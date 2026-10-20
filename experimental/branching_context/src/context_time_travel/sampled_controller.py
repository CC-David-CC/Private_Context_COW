"""Schema-3 Min-P controller and plain sampled-generation session.

This module owns controller cursor updates around an external selector.  It
does not implement a second sampler: ``min_p.choose`` remains the independent
full-vocabulary reference, while callers may provide the native selector.
"""
from __future__ import annotations

import codecs
import copy
import threading
from collections import Counter
from collections.abc import Mapping, Sequence

from . import controller as _controller
from . import min_p as _min_p
from .store import IntegrityError, InvalidOperation


MIN_P_SAMPLER_ORDER = ["grammar/support", "min-p", "temperature", "categorical"]
FILTERED_SAMPLER_ORDER = ["grammar/support", "min-p", "top-k", "temperature", "top-p", "categorical"]
GREEDY_SAMPLER_ORDER = ['grammar/support', 'greedy']


def sampler_order(policy):
    if _min_p.mirostat_enabled(policy):
        order = ['grammar/support', 'mirostat2', 'categorical']
        if _min_p.dry_enabled(policy):
            order.insert(1, 'dry')
        return order
    if policy['temperature'] == 0:
        order = list(GREEDY_SAMPLER_ORDER)
    else:
        order = list(MIN_P_SAMPLER_ORDER if policy['top_k'] == 0 and policy['top_p'] == 1.0
                    else FILTERED_SAMPLER_ORDER)
    if _min_p.dry_enabled(policy): order.insert(1, 'dry')
    if _min_p.entropy_enabled(policy):
        index = order.index('temperature')
        if policy.get('typical_p', 1) < 1:
            order.insert(index, 'typical')
        if policy.get('dynatemp_range', 0) > 0:
            order[order.index('temperature')] = 'dynatemp'
    if _min_p.xtc_enabled(policy):
        order.insert(order.index('categorical'), 'xtc')
    if _min_p.adaptive_enabled(policy):
        order.insert(order.index('categorical'), 'adaptive-p')
    return order


def replace_sampling(policy, sampled):
    """Replace this generation's sampler knobs; preserve controller policies."""
    for key in _min_p.OPTIONAL_FIELDS: policy.pop(key, None)
    policy.update(sampled)


def dry_arguments(policy, history, token_bytes):
    """Transient bounded ordered history, derived from canonical consumed tokens."""
    if not _min_p.dry_enabled(policy): return {}
    history = tuple(history[-policy['dry_penalty_last_n']:])
    breakers = []
    for token in sorted(set(history)):
        piece = _token_bytes(token_bytes, token)
        if any(value in piece for value in (b'\n', b':', b'"', b'*')):
            breakers.append(token)
    return dict({key:policy[key] for key in _min_p.DRY_DEFAULTS},
                dry_history=history, dry_breakers=tuple(breakers))


_RNG_ALGORITHM = "hmac-sha256-u53-v1"
_DRAW_LIMIT = 1 << 128
_CONTROLLER_FIELDS = {
    "schema", "status", "rng_algorithm", "stream_key", "next_draw", "policy",
    "sampler_order", "grammar_state", "seen", "stop_tail", "accepted_in_scope",
    "output_pending_hex", "output_decoder", "constraint",
}
_PROPOSAL_CONTROLLER_FIELDS = _CONTROLLER_FIELDS | {"sampler_state"}
_TEXT_STOP_CONTROLLER_FIELDS = _CONTROLLER_FIELDS | {"text_stop"}
_TEXT_STOP_PROPOSAL_FIELDS = _PROPOSAL_CONTROLLER_FIELDS | {"text_stop"}
_POLICY_FIELDS = {
    "allowed", "temperature", "repetition_penalty", "eos_tokens",
    "stop_sequences", "grammar", "rejection_threshold", "min_p", "top_k", "top_p",
}


def _history_tuple(history, vocab_size):
    if (not isinstance(history, Sequence) or isinstance(history, (str, bytes, bytearray))):
        raise IntegrityError("Controller history must be a token sequence")
    result = tuple(history)
    if any(type(token) is not int or not 0 <= token < vocab_size for token in result):
        raise IntegrityError("Controller history contains an invalid token ID")
    return result


def _eos_list(eos_tokens, vocab_size):
    if (not isinstance(eos_tokens, Sequence) or isinstance(eos_tokens, (str, bytes, bytearray))
            or not eos_tokens):
        raise InvalidOperation("EOS token IDs must be a nonempty sequence")
    eos = list(eos_tokens)
    if (any(type(token) is not int or not 0 <= token < vocab_size for token in eos)
            or len(eos) != len(set(eos))):
        raise InvalidOperation("EOS token IDs must be unique vocabulary IDs")
    return eos


def _proposal_schema(schema):
    return schema in (4, 6)


def _text_stop_schema(schema):
    return schema in (5, 6)


def start(policy, stream, history, vocab_size, eos_tokens, *, text_stops=()):
    """Create a schema-3 controller with fixed generation settings."""
    if type(vocab_size) is not int or vocab_size <= 0:
        raise InvalidOperation("Vocabulary size must be a positive integer")
    if not isinstance(stream, str) or not stream:
        raise InvalidOperation("Sampler stream must be nonempty text")
    history = _history_tuple(history, vocab_size)
    eos = _eos_list(eos_tokens, vocab_size)
    try:
        sampled = _min_p.validate_policy(policy)
    except (TypeError, ValueError) as exc:
        raise InvalidOperation("Invalid Min-P policy") from exc
    from .text_stops import normalize_stop_sequences
    try:
        normalized_stops = normalize_stop_sequences(text_stops)
    except (TypeError, ValueError) as exc:
        raise InvalidOperation("Invalid text stop configuration") from exc

    legacy = _controller.start(
        {
            "temperature": sampled["temperature"],
            "repetition_penalty": 1.0,
            "eos_tokens": eos,
        },
        stream,
        history,
        vocab_size,
    )
    legacy["schema"] = 3
    legacy["policy"].update(sampled)
    legacy["sampler_order"] = sampler_order(sampled)
    legacy["constraint"] = None
    if _min_p.xtc_enabled(sampled):
        legacy['schema'] = 4
        legacy['sampler_state'] = None
    elif _min_p.mirostat_enabled(sampled):
        from .mirostat2 import start_state
        legacy['schema'] = 4
        legacy['sampler_state'] = start_state(sampled['mirostat_tau'])
    elif _min_p.adaptive_enabled(sampled):
        from .adaptive_p import initial_state
        legacy['schema'] = 4
        legacy['sampler_state'] = initial_state(sampled['adaptive_target'], sampled['adaptive_decay'])
    if normalized_stops:
        from .text_stops import TextStopMatcher
        matcher = TextStopMatcher(normalized_stops)
        legacy['schema'] = 6 if _proposal_schema(legacy['schema']) else 5
        legacy['text_stop'] = matcher.export_state()
    return legacy


def validate_controller(state, *, history=None, vocab_size=None):
    """Validate and detach a complete schema-3/4 plain sampled controller.

    If ``history`` and ``vocab_size`` are supplied, the complete canonical
    token-count map is checked as well.  UTF-8 pending bytes and accepted
    suffix consistency are checked by ``PlainSamplingSession.restore``, which
    also has the tokenizer functions needed to reconstruct them.
    """
    if not isinstance(state, Mapping):
        raise IntegrityError("Invalid sampled controller fields")
    schema = state.get("schema")
    expected_fields = (
        _CONTROLLER_FIELDS if schema == 3 else
        _PROPOSAL_CONTROLLER_FIELDS if schema == 4 else
        _TEXT_STOP_CONTROLLER_FIELDS if schema == 5 else
        _TEXT_STOP_PROPOSAL_FIELDS if schema == 6 else None)
    if expected_fields is None or set(state) != expected_fields:
        raise IntegrityError("Invalid sampled controller fields")
    if type(schema) is not int or schema not in (3, 4, 5, 6):
        raise IntegrityError("Invalid sampled controller schema")
    base_schema = 4 if _proposal_schema(schema) else 3
    if _text_stop_schema(schema):
        from .text_stops import TextStopMatcher
        try:
            checked_stop = TextStopMatcher.restore(state.get('text_stop'))
            if not checked_stop.stop_sequences:
                raise ValueError('schema-5/6 controller requires configured text stops')
        except (TypeError, ValueError) as exc:
            raise IntegrityError('Invalid sampled controller text-stop state') from exc
    if state.get("status") not in ("active", "stopped"):
        raise IntegrityError("Invalid sampled controller status")
    if state.get("rng_algorithm") != _RNG_ALGORITHM:
        raise IntegrityError("Unsupported sampled controller RNG")
    stream_key = state.get("stream_key")
    if (not isinstance(stream_key, str) or len(stream_key) != 64
            or any(ch not in "0123456789abcdef" for ch in stream_key)):
        raise IntegrityError("Invalid sampled controller stream key")
    cursor = state.get("next_draw")
    if type(cursor) is not int or not 0 <= cursor < _DRAW_LIMIT:
        raise IntegrityError("Invalid sampled controller draw cursor")
    if state.get("grammar_state") is not None or state.get("constraint") is not None:
        raise IntegrityError("Plain Min-P controller cannot carry a grammar constraint")
    if state.get("stop_tail") != []:
        raise IntegrityError("Plain Min-P controller cannot carry stop-sequence state")
    if state.get("output_decoder") != "utf8-incremental-replace-v1":
        raise IntegrityError("Unsupported output decoder")
    pending = state.get("output_pending_hex")
    if (not isinstance(pending, str) or len(pending) % 2
            or any(ch not in "0123456789abcdef" for ch in pending)):
        raise IntegrityError("Invalid pending UTF-8 bytes")
    accepted = state.get("accepted_in_scope")
    if type(accepted) is not int or accepted < 0:
        raise IntegrityError("Invalid accepted token count")

    policy = state.get("policy")
    if (not isinstance(policy, Mapping)
            or not _POLICY_FIELDS <= set(policy)
            or set(policy) - _POLICY_FIELDS - _min_p.OPTIONAL_FIELDS):
        raise IntegrityError("Invalid sampled controller policy fields")
    try:
        normalized = _min_p.validate_policy(_min_p.sampling_policy(policy))
    except (TypeError, ValueError) as exc:
        raise IntegrityError("Invalid sampled controller policy") from exc
    if set(_min_p.sampling_policy(policy)) != set(normalized):
        raise IntegrityError('Incomplete canonical sampled policy parameter group')
    if base_schema == 3 and _min_p.xtc_enabled(normalized):
        raise IntegrityError('Active XTC requires a variable-draw schema-4 controller')
    if base_schema == 3 and _min_p.mirostat_enabled(normalized):
        raise IntegrityError('Active Mirostat v2 requires a feedback schema-4 controller')
    if base_schema == 3 and _min_p.adaptive_enabled(normalized):
        raise IntegrityError('Active Adaptive-P requires a feedback schema-4 controller')
    detached_state = None
    if base_schema == 4:
        if _min_p.mirostat_enabled(normalized):
            from .mirostat2 import validate_state
            try:
                detached_state = validate_state(state.get('sampler_state'))
            except (TypeError, ValueError) as exc:
                raise IntegrityError('Invalid sampled-controller Mirostat v2 state') from exc
            if _min_p.xtc_enabled(normalized):
                raise IntegrityError('Mirostat v2 cannot be combined with XTC')
        elif _min_p.adaptive_enabled(normalized):
            from .adaptive_p import validate_state
            try:
                detached_state = validate_state(state.get('sampler_state'))
            except (TypeError, ValueError) as exc:
                raise IntegrityError('Invalid sampled-controller Adaptive-P state') from exc
            if _min_p.xtc_enabled(normalized):
                raise IntegrityError('Adaptive-P cannot be combined with XTC')
        elif _min_p.xtc_enabled(normalized):
            if state.get('sampler_state') is not None:
                raise IntegrityError('XTC schema-4 controller requires null sampler state')
        elif state.get('sampler_state') is not None:
            raise IntegrityError('Stateless schema-4 controller requires null sampler state')
    allowed = policy.get("allowed")
    if (not isinstance(allowed, list) or not allowed
            or any(type(token) is not int for token in allowed)
            or allowed != list(range(len(allowed)))):
        raise IntegrityError("Plain Min-P support must be the complete vocabulary")
    vocab_size = len(allowed) if vocab_size is None else vocab_size
    if type(vocab_size) is not int or vocab_size <= 0 or len(allowed) != vocab_size:
        raise IntegrityError("Sampled controller vocabulary size mismatch")
    eos = policy.get("eos_tokens")
    if (not isinstance(eos, list) or not eos
            or any(type(token) is not int or not 0 <= token < vocab_size for token in eos)
            or len(eos) != len(set(eos))):
        raise IntegrityError("Invalid sampled controller EOS support")
    if policy.get("stop_sequences") != [] or policy.get("grammar") is not None:
        raise IntegrityError("Plain Min-P controller has unsupported stopping state")
    rejection = policy.get("rejection_threshold")
    if type(rejection) not in (int, float) or rejection != 1.0:
        raise IntegrityError("Plain Min-P controller has unsupported rejection policy")

    try:
        detached = copy.deepcopy(dict(state))
    except Exception as exc:
        raise IntegrityError("Sampled controller cannot be detached") from exc
    if detached['sampler_order'] != sampler_order(normalized):
        raise IntegrityError('Sampler order differs from configured filters')
    detached["policy"].update(normalized)
    if base_schema == 4 and detached_state is not None:
        detached['sampler_state'] = detached_state
    seen = detached.get('seen')
    if not isinstance(seen,dict) or any(type(count) is not int or count <= 0 for count in seen.values()):
        raise IntegrityError('Invalid sampled controller token counts')
    if history is not None:
        if vocab_size is None:
            raise IntegrityError("Vocabulary size is required with history")
        tokens = _history_tuple(history, vocab_size)
        expected = {str(token): count for token, count in Counter(tokens).items()}
        seen = detached.get("seen")
        if not isinstance(seen, dict) or seen != expected:
            raise IntegrityError("Sampled controller token counts do not match history")
        if accepted > len(tokens):
            raise IntegrityError("Accepted token count exceeds canonical history")
    else:
        seen = detached.get("seen")
        if not isinstance(seen, dict):
            raise IntegrityError("Invalid sampled controller token counts")
        for token, count in seen.items():
            if (not isinstance(token, str) or not token.isdecimal() or type(count) is not int
                    or count <= 0 or str(int(token)) != token or not 0 <= int(token) < vocab_size):
                raise IntegrityError("Invalid sampled controller token counts")
    return detached


def _mask_support(mask, vocab_size):
    width = (vocab_size + 7) // 8
    if type(mask) is not bytes or len(mask) != width:
        raise IntegrityError("Sampler mask width does not match vocabulary")
    remainder = vocab_size & 7
    if remainder and mask[-1] & (0xFF ^ ((1 << remainder) - 1)):
        raise IntegrityError("Sampler mask has nonzero padding bits")
    if not any(mask):
        raise IntegrityError("Sampler mask has empty support")


def choose_owned(state, mask, selector, *, dry_context=None):
    """Select one legal token and advance the private draw cursor on success.

    ``selector(mask, temperature, min_p, draw53, **filters)`` may be the native
    implementation or an independent reference.  It must return a token ID;
    this boundary independently enforces mask shape, padding, nonempty
    support, and selected-token legality before changing ``next_draw``.
    """
    if not isinstance(state, dict) or state.get("schema") not in (3, 5) or state.get("status") != "active":
        raise InvalidOperation("Selection requires an active private schema-3 controller")
    policy = state.get("policy")
    if not isinstance(policy, dict):
        raise IntegrityError("Invalid private sampled controller policy")
    allowed = policy.get("allowed")
    if not isinstance(allowed, list) or not allowed:
        raise IntegrityError("Invalid private sampled vocabulary")
    vocab_size = len(allowed)
    _mask_support(mask, vocab_size)
    if not callable(selector):
        raise InvalidOperation("Sampler selector must be callable")
    try:
        draw = (0 if policy['temperature'] == 0 else
                _min_p.draw53(state.get("stream_key"), state.get("next_draw")))
    except (TypeError, ValueError) as exc:
        raise IntegrityError("Invalid private sampled controller draw state") from exc
    if _min_p.entropy_enabled(policy):
        filters = dict(top_k=policy['top_k'],top_p=policy['top_p'],
                       typical_p=policy.get('typical_p',1.),
                       dynatemp_range=policy.get('dynatemp_range',0.),
                       dynatemp_exponent=policy.get('dynatemp_exponent',1.))
        if _min_p.dry_enabled(policy):
            if dry_context is None:
                raise InvalidOperation('DRY requires the branch consumed-token history')
            filters.update(dry_context)
        token = selector(mask,policy['temperature'],policy['min_p'],draw,**filters)
    elif _min_p.dry_enabled(policy):
        if dry_context is None:
            raise InvalidOperation('DRY requires the branch consumed-token history')
        token = selector(mask, policy['temperature'], policy['min_p'], draw,
                         top_k=policy['top_k'], top_p=policy['top_p'], **dry_context)
    elif policy['top_k'] == 0 and policy['top_p'] == 1.0:
        token = selector(mask, policy["temperature"], policy["min_p"], draw)
    else:
        token = selector(mask, policy["temperature"], policy["min_p"], draw,
                         top_k=policy['top_k'], top_p=policy['top_p'])
    if (type(token) is not int or not 0 <= token < vocab_size
            or not mask[token >> 3] & (1 << (token & 7))):
        raise IntegrityError("Sampler selected a token outside legal support")
    if policy['temperature'] != 0:
        state["next_draw"] += 1
    return token


def choose_proposal_owned(state, mask, selector, *, source_digest, dry_context=None):
    """Stage a schema-4 proposal without advancing canonical controller state.

    The selector receives at most two deterministic ``draw_candidates`` and
    must return a bound :class:`SelectionProposal`. Cursor/feedback state is
    committed only by the owning session after token consumption succeeds.
    """
    from .sampling_proposal import (
        SelectionProposal,
        draw_candidates,
        validate_proposal,
    )
    if (not isinstance(state, dict) or state.get("schema") not in (4, 6)
            or state.get("status") != "active"):
        raise InvalidOperation("Proposal selection requires an active schema-4 controller")
    policy = state.get("policy")
    if not isinstance(policy, dict):
        raise IntegrityError("Invalid proposal controller policy")
    allowed = policy.get("allowed")
    if not isinstance(allowed, list) or not allowed:
        raise IntegrityError("Invalid proposal controller vocabulary")
    _mask_support(mask, len(allowed))
    if not callable(selector):
        raise InvalidOperation("Proposal selector must be callable")
    candidates = draw_candidates(state)
    filters = {"top_k": policy["top_k"], "top_p": policy["top_p"]}
    mirostat_active = _min_p.mirostat_enabled(policy)
    adaptive_active = _min_p.adaptive_enabled(policy)
    mirostat_fields = {'mirostat', 'mirostat_tau', 'mirostat_eta'}
    adaptive_fields = {'adaptive_target', 'adaptive_decay'}
    filters.update({key: policy[key] for key in _min_p.OPTIONAL_FIELDS
                    if key in policy
                    and (mirostat_active or key not in mirostat_fields)
                    and (adaptive_active or key not in adaptive_fields)})
    if _min_p.dry_enabled(policy):
        if dry_context is None:
            raise InvalidOperation("DRY requires the branch consumed-token history")
        filters.update(dry_context)
    if mirostat_active:
        from .mirostat2 import validate_state
        filters['sampler_state'] = validate_state(state['sampler_state'])
    elif adaptive_active:
        from .adaptive_p import validate_state
        filters['sampler_state'] = validate_state(state['sampler_state'])
    proposal = selector(mask, policy["temperature"], policy["min_p"],
                        draw_start=state["next_draw"], draw_candidates=candidates,
                        source_digest=source_digest,
                        **filters)
    if type(proposal) is not SelectionProposal:
        raise IntegrityError("Schema-4 selector did not return a selection proposal")
    return validate_proposal(state, mask, proposal, available_draws=len(candidates),
                             expected_source_digest=source_digest)


def _token_bytes(token_bytes, token):
    piece = token_bytes(token)
    if type(piece) is not bytes:
        raise IntegrityError("Tokenizer token_bytes must return bytes")
    return piece


class PlainSamplingSession:
    """Thread-owned plain Min-P controller with a sampled-token boundary."""

    def __init__(self, state, history, vocab_size, token_bytes, is_eog):
        self._owner = threading.get_ident()
        self._state = state
        self._history = tuple(history)
        self._vocab_size = vocab_size
        self._token_bytes = token_bytes
        self._is_eog = is_eog
        self._pending_state = None
        self._pending_token = None
        self._pending_proposal = None
        self._poisoned = False
        self._generation_started = False
        if _text_stop_schema(state.get('schema')):
            from .text_stops import TextStopMatcher
            self._stop_matcher = TextStopMatcher.restore(state['text_stop'])
        else:
            self._stop_matcher = None
        if _proposal_schema(state.get("schema")):
            from .sampling_proposal import controller_digest
            self._proposal_base_digest = controller_digest(state)
            self._proposal_epoch = 0
        else:
            self._proposal_base_digest = None
            self._proposal_epoch = None

    def _proposal_source_digest(self):
        from .sampling_proposal import proposal_source_digest
        if self._proposal_base_digest is None or self._proposal_epoch is None:
            raise InvalidOperation("This sampling session has no proposal binding")
        return proposal_source_digest(self._proposal_base_digest, self._proposal_epoch,
                                      self._state["next_draw"])

    def configure_generation(self, sampling):
        """Apply fixed settings once before this call's first selection."""
        self._check_owner()
        if (self._generation_started or self._pending_state is not None
                or self._state['status'] != 'active' or self._state['schema'] not in (3,4,5,6)):
            raise InvalidOperation('Sampler settings change only at a generation boundary')
        policy = _min_p.validate_policy(sampling)
        proposal_schema = _proposal_schema(self._state['schema'])
        stop_schema = _text_stop_schema(self._state['schema'])
        previous_mirostat = _min_p.mirostat_enabled(self._state['policy'])
        previous_adaptive = _min_p.adaptive_enabled(self._state['policy'])
        previous_state = self._state.get('sampler_state')
        replace_sampling(self._state['policy'], policy)
        self._state['sampler_order'] = sampler_order(policy)
        mirostat_active = _min_p.mirostat_enabled(policy)
        adaptive_active = _min_p.adaptive_enabled(policy)
        xtc_active = _min_p.xtc_enabled(policy)
        if mirostat_active:
            if previous_mirostat and previous_state is not None:
                self._state['sampler_state'] = previous_state
            else:
                from .mirostat2 import start_state
                self._state['sampler_state'] = start_state(policy['mirostat_tau'])
        elif adaptive_active:
            if previous_adaptive and previous_state is not None:
                from .adaptive_p import validate_state
                self._state['sampler_state'] = validate_state(previous_state)
            else:
                from .adaptive_p import initial_state
                self._state['sampler_state'] = initial_state(policy['adaptive_target'], policy['adaptive_decay'])
        elif proposal_schema or xtc_active:
            self._state['sampler_state'] = None
        if proposal_schema or mirostat_active or adaptive_active or xtc_active:
            self._state['schema'] = 6 if stop_schema else 4
            from .sampling_proposal import controller_digest
            self._proposal_base_digest = controller_digest(self._state)
            self._proposal_epoch = 0
        else:
            self._state['schema'] = 5 if stop_schema else 3
            self._state.pop('sampler_state', None)
            self._proposal_base_digest = None
            self._proposal_epoch = None
        self._generation_started = True

    @classmethod
    def restore(cls, state, history, vocab_size, token_bytes, is_eog):
        if type(vocab_size) is not int or vocab_size <= 0:
            raise IntegrityError("Vocabulary size must be a positive integer")
        if not callable(token_bytes) or not callable(is_eog):
            raise InvalidOperation("Restore requires tokenizer token_bytes and is_eog functions")
        tokens = _history_tuple(history, vocab_size)
        detached = validate_controller(state, history=tokens, vocab_size=vocab_size)
        accepted = detached["accepted_in_scope"]
        if accepted > len(tokens):
            raise IntegrityError("Accepted suffix exceeds canonical history")
        suffix = tokens[len(tokens) - accepted:] if accepted else ()
        decoder = codecs.getincrementaldecoder("utf8")(errors="replace")
        stop_matcher = None
        if _text_stop_schema(detached['schema']):
            from .text_stops import TextStopMatcher
            stop_matcher = TextStopMatcher(detached['text_stop']['stop_sequences'])
        stopped = False
        for index, token in enumerate(suffix):
            piece = _token_bytes(token_bytes, token)
            eog = is_eog(token)
            if type(eog) is not bool:
                raise IntegrityError("Tokenizer is_eog must return bool")
            should_stop = eog or token in detached["policy"]["eos_tokens"]
            if stopped:
                raise IntegrityError("Canonical history continues after a stop token")
            if stop_matcher is not None:
                stop_result = stop_matcher.feed(b'' if should_stop else piece, final=should_stop)
                decoder.decode(stop_result.safe_bytes, final=False)
                stopped = stop_result.stopped or should_stop
            else:
                decoder.decode(piece, final=False)
                stopped = should_stop
            if stopped and index != len(suffix) - 1:
                raise IntegrityError("Canonical history continues after a stop token")
        pending = decoder.getstate()[0].hex()
        if pending != detached["output_pending_hex"]:
            raise IntegrityError("UTF-8 pending state does not match accepted history")
        if stop_matcher is not None and stop_matcher.export_state() != detached['text_stop']:
            raise IntegrityError('Text-stop matcher state does not match accepted history')
        expected_status = "stopped" if stopped else "active"
        if detached["status"] != expected_status:
            raise IntegrityError("Controller stop status does not match accepted history")
        return cls(detached, tokens, vocab_size, token_bytes, is_eog)

    def _check_owner(self):
        if threading.get_ident() != self._owner:
            raise InvalidOperation("Plain sampling session used from a foreign thread")
        if self._poisoned:
            raise IntegrityError("Plain sampling session is poisoned")

    @property
    def current_controller(self):
        self._check_owner()
        if self._pending_state is not None or self._pending_proposal is not None:
            raise InvalidOperation("A sampled but unconsumed token cannot be published")
        return validate_controller(self._state, history=self._history, vocab_size=self._vocab_size)

    def choose(self, mask, selector):
        self._check_owner()
        if self._pending_state is not None:
            raise InvalidOperation("A sampled token is already pending consumption")
        if self._state["status"] != "active":
            raise InvalidOperation("Generation scope is stopped")
        self._generation_started = True
        if _proposal_schema(self._state.get("schema")):
            proposal = choose_proposal_owned(
                self._state, mask, selector,
                source_digest=self._proposal_source_digest(),
                dry_context=dry_arguments(self._state['policy'], self._history, self._token_bytes),
            )
            # A non-None sentinel marks the pending boundary; unlike the
            # legacy candidate path this must not copy the full controller.
            self._pending_state = True
            self._pending_proposal = proposal
            self._pending_token = proposal.token
            return proposal.token
        candidate = dict(self._state)
        token = choose_owned(candidate, mask, selector,
            dry_context=dry_arguments(candidate['policy'], self._history, self._token_bytes))
        self._pending_state = candidate
        self._pending_token = token
        return token

    def consume(self, current, token, piece, is_eog):
        self._check_owner()
        if current != self._state:
            raise InvalidOperation("Stale plain sampling controller")
        self._consume_selected(token, piece, is_eog)
        return self.current_controller

    def _consume_selected(self, token, piece, is_eog):
        self._check_owner()
        pending = self._pending_state
        if pending is None:
            raise InvalidOperation("No sampled token is pending consumption")
        if type(token) is not int or token != self._pending_token:
            raise IntegrityError("Consumed token does not match pending sample")
        if type(piece) is not bytes or piece != _token_bytes(self._token_bytes, token):
            self._poisoned = True
            raise IntegrityError("Consumed token bytes do not match tokenizer")
        expected_eog = self._is_eog(token)
        if type(expected_eog) is not bool or type(is_eog) is not bool or is_eog != expected_eog:
            self._poisoned = True
            raise IntegrityError("Consumed EOS status does not match tokenizer")

        proposal = self._pending_proposal
        if proposal is not None:
            from .sampling_proposal import commit_proposal
            candidate = commit_proposal(
                self._state, proposal, token,
                expected_source_digest=self._proposal_source_digest())
        else:
            candidate = dict(pending)
        stop_matcher = None
        stop_result = None
        if _text_stop_schema(candidate['schema']):
            stop_matcher = self._stop_matcher
            should_stop = is_eog or token in self._state['policy']['eos_tokens']
            stop_result = stop_matcher.feed(b'' if should_stop else piece, final=should_stop)
            candidate['text_stop'] = stop_matcher.export_state()
        # restore() detached this entire request-owned state. Mutate its count
        # map in place; no controller escapes while a choice is unconsumed.
        try:
            if proposal is not None:
                feedback_state = candidate.get("sampler_state")
                candidate.pop("sampler_state")
                candidate["schema"] = 1
                _controller._consume_owned(
                    candidate, token, piece, is_eog,
                    display_piece=(stop_result.safe_bytes if stop_result is not None else None),
                    text_stop=bool(stop_result and stop_result.stopped))
                candidate["schema"] = 6 if stop_matcher is not None else 4
                candidate["sampler_state"] = feedback_state
            else:
                output_schema = candidate['schema']
                if output_schema == 5:
                    candidate['schema'] = 1
                _controller._consume_owned(
                    candidate, token, piece, is_eog,
                    display_piece=(stop_result.safe_bytes if stop_result is not None else None),
                    text_stop=bool(stop_result and stop_result.stopped))
                if output_schema == 5:
                    candidate['schema'] = output_schema
        except Exception:
            self._poisoned = True
            raise
        self._state = candidate
        self._history = self._history + (token,)
        if proposal is not None:
            self._proposal_epoch += 1
        self._pending_state = None
        self._pending_token = None
        self._pending_proposal = None

    def is_stopped(self):
        self._check_owner()
        return self._state["status"] == "stopped"

    def is_complete(self):
        self._check_owner()
        return False
