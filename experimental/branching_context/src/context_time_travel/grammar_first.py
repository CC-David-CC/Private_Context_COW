"""Opt-in GBNF/JSON-Schema decoding using the pinned llguidance matcher.

The engine constrains syntax only. It does not authorize tools, validate the
meaning of generated arguments, or execute an agent action.
"""
from __future__ import annotations

import copy
import codecs
import hashlib
import json
import math
import re
import threading
from collections.abc import Mapping, Sequence

from . import controller as _controller
from . import min_p as _min_p
from .store import IntegrityError, InvalidOperation

SCHEMA_VERSION = 2
CONSTRAINT_SCHEMA_VERSION = 1
LLGUIDANCE_VERSION = "1.9.1"
# llguidance otherwise permits Unicode escapes only for control characters.
# Enable standard printable four-hex/surrogate representations for ordinary
# JSON strings, keeping schema checks strict. Bind this host compiler policy
# into JSON-containing engine identities; old parser progress is incompatible.
_JSON_SCHEMA_COMPILER_OPTIONS = (
    ("lenient", False),
    ("coerce_one_of", False),
    ("json_allow_general_unicode_escapes", True),
)
MAX_GRAMMAR_SOURCE_BYTES = 1_000_000
MAX_STATE_NAME_BYTES = 128
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_CONSTRAINT_FIELDS = {
    "schema", "engine_id", "llguidance_version", "tokenizer_sha256",
    "spec_sha256", "agent_state", "scope_start",
}
# Keep unknown assertion-like keywords from silently becoming annotations. The
# compiler still checks these supported keywords and refuses unsupported forms.
_JSON_SCHEMA_KEYWORDS = frozenset({
    "$schema", "$id", "$ref", "$defs", "$comment", "title", "description",
    "default", "examples", "deprecated", "readOnly", "writeOnly",
    "type", "enum", "const", "properties", "patternProperties", "required",
    "additionalProperties", "minProperties", "maxProperties", "items", "prefixItems",
    "minItems", "maxItems", "minLength", "maxLength", "pattern", "format",
    "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
    "anyOf", "oneOf", "allOf",
})


def _check_json_schema_keywords(schema):
    pending = [schema]
    while pending:
        current = pending.pop()
        if type(current) is bool:
            continue
        if not isinstance(current, dict) or set(current) - _JSON_SCHEMA_KEYWORDS:
            raise InvalidOperation("Unsupported JSON Schema keyword or subschema")
        reference = current.get("$ref")
        if reference is not None and (not isinstance(reference, str) or not reference.startswith("#")):
            raise InvalidOperation("Only local JSON Schema references are supported")
        for key in ("properties", "patternProperties", "$defs"):
            if key in current:
                if not isinstance(current[key], dict):
                    raise InvalidOperation("Invalid JSON Schema property definitions")
                pending.extend(current[key].values())
        for key in ("items", "additionalProperties"):
            if key in current:
                pending.append(current[key])
        for key in ("prefixItems", "anyOf", "oneOf", "allOf"):
            if key in current:
                if not isinstance(current[key], list):
                    raise InvalidOperation("Invalid JSON Schema subschema sequence")
                pending.extend(current[key])
def _canonical_json(value) -> bytes:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise InvalidOperation("Grammar specification is not canonical JSON data") from exc


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _valid_name(value) -> bool:
    if not isinstance(value, str) or not value or any(ord(ch) < 0x20 or ord(ch) == 0x7f for ch in value):
        return False
    try:
        return len(value.encode("utf-8")) <= MAX_STATE_NAME_BYTES
    except UnicodeEncodeError:
        return False


def normalize_grammar_source(fmt, source):
    """Detach and bound a host definition before compiling or hashing it.

    JSON property order can affect the compiler's grammar. Compile the same
    canonical key order that defines its identity, so equal IDs cannot denote
    different parser behavior. Existing GBNF source bytes stay unchanged.
    """
    if fmt == "gbnf":
        if not isinstance(source, str):
            raise InvalidOperation("GBNF source must be text")
        try:
            data = source.encode("utf-8")
        except UnicodeError as exc:
            raise InvalidOperation("Grammar source must be valid UTF-8 text") from exc
        detached = source
    elif fmt == "thinking":
        from .thinking_grammar import normalize
        detached = normalize(source, normalize_grammar_source)
        data = _canonical_json(detached)
    elif fmt == "json_schema":
        if not isinstance(source, dict):
            raise InvalidOperation("JSON Schema source must be an inline JSON object")
        # Compiler extensions can redefine separators or ignore constraints.
        # This contract accepts ordinary JSON schemas, not lenient extensions.
        if "x-guidance" in source:
            raise InvalidOperation("JSON Schema compiler overrides are not supported")
        try:
            data = _canonical_json(source)
            detached = json.loads(data)
        except (RecursionError, ValueError, TypeError) as exc:
            raise InvalidOperation("Invalid JSON Schema source") from exc
        _check_json_schema_keywords(detached)
    else:
        raise InvalidOperation("Unsupported grammar format")
    if not data or len(data) > MAX_GRAMMAR_SOURCE_BYTES:
        raise InvalidOperation("Grammar source is empty or exceeds parser input limit")
    return detached, data


def validate_controller(state):
    """Validate and detach a schema-2 decoder controller without loading llguidance."""
    if not isinstance(state, Mapping) or type(state.get('schema')) is not int or state['schema'] not in (2,3,4,5,6):
        raise IntegrityError("Invalid grammar-first controller schema")
    expected_fields = {
        "schema", "status", "rng_algorithm", "stream_key", "next_draw", "policy",
        "sampler_order", "grammar_state", "seen", "stop_tail", "accepted_in_scope",
        "output_pending_hex", "output_decoder", "constraint",
    }
    if state['schema'] in (4, 6):
        expected_fields.add('sampler_state')
    if state['schema'] in (5, 6):
        expected_fields.add('text_stop')
    if set(state) != expected_fields:
        raise IntegrityError("Invalid grammar-first controller fields")
    constraint = state.get("constraint")
    if not isinstance(constraint, Mapping) or set(constraint) != _CONSTRAINT_FIELDS:
        raise IntegrityError("Invalid grammar-first constraint descriptor")
    if constraint.get("schema") != CONSTRAINT_SCHEMA_VERSION:
        raise IntegrityError("Unsupported grammar-first constraint schema")
    for name in ("engine_id", "tokenizer_sha256", "spec_sha256"):
        if not isinstance(constraint.get(name), str) or not _HASH_RE.fullmatch(constraint[name]):
            raise IntegrityError("Invalid grammar-first identity digest")
    if constraint.get("llguidance_version") != LLGUIDANCE_VERSION:
        raise IntegrityError("Unsupported grammar parser version")
    if not _valid_name(constraint.get("agent_state")):
        raise IntegrityError("Invalid grammar-first agent state")
    scope_start = constraint.get("scope_start")
    if type(scope_start) is not int or scope_start < 0:
        raise IntegrityError("Invalid grammar-first scope start")
    if state['schema'] in (3,4,5,6):
        from .sampled_controller import validate_controller as validate_sampled
        # Validate common schema-3 controller fields, then attach the checked
        # immutable grammar definition/scope descriptor to that same record.
        detached=validate_sampled(dict(state,constraint=None))
        detached['constraint']=copy.deepcopy(dict(constraint))
        return detached
    legacy = dict(state)
    legacy.pop("constraint")
    legacy["schema"] = 1
    detached = _controller._validated_state(legacy)
    detached["schema"] = SCHEMA_VERSION
    detached["constraint"] = copy.deepcopy(dict(constraint))
    if detached.get("status") not in ("active", "stopped"):
        raise IntegrityError("Invalid grammar-first generation status")
    if type(detached.get("accepted_in_scope")) is not int or detached["accepted_in_scope"] < 0:
        raise IntegrityError("Invalid grammar-first accepted token count")
    if not isinstance(detached.get("policy"), dict):
        raise IntegrityError("Invalid grammar-first controller policy")
    policy = detached["policy"]
    if (set(policy) != {"allowed", "temperature", "repetition_penalty", "eos_tokens",
                        "stop_sequences", "grammar", "rejection_threshold"}
            or policy.get("temperature") != 0 or policy.get("grammar") is not None
            or policy.get("repetition_penalty") != 1.0
            or policy.get("stop_sequences") != []
            or policy.get("rejection_threshold") != 1.0
            or not isinstance(policy.get("allowed"), list)
            or not policy.get("allowed")
            or any(type(token) is not int for token in policy['allowed'])
            or any(type(policy.get(field)) not in (int, float)
                   for field in ('temperature', 'repetition_penalty', 'rejection_threshold'))
            or policy.get("allowed") != list(range(len(policy.get("allowed", []))))
            or detached.get("next_draw") != 0
            or detached.get("grammar_state") is not None
            or detached.get('stop_tail') != []
            or not isinstance(detached.get('stream_key'), str)
            or not _HASH_RE.fullmatch(detached['stream_key'])
            or detached.get("output_decoder") != "utf8-incremental-replace-v1"):
        raise IntegrityError("Grammar-first controller must use the host-owned greedy sampler")
    if (not isinstance(detached.get("output_pending_hex"), str)
            or len(detached["output_pending_hex"]) % 2
            or not re.fullmatch(r"[0-9a-f]*", detached["output_pending_hex"])):
        raise IntegrityError("Invalid grammar-first UTF-8 decoder state")
    eos = policy.get("eos_tokens")
    if (not isinstance(eos, list) or not eos
            or any(type(token) is not int or token < 0 or token >= len(policy["allowed"]) for token in eos)
            or len(eos) != len(set(eos))):
        raise IntegrityError("Invalid grammar-first EOS support")
    return detached


class _StrataTokenizerAdapter:
    """Adapt tools.strata_tokenizer.Tokenizer to llguidance's tokenizer protocol."""

    def __init__(self, tokenizer):
        if not callable(getattr(tokenizer, "token_bytes", None)) or not callable(getattr(tokenizer, "encode", None)):
            raise InvalidOperation("Grammar engine requires the Strata tokenizer interface")
        tokens = getattr(tokenizer, "tokens", None)
        if not isinstance(tokens, list) or not tokens:
            raise InvalidOperation("Tokenizer has no finite vocabulary")
        token_types = getattr(tokenizer, "token_types", None)
        if token_types is not None and len(token_types) != len(tokens):
            raise InvalidOperation("Tokenizer token-type table does not match vocabulary")
        # Tokenizer encoding consults mutable dictionaries/lists. Freeze a
        # private snapshot so later caller mutation cannot alter compiled IDs.
        self.source = copy.copy(tokenizer)
        for attr in ("tokens", "token_types", "special_ids", "ids", "ranks", "special_tokens"):
            value = getattr(tokenizer, attr, None)
            if isinstance(value, dict):
                setattr(self.source, attr, dict(value))
            elif isinstance(value, list):
                setattr(self.source, attr, list(value))
        for attr in ("_piece_ids", "_bytes_cache"):
            if hasattr(self.source, attr):
                setattr(self.source, attr, {})
        self.tokens = [tokenizer.token_bytes(i) for i in range(len(tokens))]
        self.special_token_ids = [
            i for i, token_type in enumerate(getattr(tokenizer, "token_types", []) or [])
            if token_type == 3
        ]
        special_ids = getattr(tokenizer, "special_ids", {}) or {}
        self.eos_token_id = special_ids.get("eos_token_id")
        self.bos_token_id = special_ids.get("bos_token_id")
        if self.eos_token_id is None:
            self.eos_token_id = special_ids.get("eos")
        if self.bos_token_id is None:
            self.bos_token_id = special_ids.get("bos")

    def __call__(self, data):
        if isinstance(data, bytes):
            try:
                data = data.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError("llguidance requested tokenization of non-UTF-8 grammar bytes") from exc
        if not isinstance(data, str):
            raise TypeError("Tokenizer adapter accepts UTF-8 text or bytes")
        return self.source.encode(data, parse_special=False)


class GrammarEngine:
    """Compile immutable named grammars and create independent greedy sessions."""

    MAX_GRAMMAR_SOURCE_BYTES = MAX_GRAMMAR_SOURCE_BYTES

    def __init__(self, tokenizer, states, eos_tokens):
        try:
            import llguidance
            from llguidance import LLMatcher, LLTokenizer, TokenizerWrapper
        except Exception as exc:
            raise InvalidOperation("The pinned llguidance dependency is unavailable") from exc
        if getattr(llguidance, "__version__", None) != LLGUIDANCE_VERSION:
            raise InvalidOperation("Unexpected llguidance version")
        if not isinstance(states, Mapping) or not states:
            raise InvalidOperation("At least one named grammar state is required")
        if (not isinstance(eos_tokens, (list, tuple)) or not eos_tokens
                or any(type(token) is not int for token in eos_tokens)
                or len(set(eos_tokens)) != len(eos_tokens)):
            raise InvalidOperation("EOS token IDs must be a nonempty unique integer sequence")

        self._adapter = _StrataTokenizerAdapter(tokenizer)
        self.tokenizer = tokenizer
        self.vocab_size = len(self._adapter.tokens)
        if self.vocab_size <= 0 or any(not 0 <= token < self.vocab_size for token in eos_tokens):
            raise InvalidOperation("EOS token ID is outside tokenizer vocabulary")
        self.eos_tokens = tuple(eos_tokens)
        self._special_ids = frozenset(self._adapter.special_token_ids)
        if not set(self.eos_tokens).issubset(self._special_ids):
            raise InvalidOperation("Configured EOS token must be a tokenizer control token")
        self._lltokenizer = LLTokenizer(TokenizerWrapper(self._adapter), n_vocab=self.vocab_size,
                                         eos_token=list(self.eos_tokens), slices=[])
        self._matcher_type = LLMatcher
        self._parser_limits = llguidance.LLParserLimits(max_grammar_size=500_000,
                                                         max_lexer_states=250_000,
                                                         verbose_errors=False)
        self._llguidance = llguidance

        normalized = {}
        has_json_schema = False
        for name, value in states.items():
            if not _valid_name(name) or not isinstance(value, Mapping) or set(value) != {"format", "source"}:
                raise InvalidOperation("Invalid named grammar state")
            fmt = value["format"]
            source_descriptor, source_bytes = normalize_grammar_source(fmt, value["source"])
            final = source_descriptor['final'] if fmt == 'thinking' else dict(format=fmt, source=source_descriptor)
            if final['format'] == "gbnf":
                try:
                    grammar = llguidance.grammar_from("gbnf", final['source'])
                except Exception as exc:
                    raise InvalidOperation("GBNF grammar compilation failed") from exc
            else:
                has_json_schema = True
                try:
                    grammar = LLMatcher.grammar_from_json_schema(_canonical_json(final['source']).decode("utf-8"),
                        overrides=dict(_JSON_SCHEMA_COMPILER_OPTIONS))
                except Exception as exc:
                    raise InvalidOperation("JSON Schema compilation failed") from exc
            if LLMatcher.validate_grammar(grammar, self._lltokenizer, limits=self._parser_limits):
                raise InvalidOperation("Grammar validation failed")
            self._check_matcher(LLMatcher(self._lltokenizer, grammar, log_level=0,
                                          limits=self._parser_limits))
            normalized[name] = {"format": fmt, "source": source_descriptor, "grammar": grammar}
            if fmt == 'thinking':
                close = tokenizer.encode('</think>', parse_special=True)
                if (len(close) != 1 or tokenizer.token_types[close[0]] not in (3, 4)
                        or close[0] in self.eos_tokens or self._adapter.tokens[close[0]] != b'</think>'):
                    raise InvalidOperation('Thinking requires a dedicated non-EOS </think> token')
                normalized[name]['close_token'] = close[0]
                normalized[name]['free_grammar'] = llguidance.grammar_from('regex', r'(?s:.*)')

        self._states = normalized
        public_spec = {name: {"format": item["format"], "source": item["source"]}
                       for name, item in sorted(normalized.items())}
        self.spec_sha256 = _sha256(_canonical_json(public_spec))
        self.tokenizer_sha256 = self._tokenizer_hash(self._adapter.source)
        identity = {
            "llguidance_version": LLGUIDANCE_VERSION,
            "tokenizer_sha256": self.tokenizer_sha256,
            "spec_sha256": self.spec_sha256,
            "eos_tokens": list(self.eos_tokens),
        }
        if has_json_schema:
            identity["json_schema_compiler_options"] = dict(_JSON_SCHEMA_COMPILER_OPTIONS)
        self.engine_id = _sha256(_canonical_json(identity))

    @staticmethod
    def _check_matcher(matcher):
        if matcher.is_error():
            raise InvalidOperation("Grammar matcher initialization failed")

    @staticmethod
    def _tokenizer_hash(tokenizer):
        h = hashlib.sha256()
        for i in range(len(tokenizer.tokens)):
            token = tokenizer.token_bytes(i)
            h.update(len(token).to_bytes(8, "big"))
            h.update(token)
        for token_type in getattr(tokenizer, "token_types", []) or []:
            h.update(int(token_type).to_bytes(4, "big", signed=True))
        special = getattr(tokenizer, "special_ids", {}) or {}
        h.update(_canonical_json({str(key): int(value) for key, value in special.items()}))
        h.update(str(getattr(tokenizer, "pre", "")).encode("utf-8"))
        ranks = getattr(tokenizer, "ranks", {}) or {}
        for (left, right), rank in sorted(ranks.items(), key=lambda item: item[1]):
            for part in (left, right):
                encoded = part.encode("utf-8")
                h.update(len(encoded).to_bytes(8, "big"))
                h.update(encoded)
            h.update(int(rank).to_bytes(8, "big"))
        return h.hexdigest()

    def _matcher(self, agent_state):
        try:
            spec = self._states[agent_state]
        except KeyError as exc:
            raise InvalidOperation("Unknown host-selected grammar state") from exc
        matcher = self._matcher_type(self._lltokenizer, spec["grammar"], log_level=0,
                                     limits=self._parser_limits)
        self._check_matcher(matcher)
        if spec['format'] == 'thinking':
            from .thinking_grammar import ThinkingMatcher
            free = self._matcher_type(self._lltokenizer, spec['free_grammar'], log_level=0,
                                      limits=self._parser_limits)
            self._check_matcher(free)
            return ThinkingMatcher(self, matcher, spec['source']['max_tokens'], spec['close_token'], free)
        return matcher

    def _scope_controls(self, agent_state):
        """Permit only the dedicated thinking delimiter for that immutable scope."""
        close = self._states[agent_state].get('close_token')
        return frozenset() if close is None else frozenset((close,))

    def start(self, state, stream, history, vocab_size, *, sampling=None, text_stops=()):
        from .text_stops import normalize_stop_sequences
        try:
            normalized_stops = normalize_stop_sequences(text_stops)
        except (TypeError, ValueError) as exc:
            raise InvalidOperation('Invalid text stop configuration') from exc
        if not _valid_name(state) or state not in self._states:
            raise InvalidOperation("Unknown host-selected grammar state")
        if not isinstance(stream, str) or not stream:
            raise InvalidOperation("Grammar stream identity must be nonempty text")
        history = self._validate_history(history, vocab_size)
        if vocab_size != self.vocab_size:
            raise InvalidOperation("Grammar tokenizer vocabulary does not match execution")
        if normalized_stops and sampling is None:
            sampling = {"temperature": 0.0, "min_p": 0.05, "top_k": 0,
                        "top_p": 1.0, "repetition_penalty": 1.0}
        legacy = _controller.start({"temperature": 0.0, "eos_tokens": list(self.eos_tokens)},
                                   stream, history, vocab_size)
        legacy["schema"] = SCHEMA_VERSION
        legacy["constraint"] = self._constraint(state, len(history))
        if sampling is not None:
            from .min_p import validate_policy
            from .sampled_controller import sampler_order
            try:
                if not isinstance(sampling,Mapping):raise ValueError('Sampler policy must be a mapping')
                supplied=dict(sampling)
                supplied.setdefault('repetition_penalty',1.0)
                sampled=validate_policy(supplied)
            except (TypeError,ValueError) as exc:
                raise InvalidOperation('Invalid grammar Min-P policy') from exc
            legacy['schema']=3
            legacy['policy'].update(sampled)
            legacy['sampler_order']=sampler_order(sampled)
            from .min_p import xtc_enabled, mirostat_enabled
            if xtc_enabled(sampled):
                legacy['schema']=4
                legacy['sampler_state']=None
            elif mirostat_enabled(sampled):
                from .mirostat2 import start_state
                legacy['schema']=4
                legacy['sampler_state']=start_state(sampled['mirostat_tau'])
            elif _min_p.adaptive_enabled(sampled):
                from .adaptive_p import initial_state
                legacy['schema']=4
                legacy['sampler_state']=initial_state(sampled['adaptive_target'], sampled['adaptive_decay'])
        if normalized_stops:
            from .text_stops import TextStopMatcher
            matcher = TextStopMatcher(normalized_stops)
            legacy['schema'] = 6 if legacy['schema'] == 4 else 5
            legacy['text_stop'] = matcher.export_state()
        return validate_controller(legacy)

    def restore(self, controller_state, history):
        state = validate_controller(controller_state)
        constraint = state["constraint"]
        if constraint["engine_id"] != self.engine_id:
            raise IntegrityError("Grammar engine identity does not match controller")
        if constraint["tokenizer_sha256"] != self.tokenizer_sha256 or constraint["spec_sha256"] != self.spec_sha256:
            raise IntegrityError("Grammar tokenizer or specification identity changed")
        if state["policy"].get("eos_tokens") != list(self.eos_tokens):
            raise IntegrityError("Grammar EOS support changed")
        if len(state["policy"].get("allowed", ())) != self.vocab_size:
            raise IntegrityError("Grammar allowed-token support does not match tokenizer vocabulary")
        history = self._validate_history(history, self.vocab_size)
        scope_start = constraint["scope_start"]
        accepted = state["accepted_in_scope"]
        if scope_start > len(history) or len(history) - scope_start != accepted:
            raise IntegrityError("Grammar history does not match accepted scope length")
        seen = {}
        for token in history:
            key = str(token)
            seen[key] = seen.get(key, 0) + 1
        if (state['seen'] != seen or not isinstance(state['seen'], dict)
                or any(type(count) is not int for count in state['seen'].values())):
            raise IntegrityError('Grammar repetition history differs from canonical tokens')
        output_decoder = codecs.getincrementaldecoder('utf8')(errors='replace')
        stop_matcher = None
        if state['schema'] in (5, 6):
            from .text_stops import TextStopMatcher
            stop_matcher = TextStopMatcher(state['text_stop']['stop_sequences'])
        stop_triggered = False
        for token in history[scope_start:]:
            piece = self._adapter.tokens[token]
            eos = token in self.eos_tokens
            if stop_triggered:
                raise IntegrityError("Canonical grammar history continues after text stop")
            if stop_matcher is None:
                output_decoder.decode(piece, final=False)
            else:
                stop_result = stop_matcher.feed(b'' if eos else piece, final=eos)
                output_decoder.decode(stop_result.safe_bytes, final=False)
                stop_triggered = stop_result.stopped
        if state['output_pending_hex'] != output_decoder.getstate()[0].hex():
            raise IntegrityError('Grammar output decoder differs from canonical scope')
        matcher = self._matcher(constraint["agent_state"])
        saw_eos = False
        for token in history[scope_start:]:
            if saw_eos:
                raise IntegrityError("Canonical grammar history continues after EOS")
            if (token in self._special_ids and token not in self.eos_tokens
                    and token not in self._scope_controls(constraint['agent_state'])):
                raise IntegrityError("History contains a forbidden control token in grammar scope")
            mask = matcher.compute_bitmask()
            if token >= len(mask) * 8 or not (mask[token >> 3] & (1 << (token & 7))):
                raise IntegrityError("Canonical history violates the selected grammar")
            if token in self.eos_tokens:
                if not matcher.is_accepting():
                    raise IntegrityError("Canonical history contains premature EOS")
                saw_eos = True
            if not matcher.consume_token(token):
                raise IntegrityError("Canonical history violates the selected grammar")
        if matcher.is_error():
            raise IntegrityError("Grammar replay failed")
        if stop_matcher is not None and stop_matcher.export_state() != state['text_stop']:
            raise IntegrityError('Text-stop matcher state does not match accepted grammar history')
        if state["status"] == "stopped":
            if not ((not saw_eos and stop_triggered) or (saw_eos and matcher.is_stopped())):
                raise IntegrityError("Stopped grammar controller lacks an accepted EOS")
        elif saw_eos or stop_triggered:
            raise IntegrityError("Grammar controller omitted stopped status after EOS")
        return GrammarSession(self, state, matcher, history)

    def _constraint(self, state, scope_start):
        return {
            "schema": CONSTRAINT_SCHEMA_VERSION,
            "engine_id": self.engine_id,
            "llguidance_version": LLGUIDANCE_VERSION,
            "tokenizer_sha256": self.tokenizer_sha256,
            "spec_sha256": self.spec_sha256,
            "agent_state": state,
            "scope_start": scope_start,
        }

    @staticmethod
    def _validate_history(history, vocab_size):
        if not isinstance(history, (list, tuple)):
            raise InvalidOperation("Grammar history must be a token sequence")
        if (type(vocab_size) is not int or vocab_size <= 0
                or any(type(token) is not int or not 0 <= token < vocab_size for token in history)):
            raise InvalidOperation("Grammar history contains an invalid token ID")
        return tuple(history)


class GrammarSession:
    def __init__(self, engine, controller_state, matcher, history=()):
        self.engine = engine
        # restore() hands us an already detached and validated state.
        self._state = controller_state
        self._history = tuple(history)
        self._matcher = matcher
        self._scope_controls = engine._scope_controls(controller_state['constraint']['agent_state'])
        if controller_state['schema'] in (5, 6):
            from .text_stops import TextStopMatcher
            self._stop_matcher = TextStopMatcher.restore(controller_state['text_stop'])
        else:
            self._stop_matcher = None
        self._poisoned = False
        self._pending_sample = None
        self._generation_started = False
        self._owner_thread = threading.current_thread()
        if controller_state.get('schema') in (4, 6):
            from .sampling_proposal import controller_digest
            self._proposal_base_digest = controller_digest(controller_state)
            self._proposal_epoch = 0
        else:
            self._proposal_base_digest = None
            self._proposal_epoch = None

    def _proposal_source_digest(self):
        from .sampling_proposal import proposal_source_digest
        if self._proposal_base_digest is None or self._proposal_epoch is None:
            raise InvalidOperation('This grammar session has no proposal binding')
        return proposal_source_digest(self._proposal_base_digest, self._proposal_epoch,
                                     self._state['next_draw'])

    def configure_generation(self, sampling):
        """Set one policy for this restored generation without resetting progress."""
        from .min_p import validate_policy
        from .sampled_controller import sampler_order, replace_sampling
        self._ensure_live()
        if (self._generation_started or self._pending_sample is not None
                or self._state['schema'] not in (3,4,5,6) or self._state['status'] != 'active'):
            raise InvalidOperation('Sampler settings change only at a sampled generation boundary')
        policy = validate_policy(sampling)
        from .min_p import mirostat_enabled, xtc_enabled
        proposal_schema = self._state['schema'] in (4,6)
        stop_schema = self._state['schema'] in (5,6)
        previous_mirostat = mirostat_enabled(self._state['policy'])
        from .min_p import adaptive_enabled
        previous_adaptive = adaptive_enabled(self._state['policy'])
        previous_state = self._state.get('sampler_state')
        replace_sampling(self._state['policy'], policy)
        self._state['sampler_order'] = sampler_order(policy)
        mirostat_active = mirostat_enabled(policy)
        adaptive_active = adaptive_enabled(policy)
        xtc_active = xtc_enabled(policy)
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

    @property
    def current_controller(self):
        """Export a detached validated controller at an ownership boundary."""
        self._ensure_live()
        if self._pending_sample is not None:
            raise InvalidOperation('A sampled but unconsumed token cannot be published')
        return validate_controller(self._state)

    def thinking_progress(self):
        """A derived view; the canonical controller/history remain authoritative."""
        self._ensure_live()
        from .thinking_grammar import ThinkingMatcher
        return self._matcher.progress() if isinstance(self._matcher, ThinkingMatcher) else None

    def _ensure_live(self):
        if threading.current_thread() is not self._owner_thread:
            raise InvalidOperation("Grammar session belongs to another thread")
        if self._poisoned:
            raise IntegrityError("Grammar session is unusable after a failed transition")
        if self._matcher.is_error():
            self._poisoned = True
            raise IntegrityError("Grammar matcher entered an error state")

    def choose(self, logits):
        self._ensure_live()
        if self._state['schema'] in (4, 6):
            raise InvalidOperation('Schema-4 selection requires the checked proposal callback')
        if self._state['schema'] in (3, 5):
            from .min_p import choose
            return self._choose_min_p(lambda mask,temp,p,draw,**filters:
                choose(logits,mask,temperature=temp,min_p=p,draw53=draw,**filters)['token'])
        if self._state["status"] != "active":
            raise InvalidOperation("Generation scope is already stopped")
        if not isinstance(logits, Sequence) or isinstance(logits, (str, bytes)) or len(logits) != self.engine.vocab_size:
            raise InvalidOperation("Logit vector does not match grammar vocabulary")
        scores = []
        for value in logits:
            try:
                score = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise IntegrityError("Invalid raw logit") from exc
            if not math.isfinite(score):
                raise IntegrityError("Non-finite raw logit")
            scores.append(score)
        mask = self._matcher.compute_bitmask()
        self._ensure_live()
        if not isinstance(mask, (bytes, bytearray)) or len(mask) * 8 < self.engine.vocab_size:
            raise IntegrityError("Grammar matcher returned an invalid token mask")
        legal = []
        accepting = bool(self._matcher.is_accepting())
        for token in range(self.engine.vocab_size):
            if not (mask[token >> 3] & (1 << (token & 7))):
                continue
            if (token in self.engine._special_ids and token not in self.engine.eos_tokens
                    and token not in self._scope_controls):
                continue
            if token in self.engine.eos_tokens and not accepting:
                continue
            legal.append(token)
        if not legal:
            raise IntegrityError("Grammar has no legal token; constrained decoding cannot be bypassed")
        return max(legal, key=lambda token: (scores[token], -token))

    def _choose_masked(self, selector):
        """Delegate finite raw-logit argmax to a checked native selector.

        Grammar ownership remains here. Only the transient legal-token mask is
        sent to the backend; the parser and canonical controller do not move.
        Native selection must check all logits, including masked-out values.
        """
        self._ensure_live()
        if self._state['status'] != 'active':
            raise InvalidOperation('Generation scope is already stopped')
        mask = self._matcher.compute_bitmask()
        self._ensure_live()
        width = (self.engine.vocab_size + 7) // 8
        if not isinstance(mask, (bytes, bytearray)) or len(mask) < width:
            raise IntegrityError('Grammar matcher returned an invalid token mask')
        mask = bytearray(mask[:width])
        accepting = bool(self._matcher.is_accepting())
        for token in self.engine._special_ids:
            if token not in self.engine.eos_tokens and token not in self._scope_controls:
                mask[token >> 3] &= ~(1 << (token & 7))
        if not accepting:
            for token in self.engine.eos_tokens:
                mask[token >> 3] &= ~(1 << (token & 7))
        remaining = self.engine.vocab_size % 8
        if remaining:
            mask[-1] &= (1 << remaining) - 1
        if not any(mask):
            raise IntegrityError('Grammar has no legal token; constrained decoding cannot be bypassed')
        mask = bytes(mask)
        token = selector(mask)
        if (type(token) is not int or not 0 <= token < self.engine.vocab_size
                or not (mask[token >> 3] & (1 << (token & 7)))):
            raise IntegrityError('Native selector returned a token outside the grammar mask')
        return token

    def _choose_min_p(self,selector):
        from .sampled_controller import choose_owned, dry_arguments
        self._ensure_live()
        if self._state['schema'] not in (3,5) or self._pending_sample is not None:
            raise InvalidOperation('Min-P requires a ready schema-3 grammar controller')
        self._generation_started = True
        candidate=dict(self._state)
        context = dry_arguments(candidate['policy'], self._history,
                                lambda token: self.engine._adapter.tokens[token])
        token=self._choose_masked(lambda mask:choose_owned(candidate,mask,selector,dry_context=context))
        self._pending_sample=(token,candidate)
        return token

    def _choose_proposal(self, selector):
        from .sampled_controller import choose_proposal_owned, dry_arguments
        self._ensure_live()
        if self._state['schema'] not in (4,6) or self._pending_sample is not None:
            raise InvalidOperation('Proposal selection requires a ready schema-4 grammar controller')
        self._generation_started = True
        candidate = self._state
        context = dry_arguments(candidate['policy'], self._history,
                                lambda token: self.engine._adapter.tokens[token])
        proposal_box = []
        def choose_on_mask(mask):
            proposal = choose_proposal_owned(
                candidate, mask, selector, source_digest=self._proposal_source_digest(),
                dry_context=context)
            proposal_box.append(proposal)
            return proposal.token

        token = self._choose_masked(choose_on_mask)
        proposal = proposal_box[0]
        self._pending_sample = (token, None, proposal)
        return token

    def consume(self, controller_state, token, piece, is_eog):
        self._ensure_live()
        current = validate_controller(controller_state)
        if current != self._state:
            raise IntegrityError("Grammar session controller is stale or belongs to another branch")
        self._consume_selected(token, piece, is_eog)
        return self.current_controller

    def _consume_selected(self, token, piece, is_eog):
        """Advance request-owned state without exporting the vocabulary policy.

        Only the owning request uses this path. Its initial controller was
        detached and validated by restore(); no mutable controller escapes
        between transitions. Public consume() still checks the supplied state,
        and current_controller validates/detaches the publication boundary.
        """
        self._ensure_live()
        current = self._state
        if current['schema'] in (3,5):
            if self._pending_sample is None or type(token) is not int or token!=self._pending_sample[0]:
                raise InvalidOperation('Consumed token differs from pending Min-P selection')
            current=self._pending_sample[1]
        elif current['schema'] in (4,6):
            if (self._pending_sample is None or len(self._pending_sample) != 3
                    or type(token) is not int or token != self._pending_sample[0]):
                raise InvalidOperation('Consumed token differs from pending selection proposal')
            from .sampling_proposal import commit_proposal
            current = commit_proposal(
                self._state, self._pending_sample[2], token,
                expected_source_digest=self._proposal_source_digest())
        if current["status"] != "active":
            raise InvalidOperation("Generation scope is already stopped")
        if (type(token) is not int or not 0 <= token < self.engine.vocab_size
                or not isinstance(piece, bytes) or type(is_eog) is not bool):
            raise InvalidOperation("Invalid grammar token consumption")
        expected_piece = self.engine._adapter.tokens[token]
        if piece != expected_piece:
            raise IntegrityError("Consumed token bytes do not match tokenizer")
        is_eos = token in self.engine.eos_tokens
        if is_eog != is_eos:
            raise IntegrityError("Only the configured EOS token may stop grammar generation")
        if token in self.engine._special_ids and not is_eos and token not in self._scope_controls:
            raise IntegrityError("Grammar cannot emit a control token")
        if is_eos and not self._matcher.is_accepting():
            raise IntegrityError("Grammar EOS arrived before an accepting state")
        mask = self._matcher.compute_bitmask()
        if not (mask[token >> 3] & (1 << (token & 7))):
            raise IntegrityError("Consumed token is not legal in the current grammar state")
        if not self._matcher.consume_token(token):
            self._poisoned = True
            raise IntegrityError("Grammar matcher rejected a token previously declared legal")
        stop_result = None
        if self._stop_matcher is not None:
            stop_result = self._stop_matcher.feed(b'' if is_eos else piece, final=is_eos)
            current['text_stop'] = self._stop_matcher.export_state()
        legacy = dict(current)
        legacy.pop("constraint")
        legacy["schema"] = 1
        feedback_state = current.get('sampler_state') if current.get('schema') in (4,6) else None
        try:
            _controller._consume_owned(
                legacy, token, piece, is_eog,
                display_piece=(stop_result.safe_bytes if stop_result is not None else None),
                text_stop=bool(stop_result and stop_result.stopped))
            legacy["schema"] = current['schema']
            legacy["constraint"] = current["constraint"]
            if current['schema'] in (4,6):
                legacy['sampler_state'] = feedback_state
            self._state = legacy
            self._history += (token,)
            if current['schema'] in (4,6):
                self._proposal_epoch += 1
            self._pending_sample=None
        except Exception:
            self._poisoned = True
            raise

    def is_complete(self):
        self._ensure_live()
        return bool(self._matcher.is_accepting())

    def is_stopped(self):
        self._ensure_live()
        return self._state["status"] == "stopped"
