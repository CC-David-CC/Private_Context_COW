"""Versioned seed normalization and copy-on-write RNG stream transition."""
from __future__ import annotations

import copy
import hashlib
from collections import Counter
from collections.abc import Sequence

from .store import IntegrityError, InvalidOperation

_U64_MAX = (1 << 64) - 1


def seed_stream(seed: int) -> str:
    """Return stable stream material for a caller-supplied unsigned 64-bit seed."""
    if type(seed) is not int or not 0 <= seed <= _U64_MAX:
        raise InvalidOperation("Seed must be an unsigned 64-bit integer")
    return f"cow-seed-v1:{seed}"


def reseed_controller(state, *, seed: int, history: Sequence[int], vocab_size: int) -> dict:
    """Detach a validated controller and change only its RNG key and cursor.

    The current parser, policy, repetition counts, decoded-byte suffix, sampler
    feedback, status, and consumed history remain attached to the same scope.
    """
    from . import controller as legacy_controller

    material = seed_stream(seed)
    try:
        detached = legacy_controller._validated_state(state)
        if detached.get("status") == "idle":
            raise InvalidOperation("An idle context has no generation stream to reseed")
        if detached.get("status") not in ("active", "stopped"):
            raise InvalidOperation("Generation controller cannot be reseeded")
        if type(vocab_size) is not int or vocab_size <= 0:
            raise InvalidOperation("Vocabulary size must be positive")
        if (not isinstance(history, Sequence) or isinstance(history, (str, bytes, bytearray))
                or any(type(token) is not int or not 0 <= token < vocab_size for token in history)):
            raise InvalidOperation("Reseed history is invalid")
        expected_seen = {str(token): count for token, count in Counter(history).items()}
        if detached.get("seen") != expected_seen:
            raise IntegrityError("Reseed controller repetition state differs from canonical history")
    except (TypeError, ValueError) as error:
        if isinstance(error, (InvalidOperation, IntegrityError)):
            raise
        raise IntegrityError("Reseed controller is malformed") from error

    result = copy.deepcopy(detached)
    result["stream_key"] = hashlib.sha256(material.encode("utf-8")).hexdigest()
    result["next_draw"] = 0
    return result


def validate_reseed_source(state, *, history, vocab_size, token_bytes, is_eog,
                           grammar_engine=None) -> dict:
    """Replay a saved controller against its canonical source before reseeding.

    The controller object hash proves only that its bytes were stored intact.
    This check proves its progress fields are the result of consuming the
    accepted canonical suffix. Grammar controllers require the server-owned
    engine because the canonical record intentionally stores only its stable
    grammar identity, not a second copy of the grammar source.
    """
    from collections.abc import Sequence
    from . import controller as legacy_controller

    if (not isinstance(history, Sequence) or isinstance(history, (str, bytes, bytearray))
            or type(vocab_size) is not int or vocab_size <= 0
            or any(type(token) is not int or not 0 <= token < vocab_size for token in history)):
        raise InvalidOperation("Reseed history is invalid")
    if not callable(token_bytes) or not callable(is_eog):
        raise InvalidOperation("Reseed requires tokenizer validation callbacks")
    try:
        detached = legacy_controller._validated_state(state)
        if detached.get("status") not in ("active", "stopped"):
            raise InvalidOperation("Generation controller cannot be reseeded")
        expected_seen = {str(token): count for token, count in Counter(history).items()}
        if detached.get("seen") != expected_seen:
            raise IntegrityError("Reseed controller repetition state differs from canonical history")
        schema = detached.get("schema")
        constraint = detached.get("constraint")
        if constraint is not None:
            if (grammar_engine is None
                    or not callable(getattr(grammar_engine, "restore", None))):
                raise InvalidOperation("Grammar reseed requires its server-owned engine")
            session = grammar_engine.restore(detached, tuple(history))
            detached = session.current_controller
        elif schema in (3, 4, 5, 6):
            from .sampled_controller import PlainSamplingSession
            session = PlainSamplingSession.restore(
                detached, tuple(history), vocab_size, token_bytes, is_eog)
            detached = session.current_controller
        elif schema == 1:
            # The legacy greedy controller has no session restore helper. Its
            # generation suffix is nevertheless replayable from the saved
            # accepted count and policy; preserve the existing stream/cursor
            # while recomputing every progress-bearing field.
            accepted = detached.get("accepted_in_scope")
            if type(accepted) is not int or not 0 <= accepted <= len(history):
                raise IntegrityError("Legacy reseed controller has invalid accepted scope")
            scope_start = len(history) - accepted
            expected = legacy_controller.start(
                detached.get("policy"), "reseed-validation", history[:scope_start], vocab_size)
            expected["stream_key"] = detached.get("stream_key")
            expected["next_draw"] = detached.get("next_draw")
            events = []
            for token in history[scope_start:]:
                piece = token_bytes(token)
                end = is_eog(token)
                if type(piece) is not bytes or type(end) is not bool:
                    raise IntegrityError("Tokenizer returned invalid reseed token metadata")
                events.append((token, piece, end))
            expected = legacy_controller.consume_many(expected, events)
            if expected != detached:
                raise IntegrityError("Legacy reseed controller differs from canonical history")
        else:
            raise InvalidOperation("Unsupported generation controller for reseed")
        return copy.deepcopy(detached)
    except (TypeError, ValueError, KeyError) as error:
        if isinstance(error, (InvalidOperation, IntegrityError)):
            raise
        raise IntegrityError("Reseed controller is malformed") from error
