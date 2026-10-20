"""Independent, history-based DRY penalty reference.

This module deliberately uses a small brute-force scan rather than the native
sampler's reverse Z-algorithm. It is a correctness oracle for bounded canonical
token history, not a high-throughput sampler.
"""
from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable, Sequence

UINT32_MAX = (1 << 32) - 1
FLOAT32_MAX = 3.4028234663852886e38
DEFAULT_SEQUENCE_BREAKERS = (b"\n", b":", b'"', b"*")

__all__ = [
    "DEFAULT_SEQUENCE_BREAKERS",
    "FLOAT32_MAX",
    "penalties",
    "penalties_from_ids",
    "repeat_lengths",
]


def _as_finite_number(name: str, value: object, minimum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(number) or number < minimum:
        raise ValueError(f"{name} must be finite and at least {minimum}")
    return number


def _as_uint32(name: str, value: object) -> int:
    if type(value) is not int or value < 0 or value > UINT32_MAX:
        raise ValueError(f"{name} must be an integer in [0, 2**32 - 1]")
    return value


def _history_window(history: Iterable[int], last_n: int) -> tuple[int, ...]:
    if isinstance(history, (str, bytes, bytearray)):
        raise TypeError("history must be an ordered iterable of token IDs")
    if last_n == 0:
        return ()
    try:
        if isinstance(history, Sequence):
            result = tuple(history[-last_n:])
        else:
            result = tuple(deque(history, maxlen=last_n))
    except TypeError as exc:
        raise TypeError("history must be an ordered iterable of token IDs") from exc
    if any(type(token) is not int or token < 0 or token > UINT32_MAX for token in result):
        raise ValueError("history token IDs must be uint32 integers")
    return result


def _breaker_set(breaker_ids: Iterable[int]) -> frozenset[int]:
    if isinstance(breaker_ids, (str, bytes, bytearray)):
        raise TypeError("breaker_ids must be an iterable of token IDs")
    try:
        raw_ids = tuple(breaker_ids)
    except TypeError as exc:
        raise TypeError("breaker_ids must be an iterable of token IDs") from exc
    if any(type(token) is not int or token < 0 or token > UINT32_MAX for token in raw_ids):
        raise ValueError("breaker token IDs must be uint32 integers")
    return frozenset(raw_ids)


def repeat_lengths(
    history: Iterable[int],
    breaker_ids: Iterable[int],
    *,
    dry_allowed_length: int = 2,
    dry_penalty_last_n: int = 256,
) -> dict[int, int]:
    """Return each following candidate token's longest repeated suffix length.

    Only the current suffix is capped by the latest breaker. Earlier source
    occurrences remain searchable in the whole last-N window. Candidate tokens
    that are breakers are omitted. With allowed length zero, every non-breaker
    token with a preceding token in the window gets a zero-length minimum; any
    positive matching length still takes precedence.
    """
    allowed = _as_uint32("dry_allowed_length", dry_allowed_length)
    last_n = _as_uint32("dry_penalty_last_n", dry_penalty_last_n)
    breakers = _breaker_set(breaker_ids)
    if last_n == 0:
        return {}
    window = _history_window(history, last_n)
    if len(window) < 2:
        return {}

    latest_breaker = -1
    for index, token in enumerate(window):
        if token in breakers:
            latest_breaker = index
    current_suffix_length = len(window) - latest_breaker - 1

    longest_by_candidate: dict[int, int] = {}
    for candidate_index in range(1, len(window)):
        candidate = window[candidate_index]
        if candidate in breakers:
            continue
        limit = min(candidate_index, current_suffix_length)
        longest = 0
        if allowed == 0:
            longest_by_candidate.setdefault(candidate, 0)
        for length in range(max(1, allowed), limit + 1):
            if window[candidate_index - length:candidate_index] == window[-length:]:
                longest = length
        if longest >= allowed and longest > 0:
            prior = longest_by_candidate.get(candidate, 0)
            if longest > prior:
                longest_by_candidate[candidate] = longest
    return longest_by_candidate


def _validate_penalty_parameters(
    dry_multiplier: object,
    dry_base: object,
    dry_allowed_length: object,
    dry_penalty_last_n: object,
) -> tuple[float, float, int, int]:
    multiplier = _as_finite_number("dry_multiplier", dry_multiplier, 0.0)
    base = _as_finite_number("dry_base", dry_base, 1.0)
    allowed = _as_uint32("dry_allowed_length", dry_allowed_length)
    last_n = _as_uint32("dry_penalty_last_n", dry_penalty_last_n)
    return multiplier, base, allowed, last_n


def penalties_from_ids(
    history: Iterable[int],
    breaker_ids: Iterable[int],
    *,
    dry_multiplier: float = 0.0,
    dry_base: float = 1.75,
    dry_allowed_length: int = 2,
    dry_penalty_last_n: int = 256,
) -> dict[int, float]:
    """Return sparse DRY penalties using an already identified breaker-ID set."""
    multiplier, base, allowed, last_n = _validate_penalty_parameters(
        dry_multiplier, dry_base, dry_allowed_length, dry_penalty_last_n
    )
    breakers = _breaker_set(breaker_ids)
    if multiplier == 0.0 or last_n == 0:
        return {}
    lengths = repeat_lengths(
        history, breakers, dry_allowed_length=allowed, dry_penalty_last_n=last_n
    )
    result: dict[int, float] = {}
    for token, repeated_length in lengths.items():
        exponent = repeated_length - allowed
        try:
            value = multiplier * math.pow(base, exponent)
        except OverflowError:
            value = math.inf
        result[token] = min(value, FLOAT32_MAX) if math.isfinite(value) else FLOAT32_MAX
    return result


def penalties(
    history: Iterable[int],
    *,
    dry_multiplier: float = 0.0,
    dry_base: float = 1.75,
    dry_allowed_length: int = 2,
    dry_penalty_last_n: int = 256,
    token_bytes,
) -> dict[int, float]:
    """Return sparse penalties, identifying fixed breakers from token bytes."""
    multiplier, base, allowed, last_n = _validate_penalty_parameters(
        dry_multiplier, dry_base, dry_allowed_length, dry_penalty_last_n
    )
    if not callable(token_bytes):
        raise TypeError("token_bytes must be callable")
    if multiplier == 0.0 or last_n == 0:
        return {}
    window = _history_window(history, last_n)
    if len(window) < 2:
        return {}
    breaker_ids = set()
    byte_cache: dict[int, bytes] = {}
    for token in window:
        if token not in byte_cache:
            value = token_bytes(token)
            if not isinstance(value, bytes):
                raise TypeError("token_bytes must return bytes")
            byte_cache[token] = value
        if any(breaker in byte_cache[token] for breaker in DEFAULT_SEQUENCE_BREAKERS):
            breaker_ids.add(token)
    return penalties_from_ids(
        window,
        breaker_ids,
        dry_multiplier=multiplier,
        dry_base=base,
        dry_allowed_length=allowed,
        dry_penalty_last_n=last_n,
    )
