"""Independent binary64 reference for locally-typical support filtering."""
from __future__ import annotations

import math
from collections.abc import Sequence


def _finite_number(value, name: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a real number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _validated(scores, support, typical_p, min_keep):
    if (not isinstance(scores, Sequence) or isinstance(scores, (str, bytes, bytearray))
            or not scores):
        raise ValueError("scores must be a nonempty sequence")
    values = [_finite_number(value, f"score[{i}]") for i, value in enumerate(scores)]
    if (not isinstance(support, Sequence) or isinstance(support, (str, bytes, bytearray))
            or not support):
        raise ValueError("support must be a nonempty token sequence")
    ids = list(support)
    if any(type(token) is not int or token < 0 or token >= len(values) for token in ids):
        raise ValueError("support contains an invalid token ID")
    if len(set(ids)) != len(ids):
        raise ValueError("support token IDs must be unique")
    if type(min_keep) is not int or min_keep < 1:
        raise ValueError("min_keep must be a positive integer")
    p = _finite_number(typical_p, "typical_p")
    if not 0.0 < p <= 1.0:
        raise ValueError("typical_p must be in (0, 1]")
    ids.sort()
    return values, ids, p


def retain_typical(scores, support, typical_p, *, min_keep: int = 1):
    """Return retained IDs and pinned statistics without mutating inputs.

    Scores are the already DRY/Min-P/Top-K adjusted float32 frontier, exposed
    as Python numbers. Normalization and entropy use binary64. Every reduction
    sums in token-ID order. Typicality ties resolve to the lower token ID.
    Tokens are retained in increasing typicality order until cumulative mass
    is strictly greater than ``typical_p`` and ``min_keep`` is met; the result
    IDs are returned sorted for stable mask construction.
    """
    values, ids, p = _validated(scores, support, typical_p, min_keep)
    if p == 1.0:
        return ids[:], {
            "entropy": None,
            "ranked_ids": ids[:],
            "kept_count": len(ids),
            "retained_mass": None,
            "support_count": len(ids),
        }

    maximum = max(values[token] for token in ids)
    weights = {token: math.exp(values[token] - maximum) for token in ids}
    total = 0.0
    for token in ids:
        total += weights[token]
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("typical support has no finite positive mass")
    probabilities = {token: weights[token] / total for token in ids}
    log_probabilities = {
        token: values[token] - maximum - math.log(total) for token in ids
    }

    entropy = 0.0
    for token in ids:
        prob = probabilities[token]
        if prob > 0.0:
            entropy -= prob * log_probabilities[token]
    if not math.isfinite(entropy):
        raise ValueError("typical entropy is not finite")

    def distance(token):
        return abs(-log_probabilities[token] - entropy)

    ranked = sorted(ids, key=lambda token: (distance(token), token))
    retained = []
    cumulative = 0.0
    for token in ranked:
        retained.append(token)
        cumulative += probabilities[token]
        if len(retained) >= min_keep and cumulative > p:
            break
    # If rounding leaves the cumulative sum at p, retaining all is the safe
    # source-compatible result; the support is never empty.
    retained_set = set(retained)
    retained_mass = 0.0
    for token in ids:
        if token in retained_set:
            retained_mass += probabilities[token]
    return sorted(retained), {
        "entropy": entropy,
        "ranked_ids": ranked,
        "kept_count": len(retained),
        "retained_mass": retained_mass,
        "support_count": len(ids),
    }
