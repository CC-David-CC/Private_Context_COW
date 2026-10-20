"""Independent binary64 entropy-adaptive temperature reference."""
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


def _validate(scores, support, temperature, temp_range, exponent):
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
    ids.sort()
    t = _finite_number(temperature, "temperature")
    delta = _finite_number(temp_range, "range")
    power = _finite_number(exponent, "exponent")
    if t < 0.0:
        raise ValueError("temperature must be nonnegative")
    if delta < 0.0:
        raise ValueError("range must be nonnegative")
    if power <= 0.0:
        raise ValueError("exponent must be positive")
    low = max(0.0, t - delta)
    high = t + delta
    if not math.isfinite(high):
        raise ValueError("temperature range overflows finite values")
    return values, ids, t, delta, power, low, high


def effective_temperature(scores, support, temperature, temp_range, exponent):
    """Return dynamic temperature and entropy diagnostics without mutation.

    Entropy uses stable softmax over the supplied post-filter support. Inputs
    and reductions are visited in token-ID order. A singleton active support
    has normalized entropy zero. Range zero and base temperature zero bypass
    entropy work; base temperature zero is the caller's greedy path.
    """
    values, ids, t, delta, power, low, high = _validate(
        scores, support, temperature, temp_range, exponent)
    if t == 0.0:
        return 0.0, {"normalized_entropy": None, "effective_temperature": 0.0,
                     "entropy": None, "maximum_entropy": None, "support_count": len(ids),
                     "bypassed": True}
    if delta == 0.0:
        return t, {"normalized_entropy": None, "effective_temperature": t,
                   "entropy": None, "maximum_entropy": None, "support_count": len(ids),
                   "bypassed": True}

    if len(ids) == 1:
        entropy = 0.0
        maximum_entropy = 0.0
        normalized = 0.0
    else:
        maximum = max(values[token] for token in ids)
        weights = {token: math.exp(values[token] - maximum) for token in ids}
        total = 0.0
        for token in ids:
            total += weights[token]
        if not math.isfinite(total) or total <= 0.0:
            raise ValueError("dynamic-temperature support has no finite positive mass")
        entropy = 0.0
        for token in ids:
            prob = weights[token] / total
            if prob > 0.0:
                entropy -= prob * math.log(prob)
        maximum_entropy = math.log(len(ids))
        if not math.isfinite(entropy) or not math.isfinite(maximum_entropy) or maximum_entropy <= 0:
            raise ValueError("dynamic-temperature entropy is invalid")
        normalized = min(1.0, max(0.0, entropy / maximum_entropy))

    effective = low + (high - low) * math.pow(normalized, power)
    if not math.isfinite(effective) or not low <= effective <= high:
        raise ValueError("effective temperature is outside its finite range")
    return effective, {
        "normalized_entropy": normalized,
        "effective_temperature": effective,
        "entropy": entropy,
        "maximum_entropy": maximum_entropy,
        "support_count": len(ids),
        "bypassed": False,
    }
