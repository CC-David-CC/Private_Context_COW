"""Independent, deterministic reference for Mirostat v2 sampling.

The caller supplies legal logits, a 53-bit categorical draw, and the current
controller-owned ``mu``. This module has no RNG and never mutates input or
controller state. A caller should persist ``next_mu`` only after the selected
token has actually been consumed.

This is a binary64 reference, not a bit-compatible reproduction of llama.cpp's
float/MT19937 sampler.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


PROTOCOL = "mirostat2-reference-u53-v1"
STATE_PROTOCOL = "mirostat2-state-v1"
STATE_VERSION = 1
_DRAW_RANGE = 1 << 53
_LN2 = math.log(2.0)


def _finite_real(value, name):
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _validate_parameters(*, temperature, mu, tau, eta, draw53):
    temperature = _finite_real(temperature, "temperature")
    mu = _finite_real(mu, "mu")
    tau = _finite_real(tau, "tau")
    eta = _finite_real(eta, "eta")
    if temperature <= 0.0:
        raise ValueError("temperature must be greater than zero")
    if tau <= 0.0:
        raise ValueError("tau must be greater than zero")
    if eta < 0.0:
        raise ValueError("eta must be nonnegative")
    if type(draw53) is not int or not 0 <= draw53 < _DRAW_RANGE:
        raise ValueError("draw53 must be an integer in [0, 2^53)")
    return temperature, mu, tau, eta


def start_state(tau):
    """Return the JSON-safe versioned initial state, with ``mu = 2*tau``."""
    tau = _finite_real(tau, "tau")
    if tau <= 0.0:
        raise ValueError("tau must be greater than zero")
    mu = 2.0 * tau
    if not math.isfinite(mu):
        raise ValueError("initial mu is not finite")
    return {"protocol": STATE_PROTOCOL, "version": STATE_VERSION, "mu": mu}


def validate_state(state):
    """Validate and detach the small persisted state envelope."""
    if (not isinstance(state, Mapping)
            or set(state) != {"protocol", "version", "mu"}
            or state.get("protocol") != STATE_PROTOCOL
            or type(state.get("version")) is not int
            or state.get("version") != STATE_VERSION):
        raise ValueError("invalid Mirostat v2 state envelope")
    mu = _finite_real(state["mu"], "mu")
    return {"protocol": STATE_PROTOCOL, "version": STATE_VERSION, "mu": mu}


def choose(logits, mask, *, temperature, draw53, mu, tau, eta):
    """Select from a legal-token mask and propose the Mirostat v2 update.

    ``logits`` is a full-vocabulary finite sequence and ``mask`` is a packed
    little-bit-order byte mask, matching the independent Min-P selector. The
    function subtracts the maximum legal raw logit before temperature scaling,
    normalizes over legal support, and keeps the descending-probability prefix whose surprisal is at
    most ``mu`` (forcing the highest-probability token if that set is empty).
    It then renormalizes the retained support and uses ``draw53 / 2**53`` in a
    token-ID-ascending CDF. ``surprise_bits`` is measured under that truncated,
    renormalized distribution. ``next_mu`` is only a proposal; callers own the
    durable/consumed transition.
    """
    temperature, mu, tau, eta = _validate_parameters(
        temperature=temperature, mu=mu, tau=tau, eta=eta, draw53=draw53)
    if (not isinstance(logits, Sequence)
            or isinstance(logits, (str, bytes, bytearray)) or not logits):
        raise ValueError("logits must be a nonempty sequence")
    vocab_size = len(logits)
    expected_mask_bytes = (vocab_size + 7) // 8
    if type(mask) is not bytes or len(mask) != expected_mask_bytes:
        raise ValueError("mask width does not match vocabulary")
    remainder = vocab_size & 7
    if remainder and mask[-1] & (0xff ^ ((1 << remainder) - 1)):
        raise ValueError("mask has nonzero padding bits")

    raw_scores = []
    for token_id, value in enumerate(logits):
        raw = _finite_real(value, f"logit[{token_id}]")
        raw_scores.append(raw)

    legal_ids = [token_id for token_id in range(vocab_size)
                 if mask[token_id >> 3] & (1 << (token_id & 7))]
    if not legal_ids:
        raise ValueError("mask has empty legal support")
    maximum = max(raw_scores[token_id] for token_id in legal_ids)
    shifted = {}
    for token_id in legal_ids:
        # Subtraction before division makes common score offsets irrelevant.
        # A very distant legal score can underflow to -inf; that is a valid
        # zero-mass candidate. The legal maximum anchors normalization at zero.
        delta = raw_scores[token_id] - maximum
        score = delta / temperature
        if math.isnan(score) or score > 0.0:
            raise ValueError("shifted legal logit is invalid")
        shifted[token_id] = score
    legal_total_weight = 0.0
    for token_id in legal_ids:
        legal_total_weight += math.exp(shifted[token_id])
    if not math.isfinite(legal_total_weight) or legal_total_weight <= 0.0:
        raise ValueError("legal support has no finite positive weight")
    log_legal_weight = math.log(legal_total_weight)

    ranked_ids = sorted(legal_ids, key=lambda token_id: (-shifted[token_id], token_id))
    kept_ranked = []
    for token_id in ranked_ids:
        log_probability = shifted[token_id] - log_legal_weight
        surprise = -log_probability / _LN2
        if surprise > mu:
            break
        kept_ranked.append(token_id)
    if not kept_ranked:
        kept_ranked = [ranked_ids[0]]

    kept_ids = sorted(kept_ranked)
    kept_maximum = max(shifted[token_id] for token_id in kept_ids)
    kept_weights = {}
    total_weight = 0.0
    for token_id in kept_ids:
        weight = math.exp(shifted[token_id] - kept_maximum)
        kept_weights[token_id] = weight
        total_weight += weight
    if not math.isfinite(total_weight) or total_weight <= 0.0:
        raise ValueError("retained support has no finite positive weight")

    target = (draw53 / _DRAW_RANGE) * total_weight
    cumulative = 0.0
    selected = None
    for token_id in kept_ids:
        weight = kept_weights[token_id]
        if weight <= 0.0:
            continue
        cumulative += weight
        if cumulative > target:
            selected = token_id
            break
    if selected is None:
        # Rounding can make target equal total_weight for the largest u53.
        selected = next(token_id for token_id in reversed(kept_ids)
                        if kept_weights[token_id] > 0.0)

    selected_log_probability = (
        shifted[selected] - kept_maximum - math.log(total_weight))
    surprise_bits = -selected_log_probability / _LN2
    if not math.isfinite(surprise_bits):
        raise ValueError("observed surprise is not finite")
    error = surprise_bits - tau
    update = eta * error
    next_mu = mu - update
    if not math.isfinite(update) or not math.isfinite(next_mu):
        raise ValueError("Mirostat v2 feedback update is not finite")

    return {
        "protocol": PROTOCOL,
        "token": selected,
        "draw53": draw53,
        "legal_count": len(legal_ids),
        "legal_ids": legal_ids,
        "kept_count": len(kept_ids),
        "legal_total_weight": legal_total_weight,
        "total_weight": total_weight,
        "kept_ids": kept_ids,
        "selected_log_probability": selected_log_probability,
        "selected_probability": math.exp(selected_log_probability),
        "surprise_bits": surprise_bits,
        "next_mu": next_mu,
    }
