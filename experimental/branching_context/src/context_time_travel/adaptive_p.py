"""Deterministic binary64 reference for llama.cpp-style Adaptive-P.

The chooser is pure: it observes an explicit branch state and a caller-owned
u53 draw, then returns a private feedback proposal. ``accept`` creates the
next state only after the caller confirms that the proposed token was consumed.
The caller owns draw-cursor advancement and canonical controller publication.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence


PROTOCOL = "full-vocab-adaptive-p-u53-v1"
STATE_VERSION = "adaptive-p-state-v1"
_U53_RANGE = 1 << 53
_STATE_KEYS = {"version", "weighted_sum", "total_weight"}
_PROPOSAL_KEYS = {
    "version", "state_sha256", "token", "original_probability",
    "decay", "feedback_enabled",
}
_DISTRIBUTION_WIDTH = 0.3
_PEAK_LOGIT_VALUE = 5.0
_SHARPNESS = 10.0


def _number(value, name):
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a real number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _policy(adaptive_target, adaptive_decay):
    target = _number(adaptive_target, "adaptive_target")
    decay = _number(adaptive_decay, "adaptive_decay")
    if target > 1.0:
        raise ValueError("adaptive_target must be at most 1; negative disables adaptation")
    if not 0.0 <= decay <= 0.99:
        raise ValueError("adaptive_decay must be in [0, 0.99]")
    return target, decay


def validate_state(state):
    """Return a detached, JSON-safe EMA state with strict numeric fields."""
    if not isinstance(state, Mapping) or set(state) != _STATE_KEYS:
        raise ValueError("Adaptive-P state must contain exactly version, weighted_sum, total_weight")
    if state["version"] != STATE_VERSION:
        raise ValueError("unsupported Adaptive-P state version")
    weighted_sum = _number(state["weighted_sum"], "weighted_sum")
    total_weight = _number(state["total_weight"], "total_weight")
    if total_weight <= 0.0:
        raise ValueError("total_weight must be positive")
    return {
        "version": STATE_VERSION,
        "weighted_sum": weighted_sum,
        "total_weight": total_weight,
    }


def initial_state(adaptive_target, adaptive_decay):
    """Create llama.cpp's cold-start EMA prior for this target and decay."""
    target, decay = _policy(adaptive_target, adaptive_decay)
    total_weight = 1.0 / (1.0 - decay)
    weighted_sum = target / (1.0 - decay)
    if not math.isfinite(total_weight) or not math.isfinite(weighted_sum):
        raise ValueError("Adaptive-P initial EMA state must be finite")
    return validate_state({
        "version": STATE_VERSION,
        "weighted_sum": weighted_sum,
        "total_weight": total_weight,
    })


def _state_digest(state):
    encoded = json.dumps(
        validate_state(state), sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_inputs(logits, mask, state, adaptive_target, adaptive_decay, draw53):
    target, decay = _policy(adaptive_target, adaptive_decay)
    normalized_state = validate_state(state)
    if (not isinstance(logits, Sequence)
            or isinstance(logits, (str, bytes, bytearray))
            or len(logits) == 0):
        raise ValueError("logits must be a nonempty finite sequence")
    if type(draw53) is not int or not 0 <= draw53 < _U53_RANGE:
        raise ValueError("draw53 must be an integer in [0, 2^53)")
    expected_mask_bytes = (len(logits) + 7) // 8
    if type(mask) is not bytes or len(mask) != expected_mask_bytes:
        raise ValueError("mask width does not match vocabulary")
    remainder = len(logits) & 7
    if remainder and mask[-1] & (0xFF ^ ((1 << remainder) - 1)):
        raise ValueError("mask has nonzero padding bits")
    scores = [_number(value, f"logit[{token}]") for token, value in enumerate(logits)]
    support = [token for token in range(len(scores))
               if mask[token >> 3] & (1 << (token & 7))]
    if not support:
        raise ValueError("mask has empty legal support")
    return target, decay, normalized_state, scores, support


def _softmax(scores, support):
    maximum = max(scores[token] for token in support)
    weights = []
    total = 0.0
    for token in support:
        weight = math.exp(scores[token] - maximum)
        weights.append((token, weight))
        total += weight
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("Adaptive-P support has no finite positive categorical mass")
    return [(token, weight / total) for token, weight in weights]


def _draw_from_weights(weights, draw53):
    positive = [(token, weight) for token, weight in weights if weight > 0.0]
    if not positive:
        raise ValueError("Adaptive-P distribution has no positive categorical weight")
    total = 0.0
    for _, weight in weights:
        total += weight
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("Adaptive-P distribution has no finite positive categorical mass")
    threshold = (draw53 / _U53_RANGE) * total
    cumulative = 0.0
    for token, weight in weights:
        if weight <= 0.0:
            continue
        cumulative += weight
        if cumulative > threshold:
            return token
    return positive[-1][0]


def choose(logits, mask, state, *, adaptive_target, adaptive_decay, draw53):
    """Select a token and return a private, state-bound accepted-token proposal.

    ``mask`` is packed least-significant-bit first. The input state, logits,
    and mask are not mutated. ``draw53`` is supplied by the host and is never
    advanced here. Negative targets sample the original distribution and do
    not propose EMA feedback.
    """
    target, decay, state_value, scores, support = _validate_inputs(
        logits, mask, state, adaptive_target, adaptive_decay, draw53)
    original = _softmax(scores, support)
    feedback_enabled = target >= 0.0

    adapted_target = None
    selection_weights = original
    if feedback_enabled:
        mean_probability = state_value["weighted_sum"] / state_value["total_weight"]
        adapted_target = min(1.0, max(0.0, 2.0 * target - mean_probability))
        transformed = []
        original_by_id = dict(original)
        for token in support:
            probability = original_by_id[token]
            distance = abs((probability - adapted_target) / _DISTRIBUTION_WIDTH)
            transformed_logit = (
                _PEAK_LOGIT_VALUE
                - _SHARPNESS * distance * distance / (1.0 + distance)
            )
            transformed.append((token, transformed_logit))
        selection_weights = _softmax(
            [item[1] for item in transformed],
            list(range(len(transformed))),
        )
        selection_weights = [
            (transformed[index][0], weight)
            for index, weight in selection_weights
        ]

    token = _draw_from_weights(selection_weights, draw53)
    original_probability = dict(original)[token]
    proposal = {
        "version": PROTOCOL,
        "state_sha256": _state_digest(state_value),
        "token": token,
        "original_probability": original_probability,
        "decay": decay,
        "feedback_enabled": feedback_enabled,
    }
    return {
        "selection_protocol": PROTOCOL,
        "token": token,
        "original_probability": original_probability,
        "adapted_target": adapted_target,
        "legal_count": len(support),
        "draw53": draw53,
        "proposal": proposal,
    }


def accept(state, proposal, token):
    """Return a detached next EMA state after matching the consumed token.

    A mismatched token or proposal from a different state fails without
    mutating either input. The host should publish the returned state only
    after it has established that this selected token was actually consumed.
    """
    normalized_state = validate_state(state)
    if not isinstance(proposal, Mapping) or set(proposal) != _PROPOSAL_KEYS:
        raise ValueError("invalid Adaptive-P feedback proposal")
    if proposal["version"] != PROTOCOL:
        raise ValueError("unsupported Adaptive-P proposal version")
    proposal_token = proposal["token"]
    if type(proposal_token) is not int or proposal_token < 0:
        raise ValueError("proposal token must be a nonnegative integer")
    if type(token) is not int or token != proposal_token:
        raise ValueError("accepted token does not match Adaptive-P proposal")
    if proposal["state_sha256"] != _state_digest(normalized_state):
        raise ValueError("Adaptive-P proposal was created from a different state")
    original_probability = _number(proposal["original_probability"], "original_probability")
    if not 0.0 <= original_probability <= 1.0:
        raise ValueError("original_probability must be in [0, 1]")
    decay = _number(proposal["decay"], "decay")
    if not 0.0 <= decay <= 0.99:
        raise ValueError("decay must be in [0, 0.99]")
    if type(proposal["feedback_enabled"]) is not bool:
        raise ValueError("feedback_enabled must be a boolean")
    if not proposal["feedback_enabled"]:
        return normalized_state

    weighted_sum = original_probability + decay * normalized_state["weighted_sum"]
    total_weight = 1.0 + decay * normalized_state["total_weight"]
    if not math.isfinite(weighted_sum) or not math.isfinite(total_weight):
        raise ValueError("Adaptive-P updated EMA state must be finite")
    return validate_state({
        "version": STATE_VERSION,
        "weighted_sum": weighted_sum,
        "total_weight": total_weight,
    })
