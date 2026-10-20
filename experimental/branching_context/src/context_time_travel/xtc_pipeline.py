"""Independent full-vocabulary XTC-after-Top-P sampling reference.

The module is deliberately stateless: both u53 draws are supplied by the
caller, and the returned choice and accounting never mutate a controller.
"""
from __future__ import annotations

import math
import struct
from collections.abc import Sequence

from . import min_p


_U53_RANGE = 1 << 53


def _categorical(weights, draw53):
    total = 0.0
    positive = []
    for token, weight in weights:
        if weight > 0.0:
            positive.append((token, weight))
            total += weight
    if not positive or not math.isfinite(total) or total <= 0.0:
        raise ValueError("XTC pipeline has no positive categorical weight")
    target = (draw53 / _U53_RANGE) * total
    cumulative = 0.0
    for token, weight in weights:
        if weight <= 0.0:
            continue
        cumulative += weight
        if cumulative > target:
            return token, total
    return positive[-1][0], total


def _base_settings(settings):
    defaults = dict(temperature=1.0, min_p=0.0, top_k=0, top_p=1.0,
                    repetition_penalty=1.0, dry_multiplier=0.0, dry_base=1.75,
                    dry_allowed_length=2, dry_penalty_last_n=256,
                    typical_p=1.0, dynatemp_range=0.0, dynatemp_exponent=1.0,
                    xtc_probability=0.0, xtc_threshold=0.1)
    defaults.update({key: value for key, value in settings.items()
                     if key not in ("dry_history", "dry_breakers")})
    return min_p.validate_policy(defaults)


def _draw(value, name):
    if type(value) is not int or not 0 <= value < _U53_RANGE:
        raise ValueError(f"{name} must be an integer in [0, 2^53)")
    return value


def _second_draw(value):
    """Accept the native 2**53 absent-draw sentinel as well as a real draw."""
    if value is None:
        return _U53_RANGE
    if type(value) is not int or not 0 <= value <= _U53_RANGE:
        raise ValueError("draw53_second must be an integer in [0, 2^53]")
    return value


def choose_v5(logits, mask, *, draw53, draw53_second=None, **settings):
    """Choose from legal full-vocabulary logits with optional XTC filtering.

    Pipeline order is legal mask, DRY, Min-P, Top-K, Typical, temperature,
    Top-P, XTC, then ID-ordered categorical sampling. XTC's activation draw is
    first and categorical draw second only when an eligible multi-token
    support reaches that stage. In all other stochastic cases the first draw
    is categorical; temperature zero consumes no draw.
    """
    policy = _base_settings(settings)
    if (not isinstance(logits, Sequence)
            or isinstance(logits, (str, bytes, bytearray)) or not logits):
        raise ValueError("logits must be a nonempty finite sequence")
    if type(draw53) is not int or not 0 <= draw53 < _U53_RANGE:
        raise ValueError("draw53 must be an integer in [0, 2^53)")
    second_draw = _second_draw(draw53_second)

    xtc_configured = min_p.xtc_enabled(policy)
    legacy_settings = {key: value for key, value in policy.items()
                       if key not in (set(min_p.XTC_DEFAULTS) | {"repetition_penalty"})}
    transient_dry = {key: settings[key] for key in ("dry_history", "dry_breakers")
                     if key in settings}
    if not xtc_configured:
        result = min_p.choose(logits, mask, draw53=draw53, **legacy_settings,
                              **transient_dry)
        greedy = policy["temperature"] == 0.0
        return dict(result, selection_protocol=result["selection_protocol"],
                    draw_count=0 if greedy else 1, activation_draws=0,
                    activated=False, removed_count=0,
                    pre_xtc_count=result["kept_count"], draw53_second=second_draw)

    vocab = len(logits)
    if type(mask) is not bytes or len(mask) != (vocab + 7) // 8:
        raise ValueError("mask width does not match vocabulary")
    remainder = vocab & 7
    if remainder and mask[-1] & (0xFF ^ ((1 << remainder) - 1)):
        raise ValueError("mask has nonzero padding bits")
    scores = [min_p._number(value, f"logit[{index}]")
              for index, value in enumerate(logits)]
    legal = [token for token in range(vocab)
             if mask[token >> 3] & (1 << (token & 7))]
    if not legal:
        raise ValueError("mask has empty legal support")

    active_dry = min_p.dry_enabled(policy)
    if active_dry:
        from .dry import FLOAT32_MAX, penalties_from_ids
        history = settings.get("dry_history", ())
        breakers = settings.get("dry_breakers", ())
        if (not isinstance(history, (tuple, list))
                or not isinstance(breakers, (tuple, list))
                or any(type(token) is not int or not 0 <= token < vocab
                       for token in (*history, *breakers))):
            raise ValueError("DRY history/breakers contain invalid vocabulary IDs")
        penalties = penalties_from_ids(history, breakers,
            **{key: policy[key] for key in min_p.DRY_DEFAULTS})
        for token, penalty in penalties.items():
            adjusted = max(-FLOAT32_MAX, scores[token] - penalty)
            scores[token] = struct.unpack("<f", struct.pack("<f", adjusted))[0]

    max_legal = max(scores[token] for token in legal)
    if policy["temperature"] == 0.0:
        legacy = min_p.choose(logits, mask, draw53=draw53, **legacy_settings,
                              **transient_dry)
        return dict(legacy, selection_protocol=min_p.SAMPLING_PROTOCOL_V5,
                    draw_count=0, activation_draws=0, activated=False,
                    removed_count=0, pre_xtc_count=legacy["kept_count"],
                    draw53_second=second_draw)

    minimum = policy["min_p"]
    if minimum == 0.0:
        kept = list(legal)
    else:
        cutoff = max_legal + math.log(minimum)
        kept = [token for token in legal if scores[token] >= cutoff]
    if policy["top_k"]:
        kept = sorted(sorted(kept, key=lambda token: (-scores[token], token))[:policy["top_k"]])

    effective = policy["temperature"]
    typical_count = None
    if min_p.entropy_enabled(policy):
        from .dynatemp import effective_temperature
        from .typical import retain_typical
        kept, _ = retain_typical(scores, kept, policy["typical_p"])
        typical_count = len(kept)
        effective, _ = effective_temperature(scores, kept, policy["temperature"],
            policy["dynatemp_range"], policy["dynatemp_exponent"])
        max_legal = max(scores[token] for token in kept)
        if effective == 0.0:
            legacy = min_p.choose(logits, mask, draw53=draw53, **legacy_settings,
                                  **transient_dry)
            return dict(legacy, selection_protocol=min_p.SAMPLING_PROTOCOL_V5,
                        draw_count=1, activation_draws=0, activated=False,
                        removed_count=0, pre_xtc_count=legacy["kept_count"],
                    draw53_second=second_draw)

    weights = []
    total = 0.0
    for token in kept:
        weight = math.exp((scores[token] - max_legal) / effective)
        weights.append((token, weight))
        total += weight
    if policy["top_p"] < 1.0:
        ranked = sorted(weights, key=lambda item: (-scores[item[0]], item[0]))
        retained = set()
        cumulative = 0.0
        for token, weight in ranked:
            retained.add(token)
            cumulative += weight
            if cumulative >= policy["top_p"] * total:
                break
        weights = [(token, weight) for token, weight in weights if token in retained]
        kept = [token for token, _ in weights]
        total = 0.0
        for _, weight in weights:
            total += weight

    pre_xtc_count = len(weights)
    if pre_xtc_count <= 1:
        token, total = _categorical(weights, draw53)
        return dict(selection_protocol=min_p.SAMPLING_PROTOCOL_V5, token=token,
                    logits_count=vocab, mask_bytes=len(mask), legal_count=len(legal),
                    kept_count=len(weights), draw53=draw53, total_weight=total,
                    **({"effective_temperature": effective,
                       "typical_kept_count": typical_count} if typical_count is not None else {}),
                    draw_count=1, activation_draws=0, activated=False,
                    removed_count=0, pre_xtc_count=pre_xtc_count,
                    draw53_second=second_draw)

    categorical_draw = second_draw
    if categorical_draw == _U53_RANGE:
        raise ValueError("XTC with multi-token support requires a second draw")
    activated = draw53 / _U53_RANGE <= policy["xtc_probability"]
    removed = 0
    if activated:
        ranked = sorted(weights, key=lambda item: (-scores[item[0]], item[0]))
        total_for_threshold = total
        last_eligible = 0
        for index, (_, weight) in enumerate(ranked):
            if weight / total_for_threshold >= policy["xtc_threshold"]:
                last_eligible = index
            else:
                break
        if last_eligible > 0 and len(ranked) - last_eligible >= 1:
            removed_ids = {token for token, _ in ranked[:last_eligible]}
            removed = len(removed_ids)
            weights = [(token, weight) for token, weight in weights if token not in removed_ids]
            center = max(scores[token] for token, _ in weights)
            weights = [(token, math.exp((scores[token] - center) / effective))
                       for token, _ in weights]

    token, total = _categorical(weights, categorical_draw)
    result = dict(selection_protocol=min_p.SAMPLING_PROTOCOL_V5, token=token,
                  logits_count=vocab, mask_bytes=len(mask), legal_count=len(legal),
                  kept_count=len(weights), draw53=draw53, total_weight=total,
                  draw_count=2, activation_draws=1, activated=activated,
                  removed_count=removed, pre_xtc_count=pre_xtc_count,
                  draw53_second=second_draw)
    if typical_count is not None:
        result.update(effective_temperature=effective, typical_kept_count=typical_count)
    return result
