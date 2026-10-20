"""Independent full-vocabulary Min-P reference selection.

This module is deliberately separate from the native masked-argmax selector
and from controller mutation. Callers own draw-cursor advancement and consume
the returned token only after the corresponding model operation succeeds.
"""
from __future__ import annotations

import hashlib
import hmac
import math
import struct
from collections.abc import Mapping, Sequence


MIN_P_PROTOCOL = "full-vocab-min-p-u53-v1"
SAMPLING_PROTOCOL_V2 = "full-vocab-min-p-u53-v2"
DRY_PROTOCOL = "full-vocab-dry-min-p-u53-v3"
SAMPLING_PROTOCOL_V4 = "full-vocab-typical-dyna-min-p-u53-v4"
SAMPLING_PROTOCOL_V5 = "full-vocab-xtc-min-p-u53-v5"
SAMPLING_PROTOCOL_V6 = "full-vocab-mirostat2-u53-v6"
SAMPLING_PROTOCOL_V7 = "full-vocab-adaptive-p-u53-v7"
DRY_DEFAULTS = dict(dry_multiplier=0.0, dry_base=1.75,
                    dry_allowed_length=2, dry_penalty_last_n=256)
DYNA_DEFAULTS = dict(dynatemp_range=0.0, dynatemp_exponent=1.0)
XTC_DEFAULTS = dict(xtc_probability=0.0, xtc_threshold=0.1)
MIROSTAT_DEFAULTS = dict(mirostat=0, mirostat_tau=5.0, mirostat_eta=0.1)
ADAPTIVE_DEFAULTS = dict(adaptive_target=-1.0, adaptive_decay=0.90)
OPTIONAL_FIELDS = (set(DRY_DEFAULTS) | set(DYNA_DEFAULTS) | set(XTC_DEFAULTS)
                   | set(MIROSTAT_DEFAULTS) | set(ADAPTIVE_DEFAULTS) | {'typical_p'})
_DRAW_LIMIT = 1 << 128
_U53_RANGE = 1 << 53
_POLICY_FIELDS = {
    "temperature", "min_p", "top_k", "top_p", "repetition_penalty",
}


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


def validate_policy(policy):
    """Normalize generation settings; v1 profiles support the unfiltered subset."""
    if (not isinstance(policy, Mapping) or not _POLICY_FIELDS <= set(policy)
            or set(policy) - _POLICY_FIELDS - OPTIONAL_FIELDS):
        raise ValueError("Min-P policy must contain exactly the supported fields")
    temperature = _number(policy["temperature"], "temperature")
    min_p = _number(policy["min_p"], "min_p")
    top_p = _number(policy["top_p"], "top_p")
    repetition_penalty = _number(policy["repetition_penalty"], "repetition_penalty")
    top_k = policy["top_k"]
    if temperature < 0:
        raise ValueError("temperature must be nonnegative")
    if not 0 <= min_p <= 1:
        raise ValueError("min_p must be in [0, 1]")
    if type(top_k) is not int or not 0 <= top_k <= (1 << 64) - 1:
        raise ValueError("top_k must be a nonnegative uint64 integer")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    if repetition_penalty != 1.0:
        raise ValueError("repetition_penalty must be 1")
    result = {
        "temperature": temperature,
        "min_p": min_p,
        "top_k": top_k,
        "top_p": top_p,
        "repetition_penalty": 1.0,
    }
    if set(policy) & set(DRY_DEFAULTS):
        dry = dict(DRY_DEFAULTS)
        dry.update({key:policy[key] for key in DRY_DEFAULTS if key in policy})
        dry['dry_multiplier'] = _number(dry['dry_multiplier'], 'dry_multiplier')
        dry['dry_base'] = _number(dry['dry_base'], 'dry_base')
        if dry['dry_multiplier'] < 0 or dry['dry_base'] < 1:
            raise ValueError('DRY multiplier must be nonnegative and base at least one')
        for key in ('dry_allowed_length', 'dry_penalty_last_n'):
            if type(dry[key]) is not int or not 0 <= dry[key] < (1 << 32):
                raise ValueError(f'{key} must be a uint32 integer')
        result.update(dry)
    if 'typical_p' in policy:
        typical_p = _number(policy['typical_p'], 'typical_p')
        if not 0 < typical_p <= 1:
            raise ValueError('typical_p must be in (0, 1]')
        result['typical_p'] = typical_p
    if set(policy) & set(DYNA_DEFAULTS):
        dynamic = dict(DYNA_DEFAULTS)
        dynamic.update({key:policy[key] for key in DYNA_DEFAULTS if key in policy})
        dynamic = {key:_number(value,key) for key,value in dynamic.items()}
        if dynamic['dynatemp_range'] < 0 or dynamic['dynatemp_exponent'] <= 0:
            raise ValueError('DynaTemp range must be nonnegative and exponent positive')
        if not math.isfinite(temperature + dynamic['dynatemp_range']):
            raise ValueError('DynaTemp upper temperature must be finite')
        result.update(dynamic)
    if set(policy) & set(XTC_DEFAULTS):
        xtc = dict(XTC_DEFAULTS)
        xtc.update({key: policy[key] for key in XTC_DEFAULTS if key in policy})
        xtc = {key: _number(value, key) for key, value in xtc.items()}
        if not 0.0 <= xtc['xtc_probability'] <= 1.0:
            raise ValueError('xtc_probability must be in [0, 1]')
        if not 0.0 <= xtc['xtc_threshold'] <= 1.0:
            raise ValueError('xtc_threshold must be in [0, 1]')
        result.update(xtc)
    if set(policy) & set(MIROSTAT_DEFAULTS):
        mirostat = dict(MIROSTAT_DEFAULTS)
        mirostat.update({key: policy[key] for key in MIROSTAT_DEFAULTS if key in policy})
        mode = mirostat['mirostat']
        if type(mode) is not int or mode not in (0, 2):
            raise ValueError('mirostat must be integer mode 0 or 2')
        tau = _number(mirostat['mirostat_tau'], 'mirostat_tau')
        eta = _number(mirostat['mirostat_eta'], 'mirostat_eta')
        if tau <= 0.0 or not math.isfinite(2.0 * tau):
            raise ValueError('mirostat_tau must be positive with finite initial mu')
        if eta < 0.0:
            raise ValueError('mirostat_eta must be nonnegative')
        mirostat.update(mirostat_tau=tau, mirostat_eta=eta)
        if mode == 2 and (
                temperature <= 0.0 or min_p != 0.0 or top_k != 0 or top_p != 1.0
                or result.get('typical_p', 1.0) != 1.0
                or result.get('dynatemp_range', 0.0) != 0.0
                or xtc_enabled({**result, **mirostat})):
            raise ValueError('Mirostat mode 2 cannot be combined with other probability truncators')
        result.update(mirostat)
    if set(policy) & set(ADAPTIVE_DEFAULTS):
        adaptive = dict(ADAPTIVE_DEFAULTS)
        adaptive.update({key: policy[key] for key in ADAPTIVE_DEFAULTS if key in policy})
        target = _number(adaptive['adaptive_target'], 'adaptive_target')
        decay = _number(adaptive['adaptive_decay'], 'adaptive_decay')
        if target > 1.0:
            raise ValueError('adaptive_target must be at most 1; negative disables adaptation')
        if not 0.0 <= decay <= 0.99:
            raise ValueError('adaptive_decay must be in [0, 0.99]')
        result.update(adaptive_target=target, adaptive_decay=decay)
        if target >= 0.0 and (
                temperature <= 0.0 or mirostat_enabled(result) or xtc_enabled(result)):
            raise ValueError('Adaptive-P requires positive base temperature and cannot be combined with Mirostat 2 or active XTC')
    return result


def sampling_policy(policy):
    """Extract sampler settings from the one canonical controller policy."""
    return {key:policy[key] for key in _POLICY_FIELDS | OPTIONAL_FIELDS if key in policy}


def dry_enabled(policy):
    return policy.get('dry_multiplier', 0) > 0 and policy.get('dry_penalty_last_n', 0) > 0


def entropy_enabled(policy):
    return policy['temperature'] > 0 and (policy.get('typical_p', 1) < 1
                                         or policy.get('dynatemp_range', 0) > 0)


def xtc_enabled(policy):
    return (policy['temperature'] > 0
            and policy.get('xtc_probability', 0.0) > 0.0
            and policy.get('xtc_threshold', XTC_DEFAULTS['xtc_threshold']) <= 0.5)


def mirostat_enabled(policy):
    return policy.get('mirostat', 0) == 2


def adaptive_enabled(policy):
    return policy.get('adaptive_target', ADAPTIVE_DEFAULTS['adaptive_target']) >= 0.0


def supports_policy(protocol, policy):
    if adaptive_enabled(policy):
        return protocol == SAMPLING_PROTOCOL_V7
    if mirostat_enabled(policy):
        return protocol in (SAMPLING_PROTOCOL_V6, SAMPLING_PROTOCOL_V7)
    if xtc_enabled(policy):
        return protocol in (SAMPLING_PROTOCOL_V5, SAMPLING_PROTOCOL_V6, SAMPLING_PROTOCOL_V7)
    if entropy_enabled(policy):
        return protocol in (SAMPLING_PROTOCOL_V4, SAMPLING_PROTOCOL_V5,
                            SAMPLING_PROTOCOL_V6, SAMPLING_PROTOCOL_V7)
    if dry_enabled(policy):
        return protocol in (DRY_PROTOCOL, SAMPLING_PROTOCOL_V4, SAMPLING_PROTOCOL_V5,
                            SAMPLING_PROTOCOL_V6, SAMPLING_PROTOCOL_V7)
    if policy['temperature'] == 0 or policy['top_k'] or policy['top_p'] != 1:
        return protocol in (SAMPLING_PROTOCOL_V2, DRY_PROTOCOL, SAMPLING_PROTOCOL_V4,
                            SAMPLING_PROTOCOL_V5, SAMPLING_PROTOCOL_V6,
                            SAMPLING_PROTOCOL_V7)
    return protocol in (MIN_P_PROTOCOL, SAMPLING_PROTOCOL_V2, DRY_PROTOCOL,
                        SAMPLING_PROTOCOL_V4, SAMPLING_PROTOCOL_V5,
                        SAMPLING_PROTOCOL_V6, SAMPLING_PROTOCOL_V7)


def draw53(stream_key, cursor):
    """Return the existing HMAC-SHA256 u53 draw without advancing state."""
    if (not isinstance(stream_key, str) or len(stream_key) != 64
            or any(ch not in "0123456789abcdef" for ch in stream_key)):
        raise ValueError("stream_key must be a lowercase SHA-256 hex digest")
    if type(cursor) is not int or not 0 <= cursor < _DRAW_LIMIT - 1:
        raise ValueError("draw cursor is outside the supported range")
    key = bytes.fromhex(stream_key)
    message = b"ctt-u53-v1:" + cursor.to_bytes(16, "big")
    digest = hmac.new(key, message, hashlib.sha256).digest()
    return int.from_bytes(digest[:8], "big") >> 11


def choose(logits, mask, *, temperature, min_p, draw53, top_k=0, top_p=1.0,
           dry_multiplier=0., dry_base=1.75, dry_allowed_length=2,
           dry_penalty_last_n=256, dry_history=(), dry_breakers=(),
           typical_p=1., dynatemp_range=0., dynatemp_exponent=1.):
    """Choose one legal token with full-vocabulary Min-P and categorical sampling.

    ``mask`` packs token IDs least-significant-bit first. All raw logits are
    checked for finiteness before legality filtering. The caller's sequences
    and controller draw cursor are never modified.
    """
    if (not isinstance(logits, Sequence) or isinstance(logits, (str, bytes, bytearray))
            or len(logits) == 0):
        raise ValueError("logits must be a nonempty finite sequence")
    temperature = _number(temperature, "temperature")
    min_p = _number(min_p, "min_p")
    if temperature < 0:
        raise ValueError("temperature must be nonnegative")
    if not 0 <= min_p <= 1:
        raise ValueError("min_p must be in [0, 1]")
    policy = validate_policy(dict(temperature=temperature, min_p=min_p, top_k=top_k,
                                  top_p=top_p, repetition_penalty=1.0,
                                  dry_multiplier=dry_multiplier,dry_base=dry_base,
                                  dry_allowed_length=dry_allowed_length,dry_penalty_last_n=dry_penalty_last_n,
                                  typical_p=typical_p,dynatemp_range=dynatemp_range,
                                  dynatemp_exponent=dynatemp_exponent))
    top_k, top_p = policy['top_k'], policy['top_p']
    if type(draw53) is not int or not 0 <= draw53 < _U53_RANGE:
        raise ValueError("draw53 must be an integer in [0, 2^53)")

    vocab_size = len(logits)
    mask_bytes = (vocab_size + 7) // 8
    if type(mask) is not bytes or len(mask) != mask_bytes:
        raise ValueError("mask width does not match vocabulary")
    remainder = vocab_size & 7
    if remainder and mask[-1] & (0xFF ^ ((1 << remainder) - 1)):
        raise ValueError("mask has nonzero padding bits")

    scores = []
    for token, value in enumerate(logits):
        score = _number(value, f"logit[{token}]")
        scores.append(score)

    active_dry = dry_enabled(policy)
    if active_dry:
        from .dry import penalties_from_ids, FLOAT32_MAX
        if (not isinstance(dry_history,(tuple,list)) or not isinstance(dry_breakers,(tuple,list))
                or any(type(token) is not int or not 0 <= token < vocab_size
                       for token in (*dry_history,*dry_breakers))):
            raise ValueError('DRY history/breakers contain invalid vocabulary IDs')
        penalties = penalties_from_ids(dry_history,dry_breakers,
            **{key:policy[key] for key in DRY_DEFAULTS})
        for token, penalty in penalties.items():
            # Native frontier scores and their adjusted copy are binary32.
            adjusted=max(-FLOAT32_MAX,scores[token]-penalty)
            scores[token]=struct.unpack('<f',struct.pack('<f',adjusted))[0]

    legal = [token for token in range(vocab_size)
             if mask[token >> 3] & (1 << (token & 7))]
    if not legal:
        raise ValueError("mask has empty legal support")
    max_legal = max(scores[token] for token in legal)
    if temperature == 0:
        return dict(selection_protocol=DRY_PROTOCOL if active_dry else SAMPLING_PROTOCOL_V2,
                    token=max(legal, key=lambda token:(scores[token],-token)),
                    logits_count=vocab_size,mask_bytes=mask_bytes,legal_count=len(legal),
                    kept_count=1,draw53=draw53,total_weight=1.0)
    if min_p == 0:
        kept = legal
    else:
        cutoff = max_legal + math.log(min_p)
        kept = [token for token in legal if scores[token] >= cutoff]

    if top_k:
        kept = sorted(sorted(kept, key=lambda token: (-scores[token], token))[:top_k])

    active_entropy = entropy_enabled(policy)
    effective = temperature
    extra = {}
    if active_entropy:
        from .typical import retain_typical
        from .dynatemp import effective_temperature
        kept, _ = retain_typical(scores, kept, policy['typical_p'])
        typical_count = len(kept)
        effective, _ = effective_temperature(scores, kept, temperature,
            policy['dynatemp_range'], policy['dynatemp_exponent'])
        # Typical filtering can remove the original maximum. Center on the
        # remaining support so a tiny temperature cannot underflow every weight.
        max_legal = max(scores[token] for token in kept)
        extra = dict(effective_temperature=effective, typical_kept_count=typical_count)
        if effective == 0:
            return dict(selection_protocol=SAMPLING_PROTOCOL_V4,
                        token=max(kept,key=lambda token:(scores[token],-token)),
                        logits_count=vocab_size,mask_bytes=mask_bytes,legal_count=len(legal),
                        kept_count=1,draw53=draw53,total_weight=1.0,**extra)

    weights = []
    total_weight = 0.0
    for token in kept:
        weight = math.exp((scores[token] - max_legal) / effective)
        weights.append((token, weight))
        total_weight += weight
    if top_p < 1.0:
        ranked = sorted(weights, key=lambda item: (-scores[item[0]], item[0]))
        retained = set()
        cumulative = 0.0
        for token, weight in ranked:
            retained.add(token)
            cumulative += weight
            if cumulative >= top_p * total_weight:
                break
        weights = [(token, weight) for token, weight in weights if token in retained]
        kept = [token for token, _ in weights]
        total_weight = 0.0
        for _, weight in weights:
            total_weight += weight
    positive = [(token, weight) for token, weight in weights if weight > 0.0]
    if not positive or not math.isfinite(total_weight) or total_weight <= 0.0:
        raise ValueError("Min-P support has no positive categorical weight")

    target = (draw53 / _U53_RANGE) * total_weight
    cumulative = 0.0
    selected = None
    for token, weight in weights:
        if weight <= 0.0:
            continue
        cumulative += weight
        if cumulative > target:
            selected = token
            break
    if selected is None:
        selected = positive[-1][0]

    return {
        "selection_protocol": SAMPLING_PROTOCOL_V4 if active_entropy else DRY_PROTOCOL if active_dry else
            MIN_P_PROTOCOL if top_k == 0 and top_p == 1.0 else SAMPLING_PROTOCOL_V2,
        "token": selected,
        "logits_count": vocab_size,
        "mask_bytes": mask_bytes,
        "legal_count": len(legal),
        "kept_count": len(kept),
        "draw53": draw53,
        "total_weight": total_weight,
        **extra,
    }
