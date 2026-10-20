"""Independent Adaptive-P reference after the existing probability filters.

Only the raw frontier and explicit controller state enter this reference. It
does not import the native selector or reconstruct logits from probabilities.
All reductions and categorical sampling visit ascending token IDs.
"""
from __future__ import annotations

import math
import struct
from collections.abc import Sequence

from . import adaptive_p, min_p


def choose_v7(logits, mask, state, *, draw53, **settings):
    defaults = dict(temperature=1., min_p=0., top_k=0, top_p=1.,
                    repetition_penalty=1., **min_p.DRY_DEFAULTS,
                    typical_p=1., **min_p.DYNA_DEFAULTS,
                    adaptive_target=-1., adaptive_decay=.9)
    defaults.update({key: value for key, value in settings.items()
                     if key not in ('dry_history', 'dry_breakers')})
    policy = min_p.validate_policy(defaults)
    if not min_p.adaptive_enabled(policy):
        raise ValueError('The V7 reference requires active Adaptive-P')
    state = adaptive_p.validate_state(state)
    if type(draw53) is not int or not 0 <= draw53 < (1 << 53):
        raise ValueError('draw53 must be an integer in [0, 2^53)')
    if (not isinstance(logits, Sequence)
            or isinstance(logits, (str, bytes, bytearray)) or not logits):
        raise ValueError('logits must be a nonempty finite sequence')
    vocab = len(logits)
    if type(mask) is not bytes or len(mask) != (vocab + 7) // 8:
        raise ValueError('mask width does not match vocabulary')
    remainder = vocab & 7
    if remainder and mask[-1] & (0xff ^ ((1 << remainder) - 1)):
        raise ValueError('mask has nonzero padding bits')
    scores = [min_p._number(value, f'logit[{i}]') for i, value in enumerate(logits)]
    legal = [i for i in range(vocab) if mask[i >> 3] & (1 << (i & 7))]
    if not legal:
        raise ValueError('mask has empty legal support')
    if min_p.dry_enabled(policy):
        from .dry import FLOAT32_MAX, penalties_from_ids
        history, breakers = settings.get('dry_history', ()), settings.get('dry_breakers', ())
        if (not isinstance(history, (tuple, list)) or not isinstance(breakers, (tuple, list))
                or any(type(token) is not int or not 0 <= token < vocab
                       for token in (*history, *breakers))):
            raise ValueError('DRY history/breakers contain invalid vocabulary IDs')
        penalties = penalties_from_ids(history, breakers,
            **{key: policy[key] for key in min_p.DRY_DEFAULTS})
        for token, penalty in penalties.items():
            adjusted = max(-FLOAT32_MAX, scores[token] - penalty)
            scores[token] = struct.unpack('<f', struct.pack('<f', adjusted))[0]

    maximum = max(scores[token] for token in legal)
    if policy['min_p'] == 0.:
        kept = list(legal)
    else:
        cutoff = maximum + math.log(policy['min_p'])
        kept = [token for token in legal if scores[token] >= cutoff]
    min_p_count = len(kept)
    if policy['top_k']:
        kept = sorted(sorted(kept, key=lambda token: (-scores[token], token))[:policy['top_k']])
    top_k_count = len(kept)
    effective = policy['temperature']
    if min_p.entropy_enabled(policy):
        from .typical import retain_typical
        from .dynatemp import effective_temperature
        kept, _ = retain_typical(scores, kept, policy['typical_p'])
        effective, _ = effective_temperature(scores, kept, policy['temperature'],
            policy['dynatemp_range'], policy['dynatemp_exponent'])
        maximum = max(scores[token] for token in kept)
    typical_count = len(kept)
    if effective == 0.:
        kept = [max(kept, key=lambda token: (scores[token], -token))]
        weights = [(kept[0], 1.)]
    else:
        weights = [(token, math.exp((scores[token] - maximum) / effective)) for token in kept]
        total = 0.
        for _, weight in weights:
            total += weight
        if policy['top_p'] < 1.:
            ranked = sorted(weights, key=lambda item: (-scores[item[0]], item[0]))
            retained = set()
            cumulative = 0.
            for token, weight in ranked:
                retained.add(token)
                cumulative += weight
                if cumulative >= policy['top_p'] * total:
                    break
            weights = [(token, weight) for token, weight in weights if token in retained]
            kept = [token for token, _ in weights]

    original_total = 0.
    for _, weight in weights:
        original_total += weight
    if not math.isfinite(original_total) or original_total <= 0.:
        raise ValueError('Adaptive-P input support has no finite positive mass')
    original = [(token, weight / original_total) for token, weight in weights]
    adapted = min(1., max(0., 2. * policy['adaptive_target']
                         - state['weighted_sum'] / state['total_weight']))
    transformed = []
    for token, probability in original:
        distance = abs((probability - adapted) / .3)
        transformed.append((token, 5. - 10. * distance * distance / (1. + distance)))
    center = max(value for _, value in transformed)
    raw = [(token, math.exp(value - center)) for token, value in transformed]
    transformed_total = 0.
    for _, weight in raw:
        transformed_total += weight
    if not math.isfinite(transformed_total) or transformed_total <= 0.:
        raise ValueError('Adaptive-P transformed support has no finite positive mass')
    categorical = [(token, weight / transformed_total) for token, weight in raw]
    token = adaptive_p._draw_from_weights(categorical, draw53)
    original_probability = dict(original)[token]
    feedback = adaptive_p.validate_state(dict(state,
        weighted_sum=original_probability + policy['adaptive_decay'] * state['weighted_sum'],
        total_weight=1. + policy['adaptive_decay'] * state['total_weight']))
    return dict(selection_protocol=min_p.SAMPLING_PROTOCOL_V7, token=token, draw53=draw53,
                draw_count=1, legal_count=len(legal), kept_count=len(kept), kept_ids=kept,
                min_p_kept_count=min_p_count, top_k_kept_count=top_k_count,
                typical_kept_count=typical_count, effective_temperature=effective,
                original_probability=original_probability, adapted_target=adapted,
                feedback_state=feedback)
