"""Experimental chunk-composition algebra and explicit VJPs; not a Wen engine.

NumPy is used as an inspectable FP64 reference. No PyTorch forward or backward
is used. A separate native Wen probe must qualify the actual execution path.
"""
from dataclasses import dataclass
import hashlib
import json
import math
import numpy as np


@dataclass(frozen=True)
class Schedule:
    frequencies: tuple[float, ...]
    layout: str = 'split-half'
    attention_scale: float = 1.0

    def __post_init__(self):
        if (not self.frequencies or not all(math.isfinite(x) and x > 0 for x in self.frequencies)
                or self.layout not in ('split-half', 'interleaved')
                or not math.isfinite(self.attention_scale) or self.attention_scale <= 0):
            raise ValueError('Fixed positive frequencies, layout and attention scale required')

    @property
    def identity(self):
        return hashlib.sha256(json.dumps(self.__dict__, sort_keys=True).encode()).hexdigest()

    @classmethod
    def yarn(cls, frequencies, interpolation, gamma, **kwargs):
        omega = np.asarray(frequencies, dtype=np.float64)
        blend = np.asarray(gamma, dtype=np.float64)
        if (omega.shape != blend.shape or interpolation < 1 or not math.isfinite(interpolation)
                or np.any(~np.isfinite(blend)) or np.any((blend < 0) | (blend > 1))):
            raise ValueError('Fixed YaRN interpolation and per-pair blend required')
        return cls(tuple((1-blend)*omega/interpolation + blend*omega), **kwargs)


def rotate(x, position, schedule):
    """Pure phase rotation, including partial RoPE. Do not apply YaRN gain again.

    Cached Q/K may already contain an attention gain. Relocation applies only
    the unit rotation, otherwise composition/inverse identities are incorrect.
    """
    x = np.asarray(x)
    pairs = len(schedule.frequencies)
    if x.shape[-1] < 2*pairs:
        raise ValueError('RoPE dimensions exceed head width')
    phase = np.asarray(position, dtype=np.float64)[..., None] * np.asarray(schedule.frequencies)
    cosine, sine = np.cos(phase), np.sin(phase)
    out = x.copy()
    if schedule.layout == 'split-half':
        a, b = x[..., :pairs], x[..., pairs:2*pairs]
        out[..., :pairs] = cosine*a-sine*b
        out[..., pairs:2*pairs] = sine*a+cosine*b
    else:
        a, b = x[..., :2*pairs:2], x[..., 1:2*pairs:2]
        out[..., :2*pairs:2] = cosine*a-sine*b
        out[..., 1:2*pairs:2] = sine*a+cosine*b
    return out


@dataclass(frozen=True)
class Chunk:
    key: str
    keys: np.ndarray
    values: np.ndarray
    schedule_id: str
    writer_id: str

    @classmethod
    def create(cls, keys, values, schedule, writer_id):
        k, v = np.array(keys, copy=True), np.array(values, copy=True)
        if (k.ndim != 2 or v.ndim != 2 or len(k) != len(v) or len(k) == 0
                or not np.isfinite(k).all() or not np.isfinite(v).all() or not writer_id):
            raise ValueError('Finite nonempty token matrices and Writer identity required')
        identity = dict(schedule=schedule.identity, writer=writer_id,
                        k_shape=k.shape, v_shape=v.shape, k_dtype=k.dtype.str, v_dtype=v.dtype.str)
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()+k.tobytes()+v.tobytes()).hexdigest()
        k.flags.writeable = v.flags.writeable = False
        return cls(digest, k, v, schedule.identity, writer_id)


def compose(query, occurrences, schedule, query_positions, writer_id):
    """One shared softmax over occurrences; inverse-rotate Q, retain immutable K/V.

    query is already rotated at the Reader's global position. Logical gaps do
    not allocate dummy tokens. This is one attention head, not recurrent Wen.
    """
    q = np.asarray(query)
    positions = np.asarray(query_positions)
    if (q.ndim != 2 or positions.shape != (len(q),) or not occurrences
            or not np.isfinite(q).all() or not np.isfinite(positions).all()):
        raise ValueError('Explicit finite queries, positions and occurrences required')
    blocks, values, masks, bounds = [], [], [], []
    cursor = 0
    for chunk, origin in occurrences:
        if (type(origin) is not int or origin < 0 or chunk.schedule_id != schedule.identity
                or chunk.writer_id != writer_id or chunk.keys.shape[1] != q.shape[1]):
            raise ValueError('Incompatible chunk, Writer, schedule or origin')
        for start, end in bounds:
            if origin < end and start < origin+len(chunk.keys):
                raise ValueError('Overlapping logical occurrences require a different explicit contract')
        bounds.append((origin, origin+len(chunk.keys)))
        blocks.append(rotate(q, -origin, schedule) @ chunk.keys.T / math.sqrt(q.shape[1]))
        values.append(chunk.values)
        masks.append(origin+np.arange(len(chunk.keys))[None, :] <= positions[:, None])
        cursor += len(chunk.keys)
    scores = np.concatenate(blocks, axis=1)
    valid = np.concatenate(masks, axis=1)
    if not valid.any(axis=1).all():
        raise ValueError('A query has no visible keys')
    scores = np.where(valid, scores, -np.inf)
    probabilities = np.exp(scores-np.max(scores, axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    v = np.concatenate(values, axis=0)
    output = probabilities @ v
    tape = (q.copy(), tuple(occurrences), schedule, probabilities, v, output)
    return output, tape


def compose_vjp(tape, grad_output):
    q, occurrences, schedule, p, v, output = tape
    g = np.asarray(grad_output)
    if g.shape != output.shape or not np.isfinite(g).all():
        raise ValueError('Finite output adjoint required')
    ds = p * (g @ v.T - np.sum(g*output, axis=1, keepdims=True))
    dq = np.zeros_like(q)
    chunks = {}
    offset = 0
    for chunk, origin in occurrences:
        end = offset+len(chunk.keys)
        local_q = rotate(q, -origin, schedule)
        local_ds = ds[:, offset:end] / math.sqrt(q.shape[1])
        dq += rotate(local_ds @ chunk.keys, origin, schedule)
        dk = local_ds.T @ local_q
        dv = p[:, offset:end].T @ g
        if chunk.key in chunks:
            chunks[chunk.key][0] += dk
            chunks[chunk.key][1] += dv
        else:
            chunks[chunk.key] = [dk, dv]
        offset = end
    return dq, chunks


def lora(x, weight, a, b, alpha):
    scale = alpha/a.shape[0]
    low = x @ a.T
    return x @ weight.T + scale*low @ b.T, (x, a, b, weight, low, scale)


def lora_vjp(tape, grad_output):
    x, a, b, weight, low, scale = tape
    g = np.asarray(grad_output)
    low_grad = scale*g @ b
    return g @ weight + low_grad @ a, low_grad.T @ x, scale*g.T @ low


def cross_entropy(logits, targets):
    z = np.asarray(logits)
    ids = np.asarray(targets)
    if z.ndim != 2 or ids.shape != (len(z),) or np.any(ids < 0) or np.any(ids >= z.shape[1]):
        raise ValueError('One valid target per score row required')
    shifted = z-z.max(axis=1, keepdims=True)
    logp = shifted-np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    grad = np.exp(logp)
    grad[np.arange(len(z)), ids] -= 1
    return float(-logp[np.arange(len(z)), ids].mean()), grad/len(z)


def distance_weights(distances, gamma=1.0):
    d = np.asarray(distances, dtype=np.float64)
    if d.ndim != 1 or len(d) == 0 or np.any(~np.isfinite(d)) or np.any(d < 0) or gamma < 0:
        raise ValueError('Finite nonnegative distances and gamma required')
    weights = 1+gamma*np.log2(1+d/128)
    return weights/weights.mean()
