"""Bounded in-process attestations for exact immutable checkpoint owners.

This is only a host-side digest memo for RAM objects. It does not certify disk
contents or native/GPU state, and it does not replace canonical binding checks.
"""
from collections import OrderedDict
import hashlib
import math
import struct
import threading
import weakref


MAX_ATTESTED_OWNERS = 32
_LOCK = threading.RLock()
_OWNERS = OrderedDict()


def _compute_digest(blob, frontier):
    state_identity = blob if isinstance(blob, bytes) else blob.identity
    digest = hashlib.sha256(state_identity)
    digest.update(struct.pack('<' + str(len(frontier)) + 'f', *frontier))
    return digest.hexdigest()


def _eligible(owner):
    blob = getattr(owner, 'blob', None)
    frontier = getattr(owner, 'frontier', None)
    try:
        return (type(blob) is bytes and bool(blob) and type(frontier) is tuple and bool(frontier)
                and all(type(value) in (int, float) and math.isfinite(value) for value in frontier)
                and type(getattr(owner, 'checksum', None)) is str)
    except (OverflowError, TypeError, ValueError):
        return False


def _drop(key, owner_ref):
    with _LOCK:
        entry = _OWNERS.get(key)
        if entry is not None and entry[0] is owner_ref:
            del _OWNERS[key]


def _remember(owner, digest):
    if not _eligible(owner) or owner.checksum != digest:
        return
    key = id(owner)
    try:
        owner_ref = weakref.ref(owner, lambda ref, key=key: _drop(key, ref))
    except TypeError:
        # Weak-reference support is an optimization constraint, not part of
        # the existing checkpoint admission contract. Keep hashing normally.
        return
    entry = (owner_ref, owner.blob, owner.frontier, owner.checksum, digest)
    with _LOCK:
        _OWNERS[key] = entry
        _OWNERS.move_to_end(key)
        while len(_OWNERS) > MAX_ATTESTED_OWNERS:
            _OWNERS.popitem(last=False)


def verified_checkpoint_digest(owner):
    """Return the actual RAM digest, reusing only the exact certified owner."""
    key = id(owner)
    with _LOCK:
        entry = _OWNERS.get(key)
        if entry is not None:
            owner_ref, blob, frontier, checksum, digest = entry
            # The retained tuple was eligible when certified: its elements
            # were exact immutable builtin numbers, and the strong reference
            # below guarantees this exact tuple is still present. Rechecking
            # every element here would scan large logit frontiers per lookup.
            if (owner_ref() is owner and owner.blob is blob and owner.frontier is frontier
                    and type(owner.checksum) is str and owner.checksum == checksum == digest):
                _OWNERS.move_to_end(key)
                return digest
            del _OWNERS[key]
    digest = _compute_digest(owner.blob, owner.frontier)
    if digest == getattr(owner, 'checksum', None):
        _remember(owner, digest)
    return digest


def _attest_factory_result(owner):
    """Hash an exact factory result, set its checksum, then attest that owner."""
    digest = _compute_digest(owner.blob, owner.frontier)
    object.__setattr__(owner, 'checksum', digest)
    _remember(owner, digest)
    return digest


def attested_owner_count():
    """Bounded diagnostics for tests; not a persistence or health metric."""
    with _LOCK:
        return len(_OWNERS)
