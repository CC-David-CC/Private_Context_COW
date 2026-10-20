"""Bounded, exact SVE1 payload admission for the pinned vision encoder.

An accepted payload is immutable derived data, not a second input history.
The owner supplies the expected digest, model width and profile row bound.
"""
from dataclasses import dataclass
import errno
import hashlib
import math
import os
import re
import stat
import struct

from .store import IntegrityError, InvalidOperation, _hash, _json_bytes


_HEADER = struct.Struct('<5i')
_MAGIC = 0x31455653
_DIGEST = re.compile(r'[0-9a-f]{64}\Z')


def _digest(value):
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        raise InvalidOperation('Expected a complete lowercase SHA-256 digest')
    return value


def _bound(expected_sha256, expected_width, max_rows):
    _digest(expected_sha256)
    for value in (expected_width, max_rows):
        if type(value) is not int or not 0 < value < 2**31:
            raise InvalidOperation('Model width and row limit must be positive int32 values')
    maximum = _HEADER.size + max_rows * expected_width * 4
    if maximum >= 2**63:
        raise InvalidOperation('Embedding size bound exceeds the supported byte range')
    return maximum


def encoding_key(normalized_sha256, encoder_profile_id):
    """Index derived bytes by full input and encoder identity, never work ID."""
    return _hash(_json_bytes(dict(schema=1, kind='image-encoding',
        normalized_sha256=_digest(normalized_sha256),
        encoder_profile_id=_digest(encoder_profile_id))))


@dataclass(frozen=True)
class Embedding:
    payload: bytes
    sha256: str
    rows: int
    grid_width: int
    grid_height: int
    width: int

    @property
    def nbytes(self):
        return len(self.payload)

    @property
    def data(self):
        return memoryview(self.payload)[_HEADER.size:]


def parse_sve(payload, *, expected_sha256, expected_width, max_rows=1024):
    maximum = _bound(expected_sha256, expected_width, max_rows)
    if type(payload) is not bytes:
        raise InvalidOperation('Embedding admission requires immutable bytes')
    if not _HEADER.size <= len(payload) <= maximum:
        raise IntegrityError('SVE1 byte length is outside the configured limit')
    magic, rows, nx, ny, width = _HEADER.unpack_from(payload)
    if magic != _MAGIC:
        raise IntegrityError('Invalid SVE1 magic or endian convention')
    if not 0 < rows <= max_rows or nx <= 0 or ny <= 0 or rows != nx * ny:
        raise IntegrityError('SVE1 rows must form a rectangular grid within the row limit')
    if width != expected_width:
        raise IntegrityError('SVE1 width differs from the loaded model')
    if len(payload) != _HEADER.size + rows * width * 4:
        raise IntegrityError('SVE1 length differs from its dimensions; extra chunks are unsupported')
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected_sha256:
        raise IntegrityError('Embedding payload digest mismatch')
    # Iterate the wire-endian floats without a second materialized float array.
    data = memoryview(payload)[_HEADER.size:]
    if any(not math.isfinite(value) for (value,) in struct.iter_unpack('<f', data)):
        raise IntegrityError('SVE1 contains non-finite embedding values')
    return Embedding(payload, actual, rows, nx, ny, width)


def read_sve(path, *, expected_sha256, expected_width, max_rows=1024):
    """Read one regular bounded artifact; reject links, FIFOs and changed files."""
    maximum = _bound(expected_sha256, expected_width, max_rows)
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise IntegrityError('Embedding file must not be a symbolic link') from exc
        raise
    with os.fdopen(fd, 'rb') as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise IntegrityError('Embedding file must be regular')
        if not _HEADER.size <= before.st_size <= maximum:
            raise IntegrityError('Embedding file size exceeds its configured limit')
        payload = stream.read(before.st_size + 1)
        after = os.fstat(stream.fileno())
        if (len(payload) != before.st_size or
                (before.st_size, before.st_mtime_ns, before.st_ctime_ns) !=
                (after.st_size, after.st_mtime_ns, after.st_ctime_ns)):
            raise IntegrityError('Embedding file changed during the read')
    return parse_sve(payload, expected_sha256=expected_sha256,
                     expected_width=expected_width, max_rows=max_rows)
