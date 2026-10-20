"""Transient complete-schedule validation projections, never persisted state.

Parse and independently hash canonical inputs on every validation. Bindings
computed with the original SHA-256 equations may be reused for that exact
canonical prefix. Every receipt field is still compared. This process-local,
bounded cache contains only derived hashes, never history or model state.
"""
from bisect import bisect_right
from collections import OrderedDict
from threading import Lock
import hashlib
import struct
from .input_spans import canonical_inputs
from .store import _json_bytes


class _BindingCache:
    """Bounded derived bindings. Callers must independently verify input identity.

    Keys use the canonical prefix digest, consumed count and rotary cursor, not
    receipt-supplied identities or object addresses. Values are immutable hash
    tuples; neither history bytes nor mutable receipt dictionaries are retained.
    The module owns the equation version. Changing equations needs a new module
    execution identity, never cache migration. Races may duplicate computation;
    locks cover lookup/publication only, not hashing or model execution.
    """
    def __init__(self, capacity):
        if type(capacity) is not int or capacity <= 0:
            raise ValueError('Binding cache capacity must be a positive integer')
        self._capacity = capacity
        self._entries = OrderedDict()
        self._lock = Lock()

    def get(self, key):
        with self._lock:
            value = self._entries.get(key)
            if value is not None:
                self._entries.move_to_end(key)
            return value

    def put(self, key, value):
        with self._lock:
            self._entries[key] = value
            self._entries.move_to_end(key)
            while len(self._entries) > self._capacity:
                self._entries.popitem(last=False)

    def __len__(self):
        with self._lock:
            return len(self._entries)


# Small metadata only: 65,536 entries, irrespective of prefix/model byte size.
# Eviction changes work, never correctness; no persistence or user API flag.
_BINDINGS = _BindingCache(65536)


def _hash_binding(projection, end):
    """Original native/position equations, unchanged even on a cache miss."""
    count = bisect_right(projection._image_ends, end)
    native = hashlib.sha256(b'COW-VISION-INPUT-v1\0')
    native.update(struct.pack('<Q', end))
    native.update(projection._native[:4*end])
    native.update(struct.pack('<Q', count))
    native.update(projection._images[:projection._image_offsets[count]])
    cursor = projection._cursors[end]
    positional = hashlib.sha256(b'COW-MROPE-v1\0')
    positional.update(struct.pack('<QQ', end, cursor))
    positional.update(projection._positions[:12*end])
    return native.hexdigest(), positional.hexdigest()


class _InputScheduleProjection:
    def __init__(self, history, ends):
        spans = canonical_inputs(history.spans)
        wanted = set(ends)
        if any(type(n) is not int or not 0 <= n <= len(history) for n in wanted):
            raise ValueError('Invalid mixed request boundary')
        empty = _json_bytes(dict(schema=1, spans=[]))
        text = _json_bytes(dict(kind='text', tokens=[]))
        if not empty.endswith(b'[]}\n') or not text.endswith(b'[]}\n'):
            raise ValueError('Unsupported canonical input encoding')
        root_close, text_close = b']}\n', b']}'
        canonical = hashlib.sha256(empty[:-3])
        self._digests = {}
        ids, native, positions = [], bytearray(), bytearray()
        self._cursors = [0]
        self._text_ends = bytearray([0])
        self._image_starts, self._image_ends = [], []
        image_bytes, self._image_offsets = bytearray(), [0]
        cursor, cell = 0, 0
        for index, span in enumerate(spans):
            if index:
                canonical.update(b',')
            if span['kind'] == 'text':
                canonical.update(text[:-3])
                for offset, token in enumerate(span['tokens']):
                    if offset:
                        canonical.update(b',')
                    canonical.update(str(token).encode('ascii'))
                    ids.append(token)
                    native.extend(struct.pack('<i', token))
                    positions.extend(struct.pack('<iii', cursor, cursor, cursor))
                    cursor += 1
                    cell += 1
                    self._cursors.append(cursor)
                    self._text_ends.append(1)
                    if cell in wanted:
                        prefix = canonical.copy()
                        prefix.update(text_close + root_close)
                        self._digests[cell] = prefix.hexdigest()
                canonical.update(text_close)
            else:
                canonical.update(_json_bytes(span)[:-1])
                start = cell
                self._image_starts.append(start)
                for y in range(span['grid_height']):
                    for x in range(span['grid_width']):
                        ids.append(span['pad_token_id'])
                        native.extend(struct.pack('<i', span['pad_token_id']))
                        positions.extend(struct.pack('<iii', cursor, cursor+y, cursor+x))
                        cell += 1
                        self._cursors.append(0)  # Interior image boundaries are forbidden.
                        self._text_ends.append(0)
                cursor += max(span['grid_width'], span['grid_height'])
                self._cursors[-1] = cursor
                self._image_ends.append(cell)
                image_bytes.extend(struct.pack('<6q', start, span['rows'],
                    span['grid_width'], span['grid_height'], span['width'], span['pad_token_id']))
                image_bytes.extend(span['embedding_sha256'].encode('ascii'))
                self._image_offsets.append(len(image_bytes))
        canonical.update(root_close)
        if cell != len(history) or canonical.hexdigest() != history.digest:
            raise ValueError('Canonical mixed history identity mismatch')
        self.native_tokens = tuple(ids)
        self._native = memoryview(bytes(native))
        self._positions = memoryview(bytes(positions))
        self._images = memoryview(bytes(image_bytes))

    def boundary(self, position):
        if type(position) is not int or not 0 <= position <= len(self.native_tokens):
            raise ValueError('Invalid mixed request boundary')
        image = bisect_right(self._image_starts, position)-1
        if image >= 0 and self._image_starts[image] < position < self._image_ends[image]:
            raise ValueError('Execution boundary splits an image')

    def new_image(self, start, end):
        count = bisect_right(self._image_ends, end)
        return count > 0 and self._image_starts[count-1] >= start

    def binding(self, end):
        self.boundary(end)
        if not self._text_ends[end] or end not in self._digests:
            raise ValueError('Execution boundary must follow completed image input')
        digest, cursor = self._digests[end], self._cursors[end]
        key = (digest, end, cursor)
        value = _BINDINGS.get(key)
        if value is None:
            value = _hash_binding(self, end)
            _BINDINGS.put(key, value)
        return dict(inputs_sha256=digest, native_input_sha256=value[0],
                    position_sha256=value[1], rotary_cursor=cursor)
