"""Canonical mixed model inputs; no payload storage or inference lives here.

Image payloads are derived, evictable bytes. Their original input, encoder
profile and expected output identity remain in the immutable history.
"""
from dataclasses import dataclass
import json
import re

from .store import InvalidOperation, _hash, _json_bytes, _tokens


_SHA256 = re.compile(r'[0-9a-f]{64}\Z')
_IMAGE_FIELDS = frozenset((
    'kind', 'original_sha256', 'normalized_sha256', 'encoder_profile_id',
    'embedding_sha256', 'embedding_bytes', 'rows', 'width', 'grid_width',
    'grid_height', 'pad_token_id',
))
_DIGEST_FIELDS = ('original_sha256', 'normalized_sha256', 'encoder_profile_id',
                  'embedding_sha256')
_MAX_DIMENSION = 2**31 - 1


def canonical_inputs(spans):
    """Validate and detach spans, merging adjacent text without crossing images.

Native operation/batch boundaries belong to execution receipts, not arbitrary
text-span fragmentation. Image provenance and grid are part of this version's
input identity, so reuse never drops an encoder/profile binding.
"""
    if not isinstance(spans, (tuple, list)):
        raise InvalidOperation('Mixed inputs must be an ordered tuple/list of spans')
    result = []
    for span in spans:
        if not isinstance(span, dict):
            raise InvalidOperation('Each mixed input span must be an object')
        if span.get('kind') == 'text':
            if set(span) != {'kind', 'tokens'}:
                raise InvalidOperation('Unknown text input fields')
            tokens = _tokens(span['tokens'])
            if not tokens:
                continue
            if result and result[-1]['kind'] == 'text':
                result[-1]['tokens'].extend(tokens)
            else:
                result.append(dict(kind='text', tokens=tokens))
            continue
        if span.get('kind') != 'image' or set(span) != _IMAGE_FIELDS:
            raise InvalidOperation('Unknown mixed image input kind/fields')
        for key in _DIGEST_FIELDS:
            if not isinstance(span[key], str) or not _SHA256.fullmatch(span[key]):
                raise InvalidOperation('Invalid image identity: ' + key)
        for key in ('rows', 'width', 'grid_width', 'grid_height'):
            if type(span[key]) is not int or not 0 < span[key] <= _MAX_DIMENSION:
                raise InvalidOperation('Invalid image dimension: ' + key)
        if span['rows'] != span['grid_width'] * span['grid_height']:
            raise InvalidOperation('Image row count does not match its rectangular grid')
        expected_bytes = 20 + span['rows'] * span['width'] * 4
        if (type(span['embedding_bytes']) is not int or expected_bytes >= 2**63
                or span['embedding_bytes'] != expected_bytes):
            raise InvalidOperation('Image byte length does not match SVE1 dimensions')
        _tokens([span['pad_token_id']])
        # All accepted image fields are immutable scalars.
        result.append(dict(span))
    return result


def input_count(spans):
    """Count consumed model cells, including every image embedding row."""
    return sum(len(span['tokens']) if span['kind'] == 'text' else span['rows']
               for span in canonical_inputs(spans))


def inputs_digest(spans):
    return _hash(_json_bytes(dict(schema=1, spans=canonical_inputs(spans))))


def text_runs(spans):
    """Actual text runs; image boundaries must break repetition/DRY matching."""
    return tuple(tuple(span['tokens']) for span in canonical_inputs(spans)
                 if span['kind'] == 'text')


@dataclass(frozen=True, init=False)
class InputHistory:
    """Detached read view of canonical spans, never another persisted history.

    Deliberately not iterable as tokens: callers must choose model cells or
    actual text explicitly. Native projection must never become cache identity.
    """
    _encoded: bytes
    _count: int

    def __init__(self, spans):
        canonical = canonical_inputs(spans)
        object.__setattr__(self, '_encoded', _json_bytes(dict(schema=1, spans=canonical)))
        object.__setattr__(self, '_count', input_count(canonical))

    @property
    def spans(self):
        return json.loads(self._encoded)['spans']

    @property
    def digest(self):
        return _hash(self._encoded)

    @property
    def native_tokens(self):
        return tuple(token for span in self.spans for token in
                     (span['tokens'] if span['kind'] == 'text'
                      else [span['pad_token_id']]*span['rows']))

    @property
    def text_runs(self):
        return text_runs(self.spans)

    @property
    def text_tokens(self):
        # Frequency counts may span images. Sequence matching must use text_runs.
        return tuple(token for run in self.text_runs for token in run)

    def __len__(self):
        return self._count

    def __add__(self, tokens):
        if isinstance(tokens,InputHistory):
            return InputHistory(self.spans+tokens.spans)
        return InputHistory(self.spans + [dict(kind='text', tokens=list(tokens))])

    def starts_with(self, prefix):
        """Compare already validated immutable canonical bytes, not text meaning.

        Both objects own schema-1 canonical JSON from this class's constructor.
        An extension either appends spans or extends the last merged text span.
        Require the comma delimiter: token1 must not match token10. Counts,
        complete image metadata and all earlier spans remain exact. No history,
        hashes or decoder state are rewritten; no mutable proof is cached.
        Untrusted loading still validates/canonicalizes inputs in the constructor.
        """
        if not isinstance(prefix, InputHistory):
            raise TypeError('Expected an explicit input history prefix')
        if len(prefix) > len(self):
            return False
        if len(prefix) == len(self):
            return self == prefix
        if not prefix:
            return True
        raw = prefix._encoded
        # The root array/object close is ]}\n. An extra complete span follows
        # its predecessor with a comma; compare buffers without copying them.
        head = memoryview(raw)[:-3]
        if (self._encoded.startswith(head) and len(self._encoded) > len(head)
                and self._encoded[len(head)] == ord(',')):
            return True
        # A final text span ends ]}]}\n. Canonical adjacent text is merged,
        # so appended token IDs follow the old last token with a comma.
        if raw.endswith(b']}]}\n'):
            head = memoryview(raw)[:-5]
            return (self._encoded.startswith(head) and len(self._encoded) > len(head)
                    and self._encoded[len(head)] == ord(','))
        return False

    def segment(self, start, stop=None):
        """Read model cells without turning image rows into sampled text IDs."""
        stop = len(self) if stop is None else stop
        if (type(start) is not int or type(stop) is not int
                or not 0 <= start <= stop <= len(self)):
            raise InvalidOperation('Invalid mixed-input segment boundary')
        result, cell = [], 0
        for span in self.spans:
            size = len(span['tokens']) if span['kind']=='text' else span['rows']
            end = cell+size
            if span['kind']=='image' and (cell < start < end or cell < stop < end):
                raise InvalidOperation('Public boundary splits an image span')
            if max(start,cell) < min(stop,end):
                if span['kind']=='image':result.append(span)
                else:result.append(dict(kind='text',tokens=span['tokens'][max(start-cell,0):min(stop-cell,size)]))
            cell = end
        return InputHistory(result)

    def prefix(self, count):
        if type(count) is not int or not 0 <= count <= len(self):
            raise InvalidOperation('Invalid mixed-input prefix boundary')
        remaining, result = count, []
        for span in self.spans:
            if not remaining:
                break
            size = len(span['tokens']) if span['kind'] == 'text' else span['rows']
            if size <= remaining:
                result.append(span)
            elif span['kind'] == 'image':
                raise InvalidOperation('Public boundary splits an image span')
            else:
                result.append(dict(kind='text', tokens=span['tokens'][:remaining]))
            remaining -= min(size, remaining)
        return InputHistory(result)


def history_lineage(store, snapshot):
    """Read the one canonical Store history using its declared input version."""
    if store.get_snapshot(snapshot)['schema']==2:
        root, spans = store.input_lineage(snapshot)
        return root, InputHistory(spans)
    root, ids = store.lineage(snapshot)
    return root, tuple(ids)


def history_for_snapshot(store, snapshot):
    return history_lineage(store,snapshot)[1]


def history_segment(history, start, stop=None):
    stop = len(history) if stop is None else stop
    if isinstance(history,InputHistory):return history.segment(start,stop)
    if (type(start) is not int or type(stop) is not int
            or not 0 <= start <= stop <= len(history)):
        raise InvalidOperation('Invalid token history segment boundary')
    return tuple(history[start:stop])


def history_digest(history):
    if isinstance(history,InputHistory):return history.digest
    return _hash(_json_bytes(list(history)))


def generated_text(history):
    if isinstance(history,InputHistory):
        if any(span['kind']=='image' for span in history.spans):
            raise InvalidOperation('Generated output must contain actual text, not image rows')
        return history.text_tokens
    return tuple(history)
