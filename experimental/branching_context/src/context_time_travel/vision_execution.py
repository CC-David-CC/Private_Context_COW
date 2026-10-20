"""Transient native projection of canonical mixed inputs and receipt checks."""
import hashlib
import struct

from .input_spans import InputHistory
from .store import InvalidOperation


VISION_PROTOCOL = 'explicit-image-rows-mrope-v3'
RESUME_CONTRACT = 'identical-mixed-native-operators-v1'
POSITION_CONVENTION = 'qwen-image-grid-mrope-v1'


def placements(history):
    if not isinstance(history, InputHistory):
        raise InvalidOperation('Mixed execution requires explicit canonical input spans')
    result, cell = [], 0
    for span in history.spans:
        if span['kind'] == 'image':
            result.append(dict(start=cell, rows=span['rows'], grid_width=span['grid_width'],
                grid_height=span['grid_height'], width=span['width'],
                pad_token=span['pad_token_id'], embedding_sha256=span['embedding_sha256']))
            cell += span['rows']
        else:
            cell += len(span['tokens'])
    return result


def input_binding(history):
    images = placements(history)
    ids = history.native_tokens
    native = hashlib.sha256(b'COW-VISION-INPUT-v1\0')
    native.update(struct.pack('<Q', len(ids)))
    native.update(struct.pack('<'+'i'*len(ids), *ids))
    native.update(struct.pack('<Q', len(images)))
    for image in images:
        native.update(struct.pack('<6q', *(image[k] for k in
            ('start','rows','grid_width','grid_height','width','pad_token'))))
        native.update(image['embedding_sha256'].encode('ascii'))
    cursor, positions = 0, bytearray()
    for span in history.spans:
        if span['kind'] == 'text':
            for _ in span['tokens']:
                positions.extend(struct.pack('<iii', cursor, cursor, cursor))
                cursor += 1
        else:
            for y in range(span['grid_height']):
                for x in range(span['grid_width']):
                    positions.extend(struct.pack('<iii', cursor, cursor+y, cursor+x))
            cursor += max(span['grid_width'],span['grid_height'])
    positional = hashlib.sha256(b'COW-MROPE-v1\0'+struct.pack('<QQ',len(ids),cursor)+positions)
    return dict(native_input_sha256=native.hexdigest(), position_sha256=positional.hexdigest(),
                rotary_cursor=cursor)


def validate_boundary(history, *, capacity, vocab, width):
    if not isinstance(history, InputHistory) or not 0 < len(history) <= capacity:
        raise InvalidOperation('Invalid mixed consumed boundary or capacity')
    spans = history.spans
    if spans[-1]['kind'] != 'text':
        raise InvalidOperation('Consumed prompt must include text after its last image')
    for span in spans:
        if span['kind'] == 'image' and (span['width'] != width or span['rows'] > 1024):
            raise InvalidOperation('Image span exceeds the pinned model geometry')
    if any(not 0 <= token < vocab for token in history.native_tokens):
        raise InvalidOperation('Mixed input token is outside vocabulary')
