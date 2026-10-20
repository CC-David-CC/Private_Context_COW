"""Fresh byte construction for the unchanged input/position SHA-256 equations.

Only transient primitive arrays: no history, receipt or model-state cache, no
skipped integrity checks. vision_execution.input_binding remains the independent
scalar oracle. Exact owned InputHistory uses array slice assignment in C rather
than packing three integers in Python per text token. Unowned/subclass views
retain the complete reference path. This module is not yet GPU-qualified.
"""
from array import array
import hashlib
import struct
import sys
from .input_spans import InputHistory
from .vision_execution import input_binding as reference_input_binding


def _array_bytes(words):
    if words.itemsize != 4:
        return struct.pack('<'+'i'*len(words),*words)
    if sys.byteorder != 'little':
        words=words[:]
        words.byteswap()
    return words.tobytes()


def _i32_bytes(values):
    try:
        return _array_bytes(array('i',values))
    except OverflowError as error:
        raise struct.error('input or position is outside signed int32') from error


def _text_positions(start,count):
    if count and (start < -2147483648 or start+count-1 > 2147483647):
        raise struct.error('position is outside signed int32')
    word=array('i',range(start,start+count))
    triples=array('i',[0])*(3*count)
    triples[0::3]=word
    triples[1::3]=word
    triples[2::3]=word
    return _array_bytes(triples)


def _position_bytes(spans):
    cursor=0
    positions=bytearray()
    for span in spans:
        if span['kind']=='text':
            count=len(span['tokens'])
            positions.extend(_text_positions(cursor,count))
            cursor+=count
        else:
            nx,ny=span['grid_width'],span['grid_height']
            if cursor+max(nx,ny)-1 > 2147483647:
                raise struct.error('image position is outside signed int32')
            words=array('i',[0])*(3*nx*ny)
            words[0::3]=array('i',[cursor])*(nx*ny)
            words[1::3]=array('i',(cursor+y for y in range(ny) for _ in range(nx)))
            words[2::3]=array('i',range(cursor,cursor+nx))*ny
            positions.extend(_array_bytes(words))
            cursor+=max(nx,ny)
    return cursor,bytes(positions)


def fresh_input_binding(history):
    views=history._owned_views() if type(history) is InputHistory else None
    if views is None:
        return reference_input_binding(history)
    encoded=history._encoded
    spans=[dict(kind='text',tokens=data) if kind=='text' else dict(data)
           for kind,data in views]
    ids=history.native_tokens
    images=[]
    cell=0
    for span in spans:
        if span['kind']=='text':
            cell+=len(span['tokens'])
        else:
            images.append((cell,span))
            cell+=span['rows']
    native=hashlib.sha256(b'COW-VISION-INPUT-v1\0')
    native.update(struct.pack('<Q',len(ids)))
    native.update(_i32_bytes(ids))
    native.update(struct.pack('<Q',len(images)))
    for start,image in images:
        native.update(struct.pack('<6q',start,image['rows'],image['grid_width'],
                                  image['grid_height'],image['width'],image['pad_token_id']))
        native.update(image['embedding_sha256'].encode('ascii'))
    cursor,positions=_position_bytes(spans)
    positional=hashlib.sha256(b'COW-MROPE-v1\0')
    positional.update(struct.pack('<QQ',len(ids),cursor))
    positional.update(positions)
    if history._encoded is not encoded:
        raise ValueError('Input changed while computing a fresh binding')
    return dict(native_input_sha256=native.hexdigest(),position_sha256=positional.hexdigest(),
                rotary_cursor=cursor)
