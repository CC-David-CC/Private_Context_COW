"""Reuse only independently keyed, immutable input/position hash results.

The complete reference equations stay in vision_execution.input_binding. The
existing bounded complete-schedule cache also holds these same pure bindings.
No model state or history is stored here; native checks still query every time.
"""
import hashlib
from .input_spans import InputHistory
from .vision_execution import input_binding as reference_input_binding
from . import input_schedule_projection as projection


def cached_input_binding(history):
    views=history._owned_views() if type(history) is InputHistory else None
    if views is None:return reference_input_binding(history)
    encoded=history._encoded
    cursor=sum(len(data) if kind=='text' else max(dict(data)['grid_width'],dict(data)['grid_height'])
               for kind,data in views)
    key=(hashlib.sha256(encoded).hexdigest(),len(history),cursor)
    pair=projection._BINDINGS.get(key)
    if pair is None:
        value=reference_input_binding(history)
        if value['rotary_cursor']!=cursor or history._encoded is not encoded:
            raise ValueError('Input changed while deriving its binding')
        pair=(value['native_input_sha256'],value['position_sha256'])
        projection._BINDINGS.put(key,pair)
    if history._encoded is not encoded:raise ValueError('Input changed while reading its binding')
    return dict(native_input_sha256=pair[0],position_sha256=pair[1],rotary_cursor=cursor)
