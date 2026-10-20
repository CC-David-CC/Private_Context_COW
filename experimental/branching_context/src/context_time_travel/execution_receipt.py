"""Derived receipts for successful native batches; canonical inputs live in Store."""
import hashlib
import json
import struct


def tokens_digest(ids):
    return hashlib.sha256(struct.pack('<'+'q'*len(ids), *ids)).hexdigest()


def schedule_digest(schedule):
    return hashlib.sha256(json.dumps(schedule, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def validate_schedule(schedule, history, chunk_limit):
    return _validate_schedule(schedule, history, chunk_limit, {'batched-prefill'})


def validate_operator_schedule(schedule, history, chunk_limit):
    return _validate_schedule(schedule, history, chunk_limit, {'batched-prefill','normal-verifier-decode'})


def validate_input_schedule(schedule, history, chunk_limit):
    """Check every mixed receipt using one canonical/native byte projection."""
    from .input_spans import InputHistory
    from .input_schedule_projection import _InputScheduleProjection
    from .store import InvalidOperation
    if not isinstance(history, InputHistory) or not isinstance(schedule,(list,tuple)):
        raise ValueError('Missing explicit mixed history or schedule')
    legacy_keys = {'operator','position','count','tokens_sha256','chunks'}
    binding_keys = {'inputs_sha256','native_input_sha256','position_sha256','rotary_cursor'}
    legacy, ends = [], []
    for event in schedule:
        if not isinstance(event,dict) or set(event) != legacy_keys | binding_keys:
            raise ValueError('Unknown mixed execution receipt layout')
        if type(event['position']) is not int or type(event['count']) is not int:
            raise ValueError('Invalid mixed request boundary')
        ends.append(event['position']+event['count'])
        legacy.append({key:event[key] for key in legacy_keys})
    try:
        projection = _InputScheduleProjection(history,ends)
    except InvalidOperation as error:
        raise ValueError('Invalid canonical mixed history') from error
    for event, end in zip(schedule,ends):
        projection.boundary(event['position'])
        expected = projection.binding(end)
        if (type(event['rotary_cursor']) is not int or
                any(event[k] != value for k,value in expected.items())):
            raise ValueError('Mixed execution receipt input/position identity mismatch')
        if event['operator']=='normal-verifier-decode' and projection.new_image(event['position'],end):
            raise ValueError('Decode cannot consume image rows')
    validate_operator_schedule(legacy,projection.native_tokens,chunk_limit)
    return schedule_digest(schedule)


def _validate_schedule(schedule, history, chunk_limit, operators):
    if not isinstance(schedule, (list, tuple)) or type(chunk_limit) is not int or chunk_limit <= 0:
        raise ValueError('Missing or invalid native execution schedule')
    position = 0
    for event in schedule:
        if not isinstance(event, dict) or set(event) != {'operator','position','count','tokens_sha256','chunks'}:
            raise ValueError('Unknown execution receipt layout')
        count = event['count']
        if (not isinstance(event['operator'], str) or event['operator'] not in operators or type(count) is not int or count <= 0
                or type(event['position']) is not int or event['position'] != position
                or position+count > len(history)):
            raise ValueError('Unsupported operator or noncontiguous request')
        if event['tokens_sha256'] != tokens_digest(history[position:position+count]):
            raise ValueError('Execution receipt token identity mismatch')
        chunks = event['chunks']
        if not isinstance(chunks, (list, tuple)) or not chunks:
            raise ValueError('Missing completed native chunks')
        cursor = position
        for entry in chunks:
            if (not isinstance(entry, (list, tuple)) or len(entry) != 2
                    or any(type(v) is not int for v in entry)
                    or entry[0] != cursor or not 0 < entry[1] <= chunk_limit
                    or (event['operator'] == 'normal-verifier-decode' and entry[1] != 1)
                    or cursor+entry[1] > position+count):
                raise ValueError('Invalid completed native chunk boundary')
            cursor += entry[1]
        if cursor != position+count:
            raise ValueError('Incomplete native request')
        position += count
    if position != len(history):
        raise ValueError('Schedule does not cover consumed history')
    return schedule_digest(schedule)


class _InputScheduleCursor:
    """An owner-local proof cursor, never a persisted history or native state.

    Audit the complete schedule on entry and at publication. Between those
    boundaries, the immutable owned prefix is already checked: validate just
    the new native receipt against its exact extension and current input hash.
    Receipt dictionaries never escape until this cursor is closed. Keep this
    private; untrusted/disk schedules must still use the complete validator.
    """
    def __init__(self, schedule, history, chunk_limit):
        import copy
        import threading
        owned = copy.deepcopy(tuple(schedule))
        validate_input_schedule(owned, history, chunk_limit)
        self._owner = threading.current_thread()
        self._events = list(owned)
        self._history = history
        self._chunk_limit = chunk_limit
        self._closed = False

    def _require_owner(self):
        import threading
        if threading.current_thread() is not self._owner:
            raise ValueError('Mixed schedule cursor belongs to another thread')
        if self._closed:
            raise ValueError('Mixed schedule cursor is closed')

    def append(self, receipt, history):
        import copy
        from .input_spans import InputHistory
        from .vision_execution import input_binding
        self._require_owner()
        try:
            event = copy.deepcopy(receipt)
            keys = {'operator','position','count','tokens_sha256','chunks'}
            bindings = {'inputs_sha256','native_input_sha256','position_sha256','rotary_cursor'}
            if not isinstance(event, dict) or set(event) != keys | bindings:
                raise ValueError('Unknown mixed execution receipt layout')
            position, count = event['position'], event['count']
            if (not isinstance(history, InputHistory) or type(position) is not int
                    or type(count) is not int or position != len(self._history)
                    or count <= 0 or position+count != len(history)):
                raise ValueError('Noncontiguous mixed execution extension')
            if not history.starts_with(self._history):
                raise ValueError('Mixed execution extension rewrites owned ancestry')
            if not history.spans or history.spans[-1]['kind'] != 'text':
                raise ValueError('Execution boundary must follow completed image input')
            delta = history.segment(position)
            if (event['operator'] == 'normal-verifier-decode'
                    and any(span['kind']=='image' for span in delta.spans)):
                raise ValueError('Decode cannot consume image rows')
            expected = dict(inputs_sha256=history.digest, **input_binding(history))
            if (type(event['rotary_cursor']) is not int
                    or any(event[k] != value for k,value in expected.items())):
                raise ValueError('Mixed execution receipt input/position identity mismatch')
            # The unchanged native validator checks the new local interval.
            # Normalize only positions, never token identity or batch sizes.
            local = {key:event[key] for key in keys}
            local['position'] = 0
            chunks = event['chunks']
            if (not isinstance(chunks,(list,tuple)) or any(
                    not isinstance(chunk,(list,tuple)) or len(chunk)!=2
                    or any(type(v) is not int for v in chunk) for chunk in chunks)):
                raise ValueError('Invalid completed native chunks')
            local['chunks'] = [[start-position,size] for start,size in chunks]
            validate_operator_schedule([local],delta.native_tokens,self._chunk_limit)
            self._events.append(event)
            self._history = history
        except Exception as error:
            self._closed = True
            if isinstance(error,ValueError):
                raise
            raise ValueError('Invalid mixed execution extension') from error

    def finish(self):
        import copy
        self._require_owner()
        self._closed = True
        validate_input_schedule(self._events,self._history,self._chunk_limit)
        return tuple(copy.deepcopy(self._events))
