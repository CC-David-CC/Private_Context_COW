"""Strict, detached input metadata for opt-in presentation-bound COW work."""
import hashlib
import json
import re

from .store import InvalidOperation, _hash, _json_bytes
from .input_spans import InputHistory, history_for_snapshot, history_segment, history_digest


_FIELDS = {'schema', 'namespace_id', 'kind', 'messages', 'renderer_id', 'output_message_id'}
_KINDS = {'root', 'answer', 'initial_turn', 'user_turn', 'resume', 'reseed'}


def detach_request(value, *, expected_kind=None):
    """Validate and JSON-detach request metadata before intent or GPU admission."""
    if not isinstance(value, dict):
        raise InvalidOperation('Presentation request must be an object')
    try:
        # Canonical encoding rejects non-JSON values and severs nested aliases.
        result = json.loads(_json_bytes(value))
    except (TypeError, ValueError) as error:
        raise InvalidOperation('Presentation request is not canonical JSON') from error
    allowed_fields=(_FIELDS, _FIELDS | {'generation_settings'}, _FIELDS | {'seed'},
                    _FIELDS | {'generation_settings','seed'})
    seeded = 'seed' in result
    expected_schema = 2 if seeded else 1
    if (set(result) not in allowed_fields
            or type(result['schema']) is not int or result['schema'] != expected_schema):
        raise InvalidOperation('Invalid presentation request fields')
    if (not isinstance(result['kind'], str) or result['kind'] not in _KINDS
            or (expected_kind and result['kind'] != expected_kind)):
        raise InvalidOperation('Presentation request kind differs')
    if not isinstance(result['namespace_id'], str) or not re.fullmatch('[0-9a-f]{32}', result['namespace_id']):
        raise InvalidOperation('Invalid presentation namespace')
    if not isinstance(result['renderer_id'], str) or not re.fullmatch('[0-9a-f]{64}', result['renderer_id']):
        raise InvalidOperation('Invalid presentation renderer')
    messages = result['messages']
    if not isinstance(messages, list):
        raise InvalidOperation('Presentation messages must be a list')
    seen = set()
    for item in messages:
        if (not isinstance(item, dict) or set(item) != {'id', 'role', 'content'}
                or not isinstance(item.get('id'), str)
                or not re.fullmatch('[0-9a-f]{32}', item['id'])
                or item['id'] in seen
                or not isinstance(item.get('role'), str)
                or item.get('role') not in ('system', 'user', 'assistant')
                or not isinstance(item.get('content'), str)):
            raise InvalidOperation('Invalid presentation message')
        seen.add(item['id'])
    output = result['output_message_id']
    if 'generation_settings' in result:
        if result['kind']!='resume' or not isinstance(result['generation_settings'],dict):
            raise InvalidOperation('Generation settings are supported only for sampled resume')
    if 'seed' in result:
        if (result['kind'] not in ('answer','initial_turn','user_turn','resume','reseed')
                or type(result['seed']) is not int or not 0 <= result['seed'] < (1<<64)):
            raise InvalidOperation('Seed is invalid for this presentation action')
    if result['kind']=='reseed' and 'seed' not in result:
        raise InvalidOperation('Reseed presentation requires its unsigned seed')
    if result['kind'] == 'root':
        if not messages or output is not None:
            raise InvalidOperation('Root must bind messages and have no output message')
    else:
        if output is None or not isinstance(output, str) or not re.fullmatch('[0-9a-f]{32}', output):
            raise InvalidOperation('Presentation requires a stable output message ID')
        if result['kind'] in ('answer', 'resume', 'reseed') and messages:
            raise InvalidOperation('This presentation kind cannot add display messages')
        if result['kind'] in ('initial_turn','user_turn') and (
                len(messages) != 1 or messages[0]['role'] != 'user'):
            raise InvalidOperation('A user turn must bind exactly one user message')
        if any(item['id'] == output for item in messages):
            raise InvalidOperation('Output message ID must be distinct from input messages')
    return result


def request_digest(value):
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def binding_for_snapshot(store, request, *, operation_id, source_head, target_snapshot,
                         input_ids=(), generation_operation_id=None):
    """Construct a display binding from the actual canonical consumed delta."""
    from .presentation_binding import validate_binding

    if request['kind']=='root' and 'seed' in request:
        raise InvalidOperation('Root presentation cannot supply a generation seed')
    target = store.get_snapshot(target_snapshot)
    history = history_for_snapshot(store,target_snapshot)
    mixed = isinstance(history,InputHistory)
    if mixed:
        delta = input_ids if isinstance(input_ids,InputHistory) else InputHistory([dict(kind='text',tokens=list(input_ids))])
    else:
        if isinstance(input_ids,InputHistory):
            raise InvalidOperation('Legacy presentation cannot bind mixed input history')
        delta = tuple(input_ids)
    schema = 3 if mixed else 2 if 'seed' in request else 1
    digest_field = 'inputs_sha256' if mixed else 'token_ids_sha256'
    if request['kind'] == 'root':
        if source_head is None or source_head.revision != 0 or source_head.tip != target_snapshot:
            raise InvalidOperation('Root presentation must bind the initial branch reference')
        if history != delta:
            raise InvalidOperation('Root presentation differs from consumed prompt tokens')
        binding = dict(schema=schema, namespace_id=request['namespace_id'], kind='root',
            messages=request['messages'],
            source=dict(snapshot=None, revision=0, token_count=0),
            input_span=dict(start=0, count=len(history), **{digest_field:history_digest(history)}),
            output_span=None, renderer_id=request['renderer_id'],
            turn_complete=target['controller'].get('status') == 'stopped')
    else:
        if source_head is None:
            raise InvalidOperation('Presentation request has no source branch head')
        source_history = history_for_snapshot(store,source_head.tip)
        if (history_segment(history,0,len(source_history)) != source_history or
                history_segment(history,len(source_history),len(source_history)+len(delta)) != delta):
            raise InvalidOperation('Presentation source differs from canonical input prefix')
        consumed_inputs = len(delta)
        output_count = len(history) - len(source_history) - consumed_inputs
        if output_count < 0:
            raise InvalidOperation('Presentation input exceeds consumed history')
        binding = dict(schema=schema, namespace_id=request['namespace_id'], kind=request['kind'],
            messages=request['messages'],
            source=dict(snapshot=source_head.tip, revision=source_head.revision,
                        token_count=len(source_history)),
            input_span=dict(start=len(source_history), count=consumed_inputs,
                            **{digest_field:history_digest(delta)}),
            output_span=dict(message_id=request['output_message_id'],
                start=len(source_history)+consumed_inputs, count=output_count),
            renderer_id=request['renderer_id'],
            turn_complete=target['controller'].get('status') == 'stopped')
        if 'generation_settings' in request:
            binding['generation_settings']=request['generation_settings']
        if request['kind']=='reseed':
            binding['reseed_seed']=request['seed']
        elif 'seed' in request:
            binding['generation_seed']=request['seed']
    return validate_binding(store, source_head, target_snapshot, binding,
                            namespace_id=request['namespace_id'], operation_id=operation_id,
                            generation_operation_id=generation_operation_id)
