"""Validate display annotations against consumed canonical snapshot spans.

This does not render text, run inference, publish a head, or store another token
history. The frontend must verify text encoding before supplying annotations.
Publication must bind the returned descriptor to the same atomic move receipt.
"""
import hashlib
import re

from .store import InvalidOperation, _decode, _hash, _id, _json_bytes, _name
from .input_spans import InputHistory, history_lineage, history_segment, history_digest, generated_text


def _require(condition, message):
    if not condition: raise InvalidOperation(message)


def _integer(value):
    return type(value) is int and value >= 0


def validate_binding(store, head, target_snapshot, binding, *, namespace_id, operation_id,
                     generation_operation_id=None):
    """Check one root/answer/user/resume/rewind annotation without changing state.

    `head` is the CAS source read by the publisher under its Store transaction.
    `namespace_id` must come from that branch's immutable namespace association,
    not from an untrusted request. The caller owns atomic receipt publication.
    """
    _name(operation_id)
    if generation_operation_id is None:
        generation_operation_id = operation_id
    _name(generation_operation_id)
    target_snapshot = _id(target_snapshot)
    _require(isinstance(namespace_id,str) and re.fullmatch('[0-9a-f]{32}',namespace_id),
             'Invalid presentation namespace')
    _require(isinstance(binding,dict), 'Presentation must be an object')
    value = _decode(_json_bytes(binding))  # Detach nested caller-owned objects.
    fields={'schema','namespace_id','kind','messages','source','input_span',
            'output_span','renderer_id','turn_complete'}
    optional={'generation_settings','generation_seed','reseed_seed'}
    schema=value.get('schema')
    seed_field=('reseed_seed' if value.get('kind')=='reseed' else 'generation_seed')
    expected_fields=(fields if schema==1 else fields|{seed_field} if schema==2 else
                     fields|({seed_field} if seed_field in value else set()) if schema==3 else set())
    optional_fields={'generation_settings'}
    _require(set(value)==expected_fields or
             (expected_fields < set(value) and set(value)-expected_fields <= optional_fields),
             'Invalid presentation fields')
    _require(not ({'generation_seed','reseed_seed'} <= set(value)),
             'Presentation cannot bind both generation and reseed seeds')
    _require(type(schema) is int and schema in (1,2,3), 'Unsupported presentation schema')
    _require(value['namespace_id']==namespace_id, 'Presentation belongs to another namespace')
    _id(value['renderer_id'])
    kind=value['kind']
    _require(kind in ('root','answer','initial_turn','user_turn','resume','reseed','rewind'), 'Unsupported presentation action')
    _require('generation_settings' not in value or kind=='resume',
             'Generation settings are supported only for resume actions')
    _require('generation_seed' not in value or kind in ('answer','initial_turn','user_turn','resume'),
             'Generation seed is unsupported for this presentation action')
    _require('reseed_seed' not in value or kind=='reseed',
             'Reseed seed is unsupported for this presentation action')
    seeded = 'generation_seed' in value or 'reseed_seed' in value
    _require(schema==3 or (schema==2)==seeded, 'Presentation seed fields require schema 2 or 3')
    for key in ('generation_seed','reseed_seed'):
        if key in value:
            _require(_integer(value[key]) and value[key] < (1<<64), 'Presentation seed is invalid')
    messages=value['messages']
    _require(isinstance(messages,list), 'Presentation messages must be a list')
    seen=set()
    for message in messages:
        _require(isinstance(message,dict) and set(message)=={'id','role','content'}, 'Invalid message fields')
        _require(isinstance(message['id'],str) and re.fullmatch('[0-9a-f]{32}',message['id']), 'Invalid message ID')
        _require(message['id'] not in seen, 'Duplicate presentation message ID')
        seen.add(message['id'])
        _require(message['role'] in ('system','user','assistant') and isinstance(message['content'],str),
                 'Only plain text message roles are supported')
    source=value['source']
    _require(isinstance(source,dict) and set(source)=={'snapshot','revision','token_count'}, 'Invalid source fields')
    _require(_integer(source['revision']) and source['revision']==head.revision, 'Presentation revision changed')
    _require(_integer(source['token_count']), 'Invalid source token count')
    target_record=store.get_snapshot(target_snapshot)
    source_record=store.get_snapshot(head.tip)
    target_root,target_history=history_lineage(store,target_snapshot)
    source_root,source_history=history_lineage(store,head.tip)
    _require((schema==3)==isinstance(target_history,InputHistory)
             and isinstance(source_history,InputHistory)==isinstance(target_history,InputHistory),
             'Presentation schema differs from canonical input kind')
    _require(target_root==source_root==head.root, 'Presentation crosses computational roots')
    _require(type(value['turn_complete']) is bool and value['turn_complete']==
             (target_record['controller'].get('status')=='stopped'), 'Turn completion differs from consumed controller')

    if kind=='root':
        _require(head.revision==0 and head.tip==target_snapshot and target_record['parent'] is None,
                 'Root annotation requires the initial root reference')
        _require(source['snapshot'] is None and source['token_count']==0, 'Root has no earlier input history')
        _require(messages and value['output_span'] is None and
                 target_record['controller'].get('status')=='idle', 'Root cannot claim generated output')
        start,delta=0,target_history
    else:
        _require(source['snapshot']==head.tip and source['token_count']==len(source_history),
                 'Presentation source differs from canonical history')
        if kind=='rewind':
            _require(not messages and value['input_span'] is None and value['output_span'] is None,
                     'Rewind cannot invent messages or consumed spans')
            cursor=head.tip;visited=set()
            while cursor!=target_snapshot:
                _require(cursor is not None and cursor not in visited, 'Rewind target is not a canonical ancestor')
                visited.add(cursor);cursor=store.get_snapshot(cursor)['parent']
            return value
        _require(target_record['parent']==head.tip and history_segment(target_history,0,len(source_history))==source_history,
                 'Turn annotation requires a direct canonical append')
        start,delta=len(source_history),history_segment(target_history,len(source_history))

    span=value['input_span']
    digest_field = 'inputs_sha256' if schema==3 else 'token_ids_sha256'
    _require(isinstance(span,dict) and set(span)=={'start','count',digest_field}, 'Invalid input span fields')
    _require(_integer(span['start']) and _integer(span['count']) and span['start']==start
             and span['count']<=len(delta), 'Input span is outside consumed history')
    _require(span[digest_field]==history_digest(history_segment(delta,0,span['count'])), 'Input digest differs')
    if kind=='root':
        _require(span['count']==len(delta), 'Root must bind its complete consumed prefix')
        return value
    output=value['output_span']
    _require(isinstance(output,dict) and set(output)=={'start','count','message_id'}, 'Invalid output span fields')
    _require(_integer(output['start']) and _integer(output['count']) and
             output['start']==start+span['count'] and output['count']==len(delta)-span['count'],
             'Output span differs from consumed generated tokens')
    _require(isinstance(output['message_id'],str) and re.fullmatch('[0-9a-f]{32}',output['message_id'])
             and output['message_id'] not in seen, 'Invalid assistant message ID')
    generated_text(history_segment(delta,span['count']))
    before,after=source_record['controller'],target_record['controller']
    _require(after.get('status') in ('active','stopped'), 'Output lacks an active generation scope')
    _require(_integer(after.get('accepted_in_scope')), 'Invalid consumed output counter')
    if kind=='reseed':
        _require(not messages and span['count']==0 and before.get('status') in ('active','stopped'),
                 'Reseed must preserve an existing generation without adding inputs')
        _require('reseed_seed' in value, 'Reseed annotation lacks its seed binding')
        expected=hashlib.sha256(f"cow-seed-v1:{value['reseed_seed']}".encode('utf-8')).hexdigest()
        _require(after.get('stream_key')==expected and after.get('next_draw')==0,
                 'Reseed controller does not match its bound seed')
        before_without_rng={key:item for key,item in before.items() if key not in ('stream_key','next_draw')}
        after_without_rng={key:item for key,item in after.items() if key not in ('stream_key','next_draw')}
        _require(before_without_rng==after_without_rng and after.get('status')==before.get('status'),
                 'Reseed changed non-RNG controller state')
        stream=before.get('stream_key')
        _require(isinstance(stream,str), 'Reseed source has no active stream identity')
        expected_message=store._published_assistant_message_id(namespace_id,head.tip,stream,
                                                               branch_name=head.name)
        _require(expected_message is not None and output.get('message_id')==expected_message
                 and output['count']==0 and output['start']==start,
                 'Reseed must carry forward the published assistant identity without output')
    elif kind=='resume':
        _require(not messages and span['count']==0 and before.get('status')=='active',
                 'Resume must continue the active turn without new inputs')
        if 'generation_seed' in value:
            expected_stream=hashlib.sha256(
                f"cow-seed-v1:{value['generation_seed']}".encode('utf-8')).hexdigest()
        else:
            expected_stream=before.get('stream_key')
        _require(after.get('stream_key')==expected_stream,
                 'Resume changed its generation stream without a bound seed')
        has_generation_settings='generation_settings' in value
        generation_settings=value.get('generation_settings')
        if has_generation_settings:
            _require(isinstance(generation_settings,dict),
                     'Resume generation settings must be an object')
            try:
                from .min_p import OPTIONAL_FIELDS, validate_policy
                sampler_fields={'temperature','min_p','top_k','top_p','repetition_penalty'}|OPTIONAL_FIELDS
                actual={key:after['policy'][key] for key in sampler_fields if key in after['policy']}
                normalized=validate_policy(generation_settings)
                _require(normalized==generation_settings and actual==normalized,
                         'Resume policy differs from its bound generation descriptor')
                before_core={key:item for key,item in before['policy'].items() if key not in sampler_fields}
                after_core={key:item for key,item in after['policy'].items() if key not in sampler_fields}
                _require(before_core==after_core,
                         'Resume changed non-sampler generation settings')
            except (TypeError,ValueError,KeyError) as error:
                raise InvalidOperation('Resume generation settings are invalid') from error
        elif after.get('policy')!=before.get('policy'):
            raise InvalidOperation('Resume policy change lacks a bound generation descriptor')
        count=before.get('accepted_in_scope')
        _require(_integer(count) and after.get('accepted_in_scope')==count+output['count'],
                 'Resume output count differs from controller progress')
    else:
        if kind in ('initial_turn','user_turn'):
            _require(span['count']>0, 'A new user turn must consume its prompt suffix')
            expected_status='idle' if kind=='initial_turn' else 'stopped'
            _require(len(messages)==1 and messages[0]['role']=='user' and before.get('status')==expected_status,
                     'User turn source state or message differs')
        else:
            _require(not messages and before.get('status')=='idle', 'Initial answer requires a prepared idle prefix')
        if 'generation_seed' in value:
            expected_stream=hashlib.sha256(f"cow-seed-v1:{value['generation_seed']}".encode('utf-8')).hexdigest()
        else:
            expected_stream=hashlib.sha256(generation_operation_id.encode()).hexdigest()
        _require(after.get('stream_key')==expected_stream and
                 after.get('accepted_in_scope')==output['count'], 'Output belongs to a different generation scope')
    return value
