"""Versioned multi-turn teaching admission, independent of a curriculum package."""
import copy
import hashlib
import re

POLICIES=('verified_thinking_actions_answer','actions_answer')
SHA=re.compile(r'[a-f0-9]{64}\Z')

def validate_segments_lesson(value):
    required={'schema','title','model_contract_id','segments','target_policy','scope','evidence'}
    if set(value) not in (required,required|{'skill_id'}) or type(value['schema']) is not int or value['schema']!=2:raise ValueError('Invalid lesson schema 2')
    if value['scope']!='owned-training' or not SHA.fullmatch(str(value['model_contract_id'])):raise ValueError('Owned split and model identity required')
    if type(value['title']) is not str or not value['title'].strip() or len(value['title'].encode())>256:raise ValueError('Invalid title')
    if 'skill_id' in value and (type(value['skill_id']) is not str or not re.fullmatch(r'[a-z][a-z0-9_.-]{0,95}',value['skill_id'])):raise ValueError('Invalid skill identity')
    if value['target_policy'] not in POLICIES:raise ValueError('Unknown target policy')
    evidence=value['evidence']
    keys={'method','archive_sha256','verification_sha256','renderer_sha256','tokenizer_sha256'}
    if type(evidence) is not dict or set(evidence)!=keys or evidence['method']!='sss-verified-episode-v1':raise ValueError('Verified episode provenance required')
    if any(not SHA.fullmatch(str(evidence[k])) for k in keys-{'method'}):raise ValueError('Invalid evidence hash')
    rows=value['segments']
    if type(rows) is not list or not 2<=len(rows)<=16384:raise ValueError('Bounded multi-turn segments required')
    targets=[];total=0
    for row in rows:
        if type(row) is not dict or set(row)!={'role','kind','rendered','verified','scope'}:raise ValueError('Exact rendered spans required')
        if row['scope']!='practice' or row['role'] not in ('system','user','assistant','tool'):raise ValueError('Evaluation data cannot enter training context')
        if row['kind'] not in ('instruction','question','observation','thinking','action','answer','summary') or type(row['verified']) is not bool:raise ValueError('Invalid segment')
        if type(row['rendered']) is not str or not row['rendered']:raise ValueError('Empty or invalid rendered span')
        total+=len(row['rendered'].encode())
        if total>2*1024*1024:raise ValueError('Rendered lesson exceeds byte budget')
        selected=row['kind'] in ('thinking','action','answer','summary')
        if selected and row['role']!='assistant':raise ValueError('Assistant span has wrong role')
        if row['role']=='assistant' and not selected:raise ValueError('Unknown assistant target kind')
        if row['kind']=='thinking' and value['target_policy']=='actions_answer':selected=False
        if selected and not row['verified']:raise ValueError('Unverified text cannot be a target')
        targets.append(selected)
    if not any(targets):raise ValueError('No verified targets')
    return copy.deepcopy(value)

def tokenize_segments(lesson,tokenize,maximum):
    value=validate_segments_lesson(lesson);text='';tokens=[];labels=[]
    for row in value['segments']:
        selected=row['role']=='assistant' and (row['kind']!='thinking' or value['target_policy']=='verified_thinking_actions_answer')
        text+=row['rendered'];current=tokenize(text)
        if type(current) is not list or any(type(t) is not int or t<0 for t in current):raise ValueError('Invalid tokenizer output')
        if current[:len(tokens)]!=tokens:raise ValueError('Rendered role boundary does not align to tokenizer')
        labels.extend([selected]*(len(current)-len(tokens)));tokens=current
    if not 2<=len(tokens)-1<=maximum or not any(labels[1:]):raise ValueError('Teaching tokens exceed budget or have no targets; no truncation')
    return dict(tokens=tokens[:-1],targets=tokens[1:],loss_mask=labels[1:])
