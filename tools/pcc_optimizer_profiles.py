"""Content-addressed, owner-configured native optimizer profiles.

Schema 1 retains its full-only contract. Schema 2 explicitly binds mode and
LoRA rank; an existing policy ID never silently gains another training mode.
"""
import copy,hashlib,json,math,re
from tools.pcc_native_training import validate_optimizer_request


def identity(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def mode(profile):
    return 'full' if profile['schema']==1 else profile['mode']


def recipe(profile):
    """Compare the exact registered recipe without title/state-policy labels."""
    value=dict(mode=mode(profile),rank=profile.get('rank'),optimizer=profile['optimizer'],
        learning_rate=profile['learning_rate'],optimizer_settings=profile['optimizer_settings'])
    if 'lora_update_space' in profile:value['lora_update_space']=profile['lora_update_space']
    return value


def profiles(value,*,sequence_profile,deadline):
    if type(value) is not dict:raise ValueError('Owner optimizer profiles must be an explicit map')
    result={}
    for key,profile in value.items():
        fields={'schema','title','optimizer','learning_rate','optimizer_settings','state_policy','training_budget_seconds'}
        if type(profile) is not dict or type(profile.get('schema')) is not int or profile['schema'] not in (1,2,3):
            raise ValueError('Optimizer profile identity/schema differs')
        if profile['schema']>=2:fields|={'mode','rank'}
        if profile['schema']==3:fields.add('lora_update_space')
        if set(profile)!=fields or key!=identity(profile):raise ValueError('Optimizer profile identity/schema differs')
        if type(profile['title']) is not str or not 1<=len(profile['title'])<=160 or profile['optimizer'] not in ('adamw','muon') or profile['state_policy'] not in ('fresh','resume'):
            raise ValueError('Explicit native optimizer/title/state policy required')
        selected=mode(profile)
        if selected not in ('full','lora'):raise ValueError('Explicit full or LoRA optimizer mode required')
        if profile['schema']==3 and (selected!='lora' or profile['lora_update_space']!='factorwise-v1'):
            raise ValueError('Schema 3 requires explicit factorwise LoRA')
        if profile['schema']>=2:
            rank=profile['rank']
            if ((selected=='full' and rank is not None)
                    or (selected=='lora' and (type(rank) is not int or not 1<=rank<=256))):
                raise ValueError('LoRA profile needs its exact rank; full profile rank must be null')
        rate=profile['learning_rate'];seconds=profile['training_budget_seconds']
        if type(rate) not in (int,float) or not math.isfinite(rate) or not 0<rate<=.01:raise ValueError('Bounded positive profile rate required')
        if seconds is not None and (type(seconds) not in (int,float) or not math.isfinite(seconds) or not 0<seconds<deadline):
            raise ValueError('Profile training budget must be positive and below hard deadline')
        if selected=='lora' and ((profile['optimizer']!='adamw' and profile['schema']!=3) or seconds is not None):
            raise ValueError('Registered LoRA currently qualifies fixed-update AdamW, without a training-time budget')
        native=dict(optimizer=profile['optimizer'],mode=selected,sequence_profile=sequence_profile,
            optimizer_state_mode='fresh',optimizer_settings=profile['optimizer_settings'])
        if profile['schema']==3:native['lora_update_space']=profile['lora_update_space']
        validate_optimizer_request(native)
        result[key]=copy.deepcopy(profile)
    return result


def select(registered,request):
    if 'optimizer_profile_id' not in request:return None
    key=request['optimizer_profile_id']
    if type(key) is not str or key not in registered or identity(registered[key])!=key:
        raise ValueError('Unknown or changed owner optimizer profile')
    profile=registered[key]
    if request['mode']!=mode(profile) or request['learning_rate']!=profile['learning_rate']:
        raise ValueError('Job mode/rate differs from registered optimizer profile')
    if mode(profile)=='lora' and request['rank']!=profile['rank']:
        raise ValueError('LoRA job rank differs from registered optimizer profile')
    if profile['state_policy']=='resume' and not request.get('parent_job_id'):
        raise ValueError('Optimizer resume needs a completed parent job')
    return copy.deepcopy(profile)


def validate_parent(registered,request,parent_record):
    """New policies refuse an incompatible resume before a worker is queued.

    Legacy schema-1 behavior is retained. Physical state hash/header admission
    remains in the native executor; no large state file is read under API locks.
    """
    profile=select(registered,request)
    if profile is None or profile['schema']==1 or profile['state_policy']!='resume':return
    prior_request=parent_record['request'];candidate=parent_record['candidate']
    prior_key=prior_request.get('optimizer_profile_id');prior=registered.get(prior_key)
    if (prior is None or identity(prior)!=prior_key or recipe(prior)!=recipe(profile)
            or candidate['receipt'].get('optimizer')!=profile['optimizer']
            or type(candidate['receipt'].get('optimizer_checkpoint')) is not dict):
        raise ValueError('Resume requires a completed parent with the same registered optimizer recipe')
    if prior_request['lesson_ids']!=request['lesson_ids']:
        raise ValueError('Resume requires the same ordered lessons; choose fresh moments for a new block')
