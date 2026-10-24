"""Content-addressed, owner-configured native optimizer profiles.

Clients choose registered identities; executables, resources and state paths
remain owner configuration or registered parent artifacts.
"""
import copy,hashlib,json,math,re
from tools.pcc_native_training import validate_optimizer_request

def identity(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def profiles(value,*,sequence_profile,deadline):
    if type(value) is not dict:raise ValueError('Owner optimizer profiles must be an explicit map')
    result={}
    for key,profile in value.items():
        fields={'schema','title','optimizer','learning_rate','optimizer_settings','state_policy','training_budget_seconds'}
        if type(profile) is not dict or set(profile)!=fields or type(profile['schema']) is not int or profile['schema']!=1 or key!=identity(profile):raise ValueError('Optimizer profile identity/schema differs')
        if type(profile['title']) is not str or not 1<=len(profile['title'])<=160 or profile['optimizer'] not in ('adamw','muon') or profile['state_policy'] not in ('fresh','resume'):raise ValueError('Explicit native optimizer/title/state policy required')
        rate=profile['learning_rate'];seconds=profile['training_budget_seconds']
        if type(rate) not in (int,float) or not math.isfinite(rate) or not 0<rate<=.01:raise ValueError('Bounded positive profile rate required')
        if seconds is not None and (type(seconds) not in (int,float) or not math.isfinite(seconds) or not 0<seconds<deadline):raise ValueError('Profile training budget must be positive and below hard deadline')
        validate_optimizer_request(dict(optimizer=profile['optimizer'],mode='full',sequence_profile=sequence_profile,optimizer_state_mode='fresh',optimizer_settings=profile['optimizer_settings']))
        result[key]=copy.deepcopy(profile)
    return result

def select(registered,request):
    if 'optimizer_profile_id' not in request:return None
    key=request['optimizer_profile_id']
    if type(key) is not str or key not in registered or identity(registered[key])!=key:raise ValueError('Unknown or changed owner optimizer profile')
    profile=registered[key]
    if request['mode']!='full' or request['learning_rate']!=profile['learning_rate']:raise ValueError('Job mode/rate differs from registered optimizer profile')
    if profile['state_policy']=='resume' and not request.get('parent_job_id'):raise ValueError('Optimizer resume needs a completed parent job')
    return copy.deepcopy(profile)
