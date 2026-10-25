"""Owned lesson records and non-repeating native-training jobs. No inference ledger."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, ExitStack
from pathlib import Path
import copy
import fcntl
import hashlib
import json
import math
import os
import re
import subprocess
import threading
import uuid
from tools.pcc_native_training import execute_native, sha256, validate_examples

HEX=re.compile(r'[a-f0-9]{64}\Z')
OP=re.compile(r'[A-Za-z][A-Za-z0-9_-]{0,127}\Z')

def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()

def digest(value):return hashlib.sha256(canonical(value)).hexdigest()

def validate_lesson(value):
    if type(value) is dict and value.get('schema')==2:
        from tools.pcc_lesson_segments import validate_segments_lesson
        return validate_segments_lesson(value)
    fields={'schema','title','model_contract_id','prompt','completion','rationale','scope','evidence'}
    if type(value) is not dict or set(value) not in (fields,fields|{'skill_id'}) or type(value['schema']) is not int or value['schema']!=1:
        raise ValueError('Lesson schema 1 requires explicit teaching text and evidence')
    if 'skill_id' in value and (type(value['skill_id']) is not str or not re.fullmatch(r'[a-z][a-z0-9_.-]{0,95}',value['skill_id'])):
        raise ValueError('Invalid lesson skill_id')
    if not HEX.fullmatch(str(value['model_contract_id'])) or value['scope']!='owned-training':
        raise ValueError('Lesson requires an owned training split and exact model identity')
    for key,limit in [('title',256),('prompt',65536),('completion',65536),('rationale',8192)]:
        if type(value[key]) is not str or not value[key].strip() or len(value[key].encode())>limit:
            raise ValueError('Invalid lesson '+key)
    evidence=value['evidence']
    if type(evidence) is not dict or set(evidence)!={'method','failed_result_sha256','corrected_result_sha256','verification_sha256'}:
        raise ValueError('Keep original failure, correction and independent check identities')
    if evidence['method'] not in ('bw-verified-correction-v1','host-verified-correction-v1','host-verified-replay-v1'):
        raise ValueError('Explicit host verification method required')
    if any(not HEX.fullmatch(str(evidence[key])) for key in evidence if key!='method'):
        raise ValueError('Verification evidence requires SHA-256 identities')
    if evidence['failed_result_sha256']==evidence['corrected_result_sha256']:
        raise ValueError('Failure and correction must be distinct evidence')
    return copy.deepcopy(value)

class LessonService:
    """One optional server-owned training boundary, separate from COW history.

    Clients submit text and known candidate IDs, never executable/model paths,
    devices, environment overrides or shell commands. Verification is an
    authenticated owner's attestation, not a remote execution of client tests.
    """
    def __init__(self,directory,model_contract_id,tokenize,runner,*,max_tokens=256,max_steps=16,sequence_profile='short-v1',rendered_tokenize=None,learning_policies=None):
        if not HEX.fullmatch(model_contract_id):raise ValueError('Exact model contract required')
        self.root=Path(directory).absolute()
        if self.root.is_symlink():raise ValueError('Training store cannot be a symlink')
        self.root.mkdir(mode=0o700,parents=True,exist_ok=True)
        if self.root.stat().st_uid!=os.getuid() or self.root.stat().st_mode&0o077:
            raise ValueError('Training store must be owned and private')
        self.lease=(self.root/'service.lock').open('a+')
        fcntl.flock(self.lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.model=model_contract_id;self.tokenize=tokenize;self.runner=runner
        self.rendered_tokenize=tokenize if rendered_tokenize is None else rendered_tokenize
        if not callable(self.rendered_tokenize):raise ValueError("Rendered tokenizer callback required")
        from tools.pcc_native_training import SEQUENCE_PROFILES
        if sequence_profile not in SEQUENCE_PROFILES:raise ValueError('Unknown sequence profile')
        self.sequence_profile=sequence_profile
        if type(max_tokens) is not int or not 2<=max_tokens<=SEQUENCE_PROFILES[sequence_profile] or type(max_steps) is not int or not 1<=max_steps<=1000:
            raise ValueError('Invalid owner training limits')
        self.max_tokens=max_tokens;self.max_steps=max_steps;self.mutex=threading.RLock()
        binding=self.root/'model.json'
        if binding.exists():
            if self._read(binding)!=dict(schema=1,model_contract_id=self.model):raise ValueError('Training store belongs to another model contract')
        else:self._write(binding,dict(schema=1,model_contract_id=self.model))
        for name in ('lessons','operations','runs'):(self.root/name).mkdir(mode=0o700,exist_ok=True)
        # Never replay a training job after a crash or a lost acknowledgment.
        for path in (self.root/'operations').glob('*.json'):
            record=self._read(path)
            if record.get('state') in ('queued','running'):
                record['state']='interrupted';self._write(path,record)
        bind=getattr(self.runner,'bind',None)
        # Lessons are immutable atomic files. Admission already owns the record
        # lock; calling the public locking reader here would nest flock handles.
        if bind is not None:bind(self._lesson,self._example,self.tokenize,self.model)
        self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='pcc-training')
        self.learning=None
        if learning_policies is not None:self.configure_learning(learning_policies)

    def configure_learning(self,policies):
        if self.learning is not None:raise ValueError('Learning policies already configured')
        from tools.pcc_learning_runs import LearningRuns
        self.learning=LearningRuns(self,policies)
        self.runner.record_process_identity=True

    def learning_policies(self):
        if self.learning is None:return dict(schema=1,policies=[],enabled=False)
        return dict(self.learning.capabilities(),enabled=True)

    def learning_runs(self):
        if self.learning is None:raise ValueError('Learning runs are not configured')
        return self.learning.listing()

    def learning_run(self,identity):
        if self.learning is None:raise ValueError('Learning runs are not configured')
        return self.learning.status(identity)

    def learning_round(self,identity,number):
        if self.learning is None:raise ValueError('Learning runs are not configured')
        return self.learning.round_status(identity,number)

    def close(self):
        self.executor.shutdown(wait=True);self.lease.close()

    @contextmanager
    def locked(self):
        with self.mutex:
            with (self.root/'records.lock').open('a+') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX);yield

    def _read(self,path):
        if path.is_symlink():raise ValueError('Training record link rejected')
        value=json.loads(path.read_bytes())
        if path.parent==self.root/'operations':
            acceptance=value.get('acceptance') if type(value) is dict else None
            request=value.get('request') if type(value) is dict else None
            if (type(acceptance) is not dict or type(request) is not dict
                    or request.get('operation_id')!=path.stem or acceptance.get('operation_id')!=path.stem
                    or acceptance.get('request_sha256')!=digest([value.get('path'),request])):
                raise ValueError('Stored operation differs from its original request binding')
            if 'record_sha256' in value and value['record_sha256']!=digest({k:v for k,v in value.items() if k!='record_sha256'}):
                raise ValueError('Stored training operation checksum differs')
        return value

    def _write(self,path,value):
        if path.parent==self.root/'operations':
            value={k:v for k,v in value.items() if k!='record_sha256'}
            value['record_sha256']=digest(value)
        temporary=path.with_name('.pending-'+uuid.uuid4().hex)
        try:
            fd=os.open(temporary,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
            with os.fdopen(fd,'wb') as out:out.write(canonical(value)+b'\n');out.flush();os.fsync(out.fileno())
            os.replace(temporary,path)
            fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
            try:os.fsync(fd)
            finally:os.close(fd)
        finally:
            if temporary.exists():temporary.unlink()

    def operation(self,identity):
        if not OP.fullmatch(identity):raise ValueError('Invalid operation ID')
        with self.locked():return self._read(self.root/'operations'/(identity+'.json'))['acceptance']

    def lesson(self,identity):
        with self.locked():return self._lesson(identity)

    def _lesson(self,identity):
        if not HEX.fullmatch(identity):raise ValueError('Invalid lesson ID')
        value=self._read(self.root/'lessons'/(identity+'.json'))
        if digest(value['lesson'])!=identity:raise ValueError('Corrupt lesson identity')
        validate_lesson(value['lesson'])
        if value['lesson']['model_contract_id']!=self.model:raise ValueError('Lesson belongs to another model')
        if value['example']!=self._example(value['lesson'],identity):raise ValueError('Stored training tokens differ from the immutable lesson')
        return value

    def _example(self,lesson,identity):
        if lesson['schema']==2:
            from tools.pcc_lesson_segments import tokenize_segments
            example=dict(id=identity,**tokenize_segments(lesson,self.rendered_tokenize,self.max_tokens),verified=True,provenance_sha256=digest(lesson['evidence']))
            validate_examples([example],sequence_profile=self.sequence_profile);return example
        prefix=self.tokenize(lesson['prompt']);full=self.tokenize(lesson['prompt']+lesson['completion'])
        if not prefix or full[:len(prefix)]!=prefix or len(full)<=len(prefix):
            raise ValueError('Teaching boundary must align to the exact tokenizer; add a separator')
        if not 2<=len(full)-1<=self.max_tokens:raise ValueError('Lesson exceeds admitted token budget; no text was truncated')
        example=dict(id=identity,tokens=full[:-1],targets=full[1:],loss_mask=[i+1>=len(prefix) for i in range(len(full)-1)],verified=True,provenance_sha256=digest(lesson['evidence']))
        validate_examples([example],sequence_profile=self.sequence_profile);return example

    def lessons(self):
        with self.locked():
            rows=[self._read(p) for p in sorted((self.root/'lessons').glob('*.json'))]
            selected={}
            for path in (self.root/'operations').glob('*.json'):
                operation=self._read(path)
                if operation['path']=='/cow/training/jobs':
                    for identity in operation['request']['lesson_ids']:selected.setdefault(identity,[]).append(operation['acceptance']['job_id'])
            return dict(schema=1,model_contract_id=self.model,lessons=[dict(lesson_id=v['lesson_id'],title=v['lesson']['title'],tokens=len(v['example']['tokens']),
                skill_id=v['lesson'].get('skill_id'),selected_in=sorted(selected.get(v['lesson_id'],[])),pending=v['lesson_id'] not in selected) for v in rows])

    def skills(self):
        describe=getattr(self.runner,'describe_skills',None)
        if describe is None:raise ValueError('This server has no configured skill guard')
        return describe()

    def exports(self):
        with self.locked():
            root=self.root/'exports'
            if root.is_symlink():raise ValueError('Export store link rejected')
            from tools.pcc_model_exports import read_export
            return dict(schema=1,model_contract_id=self.model,exports=[read_export(p,self.model) for p in sorted(root.glob('*.json'))])

    def optimizer_profiles(self):
        describe=getattr(self.runner,'describe_optimizer_profiles',None)
        if describe is None:return dict(schema=1,model_contract_id=self.model,profiles=[])
        return dict(schema=1,model_contract_id=self.model,profiles=describe())

    def job(self,identity):
        if not OP.fullmatch(identity):raise ValueError('Invalid job ID')
        with self.locked():record=self._read(self.root/'operations'/(identity+'.json'))
        if record['path']!='/cow/training/jobs':raise ValueError('Operation is not a training job')
        result=dict(schema=1,job_id=identity,state=record['state'],request_sha256=record['acceptance']['request_sha256'],
                    lesson_ids=record['request']['lesson_ids'],mode=record['request']['mode'],promoted=False)
        if 'skill_guard' in record['request'] and record['state'] in ('queued','running'):
            path=self.root/'runs'/identity/'progress.json'
            if path.exists():
                progress=self._read(path)
                if progress.get('policy_id')!=record['request']['skill_guard']['policy_id']:raise ValueError('Skill progress belongs to another policy')
                result['skill_guard_progress']=progress
        if 'candidate' in record:
            c=record['candidate'];result['candidate']=dict(sha256=c['candidate_sha256'],mode=c['mode'])
            result['steps']=[{**{key:step[key] for key in ('step','before','after','examples','example_cursor')},
                'gradient_tensors':len(step.get('gradients',[])),
                'nonzero_gradients':sum(g['l2']>0 for g in step.get('gradients',[]))} for step in c['receipt']['steps']]
            result['example_cursor']=c['receipt']['example_cursor']
            if 'skill_guard' in c:result['skill_guard']={k:v for k,v in c['skill_guard'].items() if k!='protected_pass_ids'}
        if 'optimizer_profile_id' in record['request']:
            result['optimizer_profile_id']=record['request']['optimizer_profile_id']
            if 'candidate' in record:
                receipt=record['candidate']['receipt'];result['optimizer']=receipt['optimizer']
                result['optimizer_completed_steps']=receipt['optimizer_checkpoint']['completed_steps']
                result['consumed_example_presentations']=sum(row['examples'] for row in receipt['steps'])
                ids=record['request']['lesson_ids'];result['selected_lessons']=len(ids)
                consumed={index%len(ids) for row in receipt['steps'] for index in range(row['example_cursor']-row['examples'],row['example_cursor'])}
                result['consumed_lesson_ids_this_job']=[v for index,v in enumerate(ids) if index in consumed]
                result['unconsumed_lesson_ids_this_job']=[v for index,v in enumerate(ids) if index not in consumed]
                result['training_budget']=receipt.get('training_budget')
                result['optimizer_state_policy']=record['candidate']['optimizer_state_policy']
        if 'expected_parent_sha256' in record['request']:
            result['expected_parent_sha256']=record['request']['expected_parent_sha256']
        if record['state']=='failed':result['error']='Training failed; private operator logs retained. No automatic retry.'
        return result

    def submit(self,path,request):
        if path in ('/cow/training/learning-runs','/cow/training/learning-steps'):
            if self.learning is None:raise ValueError('Learning runs are not configured')
            return self.learning.submit(path,request)
        if path not in ('/cow/training/lessons','/cow/training/jobs','/cow/training/exports'):raise ValueError('Unknown training mutation')
        operation=request.get('operation_id')
        if type(operation) is not str or not OP.fullmatch(operation):raise ValueError('Explicit training operation ID required')
        binding=digest([path,request]);target=self.root/'operations'/(operation+'.json')
        with self.locked():
            if target.exists():
                record=self._read(target)
                if record['acceptance']['request_sha256']!=binding:raise ValueError('Operation ID already binds different request bytes')
                return record['acceptance']
            acceptance=dict(schema=1,operation_id=operation,request_sha256=binding)
            if path.endswith('/lessons'):
                if set(request)!={'operation_id','lesson'}:raise ValueError('Lesson request fields are invalid')
                lesson=validate_lesson(request['lesson'])
                if lesson['model_contract_id']!=self.model:raise ValueError('Lesson belongs to another model')
                identity=digest(lesson)
                example=self._example(lesson,identity)
                value=dict(schema=1,lesson_id=identity,lesson=lesson,example=example,verification='authenticated-owner-attestation')
                lesson_path=self.root/'lessons'/(identity+'.json')
                if lesson_path.exists():
                    if self._read(lesson_path)!=value:raise ValueError('Conflicting lesson bytes')
                else:self._write(lesson_path,value)
                acceptance['lesson_id']=identity;state='recorded'
            elif path=='/cow/training/exports':
                fields={'operation_id','job_id','name','expected_sha256','notes','report_sha256'}
                if set(request)!=fields or type(request['job_id']) is not str or not OP.fullmatch(request['job_id']):raise ValueError('Preservation requires a known job ID and exact model/export metadata')
                prior=self._read(self.root/'operations'/(request['job_id']+'.json'))
                if prior['path']!='/cow/training/jobs' or prior['state']!='complete':raise ValueError('Preserve only a completed registered native job')
                from tools.pcc_model_exports import preserve
                saved=preserve(self.root,self.model,prior['candidate'],job_id=request['job_id'],name=request['name'],expected_sha256=request['expected_sha256'],notes=request['notes'],report_sha256=request['report_sha256'],write=self._write)
                acceptance.update(export_id=saved['export_id'],name=saved['name'],model_sha256=saved['model_sha256'],file=saved['file'],weights_published=False);state='preserved'
            else:
                required={'operation_id','lesson_ids','mode','batch_size','steps','learning_rate','rank','parent_job_id'}
                if not required<=set(request) or set(request)-required-{'skill_guard','optimizer_profile_id','expected_parent_sha256'}:raise ValueError('Training accepts only lesson IDs and bounded update settings')
                ids=request['lesson_ids']
                if type(ids) is not list or not 1<=len(ids)<=4096 or len(set(ids))!=len(ids):raise ValueError('Select 1â€“4096 distinct lesson IDs')
                for identity in ids:self._lesson(identity)
                if request['mode'] not in ('lora','full'):raise ValueError('Explicit LoRA or full mode required')
                for key,maximum in [('batch_size',128),('steps',self.max_steps),('rank',256)]:
                    if type(request[key]) is not int or not 1<=request[key]<=maximum:raise ValueError('Invalid '+key)
                if request['steps']<(len(ids)+request['batch_size']-1)//request['batch_size']:
                    raise ValueError('Step budget must cover every selected lesson at least once')
                lr=request['learning_rate']
                if type(lr) not in (int,float) or not math.isfinite(lr) or not 0<lr<=.01:raise ValueError('Learning rate must be positive and at most 0.01')
                if 'optimizer_profile_id' in request:
                    validate_optimizer=getattr(self.runner,'validate_optimizer_request',None)
                    if validate_optimizer is None:raise ValueError('Native optimizer profiles are not configured on this runner')
                    validate_optimizer(request)
                if 'skill_guard' in request:
                    validate=getattr(self.runner,'validate_request',None)
                    if validate is None:raise ValueError('Skill guard is not configured on this server')
                    validate(request)
                parent=request['parent_job_id']
                expected=request.get('expected_parent_sha256')
                if 'expected_parent_sha256' in request:
                    if type(expected) is not str or HEX.fullmatch(expected) is None or parent is None:
                        raise ValueError('Expected parent SHA256 requires an explicit completed parent job')
                if parent is not None:
                    if type(parent) is not str or not OP.fullmatch(parent):raise ValueError('Invalid parent job')
                    prior=self._read(self.root/'operations'/(parent+'.json'))
                    if prior['path']!='/cow/training/jobs' or prior['state']!='complete' or prior['request']['mode']!=request['mode']:raise ValueError('Parent must be a complete candidate of the same mode')
                    if request['mode']=='lora' and prior['request']['rank']!=request['rank']:raise ValueError('LoRA continuation must retain its parent rank')
                    validate_parent=getattr(self.runner,'validate_optimizer_parent',None)
                    if validate_parent is not None:validate_parent(request,prior)
                    if expected is not None and prior['candidate']['candidate_sha256']!=expected:
                        raise ValueError('Completed parent job differs from the expected candidate SHA256')
                acceptance['job_id']=operation;state='queued'
            record=dict(path=path,request=copy.deepcopy(request),acceptance=acceptance,state=state)
            self._write(target,record)
            if state=='queued':self.executor.submit(self._work,operation)
            return acceptance

    def _work(self,operation):
        target=self.root/'operations'/(operation+'.json')
        try:
            # Inherit this lease into the native child: a dead HTTP parent must
            # not free the training/GPU resource leases while CUDA is still live.
            with (self.root/'worker.lock').open('a+') as lease:
                fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
                with self.locked():record=self._read(target);record['state']='running';self._write(target,record)
                request=record['request'];examples=[self.lesson(i)['example'] for i in request['lesson_ids']]
                parent=None
                if request['parent_job_id']:
                    p=self.root/'operations'/(request['parent_job_id']+'.json')
                    parent_record=self._read(p)
                    validate_parent=getattr(self.runner,'validate_optimizer_parent',None)
                    if validate_parent is not None:validate_parent(request,parent_record)
                    parent=parent_record['candidate']
                    if request.get('expected_parent_sha256',parent['candidate_sha256'])!=parent['candidate_sha256']:
                        raise ValueError('Original parent candidate changed before training execution')
                candidate=self.runner(request,examples,self.root/'runs'/operation,parent,lease.fileno())
                with self.locked():
                    record=self._read(target);record.update(state='complete',candidate=candidate);self._write(target,record)
        except Exception as error:
            with self.locked():
                record=self._read(target);record.update(state='failed',private_error=type(error).__name__+': '+str(error));self._write(target,record)

class NativeRunner:
    def __init__(self,config):
        self.config=copy.deepcopy(config)
        required={'model','base_sha256','executable','devices','device_budget_bytes','reserve_bytes','gpu_uuids'}
        if not required<=set(config) or set(config)-required-{'sequence_profile','block_devices','host_reserve_bytes','timeout_seconds','memory_profile','optimizer_profiles'}:raise ValueError('Training runner requires exact owner-configured model, executable and device limits')
        if sha256(config['model'])!=config['base_sha256']:raise ValueError('Training model hash mismatch')
        from tools.pcc_native_training import validate_devices
        names=validate_devices(config['devices'])
        from tools.pcc_native_training import validate_sequence_profile
        validate_sequence_profile(config.get('sequence_profile','short-v1'),names)
        from tools.pcc_native_training import validate_memory_profile
        validate_memory_profile(config.get('memory_profile','conservative-v1'),config.get('sequence_profile','short-v1'))
        if type(config.get('timeout_seconds',3600)) is not int or not 1<=config.get('timeout_seconds',3600)<=3600:
            raise ValueError('Owner native timeout must be 1..3600 seconds')
        from tools.pcc_optimizer_profiles import profiles
        self.registered_optimizer_profiles=profiles(config.get('optimizer_profiles',{}),sequence_profile=config.get('sequence_profile','short-v1'),deadline=config.get('timeout_seconds',3600))
        if any(p['training_budget_seconds'] is not None for p in self.registered_optimizer_profiles.values()):
            from tools.pcc_native_training import require_training_budget_capability
            require_training_budget_capability(config['executable'])
        if 'block_devices' in config:
            placement=config['block_devices']
            if type(placement) is not list or not 3<=len(placement)<=34 or any(type(i) is not int or not 0<=i<len(names) for i in placement):
                raise ValueError('Owner block placement must contain bounded device indices')
        if 'host_reserve_bytes' in config and (type(config['host_reserve_bytes']) is not int or config['host_reserve_bytes']<=0):
            raise ValueError('Positive owner host-memory reserve required')
        if names==['CPU']:
            if config['gpu_uuids']:raise ValueError('CPU training cannot reserve GPUs')
        elif names!=['CUDA'+str(i) for i in range(len(names))] or len(config['gpu_uuids'])!=len(names) or len(set(config['gpu_uuids']))!=len(names):
            raise ValueError('Training requires a distinct physical UUID per visible CUDA device')

    def describe_optimizer_profiles(self):
        from tools.pcc_optimizer_profiles import identity
        if any(identity(p)!=key for key,p in self.registered_optimizer_profiles.items()):raise ValueError('Owner optimizer profiles changed after configuration')
        return [dict(id=key,**copy.deepcopy(p),may_stop_before_all_selected_lessons=p['training_budget_seconds'] is not None) for key,p in sorted(self.registered_optimizer_profiles.items())]

    def validate_optimizer_request(self,request):
        from tools.pcc_optimizer_profiles import select
        return select(self.registered_optimizer_profiles,request)

    def validate_optimizer_parent(self,request,parent_record):
        from tools.pcc_optimizer_profiles import validate_parent
        validate_parent(self.registered_optimizer_profiles,request,parent_record)

    def __call__(self,request,examples,run,parent,worker_fd):
        c=self.config;environment=dict(os.environ);fds=[worker_fd]
        profile=self.validate_optimizer_request(request)
        if profile:environment.update(GGML_CUDA_F32_ACTIVATIONS='1',NVIDIA_TF32_OVERRIDE='0')
        with ExitStack() as stack:
            if c['gpu_uuids']:
                rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True)
                indices={p[1].strip():int(p[0]) for line in rows.splitlines() if (p:=line.split(','))}
                directory=Path.home()/'.local/state/context-cow';directory.mkdir(mode=0o700,parents=True,exist_ok=True)
                for identity in sorted(c['gpu_uuids']):
                    if identity not in indices:raise ValueError('Configured training GPU is unavailable')
                    lock=stack.enter_context((directory/f'gpu-{indices[identity]}.lock').open('a+'))
                    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fds.append(lock.fileno())
                environment.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=','.join(c['gpu_uuids']))
            native={k:c[k] for k in ['model','base_sha256','devices','device_budget_bytes','reserve_bytes']}
            native.update({k:request[k] for k in ['mode','rank','batch_size','steps','learning_rate']},examples=examples,optimizer='sgd',backprop_mode='standard')
            for key in ('sequence_profile','block_devices','host_reserve_bytes','memory_profile'):
                if key in c:native[key]=c[key]
            if parent:
                if request['mode']=='lora':native.update(initial_adapter=parent['candidate_path'],initial_adapter_sha256=parent['candidate_sha256'])
                else:native.update(model=parent['candidate_path'],base_sha256=parent['candidate_sha256'])
            native['deadline_seconds']=c.get('timeout_seconds',3600)
            if getattr(self,'record_process_identity',False):native['record_process_identity']=True
            if profile:
                native.update(optimizer=profile['optimizer'],optimizer_settings=profile['optimizer_settings'],optimizer_state_mode=profile['state_policy'])
                if profile['training_budget_seconds'] is not None:native['training_budget_seconds']=profile['training_budget_seconds']
                if profile['state_policy']=='resume':
                    if parent is None or parent.get('mode')!=request['mode']:raise ValueError('Completed same-mode parent required for optimizer resume')
                    checkpoint=parent['receipt']['optimizer_checkpoint']
                    native.update(initial_optimizer_state=checkpoint['path'],initial_optimizer_state_sha256=checkpoint['sha256'],example_cursor=parent['receipt']['example_cursor'])
            candidate=execute_native(c['executable'],native,run,timeout=c.get('timeout_seconds',3600)+15,environment=environment,pass_fds=tuple(fds))
            if profile:candidate.update(optimizer_profile_id=request['optimizer_profile_id'],optimizer_state_policy=profile['state_policy'])
            return candidate

def configured_service(config_path,profile,tokenizer):
    config=json.loads(Path(config_path).read_bytes())
    fields={'schema','directory','max_tokens','max_steps','runner'}
    if set(config) not in (fields,fields|{'skill_guard_config'}) or config['schema']!=1:
        raise ValueError('Invalid optional training configuration')
    if Path(config['runner']['model']).resolve()!=Path(profile['model_path']).resolve():
        raise ValueError('Training tokenizer must belong to the configured serving model')
    runner=NativeRunner(config['runner'])
    if 'skill_guard_config' in config:
        if runner.registered_optimizer_profiles:raise ValueError('Native optimizer profiles currently require explicit lesson blocks; legacy multi-round skill_guard integration is not yet qualified')
        try:from sss.adapters.pcc import configured_guard
        except ModuleNotFoundError as error:
            if error.name=='sss':raise ValueError('Install the optional Skills³ package from CC-David-CC/sss before enabling skill_guard_config') from None
            raise
        runner=configured_guard(config['skill_guard_config'],runner,tokenizer)
    return LessonService(config['directory'],profile['model_contract_id'],tokenizer.encode,runner,
                         max_tokens=config['max_tokens'],max_steps=config['max_steps'],sequence_profile=config['runner'].get('sequence_profile','short-v1'),
                         rendered_tokenize=lambda text:tokenizer.encode(text,parse_special=True))
