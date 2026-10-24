"""Offline PCC native-training admission and immutable candidate registration.

The optional PCC lesson/job service reuses this executor; no PyTorch training.
A training candidate is never an
automatic replacement for the model/adapter pinned by a serving request.
"""
from pathlib import Path
import copy
import hashlib
import json
import math
import os
import re
import subprocess
import uuid

def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()

def optimizer_dataset_sha256(examples):
    raw=json.dumps(examples,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()

def read_optimizer_state(path,expected_sha256):
    path=Path(path).resolve(strict=True)
    if not path.is_file() or not re.fullmatch('[0-9a-f]{64}',str(expected_sha256)) or sha256(path)!=expected_sha256:
        raise ValueError('Pinned optimizer-state artifact mismatch')
    with path.open('rb') as stream:
        if stream.read(8)!=b'PCCOPT1\n':raise ValueError('Unknown optimizer-state wire format')
        raw=stream.read(8)
        if len(raw)!=8:raise ValueError('Truncated optimizer-state length')
        length=int.from_bytes(raw,'little')
        if not 0<length<=4*1024*1024:raise ValueError('Bounded optimizer-state header required')
        raw=stream.read(length)
        if len(raw)!=length:raise ValueError('Truncated optimizer-state metadata')
        value=json.loads(raw)
    keys={'schema','kind','storage','model_sha256','dataset_sha256','example_cursor','completed_steps','config','tensors'}
    if type(value) is not dict or set(value)!=keys or value['schema']!=1 or value['kind']!='pcc.completed-optimizer-state.v1' or value['storage']!='le-f32':
        raise ValueError('Unknown optimizer-state schema')
    if type(value['completed_steps']) is not int or not 0<value['completed_steps']<=1000000000:
        raise ValueError('Completed optimizer step required')
    return value

def validate_optimizer_request(request):
    kind=request.get('optimizer','sgd')
    if kind not in ('sgd','adamw','muon'):raise ValueError('Unknown native optimizer')
    if kind=='sgd':
        if any(k in request for k in ('optimizer_settings','initial_optimizer_state','initial_optimizer_state_sha256','optimizer_state_mode','optimizer_state_output')):
            raise ValueError('SGD does not consume adaptive optimizer state/settings')
        return kind
    if request.get('sequence_profile')!='dense-llama-short-v1' or (request.get('mode')!='full' and not (kind=='adamw' and request.get('mode')=='lora')):
        raise ValueError('Adaptive optimizers require ordinary-Llama full training or explicit AdamW LoRA')
    if request.get('mode')=='lora' and request.get('optimizer_state_mode')=='resume' and not request.get('initial_adapter'):
        raise ValueError('AdamW LoRA state resume requires its exact parent adapter')
    if request.get('optimizer_state_mode') not in ('fresh','resume'):raise ValueError('Explicit fresh/resume optimizer state mode required')
    settings=request.get('optimizer_settings',{})
    allowed={'schema','aux_learning_rate','beta1','beta2','epsilon','momentum','nesterov','ns_steps','projection','scaling','weight_decay','aux_weight_decay','normalization_epsilon','refine_start_step','refine_steps','precision'}
    if type(settings) is not dict or set(settings)-allowed:raise ValueError('Unknown optimizer setting')
    if settings.get('schema',1)!=1 or settings.get('precision','native-f32-v1')!='native-f32-v1':raise ValueError('Unsupported optimizer settings/precision')
    if request.get('evaluate_only') or request.get('gradient_probe_only'):raise ValueError('Use SGD selection for probes that do not consume optimizer state')
    cursor=request.get('example_cursor',0)
    if type(cursor) is not int or not 0<=cursor<=1000000000000:raise ValueError('Bounded optimizer data cursor required')
    if request['optimizer_state_mode']=='fresh':
        if cursor or any(k in request for k in ('initial_optimizer_state','initial_optimizer_state_sha256')):
            raise ValueError('Fresh optimizer cannot silently reuse supplied moments or a cursor')
    else:
        if any(k not in request for k in ('initial_optimizer_state','initial_optimizer_state_sha256','example_cursor')):raise ValueError('Complete pinned optimizer resume request required')
    return kind

SEQUENCE_PROFILES={'short-v1':8192,'tiny-cpu-long-v1':32768,'dense-short-v1':512,'dense-llama-short-v1':512}

def validate_sequence_profile(profile,devices):
    if profile not in SEQUENCE_PROFILES:raise ValueError('Unknown sequence qualification profile')
    if profile=='tiny-cpu-long-v1' and devices!=['CPU']:
        raise ValueError('Long sequence profile is only for tiny CPU fixtures')
    return profile

def validate_memory_profile(profile,sequence_profile):
    if profile not in ('conservative-v1','allocator-v1') or (profile=='allocator-v1' and sequence_profile not in ('dense-short-v1','dense-llama-short-v1')):
        raise ValueError('Allocator admission requires the bounded dense sequence profile')
    return profile

def validate_examples(examples,*,sequence_profile='short-v1'):
    if sequence_profile not in SEQUENCE_PROFILES:raise ValueError('Unknown sequence qualification profile')
    maximum=SEQUENCE_PROFILES[sequence_profile]
    if type(examples) is not list or not examples:
        raise ValueError('A nonempty explicit training split is required')
    names=set()
    for e in examples:
        if type(e) is not dict or set(e)!={'id','tokens','targets','loss_mask','verified','provenance_sha256'}:
            raise ValueError('Explicit owned example, verification and provenance required')
        if type(e['id']) is not str or not re.fullmatch(r'[A-Za-z0-9_-]{1,128}',e['id']) or e['id'] in names:
            raise ValueError('Unique bounded example IDs required')
        names.add(e['id'])
        if e['verified'] is not True or not re.fullmatch('[a-f0-9]{64}',str(e['provenance_sha256'])):
            raise ValueError('Only independently verified examples with provenance are admitted')
        if any(type(e[k]) is not list for k in ('tokens','targets','loss_mask')):raise ValueError('Token arrays required')
        n=len(e['tokens'])
        if not 2<=n<=maximum or len(e['targets'])!=n or len(e['loss_mask'])!=n:
            raise ValueError('Bounded aligned inputs/next-token targets/masks required')
        if any(type(t) is not int or t<0 for t in e['tokens']+e['targets']):
            raise ValueError('Nonnegative integer token IDs required')
        if any(type(m) is not bool for m in e['loss_mask']) or not any(e['loss_mask']):
            raise ValueError('Nonempty explicit completion-loss mask required')
        if sequence_profile in ('tiny-cpu-long-v1','dense-short-v1','dense-llama-short-v1') and sum(e['loss_mask'])>256:raise ValueError('Balanced profiles admit at most 256 selected targets')
        if e['targets'][:-1]!=e['tokens'][1:]:
            raise ValueError('Targets must describe the next consumed token')
    return copy.deepcopy(examples)

def batch_weights(count,batch_size):
    if type(count) is not int or type(batch_size) is not int or count<=0 or batch_size<=0:
        raise ValueError('Positive example count and batch size required')
    return [1./min(batch_size,count-(i//batch_size)*batch_size) for i in range(count)]

def validate_backprop_mode(mode=None,*,devices=1,experimental=False):
    if type(devices) is not int or devices<=0:
        raise ValueError('Positive device count required')
    mode='standard' if mode is None else mode
    if mode=='standard':return mode
    if mode=='layer_checkpointed' and experimental is True:return mode
    raise ValueError('Section/checkpointed backprop is a separate explicit experimental mode')

def promotion_eligible(report):
    return all(report.get(k) is True for k in ('repairs','held_out','preservation'))

def validate_devices(devices):
    if type(devices) is not list or not devices or any(type(d) is not str or not re.fullmatch(r'CPU|CUDA(?:0|[1-9][0-9]*)',d) for d in devices):
        raise ValueError('Explicit canonical CPU or CUDA device names required')
    if len(set(devices))!=len(devices):
        raise ValueError('A physical training backend cannot be counted twice')
    if 'CPU' in devices and len(devices)!=1:
        raise ValueError('Mixed CPU/CUDA placement is not qualified')
    return list(devices)

def finalize_candidate(versions,base,candidate,mode,receipt):
    if mode not in ('lora','full') or receipt.get('native') is not True:
        raise ValueError('A completed native execution receipt and explicit mode required')
    base=Path(base).resolve(strict=True);candidate=Path(candidate).resolve(strict=True)
    if base==candidate:raise ValueError('Base model cannot be overwritten')
    if not base.is_file() or not candidate.is_file():raise ValueError('Regular immutable artifacts required')
    identity=sha256(candidate)
    versions=Path(versions);versions.mkdir(parents=True,exist_ok=True)
    destination=versions/(identity+'.json')
    value=dict(schema=1,mode=mode,base_sha256=sha256(base),candidate_sha256=identity,
               candidate_path=str(candidate),promoted=False,receipt=copy.deepcopy(receipt))
    if receipt.get('optimizer','sgd')!='sgd':
        checkpoint=receipt.get('optimizer_checkpoint',{})
        state=read_optimizer_state(checkpoint['path'],checkpoint['sha256'])
        if state['model_sha256']!=identity:raise ValueError('Optimizer state belongs to different candidate weights')
        value.update(optimizer_state_path=checkpoint['path'],optimizer_state_sha256=checkpoint['sha256'],optimizer_completed_steps=state['completed_steps'])
    # Exclusive creation: a duplicate training completion reads/reconciles this
    # version rather than silently overwriting it or issuing another update.
    temporary=versions/('.pending-'+uuid.uuid4().hex+'.json')
    try:
        with temporary.open('x',encoding='utf-8',newline='\n') as stream:
            stream.write(json.dumps(value,sort_keys=True,indent=2)+'\n')
            stream.flush();os.fsync(stream.fileno())
        # Publish a complete manifest without replacing an existing version.
        os.link(temporary,destination)
        if os.name=='posix':
            fd=os.open(versions,os.O_RDONLY|os.O_DIRECTORY)
            try:os.fsync(fd)
            finally:os.close(fd)
    finally:
        if temporary.exists():temporary.unlink()
    return value

def validate_training_budget_request(request):
    if 'training_budget_seconds' not in request:return False
    value=request['training_budget_seconds'];deadline=request.get('deadline_seconds',3600)
    if type(value) not in (int,float) or not math.isfinite(value) or value<=0 or type(deadline) is not int or not 1<=deadline<=3600 or value>=deadline:
        raise ValueError('Positive finite training phase budget below the job deadline required')
    if request.get('mode')!='full' or request.get('sequence_profile')!='dense-llama-short-v1' or request.get('evaluate_only') or request.get('gradient_probe_only'):
        raise ValueError('Training phase budget requires ordinary-Llama full standard backpropagation')
    if type(request.get('steps',1)) is not int or not 1<=request.get('steps',1)<=1000 or type(request.get('batch_size',1)) is not int or request.get('batch_size',1)<1:
        raise ValueError('Bounded positive update cap and batch size required')
    return True


def require_training_budget_capability(executable,*,environment=None,pass_fds=()):
    result=subprocess.run([str(executable),'--capabilities'],capture_output=True,text=True,timeout=10,env=environment,pass_fds=pass_fds)
    try:value=json.loads(result.stdout)
    except (ValueError,TypeError):value={}
    expected=dict(schema=1,field='training_budget_seconds',boundary='whole_update',clock='steady_training_phase',mode='full',sequence_profile='dense-llama-short-v1')
    if result.returncode or type(value) is not dict or value.get('schema')!=1 or value.get('kind')!='pcc.native-training-capabilities.v1' or value.get('completed_update_budget')!=expected:
        raise ValueError('Native executable does not advertise the completed-update training budget; no update launched')
    return value


def validate_training_budget_receipt(request,receipt):
    budget=receipt.get('training_budget');rows=receipt.get('steps');cap=request.get('steps',1)
    if type(budget) is not dict or budget.get('schema')!=1 or budget.get('boundary')!='whole_update' or budget.get('clock')!='steady_training_phase' or budget.get('requested_seconds')!=request['training_budget_seconds'] or budget.get('step_cap')!=cap:
        raise ValueError('Native training-budget receipt differs from the admitted request')
    if type(rows) is not list or not 1<=len(rows)<=cap or type(budget.get('completed_updates')) is not int or budget['completed_updates']!=len(rows):
        raise ValueError('Completed-update budget counters differ')
    numbers=[budget.get(k) for k in ('elapsed_seconds','overrun_seconds','remaining_seconds')]
    if any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in numbers):raise ValueError('Finite training-budget accounting required')
    elapsed,overrun,remaining=numbers;requested=request['training_budget_seconds'];spans=[];cursor=request.get('example_cursor',0)
    for index,row in enumerate(rows):
        timing=row.get('timing',{});values=[timing.get(k) for k in ('seconds','gradient_and_coverage_seconds','optimizer_seconds','post_update_evaluation_seconds')]
        if any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in values) or values[0]<=0 or sum(values[1:])>values[0]+1e-8:raise ValueError('Finite completed-update timing required')
        spans.append(values[0]);count=min(request.get('batch_size',1),len(request['examples'])-(cursor%len(request['examples'])));cursor+=count
        if row.get('step')!=index or row.get('examples')!=count or row.get('example_cursor')!=cursor:raise ValueError('Budgeted example cursor differs from completed batches')
    close=lambda a,b:math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-8)
    if not close(sum(spans),elapsed) or not close(overrun,max(0.,elapsed-requested)) or not close(remaining,max(0.,requested-elapsed)) or overrun>spans[-1]+1e-8:
        raise ValueError('Completed-update budget accounting is inconsistent')
    reason='step_cap' if len(rows)==cap else 'training_budget'
    if budget.get('stop_reason')!=reason or (reason=='training_budget' and elapsed<requested) or receipt.get('example_cursor')!=cursor:
        raise ValueError('Training budget did not stop at the declared completed boundary')
    return len(rows)


def execute_native(executable,request,run_directory,*,timeout=3600,environment=None,pass_fds=()):
    """Run one explicit native update request and preserve its exact receipts.

    Run-directory identity prevents an uncertain completion from turning into
    an automatic second update. Reconciliation reads that directory; it never
    blindly launches the executable again. This helper does not promote output.
    """
    request=copy.deepcopy(request)
    request['devices']=validate_devices(request['devices'])
    profile=validate_sequence_profile(request.get('sequence_profile','short-v1'),request['devices'])
    validate_memory_profile(request.get('memory_profile','conservative-v1'),profile)
    request['examples']=validate_examples(request['examples'],sequence_profile=profile)
    request['backprop_mode']=validate_backprop_mode(request.get('backprop_mode'),devices=len(request['devices']))
    if request.get('mode') not in ('lora','full'):
        raise ValueError('Explicit LoRA or direct full-weight update required')
    optimizer_kind=validate_optimizer_request(request)
    timed=validate_training_budget_request(request)
    rank=request.get('rank',4)
    if type(rank) is not int or not 0<rank<=256:
        raise ValueError('Explicit bounded LoRA rank required')
    if type(request.get('learning_rate')) not in (int,float) or not math.isfinite(request['learning_rate']) or request['learning_rate']<=0:
        raise ValueError('Positive finite learning rate required')
    base=Path(request['model']).resolve(strict=True)
    original=sha256(base)
    if request.get('base_sha256')!=original:
        raise ValueError('Pinned immutable base hash mismatch')
    parent_adapter=None
    if 'initial_adapter' in request:
        if request['mode']!='lora':raise ValueError('Direct full-weight mode does not merge a LoRA')
        adapter=Path(request['initial_adapter']).resolve(strict=True)
        parent_adapter=sha256(adapter)
        if request.get('initial_adapter_sha256')!=parent_adapter:
            raise ValueError('Pinned parent adapter hash mismatch')
        request['initial_adapter']=str(adapter)
    parent_optimizer=None
    if optimizer_kind!='sgd' and request['optimizer_state_mode']=='resume':
        state=Path(request['initial_optimizer_state']).resolve(strict=True)
        parent_optimizer=read_optimizer_state(state,request['initial_optimizer_state_sha256'])
        trained_identity=parent_adapter if request['mode']=='lora' else original
        if parent_optimizer['model_sha256']!=trained_identity or parent_optimizer['dataset_sha256']!=optimizer_dataset_sha256(request['examples']) or parent_optimizer['example_cursor']!=request['example_cursor']:
            raise ValueError('Optimizer state candidate/dataset/cursor mismatch')
        request['initial_optimizer_state']=str(state)
    executable=Path(executable).resolve(strict=True)
    capabilities=require_training_budget_capability(executable,environment=environment,pass_fds=pass_fds) if timed else None
    run_directory=Path(run_directory).resolve()
    run_directory.mkdir(mode=0o700,parents=True,exist_ok=False)
    request['model']=str(base)
    request['candidate']=str(run_directory/'candidate.gguf')
    if optimizer_kind!='sgd':request['optimizer_state_output']=str(run_directory/'optimizer-state.bin')
    config=run_directory/'request-private.json'
    config.write_text(json.dumps(request,sort_keys=True,indent=2)+'\n',encoding='utf-8',newline='\n')
    config.chmod(0o600)
    launch=dict(executable=str(executable),executable_sha256=sha256(executable),
                request_sha256=sha256(config),base_sha256=original,parent_adapter_sha256=parent_adapter)
    if capabilities is not None:launch['capabilities']=capabilities
    (run_directory/'launch.json').write_text(json.dumps(launch)+'\n')
    try:
        with (run_directory/'stdout-private.json').open('x',encoding='utf-8') as out, (run_directory/'stderr.log').open('x',encoding='utf-8') as err:
            proc=subprocess.Popen([str(executable),'--train',str(config)],stdout=out,stderr=err,
                                  start_new_session=(os.name=='posix'),env=environment,pass_fds=pass_fds)
            launch['pid']=proc.pid
            (run_directory/'launch.json').write_text(json.dumps(launch)+'\n')
            try:code=proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill();proc.wait();raise
        (run_directory/'command.exit').write_text(str(code)+'\n')
        if code:
            raise RuntimeError('Native training failed; preserve this run and inspect stderr.log')
        if sha256(base)!=original:
            raise RuntimeError('Base model changed during training; candidate is not registered')
        receipt=json.loads((run_directory/'stdout-private.json').read_text(encoding='utf-8'))
        if receipt.get('native') is not True or receipt.get('backprop_mode')!='standard' or receipt.get('mode')!=request['mode']:
            raise ValueError('Native execution receipt does not match the admitted request')
        if receipt.get('optimizer','sgd')!=optimizer_kind:raise ValueError('Native optimizer differs from the admitted request')
        completed_updates=validate_training_budget_receipt(request,receipt) if timed else request.get('steps',1)
        if optimizer_kind!='sgd':
            checkpoint=receipt.get('optimizer_checkpoint',{})
            if checkpoint.get('path')!=request['optimizer_state_output'] or checkpoint.get('candidate_sha256')!=sha256(request['candidate']) or checkpoint.get('dataset_sha256')!=optimizer_dataset_sha256(request['examples']):
                raise ValueError('Optimizer checkpoint is not bound to the completed candidate/dataset')
            state=read_optimizer_state(checkpoint['path'],checkpoint['sha256'])
            expected_steps=(0 if parent_optimizer is None else parent_optimizer['completed_steps'])+completed_updates
            if state['completed_steps']!=expected_steps or state['example_cursor']!=receipt['example_cursor'] or state['model_sha256']!=checkpoint['candidate_sha256'] or state['config']!=checkpoint['config']:
                raise ValueError('Optimizer checkpoint counters/configuration differ from completion')
        receipt.update(request_sha256=launch['request_sha256'],parent_adapter_sha256=parent_adapter)
        return finalize_candidate(run_directory/'versions',base,request['candidate'],request['mode'],receipt)
    except subprocess.TimeoutExpired:
        (run_directory/'terminal-status.txt').write_text('TIMEOUT: completion unknown; inspect original owned process and run before reconciliation\n')
        raise
