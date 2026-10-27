"""Opt-in original native completion evidence; inspection/adoption never dispatch.

NativeRunner owns normal execution and the independent forward-only reload hook.
A completed original job can later be adopted by explicit owner approval. Neither
that adoption nor its reload gate is model promotion or general qualification.
"""
from pathlib import Path
import copy, fcntl, hashlib, json, math, os, re, stat, struct, subprocess
from contextlib import contextmanager

BOUND = 2e-4
APPROVAL = 'adopt-completed-original-native-v1'
HEX = re.compile('[0-9a-f]{64}')
ROUTE = '/cow/training/job-reconciliations'

class RecoveryHold(ValueError):
    def __init__(self,code,message):
        self.code=code
        super().__init__(message)

def strict_json(raw):
    def pairs(items):
        result={}
        for key,value in items:
            if key in result:raise ValueError('Duplicate original JSON field')
            result[key]=value
        return result
    def nonfinite(value):raise ValueError('Nonfinite original JSON value')
    def floating(value):
        result=float(value)
        if not math.isfinite(result):nonfinite(value)
        return result
    try:return json.loads(raw,object_pairs_hook=pairs,parse_constant=nonfinite,parse_float=floating)
    except (RecursionError,OverflowError) as error:raise ValueError('Original JSON exceeds safe bounds') from error


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()


def digest(value):return hashlib.sha256(canonical(value)).hexdigest()


def sha(path):
    path=Path(path)
    if path.is_symlink() or not path.is_file():raise ValueError('Exact regular original artifact required')
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):
    path=Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_uid!=os.getuid() or path.stat().st_mode&0o077:
        raise ValueError('Private owned original record required')
    if path.stat().st_size>64*1024**2:raise ValueError('Original metadata exceeds bounded record size')
    value=strict_json(path.read_bytes())
    if path.name=='command.exit':
        if type(value) is not int:raise ValueError('Original command exit requires an exact integer')
    elif type(value) is not dict:raise ValueError('Original metadata requires a JSON object')
    return value


def read_version(path):
    path=Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_uid!=os.getuid():raise ValueError('Original regular owned native version required')
    if path.stat().st_size>64*1024**2:raise ValueError('Original metadata exceeds bounded record size')
    value=strict_json(path.read_bytes())
    if type(value) is not dict:raise ValueError('Original candidate version requires a JSON object')
    return value


def write(path,value):
    path=Path(path)
    with path.open('xb') as stream:
        stream.write(canonical(value)+b'\n');stream.flush();os.fsync(stream.fileno())
    path.chmod(0o600)
    fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def configuration(value):
    if value is None:return None
    if (type(value) is not dict or set(value)!={'schema','source_closure'} or type(value['schema']) is not int or value['schema']!=1
            or type(value['source_closure']) is not dict or not 1<=len(value['source_closure'])<=512):
        raise ValueError('Opt-in completion recovery needs a pinned owner source closure')
    for path,expected in value['source_closure'].items():
        if type(path) is not str or type(expected) is not str or not Path(path).is_absolute() or not HEX.fullmatch(expected) or sha(path)!=expected:
            raise ValueError('Original owner source closure differs')
    return copy.deepcopy(value)


def sources(config):
    recovery=configuration(config.get('completion_recovery'))
    if recovery is None:raise ValueError('Original completion provenance is not enabled; legacy job cannot be adopted')
    executable=str(Path(config['executable']).resolve(strict=True))
    if recovery['source_closure'].get(executable)!=sha(executable):
        raise ValueError('Native executable is absent from the qualified owner closure')
    root=Path(__file__).resolve().parents[1]
    paths=['tools/pcc_native_completion.py','tools/pcc_native_training.py','tools/pcc_lessons.py',
           'tools/pcc_optimizer_profiles.py','tools/pcc_lesson_segments.py','tools/pcc_learning_runs.py',
           'tools/strata_tokenizer.py','tools/gguf_reader.py',
           'experimental/native_training/native_train.cpp','experimental/native_training/native_optimizer.h',
           'experimental/native_training/optimizer_state.h']
    result={str(root/p):sha(root/p) for p in paths}
    for path,expected in recovery['source_closure'].items():
        if path in result and result[path]!=expected:raise ValueError('Conflicting original source closure')
        result[path]=expected
    return result


# Explicit CPU/CUDA numerical, backend-selection and loader inputs. No whole
# environment capture, credentials or unrelated application settings. The owner
# closure must separately bind native binaries/dependencies; this is not proof of
# unrecorded legacy process settings or of unsupported backends.
RUNTIME_KEYS=(
    'GGML_CUDA_F32_ACTIVATIONS','NVIDIA_TF32_OVERRIDE','CUDA_VISIBLE_DEVICES','CUDA_DEVICE_ORDER',
    'GGML_CUDA_CUBLAS_COMPUTE_TYPE','GGML_CUDA_DISABLE_FUSION','COW_QWEN35_F32_ATTENTION',
    'GGML_CUDA_DISABLE_GRAPHS','GGML_CUDA_PDL','GGML_CUDA_ALLREDUCE','GGML_CUDA_DEVICES',
    'GGML_CUDA_ENABLE_UNIFIED_MEMORY','GGML_CUDA_GRAPH_OPT','GGML_CUDA_NO_PINNED',
    'GGML_CUDA_P2P','GGML_CUDA_REGISTER_HOST','GGML_OP_OFFLOAD_MIN_BATCH','GGML_BACKEND_PATH',
    'GGML_CPU_DISABLE_FUSION','GGML_NO_IQ_PANEL','GGML_KLEIDIAI_CHUNK_MULTIPLIER',
    'GGML_KLEIDIAI_SME','GGML_TOTAL_THREADS','OMP_NUM_THREADS','OMP_DYNAMIC','OMP_PROC_BIND',
    'OMP_PLACES','OMP_WAIT_POLICY','KMP_BLOCKTIME','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS',
    'LD_LIBRARY_PATH','LD_PRELOAD')

def runtime_values(environment=None):
    environment=os.environ if environment is None else environment
    result={k:environment.get(k) for k in RUNTIME_KEYS}
    if any(value is not None and (type(value) is not str or len(value.encode('utf8'))>8192) for value in result.values()):
        raise ValueError('Bounded original numerical runtime values required')
    return result


def runtime_environment(runner,request):
    env=dict(os.environ)
    if runner.validate_optimizer_request(request):env.update(GGML_CUDA_F32_ACTIVATIONS='1',NVIDIA_TF32_OVERRIDE='0')
    if runner.config['gpu_uuids']:
        env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=','.join(runner.config['gpu_uuids']))
    return runtime_values(env)


def context(runner,request,examples,parent):
    if not hasattr(runner,'completion_lesson_reader'):raise ValueError('Recoverable runner must be bound to its actual lesson service')
    lessons=[runner.completion_lesson_reader(i) for i in request['lesson_ids']]
    if [row['example'] for row in lessons]!=examples:raise ValueError('Exact original lesson rendering differs')
    return dict(schema=1,kind='pcc.original-native-completion.v1',model_contract_id=runner.completion_model,
        job_request=copy.deepcopy(request),job_request_sha256=digest(['/cow/training/jobs',request]),
        lessons=copy.deepcopy(lessons),parent=copy.deepcopy(parent),runner_config=copy.deepcopy(runner.config),
        sources_sha256=sources(runner.config),runtime_environment=runtime_environment(runner,request),reload_policy=dict(kind='last-completed-batch-mean-v1',absolute_bound=BOUND))


def native_request(native,run):
    from tools.pcc_native_training import validate_devices,validate_examples,validate_backprop_mode
    value=copy.deepcopy(native)
    value['devices']=validate_devices(value['devices'])
    value['examples']=validate_examples(value['examples'],sequence_profile=value.get('sequence_profile','short-v1'))
    value['backprop_mode']=validate_backprop_mode(value.get('backprop_mode'),devices=len(value['devices']))
    value['model']=str(Path(value['model']).resolve(strict=True))
    if 'initial_adapter' in value:value['initial_adapter']=str(Path(value['initial_adapter']).resolve(strict=True))
    if 'initial_optimizer_state' in value:value['initial_optimizer_state']=str(Path(value['initial_optimizer_state']).resolve(strict=True))
    value['candidate']=str(Path(run).resolve()/'candidate.gguf')
    if value.get('optimizer','sgd')!='sgd':value['optimizer_state_output']=str(Path(run).resolve()/'optimizer-state.bin')
    return value


def prepare(run,value,request):
    # Normal execution only, before the native child is dispatched.
    value=copy.deepcopy(value);value['native_request']=copy.deepcopy(request)
    value['native_request_sha256']=sha(Path(run)/'request-private.json')
    value['intent_sha256']=digest(value)
    write(Path(run)/'completion-intent.json',value)
    return value


def process_identity(pid):
    raw=Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=pid,start_ticks=int(raw[19]),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())


def require_gone(identity):
    if (type(identity) is not dict or set(identity)!={'pid','start_ticks','boot_id'}
            or type(identity['pid']) is not int or identity['pid']<=0
            or type(identity['start_ticks']) is not int or identity['start_ticks']<0
            or type(identity['boot_id']) is not str or not identity['boot_id']):
        raise ValueError('Original exact native process identity is missing')
    try:current=process_identity(identity['pid'])
    except FileNotFoundError:return
    if current==identity:raise RecoveryHold('original_process_live','Original native/reload child is still present')


def optimizer_configuration(request):
    settings=request.get('optimizer_settings',{})
    if type(settings) is not dict:raise ValueError('Original optimizer settings require an object')
    def f32(value):return struct.unpack('<f',struct.pack('<f',value))[0]
    floats=dict(learning_rate=request['learning_rate'],aux_learning_rate=settings.get('aux_learning_rate',.0003),
        beta1=settings.get('beta1',.9),beta2=settings.get('beta2',.95),epsilon=settings.get('epsilon',1e-8),
        momentum=settings.get('momentum',.95),weight_decay=settings.get('weight_decay',0),
        aux_weight_decay=settings.get('aux_weight_decay',0),normalization_epsilon=settings.get('normalization_epsilon',1e-7))
    value=dict(schema=1,optimizer=request['optimizer'],precision='native-f32-v1',
        nesterov=settings.get('nesterov',True),ns_steps=settings.get('ns_steps',5),
        projection=settings.get('projection','jordan5'),scaling=settings.get('scaling','original'),
        refine_start_step=settings.get('refine_start_step',1),refine_steps=settings.get('refine_steps',5),
        **{k:f32(v) for k,v in floats.items()})
    if 'lora_update_space' in request:value['lora_update_space']=request['lora_update_space']
    return value


def validate_state_payload(path,state):
    if (type(state) is not dict or type(state.get('example_cursor')) is not int or state['example_cursor']<0
            or type(state.get('tensors')) is not list or not state['tensors']):
        raise ValueError('Original optimizer payload metadata differs')
    total=0;names=set()
    for tensor in state['tensors']:
        if (type(tensor) is not dict or set(tensor)!={'name','shape','role','optimizer','bytes'}
                or type(tensor['name']) is not str or not tensor['name'] or tensor['name'] in names
                or type(tensor['shape']) is not list or len(tensor['shape'])!=2
                or any(type(x) is not int or x<=0 for x in tensor['shape'])
                or type(tensor['bytes']) is not int or tensor['bytes']!=4*math.prod(tensor['shape'])):
            raise ValueError('Original optimizer tensor layout differs')
        names.add(tensor['name']);total+=tensor['bytes']
    with Path(path).open('rb') as stream:
        stream.seek(8);length=int.from_bytes(stream.read(8),'little');stream.seek(16+length)
        if Path(path).stat().st_size!=16+length+total:raise ValueError('Original optimizer payload length differs')
        while True:
            raw=stream.read(1<<20)
            if not raw:break
            if len(raw)%4 or any(not math.isfinite(value[0]) for value in struct.iter_unpack('<f',raw)):
                raise ValueError('Original optimizer payload is nonfinite or partial')


def validate_receipt(request,receipt):
    from tools.pcc_native_training import (validate_optimizer_request,validate_training_budget_request,
        validate_training_budget_receipt,read_optimizer_state,optimizer_dataset_sha256)
    kind=validate_optimizer_request(request)
    if (type(receipt) is not dict or receipt.get('native') is not True or receipt.get('status')!='MECHANICS_EXECUTED'
            or receipt.get('mode')!=request['mode'] or receipt.get('backprop_mode')!='standard'
            or receipt.get('optimizer','sgd')!=kind or receipt.get('promoted') is not False
            or receipt.get('save_reload_qualified') is not False
            or receipt.get('candidate')!=request['candidate']):
        raise ValueError('Original native receipt does not match the admitted execution')
    rows=receipt.get('steps')
    if type(rows) is not list or not rows:raise ValueError('Original native step rows require a nonempty list')
    for row in rows:
        if type(row) is not dict:raise ValueError('Original native step row requires an object')
        if (type(row.get('gradients')) is not list or not row['gradients']
                or any(type(g) is not dict for g in row['gradients'])):
            raise ValueError('Original native gradient rows require objects')
        if 'timing' in row and type(row['timing']) is not dict:raise ValueError('Original native timing requires an object')
    count=validate_training_budget_receipt(request,receipt) if validate_training_budget_request(request) else request.get('steps',1)
    cursor=request.get('example_cursor',0)
    if len(rows)!=count:raise ValueError('Actual completed native update count differs')
    for index,row in enumerate(rows):
        batch=min(request['batch_size'],len(request['examples'])-cursor%len(request['examples']))
        cursor+=batch
        if (type(row.get('step')) is not int or row['step']!=index or type(row.get('examples')) is not int
                or row['examples']!=batch or type(row.get('example_cursor')) is not int or row['example_cursor']!=cursor
                or any(type(row.get(k)) not in (int,float) or not math.isfinite(row[k]) for k in ('before','after'))
                or type(row.get('gradients')) is not list or not row['gradients']
                or any(type(g.get('l2')) not in (int,float) or not math.isfinite(g['l2']) or g['l2']<0 for g in row['gradients'])):
            raise ValueError('Original native batch/cursor/finite-gradient evidence differs')
    if type(receipt.get('example_cursor')) is not int or receipt['example_cursor']!=cursor:
        raise ValueError('Original native completed data cursor differs')
    if kind!='sgd':
        cp=receipt.get('optimizer_checkpoint',{})
        if (type(cp) is not dict or cp.get('path')!=request['optimizer_state_output'] or cp.get('candidate_sha256')!=sha(request['candidate'])
                or cp.get('dataset_sha256')!=optimizer_dataset_sha256(request['examples'])):
            raise ValueError('Original optimizer output path/weights/dataset differs')
        state=read_optimizer_state(cp['path'],cp['sha256']);validate_state_payload(cp['path'],state)
        if state['config']!=optimizer_configuration(request):raise ValueError('Original optimizer configuration differs from requested recipe')
        previous=None
        if request['optimizer_state_mode']=='resume':
            previous=read_optimizer_state(request['initial_optimizer_state'],request['initial_optimizer_state_sha256'])
            validate_state_payload(request['initial_optimizer_state'],previous)
            if previous['config']!=optimizer_configuration(request):raise ValueError('Original resumed optimizer recipe differs')
            identity=request.get('initial_adapter_sha256',request['base_sha256'])
            if (previous['model_sha256']!=identity or previous['dataset_sha256']!=cp['dataset_sha256']
                    or previous['example_cursor']!=request['example_cursor']):raise ValueError('Original resumed state/parent/dataset differs')
        expected_steps=(0 if previous is None else previous['completed_steps'])+count
        if (state['completed_steps']!=expected_steps or cp.get('completed_steps')!=expected_steps
                or state['example_cursor']!=cursor or cp.get('example_cursor')!=cursor
                or state['model_sha256']!=cp['candidate_sha256'] or state['dataset_sha256']!=cp['dataset_sha256']
                or state['config']!=cp.get('config')):
            raise ValueError('Original optimizer counters/configuration differ')
    return rows[-1]


def reload_request(request,receipt):
    last=validate_receipt(request,receipt)
    drop={'optimizer','optimizer_settings','optimizer_state_mode','initial_optimizer_state','initial_optimizer_state_sha256',
          'optimizer_state_output','example_cursor','training_budget_seconds','record_timing','record_process_identity',
          'steps','candidate','gradient_probe_only','logits_output','lora_update_space'}
    value={k:copy.deepcopy(v) for k,v in request.items() if k not in drop}
    start=last['example_cursor']-last['examples']
    value.update(evaluate_only=True,examples=[copy.deepcopy(request['examples'][(start+j)%len(request['examples'])])
                                             for j in range(last['examples'])])
    if request['mode']=='full':value.update(model=request['candidate'],base_sha256=sha(request['candidate']))
    else:value.update(initial_adapter=request['candidate'],initial_adapter_sha256=sha(request['candidate']))
    return value


def weight_pins(request):
    paths=[request['model'],request['candidate']]
    paths += [request[k] for k in ('initial_adapter','initial_optimizer_state','optimizer_state_output') if k in request]
    return {p:sha(p) for p in sorted(set(paths))}


def reload_gate(run,request,receipt,intent,*,executable,environment,pass_fds,timeout):
    # ONLY the opted-in normal execution path may call this forward-only hook.
    for path,expected_hash in intent['sources_sha256'].items():
        if sha(path)!=expected_hash:raise ValueError('Original reload source closure changed')
    run=Path(run);folder=run/'completion-reload';folder.mkdir(mode=0o700)
    expected=reload_request(request,receipt);pins=weight_pins(request)
    write(folder/'request-private.json',expected)
    actual_environment=runtime_values(environment)
    if actual_environment!=intent['runtime_environment']:raise ValueError('Original reload runtime environment differs')
    launch=dict(executable=str(executable),executable_sha256=sha(executable),request_sha256=sha(folder/'request-private.json'),
                intent_sha256=intent['intent_sha256'],weights_sha256=pins,runtime_environment=actual_environment)
    if intent['sources_sha256'].get(str(executable))!=launch['executable_sha256']:
        raise ValueError('Pinned reload executable changed before forward dispatch')
    with (folder/'stdout-private.json').open('x') as out,(folder/'stderr.log').open('x') as err:
        os.chmod(folder/'stdout-private.json',0o600);os.chmod(folder/'stderr.log',0o600)
        child=subprocess.Popen([str(executable),'--train',str(folder/'request-private.json')],stdout=out,stderr=err,
            env=environment,pass_fds=pass_fds,start_new_session=True)
        launch['process_identity']=process_identity(child.pid);write(folder/'launch.json',launch)
        try:code=child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            child.kill();child.wait();write(folder/'terminal-status.json',dict(status='TIMEOUT'));raise
    write(folder/'command.exit',code)
    if code!=0:raise ValueError('Original independent reload failed')
    result=read(folder/'stdout-private.json')
    gate=check_reload(request,receipt,expected,result,pins)
    for path,expected_hash in intent['sources_sha256'].items():
        if sha(path)!=expected_hash:raise ValueError('Original reload source closure changed during execution')
    gate.update(schema=1,kind='pcc.independent-save-reload.v1',intent_sha256=intent['intent_sha256'],
        request_sha256=sha(folder/'request-private.json'),stdout_sha256=sha(folder/'stdout-private.json'),
        native_stdout_sha256=sha(run/'stdout-private.json'),weights_sha256=pins,promoted=False)
    gate['gate_id']=digest(gate);write(run/'completion-gate.json',gate)
    return gate


def check_reload(request,receipt,expected,result,pins):
    if (expected!=reload_request(request,receipt) or type(result) is not dict or result.get('native') is not True
            or result.get('status')!='FORWARD_EXECUTED' or result.get('backprop_mode')!='standard'
            or result.get('promoted') is not False or type(result.get('losses')) is not list
            or len(result['losses'])!=len(expected['examples'])
            or any(type(x) not in (int,float) or not math.isfinite(x) for x in result['losses'])):
        raise ValueError('Original forward-only reload evidence differs')
    if weight_pins(request)!=pins:raise ValueError('Original reload changed model/adapter/optimizer bytes')
    error=abs(sum(result['losses'])/len(result['losses'])-receipt['steps'][-1]['after'])
    if not math.isfinite(error) or error>BOUND:raise ValueError('Original save/reload exceeds the existing numeric bound')
    return dict(passed=True,absolute_error=error,absolute_bound=BOUND,
                final_batch_start=receipt['steps'][-1]['example_cursor']-receipt['steps'][-1]['examples'],
                final_batch_examples=receipt['steps'][-1]['examples'],examples_sha256=digest(expected['examples']))


def envelope(request,receipt):
    from tools.pcc_native_training import read_optimizer_state
    value=dict(schema=1,mode=request['mode'],base_sha256=sha(request['model']),
        candidate_sha256=sha(request['candidate']),candidate_path=request['candidate'],promoted=False,
        receipt=copy.deepcopy(receipt))
    if receipt.get('optimizer','sgd')!='sgd':
        cp=receipt['optimizer_checkpoint'];state=read_optimizer_state(cp['path'],cp['sha256'])
        value.update(optimizer_state_path=cp['path'],optimizer_state_sha256=cp['sha256'],optimizer_completed_steps=state['completed_steps'])
    return value


def inspect(service,job_id,*,original_record=None):
    from tools.pcc_optimizer_profiles import select
    if service._read(service.root/'model.json')!=dict(schema=1,model_contract_id=service.model) or service.runner.completion_model!=service.model:
        raise ValueError('Original training model contract differs')
    record=service._read(service.root/'operations'/(job_id+'.json')) if original_record is None else copy.deepcopy(original_record)
    if (record.get('path')!='/cow/training/jobs' or record.get('state') not in ('interrupted','failed')
            or record['request'].get('operation_id')!=job_id
            or record['acceptance'].get('request_sha256')!=digest(['/cow/training/jobs',record['request']])):
        raise ValueError('Only an original interrupted/failed native job may be adopted')
    run=service.root/'runs'/job_id
    if run.is_symlink() or not run.is_dir():raise ValueError('Original native run is missing')
    if any((run/name).exists() for name in ('terminal-status.txt','cancelled','cancelled.json')):
        raise ValueError('Cancelled or uncertain native completion cannot be adopted')
    request=record['request'];lessons=[service._lesson(i) for i in request['lesson_ids']]
    examples=[service._example(row['lesson'],row['lesson_id']) for row in lessons]
    if examples!=[row['example'] for row in lessons]:raise ValueError('Original rendered lesson examples differ')
    parent=None
    if request['parent_job_id']:
        previous=service._read(service.root/'operations'/(request['parent_job_id']+'.json'))
        if previous.get('path')!='/cow/training/jobs' or previous.get('state')!='complete':raise ValueError('Original completed parent is missing')
        service.runner.validate_optimizer_parent(request,previous);parent=previous['candidate']
        expected_parent=service.root/'runs'/request['parent_job_id']
        for key in ('candidate_path','optimizer_state_path'):
            if key in parent and Path(parent[key]).parent.resolve()!=expected_parent.resolve():raise ValueError('Original parent artifact is outside its registered native run')
        if sha(parent['candidate_path'])!=parent['candidate_sha256']:raise ValueError('Original parent candidate bytes differ')
        if parent['mode']!=request['mode'] or request.get('expected_parent_sha256',parent['candidate_sha256'])!=parent['candidate_sha256']:
            raise ValueError('Original exact parent differs')
        if request['mode']=='lora' and previous['request']['rank']!=request['rank']:raise ValueError('Original LoRA parent rank differs')
    expected_context=context(service.runner,request,examples,parent)
    native=native_request(service.runner.native_request(request,examples,parent),run)
    intent=read(run/'completion-intent.json');original_intent=copy.deepcopy(intent);original_intent.pop('intent_sha256',None)
    if (intent.get('intent_sha256')!=digest(original_intent)
            or {k:intent.get(k) for k in expected_context}!=expected_context
            or intent.get('native_request')!=native or intent.get('native_request_sha256')!=sha(run/'request-private.json')
            or read(run/'request-private.json')!=native):raise ValueError('Original predispatch source/config/request provenance differs')
    launch=read(run/'launch.json');raw=read(run/'stdout-private.json')
    code=read(run/'command.exit')
    if (type(code) is not int or code!=0 or launch.get('request_sha256')!=sha(run/'request-private.json')
            or launch.get('executable')!=str(Path(service.runner.config['executable']).resolve())
            or launch.get('executable_sha256')!=intent['sources_sha256'].get(launch.get('executable'))
            or launch.get('runtime_environment')!=intent['runtime_environment']
            or launch.get('base_sha256')!=native['base_sha256']
            or launch.get('parent_adapter_sha256')!=native.get('initial_adapter_sha256')):
        raise ValueError('Original native launch/exit differs')
    require_gone(launch.get('process_identity'))
    if launch.get('pid')!=launch['process_identity']['pid']:raise ValueError('Original native launch PID binding differs')
    validate_receipt(native,raw)
    if sha(native['model'])!=native['base_sha256'] or ('initial_adapter' in native and sha(native['initial_adapter'])!=native['initial_adapter_sha256']):
        raise ValueError('Original input model/adapter bytes differ')
    folder=run/'completion-reload'
    code=read(folder/'command.exit')
    if (folder/'terminal-status.json').exists() or type(code) is not int or code!=0:raise ValueError('Original independent reload was not complete')
    reload_launch=read(folder/'launch.json');reload_input=read(folder/'request-private.json');reload_output=read(folder/'stdout-private.json')
    require_gone(reload_launch.get('process_identity'))
    pins=weight_pins(native)
    if (reload_launch.get('intent_sha256')!=intent['intent_sha256'] or reload_launch.get('weights_sha256')!=pins
            or reload_launch.get('executable')!=launch['executable'] or reload_launch.get('executable_sha256')!=launch['executable_sha256']
            or reload_launch.get('runtime_environment')!=intent['runtime_environment']
            or reload_launch.get('request_sha256')!=sha(folder/'request-private.json')):raise ValueError('Original reload launch provenance differs')
    measured=check_reload(native,raw,reload_input,reload_output,pins)
    gate=read(run/'completion-gate.json');payload=copy.deepcopy(gate);payload.pop('gate_id',None)
    expected_gate=dict(measured,schema=1,kind='pcc.independent-save-reload.v1',intent_sha256=intent['intent_sha256'],
        request_sha256=sha(folder/'request-private.json'),stdout_sha256=sha(folder/'stdout-private.json'),
        native_stdout_sha256=sha(run/'stdout-private.json'),weights_sha256=pins,promoted=False)
    if payload!=expected_gate or gate.get('gate_id')!=digest(payload):raise ValueError('Original independent reload gate differs')
    receipt=copy.deepcopy(raw);receipt.update(request_sha256=launch['request_sha256'],parent_adapter_sha256=launch['parent_adapter_sha256'])
    candidate=envelope(native,receipt);version=run/'versions'/(candidate['candidate_sha256']+'.json')
    if version.exists() and read_version(version)!=candidate:raise ValueError('Conflicting original native candidate version')
    profile=select(service.runner.registered_optimizer_profiles,request)
    if profile:candidate.update(optimizer_profile_id=request['optimizer_profile_id'],optimizer_state_policy=profile['state_policy'])
    candidate['completion_recovery']=dict(intent_sha256=intent['intent_sha256'],gate_id=gate['gate_id'])
    files=['request-private.json','launch.json','stdout-private.json','command.exit','completion-intent.json','completion-gate.json',
           'completion-reload/request-private.json','completion-reload/launch.json','completion-reload/stdout-private.json','completion-reload/command.exit']
    evidence=dict(files_sha256={name:sha(run/name) for name in files},weights_sha256=pins,source_sha256=intent['sources_sha256'])
    return dict(schema=1,job_id=job_id,job_record_sha256=digest(record),evidence_sha256=digest(evidence),
        candidate_sha256=candidate['candidate_sha256'],candidate=candidate,evidence=evidence,
        native_configuration_sha256=digest(intent['runner_config']),
        source_closure_sha256=digest(intent['sources_sha256']),
        runtime_environment_sha256=digest(intent['runtime_environment']),
        model_contract_id=service.model,model_calls_started=0,native_jobs_started=0,promoted=False)


@contextmanager
def existing_lock(service,path):
    # Existing LessonService locks may be0644 inside its0700 private store.
    # Their empty contents are not secret; do not mutate legacy modes or create
    # a replacement lock inode while a native child might inherit the original.
    root=service.root;parent=root.lstat()
    if root.is_symlink() or not stat.S_ISDIR(parent.st_mode) or parent.st_uid!=os.getuid() or parent.st_mode&0o077:
        raise ValueError('Original private training store differs')
    try:
        before=path.lstat()
        if (not stat.S_ISREG(before.st_mode) or before.st_uid!=os.getuid() or before.st_nlink!=1 or before.st_mode&0o7000):
            raise ValueError('Original owned regular record/worker lock is missing')
        fd=os.open(path,os.O_RDWR|os.O_NOFOLLOW|os.O_CLOEXEC)
    except FileNotFoundError:raise ValueError('Original private record/worker lock is missing') from None
    with os.fdopen(fd,'r+b') as stream:
        after=os.fstat(stream.fileno())
        if (before.st_dev,before.st_ino)!=(after.st_dev,after.st_ino):raise ValueError('Original lock identity changed')
        yield stream

@contextmanager
def worker_idle(service):
    """Caller holds record admission; never create a missing historical lock."""
    path=service.root/'worker.lock'
    for p in (service.root/'operations').glob('*.json'):
        if service._read(p).get('state') in ('queued','running'):
            raise RecoveryHold('training_boundary_busy','Original or competing work is still admitted')
    with existing_lock(service,path) as lease:
        try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RecoveryHold('training_boundary_busy','Original inherited worker is still present') from None
        yield

@contextmanager
def idle(service):
    # Existing admission and worker locks only, both nonblocking.
    path=service.root/'records.lock'
    if not service.mutex.acquire(blocking=False):raise RecoveryHold('training_boundary_busy','Original record admission is busy')
    try:
        with existing_lock(service,path) as records:
            try:fcntl.flock(records,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise RecoveryHold('training_boundary_busy','Original record admission is busy') from None
            with worker_idle(service):yield
    finally:service.mutex.release()


def approval_request(request):
    fields={'operation_id','job_id','expected_job_record_sha256','expected_evidence_sha256','expected_candidate_sha256','approval'}
    if (type(request) is not dict or set(request)!=fields or request['approval']!=APPROVAL
            or any(type(request[k]) is not str or not HEX.fullmatch(request[k]) for k in fields if k.startswith('expected_'))):
        raise ValueError('Explicit owner approval and exact original completion CAS hashes required')
    from tools.pcc_lessons import OP
    if any(type(request[k]) is not str or not OP.fullmatch(request[k]) for k in ('operation_id','job_id')):
        raise ValueError('Registered original job and adoption operation IDs required')

def acceptance_for(request):
    return dict(schema=1,operation_id=request['operation_id'],request_sha256=digest([ROUTE,request]),job_id=request['job_id'],
                candidate_sha256=request['expected_candidate_sha256'],adopted=True,promoted=False)

def pending_adoption(service,job_id):
    pending=[]
    for p in (service.root/'operations').glob('*.json'):
        row=service._read(p)
        if row.get('path')==ROUTE and row.get('request',{}).get('job_id')==job_id and row.get('state')!='complete':
            pending.append(row)
    if len(pending)>1:raise RecoveryHold('adoption_transaction_conflict','Multiple original adoption intents need operator inspection')
    return pending[0] if pending else None

def transaction_value(service,operation,*,require_complete):
    """Revalidate saved bytes AND current metadata. Caller owns record/worker boundary."""
    request=operation['request'];approval_request(request)
    if (operation.get('path')!=ROUTE or operation.get('state') not in ('recorded','complete')
            or operation.get('acceptance')!=acceptance_for(request)
            or (require_complete and operation['state']!='complete')):
        raise ValueError('Original completion adoption is not a complete exact transaction')
    original=operation['original_job_record']
    value=inspect(service,request['job_id'],original_record=original)
    if (any(value[key]!=request['expected_'+key] for key in ('job_record_sha256','evidence_sha256','candidate_sha256'))
            or operation.get('validated_evidence_sha256')!=value['evidence_sha256']):
        raise ValueError('Original adopted completion evidence changed')
    current=service._read(service.root/'operations'/(request['job_id']+'.json'))
    adoption=dict(operation_id=request['operation_id'],original_job_record_sha256=digest(original),
                  evidence_sha256=value['evidence_sha256'],approval=APPROVAL,promoted=False)
    original_content={k:v for k,v in original.items() if k!='record_sha256'}
    current_content={k:v for k,v in current.items() if k!='record_sha256'}
    expected=dict(original_content,state='complete',candidate=value['candidate'],completion_adoption=adoption)
    if current_content!=expected:
        if require_complete or operation['state']=='complete' or current_content!=original_content:
            raise ValueError('Current adopted candidate/job differs from exact original completion')
    if current_content==expected:
        version=service.root/'runs'/request['job_id']/'versions'/(value['candidate_sha256']+'.json')
        if not version.exists():raise ValueError('Original adopted candidate version is missing')
    return value

def adopted_value(service,job_id):
    job=service._read(service.root/'operations'/(job_id+'.json'))
    adoption=job.get('completion_adoption')
    if job.get('state')!='complete' or type(adoption) is not dict:
        raise ValueError('Original explicit completion adoption is missing')
    from tools.pcc_lessons import OP
    if type(adoption.get('operation_id')) is not str or not OP.fullmatch(adoption['operation_id']):
        raise ValueError('Original adoption operation requires a registered ID')
    operation=service._read(service.root/'operations'/(adoption['operation_id']+'.json'))
    if operation.get('request',{}).get('job_id')!=job_id:raise ValueError('Original adoption belongs to a different native job')
    return transaction_value(service,operation,require_complete=True)

def public_inspection(service,job_id):
    result=dict(schema=1,kind='pcc.native-completion-recovery.v1',job_id=job_id,model_contract_id=service.model,
                state='ineligible',reason_code='original_evidence_invalid',next_actions=['hold'],eligible=False,
                current_evidence_verified=False,message='',model_calls_started=0,native_jobs_started=0,promoted=False)
    config=getattr(service.runner,'config',{})
    base=copy.deepcopy(result)
    try:
        with idle(service):
            job=service._read(service.root/'operations'/(job_id+'.json'))
            if job.get('state')=='complete' and 'completion_adoption' not in job:
                return dict(result,reason_code='original_job_already_complete',next_actions=['inspect_original_job','learning_status'],
                            message='Original service job is already complete; no completion adoption is needed')
            if type(config) is not dict or config.get('completion_recovery') is None:
                return dict(result,reason_code='legacy_provenance_missing',message='Original opt-in provenance is absent; no retrospective adoption')
            pending=pending_adoption(service,job_id)
            if pending is not None:
                value=transaction_value(service,pending,require_complete=False)
                result.update(state='adoption_pending',reason_code='original_adoption_pending',next_actions=['resume_original_adoption'],
                              pending_operation_id=pending['request']['operation_id'],pending_request=copy.deepcopy(pending['request']))
            else:
                job=service._read(service.root/'operations'/(job_id+'.json'))
                if job.get('state')=='complete':
                    value=adopted_value(service,job_id)
                    result.update(state='adopted',reason_code='original_adoption_verified',next_actions=['reconcile_learning_run'],
                                  adoption_operation_id=job['completion_adoption']['operation_id'])
                else:
                    value=inspect(service,job_id)
                    result.update(state='eligible',reason_code='original_completion_verified',next_actions=['adopt_original_completion'],eligible=True)
            result.update(current_evidence_verified=True)
            for key in ('job_record_sha256','evidence_sha256','candidate_sha256','native_configuration_sha256','source_closure_sha256','runtime_environment_sha256'):
                result[key]=value[key]
    except RecoveryHold as error:
        result=dict(base,state='ineligible' if error.code=='adoption_transaction_conflict' else 'unavailable',reason_code=error.code,message=str(error)[:512])
    except FileNotFoundError as error:
        result=dict(base,state='unavailable',reason_code='original_evidence_incomplete',message=str(error)[:512])
    except (ValueError,KeyError,TypeError,OSError) as error:
        result=dict(base,state='ineligible',reason_code='original_evidence_invalid',message=str(error)[:512])
    return result


def adopt(service,request):
    approval_request(request)
    target=service.root/'operations'/(request['operation_id']+'.json');job_path=service.root/'operations'/(request['job_id']+'.json')
    binding=digest([ROUTE,request])
    with idle(service):
        prior=service._read(target) if target.exists() else None
        pending=pending_adoption(service,request['job_id'])
        if pending is not None and pending['request']['operation_id']!=request['operation_id']:
            raise ValueError('Resume the original pending adoption operation; do not create a new approval')
        if prior is not None:
            if prior.get('path')!=ROUTE or prior.get('request')!=request or prior['acceptance']['request_sha256']!=binding:
                raise ValueError('Adoption operation ID binds different original approval bytes')
            if prior['state']=='complete':
                if prior['acceptance']!=acceptance_for(request):raise ValueError('Original adoption acknowledgment differs')
                # Historical operation acknowledgment, not current artifact eligibility.
                return copy.deepcopy(prior['acceptance'])
            original=prior['original_job_record']
        else:original=service._read(job_path)
        value=inspect(service,request['job_id'],original_record=original)
        if any(value[key]!=request['expected_'+key] for key in ('job_record_sha256','evidence_sha256','candidate_sha256')):
            raise ValueError('Original job/evidence/candidate changed after owner inspection')
        acceptance=acceptance_for(request)
        if prior is None:
            prior=dict(path=ROUTE,request=copy.deepcopy(request),acceptance=acceptance,state='recorded',
                       original_job_record=original,validated_evidence_sha256=value['evidence_sha256'])
            service._write(target,prior)
        current=service._read(job_path)
        adoption=dict(operation_id=request['operation_id'],original_job_record_sha256=digest(original),
                      evidence_sha256=value['evidence_sha256'],approval=APPROVAL,promoted=False)
        original_content={k:v for k,v in original.items() if k!='record_sha256'}
        current_content={k:v for k,v in current.items() if k!='record_sha256'}
        expected=dict(original_content,state='complete',candidate=value['candidate'],completion_adoption=adoption)
        if current_content!=original_content and current_content!=expected:raise ValueError('Original job changed during completion adoption')
        # Version publication is the same immutable native candidate, not model work.
        version_candidate={k:v for k,v in value['candidate'].items() if k not in ('completion_recovery','optimizer_profile_id','optimizer_state_policy')}
        version=service.root/'runs'/request['job_id']/'versions'/(value['candidate_sha256']+'.json')
        version.parent.mkdir(mode=0o700,exist_ok=True)
        if version.exists():
            if read_version(version)!=version_candidate:raise ValueError('Conflicting original candidate version')
        else:
            from tools.pcc_native_training import finalize_candidate
            finalized=finalize_candidate(version.parent,read(service.root/'runs'/request['job_id']/'request-private.json')['model'],
                value['candidate']['candidate_path'],value['candidate']['mode'],value['candidate']['receipt'])
            if finalized!=version_candidate:raise ValueError('Reconstructed original candidate differs')
        # Recheck original bytes/source immediately before publishing service state.
        fresh=inspect(service,request['job_id'],original_record=original)
        if fresh['evidence_sha256']!=value['evidence_sha256'] or fresh['candidate']!=value['candidate']:
            raise ValueError('Original completion changed during metadata adoption')
        if current_content==original_content:service._write(job_path,expected)
        prior['state']='complete';service._write(target,prior)
        return acceptance
