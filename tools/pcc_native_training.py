"""Offline PCC native-training admission and immutable candidate registration.

No BW CLI/HTTP hook and no PyTorch training. A training candidate is never an
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

def validate_examples(examples):
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
        n=len(e['tokens'])
        if not 2<=n<=8192 or len(e['targets'])!=n or len(e['loss_mask'])!=n:
            raise ValueError('Bounded aligned inputs/next-token targets/masks required')
        if any(type(t) is not int or t<0 for t in e['tokens']+e['targets']):
            raise ValueError('Nonnegative integer token IDs required')
        if any(type(m) is not bool for m in e['loss_mask']) or not any(e['loss_mask']):
            raise ValueError('Nonempty explicit completion-loss mask required')
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

def execute_native(executable,request,run_directory,*,timeout=3600):
    """Run one explicit native update request and preserve its exact receipts.

    Run-directory identity prevents an uncertain completion from turning into
    an automatic second update. Reconciliation reads that directory; it never
    blindly launches the executable again. This helper does not promote output.
    """
    request=copy.deepcopy(request)
    request['devices']=validate_devices(request['devices'])
    request['examples']=validate_examples(request['examples'])
    request['backprop_mode']=validate_backprop_mode(request.get('backprop_mode'),devices=len(request['devices']))
    if request.get('mode') not in ('lora','full'):
        raise ValueError('Explicit LoRA or direct full-weight update required')
    if request.get('optimizer','sgd')!='sgd':
        raise ValueError('Only native SGD is currently implemented')
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
    executable=Path(executable).resolve(strict=True)
    run_directory=Path(run_directory).resolve()
    run_directory.mkdir(mode=0o700,parents=True,exist_ok=False)
    request['model']=str(base)
    request['candidate']=str(run_directory/'candidate.gguf')
    config=run_directory/'request-private.json'
    config.write_text(json.dumps(request,sort_keys=True,indent=2)+'\n',encoding='utf-8',newline='\n')
    config.chmod(0o600)
    launch=dict(executable=str(executable),executable_sha256=sha256(executable),
                request_sha256=sha256(config),base_sha256=original,parent_adapter_sha256=parent_adapter)
    (run_directory/'launch.json').write_text(json.dumps(launch)+'\n')
    try:
        with (run_directory/'stdout-private.json').open('x',encoding='utf-8') as out, (run_directory/'stderr.log').open('x',encoding='utf-8') as err:
            proc=subprocess.Popen([str(executable),'--train',str(config)],stdout=out,stderr=err,
                                  start_new_session=(os.name=='posix'))
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
        receipt.update(request_sha256=launch['request_sha256'],parent_adapter_sha256=parent_adapter)
        return finalize_candidate(run_directory/'versions',base,request['candidate'],request['mode'],receipt)
    except subprocess.TimeoutExpired:
        (run_directory/'terminal-status.txt').write_text('TIMEOUT: completion unknown; inspect original owned process and run before reconciliation\n')
        raise
