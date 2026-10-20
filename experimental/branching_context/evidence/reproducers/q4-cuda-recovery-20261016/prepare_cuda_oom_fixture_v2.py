"""Fresh profile-bound Q4 active checkpoint and independent numerical reference.

This artifact harness never retags an old checkpoint. References use the
separate uncached oracle executable, with normal generated-token operators.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
import time

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('mode',choices=['prepare','reference'])
p.add_argument('--profile',type=Path,required=True)
p.add_argument('--runtime',type=Path,required=True)
p.add_argument('--fixture',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--prepared',type=Path)
p.add_argument('--expected-profile-sha256',required=True)
p.add_argument('--preflight-only',action='store_true')
a=p.parse_args()
profile=json.loads(a.profile.read_bytes())
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while b:=f.read(8*1024**2):h.update(b)
    return h.hexdigest()
assert sha(a.profile)==a.expected_profile_sha256
runtime=a.runtime.resolve(strict=True)
api=runtime/'experimental/branching_context'
sys.path[:0]=[str(api/'src'),str(api/'scripts'),str(runtime)]
from context_time_travel.strata_backend import StrataBackend,_validate_execution,_validate_model
from context_time_travel.real import BranchingModel
from context_time_travel import controller
from native_request_probe import _record_native_shutdown
from run_root_reuse_probe import reference,expected_controller
from tools.strata_tokenizer import Tokenizer
from run_cow_client_recovery import ProgressTracker
for key in list(os.environ):
    if key.startswith(('STRATA_','CUDA_','NVIDIA_','GGML_')): del os.environ[key]
os.environ.update(profile['execution']['environment'])
_validate_execution(profile)
model_verification=_validate_model(profile)
fixture=json.loads(a.fixture.read_bytes())
ids=fixture['base_tokens']
assert ids and all(type(t) is int and 0<=t<248320 for t in ids)
assert fixture['generated_prefix']==4 and fixture['followup_outputs']==8
assert profile['execution']['prefill_chunk']==8192 and profile['execution']['capacity']==98304
assert profile['execution']['generated_operator']=='normal-verifier-decode'
imports={}
for name,module in tuple(sys.modules.items()):
    if name=='context_time_travel' or name.startswith('context_time_travel.'):
        path=Path(module.__file__).resolve(strict=True)
        assert path.is_relative_to(api/'src'),('unexpected API import',name,str(path))
        actual=sha(path)
        assert profile['native_build']['sha256'].get(str(path))==actual,('unbound API source',name)
        imports[name]=dict(path=str(path),sha256=actual)
if a.preflight_only:
    if a.mode=='reference':
        Tokenizer.from_gguf(profile['model_path'])
    assert not a.output.exists(),'preflight output already exists'
    print(json.dumps(dict(status='PREPARED_NOT_RUN',mode=a.mode,profile_sha256=sha(a.profile),
        fixture_sha256=sha(a.fixture),base_tokens=len(ids),imports=imports),sort_keys=True),flush=True)
    raise SystemExit(0)
a.output.mkdir(mode=0o700,parents=True,exist_ok=False)
record=dict(schema='cuda-oom-fresh-fixture-v1',mode=a.mode,status='RUNNING',profile=str(a.profile),profile_sha256=sha(a.profile),
    execution_profile_id=profile['execution_profile_id'],model_contract_id=profile['model_contract_id'],
    fixture_sha256=sha(a.fixture),harness_sha256=sha(Path(__file__)),base_tokens=len(ids),producer_python_pid=os.getpid(),
    model_verification=model_verification,stream=fixture['stream'],numerical_max_error=None,comparisons=0,imports=imports)
started=time.monotonic()
def publish():
    temp=a.output/'result.json.tmp'
    with temp.open('w') as f:
        json.dump(record,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temp,a.output/'result.json')
def pack(values):return struct.pack('<'+'f'*len(values),*values)
def compare(left,right,name):
    assert len(left)==len(right)==4*248320,name+': incomplete frontier'
    la,lb=struct.unpack('<248320f',left),struct.unpack('<248320f',right)
    assert all(math.isfinite(x) for x in la+lb),name+': nonfinite'
    maximum=max(abs(x-y) for x,y in zip(la,lb))
    record['numerical_max_error']=max(record['numerical_max_error'] or 0,maximum)
    if left!=right:
        (a.output/(name+'.actual.f32')).write_bytes(left)
        (a.output/(name+'.expected.f32')).write_bytes(right)
        raise AssertionError(name+': mismatch max='+str(maximum))
    record['comparisons']+=1
publish()
backend=None
progress=ProgressTracker(a.output,'cuda-oom-fresh-'+a.mode).start()
try:
    if a.mode=='prepare':
        progress.phase('fresh_native_startup')
        backend=StrataBackend(profile)
        progress.native_started(backend.process.pid)
        progress.phase('fast_batched_prefix_and_four_generated')
        record['native_pid']=backend.process.pid
        record['startup_seconds']=time.monotonic()-started
        publish()
        model=BranchingModel(a.output/'store',backend,profile)
        before=backend.work_counters()
        root=model.create(ids)
        (a.output/'base.f32').write_bytes(pack(model.cache[root].frontier))
        after_prefill=backend.work_counters()
        answer=model.run_greedy_request(root,(),4,stream=fixture['stream'])
        sid=answer['snapshot']
        assert len(answer['tokens'])==4,'fixture ended before four consumed outputs'
        assert model.controller(sid)['status']=='active','fixture not an active continuation'
        assert model.history(sid)==tuple(ids+answer['tokens'])
        (a.output/'prefix.f32').write_bytes(pack(model.cache[sid].frontier))
        (a.output/'prefix-controller.json').write_text(json.dumps(model.controller(sid),sort_keys=True,separators=(',',':'))+'\n')
        checkpoint=model.checkpoint(sid)
        head=model.store.fork('CliA',sid)
        after=backend.work_counters()
        assert after_prefill['forward_tokens']-before['forward_tokens']==len(ids)
        assert after_prefill['prefill_tokens']-before['prefill_tokens']==len(ids)
        assert after['forward_tokens']-after_prefill['forward_tokens']==4
        assert after['prefill_tokens']==after_prefill['prefill_tokens']
        manifest=json.loads(checkpoint['manifest_path'].read_bytes())['manifest']
        assert manifest['execution_profile_id']==profile['execution_profile_id'] and manifest['consumed']==len(ids)+4
        record.update(status='PASS',source_snapshot=sid,source_branch=asdict(head),generated_prefix=answer['tokens'],
            consumed=len(ids)+4,checkpoint_manifest_sha256=sha(checkpoint['manifest_path']),
            checkpoint_payload_sha256=sha(checkpoint['payload_path']),checkpoint_payload_bytes=checkpoint['payload_path'].stat().st_size,
            work_before=before,work_after=after,execution_schedule=manifest['execution_schedule'],
            base_frontier_sha256=sha(a.output/'base.f32'),prefix_frontier_sha256=sha(a.output/'prefix.f32'))
        model.cache.clear()
    else:
        assert a.prepared is not None
        prepared=json.loads((a.prepared/'result.json').read_bytes())
        assert prepared['status']=='PASS' and prepared['profile_sha256']==sha(a.profile)
        assert prepared['fixture_sha256']==sha(a.fixture)
        assert not Path('/proc',str(prepared['native_pid'])).exists(),'producer native still running'
        tokenizer=Tokenizer.from_gguf(profile['model_path'])
        references=[]
        for name in ('fresh','fresh-repeat'):
            progress.phase('independent_'+name)
            ref=reference(profile,a.output/name,name,ids,[len(ids)],12)
            assert len(ref['tokens'])==12,'fresh reference stopped before bounded output fixture'
            assert ref['tokens'][:4]==prepared['generated_prefix'],'prepared token IDs differ from fresh reference'
            assert not Path('/proc',str(ref['pid'])).exists()
            compare((a.prepared/'base.f32').read_bytes(),ref['prompt_frontiers'][0],name+'-base')
            compare((a.prepared/'prefix.f32').read_bytes(),ref['generated_frontiers'][3],name+'-prefix4')
            references.append(ref)
            record['references']=[{k:v for k,v in r.items() if k not in ('generated_frontiers','prompt_frontiers')} for r in references]
            publish()
        assert references[0]['tokens']==references[1]['tokens'],'fresh repeat tokens differ'
        for index,(left,right) in enumerate(zip(references[0]['generated_frontiers'],references[1]['generated_frontiers']),1):
            compare(left,right,'fresh-repeat-'+str(index))
        expected_prefix=expected_controller(fixture['stream'],ids,references[0]['tokens'][:4],tokenizer,profile['execution']['eos_tokens'],248320)
        assert expected_prefix==json.loads((a.prepared/'prefix-controller.json').read_bytes()),'complete prefix controller mismatch'
        expected_final=expected_controller(fixture['stream'],ids,references[0]['tokens'],tokenizer,profile['execution']['eos_tokens'],248320)
        (a.output/'expected-final-controller.json').write_text(json.dumps(expected_final,sort_keys=True,separators=(',',':'))+'\n')
        (a.output/'expected-final.f32').write_bytes(references[0]['generated_frontiers'][11])
        (a.output/'expected-prefix.f32').write_bytes(references[0]['generated_frontiers'][3])
        record.update(status='PASS',source_snapshot=prepared['source_snapshot'],expected_followup_tokens=references[0]['tokens'][4:],
            expected_final_frontier_sha256=sha(a.output/'expected-final.f32'),expected_final_controller_sha256=sha(a.output/'expected-final-controller.json'),
            prefix_controller_exact=True,reference_checkpoint_imports=0,
            boundary_scope='Native operators identical; independent oracle generates 12 uninterrupted versus producer4 plus resumed8. No arbitrary batch-arrangement equivalence claim.')
except BaseException as error:
    record.update(status='FAIL',error=dict(type=type(error).__name__,message=str(error)))
    raise
finally:
    if backend is not None:
        _record_native_shutdown(record,{0:backend.process},backend.close)
        record['native_pid_absent']=not Path('/proc',str(backend.process.pid)).exists()
    record['elapsed_seconds']=time.monotonic()-started
    try:
        progress.finish(record['status'],record.get('error'))
        progress.close()
    except BaseException as cleanup_error:
        record.update(status='FAIL',heartbeat_cleanup_error=str(cleanup_error))
    publish()
if record['status']!='PASS':raise SystemExit(1)
print(json.dumps({k:v for k,v in record.items() if k not in ('references','execution_schedule','work_before','work_after')},sort_keys=True),flush=True)
