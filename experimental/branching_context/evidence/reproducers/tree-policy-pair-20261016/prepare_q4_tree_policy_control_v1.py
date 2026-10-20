"""Old policy at the candidate's exact execution/resource settings; no inference."""
import copy,hashlib,json,os,shutil,sys
from pathlib import Path
parent=Path('/home/dflanag3/spatial-artifacts/q4-tree-retention-20261016-v1')
root=Path('/home/dflanag3/spatial-artifacts/q4-tree-policy-control-20261016-v1')
base_runtime=parent/'gpu-candidate-v1/runtime';base_path=parent/'gpu-candidate-v1/profile.json'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    with path.open('x') as stream:json.dump(value,stream,sort_keys=True,indent=2);stream.write('\n')
assert sha(base_path)=='107cab965436707d13db92271aa629e7829950030dbe60b884308076f8e31253'
original=Path('/home/dflanag3/spatial-artifacts/q4-terminal-error-profile-20261016-v3/runtime/experimental/branching_context/src/context_time_travel/coordinator.py')
assert sha(original)=='bb6a1f8d9582c35dfaec69391658a033320e25d717b61ebfb9b0bec1f6c6c32b'
base=json.loads(base_path.read_bytes());profile=copy.deepcopy(base)
for key in list(os.environ):
    if key.startswith(('STRATA_','CUDA_','NVIDIA_','GGML_')):del os.environ[key]
os.environ.update(base['execution']['environment'])
sys.path[:0]=[str(base_runtime),str(base_runtime/'tools'),str(base_runtime/'experimental/branching_context/src')]
from context_time_travel.strata_backend import _digest,_validate_execution,_validate_model
_validate_execution(base);model=_validate_model(base)
root.mkdir(mode=0o700);runtime=root/'runtime';shutil.copytree(base_runtime,runtime)
target=runtime/'experimental/branching_context/src/context_time_travel/coordinator.py'
target.chmod(0o600);shutil.copyfile(original,target);target.chmod(0o400)
def remap(name):
    path=Path(name)
    return str(runtime/path.relative_to(base_runtime)) if path.is_relative_to(base_runtime) else name
pins={}
for name,expected in base['native_build']['sha256'].items():
    mapped=remap(name);actual=sha(mapped)
    assert actual==(sha(original) if Path(mapped)==target else expected),(name,mapped)
    pins[mapped]=actual
profile['native_build']['sha256']=pins
closure=profile['native_build']['selected_runtime_source_closure']
closure['files_sha256']={name:(sha(original) if name.replace('\\','/').endswith('context_time_travel/coordinator.py') else expected)
                        for name,expected in closure['files_sha256'].items()}
closure['source_commit']='7c97c1d70889a5830eec9b6eb9f29055499863f4'
closure['source_status']='Old-policy source control at exactly the candidate execution settings; file pins authoritative'
profile['gguf_reader']=remap(profile['gguf_reader'])
assert profile['execution']==base['execution'] and profile['hardware']==base['hardware']
assert profile['execution']['checkpoint_cache']==dict(byte_limit=8*1024**3,entry_limit=8)
profile['execution_profile_id']=_digest(dict(build=profile['native_build'],execution=profile['execution'],hardware=profile['hardware']))
profile['numerical_gate_qualified']=False;profile['calibration_status']='NOT_RUN'
_validate_execution(profile);_validate_model(profile)
path=root/'profile.json';write(path,profile);path.chmod(0o400)
fixtures=json.loads((parent/'gpu-candidate-v1/fixture-set.json').read_bytes())
for item in fixtures['contexts']:assert sha(item['fixture'])==item['fixture_sha256']
for item in fixtures['imports'].values():
    item['path']=remap(item['path']);assert sha(item['path'])==item['sha256']
fixtures.update(profile_path=str(path),profile_sha256=sha(path),execution_profile_id=profile['execution_profile_id'],
    builder_sha256=sha(__file__),status='PREPARED_NOT_RUN',
    source_scope='Exact existing text/token fixtures reused as inputs; fresh old-policy control store; no native checkpoints imported')
write(root/'fixture-set.json',fixtures)
write(root/'admission.json',dict(status='PREPARED_NOT_GPU_QUALIFIED',profile_sha256=sha(path),execution_profile_id=profile['execution_profile_id'],
    compared_candidate_profile_sha256=sha(base_path),execution_settings_identical=True,hardware_identical=True,
    changed_production_file=str(target),changed_production_sha256=sha(target),model_verification=model,
    source_commit=closure['source_commit'],producer_sha256=sha(__file__),old_numerical_checkpoints_imported=0))
assert not any(p.stat().st_mode&0o222 for p in runtime.rglob('*'))
print(json.dumps(dict(status='PREPARED_NOT_GPU_QUALIFIED',profile_sha256=sha(path),execution_profile_id=profile['execution_profile_id'],model=model)))
