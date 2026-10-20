"""Derive a fresh candidate; reuse token fixtures, never numerical checkpoints."""
import copy,hashlib,json,os,shutil,sys
from pathlib import Path

root=Path('/home/dflanag3/spatial-artifacts/q4-tree-retention-20261016-v1')
base_runtime=Path('/home/dflanag3/spatial-artifacts/q4-terminal-error-profile-20261016-v3/runtime')
base_path=base_runtime.parent/'profile/profile.json'
old_fixtures=Path('/home/dflanag3/spatial-artifacts/q4-cli-trees-20261016-v1/fixtures-v1/fixture-set.json')
candidate=root/'gpu-candidate-v1';runtime=candidate/'runtime'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    with path.open('x') as stream:json.dump(value,stream,sort_keys=True,indent=2);stream.write('\n')
assert sha(base_path)=='a7d53c45ebd7a1aa841ee10a7f5ef55472d5c761be928551a176b94f7d3f1755'
coordinator=root/'green-stage-v4/source/experimental/branching_context/src/context_time_travel/coordinator.py'
assert sha(coordinator)=='7042b31f1f4bb23580cd21213052ad27e8ab861b011dca98a95ede9db2602f62'
base=json.loads(base_path.read_bytes());profile=copy.deepcopy(base)
for key in list(os.environ):
    if key.startswith(('STRATA_','CUDA_','NVIDIA_','GGML_')):del os.environ[key]
os.environ.update(base['execution']['environment'])
sys.path[:0]=[str(base_runtime),str(base_runtime/'tools'),str(base_runtime/'experimental/branching_context/src')]
from context_time_travel.strata_backend import _digest,_validate_execution,_validate_model
_validate_execution(base);model=_validate_model(base)
candidate.mkdir(mode=0o700)
shutil.copytree(base_runtime,runtime)
target=runtime/'experimental/branching_context/src/context_time_travel/coordinator.py'
target.chmod(0o600);shutil.copyfile(coordinator,target);target.chmod(0o400)
def remap(name):
    path=Path(name)
    return str(runtime/path.relative_to(base_runtime)) if path.is_relative_to(base_runtime) else name
pins={}
for name,expected in base['native_build']['sha256'].items():
    mapped=remap(name);actual=sha(mapped)
    if Path(mapped)==target:assert actual==sha(coordinator)
    else:assert actual==expected,(name,mapped)
    pins[mapped]=actual
profile['native_build']['sha256']=pins
closure=profile['native_build']['selected_runtime_source_closure']
closure['files_sha256']={name:(sha(coordinator) if name.replace('\\','/').endswith('context_time_travel/coordinator.py') else expected)
                        for name,expected in closure['files_sha256'].items()}
assert sha(coordinator) in closure['files_sha256'].values()
closure['source_commit']='eff02e29af5556305a8acca75d41e01d3ca276c9'
closure['source_status']='Qualified runtime plus the single coordinator overlay from the isolated committed candidate; exact file pins authoritative'
for field in ('gguf_reader',):profile[field]=remap(profile[field])
assert base['execution']['checkpoint_cache']==dict(byte_limit=8*1024**3,entry_limit=3)
profile['execution']['checkpoint_cache']['entry_limit']=8
profile['execution_profile_id']=_digest(dict(build=profile['native_build'],execution=profile['execution'],hardware=profile['hardware']))
profile['numerical_gate_qualified']=False
profile['calibration_status']='NOT_RUN'
assert profile['model_contract_id']==base['model_contract_id']
_validate_execution(profile);_validate_model(profile)
path=candidate/'profile.json';write(path,profile);path.chmod(0o400)
fixtures=json.loads(old_fixtures.read_bytes())
for item in fixtures['contexts']:assert sha(item['fixture'])==item['fixture_sha256']
for item in fixtures['imports'].values():
    item['path']=remap(item['path']);assert sha(item['path'])==item['sha256']
fixtures.update(profile_path=str(path),profile_sha256=sha(path),execution_profile_id=profile['execution_profile_id'],
    builder_sha256=sha(__file__),status='PREPARED_NOT_RUN',
    source_scope='Previously verified exact text/token fixtures reused only as inputs; fresh store and no native checkpoint import',
    source_fixture_set=str(old_fixtures),source_fixture_set_sha256=sha(old_fixtures))
write(candidate/'fixture-set.json',fixtures)
write(candidate/'admission.json',dict(status='PREPARED_NOT_GPU_QUALIFIED',base_profile_sha256=sha(base_path),
    profile_sha256=sha(path),execution_profile_id=profile['execution_profile_id'],model_contract_id=profile['model_contract_id'],
    source_commit=closure['source_commit'],changed_production_file=str(target),changed_production_sha256=sha(target),
    cache_before=base['execution']['checkpoint_cache'],cache_after=profile['execution']['checkpoint_cache'],model_verification=model,
    fixtures_are_input_only=True,old_numerical_checkpoints_imported=0,prepare_sha256=sha(__file__)))
assert not any(p.stat().st_mode&0o222 for p in runtime.rglob('*'))
print(json.dumps(dict(status='PREPARED_NOT_GPU_QUALIFIED',profile_sha256=sha(path),execution_profile_id=profile['execution_profile_id'],model=model)))
