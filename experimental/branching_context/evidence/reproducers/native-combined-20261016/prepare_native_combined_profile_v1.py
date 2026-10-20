"""Bind reviewed combined source to a fresh profile; never import old states."""
import copy
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

root = Path('/home/dflanag3/spatial-artifacts/native-client-combined-20261016-v1')
base_root = Path('/home/dflanag3/spatial-artifacts/q4-tree-retention-20261016-v1/gpu-candidate-v1')
base_runtime = base_root / 'runtime'
source = root / 'source-v1'
candidate = root / 'gpu-candidate-v1'
runtime = candidate / 'runtime'
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):
    return json.loads(path.read_bytes())
def write(path, value):
    with path.open('x') as output:
        json.dump(value, output, sort_keys=True, indent=2)
        output.write('\n')
assert sha(base_root / 'profile.json') == '107cab965436707d13db92271aa629e7829950030dbe60b884308076f8e31253'
assert read(root / 'cpu-v1.json')['status'] == 'PASS'
assert all((root / 'g0-command-v1' / name).read_text().strip() == '0'
           for name in ('command.exit', 'wrapper.exit'))
base = read(base_root / 'profile.json')
profile = copy.deepcopy(base)
for key in list(os.environ):
    if key.startswith(('STRATA_', 'CUDA_', 'NVIDIA_', 'GGML_')):
        del os.environ[key]
os.environ.update(base['execution']['environment'])
sys.path[:0] = [str(base_runtime), str(base_runtime / 'tools'),
               str(base_runtime / 'experimental/branching_context/src')]
from context_time_travel.strata_backend import _digest, _validate_execution, _validate_model
_validate_execution(base)
model = _validate_model(base)
candidate.mkdir(mode=0o700)
shutil.copytree(base_runtime, runtime)
changes = {
    'experimental/branching_context/src/context_time_travel/coordinator.py': 'a45638c8aa159c3ab91213d085e70822b8433d16ec9ad826e9c2a963d19d48dd',
    'experimental/branching_context/src/context_time_travel/store.py': 'db2e595f2b54e43b99ea6ae3de27f4790be4ce423475f289da33e75d7288db27',
    'serve/cow_adapter.py': '3acda1602f2100cf1e8058ef4f619f9ece0245ea42eb78e1e1e7ee42781ac6dc',
    'serve/cow_text_adapter.py': '8ed365c573d3a3db441205fbc33f24f7d879e1f3325d78ce3369650a0dc79726'}
for name, expected in changes.items():
    assert sha(source / name) == expected
    target = runtime / name
    target.chmod(0o600)
    shutil.copyfile(source / name, target)
    target.chmod(0o400)
def remap(name):
    path = Path(name)
    return str(runtime / path.relative_to(base_runtime)) if path.is_relative_to(base_runtime) else name
pins = {}
for name, expected in base['native_build']['sha256'].items():
    mapped = remap(name)
    relative = str(Path(mapped).relative_to(runtime)) if Path(mapped).is_relative_to(runtime) else None
    assert sha(mapped) == changes.get(relative, expected), mapped
    pins[mapped] = sha(mapped)
assert all(str(runtime / name) in pins for name in changes)
profile['native_build']['sha256'] = pins
closure = profile['native_build']['selected_runtime_source_closure']
updated = set()
for name, expected in list(closure['files_sha256'].items()):
    normalized = name.replace('\\', '/')
    if normalized in changes:
        closure['files_sha256'][name] = changes[normalized]
        updated.add(normalized)
assert updated == set(changes), updated
closure['source_commit'] = '97bab1a24ee53df036a3ac5bd083e18ec7f5cc71'
closure['source_status'] = 'Retention runtime plus four reviewed combined-source overlays; exact file pins authoritative'
profile['gguf_reader'] = remap(profile['gguf_reader'])
assert profile['execution']['checkpoint_cache'] == dict(byte_limit=8*1024**3, entry_limit=8)
profile['execution_profile_id'] = _digest(dict(build=profile['native_build'], execution=profile['execution'], hardware=profile['hardware']))
profile['numerical_gate_qualified'] = False
profile['calibration_status'] = 'NOT_RUN'
_validate_execution(profile)
_validate_model(profile)
write(candidate / 'profile.json', profile)
(candidate / 'profile.json').chmod(0o400)
fixtures = read(base_root / 'fixture-set.json')
for item in fixtures['contexts']:
    assert sha(item['fixture']) == item['fixture_sha256']
for item in fixtures['imports'].values():
    item['path'] = remap(item['path'])
    assert sha(item['path']) == item['sha256']
fixtures.update(profile_path=str(candidate / 'profile.json'), profile_sha256=sha(candidate / 'profile.json'),
    execution_profile_id=profile['execution_profile_id'], builder_sha256=sha(__file__), status='PREPARED_NOT_RUN',
    source_scope='Exact input text/token fixtures only; new store and no imported numerical checkpoints')
write(candidate / 'fixture-set.json', fixtures)
admission = dict(status='PREPARED_NOT_GPU_QUALIFIED', profile_sha256=sha(candidate / 'profile.json'),
    execution_profile_id=profile['execution_profile_id'], changed_files=changes, source_commit=closure['source_commit'],
    model_verification=model, old_numerical_checkpoints_imported=0, prepare_sha256=sha(__file__))
write(candidate / 'admission.json', admission)
assert not any(p.stat().st_mode & 0o222 for p in runtime.rglob('*'))
print(json.dumps(admission, sort_keys=True))
