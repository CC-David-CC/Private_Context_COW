#!/usr/bin/env python3
import copy
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import time

CANDIDATE_COMMIT = '11cf9976578ca853bf78b67a308f9c544ea6f0db'
EXPECTED_COORDINATOR = 'bb6a1f8d9582c35dfaec69391658a033320e25d717b61ebfb9b0bec1f6c6c32b'
EXPECTED_BACKEND = 'd692e00310984dcea2332d86862fde0010ff06f50b7f62b35a97d25110443de1'
EXPECTED_FAULT_HELPER = 'd4bb1469f4fb0ff58047ca88350faf95a80093a36c4ae499545d1ddab2232695'
EXPECTED_BASE_PROFILE = 'f3f54a13f20d91c9035600c9434b321e81d05b27c91dd992db67ddfadd58243c'
EXPECTED_V1_PROFILE = 'e272e975b1e171507ac4426a7f334645f50b3fbf22a1b6d143b9065d32582dc5'
EXPECTED_NATIVE = '4bd5d715d460371458be8cadad28d881977dbeea57219039187d05aa12148451'
ART = Path('/home/dflanag3/spatial-artifacts/q4-terminal-error-profile-20261016-v3').resolve()
ROOT_SOURCE = Path('/home/dflanag3/spatial-artifacts/q4-terminal-error-cpu-20261016-v1/replacement-root-green-v4/source')
SOURCE_PACKAGE = ROOT_SOURCE / 'src/context_time_travel'
SOURCE_FAULT_HELPER = ROOT_SOURCE / 'scripts/run_q4_publication_fault.py'
SOURCE_RUNTIME = Path('/home/dflanag3/spatial-artifacts/q4-publication-fault-20261015-v1/runtime-v4')
V1_PROFILE = Path('/home/dflanag3/spatial-artifacts/q4-terminal-error-profile-20261016-v1/profile/profile.json')
BASE_PROFILE = Path('/home/dflanag3/spatial-artifacts/lazy-prefix-20261015-v1/profiles-v2/q4/profile.json')
SCRIPT = ART / 'derive_terminal_error_profile_v3.py'
RUNTIME = ART / 'runtime'
API = RUNTIME / 'experimental/branching_context'
PACKAGE = API / 'src/context_time_travel'
SCRIPTS = API / 'scripts'
TOOLS = RUNTIME / 'tools'
PROFILE_DIR = ART / 'profile'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n',
                          encoding='utf-8', newline='\n')


def ensure_model_stats_unchanged(profile):
    for name, expected in profile['model_files'].items():
        info = Path(name).stat()
        actual = (info.st_size, info.st_mtime_ns, info.st_ctime_ns)
        pinned = (expected['bytes'], expected['mtime_ns'], expected['ctime_ns'])
        if actual != pinned:
            raise RuntimeError('Model stat drift; refuse profile prep before model rehash: ' + name)


def source_files(root):
    return sorted(path for path in root.rglob('*.py')
                  if '__pycache__' not in path.parts and path.is_file() and not path.is_symlink())


def main():
    if not ART.is_dir():
        raise RuntimeError('Artifact root must be created before this script is transferred')
    unexpected = [p.name for p in ART.iterdir() if p.name != SCRIPT.name]
    if unexpected:
        raise RuntimeError('Fresh artifact root contains unexpected entries: ' + repr(unexpected))
    if sha(V1_PROFILE) != EXPECTED_V1_PROFILE or sha(BASE_PROFILE) != EXPECTED_BASE_PROFILE:
        raise RuntimeError('Reviewed profile input SHA mismatch')
    if not SOURCE_PACKAGE.is_dir() or sha(SOURCE_FAULT_HELPER) != EXPECTED_FAULT_HELPER:
        raise RuntimeError('Frozen candidate API/support source is missing or differs from pin')
    if any(path.is_symlink() for path in SOURCE_RUNTIME.rglob('*')):
        raise RuntimeError('Support runtime contains symlinked inputs; refuse ambiguous source closure')
    if RUNTIME.exists() or PROFILE_DIR.exists():
        raise RuntimeError('Refusing to overwrite a candidate runtime/profile stage')

    # Preserve the support runtime but omit its old API package so the candidate package is copied to a fresh path.
    source_api_dir = SOURCE_RUNTIME / 'experimental/branching_context/src'
    def ignore_runtime(directory, names):
        ignored = {name for name in names if name in ('__pycache__', '.pytest_cache') or name.endswith('.pyc')}
        if Path(directory).resolve() == source_api_dir.resolve():
            ignored.add('context_time_travel')
        return ignored
    shutil.copytree(SOURCE_RUNTIME, RUNTIME, copy_function=shutil.copyfile, ignore=ignore_runtime)
    destination_api_dir = API / 'src'
    destination_api_dir.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
    shutil.copytree(SOURCE_PACKAGE, PACKAGE, copy_function=shutil.copyfile,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.pytest_cache'))
    SCRIPTS.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
    shutil.copyfile(SOURCE_FAULT_HELPER, SCRIPTS / 'run_q4_publication_fault.py')
    for path in source_files(RUNTIME):
        path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)

    base = json.loads(BASE_PROFILE.read_text(encoding='utf-8'))
    v1 = json.loads(V1_PROFILE.read_text(encoding='utf-8'))
    command = base['execution']['sut_command']
    native = Path(command[0]).resolve(strict=True)
    if sha(native) != EXPECTED_NATIVE:
        raise RuntimeError('Existing Q4 native binary differs from reviewed pin')
    ensure_model_stats_unchanged(base)

    files = source_files(RUNTIME)
    if not files:
        raise RuntimeError('Copied runtime has no source files')
    closure = {str(path.resolve()): sha(path) for path in files}
    coordinator_path = (PACKAGE / 'coordinator.py').resolve()
    backend_path = (PACKAGE / 'strata_backend.py').resolve()
    fault_path = (SCRIPTS / 'run_q4_publication_fault.py').resolve()
    if closure.get(str(coordinator_path)) != EXPECTED_COORDINATOR:
        raise RuntimeError('Candidate coordinator source differs from reviewed pin')
    if closure.get(str(backend_path)) != EXPECTED_BACKEND:
        raise RuntimeError('Candidate backend source differs from reviewed pin')
    if closure.get(str(fault_path)) != EXPECTED_FAULT_HELPER:
        raise RuntimeError('Selected closure helper differs from reviewed pin')
    if closure.get(str((RUNTIME / 'serve/cow_http.py').resolve())) != '9f6839a359e05cbb47e0850046c6f32f00e7468265dff80fedc3d72d9f720f31':
        raise RuntimeError('Coherent HTTP runtime source differs from reviewed pin')

    package_manifest = {str(path.resolve()): sha(path) for path in source_files(PACKAGE)}
    if package_manifest[str(coordinator_path)] != EXPECTED_COORDINATOR or package_manifest[str(backend_path)] != EXPECTED_BACKEND:
        raise RuntimeError('Candidate package manifest mismatch')
    v1_closure = copy.deepcopy(v1['native_build'].get('selected_runtime_source_closure'))
    profile = copy.deepcopy(v1)
    profile['native_build']['historical_runtime_source_closure_v1'] = {
        'status': 'PROVENANCE_ONLY_NOT_SELECTED',
        'source_commit': v1_closure.get('source_commit') if isinstance(v1_closure, dict) else None,
        'files_sha256': v1_closure.get('files_sha256') if isinstance(v1_closure, dict) else {},
        'defect': 'v1 selected old coordinator/backend source hashes while candidate hashes were only appended outside the selected runtime closure',
    }
    profile['native_build']['selected_runtime_source_closure'] = {
        'source_commit': CANDIDATE_COMMIT,
        'source_components': {
            'candidate_api_commit': CANDIDATE_COMMIT,
            'support_runtime_source': str(SOURCE_RUNTIME.resolve()),
            'support_runtime_selected_closure_from_v1': (v1_closure.get('source_commit') if isinstance(v1_closure, dict) else None),
        },
        'files_sha256': closure,
    }
    profile['native_build']['control_plane_source'] = {
        'git_commit': CANDIDATE_COMMIT,
        'source_stage': str(ROOT_SOURCE.resolve()),
        'package_sha256': package_manifest,
        'runtime_root': str(RUNTIME.resolve()),
        'selected_runtime_source_closure_sha256': hashlib.sha256(
            json.dumps(closure, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
        'candidate_commit_scope': 'candidate control-plane API; selected runtime closure includes the actual candidate package plus pinned support runtime sources; native binary unchanged',
    }
    profile['native_build']['sha256'].update(closure)
    profile['numerical_gate_qualified'] = False
    profile['calibration_status'] = 'NOT_RUN'
    for key in list(profile):
        if key.startswith(('baseline', 'admission')) or key in ('atol', 'rtol', 'tolerance_rule'):
            profile.pop(key)
    profile['execution_profile_id'] = 'pending'

    # Prove the imported Python modules originate in this copied runtime, not the worker checkout.
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(RUNTIME))
    sys.path.insert(0, str(SCRIPTS))
    sys.path.insert(0, str(API / 'src'))
    sys.path.insert(0, str(TOOLS))
    from context_time_travel.strata_backend import _digest, _validate_execution, _validate_model
    fault_spec = importlib.util.spec_from_file_location('selected_q4_publication_fault', fault_path)
    fault_module = importlib.util.module_from_spec(fault_spec)
    fault_spec.loader.exec_module(fault_module)
    closure_proof = fault_module._runtime_closure(profile, RUNTIME)
    if set(closure_proof['files']) != set(closure):
        raise RuntimeError('Actual _runtime_closure verified a different file set than selected profile closure')
    if any(row['sha256'] != closure[name] for name, row in closure_proof['files'].items()):
        raise RuntimeError('Actual _runtime_closure file hashes differ')

    module = importlib.import_module('context_time_travel.strata_backend')
    if Path(module.__file__).resolve() != backend_path or sha(module.__file__) != EXPECTED_BACKEND:
        raise RuntimeError('Imported candidate backend did not come from the v2 runtime')
    import run_root_reuse_probe as reuse_probe
    import native_request_probe as native_probe
    import pool_campaign as pool_campaign_module
    import tools.strata_tokenizer as tokenizer_module
    imported = [module, importlib.import_module('context_time_travel.coordinator'),
                importlib.import_module('context_time_travel.real'),
                importlib.import_module('context_time_travel.store'),
                importlib.import_module('context_time_travel.pools'),
                importlib.import_module('context_time_travel.execution_receipt'),
                importlib.import_module('context_time_travel.checkpoint_cache'),
                importlib.import_module('context_time_travel.controller'),
                native_probe, pool_campaign_module, reuse_probe, tokenizer_module,
                importlib.import_module('serve.cow_http')]
    import_witness = []
    for mod in imported:
        origin = Path(mod.__file__).resolve(strict=True)
        try:
            relative = origin.relative_to(RUNTIME.resolve())
        except ValueError as error:
            raise RuntimeError(f'Imported source escaped coherent runtime: {mod.__name__} at {origin}') from error
        if sha(origin) != closure.get(str(origin)):
            raise RuntimeError(f'Imported source hash is not in selected closure: {mod.__name__}')
        import_witness.append({'module': mod.__name__, 'path': str(origin), 'sha256': sha(origin)})
    profile['execution_profile_id'] = _digest(dict(
        build=profile['native_build'], execution=profile['execution'], hardware=profile['hardware']))

    for key in list(os.environ):
        if key.startswith(('STRATA_', 'CUDA_', 'NVIDIA_', 'GGML_')):
            del os.environ[key]
    os.environ.update(profile['execution']['environment'])
    _validate_execution(profile)
    model_before = ensure_model_stats_unchanged(profile)
    model_validation_before = _validate_model(profile)
    model_validation_after = _validate_model(profile)
    if profile['execution'] != base['execution']:
        raise RuntimeError('Execution knobs changed during profile derivation')
    if profile['model_contract_id'] != base['model_contract_id'] or profile['hardware'] != base['hardware']:
        raise RuntimeError('Model contract or hardware changed during profile derivation')
    if sha(native) != EXPECTED_NATIVE:
        raise RuntimeError('Native binary changed during profile derivation')

    PROFILE_DIR.mkdir(mode=0o700)
    profile_path = PROFILE_DIR / 'profile.json'
    write_json(profile_path, profile)
    write_json(ART / 'runtime-import-witness.json', {
        'runtime_root': str(RUNTIME.resolve()), 'runtime_file_count': len(closure),
        'runtime_source_closure': closure_proof, 'imports': import_witness,
        'candidate_git_commit': CANDIDATE_COMMIT,
        'candidate_coordinator_sha256': EXPECTED_COORDINATOR,
        'candidate_backend_sha256': EXPECTED_BACKEND,
        'native_binary_sha256': EXPECTED_NATIVE,
    })
    write_json(PROFILE_DIR / 'derivation.json', {
        'status': 'PREPARED_NOT_NUMERICALLY_QUALIFIED',
        'base_profile_sha256': sha(BASE_PROFILE), 'v1_profile_sha256': sha(V1_PROFILE),
        'v1_selected_closure_defect': 'candidate module hashes were not selected; old coordinator/backend remained selected',
        'candidate_git_commit': CANDIDATE_COMMIT,
        'coordinator_sha256': EXPECTED_COORDINATOR, 'strata_backend_sha256': EXPECTED_BACKEND,
        'support_runtime_source': str(SOURCE_RUNTIME.resolve()),
        'runtime_root': str(RUNTIME.resolve()), 'runtime_file_count': len(closure),
        'selected_runtime_source_closure_verified_by_actual_helper': True,
        'import_origin_witness_path': str((ART / 'runtime-import-witness.json').resolve()),
        'new_profile_sha256': sha(profile_path),
        'new_execution_profile_id': profile['execution_profile_id'],
        'native_binary': str(native), 'native_binary_sha256': sha(native),
        'model_contract_id_unchanged': profile['model_contract_id'],
        'execution_settings_unchanged': True,
        'model_stat_precheck': 'PASS',
        'model_validation_before': model_validation_before,
        'model_validation_after': model_validation_after,
        'profile_derivation': 'actual coherent runtime source closure selected; execution ID recomputed with candidate strata_backend._digest; _runtime_closure, _validate_execution, and _validate_model passed',
        'checkpoint_compatibility': 'new execution_profile_id; regenerate the 7,999-token root under this profile; never relabel/reuse old-profile checkpoint',
        'inference_performed': False, 'created_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    })
    for path in ART.rglob('*'):
        if path.is_dir():
            path.chmod(stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
        else:
            path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    ART.chmod(stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
    print(json.dumps({'status': 'PREPARED_NOT_NUMERICALLY_QUALIFIED',
        'profile_sha256': sha(profile_path), 'execution_profile_id': profile['execution_profile_id'],
        'runtime_file_count': len(closure), 'imports_verified': len(import_witness),
        'inference_performed': False}, sort_keys=True))


if __name__ == '__main__':
    main()