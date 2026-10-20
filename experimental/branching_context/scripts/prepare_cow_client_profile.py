"""Prepare a fresh profile for the staged client path; never run inference."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--binaries', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    api = repo / 'experimental/branching_context'
    base = json.loads(args.base.read_bytes())
    # Refuse drift before the existing helper could choose to rehash weights.
    for name, expected in base['model_files'].items():
        actual = Path(name).stat()
        if (actual.st_size, actual.st_mtime_ns, actual.st_ctime_ns) != (
                expected['bytes'], expected['mtime_ns'], expected['ctime_ns']):
            raise RuntimeError('Model stat drift; investigate before rehash: ' + name)
    subprocess.run([sys.executable, str(api / 'scripts/prepare_native_request_profile.py'),
        '--base', str(args.base), '--binaries', str(args.binaries), '--api', str(api),
        '--output', str(args.output), '--chunk', '8192', '--capacity', '98304',
        '--checkpoint-cache-mib', '8192', '--checkpoint-cache-entries', '3'], check=True)
    profile_path = args.output / 'profile.json'
    profile = json.loads(profile_path.read_bytes())
    template = profile['execution'].setdefault('text_template_kwargs', {'enable_thinking': False})
    if template != {'enable_thinking': False}:
        raise RuntimeError('Client campaign requires explicit non-thinking template configuration')
    # Bind the actual server Python source as well as the native/API closure.
    serving = {str(path.resolve()): sha(path) for path in sorted((repo / 'serve').glob('*.py'))
               if not path.name.startswith('test_')}
    profile['native_build']['sha256'].update(serving)
    sys.path[:0] = [str(api / 'src'), str(repo), str(api / 'scripts')]
    from context_time_travel.strata_backend import _digest, _validate_execution, _validate_model
    profile['execution_profile_id'] = _digest(dict(build=profile['native_build'],
        execution=profile['execution'], hardware=profile['hardware']))
    for key in list(os.environ):
        if key.startswith(('STRATA_', 'CUDA_', 'NVIDIA_', 'GGML_')):
            del os.environ[key]
    os.environ.update(profile['execution']['environment'])
    _validate_execution(profile)
    model = _validate_model(profile)
    modules = []
    for path in sorted((api / 'src/context_time_travel').glob('*.py')):
        name = 'context_time_travel' + ('' if path.stem == '__init__' else '.' + path.stem)
        module = importlib.import_module(name)
        actual = Path(module.__file__).resolve()
        if actual != path.resolve() or profile['native_build']['sha256'].get(str(actual)) != sha(actual):
            raise RuntimeError('Runtime source origin/pin mismatch: ' + name)
        modules.append(dict(name=name, path=str(actual), sha256=sha(actual)))
    profile_path.write_text(json.dumps(profile, indent=2) + '\n', encoding='utf-8', newline='\n')
    report = dict(status='PREPARED_NOT_GPU_QUALIFIED', base_profile=str(args.base),
        base_profile_sha256=sha(args.base), base_profile_id=base['execution_profile_id'],
        profile=str(profile_path), profile_sha256=sha(profile_path),
        execution_profile_id=profile['execution_profile_id'], model_contract_id=profile['model_contract_id'],
        module_count=len(modules), modules=modules, serving_source_files=serving,
        model_verification=model, producer_sha256=sha(__file__),
        checkpoint_compatibility='New profile identity; fresh store required; do not reuse old checkpoints')
    (args.output / 'admission.json').write_text(json.dumps(report, indent=2) + '\n',
        encoding='utf-8', newline='\n')
    print(json.dumps({key: report[key] for key in ('status', 'execution_profile_id',
        'profile_sha256', 'module_count', 'model_verification')}), flush=True)


if __name__ == '__main__':
    main()
