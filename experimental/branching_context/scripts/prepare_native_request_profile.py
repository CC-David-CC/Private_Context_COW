"""Bind the experimental request capability to measured source/build identities."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys

from prepare_speed_profile import sha, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--binaries', type=Path, required=True)
    parser.add_argument('--api', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--chunk', type=int, default=8192)
    parser.add_argument('--capacity',type=int,help='New exact context capacity; changes execution identity')
    parser.add_argument('--vram-reserve-mib',type=int,help='Reserve GPU space for measured COW/state allocations')
    parser.add_argument('--checkpoint-cache-mib',type=int,help='Conservative referenced-state extent budget, at most 32768 MiB')
    parser.add_argument('--checkpoint-cache-entries',type=int,default=3)
    args = parser.parse_args()
    if (args.chunk <= 0 or (args.capacity is not None and args.capacity < args.chunk)
            or (args.vram_reserve_mib is not None and args.vram_reserve_mib <= 0)
            or (args.checkpoint_cache_mib is not None and not 0 < args.checkpoint_cache_mib <= 32768)
            or not 0 < args.checkpoint_cache_entries <= 256):
        parser.error('Positive bounded capacities and resource budgets required')
    profile = copy.deepcopy(json.loads(args.base.read_text()))
    sys.path.insert(0, str(args.api/'src'))
    from context_time_travel.strata_backend import _validate_execution, _validate_model
    for key in list(os.environ):
        if key.startswith(('STRATA_', 'CUDA_', 'NVIDIA_', 'GGML_')):
            del os.environ[key]
    os.environ.update(profile['execution']['environment'])
    _validate_execution(profile)
    _validate_model(profile)
    receipt_path = args.binaries.resolve()/'build.json'
    receipt = json.loads(receipt_path.read_text())
    if receipt.get('status') != 'BUILT_NOT_EXECUTED' or receipt.get('native_request_protocol') != 'consumed-greedy-ack-v1':
        raise ValueError('Require an actually compiled native request capability')
    paths = {str(receipt_path): sha(receipt_path)}
    for category in ('source_sha256', 'generated_sha256'):
        for name, expected in receipt[category].items():
            if sha(name) != expected:
                raise ValueError('Compiled source changed: '+name)
            paths[name] = expected
    for name, expected in receipt['binaries'].items():
        path = args.binaries.resolve()/name
        if sha(path) != expected:
            raise ValueError('Compiled executable changed')
        paths[str(path)] = expected
    # Actual static archives used by the retained link commands.
    for command in receipt['commands']:
        for name in command['argv']:
            if name.endswith('.a'):
                path = Path(name)
                if not path.is_absolute():
                    path = Path(receipt['link_working_directory'])/path
                paths[str(path.resolve())] = sha(path)
    for path in (args.api/'src/context_time_travel').glob('*.py'):
        paths[str(path.resolve())] = sha(path)
    paths[profile['gguf_reader']] = sha(profile['gguf_reader'])
    profile['native_build'] = dict(revision=profile['backend_revision'], sha256=paths,
        probe_build_sha256=sha(receipt_path), source_sha256=receipt['source_sha256'],
        source_status='staged driver over pinned base; exact file hashes authoritative')
    execution = profile['execution']
    for name, binary in [('sut_command','strata-batched-branch'), ('oracle_command','strata-batched-fresh')]:
        command = execution[name]
        command[0] = str(args.binaries.resolve()/binary)
        command[command.index('--prefill')+1] = str(args.chunk)
        if args.capacity is not None:
            command[command.index('--max-context')+1] = str(args.capacity)
        if args.vram_reserve_mib is not None:
            if '--vram-reserve-mib' not in command:
                command += ['--vram-reserve-mib',str(args.vram_reserve_mib)]
            else:
                command[command.index('--vram-reserve-mib')+1] = str(args.vram_reserve_mib)
    if args.capacity is not None:
        execution['capacity'] = args.capacity
    if args.checkpoint_cache_mib is not None:
        execution['checkpoint_cache'] = dict(byte_limit=args.checkpoint_cache_mib*1024**2,
                                              entry_limit=args.checkpoint_cache_entries)
    execution.update(prefill_chunk=args.chunk, generated_operator='normal-verifier-decode',
        native_request_protocol='consumed-greedy-ack-v1',
        resume_contract=receipt.get('resume_contract','native-request-primitive-experiment-v1'),
        operator='batched-input/native-greedy-request/per-write-sha256')
    for key in list(profile):
        if key.startswith(('calibration','baseline','admission')) or key in ('atol','rtol','tolerance_rule'):
            profile.pop(key)
    profile.update(numerical_gate_qualified=False, calibration_status='NOT_RUN',
        oracle_binary=str(args.binaries.resolve()/'strata-batched-fresh'))
    profile['execution_profile_id'] = digest(dict(build=profile['native_build'], execution=execution,
                                                 hardware=profile['hardware']))
    _validate_execution(profile)
    _validate_model(profile)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'profile.json').write_text(json.dumps(profile, indent=2)+'\n')
    print(json.dumps(dict(status='PREPARED_NOT_EXECUTED',
        execution_profile_id=profile['execution_profile_id'], profile=str(args.output/'profile.json'))))


if __name__ == '__main__':
    main()
