"""Run the independent fresh native oracle against a pinned Q8 two-turn fixture."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time


VOCAB = 248320


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_ids(path, ids):
    Path(path).write_text(''.join(f'{int(token)}\n' for token in ids), encoding='ascii', newline='\n')


def compose(fixture, rejection=False):
    if rejection:
        return list(fixture['prefix'][:128]), [128], list(fixture['new_user_suffix'][:1]), None, 1
    first = list(fixture['prefix'])+list(fixture['first_suffix'])
    if fixture['next_limit'] != 128:
        raise ValueError('this oracle build is pinned to a 128-token second turn')
    return first, [len(fixture['prefix']), len(fixture['first_suffix'])], \
        list(fixture['new_user_suffix']), list(fixture['first_outputs']), int(fixture['first_limit'])


def profile_command(profile, binary, tokens_path, max_new):
    command = list(profile['execution']['oracle_command'])
    command[0] = str(Path(binary).resolve())
    max_new_indices = [i for i, item in enumerate(command) if item == '--max-new']
    if len(max_new_indices) != 1 or max_new_indices[0]+1 >= len(command) or command[max_new_indices[0]+1].startswith('--'):
        raise ValueError('oracle command must contain one --max-new with a value')
    command[max_new_indices[0]+1] = str(max_new)
    tokens_path = str(Path(tokens_path).resolve())
    token_flags = [i for i, item in enumerate(command) if item in ('--tokens', '--tokens-file')]
    if len(token_flags) > 1:
        raise ValueError('oracle command must contain at most one prompt-token option')
    if '--tokens' in command:
        token_index = command.index('--tokens')
        if token_index+1 >= len(command) or command[token_index+1].startswith('--'):
            raise ValueError('--tokens must have a value')
        del command[token_index:token_index+2]
        command.extend(['--tokens-file', tokens_path])
    elif '--tokens-file' in command:
        token_index = command.index('--tokens-file')
        if token_index+1 >= len(command) or command[token_index+1].startswith('--'):
            raise ValueError('--tokens-file must have a value')
        command[token_index+1] = tokens_path
    else:
        command.extend(['--tokens-file', tokens_path])
    return command


def runtime_environment(profile, inherited, output_stem, suffix_path, request_lengths, expected_path):
    environment = {
        key: value for key, value in inherited.items()
        if not key.startswith(('STRATA_', 'CUDA_', 'NVIDIA_', 'GGML_'))
    }
    environment.update(profile['execution'].get('environment', {}))
    environment.update({
        'STRATA_DUMP_FIRST_LOGITS': str(output_stem),
        'STRATA_NEXT_SUFFIX_FILE': str(suffix_path),
        'STRATA_EOS_IDS': ','.join(map(str, profile['execution']['eos_tokens'])),
        'STRATA_PREFILL_REQUESTS': ','.join(map(str, request_lengths)),
    })
    if expected_path is not None:
        environment['STRATA_EXPECT_FIRST_FILE'] = str(expected_path)
    return environment


def run(profile_path, fixture_path, binary, run_dir, rejection=False):
    profile_path, fixture_path, binary, run_dir = map(
        lambda value: Path(value).resolve(), (profile_path, fixture_path, binary, run_dir))
    if run_dir.exists():
        raise FileExistsError(f'run directory must be new: {run_dir}')
    run_dir.mkdir(parents=True)
    profile_raw, fixture_raw = profile_path.read_bytes(), fixture_path.read_bytes()
    profile, fixture = json.loads(profile_raw), json.loads(fixture_raw)
    tokens, request_lengths, suffix, expected, max_new = compose(fixture, rejection)
    tokens_path, suffix_path = run_dir/'first_prompt.ids', run_dir/'next_user_suffix.ids'
    write_ids(tokens_path, tokens)
    write_ids(suffix_path, suffix)
    expected_path = None
    if expected is not None:
        expected_path = run_dir/'expected_first_outputs.ids'
        write_ids(expected_path, expected)
    output_stem = run_dir/'oracle'
    command = profile_command(profile, binary, tokens_path, max_new)
    # Match the pinned profile while avoiding accidental inherited runtime or
    # instrumentation overrides from another native experiment.
    environment = runtime_environment(profile, os.environ, output_stem, suffix_path,
                                      request_lengths, expected_path)
    stdout_path, stderr_path = run_dir/'process.stdout', run_dir/'process.stderr'
    start = time.perf_counter()
    with stdout_path.open('wb') as out, stderr_path.open('wb') as err:
        result = subprocess.run(command, env=environment, stdout=out, stderr=err, check=False)
    elapsed = time.perf_counter()-start
    stdout = stdout_path.read_text(errors='replace')
    stderr = stderr_path.read_text(errors='replace')
    receipt = {
        'schema': 'fresh-native-multiturn-run-v1',
        'mode': 'first-turn-eos-rejection' if rejection else 'full-two-turn-oracle',
        'binary': str(binary), 'binary_sha256': sha(binary),
        'profile': str(profile_path), 'profile_sha256': sha(profile_path),
        'fixture': str(fixture_path), 'fixture_sha256': sha(fixture_path),
        'input_tokens': str(tokens_path), 'input_tokens_sha256': sha(tokens_path),
        'suffix_tokens': str(suffix_path), 'suffix_tokens_sha256': sha(suffix_path),
        'expected_first_tokens': str(expected_path) if expected_path else None,
        'expected_first_tokens_sha256': sha(expected_path) if expected_path else None,
        'command': command,
        'environment': {key: environment[key] for key in (
            'STRATA_DUMP_FIRST_LOGITS', 'STRATA_NEXT_SUFFIX_FILE', 'STRATA_EOS_IDS',
            'STRATA_PREFILL_REQUESTS')},
        'expected_first_output_count': len(expected) if expected is not None else None,
        'first_prompt_request_lengths': request_lengths,
        'actual_first_prompt_token_count': len(tokens),
        'actual_next_user_suffix_token_count': len(suffix),
        'exit_code': result.returncode,
        'elapsed_seconds': elapsed,
        'stdout': str(stdout_path), 'stdout_sha256': sha(stdout_path),
        'stderr': str(stderr_path), 'stderr_sha256': sha(stderr_path),
    }
    if rejection:
        observed_guard = next((line for line in stderr.splitlines()
                               if line.startswith('EOS_GUARD_REJECT ')), '')
        work_match = re.search(r'prefill_work_since_first_prompt=(\d+)(?:\s|$)', observed_guard)
        suffix_work = int(work_match.group(1)) if work_match else None
        sidecars_absent = not any(Path(str(output_stem)+suffix).exists() for suffix in (
            '.multiturn.json', '.second.frontier.f32', '.second.decode.f32', '.second.final.f32'))
        receipt['status'] = 'PASS_EXPECTED_EOS_GUARD_REJECTION' if (
            result.returncode == 3 and 'first turn did not stop on EOS; suffix prefill refused' in stderr
            and work_match is not None and suffix_work == 0 and sidecars_absent) \
            else 'FAIL_EXPECTED_EOS_GUARD_REJECTION'
        receipt['eos_guard_observation'] = observed_guard or None
        receipt['measured_prefill_work_since_first_prompt'] = suffix_work
        receipt['second_turn_sidecars_absent'] = sidecars_absent
    elif result.returncode == 0:
        turn_path = Path(str(output_stem)+'.multiturn.json')
        schedule_path = Path(str(output_stem)+'.schedule.json')
        chunks_path = Path(str(output_stem)+'.prefill-schedule.json')
        requests_path = Path(str(output_stem)+'.requests.f32')
        turn = json.loads(turn_path.read_text())
        schedule = json.loads(schedule_path.read_text())
        expected_schedule = {'requests': request_lengths}
        expected_first_consumed = len(tokens)+len(expected)
        if schedule.get('requests') != request_lengths:
            receipt['status'] = 'FAIL_FIRST_PREFILL_REQUEST_BOUNDARIES'
        elif turn['first_generated'] != expected or not turn['first_outputs_verified']:
            receipt['status'] = 'FAIL_FIRST_GENERATION_DIFFERS_FROM_FIXTURE'
        elif not turn['first_stopped_on_actual_eos'] or not (1 <= len(turn['second_generated']) <= 128):
            receipt['status'] = 'FAIL_SECOND_TURN_STOP_OR_COUNT'
        elif turn['first_consumed'] != expected_first_consumed or \
                turn['second_suffix_position'] != expected_first_consumed or \
                turn['second_suffix_tokens'] != len(suffix) or \
                turn['second_prefill_tokens'] != len(suffix) or \
                turn['total_forward_tokens'] != turn['normal_forward_tokens']+turn['total_prefill_tokens']:
            receipt['status'] = 'FAIL_NATIVE_POSITIONS_OR_WORK_COUNTERS'
        elif requests_path.stat().st_size != 2*VOCAB*4:
            receipt['status'] = 'FAIL_FIRST_REQUEST_FRONTIER_EXTENT'
        elif output_stem.stat().st_size != VOCAB*4 or \
                Path(str(output_stem)+'.decode.f32').stat().st_size != len(expected)*VOCAB*4 or \
                Path(str(output_stem)+'.second.frontier.f32').stat().st_size != VOCAB*4 or \
                Path(str(output_stem)+'.second.decode.f32').stat().st_size != len(turn['second_generated'])*VOCAB*4 or \
                Path(str(output_stem)+'.second.final.f32').stat().st_size != VOCAB*4:
            receipt['status'] = 'FAIL_FRONTIER_DUMP_EXTENT'
        else:
            receipt['status'] = 'PASS_FRESH_TWO_TURN_ORACLE'
        receipt.update(multiturn_report=str(turn_path), multiturn_report_sha256=sha(turn_path),
                       prefill_schedule=str(schedule_path), prefill_schedule_sha256=sha(schedule_path),
                       prefill_chunks=str(chunks_path), prefill_chunks_sha256=sha(chunks_path),
                       request_frontiers=str(requests_path), request_frontiers_sha256=sha(requests_path),
                       request_frontier_bytes=requests_path.stat().st_size,
                       expected_request_schedule=expected_schedule, native_multiturn_report=turn,
                       native_prefill_schedule=schedule,
                       second_expected_max_tokens=128,
                       first_frontier_bytes=output_stem.stat().st_size,
                       first_decode_trace_bytes=Path(str(output_stem)+'.decode.f32').stat().st_size,
                       second_prefill_frontier_bytes=Path(str(output_stem)+'.second.frontier.f32').stat().st_size,
                       second_decode_trace_bytes=Path(str(output_stem)+'.second.decode.f32').stat().st_size,
                       second_final_frontier_bytes=Path(str(output_stem)+'.second.final.f32').stat().st_size)
    else:
        receipt['status'] = 'FAIL_NATIVE_PROCESS'
    receipt_path = run_dir/'run.json'
    receipt_path.write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps(receipt, indent=2))
    return 0 if receipt['status'].startswith('PASS_') else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--rejection', action='store_true', help='small max-bound case must refuse suffix prefill')
    args = parser.parse_args()
    raise SystemExit(run(args.profile, args.fixture, args.binary, args.run_dir, args.rejection))
