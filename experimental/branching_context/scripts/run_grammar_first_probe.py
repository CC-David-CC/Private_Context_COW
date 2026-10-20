"""Bounded real-GPU grammar integration/restart probe, not G1 certification.

No tools execute. Independent parser tests qualify token masks separately.
This probe compares uninterrupted and restarted execution of the same native
operators; it does not substitute for the independent full-history oracle.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import time
import signal


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def publish(path, value):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def require(value, reason):
    if not value:
        raise AssertionError(reason)


def completed_action(result, engine, tokenizer):
    require(result['grammar_complete'] is True and result['stopped'] is True,
            'A partial action cannot qualify the action probe')
    return json.loads(b''.join(tokenizer.token_bytes(token) for token in result['tokens']
        if token not in engine.eos_tokens).decode('utf-8', errors='strict'))


def live_group_pids(group):
    members = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            fields = (entry / 'stat').read_text().rsplit(')', 1)[1].split()
            if int(fields[2]) == group and fields[0] != 'Z':
                members.append(int(entry.name))
        except (OSError, ValueError, IndexError):
            continue  # A concurrently exiting process is not a surviving job.
    return sorted(members)


def run_stage(command, stdout_path, stderr_path, *, timeout=600):
    """Own a separate session so timeout cleanup includes native descendants."""
    with Path(stdout_path).open('wb') as stdout, Path(stderr_path).open('wb') as stderr:
        child = subprocess.Popen(command, stdout=stdout, stderr=stderr, start_new_session=True)
        timed_out = False
        try:
            code = child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            code = None
        survivors = live_group_pids(child.pid)
        cleanup_needed = bool(survivors)
        if cleanup_needed:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline and live_group_pids(child.pid):
                child.poll()  # Reap our direct child while its group winds down.
                time.sleep(0.02)
            if live_group_pids(child.pid):
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        code = child.wait(timeout=10)
        return dict(pid=child.pid, exit_code=code, timed_out=timed_out,
                    cleanup_needed=cleanup_needed, live_group_pids=live_group_pids(child.pid),
                    pid_absent=not Path('/proc', str(child.pid)).exists())


def finish_backend(backend, record):
    try:
        backend.close()
    except BaseException as exc:
        record['status'] = 'FAIL'
        record['failures'].append(dict(type=type(exc).__name__, message='cleanup: ' + str(exc)))
        # Close failures must not strand this owned native child.
        if backend.process.poll() is None:
            try:
                backend.process.kill()
                backend.process.wait(timeout=30)
            except BaseException as cleanup:
                record['failures'].append(dict(type=type(cleanup).__name__, message='forced cleanup: ' + str(cleanup)))
    finally:
        record['native_exit_code'] = backend.process.returncode
        record['native_pid_absent'] = not Path('/proc', str(backend.process.pid)).exists()
        if record['native_exit_code'] != 0 or not record['native_pid_absent']:
            record['status'] = 'FAIL'


def frontier(model, snapshot):
    values = model.next_logits(snapshot)
    return struct.pack('<' + 'f' * len(values), *values)


def controller_digest(model, snapshot):
    from context_time_travel.store import _json_bytes
    return hashlib.sha256(_json_bytes(model.controller(snapshot))).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('campaign', 'prepare', 'resume'), required=True)
    for name in ('runtime-root', 'profile', 'fixture', 'run-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    runtime = args.runtime_root.resolve(strict=True)
    profile = read(args.profile)
    sys.path[:0] = [str(runtime), str(runtime / 'tools'),
                    str(runtime / 'experimental/branching_context/src'),
                    str(runtime / 'experimental/branching_context/examples'),
                    str(runtime / 'experimental/branching_context/scripts')]
    from run_cow_client_recovery import ProgressTracker
    from grammar_first_agent import load_grammars
    from context_time_travel.grammar_first import GrammarEngine
    from context_time_travel.real import BranchingModel
    from context_time_travel.strata_backend import StrataBackend, _validate_execution
    from tools.strata_tokenizer import Tokenizer
    for key in list(os.environ):
        if key.startswith(('STRATA_', 'CUDA_', 'NVIDIA_', 'GGML_')):
            del os.environ[key]
    os.environ.update(profile['execution']['environment'])
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    _validate_execution(profile)
    output = args.run_dir.absolute()
    if args.stage == 'campaign':
        output.mkdir(mode=0o700, parents=True, exist_ok=False)
        outcomes = []
        for stage in ('prepare', 'resume'):
            command = [sys.executable, '-B', str(Path(__file__).resolve()),
                       '--stage', stage, '--runtime-root', str(runtime),
                       '--profile', str(args.profile.resolve()), '--fixture', str(args.fixture.resolve()),
                       '--run-dir', str(output)]
            receipt = run_stage(command, output / (stage + '.stdout'), output / (stage + '.stderr'))
            outcomes.append(dict(stage=stage, **receipt))
            healthy = (receipt['exit_code'] == 0 and receipt['pid_absent'] and
                       not receipt['timed_out'] and not receipt['cleanup_needed'] and not receipt['live_group_pids'])
            publish(output / 'campaign.json', dict(status='RUNNING' if healthy else 'FAIL', processes=outcomes))
            require(healthy, 'Process-separated stage failed or leaked descendants')
        record = read(output / 'resume.json')
        require(record['status'] == 'PASS', 'Resume did not pass')
        publish(output / 'campaign.json', dict(status='PASS', processes=outcomes,
            numerical_max_error=record['numerical_max_error'], comparisons=record['comparisons'],
            qualification='Bounded Q4 static-GBNF integration only; no cumulative gate claim'))
        return
    require(output.is_dir(), 'Campaign output must exist')
    require(not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                                       text=True).strip(), 'GPU is already in use')
    progress_output = output / (args.stage + '-progress')
    progress_output.mkdir(mode=0o700, exist_ok=False)
    tracker = ProgressTracker(progress_output, 'grammar-first-' + args.stage).start()
    os.environ['CTT_NATIVE_TRACE'] = str(output / (args.stage + '-native.jsonl'))
    record = dict(status='RUNNING', stage=args.stage, python_pid=os.getpid(), profile_sha256=sha(args.profile),
                  script_sha256=sha(__file__), fixture_sha256=sha(args.fixture), actions=[],
                  failures=[], tools_executed=0, controller_scope='greedy with static GBNF; no random sampling')
    target = output / (args.stage + '.json')
    publish(target, record)
    backend = None
    try:
        tokenizer = Tokenizer.from_gguf(profile['model_path'])
        engine = GrammarEngine(tokenizer, load_grammars(), eos_tokens=profile['execution']['eos_tokens'])
        tracker.phase('native-start')
        backend = StrataBackend(profile)
        native_pid = backend.process.pid
        record['native_pid'] = native_pid
        tracker.native_started(native_pid)
        model = BranchingModel(output / 'store', backend, profile)
        case = read(args.fixture)
        ids = case['root_token_ids']
        require(len(ids) == 7231, 'Use the pinned actual code-context fixture')
        if args.stage == 'prepare':
            tracker.phase('prefill')
            root = model.create(ids)
            model.checkpoint(root)
            record['root'] = root
            tracker.phase('partial-read')
            partial = model.run_state_request(root, (), 4, engine=engine, state='READ', stream='read')
            require(not partial['stopped'], 'READ fixture must remain active at four tokens')
            model.checkpoint(partial['snapshot'])
            record['partial'] = partial
        else:
            prepared = read(output / 'prepare.json')
            require(prepared['status'] == 'PASS' and prepared['native_exit_code'] == 0 and prepared['native_pid_absent'],
                    'Producer did not finish cleanly')
            require(not Path('/proc', str(prepared['python_pid'])).exists() and
                    not Path('/proc', str(prepared['native_pid'])).exists(), 'Producer survived restart')
            require(prepared['native_pid'] != native_pid, 'Need a genuinely new native process')
            root = prepared['root']
            tracker.phase('strict-restore-partial')
            receipt = model.restore_checkpoint(prepared['partial']['snapshot'])
            require(receipt['forward_tokens'] == 0, 'Strict restore recomputed history')
            record['partial_restore'] = receipt
            continued = model.run_state_request(prepared['partial']['snapshot'], (), 256,
                engine=engine, state='READ', stream='ignored', resume=True)
            completed_action(dict(continued, tokens=prepared['partial']['tokens'] + continued['tokens']), engine, tokenizer)
            reference = next(row for row in prepared['actions'] if row['state'] == 'READ')
            require(prepared['partial']['tokens'] + continued['tokens'] == reference['tokens'],
                    'Mid-grammar restart changed continuation tokens')
            require(controller_digest(model, continued['snapshot']) == reference['controller_sha256'],
                    'Mid-grammar restart changed the controller')
            require(frontier(model, continued['snapshot']) == (output / reference['frontier_file']).read_bytes(),
                    'Mid-grammar restart changed the final logits')
            record['partial_exact'] = True
            record['comparisons'] = 1
            record['numerical_max_error'] = 0
            tracker.phase('strict-restore-root')
            record['root_restore'] = model.restore_checkpoint(root)
            require(record['root_restore']['forward_tokens'] == 0, 'Root restore recomputed prefix')
        original = model.snapshot_bytes(root)
        record['root_tokens'] = len(ids)
        for state in ('READ', 'EDIT', 'TEST', 'FINISH'):
            tracker.phase('generate-' + state)
            before = backend.work_counters()
            result = model.run_state_request(root, (), 256, engine=engine, state=state, stream=state.lower())
            after = backend.work_counters()
            require(after['consumed_tokens'] - before['consumed_tokens'] == len(result['tokens']),
                    'Grammar request recomputed its prefix or missed generated work')
            require(model.snapshot_bytes(root) == original, 'Grammar request mutated its parent')
            name = args.stage + '-' + state.lower() + '.f32'
            raw = frontier(model, result['snapshot'])
            (output / name).write_bytes(raw)
            row = dict(state=state, **result, frontier_file=name, frontier_sha256=sha(output / name),
                       controller_sha256=controller_digest(model, result['snapshot']),
                       native_forward_tokens=after['consumed_tokens'] - before['consumed_tokens'])
            row['action'] = completed_action(result, engine, tokenizer)
            if args.stage == 'resume':
                reference = next(item for item in prepared['actions'] if item['state'] == state)
                require(row['tokens'] == reference['tokens'] and
                        row['controller_sha256'] == reference['controller_sha256'] and
                        raw == (output / reference['frontier_file']).read_bytes(),
                        'Restart changed the same-state consumed continuation')
                record['comparisons'] += 1
            record['actions'].append(row)
            publish(target, record)
        record['status'] = 'PASS'
    except BaseException as exc:
        record['status'] = 'FAIL'
        record['failures'].append(dict(type=type(exc).__name__, message=str(exc)))
        raise
    finally:
        if backend is not None:
            finish_backend(backend, record)
        tracker.finish(record['status'], str(record['failures']) if record['failures'] else None)
        tracker.close()
        publish(target, record)
    require(record['status'] == 'PASS', 'Native cleanup failed')


if __name__ == '__main__':
    main()
