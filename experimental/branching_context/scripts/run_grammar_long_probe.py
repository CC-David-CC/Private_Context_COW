"""Two real code questions, one prepared prefix; audited and timed separately.

The standalone references use matching native operator boundaries without
checkpoints. This bounded experiment does not qualify original G1-G5 gates.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import os
import statistics
import struct
import sys
import time


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def publish(path, record):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def sample_counts(actual_count):
    if type(actual_count) is not int or actual_count <= 0:
        raise ValueError('Need a positive consumed output count')
    return sorted({n for n in (1, 16, 64, actual_count) if n <= actual_count})


def require_terminal_result(record):
    if (record.get('status') != 'PASS' or type(record.get('native_exit_code')) is not int
            or record['native_exit_code'] != 0 or record.get('native_pid_absent') is not True
            or record.get('failures') or record.get('native_cleanup_errors')):
        raise AssertionError('Stage did not finish with successful native cleanup')


def checked_sample(directory, descriptor, vocab):
    relative = Path(descriptor['path'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Reference sample path escapes its result directory')
    path = directory / relative
    if path.is_symlink() or not path.resolve(strict=True).is_relative_to(directory.resolve(strict=True)):
        raise ValueError('Reference sample must be a contained ordinary file')
    raw = path.read_bytes()
    if (len(raw) != vocab * 4 or descriptor['bytes'] != len(raw)
            or hashlib.sha256(raw).hexdigest() != descriptor['sha256']):
        raise ValueError('Reference sample hash or full-vocabulary extent differs')
    values = struct.unpack('<' + 'f' * vocab, raw)
    import math
    if not all(map(math.isfinite, values)):
        raise ValueError('Nonfinite reference sample')
    return raw


def compare_frontier(raw, expected, label):
    if len(raw) != len(expected):
        raise AssertionError('Frontier extent differs: ' + label)
    if raw != expected:
        left = struct.unpack('<' + 'f' * (len(raw) // 4), raw)
        right = struct.unpack('<' + 'f' * (len(raw) // 4), expected)
        error = max(abs(a - b) for a, b in zip(left, right))
        raise AssertionError('Matching-operator frontier mismatch %s; max_abs_error=%r' % (label, error))


def validate_answer(result, controller, history, reference, expected_history):
    for record in (result, reference):
        if (not isinstance(record['tokens'], list)
                or any(type(token) is not int or token < 0 for token in record['tokens'])
                or type(record['stopped']) is not bool or type(record['grammar_complete']) is not bool):
            raise AssertionError('Token IDs and scope flags must have their exact declared types')
    if any(type(token) is not int or token < 0 for token in (*history, *expected_history)):
        raise AssertionError('Canonical/reference histories require exact integer token IDs')
    canonical = lambda value: json.dumps(value, sort_keys=True, separators=(',', ':'),
                                         ensure_ascii=False, allow_nan=False)
    if (result['tokens'] != reference['tokens'] or canonical(controller) != canonical(reference['controller'])
            or tuple(history) != tuple(expected_history) + tuple(reference['tokens'])
            or result['stopped'] != reference['stopped']
            or result['grammar_complete'] != reference['grammar_complete']):
        raise AssertionError('Branch tokens, controller, canonical history or scope status differs from reference')


@contextmanager
def capture_samples(backend, output, label, suffix, checkpoints, samples):
    """Read consumed frontiers only during the audit repetition."""
    original_feed, original_decode = backend.consume, backend.consume_generated
    generated = 0
    def capture(lease, name):
        values = backend.logits(lease)
        raw = struct.pack('<' + 'f' * len(values), *values)
        path = output / (label + '-' + name + '.f32')
        path.write_bytes(raw)
        samples[name] = dict(path=path.name, sha256=sha(path), bytes=len(raw), consumed=lease.consumed)
    def feed(lease, tokens):
        if tuple(tokens) != tuple(suffix) or 'suffix' in samples:
            raise AssertionError('Audit request unexpectedly replays or changes its suffix')
        original_feed(lease, tokens)
        capture(lease, 'suffix')
    def decode(lease, tokens):
        nonlocal generated
        original_decode(lease, tokens)
        generated += len(tokens)
        if generated in checkpoints:
            capture(lease, 'generated-' + str(generated))
    backend.consume, backend.consume_generated = feed, decode
    try:
        yield
    finally:
        backend.consume, backend.consume_generated = original_feed, original_decode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('campaign', 'candidate'), required=True)
    for name in ('runtime-root', 'profile', 'fixture', 'grammar', 'run-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--max-outputs', type=int, default=128)
    args = parser.parse_args()
    if args.max_outputs <= 0:
        raise ValueError('Output limit must be positive')
    runtime = args.runtime_root.resolve(strict=True)
    sys.path[:0] = [str(runtime), str(runtime / 'tools'),
                   str(runtime / 'experimental/branching_context/src'),
                   str(runtime / 'experimental/branching_context/scripts')]
    # Observer helpers are independent from the frozen execution runtime.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from grammar_long_reference import validate_fixture
    from run_grammar_first_probe import run_stage, finish_backend
    from run_cow_client_recovery import ProgressTracker
    from context_time_travel.grammar_first import GrammarEngine
    from context_time_travel.real import BranchingModel
    from context_time_travel.strata_backend import StrataBackend, _validate_execution
    from context_time_travel.store import _json_bytes
    from tools.strata_tokenizer import Tokenizer
    profile = read(args.profile)
    case = read(args.fixture)
    fixtures = {branch: validate_fixture(case, branch) for branch in ('A', 'B')}
    for key in list(os.environ):
        if key.startswith(('STRATA_', 'CUDA_', 'NVIDIA_', 'GGML_')):
            del os.environ[key]
    os.environ.update(profile['execution']['environment'])
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    _validate_execution(profile)
    output = args.run_dir.absolute()
    if args.stage == 'campaign':
        output.mkdir(mode=0o700, parents=True, exist_ok=False)
        processes = []
        common = ['--runtime-root', str(runtime), '--profile', str(args.profile.resolve()),
                  '--fixture', str(args.fixture.resolve()), '--grammar', str(args.grammar.resolve()),
                  '--max-outputs', str(args.max_outputs)]
        reference_script = Path(__file__).with_name('grammar_long_reference.py')
        for branch in ('A', 'B'):
            name = 'reference-' + branch
            command = [sys.executable, '-B', str(reference_script), *common,
                       '--branch', branch, '--stream', 'review-' + branch,
                       '--run-dir', str(output / name)]
            receipt = run_stage(command, output / (name + '.stdout'), output / (name + '.stderr'))
            processes.append(dict(stage=name, **receipt))
            healthy = (receipt['exit_code'] == 0 and receipt['pid_absent'] and not receipt['timed_out']
                       and not receipt['cleanup_needed'] and not receipt['live_group_pids'])
            if healthy:
                try:
                    require_terminal_result(read(output / name / 'result.json'))
                except (OSError, ValueError, AssertionError) as exc:
                    healthy = False
                    processes[-1]['result_validation_error'] = str(exc)
            publish(output / 'campaign.json', dict(status='RUNNING' if healthy else 'FAIL', processes=processes))
            if not healthy:
                raise AssertionError('Fresh reference failed or leaked descendants')
        command = [sys.executable, '-B', str(Path(__file__).resolve()), '--stage', 'candidate',
                   *common, '--run-dir', str(output)]
        receipt = run_stage(command, output / 'candidate.stdout', output / 'candidate.stderr')
        processes.append(dict(stage='candidate', **receipt))
        healthy = (receipt['exit_code'] == 0 and receipt['pid_absent'] and not receipt['timed_out']
                   and not receipt['cleanup_needed'] and not receipt['live_group_pids'])
        if healthy:
            try:
                require_terminal_result(read(output / 'candidate/result.json'))
            except (OSError, ValueError, AssertionError) as exc:
                healthy = False
                processes[-1]['result_validation_error'] = str(exc)
        publish(output / 'campaign.json', dict(status='PASS' if healthy else 'FAIL', processes=processes))
        if not healthy:
            raise AssertionError('Candidate failed or leaked descendants')
        return
    candidate = output / 'candidate'
    candidate.mkdir(mode=0o700, exist_ok=False)
    tracker = ProgressTracker(candidate, 'grammar-long').start()
    os.environ['CTT_NATIVE_TRACE'] = str(candidate / 'native.jsonl')
    record = dict(status='RUNNING', python_pid=os.getpid(), rows=[], failures=[], comparisons=0,
        profile_sha256=sha(args.profile), fixture_sha256=sha(args.fixture), grammar_sha256=sha(args.grammar),
        script_sha256=sha(__file__), controller_scope='greedy REVIEW JSON string; no tool execution',
        numerical_max_error=0, frontier_sampling='prefix, suffix, generated1/16/64/final in audit; final only for timings')
    target = candidate / 'result.json'
    publish(target, record)
    backend = None
    try:
        references = {branch: read(output / ('reference-' + branch) / 'result.json') for branch in ('A', 'B')}
        for reference in references.values():
            require_terminal_result(reference)
            if (Path('/proc', str(reference['native_pid'])).exists()
                    or Path('/proc', str(reference['python_pid'])).exists()):
                raise AssertionError('Fresh reference processes have not exited cleanly')
        tokenizer = Tokenizer.from_gguf(profile['model_path'])
        engine = GrammarEngine(tokenizer, {'REVIEW': {'format': 'gbnf', 'source': args.grammar.read_text()}},
                               eos_tokens=profile['execution']['eos_tokens'])
        tracker.phase('native-start')
        started = time.monotonic()
        backend = StrataBackend(profile)
        record['startup_seconds'] = time.monotonic() - started
        record['native_pid'] = backend.process.pid
        tracker.native_started(backend.process.pid)
        model = BranchingModel(candidate / 'store', backend, profile)
        prefix = tuple(fixtures['A'][0])
        assert prefix == tuple(fixtures['B'][0])
        tracker.phase('fresh-prefix')
        before = backend.work_counters()
        started = time.monotonic()
        root = model.create(prefix)
        record['prefix_seconds'] = time.monotonic() - started
        record['prefix_counters_before'], record['prefix_counters_after'] = before, backend.work_counters()
        assert record['prefix_counters_after']['forward_tokens'] - before['forward_tokens'] == len(prefix)
        record['root_tokens'] = len(prefix)
        original = model.snapshot_bytes(root)
        root_image = model.cache[root].blob
        record['root_native_handle'] = root_image.handle
        tracker.phase('root-byte-audit-before')
        record['parent_audit_before'] = backend.audit(root_image)
        root_values = model.cache[root].frontier
        root_raw = struct.pack('<' + 'f' * len(root_values), *root_values)
        (candidate / 'prefix.f32').write_bytes(root_raw)
        heads = {branch: model.store.fork('Review' + branch, root) for branch in ('A', 'B')}
        for phase, repetitions in [('audit', 1), ('warmup', 1), ('measured', 3)]:
            for repetition in range(repetitions):
                for branch in ('A', 'B'):
                    tracker.phase('%s-%s-%d' % (phase, branch, repetition))
                    reference = references[branch]
                    suffix = tuple(fixtures[branch][1])
                    expected_history = prefix + suffix
                    head = model.store.head(heads[branch].name)
                    reset_started = time.monotonic()
                    reset = model.store.move(head.name, head.revision, root, '%s-%d-%s-reset' % (phase, repetition, branch))
                    reset_seconds = time.monotonic() - reset_started
                    samples = {'prefix': dict(path='prefix.f32', sha256=sha(candidate / 'prefix.f32'),
                                              bytes=len(root_raw), consumed=len(prefix))}
                    before = backend.work_counters()
                    consumed_times = []
                    request_started = time.monotonic()
                    def on_consumed(token):
                        consumed_times.append(time.monotonic())
                    if phase == 'audit':
                        with capture_samples(backend, candidate, branch, suffix,
                                             sample_counts(len(reference['tokens'])), samples):
                            result = model.run_state_request(reset.tip, suffix, args.max_outputs,
                                engine=engine, state='REVIEW', stream='review-' + branch, on_token=on_consumed)
                    else:
                        result = model.run_state_request(reset.tip, suffix, args.max_outputs,
                            engine=engine, state='REVIEW', stream='review-' + branch, on_token=on_consumed)
                    after = backend.work_counters()
                    controller = model.controller(result['snapshot'])
                    validate_answer(result, controller, model.history(result['snapshot']), reference, expected_history)
                    assert after['forward_tokens'] - before['forward_tokens'] == len(suffix) + len(result['tokens'])
                    assert after['prefill_tokens'] - before['prefill_tokens'] == len(suffix)
                    assert after['decode_timed_tokens'] - before['decode_timed_tokens'] == len(result['tokens'])
                    assert after['grammar_selection_calls'] - before['grammar_selection_calls'] == len(result['tokens'])
                    assert after['loaded_file_bytes'] == before['loaded_file_bytes'], 'Unexpected disk restoration'
                    assert after['captures'] - before['captures'] == 1
                    assert model.cache[root].blob is root_image, 'Prepared prefix checkpoint was replaced'
                    assert model.snapshot_bytes(root) == original
                    final_values = model.cache[result['snapshot']].frontier
                    final_raw = struct.pack('<' + 'f' * len(final_values), *final_values)
                    final_name = 'generated-' + str(len(result['tokens']))
                    if phase == 'audit':
                        path = candidate / (branch + '-' + final_name + '.f32')
                        path.write_bytes(final_raw)
                        samples[final_name] = dict(path=path.name, bytes=len(final_raw), consumed=len(expected_history)+len(result['tokens']), sha256=sha(path))
                        expected_keys = {'prefix', 'suffix'} | {'generated-' + str(n) for n in sample_counts(len(result['tokens']))}
                        assert set(samples) == expected_keys
                        for name in expected_keys:
                            expected = checked_sample(output / ('reference-' + branch), reference['samples'][name], backend.vocab_size)
                            assert samples[name]['consumed'] == reference['samples'][name]['consumed']
                            compare_frontier((candidate / samples[name]['path']).read_bytes(), expected, branch + ':' + name)
                            record['comparisons'] += 1
                    else:
                        expected = checked_sample(output / ('reference-' + branch), reference['samples'][final_name], backend.vocab_size)
                        compare_frontier(final_raw, expected, branch + ':' + phase)
                        record['comparisons'] += 1
                    publish_started = time.monotonic()
                    move = model.store.move(reset.name, reset.revision, result['snapshot'],
                                            '%s-%d-%s-publish' % (phase, repetition, branch))
                    publish_seconds = time.monotonic() - publish_started
                    assert len(consumed_times) == len(result['tokens'])
                    intervals = [b-a for a, b in zip(consumed_times, consumed_times[1:])]
                    row = dict(phase=phase, repetition=repetition, branch=branch, **result,
                        suffix_tokens=len(suffix), output_count=len(result['tokens']),
                        generation_tps=len(result['tokens'])/result['generation_seconds'],
                        complete_request_tps=len(result['tokens'])/result['request_seconds'],
                        branch_reset_seconds=reset_seconds, branch_publish_seconds=publish_seconds,
                        workflow_seconds=reset_seconds+result['request_seconds']+publish_seconds,
                        workflow_tps=len(result['tokens'])/(reset_seconds+result['request_seconds']+publish_seconds),
                        time_to_first_consumed_token_seconds=consumed_times[0]-request_started,
                        inter_consumed_token_seconds=intervals,
                        median_inter_consumed_token_seconds=statistics.median(intervals) if intervals else None,
                        max_inter_consumed_token_seconds=max(intervals) if intervals else None,
                        controller_sha256=hashlib.sha256(_json_bytes(controller)).hexdigest(),
                        native_counters_before=before, native_counters_after=after,
                        branch_revision=move.revision, samples=samples if phase == 'audit' else {})
                    record['rows'].append(row)
                    publish(target, record)
        tracker.phase('root-byte-audit-after')
        record['parent_audit_after'] = backend.audit(root_image)
        assert record['parent_audit_before']['sha256'] == record['parent_audit_after']['sha256'], 'Parent numerical bytes changed'
        record['final_counters'] = backend.work_counters()
        record['cache_entries'] = len(model.cache)
        record['medians'] = {branch: {key: statistics.median(row[key] for row in record['rows']
            if row['phase'] == 'measured' and row['branch'] == branch)
            for key in ('generation_seconds', 'generation_tps', 'request_seconds', 'complete_request_tps',
                        'branch_reset_seconds', 'branch_publish_seconds', 'workflow_seconds', 'workflow_tps',
                        'time_to_first_consumed_token_seconds', 'median_inter_consumed_token_seconds',
                        'max_inter_consumed_token_seconds')}
            for branch in ('A', 'B')}
        record['status'] = 'PASS'
        tracker.finish('PASS')
    except BaseException as exc:
        record['status'] = 'FAIL'
        record['failures'].append(dict(type=type(exc).__name__, message=str(exc)))
        tracker.finish('FAIL', str(exc))
        raise
    finally:
        if backend is not None:
            finish_backend(backend, record)
        tracker.close()
        publish(target, record)
    require_terminal_result(record)


if __name__ == '__main__':
    main()
