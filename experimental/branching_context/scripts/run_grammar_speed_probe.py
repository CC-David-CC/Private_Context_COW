"""Attribute resident GBNF request cost on a pinned native GPU profile.

Uses the existing completed grammar campaign as a same-operator numerical
reference. Freshly computes the fixture once; never imports its checkpoints.
This is a performance probe, not independent fresh-history gate qualification.
"""
import argparse
from collections import defaultdict
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import statistics
import struct
import sys
import time


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def publish(path, value):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    os.replace(temporary, path)


@contextmanager
def timers(targets):
    """Inclusive wall times; nested measurements must not be summed."""
    records = defaultdict(lambda: dict(calls=0, seconds=0.0))
    originals = []
    for obj, name, label in targets:
        original = getattr(obj, name)
        originals.append((obj, name, original))
        def timed(*args, _original=original, _label=label, **kwargs):
            started = time.perf_counter()
            try:
                return _original(*args, **kwargs)
            finally:
                records[_label]['calls'] += 1
                records[_label]['seconds'] += time.perf_counter() - started
        setattr(obj, name, timed)
    try:
        yield records
    finally:
        for obj, name, original in reversed(originals):
            setattr(obj, name, original)


@contextmanager
def selection_audit(backend, output, label, records):
    """Separate slow audit; never include its full reads in speed medians."""
    original = backend.choose_legal
    def audited(lease, mask):
        values = backend.logits(lease)
        expected = -max((value, -token) for token, value in enumerate(values)
                        if mask[token >> 3] & (1 << (token & 7)))[1]
        actual = original(lease, mask)
        name = '%s-%d' % (label, len(records))
        raw = struct.pack('<' + 'f' * len(values), *values)
        (output / (name + '.f32')).write_bytes(raw)
        (output / (name + '.mask')).write_bytes(mask)
        assert actual == expected, 'Native legal argmax differs from host reference'
        records.append(dict(token=actual, expected=expected, consumed=lease.consumed,
                            frontier_file=name + '.f32', mask_file=name + '.mask',
                            frontier_sha256=hashlib.sha256(raw).hexdigest(),
                            mask_sha256=hashlib.sha256(mask).hexdigest()))
        return actual
    backend.choose_legal = audited
    try:
        yield
    finally:
        backend.choose_legal = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('runtime-root', 'profile', 'fixture', 'reference', 'run-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    runtime = args.runtime_root.resolve(strict=True)
    sys.path[:0] = [str(runtime), str(runtime / 'tools'),
                   str(runtime / 'experimental/branching_context/src'),
                   str(runtime / 'experimental/branching_context/examples'),
                   str(runtime / 'experimental/branching_context/scripts')]
    from run_cow_client_recovery import ProgressTracker
    from grammar_first_agent import load_grammars
    from context_time_travel import grammar_first as gf
    from context_time_travel.real import BranchingModel
    from context_time_travel.strata_backend import StrataBackend, _validate_execution
    from context_time_travel.store import _json_bytes
    from tools.strata_tokenizer import Tokenizer

    output = args.run_dir.absolute()
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    tracker = ProgressTracker(output, 'grammar-speed').start()
    profile = read(args.profile)
    for key in list(os.environ):
        if key.startswith(('STRATA_', 'CUDA_', 'NVIDIA_', 'GGML_')):
            del os.environ[key]
    os.environ.update(profile['execution']['environment'])
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ['CTT_NATIVE_TRACE'] = str(output / 'native.jsonl')
    _validate_execution(profile)
    reference = read(args.reference / 'prepare.json')
    assert reference['status'] == 'PASS'
    assert profile['model_contract_id'] == read(args.reference.parent / 'profile-candidate-v2/profile.json')['model_contract_id']
    record = dict(status='RUNNING', python_pid=os.getpid(), profile_sha256=sha(args.profile),
                  script_sha256=sha(__file__), fixture_sha256=sha(args.fixture), rows=[],
                  reference=str(args.reference), failures=[], comparisons=0,
                  timing_scope='inclusive wall time; nested values overlap; no GPU-event timing',
                  qualification='Resident static-GBNF cost attribution; no cumulative gate claim')
    backend = None
    target = output / 'result.json'
    publish(target, record)
    try:
        tracker.phase('tokenizer-and-grammar')
        tokenizer = Tokenizer.from_gguf(profile['model_path'])
        engine = gf.GrammarEngine(tokenizer, load_grammars(), eos_tokens=profile['execution']['eos_tokens'])
        tracker.phase('native-start')
        started = time.monotonic()
        backend = StrataBackend(profile)
        record['startup_seconds'] = time.monotonic() - started
        record['native_pid'] = backend.process.pid
        tracker.native_started(backend.process.pid)
        model = BranchingModel(output / 'store', backend, profile)
        tokens = read(args.fixture)['root_token_ids']
        assert len(tokens) == reference['root_tokens']
        tracker.phase('fresh-prefix')
        before = backend.work_counters()
        started = time.monotonic()
        root = model.create(tokens)
        record['prefix_seconds'] = time.monotonic() - started
        record['prefix_counters_before'] = before
        record['prefix_counters_after'] = backend.work_counters()
        record['root_tokens'] = len(tokens)
        root_bytes = model.snapshot_bytes(root)
        targets = [(gf.GrammarSession, 'choose', 'grammar.choose'),
                   (gf.GrammarSession, 'consume', 'grammar.consume_checked'),
                   (gf, 'validate_controller', 'grammar.validate_controller'),
                   (backend, 'logits', 'backend.logits'),
                   (backend, 'consume_generated', 'backend.consume_generated'),
                   (backend, 'export', 'backend.export'),
                   (model, '_completed_schedule', 'model.completed_schedule'),
                   (model, '_save', 'model.save'),
                   (model, '_publish_feed', 'model.publish_feed')]
        if hasattr(gf.GrammarSession, '_consume_selected'):
            targets.append((gf.GrammarSession, '_consume_selected', 'grammar.consume_owned'))
        if hasattr(gf.GrammarSession, '_choose_masked'):
            targets.append((gf.GrammarSession, '_choose_masked', 'grammar.choose_masked'))
        if hasattr(backend, 'choose_legal'):
            targets.append((backend, 'choose_legal', 'backend.choose_legal'))
        phases = [('warmup', 1), ('measured', 3), ('instrumented', 1)]
        if profile['execution'].get('grammar_selection_protocol'):
            phases.append(('audit', 1))
            record['selection_audit'] = []
        for phase, repetitions in phases:
            for repetition in range(repetitions):
                for state in ('READ', 'EDIT', 'TEST', 'FINISH'):
                    tracker.phase('%s-%s-%d' % (phase, state, repetition))
                    before = backend.work_counters()
                    with timers(targets if phase == 'instrumented' else []) as costs:
                        if phase == 'audit':
                            with selection_audit(backend, output, state.lower(), record['selection_audit']):
                                result = model.run_state_request(root, (), 256, engine=engine,
                                                                state=state, stream=state.lower())
                        else:
                            result = model.run_state_request(root, (), 256, engine=engine,
                                                            state=state, stream=state.lower())
                    after = backend.work_counters()
                    expected = next(row for row in reference['actions'] if row['state'] == state)
                    raw = struct.pack('<' + 'f' * backend.vocab_size, *model.next_logits(result['snapshot']))
                    assert result['tokens'] == expected['tokens'], 'Changed tokens: ' + state
                    assert hashlib.sha256(_json_bytes(model.controller(result['snapshot']))).hexdigest() == expected['controller_sha256'], 'Changed controller'
                    assert raw == (args.reference / expected['frontier_file']).read_bytes(), 'Changed frontier'
                    assert model.snapshot_bytes(root) == root_bytes, 'Mutated parent'
                    assert after['consumed_tokens'] - before['consumed_tokens'] == len(result['tokens']), 'Unexpected prefix work'
                    assert result['stopped'] and result['grammar_complete']
                    row = dict(phase=phase, repetition=repetition, state=state, **result,
                               output_count=len(result['tokens']),
                               generation_tps=len(result['tokens']) / result['generation_seconds'],
                               costs=dict(costs), counters_before=before, counters_after=after,
                               frontier_sha256=hashlib.sha256(raw).hexdigest())
                    record['rows'].append(row)
                    record['comparisons'] += 1
                    publish(target, record)
        record['medians'] = {state: {
            key: statistics.median(row[key] for row in record['rows']
                if row['state'] == state and row['phase'] == 'measured')
            for key in ('generation_seconds', 'request_seconds', 'generation_tps')}
            for state in ('READ', 'EDIT', 'TEST', 'FINISH')}
        record['numerical_max_error'] = 0
        record['status'] = 'PASS'
    except BaseException as exc:
        record['status'] = 'FAIL'
        record['failures'].append(dict(type=type(exc).__name__, message=str(exc)))
        raise
    finally:
        if backend is not None:
            try:
                backend.close()
            finally:
                record['native_exit_code'] = backend.process.returncode
                record['native_pid_absent'] = not Path('/proc', str(backend.process.pid)).exists()
                if record['native_exit_code'] != 0 or not record['native_pid_absent']:
                    record['status'] = 'FAIL'
        tracker.close()
        publish(target, record)


if __name__ == '__main__':
    main()
