"""Run an isolated shared-physical-K-page corruption probe on real CUDA."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from context_time_travel.real import BranchingModel, FreshOracle
from context_time_travel.numerics import compare_logits
from context_time_travel.store import IntegrityError


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def raw_audit(backend, state, destination):
    destination = Path(destination)
    report = backend._request('audit ' + str(state.handle) + ' ' + json.dumps(str(destination)))
    data = destination.read_bytes()
    if len(data) != report['bytes']:
        raise AssertionError('native physical audit extent mismatch')
    return report, data


def compare(sut, oracle, profile, snapshot, history):
    actual = sut.next_logits(snapshot)
    expected = oracle.forward(history)
    result = compare_logits(actual, expected, atol=profile['atol'], rtol=profile['rtol'])
    return dict(tokens=list(history), max_absolute_error=result.max_absolute_error,
                elementwise=result.pass_elementwise,
                top1_stable=result.top1_stable_within_tolerance,
                native_consumed=sut.consumed(snapshot), oracle_tokens=oracle.last['computed_tokens'])


def run_case(profile, output, *, mutant, mutant_binary):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    original_profile = profile
    if mutant:
        profile = json.loads(json.dumps(original_profile))
        binary = str(Path(mutant_binary).resolve())
        key = next(k for k in profile['native_build']['sha256'] if k.endswith('/strata-branch'))
        profile['native_build']['sha256'][binary] = hashlib.sha256(Path(binary).read_bytes()).hexdigest()
        del profile['native_build']['sha256'][key]
        profile['execution']['sut_command'][0] = binary
        profile['execution_profile_id'] = digest_json(dict(
            build=profile['native_build'], execution=profile['execution'], hardware=profile['hardware']))
        Path(output / 'mutant-profile.json').write_text(json.dumps(profile, indent=2) + '\n')

    backend = BranchingModel.load_backend(profile)
    oracle = FreshOracle(original_profile)
    sut = BranchingModel(output / 'store', backend, profile)
    record = dict(mutant=mutant, execution_profile_id=profile['execution_profile_id'],
                  model_weights_sha256=profile['model_weights_sha256'],
                  oracle_execution_profile_id=original_profile['execution_profile_id'],
                  prefix=list(profile['fixtures']['prefix'][:9]),
                  suffix=list(profile['fixtures']['suffix_a'][:1]),
                  sibling_suffix=list(profile['fixtures']['suffix_b'][:1]),
                  comparisons=[], expected_mutant_refusal=None, witness=None)
    prefix = tuple(record['prefix'])
    suffix = tuple(record['suffix'])
    try:
        root = sut.create(prefix)
        root_state = sut.cache[root].blob
        before_report, before = raw_audit(backend, root_state, output / 'parent-before.bin')
        record['parent_before'] = before_report
        record['comparisons'].append(compare(sut, oracle, original_profile, root, prefix))
        if mutant:
            try:
                sut.feed(root, suffix)
                record['feed'] = 'published'
            except IntegrityError as error:
                record['feed'] = 'integrity-refused'
                record['feed_error'] = str(error)
            after_report, after = raw_audit(backend, root_state, output / 'parent-after.bin')
            record['parent_after'] = after_report
            record['physical_changed_bytes'] = sum(a != b for a, b in zip(before, after))
            record['physical_extent_changed'] = len(before) != len(after)
            backend._log.seek(0)
            stderr = backend._log.read().decode(errors='replace')
            (output / 'native.stderr-so-far').write_text(stderr)
            witness = next((line for line in stderr.splitlines() if line.startswith('M05_SHARED_PAGE_WRITE ')), None)
            record['witness'] = witness
            # The corrupted parent checkpoint must be rejected before it can be used.
            try:
                sut.next_logits(root)
                record['expected_mutant_refusal'] = 'accepted-corrupt-parent'
            except IntegrityError as error:
                record['expected_mutant_refusal'] = 'integrity-refused'
                record['refusal_error'] = str(error)
        else:
            child_a = sut.feed(root, suffix)
            report_a, bytes_a = raw_audit(backend, root_state, output / 'parent-after-a.bin')
            record['parent_after_a'] = report_a
            record['parent_changed_bytes_after_a'] = sum(a != b for a, b in zip(before, bytes_a))
            if len(before) != len(bytes_a) or record['parent_changed_bytes_after_a']:
                raise AssertionError('normal backend changed parent after suffix A')
            record['comparisons'].append(compare(sut, oracle, original_profile, child_a, prefix + suffix))
            suffix_b = tuple(record['sibling_suffix'])
            child_b = sut.feed(root, suffix_b)
            report_b, bytes_b = raw_audit(backend, root_state, output / 'parent-after-b.bin')
            record['parent_after_b'] = report_b
            record['parent_changed_bytes_after_b'] = sum(a != b for a, b in zip(before, bytes_b))
            if len(before) != len(bytes_b) or record['parent_changed_bytes_after_b']:
                raise AssertionError('normal backend changed parent after suffix B')
            record['parent_after'] = report_b
            after = bytes_b
            record['physical_changed_bytes'] = record['parent_changed_bytes_after_b']
            record['physical_extent_changed'] = len(before) != len(after)
            record['comparisons'].append(compare(sut, oracle, original_profile, child_b, prefix + suffix_b))
            record['comparisons'].append(compare(sut, oracle, original_profile, root, prefix))
        record['native_counters'] = dict(backend._refresh_stats())
        record['oracle_requests'] = oracle.requests
        record['oracle_tokens'] = oracle.tokens_computed
        record['status'] = 'MEASURED'
        return record
    finally:
        try:
            sut.evict_all()
        finally:
            oracle.close()
            backend.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--mutant-binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--control-only', action='store_true',
                        help='exercise two real sibling branches with the unmodified pinned backend')
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
    profile = json.loads(args.profile.read_bytes())
    mutant_hash = hashlib.sha256(args.mutant_binary.read_bytes()).hexdigest()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / 'inputs.json').write_text(json.dumps(dict(
        frozen_profile_sha256=hashlib.sha256(args.profile.read_bytes()).hexdigest(),
        mutant_binary=str(args.mutant_binary.resolve()), mutant_binary_sha256=mutant_hash,
        sanitizer_diagnostic='concurrent existing job; overlap must be disclosed'), indent=2) + '\n')
    results = {}
    cases = [('control', False)] if args.control_only else [('control', False), ('mutant', True)]
    for name, mutant in cases:
        start = time.monotonic()
        try:
            results[name] = run_case(profile, out / name, mutant=mutant, mutant_binary=args.mutant_binary)
        except BaseException as error:
            results[name] = dict(status='FAIL', error=repr(error))
            (out / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
            raise
        results[name]['seconds'] = time.monotonic() - start
        (out / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
    control = results['control']
    control_ok = (all(row['elementwise'] and row['top1_stable'] for row in control['comparisons'])
                  and control['physical_changed_bytes'] == 0
                  and control.get('parent_changed_bytes_after_a', 0) == 0
                  and control.get('parent_changed_bytes_after_b', 0) == 0)
    if args.control_only:
        summary = dict(status='NORMAL_BACKEND_CONTROL_PASS' if control_ok else 'NORMAL_BACKEND_CONTROL_FAIL',
                       control_ok=control_ok, mutant_detected=None,
                       max_control_error=max((r['max_absolute_error'] for r in control.get('comparisons', [])), default=None),
                       control_changed_bytes=control.get('physical_changed_bytes'),
                       sibling_a_changed_bytes=control.get('parent_changed_bytes_after_a'),
                       sibling_b_changed_bytes=control.get('parent_changed_bytes_after_b'),
                       results=results)
        (out / 'results.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps(dict(status=summary['status'], control_ok=control_ok,
                              sibling_a_changed_bytes=summary['sibling_a_changed_bytes'],
                              sibling_b_changed_bytes=summary['sibling_b_changed_bytes']), indent=2))
        return 0 if control_ok else 1
    broken = results['mutant']
    mutant_ok = (broken.get('witness') is not None and broken.get('physical_changed_bytes', 0) > 0
                 and broken.get('expected_mutant_refusal') == 'integrity-refused'
                 and broken.get('feed') == 'integrity-refused')
    summary = dict(status='M05_DETECTED' if control_ok and mutant_ok else 'M05_NOT_QUALIFIED',
                   control_ok=control_ok, mutant_detected=mutant_ok,
                   max_control_error=max((r['max_absolute_error'] for r in control.get('comparisons', [])), default=None),
                   control_changed_bytes=control.get('physical_changed_bytes'),
                   mutant_changed_bytes=broken.get('physical_changed_bytes'),
                   witness=broken.get('witness'), results=results)
    (out / 'results.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(dict(status=summary['status'], control_ok=control_ok,
                          mutant_detected=mutant_ok, witness=summary['witness']), indent=2))
    return 0 if control_ok and mutant_ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
