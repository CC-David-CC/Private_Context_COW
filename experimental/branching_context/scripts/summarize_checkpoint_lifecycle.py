"""Summarize measured-request observer groups without adding nested timings."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostics', type=Path, required=True)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    diagnostic_bytes = args.diagnostics.read_bytes()
    result_bytes = args.result.read_bytes()
    diagnostic = json.loads(diagnostic_bytes)
    result = json.loads(result_bytes)
    measurements = diagnostic['measurements']
    if (diagnostic['status'] != 'RETURNED' or diagnostic['exit_code'] != 0
            or result['status'] != 'PASS' or result['native_exit_codes'] != [0]
            or measurements['instrumentation_errors'] or measurements['dropped_events']):
        raise ValueError('Requires successful, complete observer and native results')
    groups = [v for v in measurements['phase_stages'].values()
              if v['phase'] is not None and v['phase']['phase'] == 'measured_request']
    expected = {(i, j) for i in range(3) for j in range(3)}
    if (len(groups) != 9 or len(result['samples']) != 9
            or {(v['phase']['root_index'], v['phase']['repeat']) for v in groups} != expected
            or {(s['root_index'], s['repeat']) for s in result['samples']} != expected):
        raise ValueError('Expected the full nine-request root/repetition matrix')
    stages = []
    for name in sorted(set().union(*(v['stages'] for v in groups))):
        values = [v['stages'].get(name, dict(calls=0, errors=0, bytes=0,
                                          inclusive_ns=0, exclusive_ns=0)) for v in groups]
        if any(v['errors'] or not 0 <= v['exclusive_ns'] <= v['inclusive_ns'] for v in values):
            raise ValueError('Unexpected stage error or invalid timing: ' + name)
        stages.append(dict(stage=name,
            calls=sum(v['calls'] for v in values),
            median_calls_per_request=statistics.median(v['calls'] for v in values),
            observed_bytes_total=sum(v['bytes'] for v in values),
            byte_scope='UTF-8 command text, not payload transfer' if name.startswith('native_request_')
                else 'Argument/result extents where observable; zero does not prove no traffic',
            median_inclusive_seconds=statistics.median(v['inclusive_ns'] for v in values)/1e9,
            median_same_thread_exclusive_seconds=statistics.median(v['exclusive_ns'] for v in values)/1e9))
    report = dict(status='PASS_MEASURED_REQUEST_ATTRIBUTION_SUMMARY',
        scope='Nine original Q4 resident requests; observational timing, not a speed improvement claim',
        result_sha256=hashlib.sha256(result_bytes).hexdigest(),
        diagnostics_sha256=hashlib.sha256(diagnostic_bytes).hexdigest(),
        profile=result['profile_identity']['execution_profile_id'],
        warmups_excluded=len(result['warmup_samples']), measured_requests=len(groups),
        output_tokens=sum(len(s['answer']['tokens']) for s in result['samples']),
        native_exit_codes=result['native_exit_codes'],
        events=len(measurements['events']), dropped_events=measurements['dropped_events'],
        instrumentation_errors=measurements['instrumentation_errors'],
        timing_semantics='Host wall intervals; inclusive contains nested calls. Exclusive subtracts observed same-thread children only. Do not sum across threads as a critical path.',
        observer_overhead_measured=False, new_full_logit_comparisons=0,
        numerical_max_error=None, stages=stages)
    if args.output.exists():
        raise FileExistsError(str(args.output))
    args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'stages'}, sort_keys=True))


if __name__ == '__main__':
    main()
