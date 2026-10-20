"""Summarize executed browser requests using one server monotonic clock.

The total ends at durable publication, not the last streamed token. This is
observational latency evidence, not a numerical gate or controlled speedup test.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def measured_seconds(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError('Missing positive measured duration')
    return value


def collect(name, report_path, trace_path):
    report_path, trace_path = Path(report_path), Path(trace_path)
    report, trace = (json.loads(path.read_bytes()) for path in (report_path, trace_path))
    if report.get('status') != 'PASS' or report.get('inference_kind') != 'real':
        raise ValueError('Expected a passing real browser report')
    events = trace.get('events')
    if (trace.get('overflowed') is not False or not isinstance(events, list)
            or trace.get('event_count') != len(events)):
        raise ValueError('Incomplete server trace')
    for index, event in enumerate(events):
        if (event.get('sequence') != index or type(event.get('monotonic_ns')) is not int
                or event.get('process_id') != trace.get('process_id')
                or event.get('phase_id') != trace.get('phase_id')):
            raise ValueError('Mixed or unordered trace identity')
        if index and event['monotonic_ns'] < events[index-1]['monotonic_ns']:
            raise ValueError('Server timestamps are not monotonic')
    identity = [report['server_model_contract_id_from_durable_receipts'],
                report['server_execution_profile_id_from_durable_receipts']]

    def request_span(operation, branch, source, result):
        key = identity + [source]
        joined = [row for row in events if row.get('key') == key
                  and row.get('operation_id') == operation and row.get('branch') == branch]
        received = [row for row in joined if row['event'] == 'http_request_received']
        durable = [row for row in joined if row['event'] == 'http_request_durable']
        if len(received) != 1 or len(durable) != 1 or durable[0].get('result_snapshot') != result:
            raise ValueError('Request does not join a unique durable trace interval')
        if received[0]['sequence'] >= durable[0]['sequence']:
            raise ValueError('Durable publication precedes request arrival')
        return measured_seconds((durable[0]['monotonic_ns']-received[0]['monotonic_ns'])/1e9)

    contexts = report['reusable_context_report']['contexts']
    rows = []
    for index, context in enumerate(contexts, 1):
        nested = context['report']
        for label, key in [('A', 'ask_a_request'), ('B', 'ask_b_request')]:
            request = nested[key]
            answer, receipt = request['answer'], request['receipt']
            tokens, counters = len(answer['tokens']), answer['counter_deltas']
            suffix = context[label+'_suffix_tokens']
            if (tokens <= 0 or counters.get('forward_tokens') != suffix+tokens
                    or counters.get('prefill_tokens') != suffix):
                raise ValueError('Request lacks exact suffix/output work accounting')
            total = request_span(request['operation_id'], receipt['branch']['name'],
                                 nested['root_snapshot'], receipt['snapshot'])
            native = measured_seconds(answer['request_seconds'])
            generation = measured_seconds(answer['generation_seconds'])
            capture = measured_seconds(answer['capture_seconds'])
            residual = total-native-capture
            if native < generation or residual < 0:
                raise ValueError('Measured phase boundaries overlap or exceed the server interval')
            rows.append(dict(case=name, context=context['name'], label=f'{index}{label}',
                operation_id=request['operation_id'], generated_tokens=tokens,
                root_tokens=context['root_token_count'], suffix_tokens=suffix,
                server_arrival_to_durable_seconds=total,
                generation_seconds=generation, other_native_request_seconds=native-generation,
                checkpoint_capture_seconds=capture, other_boundary_seconds=residual,
                generation_tps=tokens/generation, durable_request_output_tps=tokens/total,
                source_native_loaded_bytes=counters.get('loaded_file_bytes')))
    revisit = report['reusable_context_report']['revisited_oldest_branch']
    total = request_span(revisit['request']['operation_id'], revisit['request']['branch'],
                         revisit['source_tip'], revisit['new_tip'])
    count = revisit['generated_tokens']
    if (revisit['counter_deltas'].get('forward_tokens') != count
            or revisit['counter_deltas'].get('prefill_tokens') != 0 or count <= 0):
        raise ValueError('Revisit lacks exact zero-prefix-forward work')
    return dict(name=name, model_contract_id=identity[0], execution_profile_id=identity[1],
        report_sha256=sha(report_path), trace_sha256=sha(trace_path), requests=rows,
        summary=dict(requests=len(rows),
            generation_tps_median=statistics.median(row['generation_tps'] for row in rows),
            durable_request_output_tps_median=statistics.median(row['durable_request_output_tps'] for row in rows),
            server_seconds_median=statistics.median(row['server_arrival_to_durable_seconds'] for row in rows),
            server_seconds_range=[min(row['server_arrival_to_durable_seconds'] for row in rows),
                                  max(row['server_arrival_to_durable_seconds'] for row in rows)]),
        disk_revisit=dict(generated_tokens=count, server_arrival_to_durable_seconds=total,
            durable_request_output_tps=count/total, prefix_forward_tokens=0,
            native_loaded_bytes=revisit['counter_deltas'].get('loaded_file_bytes')))


def plot(cases, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(cases), 1, figsize=(10.8, 3.5*len(cases)), squeeze=False)
    phases = [('generation_seconds', 'Generation', '#257a7a'),
              ('other_native_request_seconds', 'Other request computation', '#67b8a4'),
              ('checkpoint_capture_seconds', 'Checkpoint capture', '#d6a343'),
              ('other_boundary_seconds', 'Other server/publication work', '#a7b1bf')]
    for axis, case in zip(axes[:, 0], cases):
        rows = case['requests']
        bottom = [0.0]*len(rows)
        for field, label, color in phases:
            values = [row[field] for row in rows]
            axis.bar(range(len(rows)), values, bottom=bottom, label=label, color=color)
            bottom = [a+b for a, b in zip(bottom, values)]
        for index, row in enumerate(rows):
            axis.text(index, bottom[index]+.08,
                f"{row['server_arrival_to_durable_seconds']:.2f}s / {row['durable_request_output_tps']:.1f} TPS",
                ha='center', va='bottom', fontsize=9)
        axis.set_xticks(range(len(rows)), [row['label'] for row in rows])
        axis.set_ylim(0, max(bottom)*1.2)
        axis.set_ylabel('Server request seconds')
        axis.set_title(case['name']+' — each question produces 128 tokens', loc='left', fontweight='bold')
        axis.spines[['top', 'right']].set_visible(False)
        axis.grid(axis='y', alpha=.15)
        axis.set_axisbelow(True)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False, fontsize=9)
    fig.suptitle('COW browser requests: arrival through durable publication', fontsize=15, fontweight='bold')
    fig.text(.5, .07, 'Executed ~8K source contexts + ~512-token questions; one request per GPU.\n'
        'One run per distinct question. Startup and initial root preparation excluded.\n'
        'Server-clock intervals; no client/network TTFT or token-spacing measurement.',
        ha='center', fontsize=9, color='#495463')
    fig.tight_layout(rect=(0, .18, 1, .94))
    fig.savefig(output/'latency.png', dpi=170)
    fig.savefig(output/'latency.svg')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', nargs=3, action='append', required=True,
                        metavar=('NAME', 'BROWSER_REPORT', 'POOL_TRACE'))
    parser.add_argument('--output-directory', type=Path, required=True)
    args = parser.parse_args()
    cases = [collect(*case) for case in args.case]
    output = args.output_directory
    output.mkdir(parents=True, exist_ok=False)
    result = dict(status='MEASURED_SERVER_LATENCY', producer_sha256=sha(Path(__file__)),
        cases=cases, numerical_gate='SEPARATE',
        limitations=['One executed request per distinct question, no repeated benchmark.',
            'Server arrival-to-durable interval excludes startup/root preparation and final network delivery.',
            'Generation TPS excludes prefill, capture and publication; durable request TPS includes them.',
            'Residual work is unattributed; timings do not prove that all residual time is disk writing.',
            'No matched ordinary-serving overhead percentage, TTFT or TPOT is measured here.'])
    (output/'measurements.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8', newline='\n')
    plot(cases, output)
    print(json.dumps({case['name']:case['summary'] for case in cases}, indent=2))


if __name__ == '__main__': main()
