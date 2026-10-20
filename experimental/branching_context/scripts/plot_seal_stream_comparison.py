"""Plot the completed four-arm Q4 CLI comparison, retaining all input hashes."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from summarize_client_campaign import summarize


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    inputs, arms, hash_coverage = {}, {}, set()
    fields = ('steady_received_tps', 'time_to_first_token_seconds',
              'median_tpot_seconds', 'p95_tpot_seconds', 'request_seconds',
              'durable_request_seconds')
    cases = {}
    for version, case in (('old', 'case-old4ac-8k128'), ('new', 'case-f94-8k128')):
        path = args.results / case / 'result.json'
        if not path.exists():
            path = args.results / (case + '-result.json')
        record = json.loads(path.read_bytes())
        if (record['status'] != 'PASS' or record['comparisons'] != 13
                or record['numerical_max_error'] != 0
                or record['fixture_sizes'] != dict(prefix=8192,
                    suffixes=dict(A=527, B=527), output_limit=128)):
            raise ValueError('Require the declared executed numerical case: ' + case)
        cases[version] = record
        inputs[str(path.relative_to(args.results))] = digest(path)
    for mode in ('resident', 'disk'):
        for version in ('old', 'new'):
            name = f'client-{version}-{mode}-v2'
            directory = args.results / name
            checked = summarize(directory)
            client = json.loads((directory / 'client.json').read_bytes())
            worker = json.loads((directory / 'worker.json').read_bytes())
            if checked['profile_id'] != cases[version]['profile_id']:
                raise ValueError('Timing/correctness profile identity differs')
            rows = [row for row in client['requests'].values() if not row['warmup']]
            if (len(rows) != 6 or len(client['requests']) != 8
                    or any(row['outputs'] != 128 or row['mode'] != mode for row in rows)):
                raise ValueError('Require two warmups and six measured 128-output requests')
            work = [row for row in worker['requests'] if not row['warmup']]
            for row in work:
                counters = row['counter_deltas']
                if (counters['forward_tokens'] != 655 or counters['prefill_tokens'] != 527
                        or counters['prefill_chunks'] != 1
                        or counters['captures'] != 1 or counters['restores'] != 1):
                    raise ValueError('Unexpected suffix/decode work')
                hash_coverage.add(counters['integrity_device_hashed_bytes'])
            arms[name] = dict(summary=checked,
                medians={field: statistics.median(row[field] for row in rows) for field in fields},
                individual_measurements=rows,
                median_hashed_bytes=statistics.median(row['counter_deltas']['integrity_device_hashed_bytes'] for row in work),
                median_digest_copy_bytes=statistics.median(row['counter_deltas']['integrity_digest_copy_bytes'] for row in work))
            for filename in ('client.json', 'worker.json', 'resource.txt'):
                path = directory / filename
                if filename == 'resource.txt' and not path.exists():
                    path = args.results / 'runs' / name / filename
                inputs[str(path.relative_to(args.results))] = digest(path)
    if len(hash_coverage) != 1 or next(iter(hash_coverage)) <= 0:
        raise ValueError('Changed or absent measured hashing coverage')
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.4))
    panels = (('Streaming speed', 'steady_received_tps', 'Tokens / second'),
              ('First-token wait', 'time_to_first_token_seconds', 'Seconds'),
              ('Durable completion', 'durable_request_seconds', 'Seconds'))
    for ax, (title, field, unit) in zip(axes, panels):
        for index, (version, label, color) in enumerate((
                ('old', 'Before', '#6b7b8c'), ('new', 'GPU comparison + native streaming', '#087f8c'))):
            for group, mode in enumerate(('resident', 'disk')):
                data = arms[f'client-{version}-{mode}-v2']
                value = data['medians'][field]
                x = group + (index - .5) * .36
                ax.bar(x, value, width=.32, color=color, label=label if group == 0 else None)
                samples = [row[field] for row in data['individual_measurements']]
                ax.scatter([x + (j - 2.5) * .018 for j in range(6)], samples,
                           color='#192c3d', s=14, alpha=.65, zorder=3)
                ax.annotate(f'{value:.2f}', (x, value), xytext=(0, 7),
                            textcoords='offset points', ha='center', fontsize=9)
        ax.set_title(title)
        ax.set_ylabel(unit)
        ax.set_xticks([0, 1], ['Resident prefix', 'Disk restore'])
        ax.set_ylim(0, ax.get_ylim()[1] * 1.18)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.15)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=8, loc='upper left')
    fig.suptitle('Q4 Context COW: retain fresh integrity checks, reduce GPU readback', fontsize=14)
    fig.tight_layout(rect=(0, .17, 1, .92))
    fig.text(.03, .025,
        'llm-60 / RTX PRO 6000 Blackwell / FP16 KV. 8,192 source-prefix + 527 question tokens; 128 outputs.\n'
        'Bars: medians; dots: six measured requests per arm. Two warmups excluded. Model startup excluded.\n'
        'Disk uses warm filesystem cache. Local CLI, not HTTP. Durable completion includes audit and publication.\n'
        'Timing workers exited 0. Earlier correctness-case native exit codes were not captured. Not long-context certification.',
        fontsize=8)
    for extension in ('png', 'svg'):
        path = args.output / ('q4-seal-stream-comparison.' + extension)
        if extension == 'svg':
            stream = io.StringIO()
            fig.savefig(stream, format='svg', dpi=180, bbox_inches='tight')
            # Normalize derived vector formatting, never raw measurement files.
            text = '\n'.join(line.rstrip(' \t\r') for line in stream.getvalue().splitlines()) + '\n'
            path.write_text(text, encoding='utf-8', newline='\n')
        else:
            fig.savefig(path, dpi=180, bbox_inches='tight')
    changes = {}
    for mode in ('resident', 'disk'):
        before = arms[f'client-old-{mode}-v2']['medians']
        after = arms[f'client-new-{mode}-v2']['medians']
        changes[mode] = dict(
            streaming_tps_percent_change=100*(after['steady_received_tps']/before['steady_received_tps']-1),
            first_token_seconds_percent_change=100*(after['time_to_first_token_seconds']/before['time_to_first_token_seconds']-1),
            request_seconds_percent_change=100*(after['request_seconds']/before['request_seconds']-1),
            durable_seconds_percent_change=100*(after['durable_request_seconds']/before['durable_request_seconds']-1))
    report = dict(status='EXECUTED_TIMING_COMPARISON', input_sha256=inputs, arms=arms,
        changes_vs_prior_cow_profile=changes,
        stock_native_overhead_threshold='UNMEASURED: the informational 8% target requires a matched stock baseline',
        script_sha256=digest(Path(__file__)), matplotlib_version=matplotlib.__version__,
        limits=['Same-profile greedy matching-schedule numerical references',
                'Correctness-case native numeric exit codes unrecorded; timing-worker codes explicitly zero',
                'Not a same-day stock serving overhead comparison',
                'Native streaming only; Python still reads outer payload in full',
                'Warm filesystem; repeated deterministic completions may reuse durable payloads',
                'No current 64K speed claim'])
    (args.output / 'q4-seal-stream-comparison.json').write_text(
        json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')


if __name__ == '__main__':
    main()
