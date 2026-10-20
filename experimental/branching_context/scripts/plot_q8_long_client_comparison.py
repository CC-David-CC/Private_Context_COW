"""Plot retained Q8 CLI measurements; does not run inference or qualify a gate."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('measurements', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    raw = args.measurements.read_bytes()
    data = json.loads(raw)
    if (data['context_tokens'], data['question_tokens'], data['output_bound']) != (65536, 8335, 2048):
        raise ValueError('This figure describes the qualified 64K/8335/2048 campaign')
    groups = [('resident', 'A'), ('resident', 'B'), ('disk', 'A'), ('disk', 'B')]
    metrics = [
        ('steady_received_tps', 'Streaming continuation', 'Output tokens / second', 1),
        ('durable_request_tps', 'Request through durable publication', 'Output tokens / second', 1),
        ('time_to_first_token_seconds', 'Time to first token', 'Seconds', 1),
        ('p95_tpot_seconds', '95th percentile token gap', 'Milliseconds', 1000),
    ]
    colors = {'resident': '#2366b5', 'disk': '#c96a12'}
    plt.rcParams.update({'font.size': 11, 'svg.hashsalt': 'context-cow-q8-client-v1'})
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), layout='constrained')
    for ax, (field, title, unit, scale) in zip(axes.flat, metrics):
        for index, (mode, branch) in enumerate(groups):
            group = data['arms'][mode]['branches'][branch]
            samples = [row[field] * scale for row in group['samples']]
            if len(samples) != 3 or group['outputs_each'] != [2048] * 3:
                raise ValueError('Figure requires three executed 2048-output measurements per group')
            median = statistics.median(samples)
            if median != group['median'][field] * scale:
                raise ValueError('Reported median differs from measured samples')
            ax.scatter([index - .08, index, index + .08], samples,
                       color=colors[mode], s=38, zorder=3, edgecolors='white', linewidths=.6)
            ax.hlines(median, index - .23, index + .23, color=colors[mode], linewidth=2)
            ax.annotate(f'{median:.2f}', (index, max(samples)), xytext=(0, 9),
                        textcoords='offset points', ha='center', fontsize=10)
        ax.set(title=title, ylabel=unit, xticks=range(4),
               xticklabels=['Resident A', 'Resident B', 'Disk A', 'Disk B'], xlim=(-.5, 3.5))
        ax.set_ylim(0, ax.get_ylim()[1] * 1.18)
        ax.grid(axis='y', color='#dfe5eb', linewidth=.7)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Context COW · Q8 · 64K saved prefix + 8,335-token question\n'
                 '2,048 outputs per request · three measurements per branch and mode', fontsize=15)
    startup = data['startup_seconds_reported_separately']
    fig.supxlabel('Dots: individual requests. Lines and labels: medians. Warmups excluded.\n'
                  'Same-host CLI; warm filesystem. Durable saves may reuse existing payloads.\n'
                  f'Model startup excluded: resident {startup["resident"]:.2f} s, disk {startup["disk"]:.2f} s. '
                  'No stock/HTTP comparison.', fontsize=10)
    args.output.mkdir(parents=True, exist_ok=True)
    stem = args.output / 'q8-long-client-comparison'
    fig.savefig(stem.with_suffix('.png'), dpi=170)
    fig.savefig(stem.with_suffix('.svg'), metadata={'Date': None})
    plt.close(fig)
    # Matplotlib emits trailing whitespace in SVG path data; normalize only
    # this derived figure. Raw measurements are never rewritten.
    svg = stem.with_suffix('.svg')
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n',
                   encoding='utf-8', newline='\n')
    receipt = {
        'scope': 'artifact plotting only; no backend test or additional measurement',
        'measurements_sha256': hashlib.sha256(raw).hexdigest(),
        'plotter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'matplotlib': matplotlib.__version__,
        'outputs': {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in (stem.with_suffix('.png'), svg)},
    }
    (args.output / 'plot-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n',
                                                 encoding='utf-8', newline='\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
