"""Plot executed paired return measurements from compact independent audits."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load(path):
    raw = path.read_bytes()
    result = json.loads(raw)
    if result.get('status') != 'PASS_INDEPENDENT_RETAINED_RESULT_AUDIT':
        raise ValueError('Expected an independently audited terminal result')
    samples = result['samples']
    if len(samples) != 9 or {(r['root'], r['repeat']) for r in samples} != {
            (root, repeat) for root in range(3) for repeat in range(3)}:
        raise ValueError('Expected nine measured root/repetition pairs')
    if any(r['output_tokens'] != 128 or r['prefix_forward_tokens_in_request'] != 0
           for r in samples):
        raise ValueError('Figure requires the audited 128-output, suffix-only fixture')
    return result, hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--disk', required=True, type=Path)
    parser.add_argument('--ram', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    disk, disk_sha = load(args.disk)
    ram, ram_sha = load(args.ram)
    for key in ('profile', 'model_contract', 'model_weights'):
        if ram[key] != disk[key]:
            raise ValueError('Paired arms must use the same ' + key)
    if any(row.get('tier') != 'resident_pool' for row in ram['samples']):
        raise ValueError('RAM arm must have independently audited pool residency')
    if sorted((r['root'], r['repeat'], r['suffix_tokens']) for r in ram['samples']) != sorted(
            (r['root'], r['repeat'], r['suffix_tokens']) for r in disk['samples']):
        raise ValueError('Paired arms must use identical questions')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.edgecolor': '#cbd5e1', 'axes.labelcolor': '#334155',
        'text.color': '#0f172a', 'xtick.color': '#334155', 'ytick.color': '#334155',
        'svg.fonttype': 'none'})
    colors = ['#167d9a', '#d97736']
    labels = ['RAM pool', 'Disk / warm FS']
    datasets = [ram['samples'], disk['samples']]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5))
    fig.subplots_adjust(top=.80, bottom=.18, left=.08, right=.97, hspace=.63, wspace=.28)
    fig.text(.08, .945, 'Context COW', fontsize=13, weight='bold', color=colors[0])
    fig.text(.08, .903, 'Returning to three saved 64K code contexts', fontsize=21, weight='bold')
    fig.text(.08, .866, 'Q4  |  RTX PRO 6000 Blackwell  |  FP16 KV  |  one active request', fontsize=11)
    fig.text(.08, .838, '8,315–8,336 new question tokens + 128 output tokens per request', fontsize=10, color='#475569')

    def distribution(ax, field, title, ylabel, limit):
        values = [[r[field] for r in rows] for rows in datasets]
        boxes = ax.boxplot(values, positions=[1, 2], widths=.34, patch_artist=True,
            showfliers=False, medianprops={'color': '#0f172a', 'linewidth': 2},
            whiskerprops={'color': '#64748b'}, capprops={'color': '#64748b'})
        for patch, color in zip(boxes['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(.25)
            patch.set_edgecolor(color)
        for index, (points, color) in enumerate(zip(values, colors), 1):
            jitter = [(j - (len(points)-1)/2) * .018 for j in range(len(points))]
            ax.scatter([index + offset for offset in jitter], points, s=22, color=color, zorder=3)
        medians = [statistics.median(v) for v in values]
        ax.set_xticks([1, 2], [f'{label}\nmedian {median:.2f}' for label, median in zip(labels, medians)])
        ax.set_title(title, loc='left', weight='bold', pad=10)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, limit)
        ax.grid(axis='y', color='#e2e8f0', zorder=0)
        ax.set_axisbelow(True)
        return medians

    first = distribution(axes[0, 0], 'first_token_seconds', 'First token arrives', 'Seconds · lower is better', 9)
    durable = distribution(axes[0, 1], 'durable_seconds', 'Consumed state durably saved', 'Seconds · lower is better', 19)
    rate = distribution(axes[1, 0], 'received_tps', 'Steady received generation', 'Tokens / second · higher is better', 80)
    ax = axes[1, 1]
    stages = []
    for rows in datasets:
        stages.append([statistics.median(r['capture_seconds'] for r in rows),
                       statistics.median(r['durable_seconds']-r['snapshot_seconds'] for r in rows)])
    for index, (values, color, label) in enumerate(zip(stages, colors, labels)):
        positions = [j + (index-.5)*.32 for j in range(2)]
        bars = ax.bar(positions, values, width=.28, color=color, label=label)
        ax.bar_label(bars, fmt='%.2f', padding=3, fontsize=9)
    ax.set_title('Remaining state-management cost', loc='left', weight='bold', pad=10)
    ax.set_ylabel('Seconds · medians, not stacked')
    ax.set_xticks([0, 1], ['Capture', 'Snapshot → durable'])
    ax.set_ylim(0, 4.8)
    ax.grid(axis='y', color='#e2e8f0')
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc='upper right')
    fig.text(.08, .105, '9 measured requests + 3 warmups per arm. Each dot is one request. Outputs reached the 128-token cap; no EOS.', fontsize=9)
    fig.text(.08, .080, 'Same pinned model, native build and request boundaries. Startup excluded. Coordinator-consumer timing, not HTTP/browser timing.', fontsize=9)
    fig.text(.08, .055, 'RAM means serialized host-pool state; activation and GPU transfer still occur. Disk uses a warm filesystem. No cold-device claim.', fontsize=9)
    stem = args.output_dir / 'q4-64k-paired-return'
    fig.savefig(stem.with_suffix('.png'), dpi=180, facecolor='white')
    fig.savefig(stem.with_suffix('.svg'), facecolor='white')
    svg = stem.with_suffix('.svg')
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',
                   encoding='utf-8', newline='\n')
    plt.close(fig)
    summary = {'scope': 'Paired Q4 coordinator timings; no new full-logit qualification',
        'audit_sha256': {'ram': ram_sha, 'disk': disk_sha}, 'profile': ram['profile'],
        'median_first_token_seconds': dict(zip(('ram', 'disk'), first)),
        'median_durable_seconds': dict(zip(('ram', 'disk'), durable)),
        'median_received_tps': dict(zip(('ram', 'disk'), rate)),
        'first_token_reduction_fraction': 1-first[0]/first[1],
        'durable_reduction_fraction': 1-durable[0]/durable[1],
        'numerical_max_error': None, 'new_full_logit_comparisons': 0}
    stem.with_suffix('.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps(summary, sort_keys=True))


if __name__ == '__main__':
    main()
