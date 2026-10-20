"""Plot executed request timings and a separate same-store CPU query comparison."""
import json
import statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path('/home/dflanag3/spatial-artifacts/native-client-combined-20261016-v1')
primary = json.loads((root / 'primary-v1.json').read_bytes())
campaign = json.loads((root / 'gpu-campaign-v1/result.json').read_bytes())
hot = [row for row in campaign['generations'] if row['tag'].startswith('round')]
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw={'width_ratios': [1.1, 1]})
fig.suptitle('Context COW: combined Q4 candidate on llm-60', fontsize=16, fontweight='bold', y=.98)
colors = ['#285f9c', '#b35c27', '#277c64']
labels = []
for index in range(3):
    for j, question in enumerate(('A', 'B')):
        x = index * 2 + j
        values = [row['process_seconds'] for row in hot
                  if row['context_index'] == index and row['question'] == question]
        assert len(values) == 3
        axes[0].scatter([x-.12, x, x+.12], values, color=colors[index], s=38, zorder=3)
        axes[0].plot([x-.25, x+.25], [statistics.median(values)]*2, color=colors[index], linewidth=2)
        labels.append(f'Tree {index}\n{question}')
axes[0].set_xticks(range(6), labels)
axes[0].set_ylabel('Whole CLI request latency (seconds)')
axes[0].set_ylim(bottom=0)
axes[0].set_title('18 hot requests, 128 output tokens each')
axes[0].grid(axis='y', alpha=.2)
rate = primary['hot_requests']['whole_cli_tokens_per_second']
axes[0].text(.03, .18, f'{rate:.2f} output tokens/s across summed waits\n0 native file bytes loaded on all hot calls',
             transform=axes[0].transAxes, va='top', fontsize=10)
rows = primary['metadata_query']
positions = list(range(len(rows)))
axes[1].barh([x+.17 for x in positions], [row['median_catalog_seconds']*1000 for row in rows],
             height=.3, color='#a2acb5', label='Full catalog')
axes[1].barh([x-.17 for x in positions], [row['median_root_seconds']*1000 for row in rows],
             height=.3, color='#285f9c', label='Targeted root query')
axes[1].set_yticks(positions, [f"{row['arm'].title()} tree {row['context']}" for row in rows])
axes[1].set_xlabel('Median query latency (milliseconds; logarithmic scale)')
axes[1].set_xscale('log')
axes[1].set_title('Same-store CPU metadata comparison')
axes[1].invert_yaxis()
axes[1].legend(loc='upper left', bbox_to_anchor=(0, 1.22), ncol=2, frameon=False)
axes[1].grid(axis='x', alpha=.2)
fig.text(.02, .04, 'Q4 / FP16 KV / 7.2–8.1K saved prefixes / 448–512 added tokens / one active GPU request.\n'
    'CLI timing includes durable publication and observer overhead. CPU queries return identical root metadata; this is not an isolated inference speedup estimate.',
    fontsize=9, color='#46515b')
fig.tight_layout(rect=(0, .13, 1, .92), w_pad=3)
for suffix in ('png', 'svg'):
    fig.savefig(root / ('native-combined-q4-v2.' + suffix), dpi=170, facecolor='white')
