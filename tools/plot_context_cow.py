"""Reproduce publication figures from executed JSON measurements (matplotlib)."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'docs/context-cow/results/speed-8k-20261013'
OUT = ROOT / 'docs/context-cow/figures'
NAVY, TEAL, ORANGE, LIGHT, GRAY = '#17334b', '#008c95', '#e98536', '#b6cad7', '#657785'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.titlesize': 13,
                     'axes.titleweight': 'bold', 'axes.labelcolor': NAVY, 'text.color': NAVY,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.spines.left': False, 'axes.spines.bottom': False,
                     'xtick.color': GRAY, 'ytick.color': NAVY, 'svg.fonttype': 'none',
                     'figure.facecolor': '#fbfcfd', 'axes.facecolor': '#fbfcfd',
                     'svg.hashsalt': 'context-cow-v0.1.0'})
sources = {}


def read(name):
    path = DATA / name
    sources[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text())


def save(fig, name, title, subtitle, footer):
    fig.suptitle(title, x=.065, y=.975, ha='left', fontsize=23, weight='bold')
    fig.text(.065, .885, subtitle, fontsize=10.5, color=GRAY)
    fig.text(.065, .028, footer, fontsize=9, color=GRAY, va='bottom')
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / (name+'.png'), dpi=180, facecolor=fig.get_facecolor())
    svg = OUT / (name+'.svg')
    fig.savefig(svg, facecolor=fig.get_facecolor(), metadata={'Date': None})
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',
                   encoding='utf-8', newline='\n')
    plt.close(fig)


def main():
    before = read('cow-before.json')['requests']
    timed = read('cow-timed.json')['requests']
    latest = read('cow-prepare.json')
    after = latest['requests']
    ram = read('ram-sequence.json')['requests']
    fig, ax = plt.subplots(figsize=(12, 6.4))
    fig.subplots_adjust(left=.25, right=.94, top=.81, bottom=.22)
    rows = [timed[0], after[0], ram[0]]
    labels = ['COW / separate pools', 'COW / one work list', 'RAM restore / private pages']
    y = np.arange(3)
    start = np.zeros(3)
    for key, label, color in [('compute', 'Model computation', NAVY),
                               ('prepare', 'GPU write preparation', ORANGE),
                               ('finish', 'Finish consumed state', TEAL)]:
        values = np.array([r['native_decode_seconds'][key] for r in rows])
        ax.barh(y, values, left=start, color=color, height=.52, label=label)
        for i, value in enumerate(values):
            if value > .2: ax.text(start[i]+value/2, i, f'{value:.2f} s', ha='center', va='center', color='white', weight='bold')
        start += values
    for i, row in enumerate(rows):
        ax.text(start[i]+.08, i, f"prepare {row['native_decode_seconds']['prepare']:.3f} s", va='center', fontsize=10)
    ax.set(yticks=y, yticklabels=labels, xlabel='Seconds across 128 consumed output tokens', xlim=(0, 5.5))
    ax.invert_yaxis(); ax.grid(axis='x', alpha=.12); ax.set_axisbelow(True)
    ax.legend(loc='upper left', bbox_to_anchor=(-.27, -.17), frameon=False, ncol=3, fontsize=10)
    save(fig, 'write-preparation', 'Same model work. Much less preparation.',
         'A continuation: 8,192-token saved prefix + 512-token suffix + 128 outputs | Q8_0 / FP16 KV',
         'All sealed bytes still checked. Preparation includes hashing, synchronization and page handling.\n'
         'One sequential sample per bar; native phase timers exclude Python, validation and durable publication.')

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    fig.subplots_adjust(left=.07, right=.97, top=.79, bottom=.25, wspace=.25)
    labels = ['Resident A', 'Resident B', 'RAM A', 'Disk B']
    x = np.arange(4)
    for ax, key, title in [(axes[0], 'native_call_tps', 'Native decode calls'),
                           (axes[1], 'instrumented_api_tps', 'Complete instrumented API decode')]:
        for offset, records, label, color in [(-.25,before,'Initial full-resource COW',LIGHT),
                                               (0,timed,'Adjacent-page batching',NAVY),
                                               (.25,after,'All-pool work list',TEAL)]:
            values = [r[key] for r in records]
            bars = ax.bar(x+offset, values, width=.23, color=color, label=label)
            ax.bar_label(bars, labels=[f'{v:.1f}' for v in values], padding=3, fontsize=8)
        ax.set(xticks=x, xticklabels=labels, ylabel='Output tokens / second', title=title)
        ax.set_ylim(0, 65 if key == 'native_call_tps' else 10)
        ax.grid(axis='y', alpha=.12); ax.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower left', bbox_to_anchor=(.065,.105), ncol=3, frameon=False, fontsize=10)
    save(fig,'throughput','Context COW closes most of the native gap.',
         'Each completed four-lane run: 516 full-logit comparisons, maximum absolute error 0.0',
         'RTX PRO 6000 Blackwell, 16 CPU participants, 16,730 GPU expert slots. Single sequential samples, no error bars.\n'
         'Right panel includes controller work, full-vector comparisons and durable per-token publication; startup/prefill excluded.')

    fig, ax = plt.subplots(figsize=(12,6.5))
    fig.subplots_adjust(left=.23,right=.94,top=.79,bottom=.25)
    labels=['Fresh full history A','Fresh full history B','Resident prefix A','Resident prefix B','RAM prefix A','Disk prefix B']
    prefill=[latest['oracle'][label]['native']['prefill_seconds'] for label in ('A','B')]+[r['suffix_prefill_seconds'] for r in after]
    restore=[0,0]+[r['restore_seconds'] for r in after]
    y=np.arange(6)
    ax.barh(y,restore,color=ORANGE,label='Restore saved state',height=.57)
    ax.barh(y,prefill,left=restore,color=TEAL,label='Prefill inputs',height=.57)
    for i,(p,r) in enumerate(zip(prefill,restore)):
        ax.text(p+r+.06,i,f'{p+r:.3f} s',va='center',weight='bold')
    ax.set(yticks=y,yticklabels=labels,xlabel='Seconds before decode; model already loaded',xlim=(0,3.35))
    ax.invert_yaxis();ax.grid(axis='x',alpha=.12);ax.set_axisbelow(True)
    ax.legend(loc='upper left',bbox_to_anchor=(0,-.15),ncol=2,frameon=False)
    save(fig,'prefix-reuse','Prefill the changed suffix. Keep the prefix.',
         'Fresh paths compute 8,704 tokens. Restore paths compute 512; counters confirm zero prefix recomputation.',
         f"Preparing the reusable prefix cost {latest['prefix_prepare_seconds']:.3f} s; saving it cost {latest['save_seconds']:.3f} s (excluded above).\n"
         'Disk used a warm filesystem. This is not a process-crash recovery qualification or a 128K extrapolation.')

    fig,ax=plt.subplots(figsize=(12,6.5));fig.subplots_adjust(left=.09,right=.70,top=.78,bottom=.23)
    configurations=[('prototype-serving.json','Published prototype',NAVY),('main-serving.json','Upstream main',TEAL),
                    ('cpu-launch-serving.json','Main + CPU launch',ORANGE),('prototype-fixed.json','Prototype / fixed dispatch','#9b7ba5'),
                    ('main-fixed.json','Main / fixed dispatch',GRAY)]
    for name,label,color in configurations:
        records=read(name)['requests'];ax.plot([0,1,2],[r['native_decode_tps'] for r in records],marker='o',linewidth=2,label=label,color=color)
    ax.set(xticks=[0,1,2],xticklabels=['A / first request','B / second request','A / repeated request'],ylabel='Native output tokens / second',ylim=(0,120))
    ax.grid(axis='y',alpha=.15);ax.legend(loc='upper left',bbox_to_anchor=(1.02,1),frameon=False,fontsize=9)
    fig.text(.73,.39,'TOKEN PARITY: OPEN\nRepeated A diverged in\nevery serving control.\nTiming is not evidence\nof exact serving parity.',color='#a34a28',fontsize=11,linespacing=1.5)
    save(fig,'serving-controls','Ordinary serving is fast; parity is still open.',
         'Separate uncached native serving controls, A / B / A. Same 8K + 512 fixture; 128 outputs in every request.',
         'Serving controls use different dispatch behavior and publication boundaries from the branching probe.\n'
         'Do not attribute this entire gap to COW. Raw token traces, fixed-dispatch failures and source revisions are retained.')
    (OUT/'PROVENANCE.json').write_text(json.dumps(dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                                     matplotlib_version=matplotlib.__version__,sources=sources),indent=2)+'\n',newline='\n')
    print(json.dumps(dict(figures=4,sources=len(sources),output=str(OUT))))


if __name__ == '__main__':
    main()
