"""Plot executed Q4 diagnostic layers and public-loop attribution, not estimates."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('results',type=Path)
    p.add_argument('output',type=Path)
    a = p.parse_args()
    read = lambda n: json.loads((a.results/n).read_text())
    paths = ['q4-profiled-resident.json','q4-ancestry-resident.json','q4-native-cow.json','q4-direct.json']
    data = [read(n) for n in paths]
    if any(d['status'] != 'PASS' for d in data): raise ValueError('Only completed exact runs may be plotted')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes = plt.subplots(1,2,figsize=(13,5.8),layout='constrained')
    labels = ['Public API\nbefore','Public API\n2 fixes','Native COW\nevery-token capture','Direct native\nPrefill']
    rates = [[d['branches'][branch]['tokens_per_second'] for branch in ('A','B')] for d in data]
    for i,values in enumerate(rates):
        axes[0].bar(i,sum(values)/2,color=['#8895a7','#008f82','#409bad','#6267ad'][i],width=.6)
        axes[0].scatter([i-.11,i+.11],values,color='#172b43',s=20,zorder=3)
        axes[0].text(i,max(values)+1.2,f'{min(values):.2f}–{max(values):.2f}',ha='center',fontsize=10)
    axes[0].set(xticks=range(4),xticklabels=labels,ylabel='Generated tokens / second',title='Measured diagnostic layers',ylim=(0,max(map(max,rates))*1.16))
    categories = ['Canonical reads','Restore','Capture','Host parking','Forward','Publication','Other Python / protocol']
    colours = ['#78859a','#bb8a49','#348ca0','#bc6969','#6267ad','#7c9e59','#c8ced6']
    for index,name in enumerate(['q4-wall-attribution.json','q4-ancestry-attribution.json']):
        rows = read(name)['measurements']
        total = rows['generation/API.generation_step']['inclusive_ns']/1e9
        values = [0.]*6
        for label,row in rows.items():
            if not label.startswith('generation/'): continue
            seconds = row['exclusive_ns']/1e9
            if label in ('generation/Store.get_snapshot','generation/Store.lineage','generation/BranchingModel.history'): values[0] += seconds
            elif label == 'generation/native.restore': values[1] += seconds
            elif label == 'generation/native.capture': values[2] += seconds
            elif label == 'generation/native.park': values[3] += seconds
            elif label == 'generation/native.feed': values[4] += seconds
            elif label == 'generation/Store.append': values[5] += seconds
        values.append(max(0,total-sum(values)))
        left = 0.
        for label,value,colour in zip(categories,values,colours):
            axes[1].barh(index,value,left=left,color=colour,label=label if index == 0 else None)
            if value > 1: axes[1].text(left+value/2,index,f'{value:.1f}',ha='center',va='center',fontsize=9,color='white')
            left += value
    axes[1].set(yticks=[0,1],yticklabels=['Before','2 fixes'],xlabel='Seconds for 32 generated tokens',title='Public API generation wall time')
    axes[1].invert_yaxis(); axes[1].legend(loc='upper center',bbox_to_anchor=(.5,-.16),ncol=2,fontsize=9,frameon=False)
    for axis in axes: axis.grid(axis='x' if axis is axes[1] else 'y',alpha=.15);axis.set_axisbelow(True)
    fig.suptitle('Context COW: locating the slowdown',fontsize=18,fontweight='bold')
    fig.supxlabel('Q4 / FP16 KV · 1,024 prefix + 64 suffix · 16 outputs per branch A/B · llm-60\n'
        'One run per layer; dots show branches, not statistical confidence. Startup excluded.\n'
        'Direct Prefill includes frontier audit output. Native COW omits canonical history/controller publication.\n'
        'API measurements are profiled. These are wall timings, not isolated GPU kernel timings.',fontsize=9)
    for suffix in ('.png','.svg'): fig.savefig(a.output.with_suffix(suffix),dpi=180)


if __name__ == '__main__': main()
