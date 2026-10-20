"""Plot measured restricted-resume results; do not extrapolate cache speedups."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summary',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();data=json.loads(args.summary.read_text())
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
        'axes.spines.right':False,'figure.facecolor':'#f6f8fb','axes.facecolor':'#f6f8fb'})
    figure,axes=plt.subplots(1,3,figsize=(13,4.8),layout='constrained')
    figure.suptitle('Context COW: matching-batch disk restart\n'+str(data['comparisons'])+
        ' exact frontier comparisons · '+data['status'],fontsize=17,fontweight='bold',color='#172b43')
    restores=data['restore_receipts'][:2]
    names=['Compute\nprefix','Save\nprefix','Restore\nA','Restore\nB']
    times=[data['prefill_seconds'],data['save_seconds'],*[row['seconds'] for row in restores]]
    bars=axes[0].bar(names,times,color=['#78859a','#d1a34b','#07858b','#2ba59e'],width=.65)
    axes[0].bar_label(bars,labels=[f'{value:.3f} s' for value in times],padding=5)
    axes[0].set(title='Measured API latency',ylabel='Seconds',ylim=(0,max(times)*1.3))
    work=[data['prefix_tokens'],*[row['forward_tokens'] for row in restores]]
    bars=axes[1].bar(['Compute\nprefix','Restore\nA','Restore\nB'],work,
        color=['#78859a','#07858b','#2ba59e'],width=.65)
    axes[1].bar_label(bars,labels=[str(value) for value in work],padding=5)
    axes[1].set(title='Saved-prefix forward work',ylabel='Native tokens evaluated',ylim=(0,data['prefix_tokens']*1.25))
    axes[2].bar(['Branch A','Branch B'],[data['max_logit_error']]*2,color='#07858b',width=.65)
    axes[2].text(.5,.55,f"{data['max_logit_error']:.6g} logit error\n{data['frozen_atol']:.6g} tolerance",ha='center',va='center',transform=axes[2].transAxes,
        fontsize=19,fontweight='bold',color='#07858b')
    axes[2].set(title='Independent reference\nMatching batch schedule',ylabel='Maximum absolute logit error',ylim=(0,max(1,data['max_logit_error']*1.3)))
    for axis in axes: axis.grid(axis='y',alpha=.15);axis.set_axisbelow(True)
    figure.supxlabel(f"Q8 weights / FP16 KV · {data['prefix_tokens']:,} prefix + {data['suffix_tokens']} suffix per branch\n"
        'Startup excluded and reported separately. Deterministic greedy fixture; original G1–G5 unqualified.',
        fontsize=10,color='#536477')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for suffix in ('.png','.svg'): figure.savefig(args.output.with_suffix(suffix),dpi=180)


if __name__=='__main__':main()
