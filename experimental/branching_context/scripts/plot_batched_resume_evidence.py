"""Plot executed activation differences, including the exact-zero attention input."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('results',type=Path);a=p.parse_args()
    trace=json.loads((a.results/'input-layers.json').read_text())
    rows=trace['layer_comparison']['comparisons']
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),gridspec_kw={'width_ratios':[2,1]})
    for half,label,color in [(0,'Attention input','#2563eb'),(1,'Feed-forward input','#ea580c')]:
        group=[r for r in rows if r['half']==half]
        axes[0].plot([r['layer'] for r in group],[r['max_absolute_error'] for r in group],
                     label=label,color=color,linewidth=1.8)
    axes[0].set_yscale('symlog',linthresh=1e-6)
    axes[0].set(xlabel='Layer',ylabel='Maximum absolute activation difference',title='Same history, different batch partition')
    axes[0].legend(frameon=False);axes[0].grid(alpha=.2)
    first=trace['gdn_comparison']['first_difference']['max_absolute_error']
    final=trace['numerical_comparison']['max_absolute_error']
    axes[1].bar(['First QKV\nprojection','Final\nlogits'],[first,final],color=['#2563eb','#dc2626'],width=.6)
    axes[1].set_yscale('log');axes[1].set(title='Observed error at two boundaries',ylabel='Maximum absolute difference')
    for i,value in enumerate((first,final)):axes[1].text(i,value*1.4,f'{value:.6g}',ha='center',fontsize=10)
    axes[1].set_ylim(1e-6,5);axes[1].grid(axis='y',alpha=.2)
    fig.suptitle('Batches-only preflight FAIL: 128 + 4 + 1 versus 128 + 5',fontsize=14,fontweight='bold')
    fig.text(.02,.025,'Real llm-60 GPU / Q8 model. Projection input bytes identical. Fresh-repeat error 0; frozen tolerance 0.\n'
             'Readback preserved both output hashes. These are uncached forwards; no restore speed or G2 PASS is claimed.',fontsize=9)
    fig.tight_layout(rect=(0,.13,1,.94))
    for extension in ('png','svg'):fig.savefig(a.results/('batch-partition-diagnosis.'+extension),dpi=170)
    plt.close(fig)


if __name__=='__main__':main()
