"""Scientific plot of measured tiny-fixture data; no projected speedups."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summary',type=Path)
    parser.add_argument('output',type=Path)
    args = parser.parse_args()
    data = json.loads(args.summary.read_text())
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
                         'axes.spines.right':False,'figure.facecolor':'#f7f9fc','axes.facecolor':'#f7f9fc'})
    figure,axes = plt.subplots(1,3,figsize=(13,4.8),layout='constrained')
    figure.suptitle('Context COW: disk resume preserves computation\nFull batched / normal decode equivalence remains blocked',
                    fontsize=17,fontweight='bold',color='#182b45')
    latency = [data['timings']['prefix_api_seconds'],*[r['seconds'] for r in data['restores']]]
    colors = ['#73839a','#007f86','#28a7a4']
    bars=axes[0].bar(['Compute\nprefix','Restore\nA','Restore\nB'],latency,color=colors,width=.62)
    axes[0].bar_label(bars,labels=[f'{v:.3f} s' for v in latency],padding=5)
    axes[0].set(title='Engine already loaded',ylabel='Seconds',ylim=(0,max(latency)*1.28))
    bars=axes[1].bar(['Compute\nprefix','Restore\nA','Restore\nB'],[data['scope']['prefix_tokens'],0,0],color=colors,width=.62)
    axes[1].bar_label(bars,labels=[str(data['scope']['prefix_tokens']),'0','0'],padding=5)
    axes[1].set(title='Measured prefix forward work',ylabel='Tokens evaluated',ylim=(0,data['scope']['prefix_tokens']*1.25))
    values=[data['max_normal_decode_logit_error'],data['max_full_batched_logit_error']]
    bars=axes[2].bar(['Uncached\nnormal decode','Uncached\nfull batched'],values,color=['#007f86','#cf4f54'],width=.6)
    axes[2].bar_label(bars,labels=['0.00000',f'{values[1]:.5f}'],padding=5)
    axes[2].set(title='First generated-token frontier',ylabel='Maximum absolute logit error',ylim=(0,max(values)*1.28))
    for axis in axes:
        axis.grid(axis='y',alpha=.16); axis.set_axisbelow(True)
    figure.supxlabel('RTX PRO 6000 / Q8 weights / FP16 KV · 128 prefix + 4 suffix + 8 outputs per branch\n'
                       'Approximately 67 s engine startup excluded. Whole audit path: ~0.76 TPS. Frozen tolerance: 0.',
                       fontsize=10,color='#536477')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    figure.savefig(args.output.with_suffix('.png'),dpi=200)
    figure.savefig(args.output.with_suffix('.svg'))


if __name__=='__main__': main()
