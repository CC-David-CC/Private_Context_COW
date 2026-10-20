"""Plot executed primitive timings; do not imply full-API or long-context throughput."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results',type=Path)
    args = parser.parse_args()
    cases = [('Q4: 128 + 4 / 8 outputs','q4-request-tiny.json'),
             ('Q4: 1K + 64 / 16 outputs','q4-request-medium.json'),
             ('Q4: 8K + 512 / 128 outputs','q4-request-8k.json'),
             ('Q8: 128 + 4 / 8 outputs','q8-request-tiny.json')]
    fig,ax = plt.subplots(figsize=(10,6.2))
    for i,(label,name) in enumerate(cases):
        record = json.loads((args.results/name).read_text())
        if record['status'] != 'PASS' or record['numerical_max_error'] != 0:
            raise ValueError('Only executed exact primitive results may enter this chart')
        for offset,(branch,values) in zip((-.18,.18),record['branches'].items()):
            speed = values['tokens_per_second']
            ax.barh(i+offset,speed,height=.31,color='#227c9d' if branch == 'A' else '#59a14f',
                    label='Branch '+branch if i == 0 else None)
            ax.text(speed+1,i+offset,f'{speed:.1f}',va='center',fontsize=10)
    ax.set_yticks(range(len(cases)),[label for label,_ in cases])
    ax.invert_yaxis()
    ax.set_xlim(0,125)
    ax.set_xlabel('Generated tokens / second — timed resident native request only')
    ax.set_title('Real-GPU Context COW primitive: hashes enabled, zero logit error')
    ax.legend(loc='lower right')
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='x',alpha=.15)
    fig.tight_layout(rect=(0,.12,1,1))
    fig.text(.01,.02,'llm-60 · RTX PRO 6000 Blackwell · FP16 KV · synthetic tokens · one run per branch\n'
             'Startup, suffix ingestion, final capture and disk publication excluded. Not a 64K+ throughput claim.',fontsize=8)
    fig.savefig(args.results/'resident-request-throughput.png',dpi=180,bbox_inches='tight')
    fig.savefig(args.results/'resident-request-throughput.svg',bbox_inches='tight')


if __name__ == '__main__': main()
