"""Compare executed, exact before/after long requests; never extrapolate TPS."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True)
    args = parser.parse_args()
    rows,sources = [],{}
    for model in ('q4','q8'):
        paths = [args.results/(model+'-timing-64k.json'),args.results/('digest-reuse-'+model+'-64k.json')]
        before,after = [json.loads(path.read_text()) for path in paths]
        if any(report['status'] != 'PASS' or report['numerical_max_error'] != 0 for report in (before,after)):
            raise ValueError('Require completed exact model comparisons')
        sources.update({str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths})
        for name in ('A','B'):
            old,new = before['branches'][name],after['branches'][name]
            if (old['outputs'] != 2048 or new['outputs'] != 2048 or old['tokens'] != new['tokens']
                    or old['frontier_sha256'] != new['frontier_sha256']
                    or old['hashed_bytes'] != new['hashed_bytes'] or old['digest_copy_bytes'] != new['digest_copy_bytes']
                    or new['integrity_request_counters']['integrity_host_workspace_allocations'] != 0):
                raise ValueError('Changed output, integrity coverage or missing scratch evidence')
            rows.append(dict(label=model.upper()+' '+name,
                preparation_ms=[report['ms_per_output']['decode_prepare_ns'] for report in (old,new)],
                generation_tps=[report['outputs']/report['generation_seconds'] for report in (old,new)],
                request_tps=[report['outputs']/report['request_seconds'] for report in (old,new)],
                tokens_and_final_frontier_identical=True))
    fig,axes = plt.subplots(1,2,figsize=(11,5.3),layout='constrained')
    for ax,key,title,unit in zip(axes,('preparation_ms','request_tps'),
        ('GPU write preparation · lower is faster','Complete logical request · higher is faster'),
        ('Mean milliseconds per consumed output','Output tokens / whole request seconds')):
        for index,(label,color) in enumerate([('Baseline','#7d8994'),('Reused digest scratch','#328e83')]):
            positions = [i+(index-.5)*.34 for i in range(len(rows))]
            bars = ax.bar(positions,[row[key][index] for row in rows],width=.32,label=label,color=color)
            ax.bar_label(bars,fmt='%.2f',padding=3,fontsize=9)
        ax.set_xticks(range(len(rows)),[row['label'] for row in rows])
        ax.set_title(title,fontsize=11);ax.set_ylabel(unit)
        ax.set_ylim(0,max(max(row[key]) for row in rows)*1.22)
        ax.spines[['top','right']].set_visible(False)
    axes[0].legend(fontsize=9)
    fig.suptitle('Context COW: reuse temporary storage, keep every SHA check\n65,536-token code prefix + 8,335-token question · 2,048 outputs',fontsize=13)
    fig.supxlabel('llm-60 · FP16 KV · one measured request per branch\nStartup and later durable acknowledgment excluded. Client arrival measurements are separate.',fontsize=9)
    output = args.results/'digest-reuse-64k'
    for extension in ('png','svg'): fig.savefig(output.with_suffix('.'+extension),dpi=180)
    summary = dict(status='PASS',scope='one before/after measurement per branch; no repetition confidence interval',
        source_sha256=sources,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        matplotlib_version=matplotlib.__version__,rows=rows)
    output.with_suffix('.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__ == '__main__': main()
