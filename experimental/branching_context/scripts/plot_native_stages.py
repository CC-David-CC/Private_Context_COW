"""Plot executed native stage measurements, without extrapolating performance."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--q4',type=Path,required=True)
    parser.add_argument('--q8',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    rows = []
    for model,path in [('Q4',args.q4),('Q8',args.q8)]:
        report = json.loads(path.read_text())
        if report['status'] != 'PASS': raise ValueError('Only completed passing measurements')
        for name,branch in report['branches'].items():
            timings = branch['ms_per_output']
            rows.append((model+' · '+name,
                [timings[key] for key in ('decode_prepare_ns','decode_compute_ns','decode_finish_ns')]+
                [1000*branch['unassigned_generation_seconds']/branch['outputs']],branch['outputs']))
    fig,ax = plt.subplots(figsize=(9,5.3),layout='constrained')
    bottom = [0.]*len(rows)
    colors = ['#e29c45','#4283a7','#4a9867','#9c9ca9']
    for index,label in enumerate(['Prepare GPU writes','Model work + frontier copy','Finish consumed state','Other generation work']):
        values = [row[1][index] for row in rows]
        ax.bar(range(len(rows)),values,bottom=bottom,label=label,color=colors[index],width=.62)
        bottom = [a+b for a,b in zip(bottom,values)]
    for index,total in enumerate(bottom): ax.text(index,total+.35,f'{total:.2f} ms',ha='center')
    ax.set_xticks(range(len(rows)),[row[0] for row in rows])
    ax.set_ylabel('Mean milliseconds per consumed output token')
    ax.set_title('Context COW native generation stages\n65,536-token code prefix + 8,335-token question · 2,048 outputs')
    ax.set_ylim(0,max(bottom)*1.3)
    ax.spines[['top','right']].set_visible(False)
    ax.legend(loc='upper left',fontsize=9)
    fig.supxlabel('llm-60 · FP16 KV · SHA checks enabled · one measured request per branch\nBlocking wall times; not CUDA-event durations. Startup, suffix and publication excluded.',fontsize=9)
    for extension in ('png','svg'): fig.savefig(args.output.with_suffix('.'+extension),dpi=180)
    provenance = dict(sources={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in (args.q4,args.q8)},
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),matplotlib=matplotlib.__version__,
        scope='one measurement per branch, no error bars or extrapolated throughput')
    args.output.with_suffix('.provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')


if __name__ == '__main__': main()
