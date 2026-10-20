"""Plot executed CLI arrival measurements, excluding warm-up requests."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True)
    args = parser.parse_args()
    cases = []
    manifest = dict(scope='8K synthetic prefix + 512 suffix + 128 actual outputs; same-host CLI',inputs={})
    for mode in ('resident','disk'):
        path = args.results/('q8-client-'+mode+'.json')
        worker = args.results/('q8-client-'+mode+'-worker.json')
        record,native = json.loads(path.read_text()),json.loads(worker.read_text())
        if record['status'] != 'PASS' or native['status'] != 'PASS' or not native['native_exit_observed']:
            raise ValueError('Only completed executed client campaigns may enter the chart')
        rows = [r for r in record['requests'].values() if not r['warmup']]
        if len(rows) != 6 or any(r['outputs'] != 128 or r['mode'] != mode for r in rows):
            raise ValueError('Chart requires the declared six measured 128-output requests')
        cases.append((mode,rows))
        manifest['inputs'][mode] = dict(client_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            worker_sha256=hashlib.sha256(worker.read_bytes()).hexdigest(),profile_id=native['profile_id'])
    fig,axes = plt.subplots(1,3,figsize=(13,5.2))
    panels = [('Latency (seconds)', [('First token','time_to_first_token_seconds'),
                ('Snapshot','request_seconds'),('Durable ack','durable_request_seconds')],1),
              ('Received token gaps (ms)', [('Median','median_tpot_seconds'),('p95','p95_tpot_seconds')],1000),
              ('Throughput (tokens/second)', [('Streaming','steady_received_tps'),
                ('Snapshot','request_tps'),('Durable ack','durable_request_tps')],1)]
    for ax,(title,fields,scale) in zip(axes,panels):
        for i,(mode,rows) in enumerate(cases):
            for j,(label,key) in enumerate(fields):
                values = [r[key]*scale for r in rows]
                median = statistics.median(values)
                x = j+(i-.5)*.34
                ax.bar(x,median,width=.3,color=('#157f82','#d19233')[i],
                       label=('Resident prefix','Disk restore')[i] if j == 0 else None)
                ax.scatter([x]*len(values),values,color='#243746',s=10,zorder=3,alpha=.65)
                ax.annotate(f'{median:.2f}',(x,median),xytext=(0,5),textcoords='offset points',
                            ha='center',fontsize=8)
        ax.set_title(title,fontsize=12)
        ax.set_xticks(range(len(fields)),[label for label,_ in fields],rotation=20)
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=.15)
        ax.set_axisbelow(True)
        ax.set_ylim(0,ax.get_ylim()[1]*1.15)
    axes[0].legend(frameon=False,fontsize=9)
    fig.suptitle('Q8 Context COW: what the CLI caller receives',fontsize=16)
    fig.tight_layout(rect=(0,.16,1,.92))
    fig.text(.035,.025,'llm-60 / RTX PRO 6000 Blackwell / FP16 KV / hashes enabled. Bars: medians; dots: six measured requests.\n'
        'Warm-up excluded. Token-gap p95 is per request. Startup excluded. Durable ack includes final audit and disk publication.\n'
        '8K results do not predict 64K behavior. This is a local subprocess client, not HTTP or a remote network test.',fontsize=9)
    for suffix in ('png','svg'): fig.savefig(args.results/('q8-client-timing.'+suffix),dpi=180,bbox_inches='tight')
    (args.results/'q8-client-timing-provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__ == '__main__': main()
