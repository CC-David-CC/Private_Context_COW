"""Draw retained Adaptive-P measurements; no model or GPU execution."""
import argparse, csv, hashlib, json, statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--input',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
assert sha(a.input)=='63143233d19e31123b4eeaeeccd899fd9fa256d671c0ee7e883a18aa1d861868'
r=json.loads(a.input.read_bytes())
assert r['status']=='FOCUSED-PASS' and r['numerical_max_error']==0 and r['source_commit']=='747d5d4b4ba7b10e99ca7312b918e5587e126385'
c=next(c for c in r['campaigns'] if c['name']=='gpu-adaptive-q4-8k-timing-v1')
rows=[x for x in c['timed_rows'] if x['phase']=='measured']
assert len(rows)==6 and all(x['output_count']==128 for x in rows)
assert all(x['counter_deltas']['prefill_tokens']==512 and x['counter_deltas']['loaded_file_bytes']==0 for x in rows)
a.output.mkdir(mode=0o700,parents=True,exist_ok=False)
fields=['branch','repetition','output_count','generation_tps','request_tps','request_seconds','first_consumed_seconds','median_inter_consumed_seconds','native_sampling_ms_per_output','native_decode_compute_ms_per_output','native_decode_prepare_ms_per_output','native_decode_finish_ms_per_output']
with (a.output/'measurements.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader()
    for row in rows: w.writerow({k:row[k] for k in fields})
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','axes.edgecolor':'#bcc8d1','text.color':'#152b3c','axes.labelcolor':'#152b3c','xtick.color':'#152b3c','ytick.color':'#425968'})
fig,axes=plt.subplots(2,2,figsize=(11,8.2))
colors=['#008a83','#3477ba']
metrics=[('generation_tps',1,'Generation','Tokens / second',110,1),('request_seconds',1,'Whole resident request','Seconds',2.5,3),('first_consumed_seconds',1000,'First consumed output','Milliseconds',500,1),('median_inter_consumed_seconds',1000,'Time between consumed outputs','Milliseconds',14,2)]
for ax,(key,scale,title,ylabel,limit,places) in zip(axes.flat,metrics):
    for x,branch in enumerate(('A','B')):
        vals=[row[key]*scale for row in rows if row['branch']==branch]
        assert len(vals)==3
        median=statistics.median(vals)
        ax.bar(x,median,color=colors[x],alpha=.16,width=.55,zorder=1)
        ax.scatter([x-.09,x,x+.09],vals,color=colors[x],s=40,zorder=3)
        ax.plot([x-.2,x+.2],[median,median],color='#152b3c',lw=1.8,zorder=4)
        ax.text(x,median+limit*.055,f'{median:.{places}f}',ha='center',fontweight='bold',fontsize=12)
    ax.set_title(title,pad=12);ax.set_ylabel(ylabel);ax.set_xticks([0,1],['Branch A','Branch B']);ax.set_ylim(0,limit);ax.set_xlim(-.65,1.65);ax.grid(axis='y',alpha=.20);ax.set_axisbelow(True)
fig.suptitle('Adaptive-P with Context COW: resident Q4 performance',fontsize=18,fontweight='bold',y=.975)
fig.text(.5,.928,'llm-60 / RTX PRO 6000 Blackwell / 8,192 saved prefix + 512 suffix + 128 outputs',ha='center',fontsize=11)
fig.subplots_adjust(left=.10,right=.97,top=.855,bottom=.15,hspace=.43,wspace=.30)
fig.text(.08,.075,'Dots: three measured runs. Line: median. One earlier warm-up per branch is excluded.',fontsize=10)
fig.text(.08,.046,'COW, integrity checks and publication stay on. Audit reads, startup and full checkpoint writes are outside timing.',fontsize=9)
fig.savefig(a.output/'adaptive-p-q4-8k.png',dpi=180,facecolor='white',metadata={'Software':'Context COW measured evidence'})
plt.close(fig)
provenance={'schema':'context-cow-adaptive-p-figure-v1','source_report':str(a.input),'source_report_sha256':sha(a.input),'source_commit':r['source_commit'],'profile_sha256':r['profile_sha256'],'native_elf_sha256':r['native_elf_sha256'],'fixture_source_sha256':c['fixture_source_sha256'],'selected_fixture_sha256':c['selected_fixture_sha256'],'policy':c['policy'],'matplotlib_version':matplotlib.__version__,'script_sha256':sha(__file__),'output_sha256':{name:sha(a.output/name) for name in ('measurements.csv','adaptive-p-q4-8k.png')},'measured_rows':6,'warmup_rows_excluded':2}
(a.output/'plot-inputs.json').write_text(json.dumps(provenance,indent=2,sort_keys=True)+'\n')
print(json.dumps({'output':str(a.output),'files':provenance['output_sha256'],'plot_inputs_sha256':sha(a.output/'plot-inputs.json')}))
