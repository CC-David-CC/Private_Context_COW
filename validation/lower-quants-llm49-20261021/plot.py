from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent
j=json.loads((root/'result.json').read_text())
fig,ax=plt.subplots(figsize=(10,5.5))
fig.subplots_adjust(left=.34,right=.96,top=.78,bottom=.23)
labels=[]
colors=['#3977af','#499b82']
for i,run in enumerate(j['runs']):
 labels.append(f"{run['label'].upper()} | {run['prefix_tokens']:,} prefix tokens\nExpert RAM {run['expert_ram_gib']} GiB / context pool {run['context_ram_gib']} GiB")
 for k,row in enumerate(run['stages'][1:]):
  y=i+(-.17 if k==0 else .17);v=row['streamed_callback_tps']
  ax.barh(y,v,height=.28,color=colors[k],label=('A: dog' if k==0 else 'B: cat') if i==0 else None)
  ax.text(v+.25,y,f'{v:.1f}',va='center',fontsize=10)
ax.set_yticks(range(3),labels);ax.invert_yaxis();ax.set_xlim(0,24)
ax.set_xlabel('Observed streamed callback tokens per second')
ax.xaxis.grid(True,alpha=.18);ax.set_axisbelow(True)
for edge in ['top','right','left']:ax.spines[edge].set_visible(False)
ax.legend(loc='lower right',frameon=False)
fig.suptitle('Context COW: lower-quant trials',x=.05,y=.96,ha='left',fontsize=19,fontweight='bold')
fig.text(.05,.875,'RTX 4090 / 32 GB system RAM | Greedy | FP16 KV',fontsize=11)
fig.text(.05,.07,'Two different questions per configuration, one run each; no statistical comparison.\nThe long-context run also changes RAM budgets. Callbacks include COW/HTTP and may contain thinking.\nThese are not isolated GPU decode measurements. No numerical oracle was run.',fontsize=9,color='#444444')
fig.savefig(root/'throughput.png',dpi=160);fig.savefig(root/'throughput.svg')
(root/'plot-environment.json').write_text(json.dumps({'matplotlib':matplotlib.__version__})+'\n')
