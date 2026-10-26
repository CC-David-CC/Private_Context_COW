"""Render measured/planned native resource evidence; no inference or training."""
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np

root=Path(__file__).resolve().parents[1]
record=json.loads((root/'validation/native-action-real-memory-20261026.json').read_text(encoding='utf8'))
rows=record['resource_ledger'];assert len(rows)==4 and record['real_weight_updates']==0
gib=1<<30;x=np.arange(4)
persistent=np.array([r['persistent_bytes']/gib for r in rows])
graph=np.array([r['planned_peak_graph_bytes']/gib for r in rows])
reserve=np.array([r['reserve_bytes']/gib for r in rows])
total=persistent+graph+reserve;capacity=rows[0]['total_device_bytes']/gib
fig,ax=plt.subplots(figsize=(11,7));fig.subplots_adjust(left=.095,right=.98,top=.80,bottom=.22)
ax.bar(x,persistent,color='#325d88',label='Persistent tensors')
ax.bar(x,graph,bottom=persistent,color='#d89032',label='Planned graph buffer')
ax.bar(x,reserve,bottom=persistent+graph,color='#eeeeee',edgecolor='#777777',hatch='///',label='Native reserve')
ax.axhline(capacity,color='#555555',linestyle='--',linewidth=1.6,label=f'GPU total: {capacity:.2f} GiB')
observed=rows[0]['sampled_process_peak_mib']/1024
ax.plot([0],[observed],marker='*',color='#146b45',markersize=13,zorder=4,label=f'Observed process peak: {observed:.1f} GiB')
for i,value in enumerate(total):
 ax.text(i,value+1,f'{value:.1f} GiB',ha='center',va='bottom',fontsize=11,fontweight='bold')
 ax.text(i,3,'PASS'if rows[i]['gradient_pass']else'REFUSED',ha='center',fontsize=10,color='white',fontweight='bold')
ax.set_xticks(x,['Full gradients\n3,663 tokens','Full gradients\n4,873 tokens','LoRA rank 16\n4,873 tokens','Full gradients\n8,192 tokens*'])
ax.set_ylim(0,280);ax.set_yticks(range(0,281,40));ax.set_ylabel('GiB (2^30 bytes)')
ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
fig.suptitle('A small model can still have a large training graph',fontsize=18,fontweight='bold',y=.97)
fig.text(.5,.91,'MiniCPM5-2B learned FP32 parent | dedicated RTX PRO 6000 Blackwell | no weight updates',ha='center',fontsize=10)
handles,names=ax.get_legend_handles_labels();fig.legend(handles,names,loc='upper center',bbox_to_anchor=(.5,.875),ncol=3,frameon=False,fontsize=9)
fig.text(.5,.10,'Bars: persistent tensors + planned graph buffer + reserve, before CUDA/runtime overhead.\nAdaptive optimizer state is additional. A reserve is not allocated resident memory.',ha='center',fontsize=10)
fig.text(.5,.035,'*The 8k sequence is an allocation-only extension, not approved training data.\nRefusals are resource observations, not model failures. Sampled NVML peak is not a guaranteed upper bound.',ha='center',fontsize=9,color='#555555')
target=root/'docs/results/native-action-memory-20261026.png';target.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(target,dpi=180,facecolor='white');plt.close(fig)
print(target)
