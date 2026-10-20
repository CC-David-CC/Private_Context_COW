"""Plot executed same-profile paths; no old-policy speedup is inferred."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root=Path('/home/dflanag3/spatial-artifacts/q4-tree-retention-20261016-v1')
record=json.loads((root/'primary-v1.json').read_bytes())
assert record['status']=='FOCUSED-GPU-PASS; NOT-INTEGRATED'
campaign=json.loads((root/'gpu-campaign-v1/result.json').read_bytes())
hot=record['hot_round']['requests_detail']
labels=[f'Tree {r["context"]+1} · {r["question"]}' for r in hot]
first=[r['arrival_summary']['metrics_ns']['request_to_first_token']/1e9 for r in hot]
stream=[(r['outputs']-1)*r['arrival_summary']['metrics_ns']['token_interval_mean']/1e9 for r in hot]
tail=[r['arrival_summary']['metrics_ns']['last_token_to_durable']/1e9 for r in hot]
for row,a,b,c in zip(hot,first,stream,tail):
    assert abs(a+b+c-row['arrival_summary']['metrics_ns']['request_to_durable']/1e9)<1e-6
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,(left,right)=plt.subplots(1,2,figsize=(13,6),gridspec_kw={'width_ratios':[1.2,1]},layout='constrained')
fig.suptitle('Q4 · three retained trees · one active GPU request',fontsize=18,fontweight='bold')
y=np.arange(len(hot))
left.barh(y,first,label='Request → first token',color='#4169a1')
left.barh(y,stream,left=first,label='Token stream',color='#4c9f87')
left.barh(y,tail,left=np.array(first)+stream,label='Last token → durable receipt',color='#c28b4c')
left.set_yticks(y,labels);left.invert_yaxis();left.set_xlabel('Seconds from HTTP request');left.set_title('Hot requests: 128 outputs each; zero native state bytes loaded')
left.legend(loc='lower left',bbox_to_anchor=(0,-.32),frameon=False,fontsize=9)
hot_a=[next(r for r in hot if r['context']==i and r['question']=='A') for i in range(3)]
cold=[next(r for r in campaign['generations'] if r['context_index']==i and r['tag']=='disk-restart') for i in range(3)]
x=np.arange(3)
right.bar(x-.18,[r['seconds'] for r in hot_a],.34,color='#4c9f87',label='Hot retained image')
right.bar(x+.18,[r['process_seconds'] for r in cold],.34,color='#4169a1',label='After process restart')
right.set_xticks(x,['Tree 1','Tree 2','Tree 3']);right.set_ylabel('Whole CLI request seconds');right.set_ylim(0,4.25)
right.set_title('Same A questions, same profile, 128 outputs')
right.legend(loc='lower left',bbox_to_anchor=(0,-.32),frameon=False,fontsize=9)
for i,(warm,disk) in enumerate(zip(hot_a,cold)):
    right.text(i-.18,warm['seconds']+.06,f'{warm["seconds"]:.2f}',ha='center',fontsize=9)
    right.text(i+.18,disk['process_seconds']+.06,f'{disk["process_seconds"]:.2f}',ha='center',fontsize=9)
fig.supxlabel('RTX PRO 6000 Blackwell · Q4_K_XL / FP16 KV · 7.2–8.1K prefixes + 448–512-token questions\n'
    f'{record["full_vector_comparisons"]} exact vector comparisons · one campaign, one observation per displayed case\n'
    'Same-budget old-policy speed control remains unrun; filesystem cache was not flushed.',fontsize=9)
fig.savefig(root/'bounded-tree-retention-v1.png',dpi=170)
fig.savefig(root/'bounded-tree-retention-v1.svg')
print(json.dumps({'status':'RENDERED','comparisons':record['full_vector_comparisons']}))
