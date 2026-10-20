"""Plot all executed equal-budget observations, including the confounded pair."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root=Path('/home/dflanag3/spatial-artifacts/q4-tree-retention-20261016-v1')
records=[json.loads((root/name).read_bytes()) for name in ('policy-pair-v1.json','policy-pair-v2.json')]
assert all(row['status']=='CORRECTNESS-PASS; TIMING-REPORTED' for row in records)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,2,figsize=(14,9),layout='constrained',gridspec_kw={'width_ratios':[1.55,1]})
fig.suptitle('Context COW: retain hot trees within the existing cache budget',fontsize=18,fontweight='bold')
colors={'control':'#4c6e9f','candidate':'#248d71'}
for index,(record,title) in enumerate(zip(records,('First pair: CPU fault tests overlapped candidate','Repeated pair: fault-test workers held'))):
    left,right=axes[index]
    pairs=record['paired_observations'];x=np.arange(len(pairs))
    for name,label in (('control','Old policy'),('candidate','Retain hot trees')):
        values=[row[name+'_seconds'] for row in pairs]
        left.plot(x,values,marker='o',markersize=4,color=colors[name],label=label)
    for boundary in (5.5,11.5):left.axvline(boundary,color='#aab0b8',linestyle=':',linewidth=1)
    left.set_xticks(x,[f'{row["question"]}{row["context_index"]+1}' for row in pairs],fontsize=8)
    left.set_ylim(bottom=0);left.set_ylabel('Whole CLI seconds / 128 outputs')
    left.set_title(title,loc='left',fontweight='bold');left.grid(axis='y',alpha=.15)
    if index==0:left.legend(frameon=False,ncol=2)
    left.set_xlabel('Three rounds; A/B questions across Trees 1, 2 and 3')
    metrics=('median_ttft_seconds','median_last_token_to_durable_seconds','median_cli_seconds')
    positions=np.arange(len(metrics));width=.35
    for offset,(name,label) in zip((-.175,.175),(('control','Old policy'),('candidate','Retain hot trees'))):
        values=[record['arms'][name][key] for key in metrics]
        right.bar(positions+offset,values,width,color=colors[name],label=label)
        for position,value in zip(positions+offset,values):right.text(position,value+.035,f'{value:.2f}',ha='center',fontsize=9)
    right.set_xticks(positions,['First token','Last token\nto durable','Whole CLI'])
    right.set_ylabel('Median seconds');right.set_ylim(0,max(record['arms'][name]['median_cli_seconds'] for name in colors)*1.22)
    right.set_title(f'Median reduction {record["ratio_of_medians_reduction_percent"]:+.1f}%\nTotal-time reduction {record["total_time_reduction_percent"]:+.1f}%',loc='left')
    right.grid(axis='y',alpha=.15)
fig.supxlabel('Q4_K_XL / FP16 KV | RTX PRO 6000 Blackwell | 7.2-8.1K roots + 448-512-token questions\n'
    'Both policies: 8 GiB / 8 native images, 96 GiB RAM pool, 128 GiB disk pool; one active GPU request.\n'
    'All observations retained. CLI includes capture, durable publication and observation; startup/audits excluded.\n'
    'Each arm passed 818 vector comparisons at error 0; 18 direct paired final-vector checks per pair.',fontsize=9)
fig.savefig(root/'tree-policy-pair-v2.png',dpi=170)
fig.savefig(root/'tree-policy-pair-v2.svg')
print(json.dumps(dict(status='RENDERED',pairs=2,measured_requests_per_arm_per_pair=18)))
