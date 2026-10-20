"""Measured CLI latency and native-stage chart; no inference in this script."""
import argparse,json,statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--report',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
data=json.loads(args.report.read_bytes())
assert data['status']=='FOCUSED-PASS' and data['numerical_max_error']==0
assert not args.output.with_suffix('.png').exists() and not args.output.with_suffix('.svg').exists()
rows=data['rows']
groups=[('Native snapshot',[r for r in rows if r['tag'].startswith('observed')]),
        ('Switch from RAM',[r for r in rows if r['tag'].startswith('round')]),
        ('After process restart',[r for r in rows if r['tag']=='disk-restart'])]
colors=['#296c9a','#55aaa0','#e5a85d']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,3,figsize=(15,5.5),gridspec_kw={'width_ratios':[1.4,1,1.2]})
fig.patch.set_facecolor('#f8fafc')
for axis in axes:axis.set_facecolor('#f8fafc')
labels=['Native\nsnapshot\nn=3','RAM\nswitch\nn=18','Disk\nrestart\nn=3']
first=[statistics.mean(r['ttft_seconds'] for r in group) for _,group in groups]
generation=[statistics.mean(r['request_to_durable_seconds']-r['ttft_seconds']-r['last_token_to_durable_seconds'] for r in group) for _,group in groups]
publish=[statistics.mean(r['last_token_to_durable_seconds'] for r in group) for _,group in groups]
axes[0].bar(range(3),first,label='Request to first token',color=colors[0])
axes[0].bar(range(3),generation,bottom=first,label='First to last token',color=colors[1])
axes[0].bar(range(3),publish,bottom=[a+b for a,b in zip(first,generation)],label='Last token to durable receipt',color=colors[2])
for index,total in enumerate(a+b+c for a,b,c in zip(first,generation,publish)):
    axes[0].text(index,total+.05,f'{total:.2f} s',ha='center',fontweight='bold')
axes[0].set(title='Mean request latency',ylabel='Seconds to durable receipt',xticks=range(3),xticklabels=labels)
axes[0].legend(loc='upper left',fontsize=8,frameon=False)
axes[0].set_ylim(0,4.7)
tps=[statistics.median(r['steady_arrival_tps'] for r in group) for _,group in groups]
axes[1].bar(range(3),tps,color=colors)
for index,(value,(_,group)) in enumerate(zip(tps,groups)):
    low=min(r['steady_arrival_tps'] for r in group);high=max(r['steady_arrival_tps'] for r in group)
    axes[1].errorbar(index,value,yerr=[[value-low],[high-value]],fmt='none',ecolor='#172b4d',capsize=5)
    axes[1].text(index,high+2,f'{value:.1f}',ha='center',fontweight='bold')
axes[1].set(title='Median steady client arrivals',ylabel='Tokens / second; whiskers are min/max',xticks=range(3),xticklabels=labels,ylim=(0,145))
means={name:[statistics.mean(r['native_generation_stages_ms'][name]/r['outputs'] for r in group) for _,group in groups]
       for name in ('prepare','compute','finish')}
bottom=[0.,0.,0.]
for stage,color in zip(('prepare','compute','finish'),('#e5a85d','#296c9a','#55aaa0')):
    values=means[stage];axes[2].bar(range(3),values,bottom=bottom,label=stage.capitalize(),color=color)
    bottom=[a+b for a,b in zip(bottom,values)]
axes[2].set(title='Mean timed native stages',ylabel='Milliseconds per consumed output token',xticks=range(3),xticklabels=labels,ylim=(0,11))
axes[2].legend(frameon=False,fontsize=8)
fig.suptitle('Context COW · three saved source trees · Q4',fontsize=19,fontweight='bold',x=.04,ha='left')
fig.text(.04,.885,'One active GPU request. Each measured answer contains 128 outputs. Startup is separate.',fontsize=11,color='#334155')
fig.text(.04,.075,'7,231–8,087 saved tokens + 448–512 question tokens. Greedy; same execution profile/operators. Full-byte hashes retained.',fontsize=9)
fig.text(.04,.045,f'Independent audit: {data["full_vector_comparisons"]} full-vector comparisons, maximum error 0. Candidate prefix/final sampling; not every-token certification.',fontsize=9)
fig.text(.04,.015,'Same-host CLI; warm/uncontrolled filesystem cache; no clean native-without-COW baseline or LAN measurement in this chart.',fontsize=9,color='#475569')
fig.subplots_adjust(left=.055,right=.985,top=.80,bottom=.23,wspace=.32)
fig.savefig(args.output.with_suffix('.png'),dpi=160)
fig.savefig(args.output.with_suffix('.svg'))
print(json.dumps({'png':str(args.output.with_suffix('.png')),'svg':str(args.output.with_suffix('.svg'))}))
