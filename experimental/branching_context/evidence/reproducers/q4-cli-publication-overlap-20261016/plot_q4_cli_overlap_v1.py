"""Plot observed-client timing and the separately controlled overlap witness."""
import argparse,base64,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();root=args.root
primary=json.loads((root/'primary-v1.json').read_bytes());control=json.loads((root/'controlled-v1/result.json').read_bytes())
assert primary['status']=='FOCUSED-PASS' and primary['numerical_max_error']==0
pairs=primary['benchmark']['pairs'];witness=control['controlled_witness']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(13,5.7),gridspec_kw={'width_ratios':[1,1.15]})
fig.patch.set_facecolor('#f8fafc')
for axis in axes:axis.set_facecolor('#f8fafc')
x=list(range(3));width=.34
axes[0].bar([value-width/2 for value in x],[row['serial_seconds'] for row in pairs],width,color='#296c9a',label='Wait for each client to exit')
axes[0].bar([value+width/2 for value in x],[row['queued_seconds'] for row in pairs],width,color='#55aaa0',label='Queue four independent clients')
for index,row in enumerate(pairs):
    axes[0].text(index-width/2,row['serial_seconds']+.15,f'{row["serial_seconds"]:.2f}',ha='center',fontsize=9)
    axes[0].text(index+width/2,row['queued_seconds']+.15,f'{row["queued_seconds"]:.2f}',ha='center',fontsize=9)
    axes[0].text(index,15.1,f'{row["elapsed_reduction_percent"]:.1f}% shorter',ha='center',fontweight='bold')
axes[0].set(title='Natural run · four × 128-output requests',ylabel='Observed-client group completion (seconds)',
            xticks=x,xticklabels=['Pair 1\nserial → queued','Pair 2\nqueued → serial','Pair 3\nserial → queued'],ylim=(0,17))
axes[0].legend(loc='lower left',frameon=False,fontsize=8)
begin=witness['publication_release_monotonic_ns']-witness['first_publication_held_ns']
release=witness['first_publication_held_ns']/1e9
axes[1].barh(3,release,left=0,height=.38,color='#e5a85d',label='Deliberate publication hold')
maximum=release
for index,generation in enumerate(control['generations']):
    trace=json.loads(Path(generation['trace']).read_bytes())
    events=[(row['arrival_monotonic_ns'],json.loads(base64.b64decode(row['raw_line_b64']))) for row in trace['requests'][0]['events']]
    if index==0:
        durable=(next(stamp for stamp,event in events if event['event']=='durable')-begin)/1e9
        axes[1].plot(durable,3,marker='D',color='#296c9a',markersize=6,label='First durable receipt')
        axes[1].annotate('durable',(durable,3),xytext=(4,10),textcoords='offset points',fontsize=9)
        maximum=max(maximum,durable)
        continue
    tokens=[(stamp-begin)/1e9 for stamp,event in events if event['event'] in ('text','turn_complete') and 'token_id' in event]
    y=3-index
    axes[1].plot([tokens[0],tokens[-1]],[y,y],color='#55aaa0',linewidth=5,solid_capstyle='round')
    axes[1].plot(tokens[0],y,marker='o',color='#296c9a',markersize=5)
    axes[1].annotate(f'8 outputs · first at {tokens[0]:.2f} s',(tokens[0],y),xytext=(-85,14),textcoords='offset points',fontsize=9)
axes[1].axvline(release,color='#b7791f',linestyle='--',linewidth=1)
axes[1].text(release,3.53,f'release {release:.2f} s',ha='right',fontsize=9)
axes[1].set(title='Controlled real-GPU proof · excluded from speed results',xlabel='Seconds since the first save was held',
            yticks=[3,2,1,0],yticklabels=['First save','Client 2 arrivals','Client 3 arrivals','Client 4 arrivals'],xlim=(-.15,maximum+.9),ylim=(-.55,3.9))
fig.suptitle('Context COW · queue clients, retain one GPU worker',fontsize=18,fontweight='bold',x=.04,ha='left')
fig.text(.04,.885,'Q4 · three saved source trees · one active request per GPU · no production engine changes',fontsize=11,color='#334155')
fig.text(.04,.09,'All 32 continuations match qualified references: exact tokens, full greedy controllers/history and final logits (maximum error 0).',fontsize=9)
fig.text(.04,.06,'Natural token overlap: 0/9 windows. Source acquisition preceded earlier durability in 9/9; timing changes are not causally isolated.',fontsize=9)
fig.text(.04,.03,'Same-host observer-instrumented CLI completion includes trace serialization/fsync; startup, fork setup and parent audits are excluded.',fontsize=9,color='#475569')
fig.subplots_adjust(left=.065,right=.98,top=.79,bottom=.24,wspace=.38)
assert not args.output.with_suffix('.png').exists() and not args.output.with_suffix('.svg').exists()
fig.savefig(args.output.with_suffix('.png'),dpi=160);fig.savefig(args.output.with_suffix('.svg'))
print(json.dumps(dict(png=str(args.output.with_suffix('.png')),svg=str(args.output.with_suffix('.svg')))))
