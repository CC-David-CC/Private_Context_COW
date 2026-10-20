"""Compare measured guarded and caller-managed preparation with RAM controls."""
import hashlib
import json
from pathlib import Path

import numpy as np
from plot_context_cow import ROOT, OUT, NAVY, TEAL, ORANGE, GRAY, plt, save


def main():
    sources = {}

    def read(path):
        p = ROOT/'docs/context-cow/results'/path
        sources[path] = hashlib.sha256(p.read_bytes()).hexdigest()
        return json.loads(p.read_text())

    guarded = read('speed-8k-20261013/cow-prepare.json')
    external = read('external-integrity-20261013/external-model.json')
    prepared = read('external-integrity-20261013/external-prepared-model.json')
    assert all(d['status']=='PASS' for d in (guarded, external, prepared))
    rows = [guarded['requests'][0], external['requests'][0], prepared['requests'][0],
            guarded['requests'][2], prepared['requests'][2]]
    labels = ['Guarded COW A', 'Caller-managed COW A', 'Caller-managed COW A\nreuse writable access',
              'RAM restore A', 'RAM restore A\nreuse writable access']
    fig, axes = plt.subplots(1,2,figsize=(14,7))
    fig.subplots_adjust(left=.25,right=.96,top=.80,bottom=.23,wspace=.45)
    y=np.arange(len(rows));left=np.zeros(len(rows))
    for key,label,color in [('compute','Model computation',NAVY),('prepare','GPU write preparation',ORANGE),('finish','Finish state',TEAL)]:
        values=np.array([r['native_decode_seconds'][key] for r in rows])
        axes[0].barh(y,values,left=left,color=color,height=.5,label=label);left+=values
    for i,row in enumerate(rows):
        axes[0].text(left[i]+.04,i,f"{row['native_decode_seconds']['prepare']*1000:.2f} ms prep",va='center',fontsize=8)
    axes[0].set(yticks=y,yticklabels=labels,xlabel='Seconds over 128 output tokens',title='Native phase time',xlim=(0,3.35))
    bars=axes[1].barh(y,[r['native_call_tps'] for r in rows],color=[NAVY,TEAL,TEAL,GRAY,GRAY],height=.5)
    axes[1].bar_label(bars,labels=[f"{r['native_call_tps']:.2f}" for r in rows],padding=4,fontsize=10)
    axes[1].set(yticks=y,yticklabels=[],xlabel='Native output tokens / second',title='Decode calls',xlim=(0,62))
    for ax in axes:ax.invert_yaxis();ax.grid(axis='x',alpha=.12);ax.set_axisbelow(True)
    handles,legends=axes[0].get_legend_handles_labels()
    fig.legend(handles,legends,loc='lower left',bbox_to_anchor=(.065,.105),ncol=3,frameon=False)
    save(fig,'external-integrity','What the per-write integrity check costs.',
         '8K saved prefix + 512-token A suffix + 128 outputs | Q8_0 / FP16 KV | same physical GPU and resources',
         'Caller-managed variants keep ordered range/metadata/COW checks but trust the caller to prevent sealed-page mutations.\n'
         'Independent logits and parent audits stay enabled. Single serial samples; native clock excludes validation and durable publication.')
    base = read('external-integrity-20261013/v011-guarded-control.json')
    candidate = read('external-integrity-20261013/v011-external-control.json')
    assert base['status']==candidate['status']=='PASS'
    assert base['fixture_sha256']==candidate['fixture_sha256']
    assert base['model_identity']==candidate['model_identity']
    assert base['checkpoint']==candidate['checkpoint']
    comparison = dict(status='PASS', scope='bounded paired measurement; not gate certification',
                      baseline_policy=base['write_integrity_policy'],candidate_policy=candidate['write_integrity_policy'],
                      fixture_sha256=base['fixture_sha256'],numerical_max_error=max(base['numerical_max_error'],candidate['numerical_max_error']),
                      requests=[])
    for a,b in zip(base['requests'],candidate['requests']):
        assert (a['lane'],a['label'])==(b['lane'],b['label'])
        assert a['ids']==b['ids'] and a['parent_audit']==b['parent_audit']
        assert a['counters']['copied_bytes']==b['counters']['copied_bytes']
        comparison['requests'].append(dict(lane=a['lane'],label=a['label'],native_tps_baseline=a['native_call_tps'],
                                           native_tps_candidate=b['native_call_tps'],native_tps_gain_percent=100*(b['native_call_tps']/a['native_call_tps']-1),
                                           prepare_seconds_baseline=a['native_decode_seconds']['prepare'],prepare_seconds_candidate=b['native_decode_seconds']['prepare'],
                                           api_tps_baseline=a['instrumented_api_tps'],api_tps_candidate=b['instrumented_api_tps'],
                                           decode_integrity_baseline=a['decode_integrity'],decode_integrity_candidate=b['decode_integrity']))
    fig,axes=plt.subplots(1,2,figsize=(13,6.5))
    fig.subplots_adjust(left=.12,right=.95,top=.79,bottom=.23,wspace=.25)
    y=np.arange(4);labels=['Resident A','Resident B','RAM A','Disk B']
    for ax,key,title,xlabel in [(axes[0],'native_call_tps','Native throughput','Output tokens / second'),
                                (axes[1],'prepare','Preparation cost','Milliseconds over 128 outputs')]:
        for offset,records,name,color in [(-.16,base['requests'],'v0.1.0 guarded',NAVY),(.16,candidate['requests'],'v0.1.1 caller-managed + access reuse',TEAL)]:
            values=[r[key] if key=='native_call_tps' else r['native_decode_seconds'][key]*1000 for r in records]
            bars=ax.barh(y+offset,values,height=.29,color=color,label=name)
            ax.bar_label(bars,labels=[f'{v:.2f}' for v in values],padding=3,fontsize=9)
        ax.set(yticks=y,yticklabels=labels if ax is axes[0] else [],title=title,xlabel=xlabel)
        ax.invert_yaxis();ax.grid(axis='x',alpha=.12);ax.set_axisbelow(True)
        ax.set_xlim(0,62 if key=='native_call_tps' else 225)
    handles,legends=axes[0].get_legend_handles_labels();fig.legend(handles,legends,loc='lower left',bbox_to_anchor=(.065,.105),ncol=2,frameon=False,fontsize=10)
    save(fig,'v011-paired','The fresh baseline rerun confirms the gain.',
         'Same GPU, pinned binaries/resources, token fixture and independent oracle | 516 exact comparisons per run',
         '6.4–6.6% native gain on COW lanes; 0.47% on RAM is a small observed difference, not a statistically established gain.\n'
         'Caller-managed mode transfers the sealed-byte integrity promise. Both runs retain the same copy counters and parent audits.')
    (OUT/'PAIRED_COMPARISON.json').write_text(json.dumps(comparison,indent=2)+'\n',newline='\n')
    (OUT/'EXTERNAL_PROVENANCE.json').write_text(json.dumps(dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),sources=sources),indent=2)+'\n',newline='\n')


if __name__=='__main__':
    main()
