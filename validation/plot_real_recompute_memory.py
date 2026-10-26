"""Plot required graph buffers; refused buffers were never allocated."""
from pathlib import Path
import json
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent.parent
report=json.loads((root/'validation/attention-recompute-real-memory-20261026.json').read_text(encoding='utf8'))
variants=['original','selected_head','shared_mask','recompute']
labels=['Original','Selected\nhead','Shared\nmask','Attention\nrecompute']
colors=['#7b8087','#4678ad','#228c96','#2a8550']
cases=['full-3663','lora16-4873','full-4873']
fig,axes=plt.subplots(1,3,figsize=(15,6.5),sharey=True)
fig.subplots_adjust(top=.77,bottom=.22,left=.065,right=.99,wspace=.12)
for axis,case in zip(axes,cases):
    rows={r['variant']:r for r in report['comparison']if r['case']==case}
    for i,key in enumerate(variants):
        row=rows[key];value=row['planned_graph_bytes']/(1<<30);passed=row['status']=='PASS'
        axis.bar(i,value,color=colors[i],width=.67,hatch=None if passed else'///',edgecolor='white')
        axis.text(i,value-2 if passed else value+1.7,f'{value:.1f}\n'+('PASS'if passed else'REFUSED'),
                  ha='center',va='top'if passed else'bottom',fontsize=9,color='white'if passed else'black')
    budget=rows['recompute']['graph_limit_bytes']/(1<<30)
    axis.axhline(budget,color='#333333',ls=':',lw=1.4)
    axis.text(.98,budget+1.3,f'Graph budget {budget:.1f} GiB',ha='right',transform=axis.get_yaxis_transform(),fontsize=9,color='#333333',bbox=dict(facecolor='white',edgecolor='none',pad=1))
    axis.set_title(rows['original']['title'],fontsize=12,pad=12)
    axis.set_xticks(range(4),labels,fontsize=10);axis.set_ylim(0,115)
    axis.grid(axis='y',alpha=.2);axis.set_axisbelow(True);axis.spines[['top','right']].set_visible(False)
axes[0].set_ylabel('Planned graph buffer (GiB)',fontsize=11)
fig.suptitle('MiniCPM native backprop: keeping the full lesson while reducing saved attention',fontsize=16,fontweight='bold',y=.96)
fig.text(.5,.89,'Same unchanged learned parent and exact examples | dedicated 96 GB-class RTX PRO 6000',ha='center',fontsize=11)
fig.text(.5,.06,'Graph buffers exclude weights, persistent parameter gradients, optimizer state, scratch and reserve. Dotted limits are native admission budgets.\n'
    'Hatched bars were refused before allocation. PASS means a completed gradient probe, not an optimizer update or release acceptance.\n'
    'Intermediate improvements are cumulative. No context truncation, precision reduction or end-to-end speed claim.',ha='center',fontsize=9,color='#555555')
fig.savefig(root/'docs/results/attention-recompute-real-memory-20261026.png',dpi=160,facecolor='white')
plt.close(fig)
