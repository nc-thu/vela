"""Single-column latency comparison, with the reported model DSP budgets.
Conclusion: grouped W8 execution lowers modeled latency relative to FP16;
mechanism comparisons distinguish packing, shared HNU, and snapshot/merge cost.
Horizontal bars show latency; aligned columns show the resource budgets.
No experimental replicates or error bars: deterministic model results.
"""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import fitz

P=Path(__file__).resolve().parent
data=json.loads((P/'source_table.json').read_text(encoding='utf-8'))
rows=data['rows'];full=rows[-1]['ms']
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial'],
 'font.size':7,'pdf.fonttype':42,'svg.fonttype':'none','axes.linewidth':.7})
fig=plt.figure(figsize=(3.5,3.35),facecolor='white')
ax=fig.add_axes([.08,.23,.61,.65])
resources=fig.add_axes([.71,.23,.27,.65],sharey=ax)
y=np.arange(5)[::-1]
colors=['#B4B4B4','#C9B48C','#DC7000','#8298AC','#287D94']
labels=['H1  FP16 baseline','H1  Group stall + Merge24',
        'H2  One product / DSP','H3  Dedicated nonlinear',
        'Full design: W8A8 G64']
for i,(yy,row,label,color) in enumerate(zip(y,rows,labels,colors)):
    ax.barh(yy,row['ms'],height=.30,color=color,edgecolor='black',linewidth=.6,zorder=3)
    ax.text(0,yy+.24,label,fontsize=7,weight='bold',va='bottom')
    reduction=(1-full/row['ms'])*100
    value=f"{row['ms']:.1f} ms"+(f"   (−{reduction:.1f}%)" if i<4 else '')
    ax.text(12,yy,value,va='center',fontsize=6.8,color='white' if i==4 else 'black',zorder=4)
    resources.text(.22,yy,str(row['merge_dsp']),ha='center',va='center',fontsize=7)
    resources.text(.76,yy,str(row['total_dsp']),ha='center',va='center',fontsize=7)
ax.set_xlim(0,620);ax.set_ylim(-.42,4.63)
ax.set_xticks([0,200,400,600]);ax.set_xlabel('Modeled inference latency (ms)',labelpad=3)
ax.set_yticks([]);ax.grid(axis='x',color='#DDDDDD',linewidth=.5,zorder=0)
for side in ['left','right','top']:ax.spines[side].set_visible(False)
resources.set_xlim(0,1);resources.axis('off')
resources.text(.22,4.65,'Merge\nDSP',ha='center',va='bottom',fontsize=6.7,weight='bold')
resources.text(.76,4.65,'Model\nDSP',ha='center',va='bottom',fontsize=6.7,weight='bold')
fig.text(.08,.965,'Full-design latency reduction vs. each reference',fontsize=7,weight='bold',va='top')
fig.text(.08,.043,'Reduction = (reference − full design) / reference.\nFP16 evaluates the overall quantization framework.',fontsize=6.4,va='bottom')
for ext in ['pdf','svg','png']:
    fig.savefig(P/f'fig11_ablation.{ext}',dpi=400,facecolor='white')
plt.close(fig)
doc=fitz.open(P/'fig11_ablation.pdf')
doc[0].get_pixmap(dpi=300).save(P/'fig11_pdf_preview.png')
assert not doc[0].get_images()
assert 'FP16' in doc[0].get_text()
(P/'checks.json').write_text(json.dumps({'figure_size_inches':[3.5,3.35],
 'pdf_raster_images':len(doc[0].get_images()),'reduction_pct':[(1-full/r['ms'])*100 for r in rows],
 'merge_dsp_reduction_pct':75.0,'model_dsp_reduction_pct':(1-905/977)*100},indent=2))
print('Wrote vector PDF/SVG and 400-dpi PNG.')
