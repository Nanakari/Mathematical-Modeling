"""Frozen-test diagnostics, video-cluster bootstrap and formal answer."""
import csv
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,f1_score
from p2 import metrics

ROOT=Path(__file__).resolve().parent


def readcsv(path):
    with path.open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    for r in rows:
        for k in r:
            if k not in ('model','scenario','modalities','position'):
                r[k]=float(r[k]) if r[k] else None
    return rows


def writecsv(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def bootstrap(anchors,repeats=500):
    ids=anchors['ids'];y=np.array(anchors['true_class']);r=np.array(anchors['true_intensity'])
    videos={sid.split('$_$')[0] for sid in ids};videos=sorted(videos)
    groups=[np.array([i for i,sid in enumerate(ids) if sid.split('$_$')[0]==v]) for v in videos]
    rng=np.random.default_rng(20260924);out=[]
    for scenario in ['complete']+[k.split('::',1)[1] for k in anchors['predictions'] if k.startswith('proposed::') and not k.endswith('complete')]:
        proposed=anchors['predictions']['proposed::'+scenario]
        single=anchors['predictions']['proposed_seed17::'+scenario]
        baseline=anchors['predictions']['no_augmentation::'+scenario]
        pp=np.array(proposed['probabilities']).argmax(1);pr=np.array(proposed['intensity'])
        sp=np.array(single['probabilities']).argmax(1);sr=np.array(single['intensity'])
        bp=np.array(baseline['probabilities']).argmax(1);br=np.array(baseline['intensity'])
        values=[]
        for _ in range(repeats):
            ix=np.concatenate([groups[j] for j in rng.integers(0,len(groups),len(groups))])
            f=lambda p: f1_score(y[ix],p[ix],labels=[0,1,2],average='macro',zero_division=0)
            values.append([np.mean(pp[ix]==y[ix]),f(pp),np.mean(abs(pr[ix]-r[ix])),
                           f(sp)-f(bp),np.mean(abs(sr[ix]-r[ix]))-np.mean(abs(br[ix]-r[ix]))])
        values=np.array(values)
        for i,name in enumerate(['proposed_accuracy','proposed_macro_f1','proposed_mae','augmentation_delta_macro_f1','augmentation_delta_mae']):
            lo,hi=np.quantile(values[:,i],[.025,.975])
            out.append({'scenario':scenario,'quantity':name,'ci_low':float(lo),'ci_high':float(hi),'clusters':len(videos),'bootstrap_repeats':repeats})
    return out


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',default='outputs/formal');args=parser.parse_args()
    out=ROOT/args.run_dir
    done=json.loads((out/'evaluation_complete.json').read_text(encoding='utf-8'))
    cfg=json.loads((out/'protocol.json').read_text(encoding='utf-8'))
    frozen=json.loads((out/'frozen_selection.json').read_text(encoding='utf-8'))
    anchors=json.loads((out/'test_anchors.json').read_text(encoding='utf-8'))
    rows=readcsv(out/'test_metrics.csv');groups=frozen['model_groups']
    ci=bootstrap(anchors,cfg['bootstrap_repeats']);writecsv(out/'bootstrap_confidence_intervals.csv',ci)
    condition_rows=[]
    for model in groups:
        subset=[r for r in rows if r['model']==model and r['scenario']!='complete']
        keys=sorted({(r['modalities'],r['rate'],r['position']) for r in subset})
        for combo,rate,position in keys:
            selected=[r for r in subset if (r['modalities'],r['rate'],r['position'])==(combo,rate,position)]
            condition_rows.append({'model':model,'modalities':combo,'rate':rate,'position':position,'mask_repeats':len(selected),
                **{k:float(np.mean([r[k] for r in selected])) for k in ['accuracy','f1_macro','f1_weighted','mae']},
                'mae_mask_std':float(np.std([r['mae'] for r in selected],ddof=1)) if len(selected)>1 else 0.})
    writecsv(out/'condition_summary.csv',condition_rows)
    robustness=[]
    for model in groups:
        complete=next(r for r in rows if r['model']==model and r['scenario']=='complete')
        conditions=[r for r in condition_rows if r['model']==model]
        robustness.append({'model':model,'complete_accuracy':complete['accuracy'],'complete_macro_f1':complete['f1_macro'],
                           'complete_mae':complete['mae'],'complete_pearson':complete['pearson'],
                           'condition_mean_mae':float(np.mean([r['mae'] for r in conditions])),
                           'condition_mean_macro_f1':float(np.mean([r['f1_macro'] for r in conditions])),
                           'worst_condition_mae':max(r['mae'] for r in conditions)})
    writecsv(out/'robustness_summary.csv',robustness)
    pred=anchors['predictions']['proposed::complete'];p=np.array(pred['probabilities']);z=np.array(pred['intensity'])
    y=np.array(anchors['true_class']);r=np.array(anchors['true_intensity']);pc=p.argmax(1)
    errors=[{'sample_id':sid,'true_class':int(y[i]),'pred_class':int(pc[i]),'true_intensity':float(r[i]),
             'pred_intensity':float(z[i]),'absolute_error':float(abs(z[i]-r[i])),
             'class_regression_disagree':int(pc[i])!=int(np.sign(z[i])+1)} for i,sid in enumerate(anchors['ids'])]
    writecsv(out/'test_error_cases.csv',sorted(errors,key=lambda a:a['absolute_error'],reverse=True)[:30])
    writecsv(out/'test_error_by_class.csv',[{'true_class':c,'n':int((y==c).sum()),
        'mae':float(np.mean(abs(z[y==c]-r[y==c]))),'class_accuracy':float(np.mean(pc[y==c]==y[y==c]))} for c in [0,1,2]])
    fig,axs=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    mat=confusion_matrix(y,pc,labels=[0,1,2]);axs[0].imshow(mat,cmap='Blues')
    for i in range(3):
        for j in range(3):axs[0].text(j,i,str(mat[i,j]),ha='center',va='center')
    axs[0].set(xticks=[0,1,2],yticks=[0,1,2],xlabel='Predicted polarity',ylabel='True polarity',title='Independent test: complete input')
    axs[1].scatter(r,z,s=9,alpha=.4);axs[1].plot([-3,3],[-3,3],'k--',lw=1)
    axs[1].set(xlabel='True intensity',ylabel='Predicted intensity',xlim=(-3,3),ylim=(-3,3),title='Independent test: intensity')
    fig.savefig(out/'test_diagnostics.png',dpi=170);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for model in ['proposed','proposed_seed17','no_augmentation','no_semantics','text_only']:
        base=next(x for x in rows if x['model']==model and x['scenario']=='complete')
        ss=[x for x in condition_rows if x['model']==model and x['modalities']=='text+audio+vision' and x['position']=='random']
        ss=sorted(ss,key=lambda x:x['rate'])
        for ax,key in zip(axs,['mae','f1_macro']):
            ax.plot([0]+[s['rate'] for s in ss],[base[key]]+[s[key] for s in ss],marker='.',label=model)
            ax.set(xlabel='Requested local missing ratio',ylabel=key,title='All modalities: mean over three mask seeds');ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.savefig(out/'test_robustness_curves.png',dpi=170);plt.close(fig)
    combos=['text','audio','vision','text+audio','text+vision','audio+vision','text+audio+vision'];positions=cfg['positions']
    heat=np.array([[next(x['mae'] for x in condition_rows if x['model']=='proposed' and x['modalities']==m and x['position']==pos and x['rate']==.3) for pos in positions] for m in combos])
    fig,ax=plt.subplots(figsize=(8,5),layout='constrained');im=ax.imshow(heat,cmap='YlOrRd',aspect='auto')
    ax.set(xticks=range(4),xticklabels=positions,yticks=range(7),yticklabels=combos,title='Independent test: modality/position at 30% missingness')
    for i in range(7):
        for j in range(4):ax.text(j,i,f'{heat[i,j]:.3f}',ha='center',va='center')
    fig.colorbar(im,ax=ax,label='MAE');fig.savefig(out/'test_position_heatmap.png',dpi=170);plt.close(fig)
    from formal_answer import write_answer
    write_answer(out,cfg,frozen,done,anchors,robustness,condition_rows,ci)
    print('Formal analysis and answer written',flush=True)


if __name__=='__main__':main()
