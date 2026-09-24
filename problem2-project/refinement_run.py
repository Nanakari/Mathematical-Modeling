"""Video-group OOF selection; frozen official validation; attachment 3 inference."""
import argparse,copy,csv,itertools,json,pickle,time,warnings
from pathlib import Path
import joblib
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits
from p2 import DATA,metrics
from formal_features import FeatureMap,SemanticEncoder,combine
from formal_model import ensemble
from formal_run import ROOT,js,sha,dataset,sample_list,augmented,scenario,measure
from aligned_dataset import AlignedDataset
from refinement_model import FUSIONS,fit_head,adjusted_probabilities,calibrated_intensity,RefinedModel

def candidates(cfg):
    cls=[];reg=[]
    for fusion,mass in itertools.product(cfg['fusions'],cfg['clean_mass']):
        cls.extend(dict(fusion=fusion,clean_mass=mass,C=c) for c in cfg['classification_C'])
        reg.extend(dict(fusion=fusion,clean_mass=mass,kind='ridge',alpha=a) for a in cfg['ridge_alpha'])
        reg.extend(dict(fusion=fusion,clean_mass=mass,kind='absolute_svr',C=c) for c in cfg['absolute_svr_C'])
    return cls,reg

def cases(cfg):
    result=[('complete',(),0.,'start')]
    combos=[c for n in [1,2,3] for c in itertools.combinations(('text','audio','vision'),n)]
    for combo,pos in itertools.product(combos,['start','middle','end','random']):
        result.append(('+'.join(combo)+'|'+pos+'|0.3',combo,.3,pos))
    for rate in [.1,.5,.7]:result.append((f'all|random|{rate}',('text','audio','vision'),rate,'random'))
    return result

def weights(cfg):
    n=len(cases(cfg));return np.array([cfg['complete_selection_weight']]+[(1-cfg['complete_selection_weight'])/(n-1)]*(n-1))

def changed(samples,case,cfg):
    _,combo,rate,pos=case
    if not combo:return samples
    return scenario(samples,combo,rate,pos,cfg['mask_seed'],[.1,.3,.5,.7])[0]

def writecsv(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def select(cfg,out):
    if (out/'frozen_selection.json').exists():raise ValueError('Already frozen. Use another output for another protocol.')
    out.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists():
        if json.loads((out/'protocol.json').read_text(encoding='utf-8'))!=cfg:raise ValueError('Resume protocol mismatch')
    else:js(out/'protocol.json',cfg)
    code_files=['refinement_run.py','refinement_model.py','formal_features.py','formal_run.py','pipeline.py','aligned_dataset.py']
    code_hashes={n:sha(ROOT/n) for n in code_files}
    if (out/'source_hashes.json').exists():
        if json.loads((out/'source_hashes.json').read_text())!=code_hashes:raise ValueError('Source changed during selection')
    else:js(out/'source_hashes.json',code_hashes)
    with (DATA/'附件2-数据集特征文件/aligned_50.pkl').open('rb') as f:raw=pickle.load(f)
    ds=dataset(raw['train'],'train');del raw
    samples=sample_list(ds);y=ds.part['classification_labels'].astype(int);r=ds.part['regression_labels']
    ids=list(map(str,ds.part['id']));groups=np.array([s.split('$_$')[0] for s in ids])
    cv=StratifiedGroupKFold(n_splits=cfg['folds'],shuffle=True,random_state=cfg['fold_seed'])
    splits=list(cv.split(np.zeros(len(y)),y,groups));encoder=SemanticEncoder(ROOT/cfg['encoder'])
    sem=encoder.encode(samples);cs,rs=candidates(cfg);design=cases(cfg)
    cp=np.zeros((len(cs),len(design),len(y),3),dtype=np.float64)
    rp=np.zeros((len(rs),len(design),len(y)),dtype=np.float64)
    fold_of=np.full(len(y),-1);manifest=[]
    for fold,(tr,va) in enumerate(splits):
        assert not set(groups[tr])&set(groups[va]);fold_of[va]=fold
        manifest.append({'fold':fold,'train_ids':[ids[i] for i in tr],'holdout_ids':[ids[i] for i in va],
                         'train_videos':len(set(groups[tr])),'holdout_videos':len(set(groups[va]))})
        cache=out/f'fold_{fold}_predictions.npz'
        if cache.exists():
            with np.load(cache) as saved:
                np.testing.assert_array_equal(saved['indices'],va);cp[:,:,va]=saved['classification'];rp[:,:,va]=saved['regression']
            print('Resumed fold',fold,flush=True);continue
        start=time.time();ts=[samples[i] for i in tr];vs=[samples[i] for i in va]
        mapper=FeatureMap().fit(ts,sem[tr]);aug,_=augmented(ts,cfg['augmentation_seed'],3,cfg['augmentation_rates'])
        blocks=mapper.blocks(aug,encoder.encode(aug))
        classifiers=[];regressors=[]
        for i,spec in enumerate(cs):
            classifiers.append(fit_head(blocks,y[tr],spec,'classification'))
            if i%6==0:print('Fold',fold,'classifier',i+1,'/',len(cs),'seconds',round(time.time()-start),flush=True)
        for i,spec in enumerate(rs):
            regressors.append(fit_head(blocks,r[tr],spec,'regression'))
            if i%10==0:print('Fold',fold,'regressor',i+1,'/',len(rs),'seconds',round(time.time()-start),flush=True)
        del blocks,aug
        for j,case in enumerate(design):
            ss=changed(vs,case,cfg);b=mapper.blocks(ss,encoder.encode(ss));xx={k:combine(b,**FUSIONS[k]) for k in cfg['fusions']}
            for i,(spec,model) in enumerate(zip(cs,classifiers)):cp[i,j,va]=model.predict_proba(xx[spec['fusion']])
            for i,(spec,model) in enumerate(zip(rs,regressors)):rp[i,j,va]=model.predict(xx[spec['fusion']])
            if j%8==0:print('Fold',fold,'scenario',j+1,'/',len(design),flush=True)
        np.savez_compressed(cache,indices=va,classification=cp[:,:,va],regression=rp[:,:,va])
        print('Fold complete',fold,'seconds',round(time.time()-start),flush=True)
    assert (fold_of>=0).all();js(out/'fold_manifest.json',manifest)
    w=weights(cfg);class_search=[];reg_search=[]
    # Fast macro-F1 from confusion counts, avoiding repeated sklearn overhead.
    def f1(pred):
        cm=np.bincount(y*3+pred,minlength=9).reshape(3,3)
        den=cm.sum(0)+cm.sum(1)
        return float(np.divide(2*np.diag(cm),den,out=np.zeros(3,dtype=float),where=den>0).mean())
    for i,spec in enumerate(cs):
        for neutral in cfg['neutral_weight']:
            scores=[f1(adjusted_probabilities(p,neutral).argmax(1)) for p in cp[i]]
            class_search.append({'index':i,'spec':spec,'neutral_weight':neutral,'weighted_macro_f1':float(w@scores),'scenario_scores':scores})
    for i,spec in enumerate(rs):
        for scale,offset in itertools.product(cfg['regression_scale'],cfg['regression_offset']):
            scores=np.abs(calibrated_intensity(rp[i],scale,offset)-r[None,:]).mean(1)
            reg_search.append({'index':i,'spec':spec,'scale':scale,'offset':offset,'weighted_mae':float(w@scores),'scenario_scores':scores.tolist()})
    best_c=max(class_search,key=lambda x:x['weighted_macro_f1']);best_r=min(reg_search,key=lambda x:x['weighted_mae'])
    js(out/'classification_search.json',class_search);js(out/'regression_search.json',reg_search)
    # CV scores after selection are explicitly not unbiased final performance estimates.
    old_ci=cs.index(dict(fusion='av03_quality',clean_mass=.25,C=.5))
    old_ri=rs.index(dict(fusion='av03_quality',clean_mass=.25,kind='ridge',alpha=10.))
    old_p=cp[old_ci];old_z=np.clip(rp[old_ri],-3,3)
    new_p=np.stack([adjusted_probabilities(p,best_c['neutral_weight']) for p in cp[best_c['index']]])
    new_z=calibrated_intensity(rp[best_r['index']],best_r['scale'],best_r['offset'])
    cv_rows=[]
    for name,pp,zz in [('v1_protocol_refit',old_p,old_z),('v2_selected',new_p,new_z)]:
        for j,case in enumerate(design):cv_rows.append({'model':name,'scenario':case[0],**metrics(y,r,pp[j].argmax(1),zz[j])})
    js(out/'cv_comparison.json',cv_rows)
    np.savez_compressed(out/'selected_oof.npz',classes=y,intensity=r,fold=fold_of,old_probabilities=old_p,old_intensity=old_z,new_probabilities=new_p,new_intensity=new_z)
    frozen={'classification':best_c,'regression':best_r,'train_n':len(y),'train_ids':ids,'train_videos':len(set(groups)),
            'protocol_sha256':sha(out/'protocol.json'),'source_hashes':code_hashes,'scenario_names':[c[0] for c in design],
            'classification_candidates':len(class_search),'regression_candidates':len(reg_search),
            'official_valid_used_for_selection':False,'official_test_used_for_selection':False,
            'prior_test_informed_research_direction':True,'oof_scores_are_selection_scores':True}
    js(out/'frozen_selection.json',frozen)
    print('FROZEN classification',best_c['spec'],'neutral',best_c['neutral_weight'],'score',best_c['weighted_macro_f1'],flush=True)
    print('FROZEN regression',best_r['spec'],'scale/offset',best_r['scale'],best_r['offset'],'score',best_r['weighted_mae'],flush=True)

def fit_and_validate(cfg,out):
    if (out/'validation_complete.json').exists():raise ValueError('Validation already completed')
    frozen=json.loads((out/'frozen_selection.json').read_text(encoding='utf-8'))
    assert frozen['protocol_sha256']==sha(out/'protocol.json')
    for name,digest in frozen['source_hashes'].items():assert sha(ROOT/name)==digest,name
    with (DATA/'附件2-数据集特征文件/aligned_50.pkl').open('rb') as f:raw=pickle.load(f)
    ds=dataset(raw['train'],'train');vd=dataset(raw['valid'],'valid');del raw
    train=sample_list(ds);valid=sample_list(vd);y=ds.part['classification_labels'].astype(int);r=ds.part['regression_labels']
    vy=vd.part['classification_labels'].astype(int);vr=vd.part['regression_labels']
    assert not {str(s).split('$_$')[0] for s in ds.part['id']}&{str(s).split('$_$')[0] for s in vd.part['id']}
    encoder=SemanticEncoder(ROOT/cfg['encoder']);sem=encoder.encode(train);mapper=FeatureMap().fit(train,sem)
    aug,_=augmented(train,cfg['augmentation_seed'],3,cfg['augmentation_rates']);blocks=mapper.blocks(aug,encoder.encode(aug))
    c=frozen['classification'];z=frozen['regression']
    selected=RefinedModel(c['spec'],z['spec'],c['neutral_weight'],z['scale'],z['offset']).fit(blocks,y,r)
    models={'v2_selected':selected}
    # One change at a time relative to the selected model; no validation retuning.
    m=copy.deepcopy(selected);m.neutral_weight=1.;models['default_neutral_rule']=m
    m=copy.deepcopy(selected);m.scale=1.;m.offset=0.;models['no_regression_calibration']=m
    m=copy.deepcopy(selected);m.reg_spec={**z['spec'],'kind':'ridge','alpha':10.};m.scale=1.;m.offset=0.
    m.regressor=fit_head(blocks,r,m.reg_spec,'regression');models['ridge10_regression']=m
    m=copy.deepcopy(selected);m.class_spec['fusion']='av03_quality';m.reg_spec['fusion']='av03_quality'
    m.fit(blocks,y,r);models['shared_v1_fusion']=m
    m=copy.deepcopy(selected);m.class_spec['clean_mass']=.25;m.reg_spec['clean_mass']=.25
    m.fit(blocks,y,r);models['old_augmentation_mass']=m
    joblib.dump(mapper,out/'feature_map.joblib',compress=3);joblib.dump(models,out/'models.joblib',compress=3)
    js(out/'model_manifest.json',{'feature_map_sha256':sha(out/'feature_map.joblib'),'models_sha256':sha(out/'models.joblib'),
        'selection_sha256':sha(out/'frozen_selection.json'),'model_names':list(models)})
    baseline_map=joblib.load(ROOT/'outputs/formal/feature_map.joblib');baseline=joblib.load(ROOT/'outputs/formal/models.joblib')
    rows=[];stored={};start=time.time()
    for j,case in enumerate(cases(cfg)):
        ss=changed(valid,case,cfg);sem=encoder.encode(ss);b=mapper.blocks(ss,sem);bb=baseline_map.blocks(ss,sem)
        predictions={name:model.predict(b) for name,model in models.items()}
        predictions['v1_seed17']=ensemble(baseline['proposed_seed17'],bb)
        predictions['v1_ensemble']=ensemble(baseline['proposed'],bb)
        for name,(p,zp) in predictions.items():
            row={'model':name,'scenario':case[0],**metrics(vy,vr,p.argmax(1),zp)}
            row['neutral_recall']=float(np.mean(p.argmax(1)[vy==1]==1))
            row['strong_mae']=float(np.abs(zp[abs(vr)>=2]-vr[abs(vr)>=2]).mean())
            row['opposite_sign_count']=int((((p.argmax(1)==0)&(zp>0))|((p.argmax(1)==2)&(zp<0))).sum())
            rows.append(row);stored[name+'::'+case[0]]={'probabilities':p.tolist(),'intensity':zp.tolist()}
        if j%4==0:print('Official validation',j+1,'/',len(cases(cfg)),'seconds',round(time.time()-start),flush=True)
    js(out/'validation_metrics.json',rows)
    joblib.dump({'ids':list(map(str,vd.part['id'])),'classes':vy,'intensity':vr,'predictions':stored},out/'validation_predictions.joblib',compress=3)
    summary=[];w=weights(cfg)
    for name in list(models)+['v1_seed17','v1_ensemble']:
        rr=[row for row in rows if row['model']==name]
        summary.append({'model':name,'weighted_macro_f1':float(w@[row['f1_macro'] for row in rr]),
            'weighted_mae':float(w@[row['mae'] for row in rr]),'complete_accuracy':rr[0]['accuracy'],
            'complete_macro_f1':rr[0]['f1_macro'],'complete_mae':rr[0]['mae'],'complete_pearson':rr[0]['pearson'],
            'complete_neutral_recall':rr[0]['neutral_recall'],'complete_strong_mae':rr[0]['strong_mae'],
            'complete_opposite_sign_count':rr[0]['opposite_sign_count']})
    writecsv(out/'validation_summary.csv',summary)
    special=AlignedDataset.from_attachment3(DATA/'附件3-模态缺失特征样本/对齐版本',special_token_ids=(101,102))
    ss=sample_list(special);p,zp=selected.predict(mapper.blocks(ss,encoder.encode(ss)))
    export_predictions(out/'附件3_问题2_优化版预测.csv',ss,p,zp)
    js(out/'validation_complete.json',{'valid_n':len(vy),'scenarios':len(cases(cfg)),'model_groups':len(summary),
        'special_n':len(ss),'selection_sha256':sha(out/'frozen_selection.json'),'official_test_evaluated':False,
        'probability_note':'neutral-reweighted normalized decision scores; not claimed calibrated posterior probabilities'})
    print('Validation and special inference complete',flush=True)

def export_predictions(path,samples,p,z):
    rows=[{'sample_id':s['sample_id'],'source_file':s['sample_id'].split('#')[0],
           'polarity':['Negative','Neutral','Positive'][int(p[i].argmax())],'intensity':float(z[i]),
           **{f'decision_score_{j}':float(p[i,j]) for j in range(3)}} for i,s in enumerate(samples)]
    writecsv(path,rows)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['select','validate']);ap.add_argument('--config',default='configs/refinement_protocol.json');args=ap.parse_args()
    cfg=json.loads((ROOT/args.config).read_text(encoding='utf-8'));out=ROOT/cfg['output']
    with threadpool_limits(limits=4):
        (select if args.stage=='select' else fit_and_validate)(cfg,out)

if __name__=='__main__':main()
