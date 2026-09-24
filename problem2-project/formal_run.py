"""Full-data validation selection, frozen independent test, ablations and inference."""
import argparse
import copy
import csv
import gzip
import hashlib
import itertools
import json
import pickle
import time
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from aligned_dataset import AlignedDataset
from pipeline import make_plan,apply_plan
from p2 import DATA,metrics
from formal_features import FeatureMap,SemanticEncoder,combine
from formal_model import DualModel,ensemble,fit_classifier,fit_regressor

ROOT=Path(__file__).resolve().parent
MODS=('text','audio','vision')


def js(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dataset(part,split,labeled=True):
    return AlignedDataset(AlignedDataset._select(part),labeled=labeled,source=f'attachment2/{split}',special_token_ids=(101,102))


def sample_list(ds):return [ds[i] for i in range(len(ds))]


def scenario(samples,combo,rate,position,seed,rates):
    changed=[];records=[]
    for s in samples:
        p=make_plan(s,combo,rate,position,seed,rates=rates if rate else None)
        changed.append(apply_plan(s,p));records.append(p)
    return changed,records


def augmented(samples,seed,views,rates):
    combinations=[c for n in (1,2,3) for c in itertools.combinations(MODS,n)]
    result=list(samples);records=[]
    for view in range(views):
        rate=rates[view%len(rates)]
        for s in samples:
            stable=int.from_bytes(hashlib.sha256(f'{seed}|{view}|{s["sample_id"]}'.encode()).digest()[:8],'little')
            rng=np.random.default_rng(stable)
            combo=combinations[int(rng.integers(len(combinations)))];pos=str(rng.choice(['start','middle','end','random']))
            p=make_plan(s,combo,rate,pos,seed+view,rates=rates)
            result.append(apply_plan(s,p));records.append(p)
    return result,records


def measure(classes,regression,prediction):
    probs,reg=prediction
    return metrics(classes,regression,probs.argmax(1),reg)


def scalar(scores):return float(np.mean([s['mae']+1-s['f1_macro'] for s in scores]))


def selection_cases(valid,encoder,mapper,cfg):
    cases={}
    for name in cfg['selection_scenarios']:
        if name=='complete':samples=valid
        else:
            m,rate=name.split('_');combo=MODS if m=='all' else (m,)
            samples,_=scenario(valid,combo,float(rate),'random',cfg['validation_mask_seed'],cfg['rates'])
        cases[name]=mapper.blocks(samples,encoder.encode(samples))
    return cases


def train(cfg,out):
    if (out/'frozen_selection.json').exists():raise ValueError('Selection already frozen; use new output for a distinct protocol')
    started=time.time();out.mkdir(parents=True,exist_ok=True);js(out/'protocol.json',cfg)
    with (DATA/'附件2-数据集特征文件'/cfg['feature_version']).open('rb') as f:raw=pickle.load(f)
    from verify_tokenizer import verify
    js(out/'tokenizer_verification.json',verify(raw,ROOT/cfg['text_model_dir']))
    ds_train,ds_valid=dataset(raw['train'],'train'),dataset(raw['valid'],'valid');del raw
    train_samples,valid_samples=sample_list(ds_train),sample_list(ds_valid)
    y=ds_train.part['classification_labels'].astype(int);r=ds_train.part['regression_labels']
    vy=ds_valid.part['classification_labels'].astype(int);vr=ds_valid.part['regression_labels']
    encoder=SemanticEncoder(ROOT/cfg['text_model_dir'])
    semantic=encoder.encode(train_samples)
    mapper=FeatureMap(cfg['text_ngram_max_features'],cfg['text_ngram_min_df']).fit(train_samples,semantic)
    clean=mapper.blocks(train_samples,semantic);cases=selection_cases(valid_samples,encoder,mapper,cfg)
    joblib.dump(mapper,out/'feature_map.joblib',compress=3)
    print(f'Train={len(y)} valid={len(vy)} lexical={len(mapper.lexical.vocabulary_)}; semantic encoder verified',flush=True)
    first_seed=cfg['train_seeds'][0]
    train_aug,records=augmented(train_samples,first_seed,cfg['augmentation_views'],cfg['rates'])
    with gzip.open(out/'training_masks_seed17.jsonl.gz','wt',encoding='utf-8') as f:
        for rec in records:f.write(json.dumps(rec)+'\n')
    blocks=mapper.blocks(train_aug,encoder.encode(train_aug))
    repeats=cfg['augmentation_views']+1
    ay,ar=np.tile(y,repeats),np.tile(r,repeats);weights=np.full(len(ay),1/repeats)
    tuning=[];best=None
    for av_weight in cfg['av_weight_grid']:
        feature_cfg={'av_weight':av_weight,'semantic_weight':cfg['semantic_weight'],'reliability':True,'modalities':MODS,'semantic':True}
        x=combine(blocks,**feature_cfg);vx={name:combine(b,**feature_cfg) for name,b in cases.items()}
        classifiers={};regressors={}
        for C in cfg['C_grid']:
            clf=fit_classifier(x,ay,C,cfg['class_weight'],weights)
            classifiers[C]=(clf,{k:clf.predict_proba(v) for k,v in vx.items()})
            print(f'validation search av={av_weight} C={C} converged in {clf.n_iter_.tolist()}',flush=True)
        for alpha in cfg['alpha_grid']:
            reg=fit_regressor(x,ar,alpha,weights)
            regressors[alpha]=(reg,{k:np.clip(reg.predict(v),-3,3) for k,v in vx.items()})
        for C,alpha in itertools.product(cfg['C_grid'],cfg['alpha_grid']):
            scores=[measure(vy,vr,(classifiers[C][1][name],regressors[alpha][1][name])) for name in cases]
            value=scalar(scores)
            row={'av_weight':av_weight,'C':C,'alpha':alpha,'score':value,'cases':dict(zip(cases,scores))};tuning.append(row)
            if best is None or value<best['score']:
                best=copy.deepcopy(row);best['feature_config']=feature_cfg
    js(out/'validation_search.json',tuning)
    models={};seed_scores=[]
    for seed in cfg['train_seeds']:
        if seed!=first_seed:
            train_aug,records=augmented(train_samples,seed,cfg['augmentation_views'],cfg['rates'])
            with gzip.open(out/f'training_masks_seed{seed}.jsonl.gz','wt',encoding='utf-8') as f:
                for rec in records:f.write(json.dumps(rec)+'\n')
            blocks=mapper.blocks(train_aug,encoder.encode(train_aug))
        model=DualModel(best['feature_config'],best['C'],best['alpha'],cfg['class_weight']).fit(blocks,ay,ar,weights)
        models.setdefault('proposed',[]).append(model)
        result={k:measure(vy,vr,model.predict(b)) for k,b in cases.items()}
        seed_scores.append({'seed':seed,'scores':result,'score':scalar(list(result.values()))})
        print('proposed seed',seed,'validation',seed_scores[-1]['score'],flush=True)
        if seed==first_seed:
            # Hold chosen hyperparameters constant to isolate each ablation.
            specs={
                'no_augmentation':(clean,y,r,None,dict(best['feature_config'])),
                'no_reliability':(blocks,ay,ar,weights,{**best['feature_config'],'reliability':False}),
                'no_semantics':(blocks,ay,ar,weights,{**best['feature_config'],'semantic':False}),
                'text_only':(blocks,ay,ar,weights,{**best['feature_config'],'modalities':('text',)}),
                'audio_visual_only':(blocks,ay,ar,weights,{**best['feature_config'],'modalities':('audio','vision')})}
            for name,(bb,yy,rr,ww,fc) in specs.items():
                models[name]=[DualModel(fc,best['C'],best['alpha'],cfg['class_weight']).fit(bb,yy,rr,ww)]
                print('ablation fitted',name,flush=True)
    models['proposed_seed17']=[models['proposed'][0]]
    validation_metrics={name:{k:measure(vy,vr,ensemble(group,b)) for k,b in cases.items()} for name,group in models.items()}
    js(out/'validation_selected.json',validation_metrics);js(out/'training_seed_variation.json',seed_scores)
    joblib.dump(models,out/'models.joblib',compress=3)
    constant={'majority_class':int(np.bincount(y,minlength=3).argmax()),'median_intensity':float(np.median(r))}
    frozen={'protocol_sha256':sha(out/'protocol.json'),'feature_map_sha256':sha(out/'feature_map.joblib'),
            'models_sha256':sha(out/'models.joblib'),'chosen':best,'constant_baseline':constant,
            'train_n':len(y),'valid_n':len(vy),'train_ids':list(map(str,ds_train.part['id'])),
            'valid_ids':list(map(str,ds_valid.part['id'])),'model_groups':list(models),
            'selection_seconds':time.time()-started,'test_used_for_selection':False,
            'aggregation':'mean probabilities and clipped intensities over three training augmentation seeds',
            'ablation_policy':'single seed 17, shared chosen hyperparameters; compare to proposed_seed17'}
    js(out/'frozen_selection.json',frozen)
    print('SELECTION FROZEN', {k:best[k] for k in ['C','alpha','av_weight','score']},flush=True)


def evaluate(cfg,out):
    if (out/'evaluation_complete.json').exists():raise ValueError('Independent evaluation already completed; no automatic retuning or re-evaluation')
    frozen=json.loads((out/'frozen_selection.json').read_text(encoding='utf-8'))
    for file,key in [('protocol.json','protocol_sha256'),('feature_map.joblib','feature_map_sha256'),('models.joblib','models_sha256')]:
        if sha(out/file)!=frozen[key]:raise ValueError('Frozen artifact changed: '+file)
    # This function cannot tune or fit a predictor.
    mapper=joblib.load(out/'feature_map.joblib');models=joblib.load(out/'models.joblib')
    encoder=SemanticEncoder(ROOT/cfg['text_model_dir'])
    with (DATA/'附件2-数据集特征文件'/cfg['feature_version']).open('rb') as f:raw=pickle.load(f)
    ds=dataset(raw['test'],'test');del raw
    samples=sample_list(ds);y=ds.part['classification_labels'].astype(int);r=ds.part['regression_labels']
    ids=list(map(str,ds.part['id']))
    if set(ids)&(set(frozen['train_ids'])|set(frozen['valid_ids'])):raise ValueError('Split overlap')
    combinations=[c for n in (1,2,3) for c in itertools.combinations(MODS,n)]
    cases=[('complete',(),0.,'start',0)]
    for position in cfg['positions']:
        seeds=cfg['evaluation_mask_seeds'] if position=='random' else cfg['evaluation_mask_seeds'][:1]
        for seed,combo,rate in itertools.product(seeds,combinations,cfg['rates']):
            cases.append((f'{seed}|{position}|{"+".join(combo)}|{rate}',combo,rate,position,seed))
    metric_rows=[];anchor_predictions={};started=time.time()
    with gzip.open(out/'test_predictions.jsonl.gz','wt',encoding='utf-8') as predlog,gzip.open(out/'test_masks.jsonl.gz','wt',encoding='utf-8') as masklog:
        for case_index,(name,combo,rate,position,seed) in enumerate(cases):
            changed,records=scenario(samples,combo,rate,position,seed,cfg['rates'])
            blocks=mapper.blocks(changed,encoder.encode(changed))
            for rec in records:masklog.write(json.dumps({'scenario':name,**rec})+'\n')
            for model_name,group in models.items():
                p,z=ensemble(group,blocks);s=measure(y,r,(p,z))
                row={'model':model_name,'scenario':name,'modalities':'+'.join(combo) or 'complete','rate':rate,'position':position,'mask_seed':seed,
                     **{k:v for k,v in s.items() if k!='f1_per_class'},**{f'f1_class_{i}':v for i,v in enumerate(s['f1_per_class'])}}
                metric_rows.append(row)
                if name=='complete' or (combo==MODS and rate==.3 and position=='random' and seed==cfg['evaluation_mask_seeds'][0]):
                    anchor_predictions[model_name+'::'+name]={'probabilities':p.tolist(),'intensity':z.tolist()}
                predlog.write(json.dumps({'model':model_name,'scenario':name,'probabilities':p.tolist(),'intensity':z.tolist()})+'\n')
            if case_index%10==0:print(f'Independent test scenario {case_index+1}/{len(cases)} elapsed {time.time()-started:.1f}s',flush=True)
    with (out/'test_metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(metric_rows[0]));w.writeheader();w.writerows(metric_rows)
    baseline=metrics(y,r,np.full(len(y),frozen['constant_baseline']['majority_class']),np.full(len(y),frozen['constant_baseline']['median_intensity']))
    js(out/'test_anchors.json',{'ids':ids,'true_class':y.tolist(),'true_intensity':r.tolist(),'predictions':anchor_predictions,'constant_baseline':baseline})
    special=AlignedDataset.from_attachment3(DATA/'附件3-模态缺失特征样本'/'对齐版本',special_token_ids=(101,102))
    special_samples=sample_list(special);blocks=mapper.blocks(special_samples,encoder.encode(special_samples))
    p,z=ensemble(models['proposed'],blocks)
    rows=[{'sample_id':s['sample_id'],'source_file':s['sample_id'].split('#')[0],
           'polarity':['Negative','Neutral','Positive'][int(p[i].argmax())],'intensity':float(z[i]),
           **{f'prob_{j}':float(p[i,j]) for j in range(3)}} for i,s in enumerate(special_samples)]
    with (out/'附件3_问题2_预测结果.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    # Export every row; header mapping remains documented, not silently guessed.
    js(out/'evaluation_complete.json',{'test_n':len(y),'scenario_count':len(cases),'model_groups':len(models),
        'metric_rows':len(metric_rows),'special_n':len(rows),'seconds':time.time()-started,
        'selection_hash':sha(out/'frozen_selection.json'),'test_policy':'single frozen independent evaluation',
        'output_id_policy':'source filename plus within-file index; no original video ID supplied'})
    print('Independent test and attachment3 prediction complete',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['train','evaluate','all'])
    parser.add_argument('--config',default='configs/formal_protocol.json');parser.add_argument('--output')
    args=parser.parse_args();cfg=json.loads((ROOT/args.config).read_text(encoding='utf-8'))
    if args.output:cfg['output']=args.output
    out=ROOT/cfg['output'];out.mkdir(exist_ok=True)
    with threadpool_limits(limits=4):
        if args.stage in ('train','all'):train(cfg,out)
        if args.stage in ('evaluate','all'):
            saved=json.loads((out/'protocol.json').read_text(encoding='utf-8'));evaluate(saved,out)


if __name__=='__main__':main()
