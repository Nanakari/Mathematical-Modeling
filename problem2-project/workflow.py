"""Config-driven problem 2: train, select, evaluate, analyze, infer, package."""
import argparse
import copy
import csv
import gzip
import hashlib
import importlib.metadata
import json
import platform
import shutil
import time
import zipfile
from pathlib import Path
import numpy as np
import torch
from components import construct,ModularModel,JointLoss
from pipeline import MaskedStandardizer,PreparedDataset
from scenarios import build_library
from torch_training import train_epoch,predict,save_checkpoint
from p2 import metrics

HERE=Path(__file__).resolve().parent


def write_json(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def write_csv(path,rows):
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def adapter_for(cfg):
    spec=copy.deepcopy(cfg['data']);spec['name']=spec.pop('adapter')
    for k in ('feature_path','special_folder'):
        spec[k]=str((HERE/spec[k]).resolve())
    return construct(spec,{})


def score(dataset,predictions):
    expected=list(map(str,dataset.part['id']))
    if expected!=[r['sample_id'] for r in predictions]:raise ValueError('Prediction order mismatch')
    return metrics(dataset.part['classification_labels'],dataset.part['regression_labels'],
                   [r['pred_class'] for r in predictions],[r['pred_intensity'] for r in predictions])


def read_plans(path):
    grouped={}
    with gzip.open(path,'rt',encoding='utf-8') as f:
        for line in f:
            row=json.loads(line)
            grouped.setdefault(row['scenario'],{})[row['sample_id']]=row
    return grouped


def evaluate(model,dataset,norm,plans,batch_size,device):
    prepared=PreparedDataset(dataset,norm,plans=plans)
    # Eager preparation avoids repeating corruption/normalization per batch.
    prepared=[prepared[i] for i in range(len(prepared))]
    predictions=predict(model,prepared,batch_size=batch_size,device=device)
    return score(dataset,predictions),predictions


def load_best(path):
    state=torch.load(path,map_location='cpu',weights_only=True)
    model=ModularModel(state['config']);model.load_state_dict(state['model'])
    return model,state


def run(cfg,out):
    if out.exists() and any(out.iterdir()):
        raise ValueError('Output directory is nonempty; use --output with a new run directory')
    out.mkdir(parents=True,exist_ok=True)
    started=time.time();torch.set_num_threads(cfg['threads'])
    adapter=adapter_for(cfg);train,valid=adapter.load(cfg['seed'])
    majority=int(np.bincount(train.part['classification_labels'].astype(int),minlength=3).argmax())
    median=float(np.median(train.part['regression_labels']))
    reference=metrics(valid.part['classification_labels'],valid.part['regression_labels'],
                      np.full(len(valid),majority),np.full(len(valid),median))
    write_json(out/'config.json',cfg)
    write_json(out/'split_ids.json',{'train':list(map(str,train.part['id'])),'valid':list(map(str,valid.part['id']))})
    norm=construct(cfg['preprocessor'],{'masked_standard':MaskedStandardizer}).fit(train,split='train')
    norm.save(out/'standardizer.json');norm_sha=digest(out/'standardizer.json')
    manifest=build_library(valid,out/'validation_scenarios.jsonl.gz',seeds=cfg['validation']['seeds'],rates=cfg['validation']['rates'])
    plans=read_plans(out/'validation_scenarios.jsonl.gz')
    selector=construct(cfg['selection'],{})
    loss=construct(cfg['loss'],{'joint':JointLoss})
    training_cfg=cfg['training'];device=cfg['device']
    summary=[];all_metrics=[];all_predictions=[]
    for experiment in cfg['experiments']:
        name=experiment['name'];exp_dir=out/name;exp_dir.mkdir()
        torch.manual_seed(cfg['seed'])
        model_cfg=copy.deepcopy(cfg['model']);model_cfg['fusion']={'name':experiment['fusion']}
        model=ModularModel(model_cfg).to(device)
        optimizer=torch.optim.Adam(model.parameters(),lr=training_cfg['learning_rate'])
        if experiment['augment']:
            training=construct({**cfg['augmentation'],'dataset':train,'standardizer':norm,'seed':cfg['seed']},{})
        else:training=PreparedDataset(train,norm)
        history=[];best=float('inf');best_epoch=-1;stale=0
        for epoch in range(training_cfg['epochs']):
            log=train_epoch(model,training,optimizer,epoch=epoch,seed=cfg['seed'],batch_size=training_cfg['batch_size'],device=device,loss_fn=loss)
            selected_scores=[evaluate(model,valid,norm,plans[s],training_cfg['batch_size'],device)[0]
                             for s in cfg['validation']['selection_scenarios']]
            value=selector(selected_scores)
            log.update({'selection_score':value,'selection_metrics':selected_scores})
            history.append(log)
            if value<best-1e-8:
                best,best_epoch,stale=value,epoch,0
                torch.save({'config':model.config,'model':model.state_dict(),'standardizer_sha256':norm_sha,
                            'epoch':epoch,'selection_score':best},exp_dir/'best.pt')
            else:stale+=1
            save_checkpoint(exp_dir/'last.pt',model,optimizer,epoch+1,seed=cfg['seed'],preprocessing_sha256=norm_sha)
            write_json(exp_dir/'history.json',history)
            print(f'{name} epoch={epoch+1} loss={log["loss"]:.4f} selection={value:.4f}',flush=True)
            if stale>=training_cfg['patience']:break
        model,state=load_best(exp_dir/'best.pt')
        for scenario,scenario_plans in plans.items():
            scores,predictions=evaluate(model,valid,norm,scenario_plans,training_cfg['batch_size'],device)
            flat={k:v for k,v in scores.items() if k!='f1_per_class'}
            row={'experiment':name,'scenario':scenario,**flat,**{f'f1_class_{i}':x for i,x in enumerate(scores['f1_per_class'])}}
            all_metrics.append(row)
            for i,p in enumerate(predictions):
                all_predictions.append({'experiment':name,'scenario':scenario,**p,
                    'true_class':int(valid.part['classification_labels'][i]),'true_intensity':float(valid.part['regression_labels'][i])})
        summary.append({'experiment':name,'best_epoch':best_epoch+1,'selection_score':best,
                        'trained_epochs':len(history),'parameters':sum(p.numel() for p in model.parameters())})
        print(f'{name}: evaluated {len(plans)} fixed scenarios',flush=True)
    write_csv(out/'metrics.csv',all_metrics);write_csv(out/'validation_predictions.csv',all_predictions)
    winner=min(summary,key=lambda x:x['selection_score'])['experiment']
    shutil.copyfile(out/winner/'best.pt',out/'selected_model.pt')
    selected,state=load_best(out/'selected_model.pt')
    special=adapter.special()
    special_predictions=predict(selected,PreparedDataset(special,norm),batch_size=training_cfg['batch_size'],device=device)
    exporter=construct(cfg['export'],{})
    count=exporter.export(out/'attachment3_DRAFT_predictions.csv',special_predictions)
    report={'status':'small_scale_complete_workflow_not_final_competition_result','train_n':len(train),'valid_n':len(valid),
            'normalization_fit_n':len(train),'selected_experiment':winner,'experiments':summary,'special_predictions':count,
            'scenario_library':manifest,'seconds':time.time()-started,'standardizer_sha256':norm_sha,
            'constant_reference':{'class_from_train':majority,'intensity_median_from_train':median,'metrics':reference},
            'environment':{'python':platform.python_version(),**{p:importlib.metadata.version(p) for p in ['torch','numpy','scikit-learn','matplotlib']}},
            'selection_note':'Validation used for epoch/model selection and reported diagnostics; these are not independent held-out estimates.',
            'limitations':['random token embedding, no verified pretrained tokenizer','candidate mask semantics',
                           'single training seed','draft export schema and file-based IDs','no official test evaluation']}
    write_json(out/'report.json',report)
    from workflow_report import make_report
    make_report(out,report,all_metrics,all_predictions,valid)
    print(json.dumps({'selected':winner,'train':len(train),'valid':len(valid),'special':count},ensure_ascii=False),flush=True)


def infer(out,output):
    cfg=json.loads((out/'config.json').read_text(encoding='utf-8'))
    torch.set_num_threads(cfg['threads'])
    model,state=load_best(out/'selected_model.pt')
    if digest(out/'standardizer.json')!=state['standardizer_sha256']:raise ValueError('Preprocessor mismatch')
    pre=construct(cfg['preprocessor'],{'masked_standard':MaskedStandardizer})
    norm=type(pre).load(out/'standardizer.json')
    dataset=adapter_for(cfg).special()
    rows=predict(model,PreparedDataset(dataset,norm),batch_size=cfg['training']['batch_size'],device=cfg['device'])
    construct(cfg['export'],{}).export(output,rows)
    print(f'Exported {len(rows)} unlabeled predictions to {output}',flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['run','predict','package'])
    parser.add_argument('--config',default='configs/small.json')
    parser.add_argument('--output')
    parser.add_argument('--prediction-file',default='attachment3_DRAFT_reloaded.csv')
    args=parser.parse_args()
    cfg=json.loads((HERE/args.config).read_text(encoding='utf-8'))
    out=HERE/(args.output or cfg['output'])
    if args.command=='run':run(cfg,out)
    elif args.command=='predict':infer(out,out/args.prediction_file)
    else:
        from workflow_report import package
        package(HERE,out)


if __name__=='__main__':main()
