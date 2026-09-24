"""Controlled two-seed Mini, masked temporal and MAG-style experiments."""
import argparse
import itertools
import json
import pickle
import platform
import time
from pathlib import Path
import joblib
import numpy as np
import torch
from torch.nn import functional as F
from threadpoolctl import threadpool_limits

from accuracy_model import class_weights
from accuracy_run import read, rank, score, seed_all, writecsv
from aligned_dataset import AlignedDataset
from formal_run import ROOT, sha, js, dataset, sample_list, augmented, scenario
from p2 import DATA
from pipeline import MaskedStandardizer
from temporal_model import TemporalDual, tensors, predict
from verify_tokenizer import verify


def encoder_path(cfg, stage):
    return ROOT / cfg['tiny_encoder' if stage == 'tiny_text' else 'mini_encoder']


def model_for(cfg, stage):
    return TemporalDual(encoder_path(cfg, stage), stage, cfg['temporal_width'], cfg['dropout'], cfg['mag_beta'])


def tensor_save(path, value):
    temporary = Path(str(path) + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def verify_pretrained(cfg):
    records = {}
    for key in ['tiny_encoder', 'mini_encoder']:
        folder = ROOT / cfg[key]; provenance = read(folder / 'provenance.json')
        for filename, record in provenance['files'].items():
            if sha(folder / filename) != record['sha256']:
                raise ValueError('Pretrained asset changed: ' + filename)
        records[key] = provenance
    return records


def train_candidate(cfg, out, stage, seed, batch, vb, y, r, vy, vr):
    name = f'{stage}_seed{seed}'
    record_path = out / 'candidates' / (name + '.json')
    checkpoint = 'candidates/' + name + '.pt'
    last_path = out / 'candidates' / (name + '_last.pt')
    if record_path.exists():
        row = read(record_path)
        assert sha(out / checkpoint) == row['checkpoint_sha256']
        print('Resume completed', name, flush=True)
        return row
    seed_all(seed)
    model = model_for(cfg, stage)
    initial_embedding = model.encoder.embeddings.word_embeddings.weight.detach().clone()
    optimizer = torch.optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': cfg['encoder_learning_rate']},
        {'params': [p for n, p in model.named_parameters() if not n.startswith('encoder.')], 'lr': cfg['head_learning_rate']},
    ], weight_decay=cfg['weight_decay'])
    cw = torch.tensor(class_weights(y, cfg['gamma']), dtype=torch.float32)
    ty, tr = torch.tensor(y, dtype=torch.long), torch.tensor(r, dtype=torch.float32)
    history, best, stale, start_epoch, prior_seconds = [], None, 0, 1, 0.
    if last_path.exists():
        saved = torch.load(last_path, weights_only=True, map_location='cpu')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['rng'])
        history, best, stale = saved['history'], saved['best'], saved['stale']
        start_epoch, prior_seconds = saved['next_epoch'], saved['seconds']
        print('Resume epoch', name, start_epoch, flush=True)
    started = time.time()
    if stale < cfg['patience']:
        for epoch in range(start_epoch, cfg['epochs'] + 1):
            rng = np.random.default_rng(seed + epoch)
            chosen = rng.choice(4, len(y), p=[cfg['clean_probability']] + [(1-cfg['clean_probability'])/3]*3)
            order = rng.permutation(len(y)); model.train(); total = 0.
            for start in range(0, len(y), cfg['batch_size']):
                ix = order[start:start+cfg['batch_size']]
                rows = torch.tensor(ix + chosen[ix]*len(y))
                optimizer.zero_grad(set_to_none=True)
                logits, value = model(**{k: v[rows] for k, v in batch.items()})
                loss = (F.cross_entropy(logits, ty[ix], reduction='none')*cw[ty[ix]]).mean()
                loss = loss + cfg['regression_loss_weight']*F.smooth_l1_loss(value, tr[ix])
                if not torch.isfinite(loss): raise ValueError('Non-finite loss')
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step(); total += float(loss.detach())*len(ix)
            p, z = predict(model, vb); metrics = score(vy, vr, p, z)
            history.append({'epoch': epoch, 'train_loss': total/len(y), **metrics})
            if best is None or rank(metrics) > rank(best['metrics']):
                tensor_save(out / checkpoint, model.state_dict())
                best = {'epoch': epoch, 'metrics': metrics}; stale = 0
            else:
                stale += 1
            js(out / 'candidates' / (name + '_history.json'), history)
            elapsed = prior_seconds + time.time() - started
            tensor_save(last_path, {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'rng': torch.get_rng_state(),
                                   'history': history, 'best': best, 'stale': stale, 'next_epoch': epoch+1, 'seconds': elapsed})
            print(f'{name} epoch={epoch} Accuracy={metrics["accuracy"]:.4f} F1={metrics["f1_macro"]:.4f} MAE={metrics["mae"]:.4f} elapsed={elapsed:.0f}s', flush=True)
            if stale >= cfg['patience']: break
    model.load_state_dict(torch.load(out / checkpoint, weights_only=True))
    change = float((model.encoder.embeddings.word_embeddings.weight.detach()-initial_embedding).abs().max())
    assert change > 0
    p, z = predict(model, vb)
    result = {'name': name, 'stage': stage, 'seed': seed, **best, 'checkpoint': checkpoint,
              'checkpoint_sha256': sha(out / checkpoint), 'max_embedding_change': change,
              'epochs_run': len(history), 'seconds': prior_seconds+time.time()-started}
    joblib.dump({'probabilities': p, 'intensity': z}, out / 'candidates' / (name + '_valid.joblib'), compress=3)
    js(record_path, result)
    # Remove only this completed candidate's resumable optimizer snapshot.
    if last_path.exists(): last_path.unlink()
    return result


def load_group(cfg, out, stage, candidates):
    models = []
    for row in candidates:
        if row['stage'] == stage:
            assert sha(out / row['checkpoint']) == row['checkpoint_sha256']
            model = model_for(cfg, stage)
            model.load_state_dict(torch.load(out / row['checkpoint'], weights_only=True, map_location='cpu'))
            model.eval(); models.append(model)
    assert len(models) == len(cfg['seeds'])
    return models


def ensemble(models, batch):
    values = [predict(model, batch) for model in models]
    return np.mean([v[0] for v in values], axis=0), np.mean([v[1] for v in values], axis=0)


def run(cfg, out):
    out.mkdir(parents=True, exist_ok=True); (out / 'candidates').mkdir(exist_ok=True)
    sources = {name: sha(ROOT/name) for name in ['temporal_run.py', 'temporal_model.py', 'accuracy_run.py', 'accuracy_model.py', 'formal_run.py', 'formal_model.py', 'formal_features.py', 'aligned_dataset.py', 'pipeline.py', 'p2.py', 'verify_tokenizer.py']}
    if (out / 'protocol.json').exists():
        if read(out/'protocol.json') != cfg or read(out/'source_hashes.json') != sources:
            raise ValueError('Changed protocol/source requires new output directory')
    else:
        js(out/'protocol.json', cfg); js(out/'source_hashes.json', sources)
    if (out/'complete.json').exists(): raise ValueError('Already completed; use report/predict')
    js(out/'pretrained_provenance.json', verify_pretrained(cfg))
    js(out/'environment.json', {'python': platform.python_version(), 'torch': torch.__version__, 'device': 'cpu', 'threads': cfg['threads']})
    data_path = DATA/'附件2-数据集特征文件/aligned_50.pkl'
    data_hash = sha(data_path)
    with data_path.open('rb') as f: raw = pickle.load(f)
    train_ds, valid_ds = dataset(raw['train'], 'train'), dataset(raw['valid'], 'valid')
    js(out/'tokenizer_verification.json', verify(raw, ROOT/cfg['mini_encoder']))
    del raw
    train, valid = sample_list(train_ds), sample_list(valid_ds)
    y, r = train_ds.part['classification_labels'].astype(int), train_ds.part['regression_labels']
    vy, vr = valid_ds.part['classification_labels'].astype(int), valid_ds.part['regression_labels']
    train_ids, valid_ids = [s['sample_id'] for s in train], [s['sample_id'] for s in valid]
    assert not ({s.split('$_$')[0] for s in train_ids} & {s.split('$_$')[0] for s in valid_ids})
    js(out/'split_manifest.json', {'train_ids': train_ids, 'valid_ids': valid_ids, 'data_sha256': data_hash, 'official_test_evaluated': False})
    previous = joblib.load(ROOT/'outputs/accuracy_v3/validation_predictions.joblib')
    assert previous['ids'] == valid_ids
    np.testing.assert_array_equal(previous['y'], vy); np.testing.assert_array_equal(previous['r'], vr)
    js(out/'previous_artifacts.json', {name: sha(ROOT/'outputs/accuracy_v3'/name) for name in ['validation_predictions.joblib', 'frozen_selection.json', 'model_manifest.json']})
    cache = out/'prepared.pt'
    if cache.exists():
        manifest = read(out/'prepared_manifest.json')
        assert manifest['data_sha256'] == data_hash and manifest['sha256'] == sha(cache)
        prepared = torch.load(cache, weights_only=True); batch, vb = prepared['train'], prepared['valid']
        norm = MaskedStandardizer.load(out/'standardizer.json')
        assert sha(out/'standardizer.json') == manifest['standardizer_sha256']
    else:
        norm = MaskedStandardizer().fit(train_ds, split='train'); norm.save(out/'standardizer.json')
        train_aug, _ = augmented(train, cfg['augmentation_seed'], 3, cfg['augmentation_rates'])
        batch, vb = tensors(train_aug, norm), tensors(valid, norm)
        tensor_save(cache, {'train': batch, 'valid': vb})
        js(out/'prepared_manifest.json', {'data_sha256': data_hash, 'sha256': sha(cache), 'standardizer_sha256': sha(out/'standardizer.json')})
        del train_aug
    del train_ds, train
    print('Prepared train=',len(y),'valid=',len(vy),'stages=',len(cfg['stages']),'seeds=',cfg['seeds'],flush=True)
    candidates = []
    for stage, seed in itertools.product(cfg['stages'], cfg['seeds']):
        candidates.append(train_candidate(cfg,out,stage,seed,batch,vb,y,r,vy,vr))
        js(out/'candidate_summary.json',candidates)
    del batch
    comparison, complete_predictions = [], {}
    for stage in cfg['stages']:
        rows = [row for row in candidates if row['stage']==stage]
        values = [joblib.load(out/'candidates'/(row['name']+'_valid.joblib')) for row in rows]
        p = np.mean([v['probabilities'] for v in values],axis=0); z = np.mean([v['intensity'] for v in values],axis=0)
        m = score(vy,vr,p,z); complete_predictions[stage]=(p,z)
        comparison.append({'stage':stage,'seed_accuracy_mean':float(np.mean([row['metrics']['accuracy'] for row in rows])),
                           'seed_accuracy_std':float(np.std([row['metrics']['accuracy'] for row in rows],ddof=1)), 'metrics':m})
    prior_p,prior_z = previous['predictions']['complete']['neural']
    prior_metrics = score(vy,vr,prior_p,prior_z)
    best = max(comparison,key=lambda row:rank(row['metrics']))
    selected = best['stage'] if rank(best['metrics']) > rank(prior_metrics) else 'accuracy_v3'
    js(out/'architecture_comparison.json',comparison)
    selected_rows = [row for row in candidates if row['stage']==selected]
    js(out/'frozen_selection.json',{'selected':selected,'best_new':best,'previous_metrics':prior_metrics,'selected_candidates':selected_rows,
                                  'protocol_sha256':sha(out/'protocol.json'),'official_test_evaluated':False})
    js(out/'model_manifest.json',{name:sha(out/name) for name in ['protocol.json','standardizer.json','frozen_selection.json']+[row['checkpoint'] for row in selected_rows]})
    print('Selection frozen:',selected, 'best new',best['metrics']['accuracy'], flush=True)
    evaluate(cfg,out,norm,valid,vy,vr,candidates,previous)
    export_special(cfg,out)
    js(out/'complete.json',{'status':'completed','train_n':len(y),'valid_n':len(vy),'candidates':len(candidates),
                          'selection_sha256':sha(out/'frozen_selection.json'),'official_test_evaluated':False})
    report(cfg,out)


def case_definitions():
    mods=('text','audio','vision')
    combos=[c for n in [1,2,3] for c in itertools.combinations(mods,n)]
    cases=[('complete',(),0.,'start')]
    cases += [(f'{"+".join(c)}|{pos}|0.3',c,.3,pos) for c,pos in itertools.product(combos,['start','middle','end','random'])]
    cases += [(f'all|random|{r}',mods,r,'random') for r in [.1,.5,.7]]
    return cases


def evaluate(cfg,out,norm,valid,y,r,candidates,previous):
    rows=[]; predictions={name:{'accuracy_v3':previous['predictions'][name]['neural']} for name,_,_,_ in case_definitions()}
    for stage in cfg['stages']:
        models=load_group(cfg,out,stage,candidates)
        for name,combo,rate,pos in case_definitions():
            samples=valid if not combo else scenario(valid,combo,rate,pos,cfg['validation_mask_seed'],[.1,.3,.5,.7])[0]
            p,z=ensemble(models,tensors(samples,norm))
            predictions[name][stage]=(p,z)
            rows.append({'model':stage,'scenario':name,**score(y,r,p,z)})
            print('Validation',stage,name,flush=True)
        del models
    for name,values in predictions.items():
        p,z=values['accuracy_v3']; rows.append({'model':'accuracy_v3','scenario':name,**score(y,r,p,z)})
    js(out/'validation_metrics.json',rows)
    joblib.dump({'ids':[s['sample_id'] for s in valid],'y':y,'r':r,'predictions':predictions},out/'validation_predictions.joblib',compress=3)


def export_special(cfg,out,filename='附件3_时序融合候选预测.csv'):
    for filename_hash,digest in read(out/'model_manifest.json').items():
        assert sha(out/filename_hash)==digest,filename_hash
    verify_pretrained(cfg)
    samples=sample_list(AlignedDataset.from_attachment3(DATA/'附件3-模态缺失特征样本/对齐版本',special_token_ids=(101,102)))
    selection=read(out/'frozen_selection.json'); selected=selection['selected']
    if selected=='accuracy_v3':
        from accuracy_run import predictors,base_artifacts
        old_cfg=read(ROOT/'outputs/accuracy_v3/protocol.json'); base,_=base_artifacts(old_cfg)
        for name,digest in read(out/'previous_artifacts.json').items(): assert sha(ROOT/'outputs/accuracy_v3'/name)==digest
        for name,digest in read(ROOT/'outputs/accuracy_v3/model_manifest.json').items(): assert sha(ROOT/'outputs/accuracy_v3'/name)==digest
        infer=predictors(old_cfg,ROOT/'outputs/accuracy_v3',joblib.load(ROOT/'outputs/accuracy_v3/feature_map.joblib'),joblib.load(base/'models.joblib')['proposed'])
        p,z=infer(samples)['neural']
    else:
        models=load_group(cfg,out,selected,selection['selected_candidates'])
        p,z=ensemble(models,tensors(samples,MaskedStandardizer.load(out/'standardizer.json')))
    np.testing.assert_allclose(p.sum(1),1,atol=1e-6)
    assert len(samples)==30 and np.isfinite(p).all() and np.isfinite(z).all() and (abs(z)<=3).all()
    rows=[{'sample_id':s['sample_id'],'source_file':s['sample_id'].split('#')[0],
           'polarity':['Negative','Neutral','Positive'][int(p[i].argmax())],'intensity':float(z[i]),
           **{f'prob_{j}':float(p[i,j]) for j in range(3)}} for i,s in enumerate(samples)]
    writecsv(out/filename,rows); print('Attachment3:',selected,len(rows),flush=True)


def report(cfg,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    rows=read(out/'validation_metrics.json'); selection=read(out/'frozen_selection.json')
    candidates=read(out/'candidate_summary.json'); comparison=read(out/'architecture_comparison.json')
    data=joblib.load(out/'validation_predictions.joblib'); chosen=selection['selected']
    y,r=data['y'],data['r']; p,z=data['predictions']['complete'][chosen]
    fig,axes=plt.subplots(1,2,figsize=(10,4)); cm=confusion_matrix(y,p.argmax(1),labels=[0,1,2])
    axes[0].imshow(cm,cmap='Blues')
    for i,j in itertools.product(range(3),repeat=2): axes[0].text(j,i,str(cm[i,j]),ha='center',va='center')
    axes[0].set(xlabel='Predicted',ylabel='True',xticks=[0,1,2],yticks=[0,1,2],title='Negative / Neutral / Positive')
    axes[1].scatter(r,z,s=8,alpha=.4); axes[1].plot([-3,3],[-3,3],'k--'); axes[1].set(xlabel='True intensity',ylabel='Predicted intensity',title='Validation regression')
    fig.tight_layout(); fig.savefig(out/'validation_diagnostics.png',dpi=170); plt.close(fig)
    writecsv(out/'validation_errors.csv',[{'sample_id':sid,'true_class':int(y[i]),'pred_class':int(p[i].argmax()),'true_intensity':float(r[i]),'pred_intensity':float(z[i]),'absolute_error':float(abs(r[i]-z[i]))} for i,sid in enumerate(data['ids'])])
    groups=np.array([sid.split('$_$')[0] for sid in data['ids']]); indices=[np.flatnonzero(groups==g) for g in np.unique(groups)]
    rng=np.random.default_rng(7129); selected_correct=p.argmax(1)==y
    previous_correct=data['predictions']['complete']['accuracy_v3'][0].argmax(1)==y
    deltas=[]
    for _ in range(500):
        ix=np.concatenate([indices[i] for i in rng.integers(len(indices),size=len(indices))])
        deltas.append(float(selected_correct[ix].mean()-previous_correct[ix].mean()))
    ci=np.quantile(deltas,[.025,.975]).tolist()
    js(out/'accuracy_bootstrap.json',{'selected_minus_v3':float(selected_correct.mean()-previous_correct.mean()),'95_percentile_interval':ci,'repeats':500,'accounts_for_selection':False})
    lines=['# BERT-Mini 时序编码与门控融合实验','',
           f'最终候选：{chosen}。完整输入验证Accuracy={selected_correct.mean():.4f}；'+('达到0.70验证目标。' if selected_correct.mean()>=cfg['target'] else '未达到0.70验证目标。'),'',
           '所有模型仅使用官方train的3395条样本训练，728条valid选择轮次和结构；官方test未评价。验证集此前已反复使用，本轮分数不是新的独立泛化证据。每种新结构固定两个种子，均纳入概率与强度平均，不按种子表现挑选成员。','',
           '## 同条件验证结果','','| 模型 | 完整Accuracy | Macro-F1 | MAE | Pearson | 中性召回率 | 31个缺失场景平均Accuracy |','|---|---:|---:|---:|---:|---:|---:|']
    for stage in ['accuracy_v3']+cfg['stages']:
        m=next(s for s in rows if s['model']==stage and s['scenario']=='complete')
        missing=np.mean([s['accuracy'] for s in rows if s['model']==stage and s['scenario']!='complete'])
        lines.append(f'| {stage} | {m["accuracy"]:.4f} | {m["f1_macro"]:.4f} | {m["mae"]:.4f} | {m["pearson"]:.4f} | {m["neutral_recall"]:.4f} | {missing:.4f} |')
    lines += ['','## 结构和公平对照','','tiny_text与mini_text比较两层128维Tiny和四层256维Mini，均只使用文本。mini_temporal在Mini上加入音视频各两层掩码时序卷积，隐维32、卷积核3；mini_mag在相同结构上加入MAG启发的逐位置门控残差，残差范数上限为文本表示范数的0.1倍。门控在BERT输出后作用，不是原论文在BERT内部插入模块的原样复现。','',
              '时序编码保留原50个位置，通过独立观测掩码逐层清零无效位置；不压缩缺失间隙。音视频池化不依赖文本是否可用，因此文本局部缺失不会一起删除其他模态证据。所有原始模态标准化仅在train拟合，缺失检测先于标准化。','',
              '同一Mini架构的公共模块使用相同初始化顺序。所有新模型使用相同训练配置、种子和视图抽样：完整输入概率0.6，三份连续缺失增强共享0.4。等类别权重，交叉熵+0.2×SmoothL1，编码器学习率5e-5，头部5e-4，AdamW，最多10轮，连续3轮未改善停止。accuracy_v3包含统计音视频及另一套结构，仅作为前轮参照，不将全部差异归因于编码器。','',
              '## 随机种子结果','','| 结构 | 种子 | 最佳轮次 | Accuracy | Macro-F1 | MAE |','|---|---:|---:|---:|---:|---:|']
    for row in candidates:
        m=row['metrics']; lines.append(f'| {row["stage"]} | {row["seed"]} | {row["epoch"]} | {m["accuracy"]:.4f} | {m["f1_macro"]:.4f} | {m["mae"]:.4f} |')
    lines += ['','| 结构 | 单种子Accuracy均值 | 两种子样本标准差 | 双种子集成Accuracy |','|---|---:|---:|---:|']
    for row in comparison: lines.append(f'| {row["stage"]} | {row["seed_accuracy_mean"]:.4f} | {row["seed_accuracy_std"]:.4f} | {row["metrics"]["accuracy"]:.4f} |')
    lines += ['','只有两个种子，标准差仅作描述，不能据此证明普遍稳定。结构选择使用双种子集成的完整验证Accuracy，同分依次比较Macro-F1、MAE。缺失评价在冻结后执行，不据其结果自动更换胜出模型。','',
              '## 错误分析和不确定性','',f'相对v3的Accuracy差值为{selected_correct.mean()-previous_correct.mean():+.4f}，按video_id分组500次成对bootstrap的95%描述性区间为[{ci[0]:+.4f}, {ci[1]:+.4f}]。未计入选模偏差和全部训练随机性。','',
              '![验证诊断](validation_diagnostics.png)','','| 类别 | 样本数 | 召回率 | 强度MAE |','|---|---:|---:|---:|']
    for c in range(3):
        ix=y==c; lines.append(f'| {c} | {ix.sum()} | {np.mean(p.argmax(1)[ix]==c):.4f} | {np.mean(abs(r[ix]-z[ix])):.4f} |')
    lines += ['','逐样本结果见validation_errors.csv；注意中性识别与整体Accuracy的取舍。分类与回归仍独立输出，没有用预测强度强行修正极性。','',
              '## 交付与可复现性','','附件3_时序融合候选预测.csv为30条无标签专项预测，不能计算专项Accuracy。已有formal、refinement_v2、accuracy_v3结果保持不变。核心实现为temporal_model.py与temporal_run.py，运行说明见TEMPORAL_README.md。','',
              '本轮未做缺失蒸馏，也未使用额外情感数据集或公开MOSEI微调权重。Mini是通用预训练模型，修订号、下载来源与哈希保存在pretrained_provenance.json。候选及缓存较大，不能将整个实验目录直接作为≤50MB竞赛附件；最终需仅打包选定模型并另行核验压缩或蒸馏后的推理。','',
              '方法参考：[MAG-BERT作者代码](https://github.com/WasifurRahman/BERT_multimodal_transformer)、[Google BERT-Mini](https://huggingface.co/google/bert_uncased_L-4_H-256_A-4)。代码为针对本题独立实现，未复制外部训练或评测流程。']
    (out/'时序融合优化报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('stage',choices=['run','predict','report']); ap.add_argument('--config',default='configs/temporal_protocol.json')
    args=ap.parse_args(); cfg=read(ROOT/args.config); out=ROOT/cfg['output']
    if args.stage!='run': cfg=read(out/'protocol.json')
    torch.set_num_threads(cfg['threads'])
    with threadpool_limits(limits=cfg['threads']):
        if args.stage=='run': run(cfg,out)
        elif args.stage=='predict': export_special(cfg,out,'附件3_时序融合重载预测.csv')
        else: report(cfg,out)


if __name__=='__main__': main()
