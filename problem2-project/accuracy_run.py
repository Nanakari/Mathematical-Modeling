"""Accuracy-oriented class-weight controls and real BERT-Tiny fine-tuning.

Run: python accuracy_run.py run | predict | report
Candidate/epoch selection uses complete validation only, never official test.
"""
import argparse
import csv
import itertools
import json
import pickle
import random
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from torch.nn import functional as F
from threadpoolctl import threadpool_limits

from accuracy_model import TinyDual, class_weights, tensors, predict
from aligned_dataset import AlignedDataset
from formal_features import SemanticEncoder, combine
from formal_model import fit_classifier, ensemble
from formal_run import ROOT, sha, js, dataset, sample_list, augmented, scenario, measure
from p2 import DATA


def writecsv(path, rows):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def score(y, r, p, z):
    result = measure(y, r, (p, z))
    result['neutral_recall'] = float(np.mean(p.argmax(1)[y == 1] == 1))
    result['opposite_sign_count'] = int(np.sum(((p.argmax(1) == 0) & (z > 0)) | ((p.argmax(1) == 2) & (z < 0))))
    return result


def rank(result):
    return result['accuracy'], result['f1_macro'], -result['mae']


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def base_artifacts(cfg):
    base = ROOT / cfg['formal_dir']
    frozen = read(base / 'frozen_selection.json')
    for filename, key in [('feature_map.joblib', 'feature_map_sha256'), ('models.joblib', 'models_sha256'), ('protocol.json', 'protocol_sha256')]:
        if sha(base / filename) != frozen[key]:
            raise ValueError('Original frozen artifact mismatch: ' + filename)
    return base, frozen


def load_model(out, row, cfg):
    model = TinyDual(ROOT / cfg['encoder'], frozen=row['frozen'], dropout=cfg['dropout'])
    model.load_state_dict(torch.load(out / row['checkpoint'], map_location='cpu', weights_only=True))
    return model


def train_candidate(out, cfg, spec, train_batch, valid_batch, y, r, vy, vr):
    destination = out / 'candidates' / (spec['name'] + '.json')
    if destination.exists():
        saved = read(destination)
        if sha(out / saved['checkpoint']) != saved['checkpoint_sha256']:
            raise ValueError('Candidate checkpoint changed')
        print('Resume completed', spec['name'], flush=True)
        return saved
    seed_all(cfg['seed'])
    model = TinyDual(ROOT / cfg['encoder'], frozen=spec['frozen'], dropout=cfg['dropout'])
    initial_encoder_hash = sha(ROOT / cfg['encoder'] / 'model.safetensors')
    initial_embedding = model.encoder.embeddings.word_embeddings.weight.detach().clone()
    head = [p for n, p in model.named_parameters() if not n.startswith('encoder.')]
    groups = [{'params': head, 'lr': cfg['head_learning_rate']}]
    if not spec['frozen']:
        groups.append({'params': model.encoder.parameters(), 'lr': spec['lr']})
    optimizer = torch.optim.AdamW(groups, weight_decay=cfg['weight_decay'])
    cw = torch.tensor(class_weights(y, spec['gamma']), dtype=torch.float32)
    ty, tr = torch.tensor(y, dtype=torch.long), torch.tensor(r, dtype=torch.float32)
    best, history, stale = None, [], 0
    checkpoint = 'candidates/' + spec['name'] + '.pt'
    started = time.time()
    for epoch in range(1, cfg['epochs'] + 1):
        # Same view choices and ordering for all candidates at each epoch.
        rng = np.random.default_rng(cfg['seed'] + epoch)
        chosen = rng.choice(4, len(y), p=[cfg['clean_probability']] + [(1 - cfg['clean_probability']) / 3] * 3)
        order = rng.permutation(len(y))
        model.train(); loss_sum = 0.
        for start in range(0, len(y), cfg['batch_size']):
            ix = order[start:start + cfg['batch_size']]
            rows = torch.tensor(ix + chosen[ix] * len(y))
            batch = {k: v[rows] for k, v in train_batch.items()}
            optimizer.zero_grad(set_to_none=True)
            logits, values = model(**batch)
            # Fixed population-normalized weights, avoiding per-batch class-ratio normalization.
            ce = (F.cross_entropy(logits, ty[ix], reduction='none') * cw[ty[ix]]).mean()
            loss = ce + cfg['regression_loss_weight'] * F.smooth_l1_loss(values, tr[ix])
            if not torch.isfinite(loss):
                raise ValueError('Non-finite loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step(); loss_sum += float(loss.detach()) * len(ix)
        p, z = predict(model, valid_batch)
        result = score(vy, vr, p, z)
        history.append({'epoch': epoch, 'train_loss': loss_sum / len(y), **result})
        improved = best is None or rank(result) > rank(best['metrics'])
        if improved:
            torch.save(model.state_dict(), out / checkpoint)
            best = {'epoch': epoch, 'metrics': result}; stale = 0
        else:
            stale += 1
        js(out / 'candidates' / (spec['name'] + '_history.json'), history)
        print(f"{spec['name']} epoch={epoch} acc={result['accuracy']:.4f} F1={result['f1_macro']:.4f} MAE={result['mae']:.4f} elapsed={time.time()-started:.0f}s", flush=True)
        if stale >= cfg['patience']:
            break
    model.load_state_dict(torch.load(out / checkpoint, weights_only=True))
    embedding_delta = float((model.encoder.embeddings.word_embeddings.weight.detach() - initial_embedding).abs().max())
    if spec['frozen'] and embedding_delta != 0:
        raise ValueError('Frozen encoder unexpectedly updated')
    if not spec['frozen'] and embedding_delta == 0:
        raise ValueError('Fine-tuning did not update encoder')
    result = {**spec, **best, 'checkpoint': checkpoint, 'checkpoint_sha256': sha(out / checkpoint),
              'epochs_run': len(history), 'seconds': time.time() - started,
              'initial_encoder_sha256': initial_encoder_hash, 'max_embedding_change': embedding_delta,
              'class_weights': cw.tolist()}
    js(destination, result)
    return result


def run(cfg, out):
    out.mkdir(parents=True, exist_ok=True); (out / 'candidates').mkdir(exist_ok=True)
    base, frozen = base_artifacts(cfg)
    sources = {name: sha(ROOT / name) for name in ['accuracy_run.py', 'accuracy_model.py', 'aligned_dataset.py', 'formal_features.py', 'formal_model.py', 'formal_run.py', 'pipeline.py', 'p2.py']}
    if (out / 'protocol.json').exists():
        if read(out / 'protocol.json') != cfg or read(out / 'source_hashes.json') != sources:
            raise ValueError('Protocol/source changed: use a fresh output directory')
    else:
        js(out / 'protocol.json', cfg); js(out / 'source_hashes.json', sources)
    if (out / 'complete.json').exists():
        raise ValueError('Run completed; use report or predict, not retraining')
    with (DATA / '附件2-数据集特征文件' / 'aligned_50.pkl').open('rb') as f:
        raw = pickle.load(f)
    train_ds, valid_ds = dataset(raw['train'], 'train'), dataset(raw['valid'], 'valid')
    del raw
    train, valid = sample_list(train_ds), sample_list(valid_ds)
    train_ids, valid_ids = [s['sample_id'] for s in train], [s['sample_id'] for s in valid]
    if train_ids != frozen['train_ids'] or valid_ids != frozen['valid_ids']:
        raise ValueError('Dataset differs from original train-fitted mapper')
    if {s.split('$_$')[0] for s in train_ids} & {s.split('$_$')[0] for s in valid_ids}:
        raise ValueError('Train/validation video overlap')
    js(out / 'split_manifest.json', {'train_ids': train_ids, 'valid_ids': valid_ids, 'official_test_evaluated': False})
    y, r = train_ds.part['classification_labels'].astype(int), train_ds.part['regression_labels']
    vy, vr = valid_ds.part['classification_labels'].astype(int), valid_ds.part['regression_labels']
    mapper = joblib.load(base / 'feature_map.joblib')
    shutil.copyfile(base / 'feature_map.joblib', out / 'feature_map.joblib')
    old_models = joblib.load(base / 'models.joblib')['proposed']
    train_aug, _ = augmented(train, cfg['augmentation_seed'], 3, cfg['augmentation_rates'])
    encoder = SemanticEncoder(ROOT / cfg['encoder'], threads=cfg['threads'])
    print('Prepared train', len(train), 'valid', len(valid), 'training views', len(train_aug), flush=True)
    feature_cfg = frozen['chosen']['feature_config']
    vb = mapper.blocks(valid, encoder.encode(valid))
    x = combine(mapper.blocks(train_aug, encoder.encode(train_aug)), **feature_cfg)
    vx = combine(vb, **feature_cfg)
    baseline_rows, baseline_models = [], {}
    for gamma, C in itertools.product(cfg['gamma'], cfg['C']):
        cw = dict(enumerate(class_weights(y, gamma)))
        classifier = fit_classifier(x, np.tile(y, 4), C, cw, np.full(len(train_aug), .25))
        # Preserve the original seed17 regression, isolating classifier changes.
        z = old_models[0].predict(vb)[1]
        p = classifier.predict_proba(vx)
        name = f'linear_g{gamma}_C{C}'
        result = score(vy, vr, p, z)
        baseline_rows.append({'name': name, 'gamma': gamma, 'C': C, 'metrics': result})
        baseline_models[name] = classifier
        print(name, 'accuracy', round(result['accuracy'], 4), flush=True)
    js(out / 'linear_search.json', baseline_rows)
    best_linear = max(baseline_rows, key=lambda row: rank(row['metrics']))
    joblib.dump({'classifier': baseline_models[best_linear['name']], 'regression': old_models[0], 'feature_config': feature_cfg}, out / 'linear_selected.joblib', compress=3)
    old_p, old_z = ensemble(old_models, vb)
    old_result = score(vy, vr, old_p, old_z)
    del x, vx, baseline_models, encoder
    train_batch, valid_batch = tensors(train_aug, mapper), tensors(valid, mapper)
    specs = [{'name': f'frozen_g{g}', 'gamma': g, 'frozen': True, 'lr': 0.} for g in cfg['gamma']]
    specs += [{'name': f'finetuned_g{g}_lr{lr}', 'gamma': g, 'frozen': False, 'lr': lr} for g, lr in itertools.product(cfg['gamma'], cfg['encoder_learning_rates'])]
    candidates = []
    for spec in specs:
        candidates.append(train_candidate(out, cfg, spec, train_batch, valid_batch, y, r, vy, vr))
        js(out / 'neural_search.json', candidates)
    best_neural = max(candidates, key=lambda row: rank(row['metrics']))
    chosen_kind = 'neural' if rank(best_neural['metrics']) > rank(best_linear['metrics']) else 'linear'
    if rank(old_result) >= max(rank(best_neural['metrics']), rank(best_linear['metrics'])):
        chosen_kind = 'original'
    selection = {'kind': chosen_kind, 'best_linear': best_linear, 'best_neural': best_neural,
                 'original_metrics': old_result, 'protocol_sha256': sha(out / 'protocol.json'),
                 'feature_map_sha256': sha(out / 'feature_map.joblib'), 'linear_sha256': sha(out / 'linear_selected.joblib'),
                 'official_test_evaluated': False, 'selection_split': 'previously used official validation'}
    # Freeze before missing-condition evaluation and special inference.
    js(out / 'frozen_selection.json', selection)
    shutil.copyfile(out / best_neural['checkpoint'], out / 'neural_selected.pt')
    js(out / 'model_manifest.json', {name: sha(out / name) for name in ['frozen_selection.json', 'protocol.json', 'feature_map.joblib', 'linear_selected.joblib', 'neural_selected.pt']})
    evaluate(cfg, out, mapper, valid, vy, vr, old_models)
    export_special(cfg, out)
    js(out / 'complete.json', {'status': 'completed', 'train_n': len(y), 'valid_n': len(vy), 'official_test_evaluated': False, 'selection_sha256': sha(out / 'frozen_selection.json')})
    report(cfg, out)


def predictors(cfg, out, mapper, old_models):
    selected = read(out / 'frozen_selection.json')
    linear = joblib.load(out / 'linear_selected.joblib')
    neural = load_model(out, selected['best_neural'], cfg)
    encoder = SemanticEncoder(ROOT / cfg['encoder'], threads=cfg['threads'])
    def infer(samples):
        b = mapper.blocks(samples, encoder.encode(samples))
        return {'original': ensemble(old_models, b),
                'linear': (linear['classifier'].predict_proba(combine(b, **linear['feature_config'])), linear['regression'].predict(b)[1]),
                'neural': predict(neural, tensors(samples, mapper))}
    return infer


def evaluate(cfg, out, mapper, valid, y, r, old_models):
    infer = predictors(cfg, out, mapper, old_models)
    cases = [('complete', (), 0., 'start')]
    mods = ('text', 'audio', 'vision')
    combos = [c for n in (1, 2, 3) for c in itertools.combinations(mods, n)]
    cases += [(f'{"+".join(combo)}|{pos}|0.3', combo, .3, pos) for combo, pos in itertools.product(combos, ['start', 'middle', 'end', 'random'])]
    cases += [(f'all|random|{rate}', mods, rate, 'random') for rate in [.1, .5, .7]]
    rows, predictions = [], {}
    for name, combo, rate, pos in cases:
        ss = valid if not combo else scenario(valid, combo, rate, pos, cfg['validation_mask_seed'], [.1, .3, .5, .7])[0]
        values = infer(ss)
        predictions[name] = values
        for model, (p, z) in values.items():
            rows.append({'model': model, 'scenario': name, **score(y, r, p, z)})
        print('Validation diagnostic', name, flush=True)
    js(out / 'validation_metrics.json', rows)
    joblib.dump({'ids': [s['sample_id'] for s in valid], 'y': y, 'r': r, 'predictions': predictions}, out / 'validation_predictions.joblib', compress=3)


def export_special(cfg, out, filename='附件3_Accuracy候选预测.csv'):
    for filename_hash, digest in read(out / 'model_manifest.json').items():
        if sha(out / filename_hash) != digest:
            raise ValueError('Selected artifact changed: ' + filename_hash)
    base, _ = base_artifacts(cfg)
    mapper = joblib.load(out / 'feature_map.joblib')
    old_models = joblib.load(base / 'models.joblib')['proposed']
    samples = sample_list(AlignedDataset.from_attachment3(DATA / '附件3-模态缺失特征样本' / '对齐版本', special_token_ids=(101, 102)))
    kind = read(out / 'frozen_selection.json')['kind']
    p, z = predictors(cfg, out, mapper, old_models)(samples)[kind]
    rows = [{'sample_id': s['sample_id'], 'source_file': s['sample_id'].split('#')[0], 'polarity': ['Negative', 'Neutral', 'Positive'][int(p[i].argmax())], 'intensity': float(z[i]), **{f'prob_{j}': float(p[i, j]) for j in range(3)}} for i, s in enumerate(samples)]
    assert len(rows) == len({row['sample_id'] for row in rows}) == 30
    assert np.isfinite(p).all() and np.isfinite(z).all() and (np.abs(z) <= 3).all()
    np.testing.assert_allclose(p.sum(1), 1, atol=1e-6)
    writecsv(out / filename, rows)
    print('Attachment3:', kind, len(rows), flush=True)


def report(cfg, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    selection = read(out / 'frozen_selection.json')
    rows = read(out / 'validation_metrics.json')
    data = joblib.load(out / 'validation_predictions.joblib')
    kind = selection['kind']; y, r = data['y'], data['r']
    p, z = data['predictions']['complete'][kind]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    cm = confusion_matrix(y, p.argmax(1), labels=[0, 1, 2])
    axes[0].imshow(cm, cmap='Blues')
    for i, j in itertools.product(range(3), repeat=2): axes[0].text(j, i, str(cm[i,j]), ha='center', va='center')
    axes[0].set(xlabel='Predicted', ylabel='True', xticks=[0,1,2], yticks=[0,1,2], title='Validation: Negative / Neutral / Positive')
    axes[1].scatter(r, z, s=8, alpha=.4); axes[1].plot([-3,3], [-3,3], 'k--')
    axes[1].set(xlabel='True intensity', ylabel='Predicted intensity', title='Validation regression')
    fig.tight_layout(); fig.savefig(out / 'validation_diagnostics.png', dpi=170); plt.close(fig)
    errors = [{'sample_id': sid, 'true_class': int(y[i]), 'predicted_class': int(p[i].argmax()), 'true_intensity': float(r[i]), 'predicted_intensity': float(z[i]), 'absolute_error': float(abs(r[i]-z[i]))} for i, sid in enumerate(data['ids'])]
    writecsv(out / 'validation_errors.csv', errors)
    # Cluster bootstrap is descriptive after selection, not an unbiased test CI.
    groups = np.array([sid.split('$_$')[0] for sid in data['ids']]); unique = np.unique(groups)
    indices = [np.flatnonzero(groups == g) for g in unique]
    rng = np.random.default_rng(7129); deltas = []
    old_correct = data['predictions']['complete']['original'][0].argmax(1) == y
    new_correct = p.argmax(1) == y
    for _ in range(500):
        ix = np.concatenate([indices[i] for i in rng.integers(len(unique), size=len(unique))])
        deltas.append(float(new_correct[ix].mean() - old_correct[ix].mean()))
    ci = np.quantile(deltas, [.025, .975]).tolist()
    js(out / 'accuracy_bootstrap.json', {'selected_minus_original': float(new_correct.mean()-old_correct.mean()), '95_percentile_interval': ci, 'repeats': 500, 'video_clusters': len(unique), 'accounts_for_model_selection': False})
    lines = ['# Accuracy选模与BERT-Tiny微调实验', '', '本轮只使用官方训练集拟合，在此前已使用的官方验证集选择类别权重、学习率、训练轮次与候选模型。未重新评价官方测试集；验证分数是选模结果，不是新的独立泛化证据。', '',
             f'目标：完整输入三分类Accuracy≥{cfg["target"]:.2f}。选定模型：{kind}；验证Accuracy={new_correct.mean():.4f}；' + ('达到本轮验证目标。' if new_correct.mean() >= cfg['target'] else '未达到本轮验证目标。'), '',
             '## 同条件验证对照', '', '| 模型 | 完整Accuracy | Macro-F1 | MAE | Pearson | 中性召回率 | 31个缺失情景平均Accuracy |', '|---|---:|---:|---:|---:|---:|---:|']
    for model in ['original', 'linear', 'neural']:
        complete = next(row for row in rows if row['model'] == model and row['scenario'] == 'complete')
        missing = [row['accuracy'] for row in rows if row['model'] == model and row['scenario'] != 'complete']
        lines.append(f'| {model} | {complete["accuracy"]:.4f} | {complete["f1_macro"]:.4f} | {complete["mae"]:.4f} | {complete["pearson"]:.4f} | {complete["neutral_recall"]:.4f} | {np.mean(missing):.4f} |')
    lines += ['', '## 类别权重与Accuracy选模', '', '线性对照保持原特征映射、连续缺失增强、回归分支及融合配置，只改变类别权重与分类C。gamma=0为等权，0.5为温和逆频率，1为balanced；训练样本平均权重归一至1。', '', '| gamma | C | Accuracy | Macro-F1 | 中性召回率 |', '|---|---:|---:|---:|---:|']
    for row in read(out / 'linear_search.json'):
        m = row['metrics']; lines.append(f'| {row["gamma"]} | {row["C"]} | {m["accuracy"]:.4f} | {m["f1_macro"]:.4f} | {m["neutral_recall"]:.4f} |')
    lines += ['', '## 同结构冻结与微调对照', '', 'BERT可见位置均值池化，经LayerNorm后与训练集标准化的音视频统计融合，接三分类与强度回归头。损失为类别加权交叉熵+0.2×SmoothL1。每轮每条原始样本取一个版本，完整输入概率0.6，其余在三份固定局部缺失版本中等概率选择。冻结与微调使用相同结构、初始化种子、视图选择和头部学习率；编码器冻结时保持eval状态。与线性模型相比也改变了预测头、训练目标和采样方式，不能把全部差异归因于微调。', '', '| 编码器 | gamma | 编码器学习率 | 最佳轮次 | Accuracy | Macro-F1 | MAE |', '|---|---:|---:|---:|---:|---:|---:|']
    for row in read(out / 'neural_search.json'):
        m = row['metrics']; lines.append(f'| {"冻结" if row["frozen"] else "微调"} | {row["gamma"]} | {row["lr"]} | {row["epoch"]} | {m["accuracy"]:.4f} | {m["f1_macro"]:.4f} | {m["mae"]:.4f} |')
    lines += ['', '## 验证错误分析与不确定性', '', f'选定模型相对旧集成Accuracy差值为{new_correct.mean()-old_correct.mean():+.4f}，按video_id分组500次成对bootstrap的95%描述性区间为[{ci[0]:+.4f}, {ci[1]:+.4f}]。区间未计入候选及轮次选择的不确定性，不应解读为无偏的确认性显著性检验。', '', '![验证混淆矩阵和回归散点](validation_diagnostics.png)', '', '| 类别 | 样本数 | 召回率 | 强度MAE |', '|---|---:|---:|---:|']
    for c in range(3):
        ix = y == c
        lines.append(f'| {c} | {ix.sum()} | {np.mean(p.argmax(1)[ix] == c):.4f} | {np.mean(abs(r[ix]-z[ix])):.4f} |')
    lines += ['', '完整逐样本错误表见validation_errors.csv。分类与回归仍独立输出，不强行将中性强度置零；混淆与残差用于定位问题，不能单独证明其语言或行为原因。缺失评估覆盖30%下7种模态组合×4个位置，外加全模态随机10%、50%、70%，不等于全因子穷举或真实秒数分析。', '', '## 交付与限制', '', '附件3_Accuracy候选预测.csv包含对齐版本30条无标签预测，不据此计算Accuracy。旧outputs/formal和outputs/refinement_v2保持不变。选模以完整Accuracy优先，可能牺牲Macro-F1、MAE或缺失鲁棒性，不能将分类胜出自动视为问题2综合最优。只运行一个训练种子，本轮不提供跨种子稳定性结论。运行说明见ACCURACY_README.md。']
    (out / 'Accuracy优化报告.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['run', 'predict', 'report'])
    parser.add_argument('--config', default='configs/accuracy_protocol.json')
    args = parser.parse_args(); cfg = read(ROOT / args.config); out = ROOT / cfg['output']
    if args.stage != 'run':
        cfg = read(out / 'protocol.json')
    torch.set_num_threads(cfg['threads'])
    with threadpool_limits(limits=cfg['threads']):
        if args.stage == 'run': run(cfg, out)
        elif args.stage == 'predict': export_special(cfg, out, '附件3_Accuracy候选重载预测.csv')
        else: report(cfg, out)


if __name__ == '__main__':
    main()
