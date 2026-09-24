"""One-shot complete-input official test evaluation of the frozen v4 ensemble."""
import json
import pickle
from datetime import datetime, timezone

import joblib
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error
from threadpoolctl import threadpool_limits

from accuracy_run import read, score, writecsv
from formal_run import ROOT, DATA, dataset, sample_list, sha, js
from pipeline import MaskedStandardizer
from temporal_model import tensors
from temporal_run import ensemble, load_group, verify_pretrained


def main():
    run = ROOT / 'outputs/temporal_v4'
    out = run / 'official_test'
    if out.exists():
        raise RuntimeError('Official test output already exists; inspect saved results instead of rerunning.')
    cfg = read(run / 'protocol.json')
    selected = read(run / 'frozen_selection.json')
    assert selected['selected'] == 'mini_mag'
    assert read(run / 'verification.json')['status'] == 'passed'
    assert sha(run / 'frozen_selection.json') == read(run / 'complete.json')['selection_sha256']
    source_hashes = read(run / 'source_hashes.json')
    model_hashes = read(run / 'model_manifest.json')
    for name, digest in source_hashes.items():
        assert sha(ROOT / name) == digest, name
    for name, digest in model_hashes.items():
        assert sha(run / name) == digest, name
    verify_pretrained(cfg)
    splits = read(run / 'split_manifest.json')
    data_path = DATA / '附件2-数据集特征文件/aligned_50.pkl'
    assert sha(data_path) == splits['data_sha256']
    # Freeze this evaluation's scope before loading test examples or labels.
    out.mkdir()
    protocol = {
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'model': selected['selected'], 'members': selected['selected_candidates'],
        'selection_sha256': sha(run / 'frozen_selection.json'),
        'model_manifest_sha256': sha(run / 'model_manifest.json'),
        'data_sha256': splits['data_sha256'], 'evaluator_sha256': sha(ROOT / 'temporal_test.py'),
        'split': 'test', 'scenario': 'complete', 'expected_n': 727,
        'aggregation': 'equal mean of both frozen seeds; intensity clipped per member before averaging',
        'classification': 'argmax of mean probabilities; 0 negative, 1 neutral, 2 positive',
        'refit': False, 'tuning': False, 'reselection': False,
    }
    js(out / 'evaluation_protocol.json', protocol)
    with data_path.open('rb') as f:
        raw = pickle.load(f)
    ds = dataset(raw['test'], 'test')
    del raw
    samples = sample_list(ds)
    ids = [s['sample_id'] for s in samples]
    assert len(ids) == len(set(ids)) == 727
    videos = {sid.split('$_$')[0] for sid in ids}
    for split in ['train_ids', 'valid_ids']:
        assert not videos & {sid.split('$_$')[0] for sid in splits[split]}
    torch.set_num_threads(cfg['threads'])
    with threadpool_limits(limits=cfg['threads']):
        models = load_group(cfg, run, selected['selected'], selected['selected_candidates'])
        norm = MaskedStandardizer.load(run / 'standardizer.json')
        p, z = ensemble(models, tensors(samples, norm))
    # Labels are used only after the single inference pass, for scoring.
    y = ds.part['classification_labels'].astype(int)
    r = ds.part['regression_labels']
    assert p.shape == (727, 3) and z.shape == (727,)
    assert np.isfinite(p).all() and np.isfinite(z).all() and (p >= 0).all() and (abs(z) <= 3).all()
    np.testing.assert_allclose(p.sum(1), 1, atol=1e-6)
    joblib.dump({'ids': ids, 'y': y, 'r': r, 'probabilities': p, 'intensity': z},
                out / 'predictions.joblib', compress=3)
    saved = joblib.load(out / 'predictions.joblib')
    metrics = score(saved['y'], saved['r'], saved['probabilities'], saved['intensity'])
    pred = saved['probabilities'].argmax(1)
    np.testing.assert_allclose(metrics['accuracy'], accuracy_score(y, pred))
    np.testing.assert_allclose(metrics['f1_macro'], f1_score(y, pred, average='macro'))
    np.testing.assert_allclose(metrics['mae'], mean_absolute_error(r, z))
    cm = confusion_matrix(y, pred, labels=[0, 1, 2])
    classes = [{'class': c, 'n': int((y == c).sum()),
                'recall': float((pred[y == c] == c).mean())} for c in range(3)]
    result = {'model': selected['selected'], 'split': 'test', 'scenario': 'complete',
              'n': len(y), 'correct': int((pred == y).sum()), **metrics,
              'confusion_matrix_true_rows': cm.tolist(), 'per_class': classes,
              'target_met': metrics['accuracy'] >= cfg['target']}
    js(out / 'metrics.json', result)
    writecsv(out / 'predictions.csv', [
        {'sample_id': sid, 'true_class': int(y[i]), 'pred_class': int(pred[i]),
         'true_intensity': float(r[i]), 'pred_intensity': float(z[i]),
         **{f'prob_{c}': float(p[i, c]) for c in range(3)}} for i, sid in enumerate(ids)])
    for name, digest in model_hashes.items():
        assert sha(run / name) == digest, name
    for name, digest in source_hashes.items():
        assert sha(ROOT / name) == digest, name
    valid = selected['best_new']['metrics']
    lines = ['# 冻结最优模型：官方测试集一次性评估', '',
             '模型为既有mini_mag双种子等权集成，使用冻结的轮次、标准化和预测规则。',
             '本次仅评估官方test完整输入的727条样本，不新增缺失扰动，不训练、调参或重新选模。', '',
             '| 指标 | 选模验证集 | 官方测试集 |', '|---|---:|---:|']
    for key in ['accuracy', 'f1_macro', 'f1_weighted', 'mae', 'pearson', 'neutral_recall']:
        lines.append(f'| {key} | {valid[key]:.4f} | {metrics[key]:.4f} |')
    lines += ['', f"测试集判对{result['correct']}/727条；" + ('达到' if result['target_met'] else '未达到') + '0.70目标。', '',
              '| 类别 | 样本数 | 召回率 |', '|---|---:|---:|']
    for row in classes:
        lines.append(f"| {['负向', '中性', '正向'][row['class']]} | {row['n']} | {row['recall']:.4f} |")
    lines += ['', '混淆矩阵（行=真实，列=预测；负向/中性/正向）：', '', '```', str(cm), '```', '',
              '源代码、冻结选择、模型参数、标准化及原始数据哈希核验通过；train/valid/test视频ID无交集。',
              '指标从保存的预测重算，并用sklearn独立核对Accuracy、Macro-F1和MAE。',
              '既有验证报告和冻结记录保持原样，其“未评价test”描述的是当时状态；本目录记录后续授权评估。',
              '该测试集在早期项目阶段已评价过其他模型，不能称为整个项目从未接触过的全新盲测。',
              '本次结果不用于再次选择模型；若未来据此改模型，不应再将同一test称为独立最终检验。', '']
    (out / '官方测试集评估报告.md').write_text('\n'.join(lines), encoding='utf-8')
    js(out / 'verification.json', {'status': 'passed', 'prediction_rows': 727,
                                  'frozen_artifacts_unchanged': True, 'video_overlap': False,
                                  'prediction_sha256': sha(out / 'predictions.joblib')})
    js(out / 'complete.json', {'status': 'completed', 'finished_utc': datetime.now(timezone.utc).isoformat(),
                             'evaluation_protocol_sha256': sha(out / 'evaluation_protocol.json'),
                             'metrics_sha256': sha(out / 'metrics.json')})
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
