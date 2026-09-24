"""Recompute accuracy experiment outputs without training or model selection."""
import argparse
import csv
import json
from pathlib import Path
import joblib
import numpy as np
from accuracy_run import ROOT, read, sha, score, rank, js


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--run-dir', default='outputs/accuracy_v3')
    args = parser.parse_args(); out = ROOT / args.run_dir
    cfg = read(out / 'protocol.json'); selection = read(out / 'frozen_selection.json')
    assert read(out / 'complete.json')['selection_sha256'] == sha(out / 'frozen_selection.json')
    for filename, digest in read(out / 'model_manifest.json').items():
        assert sha(out / filename) == digest, filename
    for filename, digest in read(out / 'source_hashes.json').items():
        assert sha(ROOT / filename) == digest, filename
    ids = read(out / 'split_manifest.json')
    assert len(ids['train_ids']) == 3395 and len(ids['valid_ids']) == 728
    assert not ({s.split('$_$')[0] for s in ids['train_ids']} & {s.split('$_$')[0] for s in ids['valid_ids']})
    assert selection['official_test_evaluated'] is False
    linear = read(out / 'linear_search.json'); neural = read(out / 'neural_search.json')
    assert len(linear) == 9 and len(neural) == 9
    assert max(linear, key=lambda s: rank(s['metrics'])) == selection['best_linear']
    assert max(neural, key=lambda s: rank(s['metrics'])) == selection['best_neural']
    for row in neural:
        assert sha(out / row['checkpoint']) == row['checkpoint_sha256']
        assert (row['max_embedding_change'] == 0) == row['frozen']
        history = read(out / 'candidates' / (row['name'] + '_history.json'))
        best = max(history, key=rank)
        assert best['epoch'] == row['epoch']
        for metric in ['accuracy', 'f1_macro', 'mae']:
            np.testing.assert_allclose(best[metric], row['metrics'][metric], atol=1e-12)
    original_seed17 = read(ROOT / cfg['formal_dir'] / 'validation_selected.json')['proposed_seed17']['complete']
    control = next(s for s in linear if s['gamma'] == 1 and s['C'] == .5)['metrics']
    for metric in ['accuracy', 'f1_macro', 'mae', 'pearson']:
        np.testing.assert_allclose(control[metric], original_seed17[metric], atol=1e-10)
    data = joblib.load(out / 'validation_predictions.joblib')
    assert data['ids'] == ids['valid_ids']
    metrics = read(out / 'validation_metrics.json'); assert len(metrics) == 96
    lookup = {(s['model'], s['scenario']): s for s in metrics}; assert len(lookup) == 96
    for case, predictions in data['predictions'].items():
        for model, (p, z) in predictions.items():
            assert p.shape == (728, 3) and z.shape == (728,)
            assert np.isfinite(p).all() and np.isfinite(z).all() and (p >= 0).all() and (abs(z) <= 3).all()
            np.testing.assert_allclose(p.sum(1), 1, atol=1e-6)
            measured = score(data['y'], data['r'], p, z)
            for key in ['accuracy', 'f1_macro', 'mae', 'pearson', 'neutral_recall']:
                np.testing.assert_allclose(measured[key], lookup[(model, case)][key], atol=1e-12)
    complete = {m: lookup[(m, 'complete')] for m in ['original', 'linear', 'neural']}
    chosen = max(complete, key=lambda m: rank(complete[m]))
    # Production ties retain original; neural/linear ties retain linear.
    assert chosen == selection['kind']
    for kind, source in [('linear', selection['best_linear']['metrics']), ('neural', selection['best_neural']['metrics']), ('original', selection['original_metrics'])]:
        for key in ['accuracy', 'f1_macro', 'mae']:
            np.testing.assert_allclose(complete[kind][key], source[key], atol=1e-6)
    def csvrows(name):
        with (out / name).open(encoding='utf-8-sig', newline='') as f: return list(csv.DictReader(f))
    a, b = csvrows('附件3_Accuracy候选预测.csv'), csvrows('附件3_Accuracy候选重载预测.csv')
    assert len(a) == len(b) == len({r['sample_id'] for r in a}) == 30
    difference = 0.
    for first, second in zip(a, b):
        for key in ['sample_id', 'source_file', 'polarity']: assert first[key] == second[key]
        for key in ['intensity', 'prob_0', 'prob_1', 'prob_2']:
            delta = abs(float(first[key]) - float(second[key])); difference = max(difference, delta)
            assert delta <= 1e-6
    result = {'status': 'passed', 'metric_rows_recomputed': 96, 'sample_case_predictions_checked': 96*728,
              'neural_candidates_checked': 9, 'linear_candidates_checked': 9,
              'old_balanced_seed17_control_reproduced': True, 'encoder_update_controls_passed': True,
              'selected_model': chosen, 'complete_validation_accuracy': complete[chosen]['accuracy'],
              'target_met_on_selection_validation': complete[chosen]['accuracy'] >= cfg['target'],
              'official_test_evaluated': False, 'special_rows': 30, 'reload_max_difference': difference}
    js(out / 'verification.json', result); print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
