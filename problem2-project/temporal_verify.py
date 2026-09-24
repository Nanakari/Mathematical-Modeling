"""Read-only recomputation except writing this run's verification summary."""
import argparse
import csv
import itertools
import joblib
import numpy as np
from accuracy_run import rank, score, read
from formal_run import ROOT, sha, js
from temporal_run import verify_pretrained


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run-dir',default='outputs/temporal_v4')
    args=ap.parse_args(); out=ROOT/args.run_dir
    cfg=read(out/'protocol.json'); selected=read(out/'frozen_selection.json')
    verify_pretrained(cfg)
    for name,digest in read(out/'source_hashes.json').items(): assert sha(ROOT/name)==digest,name
    for name,digest in read(out/'model_manifest.json').items(): assert sha(out/name)==digest,name
    for name,digest in read(out/'previous_artifacts.json').items(): assert sha(ROOT/'outputs/accuracy_v3'/name)==digest,name
    assert sha(out/'frozen_selection.json')==read(out/'complete.json')['selection_sha256']
    ids=read(out/'split_manifest.json')
    assert len(ids['train_ids'])==3395 and len(ids['valid_ids'])==728
    assert not ({s.split('$_$')[0] for s in ids['train_ids']} & {s.split('$_$')[0] for s in ids['valid_ids']})
    assert selected['official_test_evaluated'] is False
    token=read(out/'tokenizer_verification.json')
    assert token['train']['exact_sequence_matches']==3395 and token['valid']['exact_sequence_matches']==728
    candidates=read(out/'candidate_summary.json'); assert len(candidates)==len(cfg['stages'])*len(cfg['seeds'])
    assert {(row['stage'],row['seed']) for row in candidates}==set(itertools.product(cfg['stages'],cfg['seeds']))
    data=joblib.load(out/'validation_predictions.joblib'); assert data['ids']==ids['valid_ids']
    metrics=read(out/'validation_metrics.json'); assert len(metrics)==160
    lookup={(row['model'],row['scenario']):row for row in metrics}; assert len(lookup)==160
    for case,values in data['predictions'].items():
        for model,(p,z) in values.items():
            assert p.shape==(728,3) and z.shape==(728,)
            assert np.isfinite(p).all() and np.isfinite(z).all() and (p>=0).all() and (abs(z)<=3).all()
            np.testing.assert_allclose(p.sum(1),1,atol=1e-6)
            measured=score(data['y'],data['r'],p,z)
            for key in ['accuracy','f1_macro','f1_weighted','mae','pearson','neutral_recall']:
                np.testing.assert_allclose(measured[key],lookup[(model,case)][key],atol=1e-12)
    for row in candidates:
        assert sha(out/row['checkpoint'])==row['checkpoint_sha256']
        assert row['max_embedding_change']>0
        history=read(out/'candidates'/(row['name']+'_history.json'))
        assert max(history,key=rank)['epoch']==row['epoch']
        prediction=joblib.load(out/'candidates'/(row['name']+'_valid.joblib'))
        measured=score(data['y'],data['r'],prediction['probabilities'],prediction['intensity'])
        for key in ['accuracy','f1_macro','mae']:
            np.testing.assert_allclose(measured[key],row['metrics'][key],atol=1e-12)
    comparison=read(out/'architecture_comparison.json')
    for stage in cfg['stages']:
        members=[row for row in candidates if row['stage']==stage]
        values=[joblib.load(out/'candidates'/(row['name']+'_valid.joblib')) for row in members]
        p=np.mean([v['probabilities'] for v in values],axis=0); z=np.mean([v['intensity'] for v in values],axis=0)
        np.testing.assert_allclose(p,data['predictions']['complete'][stage][0],atol=1e-6)
        np.testing.assert_allclose(z,data['predictions']['complete'][stage][1],atol=1e-6)
        recorded=next(row for row in comparison if row['stage']==stage)
        acc=[row['metrics']['accuracy'] for row in members]
        np.testing.assert_allclose(recorded['seed_accuracy_mean'],np.mean(acc))
        np.testing.assert_allclose(recorded['seed_accuracy_std'],np.std(acc,ddof=1))
        for key in ['accuracy','f1_macro','mae']:
            np.testing.assert_allclose(recorded['metrics'][key],lookup[(stage,'complete')][key],atol=1e-6)
    best=max(comparison,key=lambda row:rank(row['metrics']))
    assert best==selected['best_new']
    expected=best['stage'] if rank(best['metrics'])>rank(selected['previous_metrics']) else 'accuracy_v3'
    assert selected['selected']==expected
    if expected!='accuracy_v3': assert len(selected['selected_candidates'])==len(cfg['seeds'])
    def csvrows(name):
        with (out/name).open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))
    a,b=csvrows('附件3_时序融合候选预测.csv'),csvrows('附件3_时序融合重载预测.csv')
    assert len(a)==len(b)==len({row['sample_id'] for row in a})==30
    difference=0.
    for first,second in zip(a,b):
        for key in ['sample_id','source_file','polarity']: assert first[key]==second[key]
        for key in ['intensity','prob_0','prob_1','prob_2']:
            delta=abs(float(first[key])-float(second[key])); difference=max(difference,delta); assert delta<=1e-6
    result={'status':'passed','candidate_count':len(candidates),'seeds_per_architecture':len(cfg['seeds']),
            'metric_rows_recomputed':len(metrics),'sample_case_predictions_checked':len(metrics)*728,
            'selected_model':expected,'complete_validation_accuracy':lookup[(expected,'complete')]['accuracy'],
            'target_met_on_selection_validation':lookup[(expected,'complete')]['accuracy']>=cfg['target'],
            'special_rows':30,'reload_max_difference':difference,'official_test_evaluated':False}
    js(out/'verification.json',result); print(result)


if __name__=='__main__': main()
