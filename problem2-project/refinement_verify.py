"""Independent consistency checks on the completed v2 artifacts."""
import argparse,csv,json
from pathlib import Path
import joblib
import numpy as np
from formal_run import sha
from p2 import metrics

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-dir',default='outputs/refinement_v2');args=ap.parse_args()
    root=Path(__file__).resolve().parent;out=root/args.run_dir
    read=lambda name:json.loads((out/name).read_text(encoding='utf-8'))
    f=read('frozen_selection.json');manifest=read('model_manifest.json');done=read('validation_complete.json')
    old=root/'outputs/formal';old_manifest=json.loads((old/'frozen_selection.json').read_text(encoding='utf-8'))
    for name,key in [('feature_map.joblib','feature_map_sha256'),('models.joblib','models_sha256'),('protocol.json','protocol_sha256')]:assert sha(old/name)==old_manifest[key]
    assert sha(out/'protocol.json')==f['protocol_sha256']
    for name,key in [('models.joblib','models_sha256'),('feature_map.joblib','feature_map_sha256'),('frozen_selection.json','selection_sha256')]:assert sha(out/name)==manifest[key]
    assert done['selection_sha256']==manifest['selection_sha256'] and done['official_test_evaluated'] is False
    for name,digest in f['source_hashes'].items():assert sha(root/name)==digest,name
    folds=read('fold_manifest.json');seen=set()
    for fold in folds:
        tr=set(fold['train_ids']);va=set(fold['holdout_ids']);assert not seen&va;seen|=va
        assert not tr&va and not {s.split('$_$')[0] for s in tr}&{s.split('$_$')[0] for s in va}
        assert tr|va==set(f['train_ids'])
    assert seen==set(f['train_ids']) and len(seen)==3395
    fold_logs_checked=all((out/f'fold_{i}_predictions.npz').exists() for i in range(3))
    if fold_logs_checked:
        for i,fold in enumerate(folds):
            with np.load(out/f'fold_{i}_predictions.npz') as cache:
                p=cache['classification'];z=cache['regression'];n=len(fold['holdout_ids'])
                assert p.shape==(24,32,n,3) and z.shape==(60,32,n)
                assert np.isfinite(p).all() and np.isfinite(z).all() and (p>=0).all()
                np.testing.assert_allclose(p.sum(-1),1,atol=1e-12)
    raw=joblib.load(out/'validation_predictions.joblib');y=raw['classes'];r=raw['intensity']
    assert not {s.split('$_$')[0] for s in seen}&{s.split('$_$')[0] for s in raw['ids']}
    rows=read('validation_metrics.json');assert len(rows)==8*32
    for row in rows:
        pred=raw['predictions'][row['model']+'::'+row['scenario']];p=np.array(pred['probabilities']);z=np.array(pred['intensity'])
        assert p.shape==(728,3) and z.shape==(728,) and np.isfinite(p).all() and np.isfinite(z).all() and (p>=0).all() and (abs(z)<=3).all()
        np.testing.assert_allclose(p.sum(1),1,atol=1e-12)
        score=metrics(y,r,p.argmax(1),z)
        for key in ['accuracy','f1_macro','mae','pearson']:np.testing.assert_allclose(score[key],row[key],atol=1e-12)
    def csvrows(name):
        with (out/name).open(encoding='utf-8-sig') as handle:return list(csv.DictReader(handle))
    a=csvrows('附件3_问题2_优化版预测.csv');b=csvrows('附件3_问题2_优化版重载预测.csv')
    assert len(a)==len(b)==len({row['sample_id'] for row in a})==30
    largest=0.
    for x,y in zip(a,b):
        for key in ['sample_id','source_file','polarity']:assert x[key]==y[key]
        for key in ['intensity','decision_score_0','decision_score_1','decision_score_2']:
            difference=abs(float(x[key])-float(y[key]));largest=max(largest,difference);assert difference<=1e-6
    report=(out/'问题2_第二轮优化报告.md').read_text(encoding='utf-8');assert not any(ord(c)<32 and c not in '\n\r\t' for c in report)
    result={'status':'passed','train_n':3395,'folds':3,'cross_fold_video_overlap':False,'train_valid_video_overlap':False,
        'recomputed_validation_metric_rows':len(rows),'checked_sample_model_case_predictions':len(rows)*728,
        'special_rows':30,'reload_classes_exact_match':True,'reload_max_absolute_difference':largest,
        'official_test_evaluated':False,'hashes_match':True,'full_fold_predictions_checked':fold_logs_checked}
    (out/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
