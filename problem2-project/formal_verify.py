"""Verify frozen artifacts without fitting or selecting models."""
import argparse,csv,gzip,hashlib,json
from pathlib import Path
import numpy as np
from p2 import metrics

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-dir',default='outputs/formal');ap.add_argument('--full-logs',action='store_true');args=ap.parse_args()
    out=Path(__file__).resolve().parent/args.run_dir
    read=lambda n:json.loads((out/n).read_text(encoding='utf-8'))
    sha=lambda n:hashlib.sha256((out/n).read_bytes()).hexdigest()
    def rows(n):
        with (out/n).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
    frozen=read('frozen_selection.json');done=read('evaluation_complete.json');anchors=read('test_anchors.json')
    for n,k in [('protocol.json','protocol_sha256'),('feature_map.joblib','feature_map_sha256'),('models.joblib','models_sha256')]:assert sha(n)==frozen[k],n
    assert sha('frozen_selection.json')==done['selection_hash'] and frozen['test_used_for_selection'] is False
    sets=[set(frozen['train_ids']),set(frozen['valid_ids']),set(anchors['ids'])]
    for groups in [sets,[{s.split('$_$')[0] for s in ids} for ids in sets]]:
        for i in range(3):
            for j in range(i):assert not groups[i]&groups[j]
    table=rows('test_metrics.csv');lookup={(r['model'],r['scenario']):r for r in table}
    assert len(table)==len(lookup)==1183 and len({r['scenario'] for r in table})==169 and len({r['model'] for r in table})==7
    original=rows('附件3_问题2_预测结果.csv');reloaded=rows('附件3_问题2_重载预测.csv')
    assert len(original)==len(reloaded)
    max_difference=0.
    for a,b in zip(original,reloaded):
        for k in ['sample_id','source_file','polarity']:assert a[k]==b[k]
        for k in ['intensity','prob_0','prob_1','prob_2']:
            difference=abs(float(a[k])-float(b[k]));max_difference=max(max_difference,difference)
            assert difference<=1e-6,(a['sample_id'],k,difference)
    assert len(original)==len({r['sample_id'] for r in original})==30
    for r in original:
        p=np.array([float(r[f'prob_{i}']) for i in range(3)])
        assert np.isfinite(p).all() and (p>=0).all() and abs(float(r['intensity']))<=3
        np.testing.assert_allclose(p.sum(),1,atol=1e-12)
    ci=rows('bootstrap_confidence_intervals.csv');assert len(ci)==10 and all(int(r['clusters'])==381 and int(r['bootstrap_repeats'])==500 for r in ci)
    report=(out/'问题2_完整解答.md').read_text(encoding='utf-8');assert not any(ord(c)<32 and c not in '\n\r\t' for c in report)
    assert '@@' not in report and chr(92)+'rho' in report
    result={'status':'passed','frozen_hashes':'matched','split_id_and_video_overlap':False,'metric_rows':1183,'scenarios':169,'model_groups':7,'special_rows':30,'reload_classes_exact_match':True,'reload_max_absolute_difference':max_difference,'reload_absolute_tolerance':1e-6,'bootstrap_repeats':500,'video_clusters':381,'full_logs_checked':args.full_logs}
    if args.full_logs:
        count=0;seen=set()
        with gzip.open(out/'test_predictions.jsonl.gz','rt',encoding='utf-8') as f:
            for line in f:
                rec=json.loads(line);key=(rec['model'],rec['scenario']);assert key not in seen;seen.add(key)
                p=np.asarray(rec['probabilities']);z=np.asarray(rec['intensity'])
                assert p.shape==(727,3) and z.shape==(727,) and np.isfinite(p).all() and np.isfinite(z).all() and (p>=0).all() and (abs(z)<=3).all()
                np.testing.assert_allclose(p.sum(1),1,atol=1e-12)
                measured=metrics(anchors['true_class'],anchors['true_intensity'],p.argmax(1),z)
                for metric in ['accuracy','f1_macro','f1_weighted','mae','pearson']:np.testing.assert_allclose(measured[metric],float(lookup[key][metric]),atol=1e-12)
                count+=727
        assert seen==set(lookup)
        nesting={};mask_count=0
        with gzip.open(out/'test_masks.jsonl.gz','rt',encoding='utf-8') as f:
            for line in f:
                rec=json.loads(line);mask_count+=1
                for mod,span in rec['intervals'].items():
                    if span:
                        a,b=span;assert 0<=a<b<=50
                        np.testing.assert_allclose(rec['actual_rates'][mod],(b-a)/rec['valid_positions'])
                    key=(rec['sample_id'],rec['seed'],rec['position'],tuple(rec['intervals']),mod)
                    nesting.setdefault(key,{})[rec['requested_rate']]=span
        assert mask_count==169*727
        for rates in nesting.values():
            previous=None
            for _,span in sorted(rates.items()):
                if previous:assert span and span[0]<=previous[0] and span[1]>=previous[1]
                if span:previous=span
        result.update(sample_model_case_predictions=count,mask_records=mask_count,recomputed_metrics_match=True,nested_intervals_verified=True)
    (out/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
