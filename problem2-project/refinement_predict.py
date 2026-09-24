"""Reload the frozen v2 model and infer unlabeled aligned attachment 3."""
import argparse,json
import joblib
from threadpoolctl import threadpool_limits
from formal_run import ROOT,sha,sample_list
from p2 import DATA
from aligned_dataset import AlignedDataset
from formal_features import SemanticEncoder
from refinement_run import export_predictions

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-dir',default='outputs/refinement_v2');args=ap.parse_args();out=ROOT/args.run_dir
    manifest=json.loads((out/'model_manifest.json').read_text())
    for filename,key in [('feature_map.joblib','feature_map_sha256'),('models.joblib','models_sha256'),('frozen_selection.json','selection_sha256')]:
        assert sha(out/filename)==manifest[key],filename
    frozen=json.loads((out/'frozen_selection.json').read_text(encoding='utf-8'))
    assert sha(out/'protocol.json')==frozen['protocol_sha256']
    cfg=json.loads((out/'protocol.json').read_text(encoding='utf-8'))
    with threadpool_limits(limits=4):
        mapper=joblib.load(out/'feature_map.joblib');model=joblib.load(out/'models.joblib')['v2_selected']
        ds=AlignedDataset.from_attachment3(DATA/'附件3-模态缺失特征样本/对齐版本',special_token_ids=(101,102))
        ss=sample_list(ds);encoder=SemanticEncoder(ROOT/cfg['encoder'])
        p,z=model.predict(mapper.blocks(ss,encoder.encode(ss)))
        export_predictions(out/'附件3_问题2_优化版重载预测.csv',ss,p,z)
    print('Reloaded inference:',len(ss),flush=True)

if __name__=='__main__':main()
