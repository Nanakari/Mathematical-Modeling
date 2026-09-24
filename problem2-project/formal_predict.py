"""Reload frozen formal artifacts and predict attachment 3 without labels."""
import argparse
import csv
import json
from pathlib import Path
import joblib
from threadpoolctl import threadpool_limits
from formal_run import ROOT,sha,sample_list
from p2 import DATA
from aligned_dataset import AlignedDataset
from formal_features import SemanticEncoder
from formal_model import ensemble


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='附件3_问题2_重载预测.csv')
    parser.add_argument('--run-dir',default='outputs/formal')
    args=parser.parse_args();out=ROOT/args.run_dir
    frozen=json.loads((out/'frozen_selection.json').read_text(encoding='utf-8'))
    cfg=json.loads((out/'protocol.json').read_text(encoding='utf-8'))
    for name,key in [('feature_map.joblib','feature_map_sha256'),('models.joblib','models_sha256'),('protocol.json','protocol_sha256')]:
        if sha(out/name)!=frozen[key]:raise ValueError('Frozen artifact hash mismatch: '+name)
    provenance=json.loads((ROOT/cfg['text_model_dir']/'provenance.json').read_text(encoding='utf-8'))
    for name,spec in provenance['files'].items():
        if sha(ROOT/cfg['text_model_dir']/name)!=spec['sha256']:raise ValueError('Pretrained asset changed: '+name)
    with threadpool_limits(limits=4):
        mapper=joblib.load(out/'feature_map.joblib');models=joblib.load(out/'models.joblib')['proposed']
        ds=AlignedDataset.from_attachment3(DATA/'附件3-模态缺失特征样本'/'对齐版本',special_token_ids=(101,102))
        samples=sample_list(ds);encoder=SemanticEncoder(ROOT/cfg['text_model_dir'])
        p,r=ensemble(models,mapper.blocks(samples,encoder.encode(samples)))
        rows=[{'sample_id':s['sample_id'],'source_file':s['sample_id'].split('#')[0],
                'polarity':['Negative','Neutral','Positive'][int(p[i].argmax())],'intensity':float(r[i]),
                **{f'prob_{j}':float(p[i,j]) for j in range(3)}} for i,s in enumerate(samples)]
        with (out/args.output).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        print('Frozen inference rows:',len(rows),flush=True)


if __name__=='__main__':main()
