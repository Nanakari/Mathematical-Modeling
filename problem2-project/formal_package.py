"""Package exact selected artifacts, source, fixed pretrained files and report."""
import argparse
import json
import zipfile
from pathlib import Path


def main():
    root=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',default='outputs/formal');args=parser.parse_args()
    out=root/args.run_dir
    names=['aligned_dataset.py','pipeline.py','p2.py','formal_features.py','formal_model.py','formal_run.py',
           'formal_predict.py','formal_analysis.py','formal_answer.py','formal_package.py','verify_tokenizer.py',
           'test_formal.py','test_pipeline.py','torch_training.py','scenarios.py',
           'formal_verify.py','FORMAL_README.md','requirements-formal.txt','configs/formal_protocol.json']
    files=[root/name for name in names]
    files+=list((root/'models/bert_tiny').glob('*'))
    names=['问题2_完整解答.md','feature_map.joblib','models.joblib','protocol.json','environment.json',
           'frozen_selection.json','tokenizer_verification.json','evaluation_complete.json','validation_search.json',
           'validation_selected.json','training_seed_variation.json','robustness_summary.csv','condition_summary.csv',
           'bootstrap_confidence_intervals.csv','test_metrics.csv','test_error_by_class.csv','test_error_cases.csv',
           'test_diagnostics.png','test_robustness_curves.png','test_position_heatmap.png','附件3_问题2_预测结果.csv',
           '附件3_问题2_重载预测.csv','test_anchors.json']
    files += [out/name for name in names]
    if (out/'verification.json').exists():files.append(out/'verification.json')
    target=out/'问题2_模型代码与结果.zip'
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for path in files:
            if not path.is_file():raise FileNotFoundError(path)
            z.write(path,Path('problem2')/path.relative_to(root))
    with zipfile.ZipFile(target) as z:
        if z.testzip():raise ValueError('Archive integrity error')
    if target.stat().st_size>50_000_000:raise ValueError('Archive exceeds 50 MB')
    print('Package bytes:',target.stat().st_size,flush=True)


if __name__=='__main__':main()
