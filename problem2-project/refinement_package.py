import zipfile
from pathlib import Path

def main():
    root=Path(__file__).resolve().parent;out=root/'outputs/refinement_v2'
    files=[root/n for n in ['aligned_dataset.py','p2.py','pipeline.py','scenarios.py','torch_training.py',
        'formal_features.py','formal_model.py','formal_run.py','verify_tokenizer.py',
        'refinement_model.py','refinement_run.py','refinement_analysis.py','refinement_predict.py',
        'refinement_verify.py','refinement_package.py','test_refinement.py','test_pipeline.py',
        'REFINEMENT_README.md','requirements-formal.txt','configs/refinement_protocol.json']]
    files+=list((root/'models/bert_tiny').glob('*'))
    files += [root/'outputs/formal'/n for n in ['models.joblib','feature_map.joblib','frozen_selection.json','protocol.json','environment.json']]
    names=['protocol.json','source_hashes.json','fold_manifest.json','classification_search.json','regression_search.json',
        'frozen_selection.json','cv_summary.json','cv_comparison.json','feature_map.joblib','models.joblib','model_manifest.json',
        'validation_complete.json','validation_summary.csv','validation_metrics.json','validation_predictions.joblib',
        'validation_paired_bootstrap.csv','validation_component_comparison.png','问题2_第二轮优化报告.md',
        '附件3_问题2_优化版预测.csv','附件3_问题2_优化版重载预测.csv','verification.json']
    files += [out/n for n in names]
    target=out/'问题2_第二轮优化代码与结果.zip'
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for path in files:
            if not path.is_file():raise FileNotFoundError(path)
            z.write(path,Path('problem2')/path.relative_to(root))
    with zipfile.ZipFile(target) as z:assert z.testzip() is None
    assert target.stat().st_size<50_000_000
    print('Package bytes:',target.stat().st_size)

if __name__=='__main__':main()
