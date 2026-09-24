"""Build preprocessing/scenarios, exercise real-data CPU training and inference."""
import csv
import hashlib
import json
import pickle
from pathlib import Path
import torch
from aligned_dataset import AlignedDataset
from p2 import DATA, metrics
from pipeline import MaskedStandardizer, PreparedDataset
from scenarios import build_library, load_scenario
from torch_training import TokenBaseline, train_epoch, predict, save_checkpoint, restore_checkpoint


def subset(dataset, n):
    return AlignedDataset({k:v[:n] for k,v in dataset.part.items()},labeled=dataset.labeled,source=dataset.source)


def write_csv(path, rows):
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    out=Path(__file__).resolve().parent/"outputs"/"infrastructure"
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.manual_seed(20260923)
    with (DATA/"附件2-数据集特征文件"/"aligned_50.pkl").open("rb") as f: raw=pickle.load(f)
    train=AlignedDataset(AlignedDataset._select(raw["train"]),labeled=True,source="attachment2/train")
    valid=AlignedDataset(AlignedDataset._select(raw["valid"]),labeled=True,source="attachment2/valid")
    del raw
    norm=MaskedStandardizer().fit(train,split="train")
    norm.save(out/"standardizer.json")
    sha=hashlib.sha256((out/"standardizer.json").read_bytes()).hexdigest()
    print("Standardizer fitted on all",len(train),"training samples",flush=True)
    library=out/"validation_scenarios.jsonl.gz"
    manifest=build_library(valid,library)
    print("Validation library:",manifest["record_count"],"records",flush=True)
    # Deliberately tiny engineering run; no model/threshold search.
    train_small, valid_small=subset(train,65),subset(valid,33)
    training=PreparedDataset(train_small,norm,augment=True,seed=17)
    plans=load_scenario(library,"17|random|text+audio+vision|0.3")
    validation=PreparedDataset(valid_small,norm,plans=plans)
    model=TokenBaseline();optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    logs=[]
    logs.append(train_epoch(model,training,optimizer,epoch=0,seed=20260923))
    save_checkpoint(out/"resume_epoch1.pt",model,optimizer,1,seed=20260923,preprocessing_sha256=sha)
    restored=TokenBaseline(); restored_opt=torch.optim.Adam(restored.parameters(),lr=.001)
    epoch,seed=restore_checkpoint(out/"resume_epoch1.pt",restored,restored_opt,preprocessing_sha256=sha)
    assert predict(model,validation)==predict(restored,validation)
    logs.append(train_epoch(restored,training,restored_opt,epoch=epoch,seed=seed))
    save_checkpoint(out/"smoke_model.pt",restored,restored_opt,2,seed=seed,preprocessing_sha256=sha)
    predictions=predict(restored,validation)
    write_csv(out/"validation_smoke_predictions.csv",predictions)
    score=metrics(valid_small.part["classification_labels"],valid_small.part["regression_labels"],
                  [r["pred_class"] for r in predictions],[r["pred_intensity"] for r in predictions])
    special=AlignedDataset.from_attachment3(DATA/"附件3-模态缺失特征样本"/"对齐版本")
    special_predictions=predict(restored,PreparedDataset(special,norm))
    write_csv(out/"attachment3_SMOKE_ONLY_NOT_SUBMISSION.csv",special_predictions)
    report={"status":"engineering_smoke_not_model_conclusion", "torch":torch.__version__, "device":"cpu",
            "model":"random token embeddings + masked means; not pretrained BERT", "train_n":65,"valid_n":33,
            "normalization_fit_n":len(train),"train_ids":train_small.part["id"],"valid_ids":valid_small.part["id"],
            "selection":"first rows of each official split for smoke only", "seed":seed,"epochs":logs,
            "class_weight":1.,"regression_weight":1.,"batch_size":16,"optimizer":"Adam", "learning_rate":.001,
            "augmentation":{"rate":.3,"position":"random","seed":17,"epoch_dependent":True},
            "checkpoint_roundtrip_equal":True,"standardizer_sha256":sha,
            "validation_scenario":"17|random|text+audio+vision|0.3", "metrics":score,
            "special_inference_n":len(special_predictions),"scenario_library":manifest,
            "limitations":["candidate extent mask","no tokenizer identity assumed","CPU epoch-boundary resume only",
                           "no official test evaluation","not full training","no model selection"]}
    (out/"run_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"epochs":logs,"special_inference_n":len(special_predictions),"metrics":score},indent=2),flush=True)


if __name__=="__main__":main()
