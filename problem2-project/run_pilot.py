"""Small end-to-end pilot; results are not formal competition conclusions."""
import argparse
import csv
import importlib.metadata
import itertools
import json
import pickle
import platform
import time
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from p2 import MODALITIES, candidate_masks, corrupt, load_features, metrics, pooled_features

HERE = Path(__file__).resolve().parent


def subset(part, limit, seed):
    # Selection uses IDs/index only, not label filtering; no split reshuffling.
    n = len(part["id"])
    idx = np.sort(np.random.default_rng(seed).choice(n, min(limit, n), replace=False))
    return {k: np.asarray(v)[idx] for k, v in part.items()}


def fit_heads(x, y, r, seed):
    if set(np.unique(y)) != {0, 1, 2}:
        raise ValueError("Training subset must cover all three classes; increase --train-limit")
    scaler = StandardScaler().fit(x)
    z = scaler.transform(x)
    cls = LogisticRegression(C=0.1, max_iter=1000, random_state=seed).fit(z, y)
    reg = Ridge(alpha=100.0, solver="lsqr").fit(z, r)
    return scaler, cls, reg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-limit", type=int, default=256)
    parser.add_argument("--valid-limit", type=int, default=96)
    parser.add_argument("--output", default="outputs/pilot")
    args = parser.parse_args()
    if args.train_limit < 3 or args.valid_limit < 3:
        parser.error("Sample limits must be at least 3")
    cfg = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    if cfg["mask_policy"] != "aligned_text_attention_candidate" or cfg["zero_policy"] != "exclude_all_zero_candidate":
        raise ValueError("Unimplemented mask policy; implement and verify it before changing the configuration")
    out = HERE / args.output
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    data = load_features(cfg["feature_version"])
    train = subset(data["train"], args.train_limit, cfg["seed"])
    valid = subset(data["valid"], args.valid_limit, cfg["seed"] + 1)
    del data  # Official test is never evaluated or used for model selection.
    xt = {m: train[m] for m in MODALITIES}
    xv = {m: valid[m] for m in MODALITIES}
    vt, at = candidate_masks(train, cfg["feature_version"])
    vv, av = candidate_masks(valid, cfg["feature_version"])
    y, r = train["classification_labels"].astype(int), train["regression_labels"]
    yt, rt = valid["classification_labels"].astype(int), valid["regression_labels"]
    clean = pooled_features(xt, at)
    corrupted, mask, _, train_records = corrupt(xt, vt, at, train["id"], modalities=MODALITIES,
                                               rate=.3, seed=cfg["seed"], position="random")
    damaged = pooled_features(corrupted, mask)
    models = {"clean_train": fit_heads(clean, y, r, cfg["seed"]),
              "missing_aug_train": fit_heads(np.concatenate([clean, damaged]), np.tile(y, 2),
                                               np.tile(r, 2), cfg["seed"])}
    with (out / "pilot_models.pkl").open("wb") as h:
        pickle.dump({"models": models, "config": cfg, "class_mapping": {0: "Negative", 1: "Neutral", 2: "Positive"},
                     "status": "pilot_only", "train_ids": train["id"].tolist()}, h)
    scenarios = [((), 0., "none")]
    for size in range(1, 4):
        for combo in itertools.combinations(MODALITIES, size):
            for rate in cfg["missing_rates"]:
                if rate > 0:
                    scenarios.extend((combo, rate, pos) for pos in cfg["positions"])
    rows, predictions = [], []
    with (out / "mask_records.jsonl").open("w", encoding="utf-8") as log:
        for row in train_records:
            log.write(json.dumps({"split": "train", "scenario": "augmentation", **row}, ensure_ascii=False) + "\n")
        for combo, rate, pos in scenarios:
            scenario = f"{'+'.join(combo) or 'complete'}|{rate}|{pos}"
            if combo:
                corrupted, mask, _, records = corrupt(xv, vv, av, valid["id"], modalities=combo,
                                                     rate=rate, position=pos, seed=cfg["seed"] + 2)
                x = pooled_features(corrupted, mask)
                for row in records:
                    log.write(json.dumps({"split": "valid", "scenario": scenario, **row}, ensure_ascii=False) + "\n")
            else:
                x = pooled_features(xv, av)
            for name, (scaler, cls, reg) in models.items():
                z = scaler.transform(x)
                pc, pr = cls.predict(z), np.clip(reg.predict(z), -3, 3)
                probabilities = cls.predict_proba(z)
                score = metrics(yt, rt, pc, pr)
                per_class = score.pop("f1_per_class")
                rows.append({"model": name, "scenario": scenario, "modalities": '+'.join(combo) or 'complete',
                             "rate": rate, "position": pos, "n_valid": len(yt), **score,
                             **{f"f1_class_{i}": v for i, v in enumerate(per_class)}})
                for i, sid in enumerate(valid["id"]):
                    predictions.append({"model": name, "scenario": scenario, "id": sid,
                                        "true_class": int(yt[i]), "pred_class": int(pc[i]),
                                        "true_intensity": float(rt[i]), "pred_intensity": float(pr[i]),
                                        **{f"prob_class_{int(c)}": float(probabilities[i, j]) for j, c in enumerate(cls.classes_)}})
    for filename, records in [("metrics.csv", rows), ("validation_predictions.csv", predictions)]:
        with (out / filename).open("w", encoding="utf-8-sig", newline="") as h:
            writer = csv.DictWriter(h, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for name in models:
        base = next(x for x in rows if x["model"] == name and x["rate"] == 0)
        selected = [x for x in rows if x["model"] == name and x["modalities"] == 'text+audio+vision' and x["position"] == 'random']
        selected = [base] + selected
        for ax, metric in zip(axes, ["mae", "f1_macro"]):
            ax.plot([x["rate"] for x in selected], [x[metric] for x in selected], marker="o", label=name)
            ax.set(xlabel="Requested local missing ratio", ylabel=metric, title="Pilot only: random intervals in all modalities")
            ax.legend(fontsize=8)
            ax.grid(alpha=.25)
    fig.savefig(out / "pilot_curves.png", dpi=160)
    plt.close(fig)
    manifest = {"status": "engineering_pilot_not_final_results", "config": cfg,
                "train_n": len(y), "valid_n": len(yt), "scenario_count": len(scenarios),
                "train_ids": train["id"].tolist(), "valid_ids": valid["id"].tolist(),
                "seed_selection_train": cfg["seed"], "seed_selection_valid": cfg["seed"] + 1,
                "seed_validation_masks": cfg["seed"] + 2,
                "python": platform.python_version(),
                "packages": {p: importlib.metadata.version(p) for p in ["numpy", "scikit-learn", "matplotlib"]},
                "hyperparameters": {"logistic_C": .1, "logistic_max_iter": 1000, "ridge_alpha": 100.,
                                     "augmentation_rate": .3, "regression_clip": [-3, 3]},
                "seconds": time.time()-start,
                "limitations": ["provisional masks", "single seed", "small subsets", "orderless mean pooling",
                                "no official test evaluation", "no attachment3 inference", "no hyperparameter selection"]}
    (out / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"train": len(y), "valid": len(yt), "scenarios": len(scenarios), "seconds": manifest["seconds"],
                      "complete_input_metrics": [x for x in rows if x["rate"] == 0]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
