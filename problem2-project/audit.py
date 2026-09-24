"""Read-only audit of supplied files. No source edits, imputation or row removal."""
import argparse
import gc
import hashlib
import json
import pickle
from collections import Counter
from pathlib import Path
import numpy as np
from p2 import DATA, MODALITIES, candidate_masks, load_features

OUT = Path(__file__).resolve().parent / "outputs"


def schema(part):
    return {k: {"type": type(v).__name__, "shape": list(v.shape) if hasattr(v, "shape") else [len(v)],
                "dtype": str(v.dtype) if hasattr(v, "dtype") else None} for k, v in part.items()}


def audit(version):
    data = load_features(version)
    report = {"version": version, "splits": {}, "overlap": {}}
    for name, part in data.items():
        ids = list(map(str, part["id"]))
        classes = np.asarray(part["classification_labels"]).astype(int)
        labels = np.asarray(part["regression_labels"])
        item = {"n": len(ids), "schema": schema(part),
                "duplicate_ids": [k for k, n in Counter(ids).items() if n > 1],
                "field_sample_counts_match": all(len(v) == len(ids) for v in part.values()),
                "class_counts": dict(Counter(map(str, classes))),
                "label_range": [float(labels.min()), float(labels.max())],
                "nonfinite_labels": int((~np.isfinite(labels)).sum()),
                "sign_mapping_mismatches": int(np.sum(classes != np.sign(labels).astype(int) + 1)),
                "features": {}}
        try:
            valid, available = candidate_masks(part, version)
            item["mask_policy"] = "candidate; BERT special tokens retained; zero rows unavailable"
        except ValueError as exc:
            valid = available = None
            item["mask_error"] = str(exc)
        attn = np.asarray(part["text_bert"])[:, 1, :]
        item["attention_nonbinary"] = int((~np.isin(attn, [0, 1])).sum())
        item["attention_nonprefix_rows"] = int(np.any(np.diff(attn, axis=1) > 0, axis=1).sum())
        for m in MODALITIES:
            x = part[m]
            zero = np.all(x == 0, axis=-1)
            stats = {"shape": list(x.shape), "nonfinite_values": 0,
                     "zero_rows": int(zero.sum())}
            # Chunk scans to avoid allocating multi-GB boolean temporaries.
            for start in range(0, len(x), 64):
                stats["nonfinite_values"] += int((~np.isfinite(x[start:start+64])).sum())
            if valid is not None:
                lengths = valid[m].sum(1)
                stats.update({"candidate_valid_length_min": int(lengths.min()),
                              "candidate_valid_length_max": int(lengths.max()),
                              "zero_rows_inside_candidate_valid": int((zero & valid[m]).sum()),
                              "nonzero_rows_outside_candidate_valid": int((~zero & ~valid[m]).sum()),
                              "samples_without_available_rows": int((available[m].sum(1) == 0).sum())})
            item["features"][m] = stats
        report["splits"][name] = item
    names = list(data)
    for i, a in enumerate(names):
        for b in names[i+1:]:
            ids_a, ids_b = set(data[a]["id"]), set(data[b]["id"])
            videos_a = {x.split("$_$")[0] for x in ids_a}
            videos_b = {x.split("$_$")[0] for x in ids_b}
            report["overlap"][a + "__" + b] = {
                "sample_ids": sorted(ids_a & ids_b),
                "video_id_overlap_count": len(videos_a & videos_b),
                "note": "Preserve supplied splits; video overlap is not duplicate clips."}
    del data
    gc.collect()
    return report


def audit_special():
    folder = next(p for p in DATA.iterdir() if p.name.startswith("附件3"))
    rows = []
    for path in sorted(folder.rglob("*.pkl")):
        with path.open("rb") as handle:
            obj = pickle.load(handle)
        part = obj.get("test", obj)
        audio_shape = np.asarray(part.get("audio", [])).shape
        version = "aligned" if len(audio_shape) == 3 and audio_shape[1] == 50 else "unaligned"
        required = ["text", "text_bert", "audio", "vision"]
        if version == "unaligned":
            required += ["audio_lengths", "vision_lengths"]
        rows.append({"file": str(path.relative_to(DATA)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                     "inferred_version_by_shape": version, "schema": schema(part),
                     "missing_for_candidate_pipeline": [k for k in required if k not in part],
                     "has_id": "id" in part})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--versions", nargs="+", default=["aligned_50.pkl", "unaligned_50.pkl"])
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    for version in args.versions:
        result = audit(version)
        (OUT / (version.replace(".pkl", "") + "_audit.json")).write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(version, {s: x["n"] for s, x in result["splits"].items()}, flush=True)
    special = audit_special()
    (OUT / "attachment3_schema.json").write_text(json.dumps(special, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Attachment 3 files:", len(special), "incompatible:", sum(bool(x["missing_for_candidate_pipeline"]) for x in special))


if __name__ == "__main__":
    main()
