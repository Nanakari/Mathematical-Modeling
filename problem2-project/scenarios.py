"""Persist label-free validation intervals; reused unchanged across models."""
import gzip
import hashlib
import itertools
import json
from pathlib import Path
from pipeline import MODALITIES, nested_intervals, mask_fingerprint


def build_library(dataset, output, *, seeds=(17, 41, 83), rates=(.1, .3, .5)):
    if getattr(dataset, "source", "").split("/")[-1] != "valid":
        raise ValueError("Scenario library requires official validation split")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    combinations = [c for size in range(1, 4) for c in itertools.combinations(MODALITIES, size)]
    count = 0
    # gzip avoids shipping tens of MB of repeated keys; contents remain JSONL.
    with gzip.open(output, "wt", encoding="utf-8") as f:
        for i in range(len(dataset)):
            s = dataset[i]
            fingerprint = mask_fingerprint(s)
            n = int(s["extent_mask"].sum())
            base = {"sample_id": s["sample_id"], "mask_fingerprint": fingerprint, "valid_positions": n}
            f.write(json.dumps({**base, "scenario": "complete", "intervals": {}, "actual_rates": {},
                                "requested_rate": 0., "seed": 0, "position": "start"})+'\n')
            count += 1
            for seed in seeds:
                for position in ("start", "middle", "end", "random"):
                    cached = {m: nested_intervals(s, m, list(rates), position, seed) for m in MODALITIES}
                    for combo in combinations:
                        for rate in rates:
                            spans = {m: cached[m][str(rate)] for m in combo}
                            actual = {m: (span[1]-span[0])/n if span and n else 0. for m, span in spans.items()}
                            row = {**base, "scenario": f"{seed}|{position}|{'+'.join(combo)}|{rate}",
                                   "seed": seed, "position": position, "requested_rate": rate,
                                   "intervals": spans, "actual_rates": actual}
                            f.write(json.dumps(row, ensure_ascii=False)+'\n')
                            count += 1
    manifest = {"format": "gzip-jsonl", "sample_count": len(dataset), "seeds": list(seeds),
                "rates": list(rates), "scenario_count": 1+len(seeds)*4*7*len(rates),
                "record_count": count, "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "nested_rates": True, "interval_unit": "sequence_position", "labels_used": False,
                "note": "Start/middle/end are deterministic, so seeds duplicate these controls; do not treat them as independent replicates."}
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_scenario(path, name):
    plans = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["scenario"] == name:
                if row["sample_id"] in plans:
                    raise ValueError("Duplicate sample in scenario")
                plans[row["sample_id"]] = row
    if not plans:
        raise ValueError(f"Unknown scenario: {name}")
    return plans
