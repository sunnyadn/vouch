#!/usr/bin/env -S uv run --quiet --with datasets --with huggingface_hub
"""
Sample 100 stratified instances from LLM-AggreFact's test split for vouch
verifier-swap calibration.

Stratification: ~9 instances per constituent dataset (11 total), balanced
labels (~half supported, half not). Deterministic given a fixed seed.

Output: bench/phase0/calibration_sample.jsonl — one JSON object per line,
each with `dataset`, `doc`, `claim`, `label`, `contamination_identifier`.
The output file is NOT committed to the OSS repo (LLM-AggreFact is CC BY-ND;
sampling could be argued to be a derivative). Each reproducer runs this
script locally against their own HuggingFace auth.

Prerequisites (one-time, manual user actions):
  1. Get an HF token at https://huggingface.co/settings/tokens (read scope is enough)
  2. Visit https://huggingface.co/datasets/lytang/LLM-AggreFact and click
     "Access repository" to accept the use-as-evaluation-benchmark terms
  3. Export HF_TOKEN=... or run `huggingface-cli login`

Usage:
  ./sample_llm_aggrefact.py
  # or
  uv run --with datasets --with huggingface_hub bench/phase0/sample_llm_aggrefact.py
"""

import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset

SEED = 20260510
TOTAL_TARGET = 100
DATASET_NAME = "lytang/LLM-AggreFact"
SPLIT = "test"
OUT_PATH = Path(__file__).parent / "calibration_sample.jsonl"


def main() -> int:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    print(f"[sample_llm_aggrefact] HF_TOKEN: {'set' if token else 'unset'}", file=sys.stderr)

    print(f"[sample_llm_aggrefact] loading {DATASET_NAME} split={SPLIT}", file=sys.stderr)
    ds = load_dataset(DATASET_NAME, split=SPLIT, token=token)
    print(f"[sample_llm_aggrefact] loaded {len(ds)} instances", file=sys.stderr)

    by_dataset = defaultdict(lambda: {"pos": [], "neg": []})
    for i, row in enumerate(ds):
        bucket = by_dataset[row["dataset"]]
        if row["label"] == 1:
            bucket["pos"].append(i)
        else:
            bucket["neg"].append(i)

    constituent = sorted(by_dataset.keys())
    print(f"[sample_llm_aggrefact] {len(constituent)} constituent datasets:", file=sys.stderr)
    for d in constituent:
        b = by_dataset[d]
        print(f"  {d}: pos={len(b['pos'])} neg={len(b['neg'])}", file=sys.stderr)

    rng = random.Random(SEED)
    per_dataset_target = TOTAL_TARGET // len(constituent)  # 9 each at 11 datasets
    extra = TOTAL_TARGET - per_dataset_target * len(constituent)  # 1 extra at 11

    picked_indices: list[int] = []
    for di, dname in enumerate(constituent):
        target_here = per_dataset_target + (1 if di < extra else 0)
        pos_target = target_here // 2
        neg_target = target_here - pos_target

        b = by_dataset[dname]
        pos = rng.sample(b["pos"], min(pos_target, len(b["pos"])))
        neg = rng.sample(b["neg"], min(neg_target, len(b["neg"])))

        # If a side ran short, top up from the other
        deficit = target_here - len(pos) - len(neg)
        if deficit > 0:
            other = b["pos"] if len(b["pos"]) > len(b["neg"]) else b["neg"]
            already = set(pos) | set(neg)
            pool = [i for i in other if i not in already]
            top_up = rng.sample(pool, min(deficit, len(pool)))
            pos.extend(i for i in top_up if i in b["pos"])
            neg.extend(i for i in top_up if i in b["neg"])

        picked_indices.extend(pos + neg)

    rng.shuffle(picked_indices)
    picked_indices = picked_indices[:TOTAL_TARGET]

    written = 0
    label_counts = defaultdict(int)
    dataset_counts = defaultdict(int)
    with OUT_PATH.open("w") as fp:
        for idx in picked_indices:
            row = ds[idx]
            entry = {
                "dataset": row["dataset"],
                "doc": row["doc"],
                "claim": row["claim"],
                "label": int(row["label"]),
                "contamination_identifier": row["contamination_identifier"],
            }
            fp.write(json.dumps(entry, ensure_ascii=False) + "\n")
            written += 1
            label_counts[entry["label"]] += 1
            dataset_counts[entry["dataset"]] += 1

    print(f"[sample_llm_aggrefact] wrote {written} entries to {OUT_PATH}", file=sys.stderr)
    print(f"[sample_llm_aggrefact] label distribution: {dict(label_counts)}", file=sys.stderr)
    print(f"[sample_llm_aggrefact] dataset distribution:", file=sys.stderr)
    for d in sorted(dataset_counts.keys()):
        print(f"  {d}: {dataset_counts[d]}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
