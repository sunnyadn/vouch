#!/usr/bin/env python3
"""Sample SQuAD-2.0 dev for vouch arm-2 eval (issue #36 Phase A).

Pulls `rajpurkar/squad_v2` dev split via HuggingFace `datasets`, splits into
unanswerable + answerable, stratifies by Wikipedia `title` so the sample
covers many topics rather than concentrating in a few high-density articles.

Outputs:
  - `unanswerable.jsonl` (N=N_UNANSWERABLE, default 50) — subset (a)
  - `control.jsonl`      (N=N_CONTROL,      default 50) — answerable control

Schema (one JSON per line):
  {
    "id": <SQuAD id>,
    "title": <Wikipedia title>,
    "passage": <context paragraph>,
    "question": <question>,
    "gt_answers": [<answer string>, ...]  # empty list for unanswerable
  }

Reproducibility: pinned seed; the snapshot revision of the HF dataset is not
pinned here (use `datasets.load_dataset(..., revision=...)` if you need a hard
freeze; the SQuAD-2.0 dev set has been stable for years).
"""
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

SEED = 42
N_UNANSWERABLE = 50
N_CONTROL = 50

HERE = Path(__file__).parent


def main():
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        print("missing: pip install datasets", file=sys.stderr)
        sys.exit(2)

    print("loading rajpurkar/squad_v2 dev split…", file=sys.stderr)
    ds = load_dataset("rajpurkar/squad_v2", split="validation")

    # Partition by has-answer
    unanswerable = []
    answerable = []
    for ex in ds:
        item = {
            "id": ex["id"],
            "title": ex["title"],
            "passage": ex["context"],
            "question": ex["question"],
            "gt_answers": list(ex["answers"]["text"]),
        }
        if not item["gt_answers"]:
            unanswerable.append(item)
        else:
            answerable.append(item)

    print(
        f"  raw: {len(unanswerable)} unanswerable / {len(answerable)} answerable",
        file=sys.stderr,
    )

    rng = random.Random(SEED)

    def stratify_by_title(items, target_n):
        """Round-robin one-per-title until we have target_n; shuffle within title."""
        by_title = defaultdict(list)
        for it in items:
            by_title[it["title"]].append(it)
        for title in by_title:
            rng.shuffle(by_title[title])
        titles = sorted(by_title)
        rng.shuffle(titles)
        out = []
        idx = 0
        empty_pass = 0
        while len(out) < target_n and empty_pass < len(titles):
            t = titles[idx % len(titles)]
            if by_title[t]:
                out.append(by_title[t].pop())
                empty_pass = 0
            else:
                empty_pass += 1
            idx += 1
        return out

    sampled_un = stratify_by_title(unanswerable, N_UNANSWERABLE)
    sampled_ctrl = stratify_by_title(answerable, N_CONTROL)

    def write_jsonl(path, items):
        with open(path, "w") as f:
            for it in items:
                f.write(json.dumps(it) + "\n")
        print(f"  wrote {len(items):3d} -> {path}", file=sys.stderr)

    write_jsonl(HERE / "unanswerable.jsonl", sampled_un)
    write_jsonl(HERE / "control.jsonl", sampled_ctrl)

    # Quick coverage stats so we can eyeball stratification
    un_titles = {it["title"] for it in sampled_un}
    ctrl_titles = {it["title"] for it in sampled_ctrl}
    print(
        f"  coverage: {len(un_titles)} unique unanswerable titles, "
        f"{len(ctrl_titles)} unique control titles",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
