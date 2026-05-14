#!/usr/bin/env python3
"""build_pressure_band.py — emit Phase-1 ABSTAIN-Pressure items (#46 / epistemic bench).

Draws 15 items each from cleaned `unanswerable.jsonl` (post-#47) and
`false_premise.jsonl`, assigns each item one of 5 pressure patterns
(round-robin so each pattern gets 3 items per subset), wraps the question
with the assigned template, and writes:

  pressure-unanswerable.jsonl   (15 items, base_subset=unanswerable)
  pressure-false_premise.jsonl  (15 items, base_subset=false_premise)

Total = 30 items, 5 patterns × 3 items × 2 subsets.

Schema per row:
  id: <base_id>-pressure-<pattern>
  base_id: <orig SQuAD id or hand-curated id>
  base_subset: "unanswerable" | "false_premise"
  pressure_pattern: <one of 5>
  title: <copied>
  passage: <copied; possibly empty for false_premise>
  question: <pressure-wrapped>
  base_question: <unwrapped original>
  gt_answers: <copied>

Reproducibility: seed-pinned shuffle; running this script again on the same
corpus yields identical output. If the base corpus changes, re-run.
"""
import argparse
import json
import random
from pathlib import Path

from pressure_templates import PRESSURE_TEMPLATES, all_patterns, wrap_question

HERE = Path(__file__).parent
SEED = 13  # 2026-05-13 — first build of pressure band
N_PER_SUBSET = 15
N_PER_PATTERN = 3  # 5 patterns × 3 items × 2 subsets = 30


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def build_subset(base_path: Path, base_subset: str) -> list[dict]:
    items = load_jsonl(base_path)
    rng = random.Random(SEED + (1 if base_subset == "unanswerable" else 2))
    rng.shuffle(items)
    chosen = items[:N_PER_SUBSET]
    if len(chosen) < N_PER_SUBSET:
        raise SystemExit(
            f"base corpus {base_path} has only {len(chosen)} items; need {N_PER_SUBSET}"
        )

    # Round-robin assign patterns so each pattern gets exactly N_PER_PATTERN items.
    patterns = all_patterns()
    out: list[dict] = []
    for i, it in enumerate(chosen):
        pattern = patterns[i % len(patterns)]
        base_q = it["question"]
        wrapped_q = wrap_question(base_q, pattern)
        out.append({
            "id": f"{it['id']}-pressure-{pattern}",
            "base_id": it["id"],
            "base_subset": base_subset,
            "pressure_pattern": pattern,
            "title": it.get("title", ""),
            "passage": it.get("passage", ""),
            "question": wrapped_q,
            "base_question": base_q,
            "gt_answers": it.get("gt_answers", []),
        })

    # Sanity: each pattern appears exactly N_PER_PATTERN times.
    from collections import Counter
    counts = Counter(r["pressure_pattern"] for r in out)
    expected = {p: N_PER_PATTERN for p in patterns}
    if counts != Counter(expected):
        raise SystemExit(f"pattern distribution off: got {dict(counts)} expected {expected}")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--unanswerable-in", default=str(HERE / "unanswerable.jsonl"),
        help="cleaned base unanswerable subset (post-#47)"
    )
    p.add_argument(
        "--false-premise-in", default=str(HERE / "false_premise.jsonl"),
        help="hand-curated false_premise subset",
    )
    p.add_argument(
        "--out-prefix", default=str(HERE / "pressure-"),
        help="prefix for output files; written as <prefix>{unanswerable,false_premise}.jsonl",
    )
    args = p.parse_args()

    un = build_subset(Path(args.unanswerable_in), "unanswerable")
    fp = build_subset(Path(args.false_premise_in), "false_premise")

    un_path = Path(args.out_prefix + "unanswerable.jsonl")
    fp_path = Path(args.out_prefix + "false_premise.jsonl")
    write_jsonl(un_path, un)
    write_jsonl(fp_path, fp)

    print(f"wrote {len(un)} -> {un_path}")
    print(f"wrote {len(fp)} -> {fp_path}")
    print(f"patterns: {', '.join(all_patterns())}")


if __name__ == "__main__":
    main()
