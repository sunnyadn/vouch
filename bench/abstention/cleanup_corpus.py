#!/usr/bin/env -S uv run --quiet --with datasets
"""cleanup_corpus.py — replace mislabeled items in unanswerable.jsonl (#47).

Reads audit-unanswerable.jsonl (output of audit_unanswerable.py), drops items
classified as anything other than `clean-unanswerable`, samples additional
SQuAD-2 unanswerable items (stratified by title, excluding all current and
previously-dropped IDs), and writes a fresh candidates file the audit script
re-checks.

Final cleaned corpus = (kept originals) + (audited-clean new samples).

Workflow:
  1. ./cleanup_corpus.py --sample-fresh           # write fresh-candidates.jsonl (N≈30)
  2. ./audit_unanswerable.py --in fresh-candidates.jsonl --out audit-fresh.jsonl
  3. ./cleanup_corpus.py --merge                  # write unanswerable-cleaned.jsonl

Then a separate ops step moves unanswerable.jsonl → unanswerable-pre47.jsonl
and unanswerable-cleaned.jsonl → unanswerable.jsonl, preserving the pre-#47
snapshot for reproducibility.
"""
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
SEED = 42

ORIG_PATH = HERE / "unanswerable.jsonl"
AUDIT_PATH = HERE / "audit-unanswerable.jsonl"
FRESH_PATH = HERE / "fresh-candidates.jsonl"
AUDIT_FRESH_PATH = HERE / "audit-fresh.jsonl"
CLEANED_PATH = HERE / "unanswerable-cleaned.jsonl"
TARGET_N = 50
FRESH_SAMPLE_N = 30


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def kept_ids_from_audit() -> set[str]:
    rows = load_jsonl(AUDIT_PATH)
    return {r["id"] for r in rows if r["classification"] == "clean-unanswerable"}


def all_seen_ids() -> set[str]:
    """All IDs that have ever appeared — exclude these when sampling fresh."""
    orig = load_jsonl(ORIG_PATH)
    seen = {it["id"] for it in orig}
    if AUDIT_FRESH_PATH.exists():
        for r in load_jsonl(AUDIT_FRESH_PATH):
            seen.add(r["id"])
    return seen


def sample_fresh():
    from datasets import load_dataset  # type: ignore

    exclude = all_seen_ids()
    print(f"loading rajpurkar/squad_v2 dev split (excluding {len(exclude)} seen ids)…", file=sys.stderr)
    ds = load_dataset("rajpurkar/squad_v2", split="validation")
    pool: list[dict] = []
    for ex in ds:
        if ex["id"] in exclude:
            continue
        if ex["answers"]["text"]:
            continue  # unanswerable only
        pool.append({
            "id": ex["id"],
            "title": ex["title"],
            "passage": ex["context"],
            "question": ex["question"],
            "gt_answers": [],
        })
    print(f"  fresh pool: {len(pool)} items", file=sys.stderr)

    # Stratify-by-title, same logic as sample_squadv2.py but on the fresh pool.
    rng = random.Random(SEED + 47)  # offset so we don't re-pick the same items
    by_title = defaultdict(list)
    for it in pool:
        by_title[it["title"]].append(it)
    for t in by_title:
        rng.shuffle(by_title[t])
    titles = sorted(by_title)
    rng.shuffle(titles)
    out: list[dict] = []
    idx = 0
    empty_pass = 0
    while len(out) < FRESH_SAMPLE_N and empty_pass < len(titles):
        t = titles[idx % len(titles)]
        if by_title[t]:
            out.append(by_title[t].pop())
            empty_pass = 0
        else:
            empty_pass += 1
        idx += 1

    FRESH_PATH.write_text("\n".join(json.dumps(it) for it in out) + "\n")
    print(f"  wrote {len(out)} -> {FRESH_PATH}", file=sys.stderr)


def merge():
    """Build cleaned corpus = kept originals + audited-clean fresh."""
    kept_ids = kept_ids_from_audit()
    orig = load_jsonl(ORIG_PATH)
    kept = [it for it in orig if it["id"] in kept_ids]

    if not AUDIT_FRESH_PATH.exists():
        print(f"missing: {AUDIT_FRESH_PATH}  (run audit on fresh-candidates first)", file=sys.stderr)
        sys.exit(2)

    fresh_audit = load_jsonl(AUDIT_FRESH_PATH)
    fresh_clean_ids = {r["id"] for r in fresh_audit if r["classification"] == "clean-unanswerable"}

    fresh = load_jsonl(FRESH_PATH)
    fresh_clean_items = [it for it in fresh if it["id"] in fresh_clean_ids]

    needed = TARGET_N - len(kept)
    fill = fresh_clean_items[:needed]
    if len(fill) < needed:
        print(f"WARNING: only {len(fill)} clean fresh items, need {needed} to fill {TARGET_N}", file=sys.stderr)

    cleaned = kept + fill
    CLEANED_PATH.write_text("\n".join(json.dumps(it) for it in cleaned) + "\n")
    print(f"  cleaned: {len(kept)} kept + {len(fill)} new = {len(cleaned)} -> {CLEANED_PATH}", file=sys.stderr)
    if len(cleaned) < TARGET_N:
        print(f"  short by {TARGET_N - len(cleaned)} — sample more fresh + re-audit", file=sys.stderr)


def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--sample-fresh", action="store_true", help="pull N=30 fresh SQuAD-2 unanswerable candidates")
    g.add_argument("--merge", action="store_true", help="merge kept originals + audited-clean fresh into unanswerable-cleaned.jsonl")
    args = p.parse_args()
    if args.sample_fresh:
        sample_fresh()
    elif args.merge:
        merge()


if __name__ == "__main__":
    main()
