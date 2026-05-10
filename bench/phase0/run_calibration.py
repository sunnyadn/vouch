#!/usr/bin/env -S uv run --quiet --with datasets --with google-genai --with accelerate --with minicheck@git+https://github.com/Liyan06/MiniCheck.git@main
"""
run_calibration.py — Phase 0 verifier-swap calibration harness.

Reads bench/phase0/calibration_sample.jsonl (100 stratified instances from
LLM-AggreFact). Runs each candidate verifier, computes balanced accuracy +
FN rate + FP rate + latency stats. Writes per-instance predictions to
calibration_predictions.jsonl and an aggregate Markdown table to
calibration_results.md.

The Vertex Gemini path mirrors vouch's production verifier prompt (see
src/verifier.ts:VERIFIER_PROMPT_TEMPLATE). The MiniCheck path uses the
official `minicheck` Python package, which is purpose-built for this
(doc, claim) -> {0, 1} task shape.

Verifiers:
  vertex-pro    Vertex AI Gemini 3.1 Pro (current vouch default)
  vertex-flash  Vertex AI Gemini Flash (cheaper tier)
  minicheck-t5  lytang/MiniCheck-Flan-T5-Large (MIT, local, ~0.8B)
  bespoke-7b    bespokelabs/Bespoke-MiniCheck-7B (CC BY-NC, ~7B, opt-in)

Usage:
  ./run_calibration.py                          # default: vertex-pro + vertex-flash + minicheck-t5
  ./run_calibration.py --include-bespoke        # add the NC-licensed SOTA reference
  ./run_calibration.py --verifier vertex-pro    # just one
  ./run_calibration.py --limit 10               # quick smoke test
"""

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).parent
SAMPLE_PATH = THIS_DIR / "calibration_sample.jsonl"
PRED_PATH = THIS_DIR / "calibration_predictions.jsonl"
RESULT_PATH = THIS_DIR / "calibration_results.md"

# Mirrors VERIFIER_PROMPT_TEMPLATE from src/verifier.ts as of 2026-05-10.
# Uses {{CLAIM}} / {{SOURCE}} placeholders + .replace() to avoid str.format()
# colliding with the literal { and } inside the JSON-format instruction.
VERIFIER_PROMPT = """You verify factual claims against source text.

CLAIM: "{{CLAIM}}"

SOURCE:
---
{{SOURCE}}
---

Question: Does the SOURCE genuinely support the CLAIM?

Rules:
- Be LENIENT on phrasing — same factual content with different words IS supported.
- Be STRICT on facts — every entity, number, dataset, baseline, or causal relationship in the claim must trace to the source.
- ABSENCE CLAIMS ("source does not mention X" / "X is missing from the paper"):
    supported = source genuinely lacks X.
    unsupported = source actually contains X (claim is wrong about absence).
- TABLE-LOOKUP CLAIMS ("system A scored Y% on dataset Z"):
    supported requires the cell at the right row × column.
    A correct number at the wrong position FAILS.
- Reject "almost-true" claims that overstate, generalize, or paraphrase beyond what the source says.

Return your verdict as JSON: { "supported": true|false, "score": 0..1, "reason": "one sentence" }."""


def load_samples(limit: int | None) -> list[dict]:
    rows = []
    with SAMPLE_PATH.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


# ---------- Vertex Gemini ----------

def make_vertex_predictor(model_id: str):
    from google import genai
    from google.genai import types

    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
    if not project:
        raise RuntimeError("GOOGLE_CLOUD_PROJECT not set in env")

    client = genai.Client(vertexai=True, project=project, location=location)

    def predict(doc: str, claim: str) -> tuple[int, dict]:
        prompt = VERIFIER_PROMPT.replace("{{CLAIM}}", claim).replace("{{SOURCE}}", doc)
        response = client.models.generate_content(
            model=model_id,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
            ),
        )
        text = response.text or ""
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            return 0, {"raw": text[:500], "parse_error": True}
        supported = bool(obj.get("supported"))
        return 1 if supported else 0, {
            "score": obj.get("score"),
            "reason": (obj.get("reason") or "")[:300],
        }
    return predict


# ---------- MiniCheck (local) ----------

def make_minicheck_predictor(model_name: str):
    import nltk
    for resource in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            print(f"[minicheck] downloading nltk resource: {resource}", file=sys.stderr)
            nltk.download(resource, quiet=True)

    from minicheck.minicheck import MiniCheck

    cache_dir = os.environ.get("MINICHECK_CACHE_DIR", str(THIS_DIR / ".minicheck_cache"))
    print(f"[minicheck] loading model_name={model_name} cache_dir={cache_dir}", file=sys.stderr)
    enable_prefix_caching = model_name == "Bespoke-MiniCheck-7B"
    scorer = MiniCheck(
        model_name=model_name,
        enable_prefix_caching=enable_prefix_caching,
        cache_dir=cache_dir,
    )

    def predict(doc: str, claim: str) -> tuple[int, dict]:
        pred_label, raw_prob, _, _ = scorer.score(docs=[doc], claims=[claim])
        return int(pred_label[0]), {"raw_prob": float(raw_prob[0])}
    return predict


# ---------- Harness core ----------

VERIFIERS = {
    "vertex-pro": dict(
        kind="vertex",
        model_id="gemini-3.1-pro-preview",
        license="Google ToS / paid",
    ),
    "vertex-flash": dict(
        kind="vertex",
        model_id="gemini-3.1-flash-lite",
        license="Google ToS / paid",
    ),
    "minicheck-t5": dict(
        kind="minicheck",
        model_id="flan-t5-large",
        license="MIT",
    ),
    "bespoke-7b": dict(
        kind="minicheck",
        model_id="Bespoke-MiniCheck-7B",
        license="CC BY-NC 4.0 (opt-in)",
    ),
}


def make_predictor(verifier_key: str):
    spec = VERIFIERS[verifier_key]
    if spec["kind"] == "vertex":
        return make_vertex_predictor(spec["model_id"])
    if spec["kind"] == "minicheck":
        return make_minicheck_predictor(spec["model_id"])
    raise ValueError(f"unknown verifier kind: {spec['kind']}")


def run_one_verifier(verifier_key: str, samples: list[dict]) -> list[dict]:
    print(f"\n=== {verifier_key} ===", file=sys.stderr)
    predict = make_predictor(verifier_key)

    # Warmup with first sample (timing dominated by cold start otherwise).
    print(f"[{verifier_key}] warming up", file=sys.stderr)
    try:
        predict(samples[0]["doc"], samples[0]["claim"])
    except Exception as e:
        print(f"[{verifier_key}] WARMUP FAILED: {e}", file=sys.stderr)
        raise

    results = []
    for i, s in enumerate(samples):
        t0 = time.perf_counter()
        try:
            pred, meta = predict(s["doc"], s["claim"])
            err = None
        except Exception as e:
            pred = -1
            meta = {"error": str(e)[:200]}
            err = str(e)[:200]
        dt = time.perf_counter() - t0
        results.append({
            "verifier": verifier_key,
            "sample_index": i,
            "dataset": s["dataset"],
            "gold_label": s["label"],
            "pred_label": pred,
            "latency_s": dt,
            "meta": meta,
            "error": err,
        })
        if (i + 1) % 10 == 0:
            print(f"[{verifier_key}] {i + 1}/{len(samples)}", file=sys.stderr)
    return results


def metrics(results: list[dict]) -> dict:
    valid = [r for r in results if r["pred_label"] in (0, 1)]
    if not valid:
        return {"n_valid": 0, "n_total": len(results)}

    tp = sum(1 for r in valid if r["gold_label"] == 1 and r["pred_label"] == 1)
    fn = sum(1 for r in valid if r["gold_label"] == 1 and r["pred_label"] == 0)
    tn = sum(1 for r in valid if r["gold_label"] == 0 and r["pred_label"] == 0)
    fp = sum(1 for r in valid if r["gold_label"] == 0 and r["pred_label"] == 1)

    n_pos = tp + fn
    n_neg = tn + fp
    sensitivity = tp / n_pos if n_pos else 0.0
    specificity = tn / n_neg if n_neg else 0.0
    bal_acc = (sensitivity + specificity) / 2

    latencies = sorted(r["latency_s"] for r in valid)
    return {
        "n_valid": len(valid),
        "n_total": len(results),
        "tp": tp, "fn": fn, "tn": tn, "fp": fp,
        "n_pos": n_pos, "n_neg": n_neg,
        "balanced_accuracy": bal_acc,
        "fn_rate": fn / n_pos if n_pos else 0.0,
        "fp_rate": fp / n_neg if n_neg else 0.0,
        "lat_median_s": statistics.median(latencies),
        "lat_p95_s": latencies[max(0, int(len(latencies) * 0.95) - 1)] if latencies else 0,
        "lat_total_s": sum(r["latency_s"] for r in valid),
    }


def write_aggregate_md(per_verifier: dict[str, dict]) -> None:
    lines = [
        "# Phase 0 verifier-swap calibration results",
        "",
        f"Sample: 100 stratified instances from LLM-AggreFact test split (seed=20260510).",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}.",
        "",
        "| Verifier | License | Bal. Acc. | FN rate | FP rate | Median lat (s) | Total wall (s) | Valid / Total |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for v_key, m in per_verifier.items():
        spec = VERIFIERS[v_key]
        if m.get("n_valid", 0) == 0:
            lines.append(
                f"| {v_key} | {spec['license']} | n/a | n/a | n/a | n/a | n/a | 0 / {m.get('n_total', 0)} |"
            )
            continue
        lines.append(
            f"| {v_key} | {spec['license']} | {m['balanced_accuracy']:.3f} | "
            f"{m['fn_rate']:.3f} | {m['fp_rate']:.3f} | "
            f"{m['lat_median_s']:.2f} | {m['lat_total_s']:.1f} | "
            f"{m['n_valid']} / {m['n_total']} |"
        )

    lines += [
        "",
        "## Verifier targets",
        "",
        "Pick the cheapest verifier whose balanced accuracy is within 3pp of vertex-pro.",
        "If minicheck-t5 (MIT, local, ~$0 marginal) clears that bar, use it for the ALCE benchmark and the $500 budget gate becomes irrelevant for the verifier side.",
    ]
    RESULT_PATH.write_text("\n".join(lines) + "\n")
    print(f"\n[aggregate] wrote {RESULT_PATH}", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verifier", action="append", default=None,
                    help="Specific verifier(s) to run (default: vertex-pro,vertex-flash,minicheck-t5)")
    ap.add_argument("--include-bespoke", action="store_true",
                    help="Include bespoke-7b (CC BY-NC, ~14 GB download)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Run on first N samples only (smoke test)")
    args = ap.parse_args()

    if args.verifier:
        verifier_keys = [v.strip() for v in ",".join(args.verifier).split(",") if v.strip()]
    else:
        verifier_keys = ["vertex-pro", "vertex-flash", "minicheck-t5"]
        if args.include_bespoke:
            verifier_keys.append("bespoke-7b")

    for v in verifier_keys:
        if v not in VERIFIERS:
            print(f"Unknown verifier: {v}. Choices: {list(VERIFIERS)}", file=sys.stderr)
            return 2

    samples = load_samples(args.limit)
    print(f"[harness] loaded {len(samples)} samples", file=sys.stderr)
    print(f"[harness] verifiers: {verifier_keys}", file=sys.stderr)

    pred_fp = PRED_PATH.open("w")
    per_verifier_metrics: dict[str, dict] = {}
    for v_key in verifier_keys:
        try:
            results = run_one_verifier(v_key, samples)
        except Exception as e:
            print(f"[{v_key}] FATAL: {e}", file=sys.stderr)
            per_verifier_metrics[v_key] = {"n_valid": 0, "n_total": len(samples), "error": str(e)[:200]}
            continue
        for r in results:
            pred_fp.write(json.dumps(r, ensure_ascii=False) + "\n")
        per_verifier_metrics[v_key] = metrics(results)
        m = per_verifier_metrics[v_key]
        if m.get("n_valid"):
            print(f"[{v_key}] bal_acc={m['balanced_accuracy']:.3f} "
                  f"fn={m['fn_rate']:.3f} fp={m['fp_rate']:.3f} "
                  f"median_lat={m['lat_median_s']:.2f}s", file=sys.stderr)
    pred_fp.close()
    print(f"\n[harness] wrote {PRED_PATH}", file=sys.stderr)

    write_aggregate_md(per_verifier_metrics)
    return 0


if __name__ == "__main__":
    sys.exit(main())
