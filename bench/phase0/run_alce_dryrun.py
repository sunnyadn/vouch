#!/usr/bin/env -S uv run --quiet --with google-genai
"""
run_alce_dryrun.py — Phase 0 ALCE-ELI5 dry-run runner.

Runs Vertex Gemini Pro (per the calibration verdict, the fidelity-reference
generator for our setup) on N stratified samples from ALCE-ELI5's 1000-instance
oracle-5-passage eval set. Two arms:

  without-vouch: ALCE default prompt + demos, generator only.
  with-vouch:    ALCE default prompt + demos + FBC discipline instruction,
                 plus post-hoc Flash-Lite verifier check on each cited
                 (passage, sentence-claim) pair. Citations whose backing
                 passage does not entail the cited sentence are stripped.

Output: bench/phase0/alce_dryrun_<arm>.json — list of records with
{question, answer, output, docs} as expected by ALCE's eval.py.

Usage:
  ./run_alce_dryrun.py --arm without-vouch --limit 3   # smoke test
  ./run_alce_dryrun.py --arm with-vouch --limit 100
  ./run_alce_dryrun.py --arm both --limit 100
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).parent
ALCE_DATA = Path("/tmp/alce/data/eli5_eval_bm25_top100_reranked_oracle.json")
ALCE_PROMPT = Path("/tmp/alce/prompts/eli5_default.json")

SEED = 20260510
GENERATOR_MODEL = "gemini-3.1-pro-preview"
VERIFIER_MODEL = "gemini-3.1-flash-lite"
# Gemini 3.x Pro is a thinking model — internal thoughts can consume hundreds
# of tokens before any visible output. Budget high enough to cover thoughts
# (typically 200-500) AND a long ELI5-style answer (~400 tokens).
MAX_NEW_TOKENS = 4096


# ---------- Prompt building ----------

def build_prompt(template: dict, question: str, docs: list[dict],
                 fbc_addendum: str = "") -> str:
    """Build the ALCE default prompt: instruction + demos + (Q + D + 'Answer:')."""
    instruction = template["instruction"]
    if fbc_addendum:
        instruction = instruction + " " + fbc_addendum
    demo_sep = template["demo_sep"]
    demo_prompt = template["demo_prompt"]
    doc_prompt = template["doc_prompt"]

    def render_docs(d_list):
        return "".join(
            doc_prompt.replace("{ID}", str(i + 1))
                      .replace("{T}", d.get("title", ""))
                      .replace("{P}", d.get("text", ""))
            for i, d in enumerate(d_list)
        )

    parts = []
    for demo in template.get("demos", []):
        parts.append(
            demo_prompt.replace("{INST}", instruction)
                       .replace("{Q}", demo["question"])
                       .replace("{D}", render_docs(demo["docs"]))
                       .replace("{A}", demo["answer"])
        )
    parts.append(
        demo_prompt.replace("{INST}", instruction)
                   .replace("{Q}", question)
                   .replace("{D}", render_docs(docs))
                   .replace("{A}", "")
    )
    return demo_sep.join(parts)


# ---------- Generator ----------

def make_generator(model_id: str):
    from google import genai
    from google.genai import types

    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
    if not project:
        raise RuntimeError("GOOGLE_CLOUD_PROJECT not set")
    client = genai.Client(vertexai=True, project=project, location=location)

    def generate(prompt: str) -> tuple[str, dict]:
        t0 = time.perf_counter()
        response = client.models.generate_content(
            model=model_id,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=MAX_NEW_TOKENS,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        dt = time.perf_counter() - t0
        text = response.text or ""
        usage = response.usage_metadata
        return text, {
            "latency_s": dt,
            "model": model_id,
            "input_tokens": usage.prompt_token_count if usage else None,
            "output_tokens": usage.candidates_token_count if usage else None,
            "thoughts_tokens": usage.thoughts_token_count if usage else None,
            "finish_reason": str(response.candidates[0].finish_reason) if response.candidates else None,
        }
    return generate


# ---------- Verifier (for with-vouch arm) ----------

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
- Reject "almost-true" claims that overstate, generalize, or paraphrase beyond what the source says.

Return your verdict as JSON: { "supported": true|false, "score": 0..1, "reason": "one sentence" }."""


def make_verifier(model_id: str):
    from google import genai
    from google.genai import types

    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
    client = genai.Client(vertexai=True, project=project, location=location)

    def verify(passage: str, claim: str) -> int:
        prompt = VERIFIER_PROMPT.replace("{{CLAIM}}", claim).replace("{{SOURCE}}", passage)
        try:
            response = client.models.generate_content(
                model=model_id,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                ),
            )
            obj = json.loads(response.text or "{}")
            return 1 if obj.get("supported") else 0
        except Exception as e:
            print(f"[verify] error: {e}", file=sys.stderr)
            return 1  # be lenient on verifier errors — don't strip on failure
    return verify


# ---------- Citation parsing + post-hoc strip ----------

CITATION_RE = re.compile(r"\[(\d+)\]")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def strip_unsupported_citations(answer: str, docs: list[dict], verify) -> tuple[str, dict]:
    """For each sentence with citations, run verifier on (cited_passage, sentence
    minus its [N] markers). If a citation's passage doesn't entail the sentence,
    drop that [N]. Returns (rewritten_answer, stats)."""
    sentences = SENTENCE_SPLIT_RE.split(answer)
    rewritten = []
    n_checked = 0
    n_stripped = 0
    for sent in sentences:
        cites = CITATION_RE.findall(sent)
        if not cites:
            rewritten.append(sent)
            continue
        bare_claim = CITATION_RE.sub("", sent).strip()
        if not bare_claim:
            rewritten.append(sent)
            continue
        kept = []
        for c in cites:
            n_checked += 1
            try:
                idx = int(c) - 1
            except ValueError:
                continue
            if idx < 0 or idx >= len(docs):
                continue
            passage = docs[idx].get("text", "")
            if verify(passage, bare_claim):
                kept.append(c)
            else:
                n_stripped += 1
        if kept:
            new_cites = "".join(f"[{c}]" for c in kept)
            stripped_sent = CITATION_RE.sub("", sent).rstrip()
            if stripped_sent and stripped_sent[-1] in ".!?":
                rewritten.append(f"{stripped_sent[:-1]} {new_cites}{stripped_sent[-1]}")
            else:
                rewritten.append(f"{stripped_sent} {new_cites}")
        else:
            stripped_sent = CITATION_RE.sub("", sent).rstrip()
            rewritten.append(stripped_sent)
    return " ".join(rewritten), {"n_checked": n_checked, "n_stripped": n_stripped}


# ---------- Main ----------

FBC_ADDENDUM = (
    "Be conservative with citations: only cite a passage if it explicitly "
    "supports your claim. When the passages don't support a sentence, prefer "
    "not citing over citing speculatively. Citations must be backed by the "
    "passage's literal content, not your prior knowledge."
)


def load_samples(limit: int | None) -> list[dict]:
    with ALCE_DATA.open() as f:
        data = json.load(f)
    import random
    rng = random.Random(SEED)
    rng.shuffle(data)
    if limit:
        data = data[:limit]
    return data


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["without-vouch", "with-vouch", "both"], default="without-vouch")
    ap.add_argument("--limit", type=int, default=100)
    args = ap.parse_args()

    samples = load_samples(args.limit)
    template = json.loads(ALCE_PROMPT.read_text())
    print(f"[harness] {len(samples)} samples, arm={args.arm}", file=sys.stderr)

    arms = ["without-vouch", "with-vouch"] if args.arm == "both" else [args.arm]

    generator = make_generator(GENERATOR_MODEL)
    verifier = make_verifier(VERIFIER_MODEL) if "with-vouch" in arms else None

    for arm in arms:
        out_path = THIS_DIR / f"alce_dryrun_{arm.replace('-', '_')}.json"
        addendum = FBC_ADDENDUM if arm == "with-vouch" else ""
        results = []
        total_gen_lat = 0.0
        total_strip_stats = {"n_checked": 0, "n_stripped": 0}

        for i, s in enumerate(samples):
            prompt = build_prompt(template, s["question"], s["docs"], fbc_addendum=addendum)
            try:
                output, meta = generator(prompt)
                total_gen_lat += meta["latency_s"]
            except Exception as e:
                print(f"[{arm}] sample {i}: generator error: {e}", file=sys.stderr)
                output = ""
                meta = {"latency_s": 0, "error": str(e)[:200]}

            strip_stats = None
            if arm == "with-vouch" and output and verifier:
                output, strip_stats = strip_unsupported_citations(output, s["docs"], verifier)
                total_strip_stats["n_checked"] += strip_stats["n_checked"]
                total_strip_stats["n_stripped"] += strip_stats["n_stripped"]

            results.append({
                "question": s["question"],
                "answer": s["answer"],
                "docs": s["docs"],
                "output": output,
                "claims": s.get("claims"),
                "_arm": arm,
                "_gen_meta": meta,
                "_strip_stats": strip_stats,
            })
            if (i + 1) % 5 == 0:
                print(f"[{arm}] {i + 1}/{len(samples)}", file=sys.stderr)

        with out_path.open("w") as f:
            json.dump({"data": results}, f, ensure_ascii=False)
        print(f"[{arm}] wrote {out_path}", file=sys.stderr)
        print(f"[{arm}] total gen latency: {total_gen_lat:.1f}s ({total_gen_lat/len(samples):.2f}s/sample)",
              file=sys.stderr)
        if arm == "with-vouch":
            print(f"[{arm}] strip stats: {total_strip_stats}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
