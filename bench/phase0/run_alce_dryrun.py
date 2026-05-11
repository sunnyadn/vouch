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
DEFAULT_VERIFIER_MODEL = "gemini-3.1-flash-lite"
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
    print(f"[verifier] using {model_id}", file=sys.stderr)
    is_pro = "pro" in model_id

    def verify(passage: str, claim: str) -> int:
        prompt = VERIFIER_PROMPT.replace("{{CLAIM}}", claim).replace("{{SOURCE}}", passage)
        try:
            response = client.models.generate_content(
                model=model_id,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                    thinking_config=types.ThinkingConfig(thinking_budget=0) if is_pro else None,
                ),
            )
            obj = json.loads(response.text or "{}")
            return 1 if obj.get("supported") else 0
        except Exception as e:
            print(f"[verify] error: {str(e)[:120]}", file=sys.stderr)
            return 1  # be lenient on verifier errors — don't strip on failure
    return verify


# ---------- Citation parsing + post-hoc strip ----------

CITATION_RE = re.compile(r"\[(\d+)\]")
# Matches [N: <any content without brackets>] — robust to multi-quote, nested
# quotes, curly-quote variants. Inner-content quote extraction is handled
# downstream in extract_quotes_from_content().
QUOTE_CITATION_RE = re.compile(r'\[(\d+):\s*([^\[\]]+?)\s*\]', re.DOTALL)
QUOTE_CITATION_STRIP_RE = re.compile(r'\[\d+:\s*[^\[\]]+?\s*\]', re.DOTALL)
INNER_QUOTE_RE = re.compile(r'"([^"]+)"')
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
QUOTE_CHARS_RE = re.compile(r"[\"'‘’“”]")


def normalize_for_quote_match(s: str) -> str:
    """Strip quote chars (incl. curly variants), collapse whitespace, lowercase.
    Stripping quotes lets nested-quote formats match cleanly."""
    s = QUOTE_CHARS_RE.sub("", s)
    return re.sub(r"\s+", " ", s).strip().lower()


def extract_quotes_from_content(content: str) -> list[str]:
    """Pull quote candidates out of the inner content of [N: ...].
    Cases handled:
      - Single  `"the cited phrase"` → ["the cited phrase"]
      - Multi   `"phrase one", "phrase two"` → ["phrase one", "phrase two"]
      - Nested  `"outer "inner" outer"` → inner runs separately; normalize
                strips quote chars so verbatim-presence still works.
      - Unquoted fallback: raw content.
    """
    matches = INNER_QUOTE_RE.findall(content)
    if matches:
        return matches
    return [content.strip()]


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


def _local_sentence_context(answer: str, start: int, end: int) -> str:
    """Extract a sentence-like context around char span [start, end). Looks
    backward for sentence-ending punctuation (or buffer start) and forward
    for the next sentence-ending punctuation (or buffer end), then strips any
    quote-citation markers + bare [N] markers from the slice."""
    left = max(answer.rfind(s, 0, start) for s in (". ", "! ", "? ", "\n"))
    if left == -1:
        left = 0
    else:
        left += 2  # past the punctuation + space
    right_candidates = [answer.find(s, end) for s in (". ", "! ", "? ", "\n")]
    right_candidates = [r for r in right_candidates if r != -1]
    right = min(right_candidates) + 1 if right_candidates else len(answer)
    raw = answer[left:right]
    cleaned = QUOTE_CITATION_STRIP_RE.sub("", raw)
    cleaned = CITATION_RE.sub("", cleaned)
    return cleaned.strip()


def strip_unsupported_citations_quote_grounded(answer: str, docs: list[dict], verify) -> tuple[str, dict]:
    """Quote-grounded variant: generator emits [N: "verbatim quote"] inline.
    Single-pass over the full answer (sentence-splitting interacts badly with
    periods inside quotes). For each quote-citation match:
      1. Quote must appear (whitespace-insensitive) in passage N
      2. NLI(quote, surrounding sentence with markers stripped) must pass
    Failures replace the marker with empty string; passes replace with [N].
    Bare [N] cites without a quote are also stripped (format violation).
    """
    n_checked = 0
    n_stripped_quote_missing = 0
    n_stripped_nli = 0
    n_no_quote_at_all = 0

    # Pass 1 — collect all quote-citation matches with verdicts.
    replacements: list[tuple[int, int, str]] = []
    for m in QUOTE_CITATION_RE.finditer(answer):
        n_checked += 1
        cite_idx_str, raw_content = m.group(1), m.group(2)
        try:
            idx = int(cite_idx_str) - 1
        except ValueError:
            replacements.append((m.start(), m.end(), ""))
            continue
        if idx < 0 or idx >= len(docs):
            replacements.append((m.start(), m.end(), ""))
            continue
        passage = docs[idx].get("text", "")
        passage_norm = normalize_for_quote_match(passage)

        candidate_quotes = extract_quotes_from_content(raw_content)
        # Quote is verbatim-present if ANY candidate is found in passage.
        present_quotes = [q for q in candidate_quotes
                          if normalize_for_quote_match(q) in passage_norm]
        if not present_quotes:
            n_stripped_quote_missing += 1
            replacements.append((m.start(), m.end(), ""))
            continue

        sentence_ctx = _local_sentence_context(answer, m.start(), m.end())
        if not sentence_ctx:
            n_stripped_nli += 1
            replacements.append((m.start(), m.end(), ""))
            continue

        # NLI on joined candidates (treat them as combined evidence).
        joined_quote = " ".join(present_quotes)
        if not verify(joined_quote, sentence_ctx):
            n_stripped_nli += 1
            replacements.append((m.start(), m.end(), ""))
            continue

        # Keep — collapse to bare [N].
        replacements.append((m.start(), m.end(), f"[{cite_idx_str}]"))

    # Pass 2 — also strip any bare [N] cites (generator format violation).
    quote_spans = [(s, e) for s, e, _ in replacements]
    for m in CITATION_RE.finditer(answer):
        if any(s <= m.start() < e for s, e in quote_spans):
            continue  # this [N] is inside or already part of a quote-citation
        n_no_quote_at_all += 1
        replacements.append((m.start(), m.end(), ""))

    # Apply replacements right-to-left to preserve indices.
    replacements.sort(key=lambda r: -r[0])
    rewritten = answer
    for start, end, sub in replacements:
        rewritten = rewritten[:start] + sub + rewritten[end:]

    # Tidy up doubled spaces / orphan punctuation introduced by deletions.
    rewritten = re.sub(r"\s{2,}", " ", rewritten)
    rewritten = re.sub(r"\s+([,.!?;:])", r"\1", rewritten)

    return rewritten, {
        "n_checked": n_checked,
        "n_stripped_quote_missing": n_stripped_quote_missing,
        "n_stripped_nli": n_stripped_nli,
        "n_stripped": n_stripped_quote_missing + n_stripped_nli,
        "n_no_quote_at_all": n_no_quote_at_all,
    }


# ---------- Main ----------

FBC_ADDENDUM = (
    "Be conservative with citations: only cite a passage if it explicitly "
    "supports your claim. When the passages don't support a sentence, prefer "
    "not citing over citing speculatively. Citations must be backed by the "
    "passage's literal content, not your prior knowledge."
)

FBC_ADDENDUM_STRICT = (
    "STRICT citation rule: only cite [N] if the passage at index N contains "
    "a near-verbatim phrase supporting your claim. Paraphrase, implication, "
    "and 'they probably mean this' do NOT count. Many sentences should have "
    "ZERO citations — uncited honest observations beat faked citations. "
    "Citations will be post-verified and stripped if unsupported, so over-"
    "citing will hurt your score."
)

FBC_ADDENDUM_QUOTE_GROUNDED = (
    "STRICT citation rule with verbatim grounding: every cited [N] must be "
    "immediately followed by a verbatim quote from passage N in the form "
    "[N: \"<exact phrase from passage N>\"]. The quoted phrase must appear "
    "word-for-word in passage N (whitespace and minor punctuation tolerated). "
    "If you cannot find a verbatim phrase that supports your claim, do NOT "
    "cite. Many sentences should have ZERO citations. Example: "
    "'The cost varies by city [1: \"varies significantly depending on the "
    "specific city or country\"].' Each citation will be post-verified for "
    "(a) verbatim presence in passage N and (b) entailment of your sentence "
    "from the quote; failures will be stripped."
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
    ap.add_argument("--strip-verifier", choices=["flash-lite", "pro"], default="flash-lite",
                    help="Verifier model used for the post-hoc citation strip step")
    ap.add_argument("--strict-fbc", action="store_true",
                    help="Use stricter FBC system-prompt addendum (verbatim-only citations)")
    ap.add_argument("--quote-grounded", action="store_true",
                    help="Require generator to emit [N: 'verbatim quote'] inline; "
                         "verify quote-in-passage AND NLI(quote, sentence)")
    ap.add_argument("--out-suffix", default="",
                    help="Append suffix to output filenames to keep multiple variants")
    args = ap.parse_args()

    samples = load_samples(args.limit)
    template = json.loads(ALCE_PROMPT.read_text())
    print(f"[harness] {len(samples)} samples, arm={args.arm}, "
          f"strip-verifier={args.strip_verifier}, strict-fbc={args.strict_fbc}",
          file=sys.stderr)

    arms = ["without-vouch", "with-vouch"] if args.arm == "both" else [args.arm]

    generator = make_generator(GENERATOR_MODEL)
    strip_verifier_id = "gemini-3.1-pro-preview" if args.strip_verifier == "pro" else DEFAULT_VERIFIER_MODEL
    verifier = make_verifier(strip_verifier_id) if "with-vouch" in arms else None

    if args.quote_grounded:
        fbc_text = FBC_ADDENDUM_QUOTE_GROUNDED
        strip_fn = strip_unsupported_citations_quote_grounded
    elif args.strict_fbc:
        fbc_text = FBC_ADDENDUM_STRICT
        strip_fn = strip_unsupported_citations
    else:
        fbc_text = FBC_ADDENDUM
        strip_fn = strip_unsupported_citations

    for arm in arms:
        out_name = f"alce_dryrun_{arm.replace('-', '_')}{args.out_suffix}.json"
        out_path = THIS_DIR / out_name
        addendum = fbc_text if arm == "with-vouch" else ""
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
                output, strip_stats = strip_fn(output, s["docs"], verifier)
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
