#!/usr/bin/env -S uv run --quiet --with google-genai
"""audit_unanswerable.py — classify each unanswerable.jsonl item (#47).

For each item we ask an LLM judge: does the passage contain a clear answer to
the question? Classification produces audit-unanswerable.jsonl which is then
human-audited (eyeball the `passage-contains-answer` + `borderline` rows) and
fed into the corpus-cleanup step.

Categories (mutually exclusive, ordered by severity of mislabel):
  passage-contains-answer — passage literally / paraphrasably contains a
    specific answer to the question. SQuAD "unanswerable" label is wrong.
  bad-question — typo / ambiguous referent / malformed question stem.
  borderline — passage is topically adjacent and a careful reader might
    construct an inference; not a clean extractive answer.
  clean-unanswerable — passage genuinely does not contain a plausible answer
    and the question is well-formed.

Env: GOOGLE_CLOUD_PROJECT, GOOGLE_APPLICATION_CREDENTIALS, GOOGLE_CLOUD_LOCATION.

Usage:
  ./audit_unanswerable.py                            # full N=50 run
  ./audit_unanswerable.py --in unanswerable.jsonl --out audit-unanswerable.jsonl
  ./audit_unanswerable.py --calibrate                # first 5 only (sanity check)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
MODEL = os.environ.get("AUDIT_MODEL", "gemini-3.1-pro-preview")


PROMPT = """You audit a single SQuAD-2 item that is labeled "unanswerable." Read the passage and the question, then classify whether the SQuAD label is correct.

QUESTION: {question}
PASSAGE: {passage}

CATEGORIES (pick exactly ONE):

1. passage-contains-answer
   The passage literally or paraphrasably contains a specific answer to the question. A careful reader extracting from the passage would give a confident, specific answer. SQuAD's "unanswerable" label is wrong on this item.
   Example: Question "Who owns ABC?" with passage "The Walt Disney Company (which also owns ABC)..." → passage-contains-answer.

2. bad-question
   The question has a typo, ambiguous referent (e.g. "the two listed teams" with no list provided), or is grammatically malformed in a way that prevents a clean read. Independent of the passage's content.

3. borderline
   The passage is topically related and a careful reader could construct an inference, but the passage does NOT directly state an answer. (E.g. passage says X applies to "ancient eukaryotes" and question asks about "jawed vertebrates" — related topic, not directly answered.)

4. clean-unanswerable
   The passage genuinely does not contain a plausible answer to the well-formed question. The SQuAD label is correct.

DECISION RULES:
- If passage explicitly contains the answer → 1 (passage-contains-answer), regardless of how the question is worded.
- If the question itself is malformed (typo / ambiguous referent / no antecedent) → 2 (bad-question), even if the passage contains something useful.
- "Borderline" is reserved for genuinely-adjacent inferences, not for clear answers.
- When in doubt between 3 and 4, prefer 4 (clean-unanswerable) — borderline should be reserved for clear topical-adjacency cases.

Output JSON: {{"classification": "<one of the four exact strings>", "reasoning": "<one short sentence>", "extracted_answer_if_any": "<verbatim or paraphrase from passage, or empty string>", "confidence": 0.0-1.0}}
"""


def make_client():
    from google import genai
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
    if not project:
        raise RuntimeError("GOOGLE_CLOUD_PROJECT not set")
    return genai.Client(vertexai=True, project=project, location=location)


def classify_one(client, item: dict) -> dict:
    import re
    from google.genai import types
    prompt = PROMPT.format(question=item["question"], passage=item["passage"])
    t0 = time.perf_counter()
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=1024),
        )
        dt = time.perf_counter() - t0
        text = response.text or ""
    except Exception as e:
        return {"classification": "_ERROR", "reasoning": str(e)[:200], "extracted_answer_if_any": "", "confidence": 0.0, "latency_s": time.perf_counter() - t0}

    # Robust JSON extraction: strip ```json fences, find the outermost {…}.
    cleaned = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```\s*$", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()
    # Also handle the partial-fence case (no closing ```)
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"```\s*$", "", cleaned).strip()
    # Last-resort: greedy match for the outermost JSON object.
    if not cleaned.startswith("{"):
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if m:
            cleaned = m.group(0)

    try:
        v = json.loads(cleaned)
        v["latency_s"] = dt
        return v
    except json.JSONDecodeError:
        # best-effort fallback: look for category keyword
        cat = "_PARSE_ERROR"
        for c in ("passage-contains-answer", "bad-question", "borderline", "clean-unanswerable"):
            if c in text.lower():
                cat = c
                break
        return {"classification": cat, "reasoning": text[:300], "extracted_answer_if_any": "", "confidence": 0.0, "latency_s": dt}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", default=str(HERE / "unanswerable.jsonl"))
    p.add_argument("--out", default=str(HERE / "audit-unanswerable.jsonl"))
    p.add_argument("--calibrate", action="store_true", help="first 5 only (sanity check)")
    args = p.parse_args()

    items = [json.loads(l) for l in Path(args.inp).read_text().splitlines() if l.strip()]
    if args.calibrate:
        items = items[:5]

    client = make_client()
    print(f"[audit] model={MODEL} N={len(items)} -> {args.out}", file=sys.stderr)
    out_rows: list[dict] = []
    counts: dict[str, int] = {}
    for i, item in enumerate(items, 1):
        v = classify_one(client, item)
        cat = v.get("classification", "_ERROR")
        counts[cat] = counts.get(cat, 0) + 1
        row = {
            "id": item["id"],
            "title": item.get("title", ""),
            "question": item["question"],
            "passage_excerpt": item["passage"][:300] + ("…" if len(item["passage"]) > 300 else ""),
            "classification": cat,
            "reasoning": v.get("reasoning", ""),
            "extracted_answer_if_any": v.get("extracted_answer_if_any", ""),
            "confidence": v.get("confidence", 0.0),
            "audit_model": MODEL,
            "audit_latency_s": v.get("latency_s", 0.0),
        }
        out_rows.append(row)
        print(f"  [{i:3d}/{len(items)}] {item['id'][:8]} {cat:25s} conf={v.get('confidence', 0):.2f}", file=sys.stderr)

    Path(args.out).write_text("\n".join(json.dumps(r) for r in out_rows) + "\n")
    print(f"[audit] wrote {len(out_rows)} -> {args.out}", file=sys.stderr)
    print(f"[audit] counts: {counts}", file=sys.stderr)


if __name__ == "__main__":
    main()
