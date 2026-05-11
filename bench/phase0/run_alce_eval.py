#!/usr/bin/env -S uv run --quiet --with google-genai --with nltk --with rouge_score --with numpy --with tqdm --with accelerate --with bitsandbytes --with minicheck@git+https://github.com/Liyan06/MiniCheck.git@main
"""
run_alce_eval.py — Phase 1 lite scorer for the Phase 0 dry-run outputs.

Replicates ALCE's compute_autoais + compute_claims metrics (from
/tmp/alce/eval.py) using MiniCheck-Flan-T5-Large as the NLI engine instead
of ALCE's canonical TRUE-T5-XXL (11B, ~22 GB). Rationale:

- MiniCheck-T5 is MIT-licensed, locally runnable, ~30x smaller, and was
  validated on Phase 0's calibration (LLM-AggreFact 100-sample) at 0.775
  balanced accuracy vs Vertex Pro 0.815 — 4pp below within statistical noise.
- Different model family from Flash-Lite (which was the with-vouch arm's
  post-hoc verifier), avoiding self-judging bias.

Documented limitation: scores produced here are NOT the canonical ALCE
numbers from the paper. For launch headlines, run with TRUE-T5-XXL after
this lite path confirms direction and Phase 1 is unblocked.

Usage:
  ./run_alce_eval.py --input alce_dryrun_without_vouch.json
  ./run_alce_eval.py --input alce_dryrun_with_vouch.json
  ./run_alce_eval.py --both     # runs both arms + prints comparison table
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

THIS_DIR = Path(__file__).parent

# Citation regex copied verbatim-equivalent from ALCE's logic.
CITATION_RE = re.compile(r"\[(\d+)\]")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def remove_citations(sent: str) -> str:
    return CITATION_RE.sub("", sent).replace(" .", ".").replace(" ,", ",").strip()


def make_minicheck_nli():
    import nltk
    for resource in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)

    from minicheck.minicheck import MiniCheck
    cache_dir = os.environ.get("MINICHECK_CACHE_DIR", str(THIS_DIR / ".minicheck_cache"))
    print(f"[nli] loading MiniCheck-Flan-T5-Large", file=sys.stderr)
    scorer = MiniCheck(model_name="flan-t5-large", enable_prefix_caching=False, cache_dir=cache_dir)

    def nli(passage: str, claim: str) -> int:
        pred_label, _, _, _ = scorer.score(docs=[passage], claims=[claim])
        return int(pred_label[0])
    return nli


VERTEX_NLI_PROMPT = """You verify factual claims against source text.

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

Return your verdict as JSON: { "supported": true|false }."""


def make_true_t5_nli(quantize_8bit: bool = False):
    """ALCE paper's canonical evaluator: google/t5_xxl_true_nli_mixture.
    Loads via Hugging Face transformers. Uses bf16 + device_map="auto" by
    default (auto-splits GPU + CPU); pass quantize_8bit=True for bitsandbytes
    8-bit (~half the memory, slight precision tradeoff).

    The NLI prompt format mirrors ALCE eval.py exactly:
      input: "premise: <passage> hypothesis: <claim>"
      output: "1" (entailed) or "0" (not entailed)
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    model_name = "google/t5_xxl_true_nli_mixture"
    print(f"[nli] loading {model_name} (quantize_8bit={quantize_8bit})", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    kwargs = dict(device_map="auto")
    if quantize_8bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )
    else:
        kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **kwargs)
    model.eval()

    def nli(passage: str, claim: str) -> int:
        text = f"premise: {passage} hypothesis: {claim}"
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
        input_ids = inputs.input_ids.to(next(model.parameters()).device)
        with torch.inference_mode():
            out = model.generate(input_ids, max_new_tokens=10)
        result = tokenizer.decode(out[0], skip_special_tokens=True).strip()
        return 1 if result == "1" else 0
    return nli


def make_vertex_nli(model_id: str):
    from google import genai
    from google.genai import types
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
    if not project:
        raise RuntimeError("GOOGLE_CLOUD_PROJECT not set")
    client = genai.Client(vertexai=True, project=project, location=location)
    print(f"[nli] using Vertex {model_id}", file=sys.stderr)

    def nli(passage: str, claim: str) -> int:
        prompt = VERTEX_NLI_PROMPT.replace("{{CLAIM}}", claim).replace("{{SOURCE}}", passage)
        try:
            response = client.models.generate_content(
                model=model_id,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                    thinking_config=types.ThinkingConfig(thinking_budget=0) if "pro" in model_id else None,
                ),
            )
            obj = json.loads(response.text or "{}")
            return 1 if obj.get("supported") else 0
        except Exception as e:
            print(f"[nli] error: {str(e)[:120]}", file=sys.stderr)
            return 1
    return nli


def format_doc(doc: dict) -> str:
    return f"Title: {doc.get('title', '')}\n{doc.get('text', '')}"


def compute_autoais(data: list[dict], nli) -> dict:
    """Mirrors ALCE compute_autoais — citation precision / recall / F1.
    For each sentence:
      - Extract cited doc indices [N]
      - Build joint passage from cited docs
      - NLI(joint passage, sentence with [N] stripped) → support flag
      - Per-citation check: NLI(passage_i alone, sentence) → over-citation flag
    """
    sent_total = 0
    sent_supported = 0
    cite_total = 0
    cite_supported = 0
    answers = []

    for item in data:
        output = item["output"]
        docs = item["docs"]
        sentences = SENTENCE_SPLIT_RE.split(output)
        per_answer_supported = 0
        per_answer_total = 0

        for sent in sentences:
            cites = CITATION_RE.findall(sent)
            if not cites:
                continue
            bare = remove_citations(sent)
            if not bare:
                continue

            cited_idxs = []
            for c in cites:
                try:
                    i = int(c) - 1
                    if 0 <= i < len(docs):
                        cited_idxs.append(i)
                except ValueError:
                    pass
            if not cited_idxs:
                continue

            joint_passage = "\n".join(format_doc(docs[i]) for i in cited_idxs)
            sent_total += 1
            per_answer_total += 1
            joint_support = nli(joint_passage, bare)
            if joint_support:
                sent_supported += 1
                per_answer_supported += 1

            for i in cited_idxs:
                cite_total += 1
                single = format_doc(docs[i])
                if nli(single, bare):
                    cite_supported += 1

        answers.append((per_answer_supported, per_answer_total))

    citation_recall = sent_supported / sent_total if sent_total else 0.0
    citation_precision = cite_supported / cite_total if cite_total else 0.0
    f1 = (2 * citation_precision * citation_recall /
          (citation_precision + citation_recall)) if (citation_precision + citation_recall) else 0.0

    return {
        "citation_recall": citation_recall * 100,
        "citation_precision": citation_precision * 100,
        "citation_f1": f1 * 100,
        "n_sentences_with_cites": sent_total,
        "n_sentences_supported": sent_supported,
        "n_individual_cites": cite_total,
        "n_individual_supported": cite_supported,
    }


def compute_claims(data: list[dict], nli) -> dict:
    """Mirrors ALCE compute_claims for ELI5 — per-claim NLI on output."""
    scores = []
    n_claims_total = 0
    n_claims_entailed = 0
    for item in data:
        output_no_cites = remove_citations(item["output"])
        claims = item.get("claims") or []
        if not claims:
            continue
        entail = 0
        for claim in claims:
            n_claims_total += 1
            if nli(output_no_cites, claim):
                entail += 1
                n_claims_entailed += 1
        scores.append(entail / len(claims))
    return {
        "claim_nli": 100 * float(np.mean(scores)) if scores else 0.0,
        "n_claims_total": n_claims_total,
        "n_claims_entailed": n_claims_entailed,
        "n_answers_with_claims": len(scores),
    }


def score_arm(input_path: Path, nli, limit: int | None = None) -> dict:
    data = json.load(input_path.open())["data"]
    if limit:
        data = data[:limit]
    print(f"[scorer] {input_path.name}: {len(data)} samples", file=sys.stderr)

    cit = compute_autoais(data, nli)
    print(f"[scorer]   cite recall: {cit['citation_recall']:.1f}%, prec: {cit['citation_precision']:.1f}%, F1: {cit['citation_f1']:.1f}%",
          file=sys.stderr)

    clm = compute_claims(data, nli)
    print(f"[scorer]   claim_nli: {clm['claim_nli']:.1f}%", file=sys.stderr)

    return {
        "input": input_path.name,
        "n_samples": len(data),
        **cit,
        **clm,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None, help="Specific JSON to score")
    ap.add_argument("--both", action="store_true",
                    help="Score both alce_dryrun_without_vouch.json and alce_dryrun_with_vouch.json")
    ap.add_argument("--nli", choices=["minicheck", "vertex-pro", "vertex-flash",
                                       "true-t5-xxl", "true-t5-xxl-8bit"],
                    default="minicheck",
                    help="NLI engine: minicheck (default, local MIT), vertex-pro "
                         "(fidelity ref), vertex-flash (cheap), true-t5-xxl "
                         "(ALCE paper canonical, bf16), true-t5-xxl-8bit "
                         "(same but bitsandbytes 8bit for tighter VRAM)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Score only first N samples per arm (for canonical-NLI subset checks)")
    ap.add_argument("--out", default=None, help="Output JSON path (default: alce_eval_results_<nli>.json)")
    args = ap.parse_args()

    if args.nli == "minicheck":
        nli = make_minicheck_nli()
    elif args.nli == "vertex-pro":
        nli = make_vertex_nli("gemini-3.1-pro-preview")
    elif args.nli == "vertex-flash":
        nli = make_vertex_nli("gemini-3.1-flash-lite")
    elif args.nli == "true-t5-xxl":
        nli = make_true_t5_nli(quantize_8bit=False)
    elif args.nli == "true-t5-xxl-8bit":
        nli = make_true_t5_nli(quantize_8bit=True)
    else:
        ap.error(f"unknown --nli {args.nli}")

    out_path = Path(args.out) if args.out else THIS_DIR / f"alce_eval_results_{args.nli.replace('-', '_')}.json"

    results = []
    if args.both:
        for arm in ("without_vouch", "with_vouch"):
            p = THIS_DIR / f"alce_dryrun_{arm}.json"
            if not p.exists():
                print(f"[scorer] missing: {p}", file=sys.stderr)
                continue
            results.append(score_arm(p, nli, limit=args.limit))
    elif args.input:
        results.append(score_arm(Path(args.input), nli, limit=args.limit))
    else:
        ap.error("--input or --both required")

    if len(results) >= 2:
        print("\n=== Comparison ===", file=sys.stderr)
        cols = [
            ("citation_recall", "Cite Rec"),
            ("citation_precision", "Cite Prec"),
            ("citation_f1", "Cite F1"),
            ("claim_nli", "Claim NLI"),
        ]
        header = f"{'metric':<14}" + "".join(f" {label:>10}" for _, label in cols)
        print(header, file=sys.stderr)
        for r in results:
            row = f"{r['input'][:14]:<14}" + "".join(
                f" {r[k]:>9.1f}%" for k, _ in cols
            )
            print(row, file=sys.stderr)

    out_path.write_text(json.dumps({"nli": args.nli, "limit": args.limit, "results": results}, indent=2))
    print(f"\n[scorer] wrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
