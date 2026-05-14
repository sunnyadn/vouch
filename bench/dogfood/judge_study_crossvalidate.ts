#!/usr/bin/env bun
/**
 * judge_study_crossvalidate.ts — blind cross-validation of the N=50 hand-
 *   classification reported in p-alpha-judge-study-2026-05-14.md.
 *
 * For each row in p-alpha-judge-study-sample.tsv, send the (proposition,
 * KB claim, entity_class, similarity, strict/loose/broad verdicts) tuple
 * to the same VERIFIER_MODEL client production NLI uses, asking which of
 * the three classes (a/b/c) the pair belongs to. Claude's class_label
 * (which lives only in the markdown writeup, not in the TSV) is reconstructed
 * separately for comparison — the judge never sees it.
 *
 * Classes (verbatim from the original report's § Methodology):
 *   (a) prompt-strictness loss — actually a contradiction strict NLI missed
 *   (b) task-definition loss — refinement / value-override / partial-
 *       contradiction; broader "useful disagreement" would catch it
 *   (c) correctly-rejected — adjacent noise, current design is right
 *
 * Output: one line per row in p-alpha-judge-study-crossvalidate.jsonl
 *   { row_id, judge_class, judge_reason }
 *
 * The post-hoc comparison (confusion matrix, kappa, disagreement diagnoses)
 * lives in the companion .md writeup, not in this script.
 */
import { readFileSync, existsSync, appendFileSync, unlinkSync } from "node:fs";
import { join } from "node:path";
import { generateObject } from "ai";
import { z } from "zod";

import { getLanguageModel } from "../../src/providers.ts";
import { classifyError } from "../../src/verifier.ts";
import { VERIFIER_MODEL } from "../../src/config.ts";

const TSV = join(import.meta.dir, "p-alpha-judge-study-sample.tsv");
const OUT = join(import.meta.dir, "p-alpha-judge-study-crossvalidate.jsonl");

interface Args {
  concurrency: number;
}
function parseArgs(): Args {
  const out: Args = { concurrency: 4 };
  const args = process.argv.slice(2);
  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (a === "--concurrency") out.concurrency = Number(args[++i]);
    else if (a === "-h" || a === "--help") {
      console.error("Usage: judge_study_crossvalidate.ts [--concurrency N]");
      process.exit(0);
    }
  }
  return out;
}

interface Row {
  idx: number;
  entity_class: string;
  similarity: string;
  strict_fires: string;
  strict_score: string;
  loose_fires: string;
  loose_score: string;
  broad_fires: string;
  broad_score: string;
  proposition: string;
  claim_text: string;
  broad_reason: string;
}

function loadTsv(path: string): Row[] {
  const lines = readFileSync(path, "utf8").split("\n").filter((l) => l.length > 0);
  const header = lines[0]!.split("\t");
  const colIdx = (name: string): number => {
    const i = header.indexOf(name);
    if (i < 0) throw new Error(`column not found: ${name}`);
    return i;
  };
  const idx_idx = colIdx("idx");
  const idx_ec = colIdx("entity_class");
  const idx_sim = colIdx("similarity");
  const idx_sf = colIdx("strict_fires");
  const idx_ss = colIdx("strict_score");
  const idx_lf = colIdx("loose_fires");
  const idx_ls = colIdx("loose_score");
  const idx_bf = colIdx("broad_fires");
  const idx_bs = colIdx("broad_score");
  const idx_prop = colIdx("proposition");
  const idx_claim = colIdx("claim_text");
  const idx_br = colIdx("broad_reason");

  const rows: Row[] = [];
  for (let i = 1; i < lines.length; i++) {
    const cells = lines[i]!.split("\t");
    rows.push({
      idx: Number(cells[idx_idx]),
      entity_class: cells[idx_ec]!,
      similarity: cells[idx_sim]!,
      strict_fires: cells[idx_sf]!,
      strict_score: cells[idx_ss]!,
      loose_fires: cells[idx_lf]!,
      loose_score: cells[idx_ls]!,
      broad_fires: cells[idx_bf]!,
      broad_score: cells[idx_bs]!,
      proposition: cells[idx_prop]!,
      claim_text: cells[idx_claim]!,
      broad_reason: cells[idx_br]!,
    });
  }
  return rows;
}

// ---------------------------------------------------------------------------
// Classifier prompt
// ---------------------------------------------------------------------------
// Classes copied verbatim from p-alpha-judge-study-2026-05-14.md § Methodology
// (and the report's class-bullet definitions at the top).
//
// What the judge sees vs. what the original hand-classifier saw: identical
// modulo Claude's own labels. The TSV has broad.reason but not strict.reason
// or loose.reason, so the judge gets broad.reason only — matching the hand
// classifier's information set.
const CLASSIFY_PROMPT_TEMPLATE = `You are classifying a (proposition A, KB claim B) pair from a P-α counter-evidence judge study.

Three NLI prompt variants were run on the pair:
  - STRICT: production logical-mutual-exclusion contract with "be CONSERVATIVE".
  - LOOSE:  same contract, conservative clause removed.
  - BROAD:  "useful disagreement" contract — fires on contradiction OR refinement / value-override / partial-contradiction that an honest writer should reconcile.

You see each variant's fire decision and score, plus the BROAD variant's stated reason.

TASK — classify the pair into EXACTLY ONE of three classes:

(a) prompt-strictness loss — A and B ARE actually mutually exclusive (a real logical contradiction), but the strict NLI prompt missed it. The fix would be to tune the strict prompt (e.g. tighten the "approximate values" carve-out). A genuine logical contradiction wearing strict's clothes.

(b) task-definition loss — A and B are NOT a logical contradiction, but B refines, value-overrides, narrows, or partially contradicts A in a way an honest writer should reconcile. Strict's mutual-exclusion contract is narrower than "usefully disagree"; only a contract change (not a prompt change) would catch it. Typical shape: A is an under-qualified numerical claim and B asserts the specific value under a specified condition.

(c) correctly-rejected — adjacent noise. A and B share the entity but make claims on different predicates / aspects; both can be simultaneously true with no reconciliation needed. Strict NLI is right to reject. If BROAD fired on this pair, BROAD is over-firing — still (c).

DECISION RULES:
- If A and B together produce a logical contradiction → (a).
- Else if B contradicts, refines, value-overrides, narrows, or partially negates a specific factual assertion in A such that A as published would be misleading → (b).
- Else (predicates simply don't overlap, OR overlap but trivially compatible) → (c).

CONTEXT:
  entity_class:   {ENTITY_CLASS}
  cosine sim:     {SIMILARITY}

  STRICT verdict: fires={STRICT_FIRES}, score={STRICT_SCORE}
  LOOSE  verdict: fires={LOOSE_FIRES}, score={LOOSE_SCORE}
  BROAD  verdict: fires={BROAD_FIRES}, score={BROAD_SCORE}
  BROAD  reason:  "{BROAD_REASON}"

PROPOSITION A (agent draft):
  "{PROP_A}"

KB CLAIM B (attested):
  "{PROP_B}"

Return JSON: { class: "a" | "b" | "c", reason: one short sentence }.`;

const JudgeSchema = z.object({
  class: z.enum(["a", "b", "c"]),
  reason: z.string().max(500),
});

interface JudgeResult {
  row_id: number;
  judge_class: "a" | "b" | "c" | "error";
  judge_reason: string;
  error?: string;
}

async function judgeRow(row: Row): Promise<JudgeResult> {
  const prompt = CLASSIFY_PROMPT_TEMPLATE
    .replace("{ENTITY_CLASS}", row.entity_class)
    .replace("{SIMILARITY}", row.similarity)
    .replace("{STRICT_FIRES}", row.strict_fires)
    .replace("{STRICT_SCORE}", row.strict_score)
    .replace("{LOOSE_FIRES}", row.loose_fires)
    .replace("{LOOSE_SCORE}", row.loose_score)
    .replace("{BROAD_FIRES}", row.broad_fires)
    .replace("{BROAD_SCORE}", row.broad_score)
    .replace("{BROAD_REASON}", row.broad_reason.replace(/"/g, "'"))
    .replace("{PROP_A}", row.proposition.replace(/"/g, "'"))
    .replace("{PROP_B}", row.claim_text.replace(/"/g, "'"));

  try {
    const { object } = await generateObject({
      model: getLanguageModel(VERIFIER_MODEL),
      schema: JudgeSchema,
      prompt,
      temperature: 0.0,
    });
    return {
      row_id: row.idx,
      judge_class: object.class,
      judge_reason: object.reason,
    };
  } catch (e: any) {
    const transient = classifyError(e);
    const kind = transient ? transient.kind : "unknown";
    return {
      row_id: row.idx,
      judge_class: "error",
      judge_reason: `judge error (${kind}): ${e?.message || String(e)}`.slice(0, 300),
      error: kind,
    };
  }
}

async function runWithConcurrency<T, R>(
  items: T[],
  concurrency: number,
  fn: (item: T, idx: number) => Promise<R>,
): Promise<R[]> {
  const results: R[] = new Array(items.length);
  let next = 0;
  const workers = Array.from({ length: Math.max(1, concurrency) }, async () => {
    while (true) {
      const idx = next++;
      if (idx >= items.length) return;
      results[idx] = await fn(items[idx]!, idx);
    }
  });
  await Promise.all(workers);
  return results;
}

async function main() {
  const args = parseArgs();
  if (!existsSync(TSV)) {
    console.error(`missing: ${TSV}`);
    console.error(`run ./bench/dogfood/judge_study_sample.ts first`);
    process.exit(1);
  }
  const rows = loadTsv(TSV);
  console.error(`loaded ${rows.length} rows from ${TSV}`);
  console.error(`model:       ${VERIFIER_MODEL}`);
  console.error(`concurrency: ${args.concurrency}`);
  console.error(`output:      ${OUT}`);
  console.error("");

  if (existsSync(OUT)) unlinkSync(OUT);

  const t0 = Date.now();
  let done = 0;
  await runWithConcurrency(rows, args.concurrency, async (row) => {
    const result = await judgeRow(row);
    appendFileSync(OUT, JSON.stringify(result) + "\n");
    done++;
    if (done % 10 === 0 || done === rows.length) {
      const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
      console.error(`  ${done}/${rows.length} done (${elapsed}s)`);
    }
    return result;
  });

  console.error("");
  console.error(`total wall: ${((Date.now() - t0) / 1000).toFixed(1)}s`);
  console.error(`output written to ${OUT}`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
