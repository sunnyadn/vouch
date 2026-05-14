#!/usr/bin/env bun
/**
 * judge_study_P_alpha.ts — judge study for the P-α (counter-evidence) funnel.
 *
 * Question (issue #51 follow-up):
 *   The P-α replay funnel went 430 propositions → 195 entity-share+cosine
 *   candidates → 3 NLI contradictions. Where do the 192 non-flipped pairs go?
 *   Specifically: are they (a) actually-contradictions the strict NLI prompt
 *   missed (prompt-tunable), (b) partial-contradictions / refinements
 *   (P-α's strict-mutual-exclusion contract is narrower than usefully-disagree,
 *   contract-tunable), or (c) genuine token-coincidence (correctly rejected)?
 *
 * Method:
 *   For each fired proposition in fires-last14d.jsonl, replay the same
 *   entity-share + cosine ≥ 0.55 funnel as production. For every (proposition,
 *   candidate-claim) pair that survives, run THREE NLI prompt variants:
 *     - V_STRICT: production prompt (CONTRADICTION_PROMPT_TEMPLATE from
 *       src/verifier.ts). Conservative; only fires on logical mutual exclusion.
 *     - V_LOOSE:  same body, "Be CONSERVATIVE..." sentence removed.
 *     - V_BROAD:  contract change — fires on refinement / partial-contradiction
 *       / value-override, not just logical negation.
 *
 *   Each variant returns { fires: boolean, score: 0..1, reason }. Output is
 *   one row per (prop, candidate) pair with all three verdicts side-by-side.
 *
 *   The Claude-judge cross-check (4th variant) is intentionally NOT
 *   implemented — requires ANTHROPIC_API_KEY which isn't configured. The brief
 *   listed it as optional / non-blocking.
 *
 * Output: fires-judge-study-P_alpha.jsonl (gitignored — same privacy rules
 *   as fires-*.jsonl).
 *
 * Usage:
 *   ./judge_study_P_alpha.ts                        # full corpus
 *   ./judge_study_P_alpha.ts --limit 10             # smoke test
 *   ./judge_study_P_alpha.ts --concurrency 4
 *
 *   Pair-level concurrency is bounded; NLI calls within a pair run in
 *   parallel (3 variants × 1 round-trip each).
 */
import { readFileSync, existsSync, appendFileSync, unlinkSync } from "node:fs";
import { join } from "node:path";
import { generateObject } from "ai";
import { z } from "zod";

import * as store from "../../src/store.ts";
import { embedOne } from "../../src/embedder.ts";
import { getLanguageModel } from "../../src/providers.ts";
import { classifyError } from "../../src/verifier.ts";
import { VERIFIER_MODEL } from "../../src/config.ts";

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------
interface Args {
  in: string;
  out: string;
  concurrency: number;
  limit?: number;
}
function parseArgs(): Args {
  const args = process.argv.slice(2);
  const out: Args = {
    in: join(import.meta.dir, "fires-last14d.jsonl"),
    out: join(import.meta.dir, "fires-judge-study-P_alpha.jsonl"),
    concurrency: Number(process.env.DOGFOOD_REPLAY_CONCURRENCY || "6"),
  };
  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (a === "--in") out.in = args[++i]!;
    else if (a === "--out") out.out = args[++i]!;
    else if (a === "--concurrency") out.concurrency = Number(args[++i]);
    else if (a === "--limit") out.limit = Number(args[++i]);
    else if (a === "-h" || a === "--help") {
      console.error(
        "Usage: judge_study_P_alpha.ts [--in PATH] [--out PATH] [--concurrency N] [--limit N]",
      );
      process.exit(0);
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Funnel constants — identical to production P-α (src/gate.ts).
// ---------------------------------------------------------------------------
const COUNTER_EVIDENCE_TOPK = 5;
const COUNTER_EVIDENCE_MIN_COS = 0.55;
const COUNTER_EVIDENCE_FIRE_SCORE = 0.75;

/** Mirrors src/gate.ts sharesPrimaryEntity. */
function sharesPrimaryEntity(entity: string, text: string): boolean {
  const ent = entity.toLowerCase().trim();
  if (ent && text.toLowerCase().includes(ent)) return true;
  const entAlnum = entity.replace(/[^a-zA-Z0-9]/g, "").toLowerCase();
  if (entAlnum.length < 3) return false;
  return text.replace(/[^a-zA-Z0-9]/g, "").toLowerCase().includes(entAlnum);
}

const DATASET_NAMES = /\b(FEVER|ALCE|FActScore|SQuAD|TriviaQA|MMLU|HELM|GSM8K|HumanEval|MS\s?MARCO|NQ|HotpotQA|BoolQ)\b/i;
const VERSION_RE = /\b(v?\d+\.\d+(\.\d+)?|\d{4}-\d{2}-\d{2})\b/;
const PRICING_RE = /[\$£€¥]|\bUSD\b|\bprice\b|\bcost\b|\b\/\s*1M\b|\bper\s+(token|request|month)\b/i;
const WORKSPACE_RE = /^(SUN-\d+|#\d+|vouch|comprisk|claude\s*code|cc-router|meta)\b|^[A-Z]{2,}-\d+$/i;
const NAMED_PRODUCT_RE = /\b(GPT-?\d|Claude|Gemini|TreeSHAP|sksurv|shap|randomForestSRC|PyTorch|TensorFlow|HuggingFace|Vertex|OpenAI|Anthropic|Mistral)\b/i;

function classifyEntity(entity: string, proposition: string): string {
  const ent = entity.trim();
  const combined = `${ent} ${proposition}`;
  if (WORKSPACE_RE.test(ent)) return "workspace-meta";
  if (DATASET_NAMES.test(ent) || DATASET_NAMES.test(combined)) return "dataset";
  if (PRICING_RE.test(combined)) return "pricing";
  if (VERSION_RE.test(ent) || VERSION_RE.test(combined)) return "version";
  if (NAMED_PRODUCT_RE.test(ent) || NAMED_PRODUCT_RE.test(combined)) return "named-product";
  if (/^[a-z][a-z0-9_-]{1,30}$/.test(ent)) return "library-or-concept";
  return "other";
}

// ---------------------------------------------------------------------------
// Prompt variants
// ---------------------------------------------------------------------------
const NliSchema = z.object({
  fires: z.boolean(),
  score: z.number().min(0).max(1),
  reason: z.string().max(500),
});

// V_STRICT: verbatim copy of production CONTRADICTION_PROMPT_TEMPLATE from
// src/verifier.ts (with `contradicts` renamed `fires` to share a schema across
// variants — semantic content unchanged).
const PROMPT_STRICT = `You are checking whether two propositions are mutually exclusive.

PROPOSITION A: "{PROP_A}"
PROPOSITION B: "{PROP_B}"

Question: Are these two propositions mutually exclusive — can a faithful reader of both conclude that one denies the other?

CONTRADICTS = true iff there is at least one factual assertion in one proposition whose **negation** is asserted in the other (e.g. A says "X treats Y as Z", B says "X does NOT treat Y as Z" / "X treats Y as W ≠ Z"). Both propositions being simultaneously true must be **logically impossible** given their literal content.

CONTRADICTS = false in all of the following:
- Different topics, different entities, or different scopes (one is silent where the other speaks).
- One is a subset / superset of the other (partial coverage, same content).
- Same factual content with different wording, paraphrase, or precision (e.g. "≈ 0.86" vs "0.862" → not contradiction unless one explicitly excludes the other's value).
- An update or refinement that adds detail without denying prior content.
- Saying nothing about a fact is NOT the same as denying it.

Be CONSERVATIVE: when in doubt, answer false. We only want clear, logical mutual exclusion.

Return JSON: { fires: boolean, score: 0..1 confidence, reason: one short sentence }.`;

// V_LOOSE: strict body, but the "Be CONSERVATIVE..." sentence is removed.
// Same logical-exclusion contract; just less calibration pressure toward false.
const PROMPT_LOOSE = `You are checking whether two propositions are mutually exclusive.

PROPOSITION A: "{PROP_A}"
PROPOSITION B: "{PROP_B}"

Question: Are these two propositions mutually exclusive — can a faithful reader of both conclude that one denies the other?

CONTRADICTS = true iff there is at least one factual assertion in one proposition whose **negation** is asserted in the other (e.g. A says "X treats Y as Z", B says "X does NOT treat Y as Z" / "X treats Y as W ≠ Z"). Both propositions being simultaneously true must be **logically impossible** given their literal content.

CONTRADICTS = false in all of the following:
- Different topics, different entities, or different scopes (one is silent where the other speaks).
- One is a subset / superset of the other (partial coverage, same content).
- Same factual content with different wording, paraphrase, or precision (e.g. "≈ 0.86" vs "0.862" → not contradiction unless one explicitly excludes the other's value).
- An update or refinement that adds detail without denying prior content.
- Saying nothing about a fact is NOT the same as denying it.

Return JSON: { fires: boolean, score: 0..1 confidence, reason: one short sentence }.`;

// V_BROAD: contract change — also fires on refinement / partial-contradiction
// / value-override. This is the "usefully disagree" contract, broader than
// strict mutual exclusion.
const PROMPT_BROAD = `You are checking whether two propositions usefully disagree.

PROPOSITION A: "{PROP_A}"
PROPOSITION B: "{PROP_B}"

Question: Does PROPOSITION B contradict, refine, override, or qualify a fact stated in PROPOSITION A in a way that an honest writer should reconcile before publishing A as-is?

FIRES = true in any of these cases:
- A and B make incompatible factual assertions on the same entity (strict contradiction).
- A asserts a value or property that B refines, narrows, qualifies, or overrides (e.g. A says "X scored Y on benchmark Z"; B says "X scored Y on Z under condition C, but Y' under condition C' " — A's unqualified value is now misleading).
- A makes a vague or under-specified claim where B's specific value or distinction changes which reading is right (e.g. A says "results aren't out"; B reports specific results on a related batch — the unqualified A is now misleading).
- A states a partial fact that B materially completes such that A in isolation would mis-frame the topic.

FIRES = false in all of:
- Different topics, different entities, different scopes (one is silent where the other speaks, on genuinely unrelated content).
- B simply adds adjacent unrelated detail without changing A's intended meaning.
- Token-level entity overlap with no shared predicate or value.
- A and B both true and compatible with no qualification needed.

Return JSON: { fires: boolean, score: 0..1 confidence, reason: one short sentence }.`;

interface NliResult {
  fires: boolean;
  score: number;
  reason: string;
  error?: string;
}

async function nliCall(prompt: string, propA: string, propB: string): Promise<NliResult> {
  const filled = prompt.replace("{PROP_A}", propA).replace("{PROP_B}", propB);
  try {
    const { object } = await generateObject({
      model: getLanguageModel(VERIFIER_MODEL),
      schema: NliSchema,
      prompt: filled,
      temperature: 0.0,
    });
    return { fires: object.fires, score: object.score, reason: object.reason };
  } catch (e: any) {
    const transient = classifyError(e);
    const kind = transient ? transient.kind : "unknown";
    return {
      fires: false,
      score: 0,
      reason: `nli error (${kind}): ${e?.message || String(e)}`.slice(0, 300),
      error: kind,
    };
  }
}

// ---------------------------------------------------------------------------
// Row schema
// ---------------------------------------------------------------------------
interface FireRow {
  ts: string;
  transcript_id: string;
  repo: string;
  propositions: Array<{ entity: string; proposition: string }>;
}

interface PairRow {
  ts: string;
  transcript_id: string;
  repo: string;
  entity: string;
  entity_class: string;
  proposition: string;
  claim_id: number;
  claim_text: string;
  dossier_slug: string | null;
  similarity: number;
  strict: NliResult;
  loose: NliResult;
  broad: NliResult;
}

// ---------------------------------------------------------------------------
// Main loop
// ---------------------------------------------------------------------------
function loadJsonl<T>(path: string): T[] {
  if (!existsSync(path)) return [];
  return readFileSync(path, "utf8")
    .split("\n")
    .filter((l) => l.trim())
    .map((l) => {
      try {
        return JSON.parse(l) as T;
      } catch {
        return null;
      }
    })
    .filter((x): x is T => x !== null);
}

async function pairsForProposition(
  entity: string,
  proposition: string,
): Promise<Array<{ claim: NonNullable<ReturnType<typeof store.getClaim>>; similarity: number }>> {
  if (!entity || entity.length < 2) return [];
  let queryEmb: Float32Array;
  try {
    queryEmb = await embedOne(`${entity}. ${proposition}`);
  } catch {
    return [];
  }
  const hits = store.searchHybrid(queryEmb, COUNTER_EVIDENCE_TOPK).filter((h) => h.kind === "claim");
  const survivors: Array<{ claim: NonNullable<ReturnType<typeof store.getClaim>>; similarity: number }> = [];
  for (const h of hits) {
    if (h.id == null) continue;
    if (h.similarity < COUNTER_EVIDENCE_MIN_COS) continue;
    const claim = store.getClaim(h.id);
    if (!claim) continue;
    if (claim.status !== "supported") continue;
    if (claim.superseded_by != null) continue;
    if (!sharesPrimaryEntity(entity, claim.claim_text)) continue;
    survivors.push({ claim, similarity: h.similarity });
  }
  return survivors;
}

async function judgePair(
  row: FireRow,
  entity: string,
  proposition: string,
  claim: NonNullable<ReturnType<typeof store.getClaim>>,
  similarity: number,
): Promise<PairRow> {
  // Run all three variants in parallel — independent calls, no shared state.
  const [strict, loose, broad] = await Promise.all([
    nliCall(PROMPT_STRICT, proposition, claim.claim_text),
    nliCall(PROMPT_LOOSE, proposition, claim.claim_text),
    nliCall(PROMPT_BROAD, proposition, claim.claim_text),
  ]);
  return {
    ts: row.ts,
    transcript_id: row.transcript_id,
    repo: row.repo,
    entity,
    entity_class: classifyEntity(entity, proposition),
    proposition,
    claim_id: claim.id,
    claim_text: claim.claim_text,
    dossier_slug: claim.dossier_slug || null,
    similarity,
    strict,
    loose,
    broad,
  };
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

function pct(n: number, d: number): string {
  return d > 0 ? `${((n / d) * 100).toFixed(1)}%` : "—";
}

function printSummary(pairs: PairRow[]): void {
  const total = pairs.length;
  const strictFires = pairs.filter((p) => p.strict.fires && p.strict.score >= COUNTER_EVIDENCE_FIRE_SCORE).length;
  const looseFires = pairs.filter((p) => p.loose.fires && p.loose.score >= COUNTER_EVIDENCE_FIRE_SCORE).length;
  const broadFires = pairs.filter((p) => p.broad.fires && p.broad.score >= COUNTER_EVIDENCE_FIRE_SCORE).length;

  console.error("");
  console.error(`pair distribution (total = ${total})`);
  console.error(`  strict fires (production)   ${strictFires.toString().padStart(4)}   ${pct(strictFires, total)}`);
  console.error(`  loose  fires                ${looseFires.toString().padStart(4)}   ${pct(looseFires, total)}`);
  console.error(`  broad  fires                ${broadFires.toString().padStart(4)}   ${pct(broadFires, total)}`);
  console.error("");

  // Cross-tab: strict vs broad disagreement is the interesting cell.
  const strictNoBroadYes = pairs.filter(
    (p) =>
      !(p.strict.fires && p.strict.score >= COUNTER_EVIDENCE_FIRE_SCORE) &&
      p.broad.fires && p.broad.score >= COUNTER_EVIDENCE_FIRE_SCORE,
  ).length;
  const strictNoLooseYes = pairs.filter(
    (p) =>
      !(p.strict.fires && p.strict.score >= COUNTER_EVIDENCE_FIRE_SCORE) &&
      p.loose.fires && p.loose.score >= COUNTER_EVIDENCE_FIRE_SCORE,
  ).length;
  console.error(`disagreement cells (strict says no-fire)`);
  console.error(`  loose-only fires            ${strictNoLooseYes.toString().padStart(4)}   ${pct(strictNoLooseYes, total - strictFires)}`);
  console.error(`  broad-only fires            ${strictNoBroadYes.toString().padStart(4)}   ${pct(strictNoBroadYes, total - strictFires)}`);
  console.error("");

  // Entity coverage
  const classCounts: Record<string, number> = {};
  const classFires: Record<string, { strict: number; loose: number; broad: number }> = {};
  for (const p of pairs) {
    classCounts[p.entity_class] = (classCounts[p.entity_class] || 0) + 1;
    if (!classFires[p.entity_class]) classFires[p.entity_class] = { strict: 0, loose: 0, broad: 0 };
    if (p.strict.fires && p.strict.score >= COUNTER_EVIDENCE_FIRE_SCORE) classFires[p.entity_class]!.strict++;
    if (p.loose.fires && p.loose.score >= COUNTER_EVIDENCE_FIRE_SCORE) classFires[p.entity_class]!.loose++;
    if (p.broad.fires && p.broad.score >= COUNTER_EVIDENCE_FIRE_SCORE) classFires[p.entity_class]!.broad++;
  }
  console.error(`entity_coverage (per-class fire rates)`);
  console.error(`  class                  pairs   strict   loose   broad`);
  for (const c of Object.keys(classCounts).sort()) {
    const n = classCounts[c]!;
    const f = classFires[c]!;
    console.error(
      `  ${c.padEnd(22)} ${n.toString().padStart(5)}   ${pct(f.strict, n).padStart(6)}   ${pct(f.loose, n).padStart(5)}   ${pct(f.broad, n).padStart(5)}`,
    );
  }
  console.error("");
}

async function main() {
  const args = parseArgs();
  const all = loadJsonl<FireRow>(args.in);
  if (!all.length) {
    console.error(`no fires in ${args.in}`);
    process.exit(1);
  }

  if (existsSync(args.out)) unlinkSync(args.out);

  const rows = args.limit ? all.slice(0, args.limit) : all;

  // Phase 1: build (proposition, candidate) pair work list. Embed + retrieve
  // are sequential within a row but rows run in parallel under concurrency.
  console.error(`input:       ${args.in}  (${all.length} fires)`);
  console.error(`output:      ${args.out}`);
  console.error(`to replay:   ${rows.length}`);
  console.error(`concurrency: ${args.concurrency}`);
  console.error(`variants:    strict, loose, broad`);
  console.error("");

  const t0 = Date.now();

  interface WorkItem {
    row: FireRow;
    entity: string;
    proposition: string;
    claim: NonNullable<ReturnType<typeof store.getClaim>>;
    similarity: number;
  }
  const work: WorkItem[] = [];

  // Build the pair list sequentially-by-row but parallelize per-row via
  // concurrency. embedOne is the expensive step here.
  let propsDone = 0;
  await runWithConcurrency(rows, args.concurrency, async (row) => {
    for (const p of row.propositions) {
      const survivors = await pairsForProposition(p.entity, p.proposition);
      for (const s of survivors) {
        work.push({ row, entity: p.entity, proposition: p.proposition, claim: s.claim, similarity: s.similarity });
      }
      propsDone++;
      if (propsDone % 50 === 0) {
        console.error(`  phase 1: ${propsDone} propositions retrieved, ${work.length} pairs queued`);
      }
    }
  });

  const t1 = Date.now();
  console.error(`phase 1 done: ${work.length} pairs in ${((t1 - t0) / 1000).toFixed(1)}s`);
  console.error("");

  // Phase 2: judge every pair under 3 variants. Each pair = 3 parallel NLI
  // calls; we run pairs themselves with the same concurrency budget.
  let judged = 0;
  const pairs = await runWithConcurrency(work, args.concurrency, async (w) => {
    const result = await judgePair(w.row, w.entity, w.proposition, w.claim, w.similarity);
    appendFileSync(args.out, JSON.stringify(result) + "\n");
    judged++;
    if (judged % 25 === 0 || judged === work.length) {
      const elapsed = ((Date.now() - t1) / 1000).toFixed(1);
      const rate = (judged / Math.max(1, (Date.now() - t1) / 1000)).toFixed(2);
      console.error(`  phase 2: ${judged}/${work.length} pairs (${elapsed}s, ${rate}/s)`);
    }
    return result;
  });

  console.error("");
  console.error(`total wall: ${((Date.now() - t0) / 1000).toFixed(1)}s`);

  printSummary(pairs);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
