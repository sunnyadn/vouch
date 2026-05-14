#!/usr/bin/env bun
/**
 * counterfactual_P_alpha.ts — replay for P-α (KB counter-evidence pull,
 * issue #51, Axis-3 measurement of the comprehensiveness axis).
 *
 * Mirrors counterfactual_E.ts shape, but instead of asking "would the new
 * fire message change the revise?" it asks "would P-α have fired a
 * grounded→ungrounded flip given the current KB?".
 *
 * Methodology — fires as proposition corpus
 *   The production P-α primitive runs on GROUNDED ASSERTs (see src/gate.ts
 *   around the `VOUCH_GATE_COUNTER_EVIDENCE === "1"` block). For each
 *   grounded pair it does:
 *
 *     1. embed(`${entity}. ${proposition}`)
 *     2. searchHybrid(queryEmb, COUNTER_EVIDENCE_TOPK=5)
 *     3. filter to supported, non-superseded, sharesPrimaryEntity claims
 *        with similarity ≥ COUNTER_EVIDENCE_MIN_COS (0.55)
 *     4. verifyContradiction(prop, claim.text)
 *     5. flip grounded → ungrounded when contradicts && score ≥ 0.75
 *
 *   Steps 1-5 are IDENTICAL regardless of how the proposition originally
 *   grounded (KB-NLI vs session-source autoground). So to measure whether
 *   the KB is dense enough with counter-claims to be useful, we can run
 *   steps 1-5 on ANY proposition the agent has historically asserted —
 *   they don't need to be "grounded" in the original session.
 *
 *   The fires-last14d.jsonl corpus is the cleanest available corpus of
 *   propositions the agent has drafted in real sessions. We treat each
 *   fired proposition AS IF grounded and ask whether the current KB
 *   carries a contradicting claim.
 *
 *   Caveat: this measures STRUCTURAL flip-rate (how often does KB carry a
 *   contradiction on the entities we draft about). For the production
 *   flip-rate on naturally-grounded ASSERTs, the population skews toward
 *   entities the agent already has KB coverage on (selection effect:
 *   grounded ASSERTs are likely on well-covered entities). True-population
 *   flip-rate will be ≥ this measurement's flip-rate.
 *
 * Output: fires-counterfactual-P_alpha.jsonl
 *   Row schema:
 *     {
 *       ts, transcript_id, repo,
 *       propositions: [
 *         { entity, proposition,
 *           kb_candidates_seen,        // # passing cosine + entity-share filter
 *           contradictions: [          // up to 2
 *             { claim_id, claim_text, dossier_slug,
 *               contradiction_score, contradiction_reason }
 *           ]
 *         }
 *       ],
 *       would_flip,                    // any proposition has ≥1 contradiction
 *       max_contradiction_score,
 *       entity_class                   // heuristic taxonomy for entity_coverage metric
 *     }
 *
 * Usage:
 *   ./counterfactual_P_alpha.ts                  # full 245-row corpus
 *   ./counterfactual_P_alpha.ts --limit 20       # smoke test
 *   ./counterfactual_P_alpha.ts --concurrency 4
 *   ./counterfactual_P_alpha.ts --in fires-last14d.jsonl --out fires-counterfactual-P_alpha.jsonl
 */
import { readFileSync, existsSync, appendFileSync, unlinkSync } from "node:fs";
import { join } from "node:path";
import { verifyContradiction } from "../../src/verifier.ts";
import * as store from "../../src/store.ts";
import { embedOne } from "../../src/embedder.ts";

interface Args {
  in: string;
  out: string;
  concurrency: number;
  limit?: number;
  resume: boolean;
}

function parseArgs(): Args {
  const args = process.argv.slice(2);
  const out: Args = {
    in: join(import.meta.dir, "fires-last14d.jsonl"),
    out: join(import.meta.dir, "fires-counterfactual-P_alpha.jsonl"),
    concurrency: Number(process.env.DOGFOOD_REPLAY_CONCURRENCY || "6"),
    resume: false,
  };
  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (a === "--in") out.in = args[++i]!;
    else if (a === "--out") out.out = args[++i]!;
    else if (a === "--concurrency") out.concurrency = Number(args[++i]);
    else if (a === "--limit") out.limit = Number(args[++i]);
    else if (a === "--resume") out.resume = true;
    else if (a === "-h" || a === "--help") {
      console.error(
        "Usage: counterfactual_P_alpha.ts [--in PATH] [--out PATH] [--concurrency N] [--limit N] [--resume]",
      );
      process.exit(0);
    }
  }
  return out;
}

interface FireRow {
  ts: string;
  transcript_id: string;
  repo: string;
  propositions: Array<{ entity: string; proposition: string }>;
}

interface PropResult {
  entity: string;
  proposition: string;
  kb_candidates_seen: number;
  contradictions: Array<{
    claim_id: number;
    claim_text: string;
    dossier_slug: string | null;
    similarity: number;
    contradiction_score: number;
    contradiction_reason: string;
  }>;
}

interface ReplayRow {
  ts: string;
  transcript_id: string;
  repo: string;
  propositions: PropResult[];
  would_flip: boolean;
  max_contradiction_score: number | null;
  entity_class: string;
}

// Same thresholds as production P-α (src/gate.ts lines 662-664).
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

/** Heuristic entity taxonomy for the entity_coverage metric.
 *  Categories are coarse — the goal is "which entity SHAPES does P-α fire on
 *  most", not a careful ontology. */
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
  // Library/tool fallback: entity is a single-token lowercase identifier
  if (/^[a-z][a-z0-9_-]{1,30}$/.test(ent)) return "library-or-concept";
  return "other";
}

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

async function replayOneProposition(
  entity: string,
  proposition: string,
): Promise<PropResult> {
  const result: PropResult = {
    entity,
    proposition,
    kb_candidates_seen: 0,
    contradictions: [],
  };
  if (!entity || entity.length < 2) return result;

  let queryEmb: Float32Array;
  try {
    queryEmb = await embedOne(`${entity}. ${proposition}`);
  } catch {
    return result;
  }

  const hits = store.searchHybrid(queryEmb, COUNTER_EVIDENCE_TOPK).filter((h) => h.kind === "claim");
  const candidates: Array<{ claim: ReturnType<typeof store.getClaim>; similarity: number }> = [];
  for (const h of hits) {
    if (h.id == null) continue;
    if (h.similarity < COUNTER_EVIDENCE_MIN_COS) continue;
    const claim = store.getClaim(h.id);
    if (!claim) continue;
    if (claim.status !== "supported") continue;
    if (claim.superseded_by != null) continue;
    if (!sharesPrimaryEntity(entity, claim.claim_text)) continue;
    candidates.push({ claim, similarity: h.similarity });
  }
  result.kb_candidates_seen = candidates.length;

  for (const { claim, similarity } of candidates) {
    if (!claim) continue;
    try {
      const verdict = await verifyContradiction(proposition, claim.claim_text);
      if (verdict.contradicts && verdict.score >= COUNTER_EVIDENCE_FIRE_SCORE) {
        result.contradictions.push({
          claim_id: claim.id,
          claim_text: claim.claim_text,
          dossier_slug: claim.dossier_slug || null,
          similarity,
          contradiction_score: verdict.score,
          contradiction_reason: verdict.reason,
        });
        if (result.contradictions.length >= 2) break; // same cap as production
      }
    } catch {
      // transient — skip this candidate (matches production behavior)
    }
  }
  return result;
}

async function replayOneRow(row: FireRow): Promise<ReplayRow> {
  const propResults: PropResult[] = [];
  for (const p of row.propositions) {
    propResults.push(await replayOneProposition(p.entity, p.proposition));
  }
  const allContradictions = propResults.flatMap((p) => p.contradictions);
  const maxScore = allContradictions.length
    ? Math.max(...allContradictions.map((c) => c.contradiction_score))
    : null;
  const entityClass =
    propResults.find((p) => p.contradictions.length > 0) ??
    propResults[0] ??
    null;
  return {
    ts: row.ts,
    transcript_id: row.transcript_id,
    repo: row.repo,
    propositions: propResults,
    would_flip: allContradictions.length > 0,
    max_contradiction_score: maxScore,
    entity_class: entityClass
      ? classifyEntity(entityClass.entity, entityClass.proposition)
      : "other",
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

function printSummary(rows: ReplayRow[]): void {
  const total = rows.length;
  const flipped = rows.filter((r) => r.would_flip);
  const flipRate = total > 0 ? (flipped.length / total) * 100 : 0;

  const pct = (n: number, d: number) => (d > 0 ? `${((n / d) * 100).toFixed(1)}%` : "—");

  console.error("");
  console.error(`replay distribution (total = ${total} fire rows)`);
  console.error(`  would_flip       ${flipped.length.toString().padStart(4)}   ${pct(flipped.length, total)}`);
  console.error(`  no_flip          ${(total - flipped.length).toString().padStart(4)}   ${pct(total - flipped.length, total)}`);
  console.error("");

  // NLI confidence histogram on the flipped subset.
  const buckets = { "0.75-0.80": 0, "0.80-0.85": 0, "0.85-0.90": 0, "0.90-0.95": 0, "0.95-1.00": 0 };
  for (const r of flipped) {
    const s = r.max_contradiction_score!;
    if (s < 0.80) buckets["0.75-0.80"]++;
    else if (s < 0.85) buckets["0.80-0.85"]++;
    else if (s < 0.90) buckets["0.85-0.90"]++;
    else if (s < 0.95) buckets["0.90-0.95"]++;
    else buckets["0.95-1.00"]++;
  }
  console.error(`NLI confidence distribution (max_contradiction_score on flipped subset, n=${flipped.length})`);
  for (const [k, v] of Object.entries(buckets)) {
    console.error(`  ${k}   ${v.toString().padStart(4)}   ${pct(v, flipped.length)}`);
  }
  console.error("");

  // entity_class breakdown.
  const classCountsAll: Record<string, number> = {};
  const classCountsFlipped: Record<string, number> = {};
  for (const r of rows) classCountsAll[r.entity_class] = (classCountsAll[r.entity_class] || 0) + 1;
  for (const r of flipped) classCountsFlipped[r.entity_class] = (classCountsFlipped[r.entity_class] || 0) + 1;
  const classes = Array.from(new Set([...Object.keys(classCountsAll), ...Object.keys(classCountsFlipped)])).sort();
  console.error(`entity_coverage (which entity classes P-α fires on most)`);
  console.error(`  class                  all      flipped   flip_rate_within_class`);
  for (const c of classes) {
    const all = classCountsAll[c] || 0;
    const fl = classCountsFlipped[c] || 0;
    console.error(
      `  ${c.padEnd(22)} ${all.toString().padStart(4)}     ${fl.toString().padStart(4)}      ${pct(fl, all)}`,
    );
  }
  console.error("");

  console.error(`headline metric:  flip_rate = ${flipped.length} / ${total} = ${flipRate.toFixed(1)}%`);
}

async function main() {
  const args = parseArgs();
  const all = loadJsonl<FireRow>(args.in);
  if (!all.length) {
    console.error(`no fires in ${args.in}`);
    process.exit(1);
  }

  if (existsSync(args.out)) {
    if (args.resume) {
      console.error(`(--resume not supported, would need ts-keyed dedupe; aborting)`);
      process.exit(1);
    } else {
      unlinkSync(args.out);
    }
  }

  const todo = args.limit ? all.slice(0, args.limit) : all;
  console.error(`input:       ${args.in}  (${all.length} fires)`);
  console.error(`output:      ${args.out}`);
  console.error(`to replay:   ${todo.length}`);
  console.error(`concurrency: ${args.concurrency}`);
  console.error("");

  const t0 = Date.now();
  let done = 0;
  let flipsSoFar = 0;

  const rows = await runWithConcurrency(todo, args.concurrency, async (row) => {
    const replay = await replayOneRow(row);
    done++;
    if (replay.would_flip) flipsSoFar++;
    appendFileSync(args.out, JSON.stringify(replay) + "\n");
    if (done % 10 === 0 || done === todo.length) {
      const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
      const rate = (done / Math.max(1, (Date.now() - t0) / 1000)).toFixed(2);
      console.error(`  ${done}/${todo.length} done (${elapsed}s, ${rate}/s, flips=${flipsSoFar})`);
    }
    return replay;
  });

  printSummary(rows);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
