#!/usr/bin/env bun
/**
 * counterfactual_E.ts — synthetic A/B for #50 (E) on the dogfood baseline.
 *
 * For each `dodge`-labeled fire in fires-labeled.jsonl:
 *   1. Query current KB for the top-2 supported claims whose claim_text
 *      shares the firing entity (mirroring src/gate.ts sharesPrimaryEntity).
 *   2. If no KB candidates exist for this entity → label as "would-not-change"
 *      ((E) has no effect; the bare `vouch search` fallback fires).
 *   3. Otherwise → send Gemini Pro a synthesis of:
 *         prior_user, draft, OLD fire_text, the ACTUAL post-fire dodge,
 *         and the synthetic (E) addendum (KB nearest claims with slugs).
 *      Judge classifies the counterfactual revise: would-verify / would-hedge /
 *      still-dodge / unclear.
 *
 * Caveats stated explicitly:
 *   - This is an LLM-judge counterfactual, NOT a real A/B. The judge has
 *     "anything plausibly helps" bias. Treat the would-verify + would-hedge
 *     total as an UPPER BOUND on (E)'s real-world lift; the realized lift
 *     in next-session dogfood is the binding number.
 *   - We use the CURRENT KB (today, 2026-05-14). Historical fires from May
 *     9-11 didn't have access to today's claims; the counterfactual asks
 *     "if those fires happened today with current KB, would (E) help."
 *   - We do NOT re-extract propositions or re-run NLI. We use the historical
 *     entity / proposition / fire_text as-is.
 *
 * Output: fires-counterfactual-E.jsonl (gitignored). Plus printed summary.
 */
import { readFileSync, existsSync, appendFileSync } from "node:fs";
import { join } from "node:path";
import { z } from "zod";
import { generateObject } from "ai";
import { getLanguageModel } from "../../src/providers.ts";
import * as store from "../../src/store.ts";
import { embedOne } from "../../src/embedder.ts";

interface Args {
  in: string;
  out: string;
  judge: string;
  concurrency: number;
  limit?: number;
}

function parseArgs(): Args {
  const args = process.argv.slice(2);
  const out: Args = {
    in: join(import.meta.dir, "fires-labeled.jsonl"),
    out: join(import.meta.dir, "fires-counterfactual-E.jsonl"),
    judge: process.env.DOGFOOD_JUDGE_MODEL || "gemini-3.1-pro-preview",
    concurrency: Number(process.env.DOGFOOD_JUDGE_CONCURRENCY || "6"),
  };
  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (a === "--in") out.in = args[++i]!;
    else if (a === "--out") out.out = args[++i]!;
    else if (a === "--judge") out.judge = args[++i]!;
    else if (a === "--concurrency") out.concurrency = Number(args[++i]);
    else if (a === "--limit") out.limit = Number(args[++i]);
  }
  return out;
}

interface DodgeRow {
  ts: string;
  transcript_id: string;
  propositions: Array<{ entity: string; proposition: string }>;
  draft: string;
  prior_user: string;
  fire_text: string;
  post_fire_draft?: string;
  manual_label: { class: string; reasoning?: string };
}

interface KbCandidate {
  claim_id: number;
  claim_text: string;
  dossier_slug: string | null;
  similarity: number;
}

/** Mirrors src/gate.ts sharesPrimaryEntity. Substring or alphanum-only
 *  substring (length ≥ 3). */
function sharesPrimaryEntity(entity: string, text: string): boolean {
  const ent = entity.toLowerCase().trim();
  if (ent && text.toLowerCase().includes(ent)) return true;
  const entAlnum = entity.replace(/[^a-zA-Z0-9]/g, "").toLowerCase();
  if (entAlnum.length < 3) return false;
  return text.replace(/[^a-zA-Z0-9]/g, "").toLowerCase().includes(entAlnum);
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

/** Look up KB candidates as production gate (#50 E) would surface them:
 *  embed `<entity>. <proposition>`, hybrid-search the KB (embedding +
 *  full-text), take supported non-superseded claims that ALSO share the
 *  firing entity, top-2 by similarity. Mirrors src/gate.ts groundExtracted
 *  Pairs Phase 1 + sharesPrimaryEntity display filter — same primitive,
 *  same selection criteria. */
const WIDER_TOPK = 8;
async function lookupKbCandidates(
  entity: string,
  proposition: string,
): Promise<KbCandidate[]> {
  if (!entity || entity.length < 2) return [];
  let queryEmb: Float32Array;
  try {
    queryEmb = await embedOne(`${entity}. ${proposition}`);
  } catch {
    return [];
  }
  const hits = store
    .searchHybrid(queryEmb, WIDER_TOPK)
    .filter((h) => h.kind === "claim");
  const out: KbCandidate[] = [];
  for (const h of hits) {
    if (out.length >= 2) break;
    if (h.id == null) continue;
    const claim = store.getClaim(h.id);
    if (!claim) continue;
    if (claim.status !== "supported") continue;
    if (claim.superseded_by != null) continue;
    if (!sharesPrimaryEntity(entity, claim.claim_text)) continue;
    out.push({
      claim_id: claim.id,
      claim_text: claim.claim_text,
      dossier_slug: claim.dossier_slug || null,
      similarity: h.similarity,
    });
  }
  return out;
}

const JudgeSchema = z.object({
  counterfactual: z.enum(["would-verify", "would-hedge", "still-dodge", "unclear"]),
  reasoning: z.string().max(400),
});

const JUDGE_PROMPT = `You evaluate a counterfactual: GIVEN that vouch's gate now surfaces KB-nearest candidates inline in the fire message (issue #50 E), would the coding-agent (Claude Code) have produced a DIFFERENT revise on this historical fire?

The agent's ACTUAL revise on this fire was classified as "dodge" — meaning one of:
  (4a) argued the fire was a false positive without verifying
  (4b) silently rephrased to remove the named entity, leaving the claim implicit

Now consider what (E) changes: the new fire message ADDS a "KB nearest:" block listing up to 2 existing supported KB claims whose text contains the firing entity, with their dossier slugs and the NLI gap reasons. Two productive paths are surfaced explicitly: (a) rephrase the claim to match an existing dossier; (b) fetch a fresh source via vouch search.

Classify the counterfactual revise:
  - would-verify: the agent now plausibly runs \`vouch claim ... --dossier <slug>\` (rephrasing the claim to match an existing dossier surfaced in the KB block) OR \`vouch fetch <url>\` (fetching a fresh source via the suggested search). Net: KB grows, claim grounded.
  - would-hedge: the agent now plausibly keeps the claim but adds explicit "(Unverified, …)" provenance tag near the entity — the surfaced KB info clarifies that there's adjacent ground truth but not exact entailment.
  - still-dodge: the surfaced KB info doesn't change the agent's behavior; the dodge pattern (4a meta-rationalize / 4b silent rephrase) still wins.
  - unclear: the dodge is ambiguous, the KB candidates are too far from the claim, or no clear counterfactual can be inferred from the row alone.

Be CONSERVATIVE on would-verify and would-hedge. The dodge pattern is well-rehearsed in the agent; a more informative fire is necessary but not always sufficient.

Context for this case:

PRIOR USER PROMPT (what the human asked):
\`\`\`
{PRIOR_USER}
\`\`\`

AGENT DRAFT that fired the gate:
\`\`\`
{DRAFT}
\`\`\`

OLD FIRE TEXT (what the agent saw):
\`\`\`
{OLD_FIRE}
\`\`\`

NEW (E) ADDENDUM (what would be added to the fire under #50 E):
\`\`\`
{NEW_ADDENDUM}
\`\`\`

AGENT'S ACTUAL POST-FIRE REVISE (classified as dodge by prior judge):
\`\`\`
{POST_FIRE}
\`\`\`

Original judge's reasoning for "dodge": {DODGE_REASON}

Return JSON: { counterfactual, reasoning (≤2 sentences, point at specific text spans). }`;

function renderEAddendum(candidates: KbCandidate[]): string {
  if (!candidates.length) {
    return "(no KB candidates pass cosine + entity-share filter — (E) has no effect; fire would suggest `vouch search` only)";
  }
  const lines = [
    "Additional fire content under #50 E:",
    "  KB nearest:",
    ...candidates.map(
      (c) =>
        `    [claim ${c.claim_id}, cos=${c.similarity.toFixed(2)}] "${c.claim_text.slice(0, 160)}${c.claim_text.length > 160 ? "…" : ""}"` +
        (c.dossier_slug ? `\n         dossier: ${c.dossier_slug}` : ""),
    ),
    "  → if rephrasing the claim closes the gap: vouch claim \"<rephrased>\" --type ATOMIC --dossier " +
      (candidates[0]!.dossier_slug || "<slug>"),
    "  → for a fresh source: vouch search \"<entity>\"",
  ];
  return lines.join("\n");
}

async function classifyOne(
  row: DodgeRow,
  judge: string,
): Promise<{
  candidates_found: number;
  counterfactual: string;
  reasoning: string;
} | null> {
  // Take the first proposition's entity as the primary firing entity.
  const firstProp = row.propositions[0];
  if (!firstProp) return null;
  const candidates = await lookupKbCandidates(firstProp.entity, firstProp.proposition);

  // Fast-path: no candidates → (E) has no effect, no need to spend a judge call.
  if (candidates.length === 0) {
    return {
      candidates_found: 0,
      counterfactual: "still-dodge",
      reasoning: "(E) surfaces no KB candidates for this entity; fire would still just suggest `vouch search`.",
    };
  }

  const prompt = JUDGE_PROMPT
    .replace("{PRIOR_USER}", (row.prior_user || "(empty)").slice(0, 1500))
    .replace("{DRAFT}", (row.draft || "(empty)").slice(0, 2500))
    .replace("{OLD_FIRE}", (row.fire_text || "(empty)").slice(0, 1500))
    .replace("{NEW_ADDENDUM}", renderEAddendum(candidates))
    .replace("{POST_FIRE}", (row.post_fire_draft || "(empty — session truncated)").slice(0, 2500))
    .replace("{DODGE_REASON}", row.manual_label.reasoning || "(no prior reasoning)");

  try {
    const { object } = await generateObject({
      model: getLanguageModel(judge),
      schema: JudgeSchema,
      prompt,
      temperature: 0.0,
    });
    return {
      candidates_found: candidates.length,
      counterfactual: object.counterfactual,
      reasoning: object.reasoning,
    };
  } catch (e: any) {
    console.error(`  [judge err] ts=${row.ts}: ${e?.message || e}`);
    return null;
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
  const rows = loadJsonl<DodgeRow>(args.in);
  const dodges = rows.filter((r) => r.manual_label?.class === "dodge");
  const todo = args.limit ? dodges.slice(0, args.limit) : dodges;

  console.error(`input:    ${args.in}  (${rows.length} labeled, ${dodges.length} dodges)`);
  console.error(`to judge: ${todo.length}`);
  console.error(`judge:    ${args.judge}  (concurrency=${args.concurrency})`);
  console.error("");

  const t0 = Date.now();
  let done = 0;
  const counts: Record<string, number> = {
    "would-verify": 0,
    "would-hedge": 0,
    "still-dodge": 0,
    "unclear": 0,
  };
  let zeroCandidates = 0;
  const results: Array<{ row: DodgeRow; verdict: NonNullable<Awaited<ReturnType<typeof classifyOne>>> }> = [];

  await runWithConcurrency(todo, args.concurrency, async (row) => {
    const verdict = await classifyOne(row, args.judge);
    done++;
    if (!verdict) return;
    counts[verdict.counterfactual]!++;
    if (verdict.candidates_found === 0) zeroCandidates++;
    results.push({ row, verdict });
    appendFileSync(args.out, JSON.stringify({ ts: row.ts, transcript_id: row.ts, ...verdict }) + "\n");
    if (done % 10 === 0 || done === todo.length) {
      const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
      console.error(`  ${done}/${todo.length} done (${elapsed}s, ${zeroCandidates} no-KB-candidate)`);
    }
  });

  const total = todo.length;
  const pct = (n: number) => `${((n / total) * 100).toFixed(1)}%`;

  console.error("");
  console.error(`counterfactual distribution  (total = ${total} dodges)`);
  console.error(`  would-verify    ${counts["would-verify"].toString().padStart(4)}   ${pct(counts["would-verify"]!)}`);
  console.error(`  would-hedge     ${counts["would-hedge"].toString().padStart(4)}   ${pct(counts["would-hedge"]!)}`);
  console.error(`  still-dodge     ${counts["still-dodge"].toString().padStart(4)}   ${pct(counts["still-dodge"]!)}`);
  console.error(`  unclear         ${counts["unclear"].toString().padStart(4)}   ${pct(counts["unclear"]!)}`);
  console.error("");
  console.error(`upper-bound (E) lift on dodge_rate:`);
  const lift = counts["would-verify"]! + counts["would-hedge"]!;
  console.error(`  (would-verify + would-hedge) / total = ${lift} / ${total} = ${pct(lift)}`);
  console.error("");
  console.error(`coverage:`);
  console.error(`  ${zeroCandidates}/${total} had NO KB candidates after entity-share filter`);
  console.error(`  (these auto-class to still-dodge — (E) has no effect)`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
