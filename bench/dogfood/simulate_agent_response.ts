#!/usr/bin/env bun
// simulate_agent_response.ts — does vouch fire MESSAGE shape change the
// agent's revise action distribution? Specifically: does an explicit
// fetch-driving message increase fetch% above the observed dogfood
// baseline (remove 87% / fetch 8% / hedge 4%)?
//
// Sample ~15 fires (mix of P-α NLI, value-reconcile, category-mismatch).
// For each, construct TWO gate fire messages:
//   - v1 "suppression" — current shape, mentions KB counter / no source,
//     no explicit tool-call suggestion
//   - v2 "fetch-driving" — same evidence, but explicitly proposes a
//     concrete `vouch search`/`vouch fetch` action and frames the draft
//     as recoverable-via-verification
// Then for each (case, version) ask the verifier LLM acting as the agent
// to choose one of {remove, hedge, fetch, continue} and explain.
//
// Methodological limitation: agent ≈ Gemini-3.1-Pro here, not the real
// Claude Code agent that vouch usually runs against. The intent is
// direction-of-effect, not point-estimate. Same model both arms = paired
// design controls model drift.

import { readFileSync, writeFileSync } from "fs";
import { dirname, join } from "path";
import { fileURLToPath } from "url";
import { generateObject } from "ai";
import { z } from "zod";
import { getLanguageModel } from "../../src/providers.ts";
import { VERIFIER_MODEL } from "../../src/config.ts";

const HERE = dirname(fileURLToPath(import.meta.url));

type Case = {
  source: "p-alpha-strict" | "value-reconcile" | "category-mismatch";
  entity: string;
  draft_proposition: string;
  kb_claim: string;
  fire_reason: string;
};

// Pull the fire output files for 3 detectors and sample a mix.
function load(path: string): any[] {
  const txt = readFileSync(join(HERE, path), "utf8").trim();
  if (!txt) return [];
  return txt.split("\n").map(l => JSON.parse(l));
}

const valueReconcileFires = load("value-reconcile-probe-fires.jsonl");
const categoryMismatchFires = load("category-mismatch-probe-fires.jsonl");
const judgeStudyPairs = load("fires-judge-study-P_alpha.jsonl");
const pAlphaFires = judgeStudyPairs.filter((p: any) => p.strict.fires);

function pick<T>(arr: T[], n: number, seed = 42): T[] {
  // deterministic-ish: stride sample
  if (arr.length <= n) return arr;
  const step = Math.floor(arr.length / n);
  return Array.from({ length: n }, (_, i) => arr[i * step]);
}

const cases: Case[] = [
  ...pAlphaFires.slice(0, 3).map((p: any) => ({
    source: "p-alpha-strict" as const,
    entity: p.entity,
    draft_proposition: p.proposition,
    kb_claim: p.claim_text,
    fire_reason: `KB attests the opposite: "${p.strict.reason}"`,
  })),
  ...pick(valueReconcileFires, 5).map((f: any) => ({
    source: "value-reconcile" as const,
    entity: f.entity,
    draft_proposition: f.proposition,
    kb_claim: f.claim_text,
    fire_reason: `value mismatch: ${(f.meta.mismatched as any[]).map((m: any) => `${m.prop_num}≠${m.claim_num} on '${m.shared_anchors.slice(0, 2).join(", ")}'`).join("; ")}`,
  })),
  ...pick(categoryMismatchFires, 5).map((f: any) => ({
    source: "category-mismatch" as const,
    entity: f.entity,
    draft_proposition: f.proposition,
    kb_claim: f.claim_text,
    fire_reason: `category mismatch: draft says ${(f.meta.prop_categories as string[]).join("/")}, KB says ${(f.meta.claim_categories as string[]).join("/")}`,
  })),
];

console.error(`Loaded ${cases.length} cases (p-α=3, value-reconcile=5, category-mismatch=5+)`);

// ──────────────────────────────────────────────────────────────────────────
// Two fire message variants
// ──────────────────────────────────────────────────────────────────────────

// v1 = the OLD production guidance string (pre-this-PR baseline)
function suppressionMessage(c: Case): string {
  return `[vouch-gate] Detected ungrounded factual claim(s) about named external entities.
  • ${c.entity}: "${c.draft_proposition}"
    KB nearest: "${c.kb_claim}"
    ${c.fire_reason}

Before answering, ground each claim:
  • vouch search "<keyword>" — check the KB
  • vouch fetch <url> — pull the source
  • vouch claim "<text>" --type ATOMIC --dossier <slug> --source-quote "..."
Or hedge explicitly with "(unverified, from training memory)" near the claim.`;
}

// v2 = the NEW production guidance — tighter + ANSI bold-red on entity
function fetchDrivingMessage(c: Case): string {
  const red = (s: string) => `\x1b[1;31m${s}\x1b[0m`;
  return `[vouch-gate] Detected ungrounded factual claim(s) about named external entities.
  • ${red(c.entity)}: "${c.draft_proposition}"
    KB nearest: "${c.kb_claim}"
    ${c.fire_reason}

Verify; don't delete. The source exists.
  • vouch search "${c.entity}"
  • vouch fetch <url>
  • vouch claim "<text>" --dossier <slug>
If the source contradicts your draft, rephrase. Silent delete hides what you didn't check. Hedge "(unverified, from training memory)" only if no source can verify.`;
}

// ──────────────────────────────────────────────────────────────────────────
// Agent-response schema
// ──────────────────────────────────────────────────────────────────────────

const ActionSchema = z.object({
  action: z.enum(["remove", "hedge", "fetch", "continue"]).describe(
    "remove = silently delete the claim from the draft. hedge = keep the claim but mark it as '(unverified, from training memory)' or similar. fetch = invoke vouch search / vouch fetch to find evidence and verify. continue = stand by the original draft, override the gate."
  ),
  why: z.string().max(300).describe("One-sentence reason for the action."),
});

async function simulateAgent(c: Case, message: string): Promise<{ action: string; why: string }> {
  try {
    const { object } = await generateObject({
      model: getLanguageModel(VERIFIER_MODEL),
      schema: ActionSchema,
      prompt: `You are an AI writing agent (Claude Code-style) that just wrote this draft proposition:

"${c.draft_proposition}"

The vouch gate (an external fact-check hook) fired with this feedback message:

---
${message}
---

What is your next action? Choose ONE of:
- remove: silently delete the claim from the draft
- hedge: keep the claim but mark it explicitly as unverified
- fetch: run \`vouch search\` or \`vouch fetch\` to find the actual source and verify
- continue: stand by the original draft, override the gate

Pick the action you would actually do given typical workflow incentives (you want to ship the draft, you have other work, the gate is one of many feedback signals).`,
    });
    return { action: object.action, why: object.why };
  } catch (e: any) {
    return { action: "ERROR", why: e.message };
  }
}

// ──────────────────────────────────────────────────────────────────────────
// Run paired simulation
// ──────────────────────────────────────────────────────────────────────────

type Row = {
  source: string;
  entity: string;
  draft: string;
  v1_action: string;
  v1_why: string;
  v2_action: string;
  v2_why: string;
};

const rows: Row[] = [];
const t0 = Date.now();

for (const c of cases) {
  const v1 = await simulateAgent(c, suppressionMessage(c));
  const v2 = await simulateAgent(c, fetchDrivingMessage(c));
  rows.push({
    source: c.source,
    entity: c.entity,
    draft: c.draft_proposition.slice(0, 80),
    v1_action: v1.action,
    v1_why: v1.why,
    v2_action: v2.action,
    v2_why: v2.why,
  });
  console.error(`  [${rows.length}/${cases.length}] v1=${v1.action} v2=${v2.action} (${c.entity})`);
}

const wall = (Date.now() - t0) / 1000;

// Tally
function tally(actions: string[]): Record<string, number> {
  const counts: Record<string, number> = { remove: 0, hedge: 0, fetch: 0, continue: 0 };
  for (const a of actions) counts[a] = (counts[a] || 0) + 1;
  return counts;
}

const v1Tally = tally(rows.map(r => r.v1_action));
const v2Tally = tally(rows.map(r => r.v2_action));

// By source bucket
const bySource: Record<string, { v1: Record<string, number>; v2: Record<string, number> }> = {};
for (const r of rows) {
  if (!bySource[r.source]) bySource[r.source] = { v1: tally([]), v2: tally([]) };
  bySource[r.source].v1[r.v1_action] = (bySource[r.source].v1[r.v1_action] || 0) + 1;
  bySource[r.source].v2[r.v2_action] = (bySource[r.source].v2[r.v2_action] || 0) + 1;
}

const summary = {
  cases_n: rows.length,
  wall_seconds: wall,
  llm_calls: rows.length * 2,
  baseline_observed_dogfood: { remove: 21, fetch: 2, hedge: 1, continue: 0 },
  v1_suppression_message: v1Tally,
  v2_fetch_driving_message: v2Tally,
  delta_fetch_pp: ((v2Tally.fetch ?? 0) - (v1Tally.fetch ?? 0)) / rows.length * 100,
  delta_remove_pp: ((v2Tally.remove ?? 0) - (v1Tally.remove ?? 0)) / rows.length * 100,
  by_source: bySource,
};

console.log("\n" + JSON.stringify(summary, null, 2));

writeFileSync(join(HERE, "simulate-agent-response-rows.jsonl"), rows.map(r => JSON.stringify(r)).join("\n"));
writeFileSync(join(HERE, "simulate-agent-response-summary.json"), JSON.stringify(summary, null, 2));
console.error(`\nWrote rows + summary to bench/dogfood/`);
