#!/usr/bin/env bun
// simulate_inject_vs_resolve.ts — three-way simulation: does Exa-fetched
// content change agent's verify behavior?
//
//   v2 (current new production)     : fire message with vouch search/fetch
//                                     suggestion, NO inline content
//   v3a (option A — inject context) : v2 + top-3 (URL, excerpt) inline
//   v3b (option B — gate resolves)  : vouch runs NLI itself; supports →
//                                     silent auto-ground; contradicts →
//                                     surface contradicting source
//
// Inputs: 8 "fetchable but silently deleted" cases from the session_claims
// audit (2026-05-15). Per case: Exa search-with-text → 3 (url, excerpt);
// NLI each excerpt against the agent's draft proposition; classify
// supports/contradicts/neither; build 3 fire messages; simulate agent
// response; tally action distribution per variant.

import { writeFileSync } from "fs";
import { dirname, join } from "path";
import { fileURLToPath } from "url";
import { generateObject } from "ai";
import { z } from "zod";
import { getLanguageModel } from "../../src/providers.ts";
import { VERIFIER_MODEL } from "../../src/config.ts";
import { verifyClaimAgainstSource } from "../../src/verifier.ts";

const HERE = dirname(fileURLToPath(import.meta.url));

// 8 cases from the audit. Proposition = what the agent wrote (and got
// silent-deleted). Some are factually wrong (compromise size, wink-nlp
// size) — auto-fetch should surface contradiction. Others are correct
// (marimo reactive, KernelSHAP slower) — auto-fetch should support.
const CASES = [
  { entity: "compromise", context: "JavaScript NLP library", proposition: "compromise is a pure-JS library with noun chunking, POS, and NER capabilities." },
  { entity: "compromise", context: "JavaScript NLP library", proposition: "compromise is approximately 400KB in size and licensed under BSD-2." },
  { entity: "wink-nlp", context: "JavaScript NLP library", proposition: "wink-nlp is a pure-JS library providing POS tagging and lemmatization." },
  { entity: "wink-nlp", context: "JavaScript NLP library", proposition: "wink-nlp is approximately 3MB in size and licensed under MIT." },
  { entity: "Vertex Gemini 3.1 Pro", context: "Google Cloud language model service", proposition: "Vertex Gemini 3.1 Pro is available at location=global." },
  { entity: "marimo", context: "Python reactive notebook", proposition: "marimo runs Python notebook cells reactively when dependencies change." },
  { entity: "KernelSHAP", context: "model explainability algorithm", proposition: "KernelSHAP is slower than TreeSHAP." },
  { entity: "fastcmprsk", context: "R competing-risks survival package", proposition: "fastcmprsk's auto lambda_max is broken." },
];

const EXA_KEY = process.env.EXA_API_KEY;
if (!EXA_KEY) throw new Error("EXA_API_KEY required");

// ──────────────────────────────────────────────────────────────────────────
// (1) Exa search-with-text for each case
// ──────────────────────────────────────────────────────────────────────────

type ExaResult = { url: string; title: string; text: string };

async function exaSearch(query: string): Promise<ExaResult[]> {
  const r = await fetch("https://api.exa.ai/search", {
    method: "POST",
    headers: { "Content-Type": "application/json", "x-api-key": EXA_KEY! },
    body: JSON.stringify({
      query,
      numResults: 3,
      type: "auto",
      contents: { text: { maxCharacters: 1200 } },
    }),
  });
  if (!r.ok) throw new Error(`Exa ${r.status}: ${await r.text()}`);
  const data: any = await r.json();
  return (data.results ?? []).slice(0, 3).map((x: any) => ({
    url: x.url ?? "",
    title: x.title ?? "",
    text: (x.text ?? "").slice(0, 1200),
  }));
}

// ──────────────────────────────────────────────────────────────────────────
// (2) Per-candidate NLI (proposition vs Exa text). Reuses vouch's verifier.
// ──────────────────────────────────────────────────────────────────────────

type NliVerdict = { supports: boolean; score: number; reason: string };

async function nliPropVsText(prop: string, sourceText: string): Promise<NliVerdict> {
  const result = await verifyClaimAgainstSource(prop, sourceText, "");
  // verifyClaimAgainstSource returns { supported, score, reason }
  return {
    supports: result.supported,
    score: result.score,
    reason: result.reason ?? "",
  };
}

// ──────────────────────────────────────────────────────────────────────────
// (3) Build the 3 fire messages
// ──────────────────────────────────────────────────────────────────────────

function v2Message(c: typeof CASES[0]): string {
  return `[vouch-gate] Detected ungrounded factual claim(s) about named external entities.
  • \x1b[1;31m${c.entity}\x1b[0m: "${c.proposition}"
    KB nearest: (none — no candidate claim in KB)

Verify; don't delete. The source exists.
  • vouch search "${c.entity}"
  • vouch fetch <url>
  • vouch claim "<text>" --dossier <slug>
If the source contradicts your draft, rephrase. Silent delete hides what you didn't check. Hedge "(unverified, from training memory)" only if no source can verify.`;
}

function v3aMessage(c: typeof CASES[0], exaResults: ExaResult[]): string {
  const inlined = exaResults
    .slice(0, 3)
    .map((r, i) => `    [${i + 1}] ${r.url}\n        "${r.text.slice(0, 250).replace(/\s+/g, " ")}…"`)
    .join("\n");
  return `[vouch-gate] Detected ungrounded factual claim(s) about named external entities.
  • \x1b[1;31m${c.entity}\x1b[0m: "${c.proposition}"
    KB nearest: (none — no candidate claim in KB)
    Fresh web sources for ${c.entity}:
${inlined}

Verify; don't delete. Read the sources above and reconcile.`;
}

function v3bMessageForContradict(c: typeof CASES[0], top: ExaResult, nli: NliVerdict): string {
  return `[vouch-gate] Auto-fetched contradicting source for \x1b[1;31m${c.entity}\x1b[0m:
  • Your draft: "${c.proposition}"
  • Source [${top.url}]: "${top.text.slice(0, 300).replace(/\s+/g, " ")}…"
  • NLI verdict: contradicts (score ${nli.score.toFixed(2)}) — ${nli.reason.slice(0, 120)}

Rephrase the draft to match the source, or supersede if you can verify your version with a different source.`;
}

function v3bMessageForNeither(c: typeof CASES[0], top: ExaResult): string {
  return `[vouch-gate] Auto-fetched candidate source for \x1b[1;31m${c.entity}\x1b[0m (NLI inconclusive — neither supports nor contradicts):
  • Your draft: "${c.proposition}"
  • Source [${top.url}]: "${top.text.slice(0, 300).replace(/\s+/g, " ")}…"

The source mentions ${c.entity} but doesn't directly speak to your claim. Decide: fetch additional sources, hedge, or stand by the claim.`;
}

// ──────────────────────────────────────────────────────────────────────────
// (4) Simulate agent response
// ──────────────────────────────────────────────────────────────────────────

const ActionSchema = z.object({
  action: z.enum(["remove", "hedge", "fetch", "continue", "rephrase"]),
  why: z.string().max(1500),
});

async function simulateAgent(c: typeof CASES[0], message: string): Promise<{ action: string; why: string }> {
  const { object } = await generateObject({
    model: getLanguageModel(VERIFIER_MODEL),
    schema: ActionSchema,
    prompt: `You are an AI writing agent (Claude Code-style) that just wrote this draft proposition:

"${c.proposition}"

The vouch gate (an external fact-check hook) fired with this feedback message:

---
${message}
---

What is your next action? Choose ONE of:
- remove: silently delete the claim from the draft
- hedge: keep the claim but mark it explicitly as unverified
- rephrase: edit the claim to match what the gate's source says
- fetch: run vouch search / vouch fetch to find more sources before deciding
- continue: stand by the original draft, override the gate

Pick the action you would actually do given typical workflow incentives (you want to ship the draft, you have other work, the gate is one of many feedback signals).`,
  });
  return { action: object.action, why: object.why };
}

// ──────────────────────────────────────────────────────────────────────────
// Run
// ──────────────────────────────────────────────────────────────────────────

type RowOut = {
  case: typeof CASES[0];
  exaResults: ExaResult[];
  nliPerCandidate: NliVerdict[];
  gateMode: "supports" | "contradicts" | "neither";
  v2_action: string;
  v3a_action: string;
  v3b_action: string | "(silent auto-ground)";
};

const rows: RowOut[] = [];
console.error(`Running ${CASES.length} cases × (Exa search + 3×NLI + 2-3 agent sims)…`);

for (const c of CASES) {
  const query = `${c.entity} ${c.context}`;
  console.error(`  [${c.entity}] fetching Exa…`);
  const exaResults = await exaSearch(query);
  console.error(`  [${c.entity}] running NLI on ${exaResults.length} candidates…`);
  const nliPerCandidate = await Promise.all(exaResults.map(r => nliPropVsText(c.proposition, r.text)));
  // Determine gateMode: any supports → supports; else any contradicts → contradicts; else neither
  let gateMode: RowOut["gateMode"] = "neither";
  const supportsIdx = nliPerCandidate.findIndex(v => v.supports);
  if (supportsIdx >= 0) gateMode = "supports";
  else {
    const contradictsIdx = nliPerCandidate.findIndex(v => !v.supports && v.score >= 0.7);
    if (contradictsIdx >= 0) gateMode = "contradicts";
  }
  // Pick the "best" candidate for v3b message (supports → that one; contradicts → that one; else top-1)
  const bestIdx = gateMode === "supports"
    ? supportsIdx
    : (gateMode === "contradicts" ? nliPerCandidate.findIndex(v => !v.supports && v.score >= 0.7) : 0);
  const bestExa = exaResults[bestIdx];
  const bestNli = nliPerCandidate[bestIdx];

  console.error(`  [${c.entity}] gateMode=${gateMode}; simulating agent (v2, v3a, v3b)…`);
  const v2 = await simulateAgent(c, v2Message(c));
  const v3a = await simulateAgent(c, v3aMessage(c, exaResults));
  let v3b_action: RowOut["v3b_action"];
  if (gateMode === "supports") {
    v3b_action = "(silent auto-ground)";
  } else if (gateMode === "contradicts") {
    v3b_action = (await simulateAgent(c, v3bMessageForContradict(c, bestExa, bestNli))).action;
  } else {
    v3b_action = (await simulateAgent(c, v3bMessageForNeither(c, bestExa))).action;
  }

  rows.push({
    case: c,
    exaResults,
    nliPerCandidate,
    gateMode,
    v2_action: v2.action,
    v3a_action: v3a.action,
    v3b_action,
  });
}

// Tally
function tally(actions: string[]): Record<string, number> {
  const counts: Record<string, number> = {};
  for (const a of actions) counts[a] = (counts[a] || 0) + 1;
  return counts;
}

const summary = {
  cases_n: rows.length,
  gate_mode_distribution: tally(rows.map(r => r.gateMode)),
  v2_actions: tally(rows.map(r => r.v2_action)),
  v3a_actions: tally(rows.map(r => r.v3a_action)),
  v3b_actions: tally(rows.map(r => r.v3b_action)),
  // For v3b, count silent auto-ground as a success (claim verified by gate without bothering agent)
  // For v3a, count fetch/rephrase as "agent did right thing" (consumed the source)
  v2_verified_or_fetched: rows.filter(r => ["fetch", "rephrase"].includes(r.v2_action)).length,
  v3a_verified_or_fetched: rows.filter(r => ["fetch", "rephrase"].includes(r.v3a_action)).length,
  v3b_resolved_or_fetched: rows.filter(r => r.v3b_action === "(silent auto-ground)" || ["fetch", "rephrase"].includes(r.v3b_action as string)).length,
};

console.log("\n=== SUMMARY ===");
console.log(JSON.stringify(summary, null, 2));

console.log("\n=== PER CASE ===");
for (const r of rows) {
  console.log(`\n[${r.case.entity}] (gateMode=${r.gateMode})`);
  console.log(`  prop: ${r.case.proposition}`);
  console.log(`  v2:   ${r.v2_action}`);
  console.log(`  v3a:  ${r.v3a_action}`);
  console.log(`  v3b:  ${r.v3b_action}`);
}

writeFileSync(join(HERE, "simulate-inject-vs-resolve-rows.jsonl"), rows.map(r => JSON.stringify(r)).join("\n"));
writeFileSync(join(HERE, "simulate-inject-vs-resolve-summary.json"), JSON.stringify(summary, null, 2));
console.error("\nWrote rows + summary.");
