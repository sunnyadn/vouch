// Two-stage replay: KIMI as the high-recall DETECTOR, a single-shot adversarial VERIFIER
// as the precision stage — measured offline on the deepseek-eval gold.
//
// Why: measured tradeoff (handoff 2026-06-09) — kimi recall ≫ deepseek but kimi false-fires
// on grounded controls (coding controls 1/5). Prompt-tuning the detector was inert; the
// candidate fix is STRUCTURAL: verify each fired issue independently before surfacing it.
// This bench answers, on existing gold, the make-or-break question for that path:
//   FP-KILL: of detector fires on NOFIRE controls, how many does the verifier reject?
//   TP-SURVIVAL: of detector fires on FIRE cases, how many does the verifier uphold?
// Both deepseek and kimi run as the verifier (round-18 lesson: don't assume cross-model
// is better — kimi-as-checker beat deepseek 4/4 vs 1/4 on the buried overreach).
//
// Run:  bun bench/verify-replay/run.ts             (REPS=2 detector, VREPS=2 verifier)
//       REPS=3 VREPS=3 bun bench/verify-replay/run.ts

import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import type { ReviewIssue } from "../../src/core/reviewer.ts";
import { verifyFlags } from "./verifier.ts";
import { type Case, CASES } from "../deepseek-eval/cases.ts";

const ROOT = join(import.meta.dir, "..", "..");
const envFile = readFileSync(join(ROOT, ".env"), "utf8");
const envOf = (k: string) => envFile.match(new RegExp(`^${k}=(.*)$`, "m"))?.[1] ?? "";

const KIMI = { name: "kimi", apiKey: envOf("KIMI_API_KEY"), baseURL: envOf("KIMI_BASE_URL"), model: envOf("KIMI_MODEL") };
const DEEPSEEK = { name: "deepseek", apiKey: envOf("DEEPSEEK_API_KEY"), baseURL: envOf("DEEPSEEK_BASE_URL"), model: envOf("DEEPSEEK_MODEL") };
if (!KIMI.apiKey || !DEEPSEEK.apiKey) {
  console.error("need KIMI_* and ANTHROPIC_* creds in .env");
  process.exit(1);
}

// ---- Stage 1: kimi detector = the DEPLOYED agentic reviewer, env flipped to kimi ----
process.env.ANTHROPIC_API_KEY = KIMI.apiKey;
process.env.ANTHROPIC_BASE_URL = KIMI.baseURL;
process.env.VOUCH_REVIEWER_MODEL = KIMI.model;
const { anthropicReviewerAgentic } = await import("../../src/core/reviewer-agentic.ts");

const REPS = Number(process.env.REPS ?? 2);
const VREPS = Number(process.env.VREPS ?? 2);

// ---- Stage 2: single-shot verifier — shared module (see verifier.ts) ----
// The gold traces are tiny (0-6 events) so the complete history fits in the prompt; the
// production form would reuse query_history, but for path-validation inline is exact.

async function verify(model: typeof KIMI, c: Case, issues: ReviewIssue[]) {
  return verifyFlags({ name: model.name, apiKey: model.apiKey, baseURL: model.baseURL, model: model.model }, { action: c.action, events: c.events }, issues);
}

// ---- run ----
console.log(`verify-replay — detector=${KIMI.model} REPS=${REPS}, verifiers={${DEEPSEEK.model}, ${KIMI.model}} VREPS=${VREPS}\n`);

interface RepResult { fired: boolean; issues: ReviewIssue[]; uphold: Record<string, boolean | null> } // verifier name → majority uphold
interface CaseResult { c: Case; reps: RepResult[]; failOpens: number }
const results: CaseResult[] = [];

for (const c of CASES) {
  const reps: RepResult[] = [];
  let failOpens = 0;
  for (let i = 0; i < REPS; i++) {
    let v = await anthropicReviewerAgentic({ action: c.action, actionType: "stop-response", events: c.events, projectFindings: [] });
    for (let r = 0; v.status === "failed" && r < 3; r++) {
      failOpens++;
      await new Promise((res) => setTimeout(res, 2500 * (r + 1)));
      v = await anthropicReviewerAgentic({ action: c.action, actionType: "stop-response", events: c.events, projectFindings: [] });
    }
    if (v.status === "failed") continue; // dead rep — excluded from the denominator
    const rep: RepResult = { fired: v.issues.length > 0, issues: v.issues, uphold: {} };
    if (rep.fired) {
      for (const verifier of [DEEPSEEK, KIMI]) {
        let yes = 0, valid = 0;
        for (let k = 0; k < VREPS; k++) {
          const verdict = await verify(verifier, c, v.issues);
          if (verdict === null) continue;
          valid++;
          if (verdict.upheld) yes++;
        }
        rep.uphold[verifier.name] = valid === 0 ? null : yes * 2 > valid;
      }
    }
    reps.push(rep);
  }
  results.push({ c, reps, failOpens });

  const fired = reps.filter((r) => r.fired).length;
  const up = (name: string) => reps.filter((r) => r.fired).map((r) => (r.uphold[name] === null ? "?" : r.uphold[name] ? "U" : "k")).join("") || "-";
  console.log(
    `[${c.expect}] ${c.id}: kimi fired ${fired}/${reps.length}${failOpens ? ` (${failOpens} fail-open)` : ""}` +
      `  verify→ ds:${up("deepseek")} kimi:${up("kimi")}   (U=upheld k=killed)`,
  );
}

// ---- aggregate ----
// Rep-level pipeline verdict: fired AND verifier-upheld. Case verdict: majority over valid reps.
function pipelineScore(verifierName: string | null) {
  let pass = 0, total = 0;
  const rows: string[] = [];
  for (const { c, reps } of results) {
    if (reps.length === 0) continue;
    total++;
    const fires = reps.filter((r) => r.fired && (verifierName === null || r.uphold[verifierName] === true)).length;
    const majorityFire = fires * 2 > reps.length;
    const ok = (c.expect === "FIRE") === majorityFire;
    if (ok) pass++;
    rows.push(`  ${ok ? "✅" : "❌"} ${c.id}: want ${c.expect}, pipeline fired ${fires}/${reps.length}`);
  }
  return { pass, total, rows };
}

// Flag-level stats: of all fired reps on NOFIRE cases (false alarms), how many killed; on FIRE cases, how many upheld.
function flagStats(verifierName: string) {
  let fpFired = 0, fpKilled = 0, tpFired = 0, tpUpheld = 0;
  for (const { c, reps } of results)
    for (const r of reps) {
      if (!r.fired || r.uphold[verifierName] === null) continue;
      if (c.expect === "NOFIRE") { fpFired++; if (!r.uphold[verifierName]) fpKilled++; }
      else { tpFired++; if (r.uphold[verifierName]) tpUpheld++; }
    }
  return { fpFired, fpKilled, tpFired, tpUpheld };
}

for (const name of [null, "deepseek", "kimi"] as const) {
  const label = name === null ? "kimi SOLO (no verifier — baseline)" : `kimi + ${name}-verifier`;
  const { pass, total, rows } = pipelineScore(name);
  console.log(`\n--- ${label}: ${pass}/${total} cases ---`);
  for (const r of rows) console.log(r);
  if (name !== null) {
    const s = flagStats(name);
    console.log(`  flag-level: FP-kill ${s.fpKilled}/${s.fpFired} (false alarms killed) · TP-survival ${s.tpUpheld}/${s.tpFired} (real catches kept)`);
  }
}

writeFileSync("/tmp/verify-replay-results.json", JSON.stringify(results, null, 2));
console.log("\nraw results → /tmp/verify-replay-results.json");
