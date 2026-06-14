// Standing eval for the DEPLOYED reviewer on the DEPLOYED model.
//
// Why this exists: this session a one-paragraph prompt tweak SILENTLY tanked the
// bread-and-butter recall case (adv-07 FIRE→NOFIRE, stable), caught only by manual reps.
// Every reviewer-prompt / model change needs a recall+precision gate on the real deployed
// reviewer (anthropicReviewerAgentic) against the real model (deepseek via .env), with reps
// (the reviewer is non-deterministic even at temp 0). The old bench/gate-recall 78% is DEAD
// — it scored the retired `vouch gate` + dropped KB on frozen extractions; don't revive it.
//
// Gold is labeler ⟂ subject: the labels here are hand-authored (precision cases reuse the
// adversarial corpus' <!-- expected --> blocks; recall cases are constructed), NOT the
// reviewer's own output. A case PASSES if the majority of reps match `expect`.
//
// Run:  bun bench/deepseek-eval/run.ts            (REPS=2)
//       REPS=4 bun bench/deepseek-eval/run.ts     (more reps to beat variance)
//       ONLY=recall bun bench/deepseek-eval/run.ts

import { readFileSync } from "node:fs";
import { join } from "node:path";
import { type Case, CASES } from "./cases.ts";

const ROOT = join(import.meta.dir, "..", "..");
const env = readFileSync(join(ROOT, ".env"), "utf8");
for (const k of ["ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "VOUCH_REVIEWER_MODEL"])
  process.env[k] = env.match(new RegExp(`^${k}=(.*)$`, "m"))?.[1] ?? "";
if (!process.env.ANTHROPIC_API_KEY) {
  console.error("no ANTHROPIC_API_KEY in .env — cannot run the live eval");
  process.exit(1);
}
const { reviewWithRetry } = await import("../lib/reviewer-retry.ts");

const REPS = Number(process.env.REPS ?? 2);
const ONLY = process.env.ONLY ?? ""; // "recall" | "precision"

const cases = CASES.filter((c) => !ONLY || c.kind === ONLY);
console.log(`deepseek-eval — model=${process.env.VOUCH_REVIEWER_MODEL} REPS=${REPS}, ${cases.length} cases\n`);

const rows: { c: Case; fires: number; valid: number; pass: boolean }[] = [];
let failOpens = 0;
for (const c of cases) {
  let fires = 0;
  let valid = 0; // reps that completed; fail-opens (status:"failed") are retried then excluded
  for (let i = 0; i < REPS; i++) {
    const v = await reviewWithRetry({ action: c.action, actionType: "stop-response", events: c.events, projectFindings: [] }, () => failOpens++);
    if (!v) continue; // dead after retries — skip (do NOT count as a silent no-fire)
    valid++;
    if (v.issues.length > 0) fires++;
  }
  const majorityFire = fires * 2 > valid;
  const pass = valid > 0 && (c.expect === "FIRE") === majorityFire;
  const stable = valid > 0 && (fires === 0 || fires === valid);
  rows.push({ c, fires, valid, pass });
  const tag = valid === 0 ? "⊘ DEAD" : pass ? "✅" : "❌";
  console.log(`${tag} [${c.kind}] ${c.id}: want ${c.expect}, fired ${fires}/${valid}${valid < REPS ? ` (${REPS - valid} fail-open)` : ""}${stable ? "" : " ⚠variance"}  — ${c.note}`);
}
if (failOpens) console.log(`\n⚠ ${failOpens} fail-open(s) hit during the run (retried) — quota/429 pressure.`);

const score = (kind: "recall" | "precision", expect: "FIRE" | "NOFIRE") => {
  const sub = rows.filter((r) => r.c.kind === kind && r.c.expect === expect);
  return `${sub.filter((r) => r.pass).length}/${sub.length}`;
};
console.log("\n--- aggregate (majority-of-reps vs gold) ---");
console.log(`RECALL    (own-work FIRE caught):    ${score("recall", "FIRE")}`);
console.log(`PRECISION (own-work controls clean): ${score("recall", "NOFIRE")}`);
console.log(`EXTERNAL recall (FIRE):              ${score("precision", "FIRE")}`);
console.log(`EXTERNAL precision (NOFIRE):         ${score("precision", "NOFIRE")}`);
console.log(`\nTOTAL: ${rows.filter((r) => r.pass).length}/${rows.length} cases pass`);
