// Consensus validation (2026-06-15): does K-of-N majority consensus fix the DEPLOYED gate's
// variance-driven cry-wolf — and what's its CEILING? Prior consensus work was xtree-era (cross-model
// probe4–9 = out-of-sample refuted as human-replacement; same-model majority probe17 = "kills noisy
// false-flags"); never validated on the CURRENT vouch reviewer + CURRENT 29-case gold. This does that.
//
// KEY FRAMING: the deployed gate fires on a SINGLE review (N=1). The other benches score by
// majority-of-REPS already, so their precision is NOT the deployed experience. So we measure each
// case's true fire-RATE over N reps, then compare aggregation rules:
//   - N=1 deployed (EXPECTED): expected #controls-clean = Σ(1-fireRate); expected #fires-caught = Σ fireRate.
//   - K-of-N consensus: fire iff fires≥K. Sweep K=1 (any-fire, max recall) … N (unanimous, max precision).
// CEILING: consensus only moves VARIANCE cases (0<fireRate<1). STABLE fires (fireRate≈1) are IMMUNE —
// they fire under every K. So this also classifies each control as variance-FP (consensus-fixable) vs
// stable-FP (consensus-PROOF) — the latter caps how much precision consensus can buy.
//
// Run: REPS=8 bun bench/decision-audit/consensus-experiment.ts   (29×8 = 232 kimi reviews)

import { type Case, CASES } from "./cases.ts";
import { setModelEnv } from "../lib/models-config.ts";

setModelEnv("kimi"); // deployed prompt, no extra
const N = Number(process.env.REPS ?? 8);
const { reviewWithRetry } = await import("../lib/reviewer-retry.ts");

interface Row { c: Case; fires: number; valid: number }
const rows: Row[] = [];
let failOpens = 0;
for (const c of CASES) {
  let fires = 0, valid = 0;
  for (let i = 0; i < N; i++) {
    const v = await reviewWithRetry({ action: c.action, actionType: "stop-response", events: c.events, projectFindings: [] }, () => failOpens++);
    if (!v) continue;
    valid++;
    if (v.issues.length > 0) fires++;
  }
  rows.push({ c, fires, valid });
}

const FIRES = rows.filter((r) => r.c.expect === "FIRE");
const CTRL = rows.filter((r) => r.c.expect === "NOFIRE");
const rate = (r: Row) => (r.valid ? r.fires / r.valid : 0);

console.log(`consensus-experiment | ${CASES.length} cases (${FIRES.length} FIRE / ${CTRL.length} NOFIRE) | N=${N} reps | kimi`);
if (failOpens) console.log(`⚠ ${failOpens} fail-open(s) retried`);

// 1) per-case fire-rate, classified
const cls = (r: Row) => (r.fires === 0 ? "stable-clean" : r.fires === r.valid ? "STABLE" : "var");
console.log(`\n── per-case fire-rate (fires/valid) ──`);
for (const kind of ["FIRE", "NOFIRE"] as const) {
  console.log(`  [${kind}]`);
  for (const r of rows.filter((x) => x.c.expect === kind).sort((a, b) => rate(b) - rate(a))) {
    const want = kind === "FIRE" ? "fire" : "clean";
    const bad = (kind === "FIRE" && r.fires === 0) || (kind === "NOFIRE" && r.fires === r.valid);
    console.log(`    ${r.fires}/${r.valid}  ${(rate(r) * 100).toFixed(0).padStart(3)}%  ${r.c.id.padEnd(36)} ${cls(r)}${bad ? "  ← STABLE-WRONG (consensus-proof)" : ""}`);
  }
}

// 2) N=1 deployed (expected, fractional case-counts) vs K-of-N consensus sweep
const expClean = CTRL.reduce((s, r) => s + (1 - rate(r)), 0); // expected #controls a single review leaves clean
const expCatch = FIRES.reduce((s, r) => s + rate(r), 0);      // expected #fires a single review catches
console.log(`\n── aggregation rules (precision = controls clean /${CTRL.length}, recall = fires caught /${FIRES.length}) ──`);
console.log(`  N=1 deployed (expected): P ${expClean.toFixed(1)}/${CTRL.length}   R ${expCatch.toFixed(1)}/${FIRES.length}   [the CURRENT single-review gate]`);
for (let K = 1; K <= N; K++) {
  const prec = CTRL.filter((r) => r.fires < K).length;   // control stays clean iff fewer than K reps fire
  const rec = FIRES.filter((r) => r.fires >= K).length;  // fire caught iff ≥K reps fire
  const tag = K === 1 ? " any-fire (max recall)" : K === Math.ceil(N / 2) || K === Math.floor(N / 2) + 1 ? " ~majority" : K === N ? " unanimous (max precision)" : "";
  console.log(`  K=${K}/${N}: P ${prec}/${CTRL.length}  R ${rec}/${FIRES.length}  T ${prec + rec}/${CASES.length}${tag}`);
}

// 3) the ceiling
const stableFP = CTRL.filter((r) => r.fires === r.valid).map((r) => r.c.id);
console.log(`\n── CEILING: stable-FP controls (fire on ALL reps → consensus-PROOF, no K fixes them) ──`);
console.log(`  ${stableFP.length ? stableFP.join(", ") : "(none)"}`);
console.log(`  ⇒ max precision any consensus can reach ≈ ${CTRL.length - stableFP.length}/${CTRL.length} (the rest are variance-fixable).`);
