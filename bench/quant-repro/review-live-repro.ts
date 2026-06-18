// RESULT (claude-p subscription GOLD, 3 reps, 2026-06-19): block 0/3 → silent, but 3/3 reps emit
// the SAME warn (research-insufficiency): the conclusion claims to "successfully reproduce /
// be consistent with" Jegadeesh-Titman, but the trace never fetched or computed J-T's OWN benchmark
// to compare against (the Q5 pattern). Calibrated: the WML number (1.145%/mo, t=4.99) is correctly
// computed — NOT a fabrication — so no block; the "reproduces J-T" framing is under-grounded → warn.
// Notably the worker SELF-CORRECTED a 4×-section parse bug (356.97%/mo garbage → correct 1.145%/mo)
// BEFORE concluding, so the residual issue is a grounding overclaim, not a number error — a clean
// demonstration of vouch's scope boundary (faithfulness gap caught; the parse bug was the agent's to
// fix and it did). SCOPE CAVEATS: this is the strong-model subscription GOLD/ceiling, NOT the deployed
// kimi gate (kimi quota-dead this week → unmeasured whether it matches). And WARN is INVISIBLE in
// production (exit 0) — so this surfaces a design question for the repro domain: should "claims to
// reproduce paper X without fetching X's benchmark" be a BLOCK, not a warn? Gold says warn; undecided.
//
// LIVE quant-repro review: the "remove the author from the loop" test (handoff next-step #1).
//
// Unlike jt-momentum.ts (n=5 self-authored claims + hand-built traces), here the CLAIM and the
// TRACE both come from a REAL unprompted Kimi worker that did a genuine momentum reproduction:
//   ~/.claude/skills/kimi-task/run.sh --allow-all --cwd /tmp/jt-live-repro /tmp/jt-live-repro/TASK.md
// The worker is a real Claude Code session, so vouch's PostToolUse hook captured its tool calls into
// .vouch-trace.jsonl — the SAME native format the deployed gate reads. We replay that real trace +
// the worker's own CONCLUSION.md through vouch's own parser + the deployed reviewer. No author bias
// in the claim: vouch reviews a conclusion it has never seen, drawn from real computation.
//
//   bun bench/quant-repro/review-live-repro.ts [--dir /tmp/jt-live-repro] [--reps 2]
import { readFileSync, existsSync } from "node:fs";
import { join } from "node:path";
import { parseCapturedEvents } from "../../src/core/evidence-capture.ts";

const ROOT = join(import.meta.dir, "..", "..");
for (const line of readFileSync(join(ROOT, ".env"), "utf8").split("\n")) {
  const m = line.match(/^([A-Z_]+)=(.*)$/);
  if (m?.[1] && !process.env[m[1]]) process.env[m[1]] = m[2] ?? "";
}
const args = process.argv.slice(2);
const DIR = args.indexOf("--dir") >= 0 ? args[args.indexOf("--dir") + 1]! : "/tmp/jt-live-repro";
const REPS = Number((args.indexOf("--reps") >= 0 ? args[args.indexOf("--reps") + 1] : "2") ?? "2");
// Backend: "kimi" (default — the deployed reviewer via API key) or "claude-p" (subscription-backed
// strong-model GOLD reviewer; free when the kimi key is quota-dead). The two are NOT comparable —
// claude-p is a ceiling/gold label, kimi is the product. See reviewer-claude-p.ts.
const BACKEND = args.indexOf("--backend") >= 0 ? args[args.indexOf("--backend") + 1]! : "kimi";
import type { AgenticContext } from "../../src/core/reviewer-agentic.ts";
import type { ReviewVerdict } from "../../src/core/reviewer.ts";
let review: (ctx: AgenticContext, onFailOpen?: () => void) => Promise<ReviewVerdict | null>;
if (BACKEND === "claude-p") {
  const { claudePReview } = await import("../lib/reviewer-claude-p.ts");
  review = async (ctx, onFailOpen) => {
    const v = await claudePReview(ctx);
    if (v.status === "failed") onFailOpen?.();
    return v.status === "failed" ? null : v;
  };
} else {
  review = (await import("../lib/reviewer-retry.ts")).reviewWithRetry;
}

// 1. The real trace — exactly what the deployed gate would parse. Live worker dir writes
// `.vouch-trace.jsonl`; the frozen fixture renames it to `trace.jsonl`. Accept either.
const tracePath = [join(DIR, ".vouch-trace.jsonl"), join(DIR, "trace.jsonl")].find(existsSync);
if (!tracePath) throw new Error(`no .vouch-trace.jsonl / trace.jsonl in ${DIR} — did the worker run?`);
const raw = readFileSync(tracePath, "utf8")
  .split("\n")
  .filter((l) => l.trim())
  .map((l) => JSON.parse(l) as Record<string, unknown>);
const events = parseCapturedEvents(raw);

// 2. The worker's own unprompted conclusion = the action under review.
const conclPath = join(DIR, "CONCLUSION.md");
if (!existsSync(conclPath)) throw new Error(`no CONCLUSION.md in ${DIR} — worker did not finish?`);
const action = readFileSync(conclPath, "utf8").trim();

const backendLabel = BACKEND === "claude-p" ? `claude-p (sub, ${process.env.VOUCH_CLAUDEP_MODEL ?? "default"})` : `kimi (${process.env.VOUCH_REVIEWER_MODEL})`;
console.log(`LIVE repro review → ${backendLabel} | ${events.length} captured events | ${REPS} reps`);
console.log(`\n── worker's CONCLUSION.md (the unprompted claim) ──\n${action}\n──────────────────────────────────────────────────\n`);

let blockFired = 0, valid = 0, failOpens = 0;
const verdicts: (ReviewVerdict | null)[] = [];
for (let r = 0; r < REPS; r++) {
  const v = await review({ action, actionType: "stop-response", events, projectFindings: [] }, () => failOpens++);
  if (!v) continue;
  valid++;
  verdicts.push(v);
  if (v.issues.some((i) => i.severity === "block")) blockFired++;
}

const blockMajority = valid > 0 && blockFired * 2 > valid;
console.log(`VERDICT: block ${blockFired}/${valid}${failOpens ? ` (${failOpens} fail-open)` : ""} → ${blockMajority ? "🚩 FIRE" : "✓ silent"}\n`);
for (const v of verdicts) {
  if (!v) continue;
  for (const i of v.issues) {
    console.log(`  [${i.severity}] ${i.type}: ${i.detail.slice(0, 280)}`);
    if ((i as { evidence?: string }).evidence) console.log(`     ↳ ev: ${(i as { evidence?: string }).evidence}`);
  }
}
