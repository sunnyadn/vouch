// Measure the CURRENT single-stage reviewer on the decision-audit gold — both models, no
// two-stage, no prompt change yet. Step 1 of the 2026-06-12 reframe: find the real gap on the
// reasoning/attribution failure class BEFORE touching anything. The reviewer already has a
// FALSIFIABILITY / premise-unexamined clause, so part of this class may already be covered —
// this run quantifies how much, and where the hole is.
//
// Runs both deepseek (deployed) and kimi by flipping the 3 env vars between calls (the reviewer
// reads apiKey/model/baseURL at call time). Fail-open-aware (c4d002e parity): status:"failed"
// reps retry with backoff and are excluded from the denominator, never counted as no-fire.
//
// GATE-BACKEND BAKE-OFF: this is the台子 to vet ANY candidate gate backend on the SAME gold before
// a live swap (the brittle-reviewer rule: never swap the deployed reviewer without a recall+precision
// reps-eval). Direct-API backends (kimi/glm/deepseek) just need their creds in .env; "claude-p" is the
// subscription GOLD/ceiling backend (no api key, strong model, free) — useful as a reference upper bound,
// not a like-for-like for the deployed direct-API gate.
//
// Run:  bun bench/decision-audit/run.ts                    (REPS=2, deepseek+kimi)
//       REPS=4 bun bench/decision-audit/run.ts             (beat variance)
//       MODELS=kimi bun bench/decision-audit/run.ts        (one backend only)
//       MODELS=kimi,glm bun bench/decision-audit/run.ts    (bake-off two candidate keys)
//       MODELS=claude-p bun bench/decision-audit/run.ts    (subscription gold/ceiling, free)

import { readFileSync } from "node:fs";
import { join } from "node:path";
import { verifyMajority } from "../verify-replay/verifier.ts";
import { filterProcessNarrationFires } from "../../src/core/process-narration.ts";
import { type Case, CASES } from "./cases.ts";

const ROOT = join(import.meta.dir, "..", "..");
const envFile = readFileSync(join(ROOT, ".env"), "utf8");
const envOf = (k: string) => envFile.match(new RegExp(`^${k}=(.*)$`, "m"))?.[1] ?? "";

// Direct-API backends: a swap is just 3 env vars (apiKey/baseURL/model) read at call time. Drop
// a new backend's creds in .env (e.g. GLM_API_KEY/GLM_BASE_URL/GLM_MODEL) and it's comparable here.
// The special "claude-p" backend is NOT in this map — it routes through the subscription adapter
// (no api key) instead of reviewWithRetry; see runModel. So this台子 compares ANY candidate gate
// backend (kimi / glm / a cheaper Anthropic key / subscription) on the SAME gold before a live swap.
const MODELS_DEF: Record<string, { apiKey: string; baseURL: string; model: string }> = {
  deepseek: { apiKey: envOf("DEEPSEEK_API_KEY"), baseURL: envOf("DEEPSEEK_BASE_URL"), model: envOf("DEEPSEEK_MODEL") },
  kimi: { apiKey: envOf("KIMI_API_KEY"), baseURL: envOf("KIMI_BASE_URL"), model: envOf("KIMI_MODEL") },
  glm: { apiKey: envOf("GLM_API_KEY"), baseURL: envOf("GLM_BASE_URL"), model: envOf("GLM_MODEL") },
};
const which = (process.env.MODELS ?? "deepseek,kimi").split(",").map((s) => s.trim());

const { reviewWithRetry } = await import("../lib/reviewer-retry.ts");
const REPS = Number(process.env.REPS ?? 2);
const VERIFY = process.env.VERIFY === "1"; // two-stage: re-judge each fire with a same-model verifier
const VREPS = Number(process.env.VREPS ?? 2);

// `fires` = reps with ANY issue (warn or block); `blockFires` = reps with a BLOCK-severity issue.
// The deployed gate only HALTS on block (warn is advisory, exit 0), so block-level precision/recall
// is the deployment-relevant number — the all-issue count scores a helpful advisory warn as a "fire"
// and overstates the cry-wolf rate (2026-06-16 finding: warn vs block split the 7/13 picture).
interface Row { c: Case; fires: number; blockFires: number; filtBlockFires: number; valid: number; pass: boolean; blockPass: boolean; filtBlockPass: boolean; stable: boolean; types: Set<string> }

async function runModel(name: string): Promise<{ rows: Row[]; failOpens: number }> {
  // "claude-p" = subscription-backed reviewer (no api key); see reviewer-claude-p.ts. It's a GOLD/
  // ceiling backend, NOT directly comparable to the deployed direct-API gate (different harness) —
  // but it answers "what does a strong reviewer do on this gold" for free. Two-stage VERIFY is API-
  // only (needs verifier creds), so it's skipped for claude-p.
  const isSub = name === "claude-p";
  let review: typeof reviewWithRetry;
  let verifierCreds: { apiKey: string; baseURL: string; model: string } | null = null;
  if (isSub) {
    const { claudePReview } = await import("../lib/reviewer-claude-p.ts");
    review = async (ctx, onFailOpen) => {
      const v = await claudePReview(ctx);
      if (v.status === "failed") onFailOpen?.();
      return v.status === "failed" ? null : v;
    };
    console.log(`\n══ claude-p (subscription, ${process.env.VOUCH_CLAUDEP_MODEL ?? "default"}) — REPS=${REPS} ══`);
  } else {
    const m = MODELS_DEF[name];
    if (!m?.apiKey) {
      console.log(`\n⊘ ${name}: no creds in .env — skipped\n`);
      return { rows: [], failOpens: 0 };
    }
    process.env.ANTHROPIC_API_KEY = m.apiKey;
    process.env.ANTHROPIC_BASE_URL = m.baseURL;
    process.env.VOUCH_REVIEWER_MODEL = m.model;
    review = reviewWithRetry;
    verifierCreds = { apiKey: m.apiKey, baseURL: m.baseURL, model: m.model };
    console.log(`\n══ ${name} (${m.model}) @ ${m.baseURL} — REPS=${REPS}${VERIFY ? ` + two-stage self-verify ×${VREPS}` : ""} ══`);
  }
  const rows: Row[] = [];
  let failOpens = 0;
  // LIMIT caps the case count — for cheaply smoke-testing a new/slow backend (e.g. claude-p) before
  // committing to the full ~13-case × REPS bake-off. 0 = full corpus.
  const LIMIT = Number(process.env.LIMIT ?? 0);
  const cases = LIMIT > 0 ? CASES.slice(0, LIMIT) : CASES;
  for (const c of cases) {
    let fires = 0, blockFires = 0, filtBlockFires = 0, valid = 0;
    const types = new Set<string>();
    for (let i = 0; i < REPS; i++) {
      const v = await review({ action: c.action, actionType: "stop-response", events: c.events, projectFindings: [] }, () => failOpens++);
      if (!v) continue;
      valid++;
      if (v.issues.length > 0) {
        if (VERIFY && verifierCreds) {
          const verifier = { name: "self", ...verifierCreds };
          const upheld = await verifyMajority(verifier, { action: c.action, events: c.events }, v.issues, VREPS);
          if (upheld === false) continue; // stage-2 rejected every flag — rep does NOT fire
        }
        fires++;
        if (v.issues.some((iss) => iss.severity === "block")) blockFires++;
        for (const iss of v.issues) types.add(iss.type);
      }
      // PAIRED: the narration filter is deterministic given the verdict, so score the SAME verdict
      // with it applied — the raw-vs-filtered delta is PURE filter effect, zero run-to-run variance.
      const filtered = filterProcessNarrationFires(v);
      if (filtered.issues.some((iss) => iss.severity === "block")) filtBlockFires++;
    }
    const majorityFire = fires * 2 > valid;
    const blockMajority = blockFires * 2 > valid;
    const filtBlockMajority = filtBlockFires * 2 > valid;
    const pass = valid > 0 && (c.expect === "FIRE") === majorityFire;
    const blockPass = valid > 0 && (c.expect === "FIRE") === blockMajority;
    const filtBlockPass = valid > 0 && (c.expect === "FIRE") === filtBlockMajority;
    const stable = valid > 0 && (fires === 0 || fires === valid);
    rows.push({ c, fires, blockFires, filtBlockFires, valid, pass, blockPass, filtBlockPass, stable, types });
    const tag = valid === 0 ? "⊘ DEAD" : pass ? "✅" : "❌";
    const filtTag = filtBlockFires !== blockFires ? ` →filt block ${filtBlockFires}/${valid}` : "";
    console.log(
      `${tag} [${c.expect}] ${c.id}: fired ${fires}/${valid} (block ${blockFires}/${valid}${filtTag})${valid < REPS ? ` (${REPS - valid} fail-open)` : ""}${stable ? "" : " ⚠var"}` +
        `${types.size ? `  {${[...types].join(",")}}` : ""}`,
    );
  }
  if (failOpens) console.log(`⚠ ${failOpens} fail-open(s) (retried) — quota/429 pressure.`);

  const recall = rows.filter((r) => r.c.expect === "FIRE");
  const prec = rows.filter((r) => r.c.expect === "NOFIRE");
  const score = (rs: Row[]) => `${rs.filter((r) => r.pass).length}/${rs.length}`;
  const blockScore = (rs: Row[]) => `${rs.filter((r) => r.blockPass).length}/${rs.length}`;
  const filtScore = (rs: Row[]) => `${rs.filter((r) => r.filtBlockPass).length}/${rs.length}`;
  // All-issue (warn|block) AND block-only. Block is the deployment number — the gate halts only on
  // block; warn is advisory. Block-precision is the cry-wolf rate that actually interrupts the agent.
  console.log(`  ANY-ISSUE        RECALL ${score(recall)}   PRECISION ${score(prec)}   TOTAL ${score(rows)}`);
  console.log(`  BLOCK-ONLY       RECALL ${blockScore(recall)}   PRECISION ${blockScore(prec)}   TOTAL ${blockScore(rows)}   ← deployment-relevant (gate halts on block)`);
  // PAIRED filter delta (same verdicts, narration filter applied): the ONLY clean way to read the
  // filter's effect, since run-to-run precision swings ±2 from variance.
  console.log(`  +NARRATION-FILTER RECALL ${filtScore(recall)}   PRECISION ${filtScore(prec)}   TOTAL ${filtScore(rows)}   ← paired Δ vs BLOCK-ONLY (pure filter effect)`);
  return { rows, failOpens };
}

console.log(`decision-audit — ${CASES.length} cases (${CASES.filter((c) => c.expect === "FIRE").length} FIRE / ${CASES.filter((c) => c.expect === "NOFIRE").length} NOFIRE)`);
for (const name of which) await runModel(name);
