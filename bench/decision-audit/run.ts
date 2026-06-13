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
// Run:  bun bench/decision-audit/run.ts              (REPS=2, both models)
//       REPS=4 bun bench/decision-audit/run.ts       (beat variance)
//       MODELS=deepseek bun bench/decision-audit/run.ts   (one model only)

import { readFileSync } from "node:fs";
import { join } from "node:path";
import { verifyMajority } from "../verify-replay/verifier.ts";
import { type Case, CASES } from "./cases.ts";

const ROOT = join(import.meta.dir, "..", "..");
const envFile = readFileSync(join(ROOT, ".env"), "utf8");
const envOf = (k: string) => envFile.match(new RegExp(`^${k}=(.*)$`, "m"))?.[1] ?? "";

const MODELS_DEF: Record<string, { apiKey: string; baseURL: string; model: string }> = {
  deepseek: { apiKey: envOf("DEEPSEEK_API_KEY"), baseURL: envOf("DEEPSEEK_BASE_URL"), model: envOf("DEEPSEEK_MODEL") },
  kimi: { apiKey: envOf("KIMI_API_KEY"), baseURL: envOf("KIMI_BASE_URL"), model: envOf("KIMI_MODEL") },
};
const which = (process.env.MODELS ?? "deepseek,kimi").split(",").map((s) => s.trim());

const { anthropicReviewerAgentic } = await import("../../src/core/reviewer-agentic.ts");
const REPS = Number(process.env.REPS ?? 2);
const VERIFY = process.env.VERIFY === "1"; // two-stage: re-judge each fire with a same-model verifier
const VREPS = Number(process.env.VREPS ?? 2);

interface Row { c: Case; fires: number; valid: number; pass: boolean; stable: boolean; types: Set<string> }

async function runModel(name: string): Promise<{ rows: Row[]; failOpens: number }> {
  const m = MODELS_DEF[name];
  if (!m?.apiKey) {
    console.log(`\n⊘ ${name}: no creds in .env — skipped\n`);
    return { rows: [], failOpens: 0 };
  }
  process.env.ANTHROPIC_API_KEY = m.apiKey;
  process.env.ANTHROPIC_BASE_URL = m.baseURL;
  process.env.VOUCH_REVIEWER_MODEL = m.model;

  console.log(`\n══ ${name} (${m.model}) @ ${m.baseURL} — REPS=${REPS}${VERIFY ? ` + two-stage self-verify ×${VREPS}` : ""} ══`);
  const rows: Row[] = [];
  let failOpens = 0;
  for (const c of CASES) {
    let fires = 0, valid = 0;
    const types = new Set<string>();
    for (let i = 0; i < REPS; i++) {
      let v = await anthropicReviewerAgentic({ action: c.action, actionType: "stop-response", events: c.events, projectFindings: [] });
      for (let r = 0; v.status === "failed" && r < 3; r++) {
        failOpens++;
        await new Promise((res) => setTimeout(res, 2500 * (r + 1)));
        v = await anthropicReviewerAgentic({ action: c.action, actionType: "stop-response", events: c.events, projectFindings: [] });
      }
      if (v.status === "failed") continue;
      valid++;
      if (v.issues.length > 0) {
        if (VERIFY) {
          const verifier = { name: "self", apiKey: m.apiKey, baseURL: m.baseURL, model: m.model };
          const upheld = await verifyMajority(verifier, { action: c.action, events: c.events }, v.issues, VREPS);
          if (upheld === false) continue; // stage-2 rejected every flag — rep does NOT fire
        }
        fires++;
        for (const iss of v.issues) types.add(iss.type);
      }
    }
    const majorityFire = fires * 2 > valid;
    const pass = valid > 0 && (c.expect === "FIRE") === majorityFire;
    const stable = valid > 0 && (fires === 0 || fires === valid);
    rows.push({ c, fires, valid, pass, stable, types });
    const tag = valid === 0 ? "⊘ DEAD" : pass ? "✅" : "❌";
    console.log(
      `${tag} [${c.expect}] ${c.id}: fired ${fires}/${valid}${valid < REPS ? ` (${REPS - valid} fail-open)` : ""}${stable ? "" : " ⚠var"}` +
        `${types.size ? `  {${[...types].join(",")}}` : ""}`,
    );
  }
  if (failOpens) console.log(`⚠ ${failOpens} fail-open(s) (retried) — quota/429 pressure.`);

  const recall = rows.filter((r) => r.c.expect === "FIRE");
  const prec = rows.filter((r) => r.c.expect === "NOFIRE");
  const score = (rs: Row[]) => `${rs.filter((r) => r.pass).length}/${rs.length}`;
  console.log(`  RECALL (catch overreach): ${score(recall)}   PRECISION (controls clean): ${score(prec)}   TOTAL ${score(rows)}`);
  return { rows, failOpens };
}

console.log(`decision-audit — ${CASES.length} cases (${CASES.filter((c) => c.expect === "FIRE").length} FIRE / ${CASES.filter((c) => c.expect === "NOFIRE").length} NOFIRE)`);
for (const name of which) await runModel(name);
