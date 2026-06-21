// Eval ONE GLM model over the decision-audit + frame-reality cases, setting the reviewer model via
// process.env.VOUCH_REVIEWER_MODEL (which anthropicReviewerAgentic actually honors). This bypasses
// run.ts's envOf() which reads GLM_MODEL from the .env FILE — the reason a command-line
// `GLM_MODEL=glm-5.2` override silently did NOTHING (both bake-off arms ran the .env's glm-4.7).
//
// Run:  EVAL_MODEL=glm-5.2 bun bench/decision-audit/eval-model.ts
//       EVAL_MODEL=glm-5.2 LIMIT=14 REPS=2 bun bench/decision-audit/eval-model.ts

import { readFileSync } from "node:fs";
import { join } from "node:path";
import { CASES } from "./cases.ts";
import { FR_CASES } from "./frame-reality-cases.ts";

const ROOT = join(import.meta.dir, "..", "..");
const env = readFileSync(join(ROOT, ".env"), "utf8");
const o = (k: string) => process.env[k] || (env.match(new RegExp(`^${k}=(.*)$`, "m"))?.[1] ?? "");

const MODEL = process.env.EVAL_MODEL ?? "glm-5.2";
process.env.ANTHROPIC_API_KEY = o("GLM_API_KEY");
process.env.ANTHROPIC_BASE_URL = o("GLM_BASE_URL");
process.env.VOUCH_REVIEWER_MODEL = MODEL; // <-- the override that actually takes effect
delete process.env.VOUCH_REVIEWER_PROMPT_EXTRA;

const REPS = Number(process.env.REPS ?? 2);
const LIMIT = Number(process.env.LIMIT ?? 14);
const { reviewWithRetry } = await import("../lib/reviewer-retry.ts");

async function evalSet(name: string, cases: { id: string; expect: string; action: string; events: unknown[] }[]) {
  console.log(`\n══ ${MODEL} × ${name} (REPS=${REPS}) ══`);
  let rHit = 0, rTot = 0, pHit = 0, pTot = 0, failOpens = 0;
  for (const c of cases) {
    let block = 0, valid = 0;
    const types = new Set<string>();
    for (let i = 0; i < REPS; i++) {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const v = await reviewWithRetry({ action: c.action, actionType: "stop-response", events: c.events as any, projectFindings: [] }, () => failOpens++);
      if (!v) continue;
      valid++;
      if (v.issues.some((iss) => iss.severity === "block")) block++;
      for (const iss of v.issues) types.add(iss.type);
    }
    const blockMaj = valid > 0 && block * 2 > valid;
    const pass = valid > 0 && (c.expect === "FIRE") === blockMaj;
    if (c.expect === "FIRE") { rTot++; if (pass) rHit++; } else { pTot++; if (pass) pHit++; }
    console.log(`${valid === 0 ? "⊘DEAD" : pass ? "✅" : "❌"} [${c.expect}] ${c.id}: block ${block}/${valid}  {${[...types].join(",") || "none"}}`);
  }
  if (failOpens) console.log(`⚠ ${failOpens} fail-open(s)`);
  console.log(`  BLOCK RECALL ${rHit}/${rTot}   PRECISION ${pHit}/${pTot}`);
  return { rHit, rTot, pHit, pTot };
}

await evalSet("decision-audit", CASES.slice(0, LIMIT));
await evalSet("frame-reality", FR_CASES);
