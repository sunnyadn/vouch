// GO/NO-GO probe: run the frame→reality cases through the CURRENT deployed reviewer (no prompt
// clause) and see whether the existing 6 dimensions catch them. A MISS on the FIRE cases proves
// frame→reality is a real residue worth an A/B clause; a catch (esp. by research-insufficiency)
// means it collapses into an existing dimension → kill it.
//
// Run:  bun bench/decision-audit/frame-reality-probe.ts                 (REPS=3, deepseek+kimi)
//       MODELS=kimi REPS=4 bun bench/decision-audit/frame-reality-probe.ts
//
// Prints the firing dimension TYPES per case — so we see not just "did it fire" but "via WHICH
// dimension" (the redundancy check: if research-insufficiency catches it, it's not a new class).

import { readFileSync } from "node:fs";
import { join } from "node:path";
import { FR_CASES } from "./frame-reality-cases.ts";
import type { AgenticContext } from "../../src/core/reviewer-agentic.ts";
import type { ReviewVerdict } from "../../src/core/reviewer.ts";

const ROOT = join(import.meta.dir, "..", "..");
const envFile = readFileSync(join(ROOT, ".env"), "utf8");
// Prefer process.env (bun loads .env into it; a command-line override lands here) then fall back to
// the file — so `GLM_MODEL=…` overrides actually take effect (the file-only read ignored them).
const envOf = (k: string) => process.env[k] || (envFile.match(new RegExp(`^${k}=(.*)$`, "m"))?.[1] ?? "");

const MODELS_DEF: Record<string, { apiKey: string; baseURL: string; model: string }> = {
  deepseek: { apiKey: envOf("DEEPSEEK_API_KEY"), baseURL: envOf("DEEPSEEK_BASE_URL"), model: envOf("DEEPSEEK_MODEL") },
  kimi: { apiKey: envOf("KIMI_API_KEY"), baseURL: envOf("KIMI_BASE_URL"), model: envOf("KIMI_MODEL") },
  glm: { apiKey: envOf("GLM_API_KEY"), baseURL: envOf("GLM_BASE_URL"), model: envOf("GLM_MODEL") },
};
const which = (process.env.MODELS ?? "deepseek,kimi").split(",").map((s) => s.trim());
const REPS = Number(process.env.REPS ?? 3);

const { reviewWithRetry } = await import("../lib/reviewer-retry.ts");

async function runModel(name: string) {
  // Ensure NO experimental clause — we are measuring the DEPLOYED prompt/criteria as-is.
  delete process.env.VOUCH_REVIEWER_PROMPT_EXTRA;

  // "claude-p" = free subscription ceiling backend (same criteria, one-shot harness). Used as the
  // LIVE reviewer when the direct-API keys are quota-dead. NOT the deployed gate (different model +
  // harness) — but a prompt-level GO/NO-GO: if even the strong ceiling misses with the SAME
  // criteria, the hole is in the CRITERIA, not the model.
  const isSub = name === "claude-p";
  let review: (ctx: AgenticContext, onFailOpen: () => void) => Promise<ReviewVerdict | null>;
  if (isSub) {
    const { claudePReview } = await import("../lib/reviewer-claude-p.ts");
    review = async (ctx, onFailOpen) => {
      const v = await claudePReview(ctx);
      if (v.status === "failed") { onFailOpen(); return null; }
      return v;
    };
    console.log(`\n══ claude-p (subscription ${process.env.VOUCH_CLAUDEP_MODEL ?? "default"}) — REPS=${REPS}, deployed criteria (no clause) ══`);
  } else {
    const m = MODELS_DEF[name];
    if (!m?.apiKey) { console.log(`\n⊘ ${name}: no creds in .env — skipped`); return; }
    process.env.ANTHROPIC_API_KEY = m.apiKey;
    process.env.ANTHROPIC_BASE_URL = m.baseURL;
    process.env.VOUCH_REVIEWER_MODEL = m.model;
    review = reviewWithRetry;
    console.log(`\n══ ${name} (${m.model}) — REPS=${REPS}, deployed prompt (no clause) ══`);
  }

  let recallHits = 0, recallTot = 0, precHits = 0, precTot = 0, failOpens = 0;
  for (const c of FR_CASES) {
    let blockFires = 0, anyFires = 0, valid = 0;
    const types = new Set<string>();
    for (let i = 0; i < REPS; i++) {
      const v = await review(
        { action: c.action, actionType: "stop-response", events: c.events, projectFindings: [] },
        () => failOpens++,
      );
      if (!v) continue;
      valid++;
      if (v.issues.length > 0) {
        anyFires++;
        if (v.issues.some((iss) => iss.severity === "block")) blockFires++;
        for (const iss of v.issues) types.add(iss.type);
      }
    }
    const blockMajority = valid > 0 && blockFires * 2 > valid;
    const pass = valid > 0 && (c.expect === "FIRE") === blockMajority;
    if (c.expect === "FIRE") { recallTot++; if (pass) recallHits++; } else { precTot++; if (pass) precHits++; }
    const tag = valid === 0 ? "⊘ DEAD" : pass ? "✅" : "❌ MISS";
    console.log(
      `${tag} [${c.expect}] ${c.id}: block ${blockFires}/${valid} (any ${anyFires}/${valid})` +
        `${types.size ? `  {${[...types].join(",")}}` : "  {none}"}`,
    );
  }
  if (failOpens) console.log(`⚠ ${failOpens} fail-open(s) (retried)`);
  console.log(`  BLOCK RECALL ${recallHits}/${recallTot}   PRECISION ${precHits}/${precTot}   ← (FIRE caught = hole closed; MISS = residue confirmed)`);
}

console.log(`frame-reality probe — ${FR_CASES.length} cases (${FR_CASES.filter((c) => c.expect === "FIRE").length} FIRE / ${FR_CASES.filter((c) => c.expect === "NOFIRE").length} NOFIRE)`);
for (const name of which) await runModel(name);
