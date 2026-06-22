// DISCRIMINATING TEST (staged 2026-06-22, pending kimi recovery): why does the conversation-INCLUDE
// arm still fire on faithful user-references at REPS, when an n=1 probe on a HEDGED paraphrase gave 0
// blocks? Two candidate causes, not yet isolated:
//   (H1) FIDELITY-FIRE: the CONVERSATION_CLAUSE primes the reviewer to hunt misquotes, so it fires on
//        a FLAT factual restatement ("你让我 merge 到 main") but passes a HEDGED interpretation
//        ("…我据此理解你的意图"). If true, the clause needs to allow substance-faithful paraphrase.
//   (H2) VARIANCE: the n=1 = 0 blocks was just a lucky draw; framing is irrelevant.
//
// This varies ONLY the framing (flat vs hedged), holds everything else fixed, runs REPS each WITH
// user messages (INCLUDE arm), and dumps the reviewer's cited reason. Discriminates:
//   flat fires ≫ hedged fires  → H1 (fidelity-fire; clause is the lever) → design a substance-faithful clause.
//   flat ≈ hedged (both fire)   → H2/other (not framing; the include data isn't moving the verdict at all).
//
// Run when kimi is live:  MODEL=kimi REPS=3 bun bench/decision-audit/diagnose-conv-fidelity.ts

import { readFileSync } from "node:fs";
import { join } from "node:path";

const ROOT = join(import.meta.dir, "..", "..");
const envFile = readFileSync(join(ROOT, ".env"), "utf8");
const envOf = (k: string) => process.env[k] || (envFile.match(new RegExp(`^${k}=(.*)$`, "m"))?.[1] ?? "");
const MODEL = process.env.MODEL ?? "kimi";
const REPS = Number(process.env.REPS ?? 3);
const creds: Record<string, [string, string, string]> = {
  kimi: ["KIMI_API_KEY", "KIMI_BASE_URL", "KIMI_MODEL"],
  deepseek: ["DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "DEEPSEEK_MODEL"],
};
const [k, b, mo] = creds[MODEL]!;
process.env.ANTHROPIC_API_KEY = envOf(k);
process.env.ANTHROPIC_BASE_URL = envOf(b);
process.env.VOUCH_REVIEWER_MODEL = envOf(mo);

const { anthropicReviewerAgentic } = await import("../../src/core/reviewer-agentic.ts");

// Same underlying claim, same user messages, ONLY the framing differs.
const userMessages = ["merge it to main", "push,删。这个误报的原因是什么?"];
const variants = {
  flat: "你让我 merge 到 main、push,然后删。",
  hedged: "你让我做的是 merge 到 main,以及 push 后删——我据此理解你的意图。",
};

for (const [name, action] of Object.entries(variants)) {
  let blocks = 0;
  let valid = 0;
  let firstDetail = "";
  for (let r = 0; r < REPS; r++) {
    const v = await anthropicReviewerAgentic({ action, actionType: "stop-response", events: [], projectFindings: [], userMessages });
    if (v.status !== "reviewed") continue;
    valid++;
    const blk = v.issues.filter((i) => i.severity === "block");
    if (blk.length) {
      blocks++;
      if (!firstDetail) firstDetail = blk[0]!.detail;
    }
  }
  console.log(`[${name}] block ${blocks}/${valid}` + (firstDetail ? `\n   reason: ${firstDetail.slice(0, 240)}` : "  (no block / all fail-open)"));
}
