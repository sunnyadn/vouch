// Throwaway diagnostic (2026-06-15): WHAT does kimi actually flag on C6 (ablation-present) and
// C13 (external-claim-with-search)? The precision-experiment proved a prompt clause won't fix the
// cry-wolf; the bench only counts issues>0. This prints the full verdict DETAIL (type/quote/detail)
// per rep so we can see the real root cause and route any fix.
// Run: REPS=3 bun bench/decision-audit/diagnose-c6c13.ts

import { CASES } from "./cases.ts";
import { setModelEnv } from "../lib/models-config.ts";

setModelEnv("kimi"); // deployed prompt (no extra)

const REPS = Number(process.env.REPS ?? 3);
const { reviewWithRetry } = await import("../lib/reviewer-retry.ts");

const TARGETS = ["C6-ran-the-ablation", "C13-external-claim-with-search"];

for (const id of TARGETS) {
  const c = CASES.find((x) => x.id === id)!;
  console.log(`\n══════ ${id} (expect NOFIRE) ══════`);
  console.log(`ACTION: ${c.action}\n`);
  for (let i = 0; i < REPS; i++) {
    const v = await reviewWithRetry({ action: c.action, actionType: "stop-response", events: c.events, projectFindings: [] }, () => {});
    if (!v) { console.log(`  rep ${i + 1}: FAIL-OPEN (no verdict)`); continue; }
    if (v.issues.length === 0) { console.log(`  rep ${i + 1}: ✓ clean (no fire)`); continue; }
    console.log(`  rep ${i + 1}: 🔥 FIRED ${v.issues.length} issue(s)`);
    for (const iss of v.issues) {
      console.log(`     [${iss.type}/${iss.severity}]`);
      console.log(`       quote: ${JSON.stringify(iss.quote)}`);
      console.log(`       detail: ${iss.detail}`);
    }
  }
}
