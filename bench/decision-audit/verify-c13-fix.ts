// Throwaway verification (2026-06-15): after the C13 gold fix — does repaired C13 now PASS as a
// clean grounded-external NOFIRE control, and does the new D16 (the split-out overreach) FIRE? —
// on the DEPLOYED model (kimi). VERIFY, don't assume (this session's discipline). REPS each.
// Run: REPS=4 bun bench/decision-audit/verify-c13-fix.ts

import { CASES } from "./cases.ts";
import { setModelEnv } from "../lib/models-config.ts";

setModelEnv("kimi"); // deployed prompt (no extra)
const REPS = Number(process.env.REPS ?? 4);
const { reviewWithRetry } = await import("../lib/reviewer-retry.ts");

const TARGETS = ["C13-external-claim-with-search", "D16-external-overreach-beyond-fetch"];

for (const id of TARGETS) {
  const c = CASES.find((x) => x.id === id)!;
  console.log(`\n══════ ${id} (expect ${c.expect}) ══════`);
  console.log(`ACTION: ${c.action}\n`);
  let fires = 0, valid = 0;
  for (let i = 0; i < REPS; i++) {
    const v = await reviewWithRetry({ action: c.action, actionType: "stop-response", events: c.events, projectFindings: [] }, () => {});
    if (!v) { console.log(`  rep ${i + 1}: FAIL-OPEN`); continue; }
    valid++;
    if (v.issues.length === 0) { console.log(`  rep ${i + 1}: clean (no fire)`); continue; }
    fires++;
    console.log(`  rep ${i + 1}: 🔥 ${v.issues.map((x) => `[${x.type}] ${x.detail.slice(0, 140)}`).join(" | ")}`);
  }
  const majorityFire = fires * 2 > valid;
  const pass = valid > 0 && (c.expect === "FIRE") === majorityFire;
  console.log(`  → fired ${fires}/${valid} | ${pass ? "✓ PASS" : "✗ FAIL"} (expect ${c.expect})`);
}
