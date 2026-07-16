// Parameterized verdict-detail diagnostic: prints WHAT the deployed reviewer (kimi) actually flags
// on a chosen set of gold cases (type/quote/detail per rep), so a fire can be root-caused rather
// than just counted. Supersedes the C6/C13-hardcoded diagnose-c6c13.ts.
// Run: CASES=C1,C6,C11,C12 REPS=3 bun bench/decision-audit/diagnose.ts
//      (default targets = the 4 stable-FP controls from the 2026-06-15 consensus experiment)

import { CASES } from "./cases.ts";
import { setModelEnv } from "../lib/models-config.ts";

setModelEnv("kimi"); // deployed prompt, no extra
const REPS = Number(process.env.REPS ?? 3);
const TARGETS = (process.env.CASES ?? "C1,C6,C11,C12").split(",").map((s) => s.trim());
const { reviewWithRetry } = await import("../lib/reviewer-retry.ts");

for (const t of TARGETS) {
  const c = CASES.find((x) => x.id === t || x.id.startsWith(`${t}-`));
  if (!c) { console.log(`\n══════ ${t}: NOT FOUND ══════`); continue; }
  console.log(`\n══════ ${c.id} (expect ${c.expect}) ══════`);
  console.log(`ACTION: ${c.action}`);
  console.log(`EVENTS: ${c.events.map((e) => e.command ?? `${e.tool} ${e.filePath ?? ""}`).join(" | ")}\n`);
  for (let i = 0; i < REPS; i++) {
    const v = await reviewWithRetry({ action: c.action, actionType: "stop-response", events: c.events, projectFindings: [] }, () => {});
    if (!v) { console.log(`  rep ${i + 1}: FAIL-OPEN`); continue; }
    if (v.issues.length === 0) { console.log(`  rep ${i + 1}: ✓ clean (no fire)`); continue; }
    console.log(`  rep ${i + 1}: 🔥 ${v.issues.length} issue(s)`);
    for (const iss of v.issues) {
      console.log(`     [${iss.type}/${iss.severity}] quote: ${JSON.stringify(iss.quote)}`);
      console.log(`       detail: ${iss.detail}`);
    }
  }
}
