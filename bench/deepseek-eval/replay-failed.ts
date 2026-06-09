// replay-failed.ts — pull every `status:"failed"` record from the live corpus and re-run it
// through the DEPLOYED reviewer with VOUCH_DIAG on, to pin WHY a big trace fails open:
// an API error (line 209 catch) vs a no-query_history bail (line 213). The corpus stores the
// exact events, so each record is replayable through the real reviewer unchanged.
//
//   bun bench/deepseek-eval/replay-failed.ts
//
// Reads ~/.claude/vouch-corpus.jsonl (or $VOUCH_CORPUS_PATH). Loads .env for the deepseek key.
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { corpusPath } from "../../src/core/corpus.ts";
import type { CapturedEvent } from "../../src/core/evidence-capture.ts";

const ROOT = join(import.meta.dir, "..", "..");
for (const line of readFileSync(join(ROOT, ".env"), "utf8").split("\n")) {
  const m = line.match(/^([A-Z_]+)=(.*)$/);
  if (m?.[1] && !process.env[m[1]]) process.env[m[1]] = m[2] ?? "";
}
process.env.VOUCH_DIAG = "1";

const { anthropicReviewerAgentic } = await import("../../src/core/reviewer-agentic.ts");

type Rec = { ts: string; actionType: string; action: string; status?: string; events: CapturedEvent[] };
const failed: Rec[] = readFileSync(corpusPath(), "utf8")
  .split("\n")
  .filter(Boolean)
  .map((l) => JSON.parse(l) as Rec)
  .filter((r) => r.status === "failed");

console.log(`replaying ${failed.length} failed records through the live deepseek reviewer\n`);
for (const r of failed) {
  console.error(`\n===== ${r.ts} (${r.actionType}, ${r.events.length} events) =====`);
  console.error(`action: ${r.action.slice(0, 80).replace(/\n/g, " ")}`);
  const t0 = performance.now();
  const v = await anthropicReviewerAgentic({
    action: r.action,
    actionType: r.actionType as "commit" | "stop-response" | "edit",
    events: r.events,
    projectFindings: [],
  });
  const dt = ((performance.now() - t0) / 1000).toFixed(1);
  console.error(`=> status=${v.status} issues=${v.issues.length} (${dt}s)`);
}
