// Faithful cry-wolf replay: re-run a PAST corpus record through the instrumented reviewer using
// THAT RECORD'S OWN stored events (the documented replay contract — the corpus snapshots what the
// reviewer actually saw, trimmed to last ~120 / 1200 chars-per-output). This is the correct way
// to re-examine #5/#6 — the LIVE trace has drifted since 06-15, so replaying against it would be
// a different input. Pass a run-ID (or any action substring) per arg; replays the first BLOCKED
// reviewed record matching it. Uses deployed creds from .env (ANTHROPIC_* = kimi).
import { homedir } from "node:os";
import { join } from "node:path";
import type { CapturedEvent } from "../../src/core/evidence-capture.ts";
import { anthropicReviewerAgentic } from "../../src/core/reviewer-agentic.ts";

const corpusPath = process.env.VOUCH_CORPUS_PATH ?? join(homedir(), ".claude", "vouch-corpus.jsonl");
const lines = (await Bun.file(corpusPath).text()).split("\n").filter(Boolean).map((l) => JSON.parse(l));

const needles = process.argv.slice(2);
if (needles.length === 0) {
  console.error("usage: bun replay-crywolf-corpus.ts <action-substring> [<action-substring> ...]");
  process.exit(1);
}

for (const needle of needles) {
  const rec = lines.find(
    (r) => r.blocked === true && r.status === "reviewed" && typeof r.action === "string" && r.action.includes(needle),
  );
  console.log("\n══════════════════════════════════════════════════════");
  if (!rec) {
    console.log(`NO blocked/reviewed corpus record contains "${needle}"`);
    continue;
  }
  const events = rec.events as CapturedEvent[];
  console.log(`REPLAY "${needle}" — original ts=${rec.ts}, stored events=${events.length}`);
  console.log("ACTION:", JSON.stringify(rec.action.slice(0, 300)));
  const verdict = await anthropicReviewerAgentic({ action: rec.action, actionType: "stop-response", events });
  console.log("→ status:", verdict.status, "| blocked:", verdict.issues.some((i) => i.severity === "block"));
  console.log("→ queries:", JSON.stringify(verdict.queries));
  console.log(
    "→ issues:",
    JSON.stringify(verdict.issues.map((i) => ({ type: i.type, sev: i.severity, quote: i.quote?.slice(0, 80) }))),
  );
}
