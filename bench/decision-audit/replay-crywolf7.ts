// One-off: replay THIS session's 7th cry-wolf through the real reviewer to (a) verify the new
// query-trail instrumentation flows end-to-end and (b) capture what kimi actually queries.
// Uses the deployed reviewer creds from .env (ANTHROPIC_* = kimi). Throwaway diagnostic.
import { parseCapturedEvents } from "../../src/core/evidence-capture.ts";
import { anthropicReviewerAgentic } from "../../src/core/reviewer-agentic.ts";

const raw = (await Bun.file(".vouch-trace.jsonl").text())
  .split("\n")
  .filter(Boolean)
  .map((l) => JSON.parse(l));
const events = parseCapturedEvents(raw);

// The exact draft span the Stop gate fired on this session.
const action =
  "Let me read the two key recent blocks to nail down the current conclusion and the real next step.";

const verdict = await anthropicReviewerAgentic({ action, actionType: "stop-response", events });

console.log("model:", process.env.VOUCH_REVIEWER_MODEL ?? "(default)");
console.log("status:", verdict.status, "| blocked:", verdict.issues.some((i) => i.severity === "block"));
console.log("queries:", JSON.stringify(verdict.queries, null, 2));
console.log("issues:", JSON.stringify(verdict.issues.map((i) => ({ type: i.type, sev: i.severity, quote: i.quote })), null, 2));
