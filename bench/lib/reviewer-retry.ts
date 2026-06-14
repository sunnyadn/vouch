// Shared fail-open-aware retry around the DEPLOYED reviewer — the single source of the contract
// every bench now relies on (introduced piecemeal in c4d002e, then copy-pasted into 5 runners).
//
// A status:"failed" rep (429/quota/timeout — the reviewer fails OPEN) is retried up to 3x with
// backoff; if still dead, returns null so the caller EXCLUDES it from the denominator instead of
// silently counting a dead rep as a no-fire (which corrupts recall). `onFailOpen` fires once per
// retry attempt, for the caller's fail-open tally.
//
// Centralizing this keeps the backoff/retry/exclusion semantics from drifting across benches —
// a divergence there would silently change what "fail-open-aware" means per bench.

import { anthropicReviewerAgentic, type AgenticContext } from "../../src/core/reviewer-agentic.ts";
import type { ReviewVerdict } from "../../src/core/reviewer.ts";

export async function reviewWithRetry(
  ctx: AgenticContext,
  onFailOpen?: () => void,
): Promise<ReviewVerdict | null> {
  let v = await anthropicReviewerAgentic(ctx);
  for (let r = 0; v.status === "failed" && r < 3; r++) {
    onFailOpen?.();
    await new Promise((res) => setTimeout(res, 2500 * (r + 1))); // back off the 429
    v = await anthropicReviewerAgentic(ctx);
  }
  return v.status === "failed" ? null : v; // null = dead after retries → caller excludes it
}
