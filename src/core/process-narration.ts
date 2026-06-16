// Surgical fix for the dominant live cry-wolf: the reviewer firing fabrication on a present-tense
// PROCESS/INTENT narration span ("Let me read X", "Starting Y now", "Launching Z", "Now recording W").
// Such a span describes an action the agent is about-to/doing/planning — it asserts no result, so it
// CANNOT be a fabrication. This session hit it ~9× (#7, #9, #12, #13 + the C12 gold case).
//
// A global prompt clause was tried and DISCARDED (env-gated A/B): it replicably softened the
// reviewer's block-severity on FIRE cases with intent phrasing, costing block-recall 16→14/13.
// This deterministic post-filter is RECALL-SAFE BY CONSTRUCTION: it suppresses a fire ONLY when the
// fire's ENTIRE flagged quote is pure process narration, never touching the reviewer's judgment on
// anything else. A real overreach is a claim ABOUT a result/attribution — it carries an outcome
// marker (passed/caused/is/…) or doesn't open with an intent marker — so it is never matched.
//
// Bias is deliberately asymmetric: err toward NOT matching. A false negative leaves a narration
// cry-wolf (mild). A false positive suppresses a real fire (recall loss) — the thing we must not do.

import type { ReviewVerdict } from "./reviewer.ts";

// Opens with a first-person-future / imperative-intent / leading-gerund action marker.
const INTENT_OPENER =
  /^[\s"'(\[]*(let me\b|let's\b|i'?ll\b|i will\b|i'?m (?:going|about) to\b|i am (?:going|about) to\b|now,?\s+(?:i'?ll|i will|let me|i'?m|i am)\b|now\s+\w+ing\b|next,?\s+(?:i'?ll|i will|let me)\b|first,?\s+(?:i'?ll|let me)\b|going to\b|about to\b|(?:starting|launching|recording|running|checking|reading|writing|building|adding|kicking off|spinning up|firing off)\b)/i;

// ANY signal the span asserts an OUTCOME / observation / completed result (not merely a planned
// action). Broad on purpose: anything that smells like a result disqualifies → no suppression.
// (Stative is/are/was/were/has/have and result verbs catch embedded claims like
// "Let me confirm: all tests pass" — the "pass" disqualifies, so it is NOT suppressed.)
// NOTE: "pass"/"fail" are ambiguous (noun "scaled passes" vs result "tests pass"). They count as an
// outcome ONLY when past-tense (passed/failed/failure) or in result context (tests/all/N pass) — a
// bare noun "passes" does not disqualify. Other verbs are past/3rd-person result forms only.
const OUTCOME_SIGNAL =
  /\b(passed|failed|failure|returned|shows?|showed|confirm(?:ed|s)?|found|fix(?:ed|es)?|caused?|works?|worked|verif(?:ied|ies|y)|prov(?:ed|es|en)|reveal(?:ed|s)?|dropped|rose|improv(?:ed|es|ement)|regress(?:ed|es)|contradict(?:ed|s)?|exists?|succeed(?:ed|s)?|crash(?:ed|es)?|broke|is|are|was|were|been|has|have|had|did|does)\b|\d\s*\/\s*\d|\b\d+\s*(?:%|errors?|hits?|cases?|reps?)\b|\b(?:tests?|checks?|cases?|all|everything|\d+)\s+(?:pass|fail)\w*\b/i;

// Single-action narration is short; cap length so a long multi-claim span is never suppressed.
const MAX_NARRATION_LEN = 280;

/**
 * True iff the flagged quote is PURE process/intent narration: it opens with an action-intent
 * marker AND carries no outcome/result/observation signal AND is short. Recall-safe (see header).
 */
export function isProcessNarration(quote: string): boolean {
  const q = quote.trim();
  if (!q || q.length > MAX_NARRATION_LEN) return false;
  if (!INTENT_OPENER.test(q)) return false;
  if (OUTCOME_SIGNAL.test(q)) return false;
  return true;
}

/**
 * Drop any issue whose entire quote is process narration; recompute ok. Returns the verdict
 * unchanged if nothing matched (identity fast-path).
 */
export function filterProcessNarrationFires(verdict: ReviewVerdict): ReviewVerdict {
  const kept = verdict.issues.filter((i) => !(i.quote && isProcessNarration(i.quote)));
  if (kept.length === verdict.issues.length) return verdict;
  return { ...verdict, issues: kept, ok: kept.length === 0 };
}
