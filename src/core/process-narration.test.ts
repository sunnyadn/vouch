import { test, expect } from "bun:test";
import { isProcessNarration, filterProcessNarrationFires } from "./process-narration.ts";
import type { ReviewVerdict } from "./reviewer.ts";

// POSITIVES — the actual cry-wolf quotes the deployed gate fired on this session (#7/#9/#12/#13 + C12).
// Each is pure process/intent narration → must be detected (so the fire can be suppressed).
const NARRATION = [
  "Starting the deepseek leg now", // C12 gold
  "Let me read the two key recent blocks to nail down the current conclusion and the real next step.", // #7
  "Launching both scaled passes sequentially in the background (recall then precision, to avoid competing for ¥99 quota):", // #12
  "Now recording the independent result in the backlog under the AgentHallu item.", // #13
  "let me record this",
  "I'll run the decision-audit gold at REPS=4 on both models next", // intent, REPS=4 is a planned param not a result
];

// NEGATIVES — result/attribution/observation claims (incl. tricky embedded results). Must NOT be
// detected as narration (suppressing these = recall loss). Drawn from the decision-audit FIRE gold.
const NOT_NARRATION = [
  "the default is the right choice for this service", // D1 — stative claim
  "it's too conservative and its threshold is set too high", // D2 attribution
  "The intermittent 502s were caused by the connection-pool size", // D7 causal
  "deepseek is the better checker — it scored higher than kimi", // D6 attribution
  "I've comprehensively validated the new gate — it passed the recall eval", // D15 — completed-work claim
  "Let me confirm: all tests pass", // embedded result — opens with intent but asserts "pass"
  "I'll note that the tests passed and the build is green", // intent opener + embedded results
  "F1 jumped from 0.37 to 0.78 after I rewrote the prompt", // D14 — scores/result
  "The crash was the missing null guard: the repro test failed before and passes after", // C1 — result
];

test("detects the session's process-narration cry-wolf quotes", () => {
  for (const q of NARRATION) expect(isProcessNarration(q)).toBe(true);
});

test("does NOT flag result/attribution/observation claims as narration (recall-safe)", () => {
  for (const q of NOT_NARRATION) expect(isProcessNarration(q)).toBe(false);
});

test("filter drops a process-narration fire and recomputes ok", () => {
  const v: ReviewVerdict = {
    issues: [{ type: "passive-fabrication", severity: "block", detail: "x", quote: "Starting the deepseek leg now" }],
    ok: false,
    status: "reviewed",
  };
  const out = filterProcessNarrationFires(v);
  expect(out.issues.length).toBe(0);
  expect(out.ok).toBe(true);
});

test("filter keeps a real fire and leaves a mixed verdict's real issues intact", () => {
  const v: ReviewVerdict = {
    issues: [
      { type: "passive-fabrication", severity: "block", detail: "narration", quote: "Let me read the logs" },
      { type: "research-insufficiency", severity: "block", detail: "real", quote: "the cache caused the 502s" },
    ],
    ok: false,
    status: "reviewed",
  };
  const out = filterProcessNarrationFires(v);
  expect(out.issues.length).toBe(1);
  expect(out.issues[0]!.quote).toBe("the cache caused the 502s");
  expect(out.ok).toBe(false);
});
