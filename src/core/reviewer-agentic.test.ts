import { test, expect } from "bun:test";
import { searchConversation } from "./conversation-capture.ts";
import type { CapturedEvent } from "./evidence-capture.ts";
import { parseCapturedEvents } from "./evidence-capture.ts";
import {
  anthropicReviewerAgentic,
  buildConversationBlock,
  composeSystemPrompt,
  formatHits,
  queryHistory,
} from "./reviewer-agentic.ts";

const ev = (o: Partial<CapturedEvent>): CapturedEvent => ({
  tool: "Bash", command: undefined, filePath: undefined, stdout: "", stderr: "", exitCode: 0, isNegative: false, ...o,
});

test("queryHistory finds a command by name and returns its FULL output (the un-truncated grounding)", () => {
  const events = [
    ev({ command: "bun test", stdout: "line1\nline2\n…\n141 pass, 0 fail" }), // grounding at the END
    ev({ command: "ls", stdout: "a b c" }),
  ];
  const hits = queryHistory(events, "bun test");
  expect(hits.length).toBe(1);
  expect(hits[0]!.output).toContain("141 pass"); // tail visible — the truncation cry-wolf can't happen
});

test("queryHistory windows on the MATCH so a fact in the TAIL of a long output stays visible", () => {
  // The R6 truncation cry-wolf: a 1500-char HEAD slice hid reps reported from the tail, so the
  // gate "saw 3 of 6" and flagged a true claim. Centering on the match keeps the tail visible.
  const filler = "x".repeat(3000);
  const hits = queryHistory([ev({ command: "run probe", stdout: `${filler}\nFINAL: R6 fired 6/6` })], "R6 fired");
  expect(hits.length).toBe(1);
  expect(hits[0]!.output).toContain("R6 fired 6/6");
});

test("queryHistory marks truncation so a cut-off view is not read as absence-of-evidence", () => {
  const hits = queryHistory([ev({ command: "big", stdout: `HEAD-FACT\n${"y".repeat(5000)}` })], "HEAD-FACT");
  expect(hits[0]!.output).toContain("HEAD-FACT");
  expect(hits[0]!.output).toMatch(/\[\+\d+ more chars\]/); // explicit marker → reviewer knows there's more
});

test("queryHistory returns the WHOLE output when it fits the budget (short outputs unchanged)", () => {
  expect(queryHistory([ev({ command: "bun test", stdout: "line1\n141 pass, 0 fail" })], "bun test")[0]!.output).toBe(
    "line1\n141 pass, 0 fail",
  );
});

test("queryHistory matches file reads and output text; empty when nothing matches", () => {
  const events = [
    ev({ tool: "Read", filePath: "src/auth.ts" }),
    ev({ command: "git show HEAD", stdout: "commit abc123\n+function sampleStdout" }),
  ];
  expect(queryHistory(events, "auth\\.ts").length).toBe(1); // by file path
  expect(queryHistory(events, "sampleStdout").length).toBe(1); // by output text
  expect(queryHistory(events, "abc123").length).toBe(1); // commit hash in output → post-commit summary groundable
  expect(queryHistory(events, "nope").length).toBe(0);
  expect(formatHits([])).toContain("no matching");
});

test("a background launch id is searchable so a 'Launched (id)' claim is verifiable (not a cry-wolf)", () => {
  // The harness returns empty stdout + the shell id ONLY in backgroundTaskId. Pre-fix that id was
  // unexposed → query_history for it returned 0 hits → the reviewer flagged the truthful launch as
  // fabrication (the #5/#6 background-launch cry-wolf class).
  const events = parseCapturedEvents([
    {
      hook_event_name: "PostToolUse",
      tool_name: "Bash",
      tool_input: { command: "REPS=8 bun consensus.ts > /tmp/c.log 2>&1" },
      tool_response: { stdout: "", stderr: "", backgroundTaskId: "bapjzx7ds" },
    },
  ]);
  expect(queryHistory(events, "bapjzx7ds").length).toBe(1);
});

test("no API key → verdict.status is 'skipped', NOT an indistinguishable empty pass", async () => {
  // The silent fail-open gap: a clean review and a dead/absent reviewer both have issues:[].
  // status disambiguates so a quiet vouch can't masquerade as a working one.
  const saved = process.env.ANTHROPIC_API_KEY;
  delete process.env.ANTHROPIC_API_KEY;
  try {
    const v = await anthropicReviewerAgentic({ action: "tests pass", actionType: "stop-response", events: [] });
    expect(v.issues).toEqual([]);
    expect(v.status).toBe("skipped"); // intentional no-op — distinct from a 'reviewed' clean pass
  } finally {
    if (saved !== undefined) process.env.ANTHROPIC_API_KEY = saved;
  }
});

test("PRODUCTION INVARIANT: feature OFF → prompt is byte-identical to the trace-only path", () => {
  // The structural "zero production impact" claim, made testable (research-insufficiency catch
  // 2026-06-24). With no conversation layer supplied (VOUCH_INCLUDE_CONVERSATION unset → arrays
  // undefined), the user-message block adds ZERO bytes and the system prompt gets NO clause.
  expect(buildConversationBlock({})).toBe(""); // off → no bytes in the user message
  expect(buildConversationBlock({ userMessages: [], priorVerdicts: [], assistantMessages: [] })).toBe(""); // empty arrays too

  const off = composeSystemPrompt(false);
  const on = composeSystemPrompt(true);
  expect(off).not.toContain("CONVERSATION SCOPE"); // the clause marker is absent on the off path
  expect(off).not.toContain("YOUR OWN PRIOR RESPONSES");
  expect(on.startsWith(off)).toBe(true); // on = off + appended clause → off path is byte-identical
  expect(on).toContain("CONVERSATION SCOPE"); // clause present only when conversation is supplied
  expect(on).toContain("YOUR OWN PRIOR RESPONSES");
});

test("WINDOW DECOUPLING (red→green): a self-reference that the prompt window DROPS is still REACHABLE", () => {
  // The aged-out self-reference cry-wolf (this session: referenced prose at index 4 of 30, outside a
  // 15-turn window). Discriminating test: SAME input, the windowed prompt block does NOT surface the
  // target (RED — windowed-only would re-fire), but searchConversation over the full array DOES (GREEN).
  const target = "I will first confirm the architecture before proposing";
  const assistantMessages = [target, ...Array.from({ length: 20 }, (_, i) => `later filler turn ${i}`)];

  // RED: the prompt window (12) drops the oldest turn → the referenced prose is NOT in the block.
  const windowed = buildConversationBlock({ assistantMessages }, 12);
  expect(windowed).not.toContain("first confirm the architecture");

  // GREEN: query_history's conversation search reaches the FULL array → the prose is found.
  const reached = searchConversation("first confirm the architecture", { assistantMessages });
  expect(reached).toContain("first confirm the architecture");
  expect(reached).toContain("YOUR OWN PRIOR RESPONSES"); // existence-only labelling carried into the hit
});

test("queryHistory is window-agnostic: a pre-commit event is still found after a commit", () => {
  // the exact post-commit-summary cry-wolf: build happened, THEN commit. eventsSinceLastCommit
  // would drop the build; query_history over the full trace still finds it.
  const events = [
    ev({ command: "bun run build", stdout: "compile plugin/bin/vouch" }),
    ev({ command: "git commit -m x", stdout: "[branch f312ecc] x" }),
  ];
  expect(queryHistory(events, "bun run build").length).toBe(1);
  expect(queryHistory(events, "f312ecc").length).toBe(1);
});
