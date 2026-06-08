import { test, expect } from "bun:test";
import type { CapturedEvent } from "./evidence-capture.ts";
import { anthropicReviewerAgentic, queryHistory, formatHits } from "./reviewer-agentic.ts";

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
