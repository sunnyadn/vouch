import { describe, expect, test } from "bun:test";
import type { CapturedEvent } from "../src/core/evidence-capture.ts";
import {
  buildEvidenceSummary,
  formatReviewerHealthNote,
  formatReviewMessage,
  type ReviewContext,
  type ReviewFn,
  type ReviewVerdict,
} from "../src/core/reviewer.ts";

function event(overrides: Partial<CapturedEvent>): CapturedEvent {
  return {
    tool: "Bash",
    stdout: "",
    stderr: "",
    exitCode: 0,
    isNegative: false,
    ...overrides,
  };
}

describe("formatReviewerHealthNote", () => {
  const verdict = (status?: ReviewVerdict["status"]): ReviewVerdict => ({
    issues: [],
    ok: true,
    status,
  });
  test("warns ONLY when the reviewer failed open — the visible signal for a silent death", () => {
    expect(formatReviewerHealthNote(verdict("failed"))).toContain("vouch reviewer unavailable");
    expect(formatReviewerHealthNote(verdict("failed"))).toContain("vouch doctor");
  });
  test("stays silent for healthy, intentional, and legacy verdicts (no noise on a normal turn)", () => {
    expect(formatReviewerHealthNote(verdict("reviewed"))).toBe(""); // clean pass
    expect(formatReviewerHealthNote(verdict("skipped"))).toBe(""); // no key configured — by design
    expect(formatReviewerHealthNote(verdict(undefined))).toBe(""); // deterministic-gate / legacy
  });
});

describe("buildEvidenceSummary", () => {
  test("collects files read, edited, and bash commands", () => {
    const events: CapturedEvent[] = [
      event({ tool: "Read", filePath: "/src/a.ts", stdout: "content" }),
      event({ tool: "Read", filePath: "/src/b.ts", stdout: "content" }),
      event({ tool: "Bash", command: "bun test", stdout: "3 pass", exitCode: 0 }),
      event({ tool: "Edit", filePath: "/src/a.ts", stdout: "" }),
      event({ tool: "Bash", command: "git status", stdout: "clean", exitCode: 0 }),
    ];
    const summary = buildEvidenceSummary(events);
    expect(summary.filesRead).toEqual(["/src/a.ts", "/src/b.ts"]);
    expect(summary.filesEdited).toEqual(["/src/a.ts"]);
    expect(summary.bashCommands).toHaveLength(2);
    expect(summary.totalToolCalls).toBe(5);
    expect(summary.unresolvedFailures).toHaveLength(0);
  });

  test("tracks unresolved failures", () => {
    const events: CapturedEvent[] = [
      event({ tool: "Bash", command: "bun test", stdout: "1 fail", exitCode: 1, isNegative: true }),
      event({ tool: "Bash", command: "git status", stdout: "clean", exitCode: 0 }),
    ];
    const summary = buildEvidenceSummary(events);
    expect(summary.unresolvedFailures).toEqual(["bun test"]);
  });

  test("resolves failure when same command succeeds later", () => {
    const events: CapturedEvent[] = [
      event({ tool: "Bash", command: "bun test", stdout: "1 fail", exitCode: 1, isNegative: true }),
      event({ tool: "Bash", command: "bun test", stdout: "3 pass", exitCode: 0, isNegative: false }),
    ];
    const summary = buildEvidenceSummary(events);
    expect(summary.unresolvedFailures).toHaveLength(0);
  });

  test("deduplicates file paths", () => {
    const events: CapturedEvent[] = [
      event({ tool: "Read", filePath: "/src/a.ts" }),
      event({ tool: "Read", filePath: "/src/a.ts" }),
    ];
    const summary = buildEvidenceSummary(events);
    expect(summary.filesRead).toEqual(["/src/a.ts"]);
  });
});

describe("formatReviewMessage", () => {
  test("returns empty string for ok verdict", () => {
    expect(formatReviewMessage({ issues: [], ok: true })).toBe("");
  });

  test("formats block issues", () => {
    const verdict: ReviewVerdict = {
      ok: false,
      issues: [
        {
          type: "active-fabrication",
          severity: "block",
          detail: "commit says 7 pass but evidence shows 3",
          suggestion: "fix the count",
        },
      ],
    };
    const msg = formatReviewMessage(verdict);
    expect(msg).toContain("⛔ vouch reviewer (BLOCK)");
    expect(msg).toContain("7 pass but evidence shows 3");
    expect(msg).toContain("fix the count");
  });

  test("formats warn issues", () => {
    const verdict: ReviewVerdict = {
      ok: false,
      issues: [
        {
          type: "research-insufficiency",
          severity: "warn",
          detail: "editing 5 files but only read 2",
        },
      ],
    };
    const msg = formatReviewMessage(verdict);
    expect(msg).toContain("⚠ vouch reviewer (advise)");
    expect(msg).toContain("editing 5 files but only read 2");
  });

  test("formats mixed block + warn", () => {
    const verdict: ReviewVerdict = {
      ok: false,
      issues: [
        { type: "passive-fabrication", severity: "block", detail: "no evidence of reading auth" },
        { type: "omission", severity: "warn", detail: "test failures not mentioned" },
      ],
    };
    const msg = formatReviewMessage(verdict);
    expect(msg).toContain("BLOCK");
    expect(msg).toContain("advise");
  });
});

describe("reviewer integration (injectable)", () => {
  const fakeReviewer: ReviewFn = async (ctx) => {
    if (ctx.evidence.filesRead.length === 0 && ctx.action.includes("fixed")) {
      return {
        ok: false,
        issues: [
          {
            type: "passive-fabrication",
            severity: "block",
            detail: "claims fix but no files were read",
            suggestion: "read the relevant files first",
          },
        ],
      };
    }
    if (ctx.evidence.unresolvedFailures.length > 0 && ctx.action.includes("all tests pass")) {
      return {
        ok: false,
        issues: [
          {
            type: "active-fabrication",
            severity: "block",
            detail: "claims all tests pass but there are unresolved failures",
          },
        ],
      };
    }
    return { ok: true, issues: [] };
  };

  test("passive fabrication: fix claim with no file reads", async () => {
    const ctx: ReviewContext = {
      action: "fixed the auth bug",
      actionType: "commit",
      evidence: buildEvidenceSummary([]),
    };
    const verdict = await fakeReviewer(ctx);
    expect(verdict.ok).toBe(false);
    expect(verdict.issues[0]!.type).toBe("passive-fabrication");
  });

  test("active fabrication: all pass claim with failures", async () => {
    const ctx: ReviewContext = {
      action: "feat: all tests pass after refactor",
      actionType: "commit",
      evidence: buildEvidenceSummary([
        event({ tool: "Bash", command: "bun test", stdout: "1 fail", exitCode: 1, isNegative: true }),
      ]),
    };
    const verdict = await fakeReviewer(ctx);
    expect(verdict.ok).toBe(false);
    expect(verdict.issues[0]!.type).toBe("active-fabrication");
  });

  test("clean commit with evidence passes", async () => {
    const ctx: ReviewContext = {
      action: "refactor: extract helper function",
      actionType: "commit",
      evidence: buildEvidenceSummary([
        event({ tool: "Read", filePath: "/src/utils.ts" }),
        event({ tool: "Edit", filePath: "/src/utils.ts" }),
        event({ tool: "Bash", command: "bun test", stdout: "3 pass", exitCode: 0 }),
      ]),
    };
    const verdict = await fakeReviewer(ctx);
    expect(verdict.ok).toBe(true);
  });
});
