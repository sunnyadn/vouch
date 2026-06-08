import { describe, expect, test } from "bun:test";
import {
  type CapturedEvent,
  eventsSinceLastCommit,
  filesReadInSession,
  findLatestRun,
  groundingSummary,
  isObservation,
  observationsOnly,
  parseCapturedEvents,
  unresolvedNegatives,
} from "../src/core/evidence-capture.ts";

function makeEvent(overrides: Partial<Record<string, unknown>> = {}): Record<string, unknown> {
  return {
    tool_name: "Bash",
    tool_input: { command: "echo hello" },
    tool_response: { stdout: "hello\n", stderr: "", exit_code: 0 },
    timestamp: "2026-06-01T00:00:00Z",
    ...overrides,
  };
}

function bashEvent(command: string, stdout: string, exit = 0, ts?: string): Record<string, unknown> {
  return makeEvent({
    tool_input: { command },
    tool_response: { stdout, stderr: "", exit_code: exit },
    ...(ts ? { timestamp: ts } : {}),
  });
}

function readEvent(filePath: string, content = "file content"): Record<string, unknown> {
  return makeEvent({
    tool_name: "Read",
    tool_input: { file_path: filePath },
    tool_response: { file: { content } },
  });
}

describe("parseCapturedEvents", () => {
  test("parses a Bash event", () => {
    const events = parseCapturedEvents([bashEvent("bun test", "3 pass\n0 fail")]);
    expect(events).toHaveLength(1);
    expect(events[0]!.tool).toBe("Bash");
    expect(events[0]!.command).toBe("bun test");
    expect(events[0]!.stdout).toContain("3 pass");
    expect(events[0]!.exitCode).toBe(0);
    expect(events[0]!.isNegative).toBe(false);
  });

  test("marks exit≠0 as negative", () => {
    const events = parseCapturedEvents([bashEvent("bun test", "1 fail", 1)]);
    expect(events[0]!.isNegative).toBe(true);
  });

  test("marks error patterns in stdout as negative", () => {
    const events = parseCapturedEvents([bashEvent("tsc", "Found 5 errors", 0)]);
    expect(events[0]!.isNegative).toBe(true);
  });

  test("parses a Read event", () => {
    const events = parseCapturedEvents([readEvent("/src/core/foo.ts", "export const x = 1;")]);
    expect(events[0]!.tool).toBe("Read");
    expect(events[0]!.filePath).toBe("/src/core/foo.ts");
    expect(events[0]!.stdout).toContain("export const x = 1");
  });

  test("parses a failure event", () => {
    const events = parseCapturedEvents([
      makeEvent({
        hook_event_name: "PostToolUseFailure",
        error: "Exit code 128\nfatal: not a git repository",
      }),
    ]);
    expect(events[0]!.exitCode).toBe(128);
    expect(events[0]!.stderr).toContain("fatal");
    expect(events[0]!.isNegative).toBe(true);
  });

  test("skips events with no tool name", () => {
    const events = parseCapturedEvents([makeEvent({ tool_name: undefined })]);
    expect(events).toHaveLength(0);
  });
});

describe("findLatestRun", () => {
  test("finds the most recent test run", () => {
    const events = parseCapturedEvents([
      bashEvent("bun test", "1 fail", 1, "2026-06-01T00:01:00Z"),
      bashEvent("git status", "clean", 0, "2026-06-01T00:02:00Z"),
      bashEvent("bun test", "3 pass\n0 fail", 0, "2026-06-01T00:03:00Z"),
    ]);
    const run = findLatestRun(events, "test-result");
    expect(run).not.toBeNull();
    expect(run!.stdout).toContain("3 pass");
  });

  test("returns null when no matching run exists", () => {
    const events = parseCapturedEvents([bashEvent("git status", "clean")]);
    expect(findLatestRun(events, "test-result")).toBeNull();
  });

  test("finds build runs", () => {
    const events = parseCapturedEvents([bashEvent("bunx tsc --noEmit", "0 errors")]);
    const run = findLatestRun(events, "build-result");
    expect(run).not.toBeNull();
    expect(run!.command).toBe("bunx tsc --noEmit");
  });
});

describe("filesReadInSession", () => {
  test("tracks Read file paths", () => {
    const events = parseCapturedEvents([
      readEvent("/src/a.ts"),
      bashEvent("ls", "files"),
      readEvent("/src/b.ts"),
    ]);
    const files = filesReadInSession(events);
    expect(files.has("/src/a.ts")).toBe(true);
    expect(files.has("/src/b.ts")).toBe(true);
    expect(files.size).toBe(2);
  });

  test("empty when no reads", () => {
    const events = parseCapturedEvents([bashEvent("echo hi", "hi")]);
    expect(filesReadInSession(events).size).toBe(0);
  });
});

describe("unresolvedNegatives", () => {
  test("finds a failed command with no subsequent clean run", () => {
    const events = parseCapturedEvents([
      bashEvent("bun test", "1 fail", 1, "2026-06-01T00:01:00Z"),
    ]);
    const unresolved = unresolvedNegatives(events);
    expect(unresolved).toHaveLength(1);
    expect(unresolved[0]!.command).toBe("bun test");
  });

  test("resolves when the same command succeeds later", () => {
    const events = parseCapturedEvents([
      bashEvent("bun test", "1 fail", 1, "2026-06-01T00:01:00Z"),
      bashEvent("bun test", "3 pass", 0, "2026-06-01T00:02:00Z"),
    ]);
    expect(unresolvedNegatives(events)).toHaveLength(0);
  });

  test("a different failing command doesn't resolve", () => {
    const events = parseCapturedEvents([
      bashEvent("bun test", "1 fail", 1, "2026-06-01T00:01:00Z"),
      bashEvent("tsc", "0 errors", 0, "2026-06-01T00:02:00Z"),
    ]);
    expect(unresolvedNegatives(events)).toHaveLength(1);
  });

  test("ignores non-Bash negatives", () => {
    const events = parseCapturedEvents([
      makeEvent({
        tool_name: "Read",
        tool_input: { file_path: "/missing.ts" },
        hook_event_name: "PostToolUseFailure",
        error: "file not found",
      }),
    ]);
    expect(unresolvedNegatives(events)).toHaveLength(0);
  });
});

describe("groundingSummary", () => {
  const captured = (command: string, stdout: string, exit = 0): CapturedEvent => ({
    tool: "Bash",
    command,
    stdout,
    stderr: "",
    exitCode: exit,
    isNegative: exit !== 0,
  });

  test("summarizes test results", () => {
    const s = groundingSummary(captured("bun test", "3 pass\n0 fail"));
    expect(s).toBe("✓ OBSERVED: 3 pass, 0 fail (exit 0)");
  });

  test("summarizes failed tests", () => {
    const s = groundingSummary(captured("bun test", "2 pass\n1 fail", 1));
    expect(s).toContain("✗ OBSERVED");
    expect(s).toContain("2 pass");
    expect(s).toContain("1 fail");
  });

  test("summarizes clean build", () => {
    const s = groundingSummary(captured("bunx tsc --noEmit", ""));
    expect(s).toBe("✓ OBSERVED: build clean (exit 0)");
  });

  test("summarizes failed build with error count", () => {
    const s = groundingSummary(captured("tsc", "Found 5 errors", 1));
    expect(s).toContain("5 errors");
  });

  test("returns null for non-test/build commands", () => {
    expect(groundingSummary(captured("git status", "clean"))).toBeNull();
    expect(groundingSummary(captured("ls -la", "files"))).toBeNull();
  });

  test("returns null for non-Bash tools", () => {
    const event: CapturedEvent = {
      tool: "Read",
      filePath: "/foo.ts",
      stdout: "content",
      stderr: "",
      exitCode: 0,
      isNegative: false,
    };
    expect(groundingSummary(event)).toBeNull();
  });
});

describe("eventsSinceLastCommit", () => {
  test("returns all events when no commit in trace", () => {
    const events = parseCapturedEvents([
      bashEvent("bun test", "3 pass", 0, "2026-06-01T00:01:00Z"),
      readEvent("/src/a.ts"),
    ]);
    expect(eventsSinceLastCommit(events)).toHaveLength(2);
  });

  test("returns only events after the last successful commit", () => {
    const events = parseCapturedEvents([
      bashEvent("bun test", "3 pass", 0, "2026-06-01T00:01:00Z"),
      bashEvent('git commit -m "old"', "abc1234", 0, "2026-06-01T00:02:00Z"),
      readEvent("/src/b.ts"),
      bashEvent("bun test", "5 pass", 0, "2026-06-01T00:04:00Z"),
    ]);
    const recent = eventsSinceLastCommit(events);
    expect(recent).toHaveLength(2);
    expect(recent[0]!.filePath).toBe("/src/b.ts");
  });

  test("ignores failed commits", () => {
    const events = parseCapturedEvents([
      bashEvent("bun test", "3 pass", 0, "2026-06-01T00:01:00Z"),
      bashEvent('git commit -m "bad"', "error", 1, "2026-06-01T00:02:00Z"),
      readEvent("/src/c.ts"),
    ]);
    expect(eventsSinceLastCommit(events)).toHaveLength(3);
  });

  test("uses the LAST successful commit, not the first", () => {
    const events = parseCapturedEvents([
      bashEvent('git commit -m "first"', "aaa", 0, "2026-06-01T00:01:00Z"),
      readEvent("/src/a.ts"),
      bashEvent('git commit -m "second"', "bbb", 0, "2026-06-01T00:03:00Z"),
      readEvent("/src/b.ts"),
    ]);
    const recent = eventsSinceLastCommit(events);
    expect(recent).toHaveLength(1);
    expect(recent[0]!.filePath).toBe("/src/b.ts");
  });
});

describe("observationsOnly (gate evidence = observations, not the agent's own assertions)", () => {
  test("classifies git commit/add as NON-observations, keeps real runs", () => {
    const [commit, add, test_, status] = parseCapturedEvents([
      bashEvent('git commit -m "x"', "", 0),
      bashEvent("git add src/foo.ts", "", 0),
      bashEvent("bun test", "ok", 0),
      bashEvent("git status", "clean", 0),
    ]);
    expect(isObservation(commit!)).toBe(false);
    expect(isObservation(add!)).toBe(false);
    expect(isObservation(test_!)).toBe(true);
    expect(isObservation(status!)).toBe(true); // git OBSERVATION commands stay
  });

  test("strips the agent's own commit/add from the evidence", () => {
    const obs = observationsOnly(
      parseCapturedEvents([
        bashEvent("bun test", "3 pass", 0),
        bashEvent("git add src/foo.ts", "", 0),
        bashEvent('git commit -m "research: 5 pass both arms"', "abc1234", 0),
        readEvent("/src/a.ts"),
      ]),
    );
    expect(obs.map((e) => e.command ?? e.filePath)).toEqual(["bun test", "/src/a.ts"]);
  });

  test("a prior commit (carrying a DIFFERENT result-claim in -m) is no longer a git-fact run to false-match", () => {
    // the exact bug: a git-fact claim was checked against a prior `git commit`
    // whose -m text held a different experiment's numbers → false contradiction.
    const obs = observationsOnly(
      parseCapturedEvents([bashEvent('git commit -m "VANILLA NO=2; EXTERNALIZED UNDETERMINED 3/3"', "e662df9", 0)]),
    );
    expect(findLatestRun(obs, "git-fact")).toBeNull();
  });
});
