import { expect, test } from "bun:test";
import { readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { captureVerdict } from "./corpus.ts";

test("captureVerdict appends a replayable JSONL record", () => {
  const path = join(tmpdir(), `vouch-corpus-test-${process.pid}.jsonl`);
  process.env.VOUCH_CORPUS_PATH = path;
  try {
    captureVerdict({
      actionType: "stop-response",
      action: "all tests pass and the cache fix is verified",
      events: [
        {
          tool: "Read",
          filePath: "src/cache.ts",
          stdout: "",
          stderr: "",
          exitCode: 0,
          isNegative: false,
        },
        {
          tool: "Bash",
          command: "bun test",
          stdout: "1 fail",
          stderr: "1 fail",
          exitCode: 1,
          isNegative: true,
        },
      ],
      verdict: {
        ok: false,
        issues: [
          {
            type: "active-fabrication",
            severity: "block",
            detail: "claims pass but bun test failed",
          },
        ],
      },
    });
    const rec = JSON.parse(readFileSync(path, "utf8").trim());
    expect(rec.blocked).toBe(true);
    expect(rec.actionType).toBe("stop-response");
    // replayable through the agentic reviewer: the events it judged round-trip
    expect(
      rec.events.map((e: { command?: string; filePath?: string }) => e.filePath ?? e.command),
    ).toEqual(["src/cache.ts", "bun test"]);
    expect(rec.issues[0].type).toBe("active-fabrication");
  } finally {
    delete process.env.VOUCH_CORPUS_PATH;
    try {
      rmSync(path);
    } catch {
      /* already gone */
    }
  }
});

test("captureVerdict records blocked=false for advise-only verdicts", () => {
  const path = join(tmpdir(), `vouch-corpus-test2-${process.pid}.jsonl`);
  process.env.VOUCH_CORPUS_PATH = path;
  try {
    captureVerdict({
      actionType: "commit",
      action: " src/x.ts | 2 +-",
      events: [
        {
          tool: "Edit",
          filePath: "src/x.ts",
          stdout: "",
          stderr: "",
          exitCode: 0,
          isNegative: false,
        },
      ],
      verdict: {
        ok: false,
        issues: [
          { type: "research-insufficiency", severity: "warn", detail: "edited without reading" },
        ],
      },
    });
    const rec = JSON.parse(readFileSync(path, "utf8").trim());
    expect(rec.blocked).toBe(false);
    expect(rec.issues[0].severity).toBe("warn");
  } finally {
    delete process.env.VOUCH_CORPUS_PATH;
    try {
      rmSync(path);
    } catch {
      /* already gone */
    }
  }
});
