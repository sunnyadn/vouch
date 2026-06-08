// Tool-trace append: always-on capture. Every tool call is evidence; the trace
// file is created on first write if it doesn't exist.

import { describe, expect, test } from "bun:test";
import { existsSync } from "node:fs";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { appendTrace, readTrace } from "../src/core/active-task.ts";

async function withTmp(fn: (dir: string) => Promise<void>): Promise<void> {
  const dir = await mkdtemp(join(tmpdir(), "vouch-trace-"));
  try {
    await fn(dir);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
}

describe("appendTrace", () => {
  test("always-on: creates file on first write", async () => {
    await withTmp(async (dir) => {
      const wrote = await appendTrace({ tool_name: "Bash" }, dir);
      expect(wrote).toBe(true);
      expect(existsSync(join(dir, ".vouch-trace.jsonl"))).toBe(true);
    });
  });

  test("appends one compacted line per event, round-trips via readTrace", async () => {
    await withTmp(async (dir) => {
      expect(await appendTrace({ tool_name: "Bash", n: 1 }, dir)).toBe(true);
      expect(await appendTrace({ tool_name: "Edit", n: 2 }, dir)).toBe(true);
      const events = await readTrace(dir);
      expect(events).toHaveLength(2);
      expect(events[0]).toEqual({ tool_name: "Bash", n: 1 });
      expect(events[1]).toEqual({ tool_name: "Edit", n: 2 });
    });
  });
});
