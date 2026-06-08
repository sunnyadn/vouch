import { expect, test } from "bun:test";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { formatDoctor, runDoctor } from "./doctor.ts";

// The headline guarantee: a missing key must be LOUD, not silent. (The whole reason
// doctor exists — most subscription users have no ANTHROPIC_API_KEY and would otherwise
// never know the LLM reviewer is off.) No key → no network round-trip either.
test("doctor flags missing ANTHROPIC_API_KEY as blocking and skips the network round-trip", async () => {
  const saved = process.env.ANTHROPIC_API_KEY;
  delete process.env.ANTHROPIC_API_KEY;
  const dir = mkdtempSync(join(tmpdir(), "vouch-doctor-")); // isolate the trace-writability probe
  try {
    const checks = await runDoctor({ dir });
    const reviewer = checks.find((c) => c.name === "LLM reviewer");
    expect(reviewer?.ok).toBe(false);
    expect(reviewer?.detail).toContain("ANTHROPIC_API_KEY");
    expect(checks.find((c) => c.name === "round-trip")).toBeUndefined(); // no key → no API call
    const out = formatDoctor(checks);
    expect(out).toContain("✗");
    expect(out).toContain("blocking issue");
  } finally {
    if (saved !== undefined) process.env.ANTHROPIC_API_KEY = saved;
  }
});

test("formatDoctor renders an all-green report with no blocking line", () => {
  const out = formatDoctor([{ name: "deterministic gate", ok: true, detail: "always on." }]);
  expect(out).toContain("✓");
  expect(out).toContain("all green");
  expect(out).not.toContain("blocking issue");
});
