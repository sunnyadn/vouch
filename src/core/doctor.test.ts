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

// The plugin is dead if the hooks can't resolve bare `vouch` on PATH — and a green
// round-trip hides that (doctor can be reached via `bun run src/cli.ts` with no link).
// doctor must flag it as blocking, not stay silent.
test("doctor flags `vouch` not resolving on PATH as a blocking hook-command issue", async () => {
  const savedPath = process.env.PATH;
  const savedKey = process.env.ANTHROPIC_API_KEY;
  delete process.env.ANTHROPIC_API_KEY; // avoid a real network round-trip
  process.env.PATH = "/nonexistent-doctor-test-dir"; // no `vouch` here
  const dir = mkdtempSync(join(tmpdir(), "vouch-doctor-"));
  try {
    const checks = await runDoctor({ dir });
    const hook = checks.find((c) => c.name === "hook command");
    expect(hook?.ok).toBe(false);
    expect(hook?.detail).toContain("not on PATH");
    expect(formatDoctor(checks)).toContain("blocking issue");
  } finally {
    if (savedPath !== undefined) process.env.PATH = savedPath;
    if (savedKey !== undefined) process.env.ANTHROPIC_API_KEY = savedKey;
  }
});

test("formatDoctor renders an all-green report with no blocking line", () => {
  const out = formatDoctor([{ name: "deterministic gate", ok: true, detail: "always on." }]);
  expect(out).toContain("✓");
  expect(out).toContain("all green");
  expect(out).not.toContain("blocking issue");
});
