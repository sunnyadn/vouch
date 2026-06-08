// vouch doctor — diagnose whether vouch is actually wired to catch things in THIS
// environment. The headline failure it surfaces: the LLM reviewer SILENTLY no-ops
// when ANTHROPIC_API_KEY is absent (most subscription/OAuth Claude Code logins do not
// export a key), so the user believes vouch is reviewing when only the deterministic
// gate runs. doctor makes that — and the real per-call latency — visible.

import { rm } from "node:fs/promises";
import { join } from "node:path";
import Anthropic from "@anthropic-ai/sdk";
import { baseDir } from "./active-task.ts";
import { DEFAULT_MODEL, loadProjectFindings } from "./reviewer.ts";

export interface DoctorCheck {
  name: string;
  ok: boolean | "warn";
  detail: string;
}

export async function runDoctor(opts: { dir?: string } = {}): Promise<DoctorCheck[]> {
  const checks: DoctorCheck[] = [];
  const apiKey = process.env.ANTHROPIC_API_KEY;
  const reviewerOff = !!process.env.VOUCH_REVIEWER_OFF;
  const model = process.env.VOUCH_REVIEWER_MODEL ?? DEFAULT_MODEL;

  // 1. The make-or-break: without a key the LLM reviewer is OFF, silently.
  if (!apiKey) {
    checks.push({
      name: "LLM reviewer",
      ok: false,
      detail:
        "ANTHROPIC_API_KEY not set → LLM reviewer is OFF (only the deterministic gate runs). " +
        "A Claude Code subscription/OAuth login does NOT export a key; set ANTHROPIC_API_KEY to enable it.",
    });
  } else if (reviewerOff) {
    checks.push({
      name: "LLM reviewer",
      ok: "warn",
      detail: "key set, but VOUCH_REVIEWER_OFF=1 disables it.",
    });
  } else {
    checks.push({ name: "LLM reviewer", ok: true, detail: "enabled (ANTHROPIC_API_KEY set)." });
  }

  // 2. Where the agent's diffs/responses are sent (privacy) + which model.
  const baseURL = process.env.ANTHROPIC_BASE_URL;
  checks.push({
    name: "endpoint",
    ok: true,
    detail: baseURL
      ? `${baseURL} (custom — your diffs/responses are sent here)`
      : "api.anthropic.com (default)",
  });
  checks.push({ name: "model", ok: true, detail: model });

  // 3. reviewer mode (agentic — queries the full trace; 1-6 sequential calls per fire).
  checks.push({
    name: "reviewer mode",
    ok: true,
    detail: "agentic: 1-6 sequential calls per Stop/commit.",
  });

  // 4. A REAL round-trip — anthropicReviewer is fail-open and would HIDE a bad key/endpoint,
  //    so call the client directly here to surface errors and measure the real latency.
  if (apiKey && !reviewerOff) {
    try {
      const client = new Anthropic({ apiKey });
      const t0 = performance.now();
      await client.messages.create({
        model,
        max_tokens: 4,
        temperature: 0,
        messages: [{ role: "user", content: "reply ok" }],
      });
      const ms = Math.round(performance.now() - t0);
      checks.push({
        name: "round-trip",
        ok: true,
        detail: `${ms}ms for ONE call. The agentic Stop reviewer makes 1-6 of these (sequential) per turn-end.`,
      });
    } catch (e) {
      checks.push({
        name: "round-trip",
        ok: false,
        detail: `reviewer call FAILED: ${(e instanceof Error ? e.message : String(e)).slice(0, 160)}`,
      });
    }
  }

  // 5. Deterministic gate — always available, no key required.
  checks.push({
    name: "deterministic gate",
    ok: true,
    detail: "always on (test-count contradiction; no key needed).",
  });

  // 6. Project memory — the source for decision-contradicts-finding.
  const findings = await loadProjectFindings();
  checks.push({
    name: "project memory",
    ok: findings.length > 0 ? true : "warn",
    detail:
      findings.length > 0
        ? `${findings.length} findings loaded (decision-contradicts-finding can fire).`
        : "no memory findings found — decision-contradicts-finding checks won't fire.",
  });

  // 7. Trace capture — vouch must be able to write its trace into the project dir.
  const dir = opts.dir ?? baseDir();
  const probe = join(dir, ".vouch-doctor-probe");
  try {
    await Bun.write(probe, "ok");
    await rm(probe, { force: true });
    checks.push({
      name: "trace capture",
      ok: true,
      detail: `writable (${join(dir, ".vouch-trace.jsonl")}).`,
    });
  } catch (e) {
    checks.push({
      name: "trace capture",
      ok: false,
      detail: `cannot write to ${dir}: ${(e instanceof Error ? e.message : String(e)).slice(0, 100)}`,
    });
  }

  return checks;
}

export function formatDoctor(checks: DoctorCheck[]): string {
  const icon = (ok: boolean | "warn") => (ok === true ? "✓" : ok === "warn" ? "⚠" : "✗");
  const lines = ["vouch doctor\n"];
  for (const c of checks) lines.push(`  ${icon(c.ok)} ${c.name.padEnd(19)} ${c.detail}`);
  const broken = checks.filter((c) => c.ok === false);
  lines.push("");
  lines.push(
    broken.length === 0
      ? "  all green — vouch will review and gate in this environment."
      : `  ${broken.length} blocking issue(s): ${broken.map((c) => c.name).join(", ")}.`,
  );
  return lines.join("\n");
}
