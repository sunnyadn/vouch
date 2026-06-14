// Always-on session evidence from the PostToolUse trace.
//
// Every tool call's result is captured by the PostToolUse hook into
// .vouch-trace.jsonl. This module reads that trace and provides structured
// queries so the reviewer checks (commit-block, research-sufficiency, grounding
// injection, omission) can verify claims against observed evidence WITHOUT
// requiring cooperative `vouch evidence exec` recording.
//
// The two rules:
//   PostToolUse(*) → record as evidence
//   PreToolUse(*) + Stop → check claims against evidence

import { commandMatchesKind, type RunRow } from "./contradiction.ts";
import type { OwnWorkKind } from "./extractor.ts";

export interface CapturedEvent {
  tool: string;
  command?: string;
  filePath?: string;
  stdout: string;
  stderr: string;
  exitCode: number;
  timestamp?: string;
  isNegative: boolean;
}

const TRACE_MAX = 16_000;
function cap(s: string): string {
  return s.length > TRACE_MAX
    ? `${s.slice(0, TRACE_MAX)}\n…[truncated ${s.length - TRACE_MAX} chars]`
    : s;
}

function extractToolInfo(event: Record<string, unknown>): {
  tool: string;
  command?: string;
  filePath?: string;
} {
  const tool = typeof event.tool_name === "string" ? event.tool_name : "";
  const input = (event.tool_input ?? {}) as Record<string, unknown>;
  const command = tool === "Bash" && typeof input.command === "string" ? input.command : undefined;
  const filePath = typeof input.file_path === "string" ? input.file_path : undefined;
  return { tool, command, filePath };
}

function extractOutput(event: Record<string, unknown>): {
  stdout: string;
  stderr: string;
  exitCode: number;
} {
  if (event.hook_event_name === "PostToolUseFailure") {
    const errText = typeof event.error === "string" ? event.error : "";
    const m = errText.match(/^Exit code (\d+)\n?/);
    if (m) return { stdout: "", stderr: cap(errText.slice(m[0].length)), exitCode: Number(m[1]) };
    return { stdout: "", stderr: cap(errText), exitCode: 1 };
  }
  const resp = event.tool_response;
  if (typeof resp === "string") return { stdout: cap(resp), stderr: "", exitCode: 0 };
  if (resp && typeof resp === "object") {
    const r = resp as Record<string, unknown>;
    const file = r.file as Record<string, unknown> | undefined;
    if (file && typeof file.content === "string") {
      return { stdout: cap(file.content), stderr: "", exitCode: 0 };
    }
    const stdout = cap(String(r.stdout ?? r.output ?? r.result ?? JSON.stringify(r)));
    const stderr = cap(String(r.stderr ?? r.error ?? ""));
    const exitCode =
      Number(r.exit_code ?? r.exitCode ?? (r.error || r.is_error ? 1 : 0)) || 0;
    return { stdout, stderr, exitCode };
  }
  return { stdout: "", stderr: "", exitCode: 0 };
}

// Edit/Write/MultiEdit inputs ARE the change applied — an OBSERVED state mutation (the file
// now contains new_string), confirmed by the success response, not a free-floating assertion.
// Capturing the hunk lets the reviewer verify edit narration ("renamed X→Y") via query_history;
// without it the trace holds the file PATH but not WHAT changed, so honest edit reports read as
// ungrounded (the R5 false-positive class). Returns undefined for non-edit tools / malformed input.
function extractEditDiff(tool: string, input: Record<string, unknown>): string | undefined {
  if (tool === "Write" && typeof input.content === "string") return `Wrote file:\n${input.content}`;
  if (tool === "Edit" && typeof input.old_string === "string" && typeof input.new_string === "string") {
    return `Edited file:\n- ${input.old_string}\n+ ${input.new_string}`;
  }
  if (tool === "MultiEdit" && Array.isArray(input.edits)) {
    const hunks = input.edits
      .filter((e): e is { old_string: string; new_string: string } => {
        const o = e as { old_string?: unknown; new_string?: unknown } | null;
        return typeof o?.old_string === "string" && typeof o?.new_string === "string";
      })
      .map((e) => `- ${e.old_string}\n+ ${e.new_string}`);
    if (hunks.length) return `Edited file (${hunks.length} hunks):\n${hunks.join("\n")}`;
  }
  return undefined;
}

// Non-zero count before a failure word ("5 errors", "2 failed", "1 failure").
// "0 fail" is a success, not a negative signal.
const NEGATIVE_COUNT_RE = /\b[1-9]\d*\s+(?:fail|error|fatal)\w*/i;
// Standalone keywords that signal failure regardless of count.
const NEGATIVE_KEYWORD_RE = /\b(?:fatal|panic|exception|abort|denied|refused)\b/i;

export function parseCapturedEvents(raw: Record<string, unknown>[]): CapturedEvent[] {
  const events: CapturedEvent[] = [];
  for (const r of raw) {
    const { tool, command, filePath } = extractToolInfo(r);
    if (!tool) continue;
    const out = extractOutput(r);
    const { stderr, exitCode } = out;
    let stdout = out.stdout;
    // For a SUCCESSFUL edit, replace the bland "updated successfully" response with the actual
    // hunk so query_history surfaces what changed; a FAILED edit keeps its error output.
    const editDiff = exitCode === 0 ? extractEditDiff(tool, (r.tool_input ?? {}) as Record<string, unknown>) : undefined;
    if (editDiff) stdout = cap(editDiff);
    const ts = typeof r.timestamp === "string" ? r.timestamp : undefined;
    const combined = `${stdout}\n${stderr}`;
    // Don't keyword-scan edit CONTENT for negatives — a hunk may legitimately contain
    // "error"/"fail" text (editing error-handling code) and that's not a failure signal.
    const isNegative =
      exitCode !== 0 || (!editDiff && (NEGATIVE_COUNT_RE.test(combined) || NEGATIVE_KEYWORD_RE.test(combined)));
    events.push({ tool, command, filePath, stdout, stderr, exitCode, timestamp: ts, isNegative });
  }
  return events;
}

export function findLatestRun(events: CapturedEvent[], kind: OwnWorkKind): RunRow | null {
  for (let i = events.length - 1; i >= 0; i--) {
    const e = events[i]!;
    if (e.tool !== "Bash" || !e.command) continue;
    if (commandMatchesKind(kind, e.command)) {
      const combined = [e.stdout, e.stderr].filter((s) => s.trim()).join("\n");
      return { command: e.command, stdout: combined };
    }
  }
  return null;
}

export function hasWebSearch(events: CapturedEvent[]): boolean {
  return events.some(
    (e) =>
      e.tool === "WebSearch" ||
      e.tool === "WebFetch" ||
      (e.tool === "Bash" && e.command && /\bcurl\b|\bwget\b|\bfetch\b/.test(e.command)),
  );
}

const ABSENCE_CLAIM_RE =
  /\b(?:I don't know of|I'm not aware of|there(?:'s| is) no|doesn't exist|no (?:open|existing|known|public|widely[- ]used) (?:benchmark|dataset|tool|library|framework|standard))\b/i;

export function hasAbsenceClaimWithoutSearch(
  response: string,
  events: CapturedEvent[],
): boolean {
  return ABSENCE_CLAIM_RE.test(response) && !hasWebSearch(events);
}

export function filesReadInSession(events: CapturedEvent[]): Set<string> {
  const paths = new Set<string>();
  for (const e of events) {
    if (e.tool === "Read" && e.filePath) paths.add(e.filePath);
  }
  return paths;
}

export function unresolvedNegatives(events: CapturedEvent[]): CapturedEvent[] {
  const laterSuccess = new Map<string, boolean>();
  const negatives: CapturedEvent[] = [];
  for (let i = events.length - 1; i >= 0; i--) {
    const e = events[i]!;
    if (e.tool !== "Bash" || !e.command) continue;
    if (!e.isNegative) {
      laterSuccess.set(e.command, true);
    } else if (!laterSuccess.get(e.command)) {
      negatives.push(e);
    }
  }
  negatives.reverse();
  return negatives;
}

const TEST_CMD_RE = /\b(test|tests|jest|vitest|pytest|mocha|spec)\b/i;
const BUILD_CMD_RE = /\b(build|tsc|tsgo|biome|compile|lint|eslint)\b/i;

export function groundingSummary(event: CapturedEvent): string | null {
  if (event.tool !== "Bash" || !event.command) return null;
  const combined = [event.stdout, event.stderr].filter((s) => s.trim()).join("\n");

  if (TEST_CMD_RE.test(event.command)) {
    const passMatch = combined.match(/(\d+)\s+pass(?:ed|ing)?/i);
    const failMatch = combined.match(/(\d+)\s+fail(?:ed|ing|ure)?/i);
    if (passMatch || failMatch) {
      const pass = passMatch ? passMatch[1] : "?";
      const fail = failMatch ? failMatch[1] : "0";
      const icon = event.exitCode === 0 ? "✓" : "✗";
      return `${icon} OBSERVED: ${pass} pass, ${fail} fail (exit ${event.exitCode})`;
    }
  }

  if (BUILD_CMD_RE.test(event.command)) {
    const errorMatch = combined.match(/(\d+)\s+error/i);
    if (event.exitCode === 0) {
      return `✓ OBSERVED: build clean (exit 0)`;
    }
    if (errorMatch) {
      return `✗ OBSERVED: ${errorMatch[1]} errors (exit ${event.exitCode})`;
    }
    if (event.exitCode !== 0) {
      return `✗ OBSERVED: build failed (exit ${event.exitCode})`;
    }
  }

  return null;
}

export const GIT_COMMIT_RE = /\bgit\b[^|&;]*\bcommit\b/;

export function eventsSinceLastCommit(events: CapturedEvent[]): CapturedEvent[] {
  let cutoff = -1;
  for (let i = events.length - 1; i >= 0; i--) {
    const e = events[i]!;
    if (
      e.tool === "Bash" &&
      e.command &&
      GIT_COMMIT_RE.test(e.command) &&
      e.exitCode === 0
    ) {
      cutoff = i;
      break;
    }
  }
  return cutoff === -1 ? events : events.slice(cutoff + 1);
}

// A `git commit`/`git add` is the agent ASSERTING + ACTING, not an OBSERVATION:
// the command text carries the agent's own claims (the -m message), and a FAILED
// attempt reads as "the work failed". Letting a gate adjudicate a grounding claim
// against it compares claim-to-claim (or to a self-action) and manufactures false
// contradictions / "fabricated" verdicts — a category error. Gate evidence must be
// OBSERVED RESULTS (tool outputs), never the agent's own assertions; strip these
// before any gate reads the trace.
const SELF_ASSERTION_RE = /\bgit\s+(?:commit|add)\b/;
export function isObservation(e: CapturedEvent): boolean {
  return !(e.tool === "Bash" && !!e.command && SELF_ASSERTION_RE.test(e.command));
}
export function observationsOnly(events: CapturedEvent[]): CapturedEvent[] {
  return events.filter(isObservation);
}
