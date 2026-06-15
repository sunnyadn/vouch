// Reviewer agent — the LLM-powered anti-hallucination reviewer.
//
// Triggered at PreToolUse (commit) and Stop, the reviewer gets the full session
// evidence context and reasons about:
//   1. Active fabrication — claim contradicts observed evidence
//   2. Passive fabrication — claim has NO supporting evidence (conclusion
//      without investigation)
//   3. Research sufficiency — scope of claim vs breadth of evidence gathered
//   4. Decision contradicts finding — repeats a documented mistake
//   5. Premise unexamined — executes an instruction without checking whether the
//      path is still warranted (a DECISION error, not a grounding error). This
//      lens came from a falsification experiment: the active ingredient is
//      RE-FRAMING the agent's action as a decision to evaluate and handing it to
//      a fresh reasoner (which the reviewer already is) — not any specific
//      question set. See bench/premise-review/probe.ts.
//   6. Omission — important negative signals not acknowledged
//
// The reviewer replaces the human who reads the terminal and catches bullshit.
// It is an LLM, not a regex engine — it can reason about research sufficiency
// in ways deterministic checks can't ("you edited 5 files but only read 2 of
// them", "you claim the auth bug is fixed but never reproduced it").
//
// INJECTABLE: the reviewer function is injectable so gate logic is testable
// with a deterministic fake. FAIL-OPEN: a reviewer failure never blocks.

import { extractJsonObject } from "./contradiction.ts";
import { type CapturedEvent, unresolvedNegatives } from "./evidence-capture.ts";
import { hasAnthropicCreds } from "./nli.ts";

export interface ReviewIssue {
  type:
    | "active-fabrication"
    | "passive-fabrication"
    | "research-insufficiency"
    | "decision-contradicts-finding"
    | "premise-unexamined"
    | "omission";
  severity: "block" | "warn";
  detail: string;
  /** The exact verbatim span from the action that is ungrounded — rendered highlighted so a human spots it. */
  quote?: string;
  suggestion?: string;
}

export interface ReviewVerdict {
  issues: ReviewIssue[];
  ok: boolean;
  /**
   * Whether the LLM reviewer actually completed a round-trip on this call. An empty
   * `issues` array is ambiguous on its own: it could mean "reviewed, found nothing"
   * OR "couldn't review at all". Without this distinction a DEAD reviewer (drained
   * quota, bad key, timeout) is byte-identical to a CLEAN pass — the silent fail-open.
   *   reviewed — a real round-trip completed (clean or with issues)
   *   skipped  — intentionally not run (no API key configured)
   *   failed   — a key IS configured but the call errored / ran out of turns (fail-open)
   * Optional: the deterministic gate and callers that don't reach the LLM leave it unset.
   */
  status?: "reviewed" | "skipped" | "failed";
  /**
   * The reviewer's query_history trail this review: every pattern it searched and how many
   * events matched. Lets a cry-wolf post-mortem distinguish "never queried" from "queried the
   * wrong term and got 0 hits" — the two have different fixes and were previously only visible
   * on ephemeral VOUCH_DIAG stderr. Persisted into the corpus next to the verdict.
   */
  queries?: { pattern: string; hits: number }[];
}

export type ReviewFn = (context: ReviewContext) => Promise<ReviewVerdict>;

export interface ReviewContext {
  action: string;
  actionType: "commit" | "stop-response" | "edit";
  evidence: EvidenceSummary;
  projectFindings?: string[];
}

export interface EvidenceSummary {
  filesRead: string[];
  filesEdited: string[];
  bashCommands: { command: string; exitCode: number; stdoutSnippet: string }[];
  webSearches: number;
  unresolvedFailures: string[];
  totalToolCalls: number;
}

// Keep the HEAD and TAIL of a command's stdout. Verification results, final counts, and
// cmp/diff/grep outcomes print at the END — the old head-only `slice(0,500)` dropped them,
// so the reviewer couldn't see grounding at the bottom of a multi-line command and
// false-flagged the claim as fabrication (the truncation cry-wolf). The trace already holds
// the full output; keep both ends, elide only a genuinely huge middle.
function sampleStdout(s: string, head = 1200, tail = 600): string {
  if (s.length <= head + tail) return s;
  return `${s.slice(0, head)}\n…[${s.length - head - tail} chars elided]…\n${s.slice(-tail)}`;
}

export function buildEvidenceSummary(events: CapturedEvent[]): EvidenceSummary {
  const filesRead: string[] = [];
  const filesEdited: string[] = [];
  const bashCommands: EvidenceSummary["bashCommands"] = [];
  let webSearches = 0;

  for (const e of events) {
    if (e.tool === "Read" && e.filePath) filesRead.push(e.filePath);
    if ((e.tool === "Edit" || e.tool === "Write") && e.filePath) filesEdited.push(e.filePath);
    if (e.tool === "WebSearch" || e.tool === "WebFetch") webSearches++;
    if (e.tool === "Bash" && e.command) {
      bashCommands.push({
        command: e.command,
        exitCode: e.exitCode,
        stdoutSnippet: sampleStdout(e.stdout),
      });
    }
  }

  return {
    filesRead: [...new Set(filesRead)],
    filesEdited: [...new Set(filesEdited)],
    bashCommands,
    webSearches,
    unresolvedFailures: unresolvedNegatives(events).map((e) => e.command!),
    totalToolCalls: events.length,
  };
}

export const DEFAULT_MODEL = "claude-sonnet-4-5-20250929";

const VALID_TYPES = new Set([
  "active-fabrication",
  "passive-fabrication",
  "research-insufficiency",
  "decision-contradicts-finding",
  "premise-unexamined",
  "omission",
]);
const VALID_SEVERITIES = new Set(["block", "warn"]);

export function parseReviewResponse(text: string): ReviewVerdict {
  const json = extractJsonObject(text);
  if (!json) return { issues: [], ok: true };
  const parsed = JSON.parse(json) as { issues?: unknown[] };
  if (!Array.isArray(parsed.issues)) return { issues: [], ok: true };
  const issues: ReviewIssue[] = [];
  for (const raw of parsed.issues) {
    if (!raw || typeof raw !== "object") continue;
    const r = raw as Record<string, unknown>;
    const type =
      typeof r.type === "string" && VALID_TYPES.has(r.type)
        ? (r.type as ReviewIssue["type"])
        : null;
    const severity =
      typeof r.severity === "string" && VALID_SEVERITIES.has(r.severity)
        ? (r.severity as ReviewIssue["severity"])
        : "warn";
    const detail = typeof r.detail === "string" ? r.detail : "";
    const quote = typeof r.quote === "string" && r.quote.trim() ? r.quote.trim() : undefined;
    const suggestion = typeof r.suggestion === "string" ? r.suggestion : undefined;
    if (type && detail) issues.push({ type, severity, detail, quote, suggestion });
  }
  return { issues, ok: issues.length === 0 };
}

// ANSI color for the ungrounded span. Coloring the offending text reads cleaner than an
// emoji marker line (user pref); honors the NO_COLOR convention for displays without ANSI.
const ANSI = { red: "\x1b[31m", yellow: "\x1b[33m", reset: "\x1b[0m" };
function paint(c: "red" | "yellow", s: string): string {
  return process.env.NO_COLOR != null ? s : `${ANSI[c]}${s}${ANSI.reset}`;
}

// Plain-language reason shown next to the colored span, so a human sees WHY it's flagged
// without decoding the jargon type name.
const ISSUE_LABEL: Record<ReviewIssue["type"], string> = {
  "active-fabrication": "contradicts evidence",
  "passive-fabrication": "no evidence",
  "research-insufficiency": "under-researched",
  "decision-contradicts-finding": "contradicts a known finding",
  "premise-unexamined": "premise unchecked",
  "omission": "ignores a failure",
};

// The reviewer failed open (a key IS configured but the call errored / timed out). Surface
// it as a NON-breaking advise line so an outage is VISIBLE instead of silently catching
// nothing — the one signal that turns "vouch went quiet" from a trap into a prompt to act.
// Returns "" for the healthy/skipped cases so it never adds noise to a normal turn.
export function formatReviewerHealthNote(verdict: ReviewVerdict): string {
  if (verdict.status !== "failed") return "";
  return (
    "⚠ vouch reviewer unavailable — it failed open and reviewed nothing this turn " +
    "(drained quota, bad key, or timeout). You are UNGATED until it recovers. Run `vouch doctor`."
  );
}

export function formatReviewMessage(verdict: ReviewVerdict): string {
  if (verdict.ok) return "";
  const blocks = verdict.issues.filter((i) => i.severity === "block");
  const warns = verdict.issues.filter((i) => i.severity === "warn");
  const lines: string[] = [];

  if (blocks.length > 0) {
    const hasFinding = blocks.some((i) => i.type === "decision-contradicts-finding");
    const header = hasFinding
      ? "⛔ vouch reviewer (BLOCK): action contradicts project knowledge:"
      : "⛔ vouch reviewer (BLOCK): fabrication detected:";
    lines.push(header);
    for (const i of blocks) {
      lines.push(`  • [${i.type}] ${i.detail}`);
      if (i.quote) lines.push(`    ${paint("red", `${ISSUE_LABEL[i.type]}: "${i.quote}"`)}`);
      if (i.suggestion) lines.push(`    → ${i.suggestion}`);
    }
  }

  if (warns.length > 0) {
    lines.push("⚠ vouch reviewer (advise): research or grounding concerns:");
    for (const w of warns) {
      lines.push(`  • [${w.type}] ${w.detail}`);
      if (w.quote) lines.push(`    ${paint("yellow", `${ISSUE_LABEL[w.type]}: "${w.quote}"`)}`);
      if (w.suggestion) lines.push(`    → ${w.suggestion}`);
    }
  }

  return `${lines.join("\n")}\n`;
}

export function shouldCallReviewer(): boolean {
  return hasAnthropicCreds() && !process.env.VOUCH_REVIEWER_OFF;
}

// The reviewer's project knowledge is the project's auto-memory — the same
// `~/.claude/projects/<encoded>/memory/*.md` files the SDK loads for the agent.
// The reviewer reads them so it can check whether the agent APPLIED a lesson it
// already knows. (There is no separate finding store; it was retired once the
// audit showed it was redundant with auto-memory.)
export async function loadProjectFindings(): Promise<string[]> {
  const results: string[] = [];
  try {
    const { existsSync, readdirSync, readFileSync } = await import("node:fs");
    const { join } = await import("node:path");
    const homedir = (await import("node:os")).homedir();
    const { baseDir } = await import("./active-task.ts");
    const encoded = baseDir().replace(/\//g, "-");
    const memoryDir = join(homedir, ".claude/projects", encoded, "memory");
    if (existsSync(memoryDir)) {
      for (const file of readdirSync(memoryDir)) {
        if (!file.endsWith(".md") || file === "MEMORY.md") continue;
        try {
          const content = readFileSync(join(memoryDir, file), "utf8");
          const descMatch = content.match(/^description:\s*(.+)$/m);
          if (descMatch) results.push(`[memory/${file}] ${descMatch[1]}`);
        } catch { /* unreadable file → skip */ }
      }
    }
  } catch { /* no memory → skip */ }

  return results;
}
