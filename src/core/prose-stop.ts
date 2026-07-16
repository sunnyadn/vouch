// Stop-hook + commit-gate helpers for the prose-gate.
//
// Live functions:
//   1. extractLastAssistantText — pull the agent's outgoing draft from JSONL
//   2. extractToolCallClaimText — claim-bearing tool-call arg text from current turn
//   3. proseCommitGate — PreToolUse commit BLOCK (deterministic + LLM paths)

import {
  anthropicContradictionJudge,
  type ContradictionJudge,
  commandMatchesKind,
  contradictionGate,
  deterministicCountContradiction,
  type RunRow,
} from "./contradiction.ts";
import {
  anthropicExtractor,
  type ExtractOwnWork,
  type OwnWorkClaim,
  type OwnWorkKind,
} from "./extractor.ts";
import { hasAnthropicCreds } from "./nli.ts";
import { looksLikeOwnWorkClaim } from "./prose-gate.ts";

// ---- 1. draft extraction --------------------------------------------------

interface TranscriptBlock {
  type?: string;
  text?: string;
}
interface ToolUseBlock {
  type?: string;
  name?: string;
  input?: Record<string, unknown>;
}
interface TranscriptRecord {
  type?: string;
  isMeta?: boolean;
  isSidechain?: boolean;
  message?: { role?: string; content?: unknown };
}

// A harness/API ERROR surfaced as an assistant-role text block is NOT the agent's prose. Reviewing it
// makes the gate fire `active-fabrication` on a SYSTEM string the agent never wrote (this session: the
// rate-limit error "API Error: Server is temporarily limiting requests …" was flagged as the agent's
// fabricated termination reason — eye-problem #5). Skip such drafts so the reviewer always reviews the
// agent's last GENUINE response, never a harness error.
//
// Each keyword must be followed by ERROR-shaped continuation (a colon, a standard HTTP status phrase, or
// end-of-string) — NOT narrative prose. A loose prefix match (`api error\b`, `429\b`, `rate[- ]?limit`)
// had a RECALL HOLE: a genuine draft STARTING with the keyword ("API error handling: I added…", "Rate
// limit handling is now implemented", "429 handling: …") was wrongly skipped → dropped from review
// entirely, so a hallucination hidden in such a draft would never be seen. Tightening closes that hole
// (falsification-tested 2026-06-24: 3 genuine-draft holes → 0, while all real errors still skip → the
// #5 cry-wolf is not reopened). Mid-sentence mentions were always safe (anchored `^`).
const HARNESS_ERROR =
  /^\s*(api error:|request (?:was )?aborted\b|request timed out\b|server is temporarily limiting|rate[- ]?limit(?:ed|\s+exceeded|\s+error|:|\s*$)|overloaded_error\b|connection error\b|fetch failed\b|network error\b|(?:429|503)\b(?:\s*$|[:.]|\s+(?:too many|service unavailable|bad gateway|temporarily|unavailable|gateway))|\(eval\):)/i;

export function isHarnessError(text: string): boolean {
  return HARNESS_ERROR.test(text.trim());
}

export function extractLastAssistantText(jsonl: string): string {
  let last = "";
  for (const line of jsonl.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    let rec: TranscriptRecord;
    try {
      rec = JSON.parse(trimmed) as TranscriptRecord;
    } catch {
      continue;
    }
    if (rec.message?.role !== "assistant") continue;
    const content = rec.message.content;
    if (!Array.isArray(content)) continue;
    const text = (content as TranscriptBlock[])
      .filter((b) => b?.type === "text" && typeof b.text === "string")
      .map((b) => b.text as string)
      .join("");
    // Skip a harness/API error masquerading as the agent's draft — keep the last GENUINE response.
    if (text.length > 0 && !isHarnessError(text)) last = text;
  }
  return last;
}

// ---- 1b. tool-call argument text (the OTHER fabrication surface) -----------

function escapeRe(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\-]/g, "\\$&");
}

const CLAIM_ARG_FLAGS = ["--message", "--note", "--reason", "--statement", "--summary", "-m"];

const CLAIM_ARG_RE_SRC =
  `(?<![\\w-])(?:${CLAIM_ARG_FLAGS.map(escapeRe).join("|")})(?:\\s+|=|(?=["']))` +
  `("(?:[^"\\\\]|\\\\.)*"|'[^']*'|[^\\s"'<>|&;()]+)`;

const CLAIM_SEP = "\n· ";

function unquoteShell(raw: string): string {
  if (raw.length >= 2 && raw[0] === '"' && raw[raw.length - 1] === '"') {
    return raw.slice(1, -1).replace(/\\(["\\$`])/g, "$1");
  }
  if (raw.length >= 2 && raw[0] === "'" && raw[raw.length - 1] === "'") {
    return raw.slice(1, -1);
  }
  return raw;
}

function isClaimAuthoringCommand(cmd: string): boolean {
  return /\bgit\b[^|&;]*\bcommit\b/.test(cmd) || /\bvouch\b/.test(cmd) || /\bcli\.ts\b/.test(cmd);
}

export function extractClaimArgsFromCommand(cmd: string): string[] {
  const out: string[] = [];
  for (const m of cmd.matchAll(new RegExp(CLAIM_ARG_RE_SRC, "g"))) {
    if (m[1]) out.push(unquoteShell(m[1]));
  }
  return out;
}

function claimTextFromToolUse(block: ToolUseBlock): string[] {
  const input = block.input;
  if (!input || typeof input !== "object") return [];
  const parts: string[] = [];
  if (typeof input.command === "string" && isClaimAuthoringCommand(input.command)) {
    parts.push(...extractClaimArgsFromCommand(input.command));
  }
  for (const key of ["message", "note", "reason", "statement", "summary"]) {
    const v = (input as Record<string, unknown>)[key];
    if (typeof v === "string" && v.trim().length > 0) parts.push(v.trim());
  }
  return parts;
}

function isUserPromptBoundary(content: unknown): boolean {
  if (typeof content === "string") return true;
  if (!Array.isArray(content)) return false;
  return (content as TranscriptBlock[]).some((b) => b?.type === "text");
}

export function extractToolCallClaimText(jsonl: string): string {
  let buf: string[] = [];
  for (const line of jsonl.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    let rec: TranscriptRecord;
    try {
      rec = JSON.parse(trimmed) as TranscriptRecord;
    } catch {
      continue;
    }
    if (rec.isMeta || rec.isSidechain) continue;
    const role = rec.message?.role;
    const content = rec.message?.content;
    if (role === "user" && isUserPromptBoundary(content)) {
      buf = [];
      continue;
    }
    if (role !== "assistant" || !Array.isArray(content)) continue;
    for (const block of content as ToolUseBlock[]) {
      if (block?.type !== "tool_use") continue;
      buf.push(...claimTextFromToolUse(block));
    }
  }
  return buf.join(CLAIM_SEP);
}

// ---- 3. PreToolUse commit gate: BLOCK -------------------------------------

import { GIT_COMMIT_RE } from "./evidence-capture.ts";

export interface CommitGateFired {
  claim: string;
  kind: OwnWorkKind;
  evidenceId?: string;
  matchedCommand?: string;
  reason: string;
}

export interface CommitGateResult {
  blocks: boolean;
  fired: CommitGateFired[];
  message: string;
}

export type CommitRunsProvider = (kind: OwnWorkKind) => Promise<RunRow[]>;

export async function proseCommitGate(args: {
  command: string;
  runsProvider: CommitRunsProvider;
  extract?: ExtractOwnWork;
  judge?: ContradictionJudge;
}): Promise<CommitGateResult> {
  const empty: CommitGateResult = { blocks: false, fired: [], message: "" };
  if (!GIT_COMMIT_RE.test(args.command)) return empty;
  const msg = extractClaimArgsFromCommand(args.command).join(CLAIM_SEP);
  if (!msg || !looksLikeOwnWorkClaim(msg)) return empty;

  const runsFor = args.runsProvider;

  // Deterministic count path (no LLM, no creds — airtight)
  const testRuns = await runsFor("test-result");
  if (testRuns[0]) {
    const det = deterministicCountContradiction(msg, testRuns[0].stdout);
    if (det?.contradicted) {
      const f: CommitGateFired = {
        claim: `${det.claimed} tests passing`,
        kind: "test-result",
        evidenceId: testRuns[0].id,
        matchedCommand: testRuns[0].command,
        reason: `deterministic count check: message claims ${det.claimed} passing, recorded run shows ${det.actual}`,
      };
      return { blocks: true, fired: [f], message: proseCommitBlockMessage([f]) };
    }
  }

  // LLM path (non-count own-work claims)
  const extract: ExtractOwnWork | null =
    args.extract ?? (hasAnthropicCreds() ? anthropicExtractor : null);
  if (!extract) return empty;
  const judge: ContradictionJudge = args.judge ?? anthropicContradictionJudge;

  let claims: OwnWorkClaim[];
  try {
    claims = await extract(msg);
  } catch {
    return empty;
  }
  if (claims.length === 0) return empty;

  const fired: CommitGateFired[] = [];
  for (const c of claims) {
    let runs: RunRow[];
    try {
      runs = await runsFor(c.kind);
    } catch {
      continue;
    }
    if (runs.length === 0) continue;
    const v = await contradictionGate(c.claim, c.kind, runs, judge);
    if (v.fires) {
      fired.push({
        claim: c.claim,
        kind: c.kind,
        evidenceId: v.matchedId,
        matchedCommand: v.matchedCommand,
        reason: v.reason,
      });
    }
  }
  if (fired.length === 0) return empty;
  return { blocks: true, fired, message: proseCommitBlockMessage(fired) };
}

export function proseCommitBlockMessage(fired: CommitGateFired[]): string {
  const lines = [
    "⛔ vouch prose-gate (BLOCK): this commit's message makes an own-work claim that a RECORDED run contradicts:",
  ];
  for (const f of fired) {
    lines.push(`  • claim: "${f.claim}"`);
    lines.push(
      `    contradicted by ${f.evidenceId ? `${f.evidenceId} ` : ""}\`${f.matchedCommand ?? "?"}\` — ${f.reason}`,
    );
  }
  lines.push(
    "  Fix the claim to match the recorded result, then re-commit.",
  );
  return `${lines.join("\n")}\n`;
}
