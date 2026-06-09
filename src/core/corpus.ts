// corpus.ts — persist every reviewer verdict to an append-only JSONL so dogfooding
// ACCUMULATES a labeled corpus (claim + the trace it was judged against + verdict +
// blocked?) instead of writing each catch to stderr where it evaporates. This is what
// turns "I ran vouch for a week" into "here are the N conclusions it flagged — K real,
// M cry-wolf."
//
// Each record stores the EVENTS the agentic (deployed) reviewer actually saw, so a record
// is REPLAYABLE THROUGH THE REAL REVIEWER: anthropicReviewerAgentic({action, events}) re-runs
// the exact case under a changed prompt / model / consensus. (We store raw events, not a
// pre-baked EvidenceSummary, because the deployed reviewer consumes events — and the summary
// is derivable from them via buildEvidenceSummary, so events are strictly more useful.)
//
// Local-only (never sent anywhere; the reviewer already ships this evidence to the LLM,
// so a local log is strictly less exposure). Path: ~/.claude/vouch-corpus.jsonl, or
// $VOUCH_CORPUS_PATH. Fail-open: a logging error must NEVER break the session.

import { appendFileSync, readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import type { CapturedEvent } from "./evidence-capture.ts";
import type { ReviewVerdict } from "./reviewer.ts";

// Exported so corpus readers (e.g. bench/deepseek-eval) resolve the SAME location the writer
// uses — a hard-copied default would silently read the wrong/empty file if this ever drifts.
export function corpusPath(): string {
  return process.env.VOUCH_CORPUS_PATH ?? join(homedir(), ".claude", "vouch-corpus.jsonl");
}

// Human-labeled reviewer MISSES — cases where vouch reviewed and stayed silent but a human
// judged a claim fabricated/ungrounded. Separate from the corpus because these carry GOLD
// labels (the human is the labeler, ⟂ the reviewer being judged), which is exactly what the
// corpus's self-labels can't give us. Feeds bench/deepseek-eval as recall (expect FIRE) cases.
export function missesPath(): string {
  return process.env.VOUCH_MISSES_PATH ?? join(homedir(), ".claude", "vouch-misses.jsonl");
}

// Record a reviewer MISS: snapshot the MOST RECENT corpus record (the review that just
// happened and missed) — its response + the exact trace vouch saw — alongside the human's note
// of which claim was ungrounded. The latest record is the right one in the common case (the
// user calls out the response right after it lands). Self-contained + replayable. Fail-open:
// a logging error must never break the session.
export function flagMiss(note: string): { sourceTs?: string; events?: number; ageSec?: number } {
  let source: { ts?: string; events?: CapturedEvent[] } | undefined;
  try {
    const lines = readFileSync(corpusPath(), "utf8").split("\n").filter(Boolean);
    const last = lines.at(-1);
    if (last) source = JSON.parse(last);
  } catch {
    // no corpus yet — still record the note (just without an attached trace)
  }
  try {
    // Store the whole source record as-is: keeping vouch's own verdict (status/issues) next to
    // the human's FIRE label IS the signal — "reviewed clean, but a human caught it".
    const entry = {
      ts: new Date().toISOString(),
      kind: "missed", // vouch reviewed and stayed silent; a human says it should have FIRED
      expect: "FIRE",
      note,
      source: source ?? null,
    };
    appendFileSync(missesPath(), `${JSON.stringify(entry)}\n`);
  } catch {
    // never break the session
  }
  // Surface the source's age so attaching the WRONG (stale) trace is auditable, not silent —
  // the latest record is only the right one if the user flags right after the response lands.
  const ageSec = source?.ts
    ? Math.round((Date.now() - new Date(source.ts).getTime()) / 1000)
    : undefined;
  return { sourceTs: source?.ts, events: source?.events?.length, ageSec };
}

// Bound the stored events so a record stays ~KB-sized but stays REPLAYABLE: the agentic
// reviewer queries command/filePath/stdout, so keep all fields, cap each captured output near
// the reviewer's own per-hit cap (1500), and keep the most recent events (its index shows the
// last ~40 and queries reach the rest). A replay then sees essentially what the reviewer saw.
const MAX_EVENTS = 120;
const PER_OUTPUT = 1200;
function trimEvents(events: CapturedEvent[]): CapturedEvent[] {
  return events.slice(-MAX_EVENTS).map((e) => ({
    ...e,
    stdout: (e.stdout ?? "").slice(0, PER_OUTPUT),
    stderr: (e.stderr ?? "").slice(0, PER_OUTPUT),
  }));
}

export function captureVerdict(rec: {
  actionType: string;
  action: string;
  events: CapturedEvent[];
  verdict: ReviewVerdict;
}): void {
  try {
    const line = JSON.stringify({
      ts: new Date().toISOString(),
      project: process.cwd(),
      actionType: rec.actionType,
      // reviewed | skipped | failed — without this an empty `issues` is ambiguous: a clean
      // pass and a fail-open death both look like []. This is what makes the corpus a
      // measurable precision sample (drop `failed` records — they reviewed nothing).
      status: rec.verdict.status,
      blocked: rec.verdict.issues.some((i) => i.severity === "block"),
      issues: rec.verdict.issues.map((i) => ({
        type: i.type,
        severity: i.severity,
        detail: i.detail,
      })),
      action: rec.action.slice(0, 2000),
      events: trimEvents(rec.events),
    });
    appendFileSync(corpusPath(), `${line}\n`);
  } catch {
    // a hook must never break the session
  }
}
