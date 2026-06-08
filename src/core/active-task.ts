// Active-task pointer — which (store, task) the agent is currently working on.
//
// Lives as a small JSON file at <baseDir>/.vouch-active (see baseDir below) so
// hooks (which run as separate processes, with no store handle) can find the
// active task without guessing a store. The pointer names the store root
// explicitly, so it is unambiguous even while multiple stores exist (it does
// not assume a single canonical store).
import { existsSync } from "node:fs";
import { rm } from "node:fs/promises";
import { join } from "node:path";

export interface ActiveTask {
  /** store root, e.g. ".vouch" or ".vouch-design-rc-independent" */
  root: string;
  task_id: string;
}

// The PostToolUse hook writes the pointer + trace under $CLAUDE_PROJECT_DIR.
// Anchor the TS side to the same dir so `vouch hook ingest` (run as a SessionEnd
// hook, whose cwd may differ) reads from where the events were actually written.
// `||` (not `??`) so a set-but-empty var also falls through to cwd; a real
// project dir is always a non-empty path. When the var is absent (e.g. `vouch
// use` run directly from a terminal), this falls back to cwd as before.
export function baseDir(): string {
  return process.env.CLAUDE_PROJECT_DIR || process.cwd();
}

function pointerPath(dir: string): string {
  return join(dir, ".vouch-active");
}

export async function writeActiveTask(a: ActiveTask, dir = baseDir()): Promise<void> {
  await Bun.write(pointerPath(dir), JSON.stringify(a));
}

export async function readActiveTask(dir = baseDir()): Promise<ActiveTask | null> {
  const p = pointerPath(dir);
  if (!existsSync(p)) return null;
  try {
    const a = (await Bun.file(p).json()) as Partial<ActiveTask>;
    if (typeof a.root === "string" && typeof a.task_id === "string") {
      return { root: a.root, task_id: a.task_id };
    }
    return null;
  } catch {
    return null;
  }
}

export async function clearActiveTask(dir = baseDir()): Promise<void> {
  await rm(pointerPath(dir), { force: true });
}

// PostToolUse trace log — a cheap append-only JSONL of tool events. The hook
// appends each event with a fast shell command (no vouch spawn); `vouch hook
// ingest` later turns the log into observed evidence on the active task. The
// file's existence is the capture switch: `vouch use` creates it (opt-in),
// ingest removes it (so capture stays off until the next `use`).
function tracePath(dir: string): string {
  return join(dir, ".vouch-trace.jsonl");
}

export async function enableTrace(dir = baseDir()): Promise<void> {
  await Bun.write(tracePath(dir), "");
}

// Append one compacted JSON line to the trace. Always-on: creates the file on
// first write if it doesn't exist. Every tool call is evidence; gating capture
// behind `vouch use` left a cooperative gap where vanilla tool calls produced
// no evidence for the commit-block to check against.
export async function appendTrace(event: unknown, dir = baseDir()): Promise<boolean> {
  const p = tracePath(dir);
  const line = `${JSON.stringify(event)}\n`;
  const fh = await import("node:fs/promises");
  await fh.appendFile(p, line);
  return true;
}

export async function readTrace(dir = baseDir()): Promise<Record<string, unknown>[]> {
  const p = tracePath(dir);
  if (!existsSync(p)) return [];
  const text = await Bun.file(p).text();
  const events: Record<string, unknown>[] = [];
  for (const line of text.split("\n")) {
    const t = line.trim();
    if (!t) continue;
    try {
      events.push(JSON.parse(t) as Record<string, unknown>);
    } catch {
      // skip a malformed line rather than abort the whole ingest
    }
  }
  return events;
}

export async function clearTrace(dir = baseDir()): Promise<void> {
  await rm(tracePath(dir), { force: true });
}
