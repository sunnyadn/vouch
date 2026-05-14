#!/usr/bin/env bun
/**
 * session_metrics.ts — per-transcript humility + fire metrics aggregator.
 *
 * Reads session_claims from ~/.vouch/store.db across all known Claude Code
 * transcripts (basename of *.jsonl under ~/.claude/projects/) and emits a
 * row per transcript with:
 *
 *   - total claims (gate processed)
 *   - verdict breakdown (grounded / ungrounded / reclassified / retracted)
 *   - stance breakdown (asserts / hedges / speculates / other)
 *   - humility ratio (H+S over A+H+S)
 *   - revise outcomes (addressed_via: fetch / hedge / remove / null)
 *   - awaiting_revise backlog
 *
 * Forward-looking measurement: rerun weekly to track whether today's
 * primitive ship moves the dial. The 2026-05-14 baseline numbers are the
 * starting point for comparison.
 *
 * Outputs JSONL by default; --format csv for spreadsheet ingestion;
 * --aggregate for a single summary row over all transcripts.
 *
 * Usage:
 *   ./session_metrics.ts                                   # all transcripts, jsonl
 *   ./session_metrics.ts --since 2026-05-14                # filter by transcript mtime
 *   ./session_metrics.ts --aggregate                       # single summary row
 *   ./session_metrics.ts --format csv > metrics.csv
 */
import { readdirSync, statSync, readFileSync } from "node:fs";
import { homedir } from "node:os";
import { basename, join } from "node:path";
import * as store from "../../src/store.ts";

interface Args {
  since?: string; // ISO date — filter transcripts whose mtime >= this
  format: "jsonl" | "csv";
  aggregate: boolean;
}

function parseArgs(): Args {
  const args = process.argv.slice(2);
  const out: Args = { format: "jsonl", aggregate: false };
  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (a === "--since") out.since = args[++i];
    else if (a === "--format") {
      const v = args[++i];
      if (v !== "jsonl" && v !== "csv") {
        console.error(`bad --format: ${v} (jsonl | csv)`);
        process.exit(2);
      }
      out.format = v;
    } else if (a === "--aggregate") out.aggregate = true;
    else if (a === "-h" || a === "--help") {
      console.error("Usage: session_metrics.ts [--since YYYY-MM-DD] [--format jsonl|csv] [--aggregate]");
      process.exit(0);
    }
  }
  return out;
}

interface SessionMetricRow {
  transcript_id: string;
  project_dir: string;
  total_claims: number;
  grounded: number;
  ungrounded: number;
  reclassified: number;
  retracted: number;
  asserts: number;
  hedges: number;
  speculates: number;
  truth_bearing: number;
  humility_pct: number | null;
  awaiting_revise: number;
  addressed_fetch: number;
  addressed_hedge: number;
  addressed_remove: number;
  addressed_total: number;
}

function getAddressedCounts(transcript_id: string): {
  fetch: number;
  hedge: number;
  remove: number;
} {
  const rows = store
    .getDb()
    .prepare(
      `SELECT addressed_via, COUNT(*) AS n
       FROM session_claims
       WHERE transcript_id = ? AND addressed_via IS NOT NULL
       GROUP BY addressed_via`,
    )
    .all(transcript_id) as Array<{ addressed_via: string; n: number }>;
  const counts = { fetch: 0, hedge: 0, remove: 0 };
  for (const r of rows) {
    if (r.addressed_via === "fetch") counts.fetch = r.n;
    else if (r.addressed_via === "hedge") counts.hedge = r.n;
    else if (r.addressed_via === "remove") counts.remove = r.n;
  }
  return counts;
}

function rowFor(transcript_id: string, project_dir: string): SessionMetricRow {
  const c = store.getSessionFireCounts(transcript_id);
  const truth = c.asserts + c.hedges + c.speculates;
  const hedged = c.hedges + c.speculates;
  const humility_pct = truth > 0 ? Number(((hedged / truth) * 100).toFixed(1)) : null;
  const addressed = getAddressedCounts(transcript_id);
  return {
    transcript_id,
    project_dir,
    total_claims: c.total,
    grounded: c.grounded,
    ungrounded: c.ungrounded,
    reclassified: c.reclassified,
    retracted: c.retracted,
    asserts: c.asserts,
    hedges: c.hedges,
    speculates: c.speculates,
    truth_bearing: truth,
    humility_pct,
    awaiting_revise: c.awaiting_revise,
    addressed_fetch: addressed.fetch,
    addressed_hedge: addressed.hedge,
    addressed_remove: addressed.remove,
    addressed_total: addressed.fetch + addressed.hedge + addressed.remove,
  };
}

function listTranscripts(since?: string): Array<{ transcript_id: string; project_dir: string; mtime: number }> {
  const root = join(homedir(), ".claude", "projects");
  const out: Array<{ transcript_id: string; project_dir: string; mtime: number }> = [];
  let dirs: string[];
  try {
    dirs = readdirSync(root);
  } catch {
    return [];
  }
  const sinceMs = since ? Date.parse(since) : 0;
  for (const d of dirs) {
    const fullDir = join(root, d);
    let stat;
    try {
      stat = statSync(fullDir);
    } catch {
      continue;
    }
    if (!stat.isDirectory()) continue;
    let files: string[];
    try {
      files = readdirSync(fullDir);
    } catch {
      continue;
    }
    for (const f of files) {
      if (!f.endsWith(".jsonl")) continue;
      const fullPath = join(fullDir, f);
      try {
        const fstat = statSync(fullPath);
        if (fstat.mtimeMs < sinceMs) continue;
        out.push({
          transcript_id: basename(f, ".jsonl"),
          project_dir: d,
          mtime: fstat.mtimeMs,
        });
      } catch {}
    }
  }
  out.sort((a, b) => b.mtime - a.mtime);
  return out;
}

function aggregate(rows: SessionMetricRow[]): SessionMetricRow {
  const sum = (k: keyof SessionMetricRow) =>
    rows.reduce((acc, r) => acc + ((r[k] as number) || 0), 0);
  const total = sum("total_claims");
  const asserts = sum("asserts");
  const hedges = sum("hedges");
  const speculates = sum("speculates");
  const truth = asserts + hedges + speculates;
  const hedged = hedges + speculates;
  return {
    transcript_id: "<aggregate>",
    project_dir: `${rows.length} transcripts`,
    total_claims: total,
    grounded: sum("grounded"),
    ungrounded: sum("ungrounded"),
    reclassified: sum("reclassified"),
    retracted: sum("retracted"),
    asserts,
    hedges,
    speculates,
    truth_bearing: truth,
    humility_pct: truth > 0 ? Number(((hedged / truth) * 100).toFixed(1)) : null,
    awaiting_revise: sum("awaiting_revise"),
    addressed_fetch: sum("addressed_fetch"),
    addressed_hedge: sum("addressed_hedge"),
    addressed_remove: sum("addressed_remove"),
    addressed_total: sum("addressed_total"),
  };
}

function emit(rows: SessionMetricRow[], format: "jsonl" | "csv"): void {
  if (format === "csv") {
    const cols = Object.keys(rows[0] ?? {}) as Array<keyof SessionMetricRow>;
    if (!cols.length) return;
    console.log(cols.join(","));
    for (const r of rows) {
      console.log(cols.map((c) => String(r[c] ?? "")).join(","));
    }
  } else {
    for (const r of rows) console.log(JSON.stringify(r));
  }
}

function main(): void {
  const args = parseArgs();
  const transcripts = listTranscripts(args.since);
  const rows: SessionMetricRow[] = [];
  for (const t of transcripts) {
    const r = rowFor(t.transcript_id, t.project_dir);
    if (r.total_claims === 0) continue; // no claims for this transcript
    rows.push(r);
  }
  if (args.aggregate) {
    if (!rows.length) {
      console.error("(no transcripts with claims to aggregate)");
      return;
    }
    emit([aggregate(rows)], args.format);
  } else {
    emit(rows, args.format);
  }
  console.error(`(${rows.length} transcripts with claims${args.since ? `, since ${args.since}` : ""})`);
}

main();
