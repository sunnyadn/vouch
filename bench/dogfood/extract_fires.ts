#!/usr/bin/env bun
/**
 * extract_fires.ts — mine vouch gate-fires from live Claude Code transcripts
 * for vouch's dogfooding-based bench (no SQuAD-2 noise, no synthetic prompts).
 *
 * Walks all ~/.claude/projects/<repo>/<uuid>.jsonl transcripts modified in the
 * last N days, finds Stop-hook fire events (user-isMeta messages whose text
 * starts with "Stop hook feedback:" and contains a "vouch-gate ... Detected
 * ungrounded" block), and emits one JSONL row per fire with:
 *
 *   ts, transcript_id, repo (dir basename), git_branch, cwd,
 *   fire_text         — the full gate stderr block (for human classification)
 *   propositions[]    — { entity, proposition, candidates_count, reason }
 *   draft             — the assistant text that fired (preceding assistant turn)
 *   prior_user        — the user turn that preceded the draft (context)
 *   post_fire_draft   — the FIRST assistant text turn after the fire = the
 *                       revise. Needed to classify revise shape per #50:
 *                       verified / hedged / continued-confab / dodge.
 *   manual_label      — placeholder for human pass (filled by classify_fires.ts)
 *
 * Output: bench/dogfood/fires-<from>--<to>.jsonl (gitignored — contains
 * user-private session content).
 *
 * Usage:
 *   ./extract_fires.ts                       # last 14 days, all projects
 *   ./extract_fires.ts --days 7              # only last 7 days
 *   ./extract_fires.ts --project redacted-meta  # filter to one project dir
 *   ./extract_fires.ts --out fires.jsonl     # custom output path
 */
import { readFileSync, readdirSync, statSync, writeFileSync } from "node:fs";
import { join, basename } from "node:path";
import { homedir } from "node:os";

interface Args {
  days: number;
  project?: string;
  out?: string;
}

function parseArgs(): Args {
  const args = process.argv.slice(2);
  const out: Args = { days: 14 };
  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--days") out.days = Number(args[++i]);
    else if (args[i] === "--project") out.project = args[++i];
    else if (args[i] === "--out") out.out = args[++i];
    else if (args[i] === "-h" || args[i] === "--help") {
      console.error(
        "Usage: extract_fires.ts [--days N] [--project NAME] [--out PATH]",
      );
      process.exit(0);
    }
  }
  return out;
}

function getEventText(ev: any): string {
  const c = ev?.message?.content;
  if (typeof c === "string") return c;
  if (Array.isArray(c)) {
    return c
      .filter((b: any) => b?.type === "text")
      .map((b: any) => b.text || "")
      .join("\n");
  }
  return "";
}

function isFireEvent(ev: any): boolean {
  if (ev?.type !== "user") return false;
  if (!ev?.isMeta) return false;
  const t = getEventText(ev);
  return t.includes("Stop hook feedback") &&
    t.includes("vouch-gate") &&
    t.includes("Detected ungrounded");
}

/** Parse the gate-stderr block into structured proposition rows. */
function parseFireText(text: string): Array<{
  entity: string;
  proposition: string;
  candidates_count: number | null;
  reason: string;
}> {
  // Format inside the block:
  //   • <entity>: "<proposition>" (N candidate(s) found but none entailed the proposition)
  // We grep bullet rows then peel apart the pieces.
  const out = [];
  const rows = text.match(/^\s*•\s.+$/gm) || [];
  for (const row of rows) {
    const m = row.match(
      /^\s*•\s+([^:]+):\s*"([^"]+)"\s*\(([^)]*)\)\s*(.*)$/,
    );
    if (!m) continue;
    const entity = m[1]!.trim();
    const proposition = m[2]!;
    const candStr = m[3]!;
    const tail = (m[4] || "").trim();
    const candMatch = candStr.match(/(\d+)\s+candidate/);
    out.push({
      entity,
      proposition,
      candidates_count: candMatch ? Number(candMatch[1]) : null,
      reason: candStr + (tail ? " " + tail : ""),
    });
  }
  return out;
}

interface FireRow {
  ts: string;
  transcript_id: string;
  repo: string;
  git_branch?: string;
  cwd?: string;
  fire_text: string;
  propositions: ReturnType<typeof parseFireText>;
  draft: string;
  prior_user: string;
  /** First assistant text turn AFTER the fire — the revise. Empty if the
   *  session ended at the fire (rare: harness sometimes truncates). */
  post_fire_draft: string;
  manual_label: { class: string; notes?: string } | null;
}

function extractFromTranscript(path: string): FireRow[] {
  const lines = readFileSync(path, "utf8").split("\n").filter((l) => l.trim());
  const rows: FireRow[] = [];

  // Pre-parse so we can look backward for context. Skip bad lines quietly.
  const events: any[] = [];
  for (const line of lines) {
    try {
      events.push(JSON.parse(line));
    } catch {
      events.push(null);
    }
  }

  const dirName = basename(path.split("/").slice(-2)[0]!);
  const transcript_id = basename(path, ".jsonl");

  for (let i = 0; i < events.length; i++) {
    const ev = events[i];
    if (!isFireEvent(ev)) continue;

    const fire_text = getEventText(ev);
    const propositions = parseFireText(fire_text);

    // Look backward for the closest assistant text turn = the draft.
    let draft = "";
    for (let j = i - 1; j >= 0 && j >= i - 50; j--) {
      const pe = events[j];
      if (pe?.type !== "assistant") continue;
      const text = getEventText(pe);
      if (!text) continue;
      draft = text;
      break;
    }

    // Look backward beyond the draft for the user turn (the prompt that
    // triggered the draft). Skip isMeta user-events (they are hook feedback,
    // not real user turns).
    let prior_user = "";
    let sawDraft = false;
    for (let j = i - 1; j >= 0 && j >= i - 80; j--) {
      const pe = events[j];
      if (!pe) continue;
      if (!sawDraft) {
        if (pe.type === "assistant" && getEventText(pe)) sawDraft = true;
        continue;
      }
      if (pe.type === "user" && !pe.isMeta) {
        const text = getEventText(pe);
        if (text) {
          prior_user = text;
          break;
        }
      }
    }

    // Look forward for the FIRST assistant text turn after the fire — this
    // is the revise that determines the label class. Cap the search window
    // to ~50 events (typical revise lands within a handful of turns).
    let post_fire_draft = "";
    for (let j = i + 1; j < events.length && j <= i + 50; j++) {
      const pe = events[j];
      if (pe?.type !== "assistant") continue;
      const text = getEventText(pe);
      if (!text) continue;
      post_fire_draft = text;
      break;
    }

    rows.push({
      ts: ev.timestamp || "",
      transcript_id,
      repo: dirName,
      git_branch: ev.gitBranch || undefined,
      cwd: ev.cwd || undefined,
      fire_text,
      propositions,
      draft,
      prior_user,
      post_fire_draft,
      manual_label: null,
    });
  }

  return rows;
}

function main() {
  const args = parseArgs();
  const root = join(homedir(), ".claude", "projects");

  const cutoffMs = Date.now() - args.days * 24 * 3600 * 1000;
  const projectDirs = readdirSync(root)
    .filter((d) => !args.project || d.includes(args.project))
    .map((d) => join(root, d))
    .filter((d) => {
      try {
        return statSync(d).isDirectory();
      } catch {
        return false;
      }
    });

  const allRows: FireRow[] = [];
  let scanned = 0;
  let skipped = 0;
  for (const dir of projectDirs) {
    let files: string[];
    try {
      files = readdirSync(dir).filter((f) => f.endsWith(".jsonl"));
    } catch {
      continue;
    }
    for (const f of files) {
      const path = join(dir, f);
      let mtime: number;
      try {
        mtime = statSync(path).mtimeMs;
      } catch {
        continue;
      }
      if (mtime < cutoffMs) {
        skipped++;
        continue;
      }
      scanned++;
      try {
        const rows = extractFromTranscript(path);
        if (rows.length) {
          allRows.push(...rows);
          console.error(`  ${rows.length.toString().padStart(3)} fires  ${basename(dir)}/${basename(path)}`);
        }
      } catch (e) {
        console.error(`  [SKIP] ${path}: ${(e as Error).message}`);
      }
    }
  }

  // Sort by timestamp (lexicographic ISO works).
  allRows.sort((a, b) => a.ts.localeCompare(b.ts));

  const out = args.out ||
    join(
      import.meta.dir,
      `fires-last${args.days}d.jsonl`,
    );
  writeFileSync(out, allRows.map((r) => JSON.stringify(r)).join("\n") + "\n");

  // Summary
  console.error("");
  console.error(`scanned ${scanned} transcripts (${skipped} older than ${args.days}d skipped)`);
  console.error(`extracted ${allRows.length} fires across ${new Set(allRows.map((r) => r.transcript_id)).size} transcripts`);
  console.error(`wrote ${out}`);

  // Per-repo breakdown
  const byRepo: Record<string, number> = {};
  for (const r of allRows) byRepo[r.repo] = (byRepo[r.repo] || 0) + 1;
  console.error("");
  console.error("per-repo fire counts:");
  for (const [r, n] of Object.entries(byRepo).sort((a, b) => b[1] - a[1])) {
    console.error(`  ${n.toString().padStart(4)}  ${r}`);
  }

  // Proposition frequency
  const entityCounts: Record<string, number> = {};
  for (const r of allRows) {
    for (const p of r.propositions) {
      entityCounts[p.entity] = (entityCounts[p.entity] || 0) + 1;
    }
  }
  const topEntities = Object.entries(entityCounts).sort((a, b) => b[1] - a[1]).slice(0, 15);
  console.error("");
  console.error("top 15 entities fire'd on:");
  for (const [e, n] of topEntities) {
    console.error(`  ${n.toString().padStart(3)}  ${e}`);
  }
}

main();
