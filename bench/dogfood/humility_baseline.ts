#!/usr/bin/env bun
/**
 * humility_baseline.ts — offline baseline for #52.
 *
 * Produces a per-session row with humility + blind-spot metrics across two
 * historical sources:
 *
 *   1. Claude Code .jsonl transcripts (assistant draft text per turn) — pairs
 *      with vouch's session_claims for the 3 transcripts that were actually
 *      processed by the gate. Other Claude Code sessions weren't running
 *      vouch, so their stance breakdown is unavailable.
 *
 *   2. Meta vault .txt transcripts at ~/Projects/meta/sunny/pages/inbox/
 *      transcripts/*.txt — older sessions exported before vouch was wired in.
 *      Stance unavailable (no extractor pass); blind-spot regex still works.
 *
 * Blind-spot regex is the SAME `countBlindSpots` shipped in `src/gate.ts`
 * (regex-only, no LLM). Stance / humility ratio comes from `session_claims`
 * because the stance label is LLM-extracted at gate time; we don't re-run
 * the extractor offline.
 *
 * Output: JSONL to stdout — one row per session. The report markdown is
 * generated separately from this output.
 *
 * Usage:
 *   ./humility_baseline.ts > humility-baseline.jsonl
 */
import { readdirSync, readFileSync, statSync } from "node:fs";
import { homedir } from "node:os";
import { basename, join } from "node:path";
import { countBlindSpots } from "../../src/gate.ts";
import * as store from "../../src/store.ts";

/** Per-turn ASSERT counts from session_claims, keyed by turn_idx. Only present
 *  for sessions that vouch processed at gate time. */
function getPerTurnAssertCounts(transcript_id: string): Map<number, number> {
  const rows = store
    .getDb()
    .prepare(
      `SELECT turn_idx, COUNT(*) AS n FROM session_claims
       WHERE transcript_id = ? AND stance = 'ASSERT'
       GROUP BY turn_idx`,
    )
    .all(transcript_id) as Array<{ turn_idx: number; n: number }>;
  const out = new Map<number, number>();
  for (const r of rows) out.set(r.turn_idx, r.n);
  return out;
}

interface AssistantTurn {
  text: string;
}

interface SessionRow {
  source: "claude-code-jsonl" | "meta-vault-txt";
  session_id: string;
  origin_path: string;
  content_tag: string;
  date: string;
  // Draft-derived
  assistant_turns: number;
  total_words: number;
  blind_spots_total: number;
  blind_spots_explicit: number;
  blind_spots_phrase: number;
  blind_spots_per_turn: number;
  // Heavy-turn definitions:
  //   _db: turns from session_claims where ASSERT count >= 3 (real, gate-derived)
  //   _proxy: assistant turns where named-entity sentence count >= 3 (rough offline proxy)
  heavy_turns_db: number | null;
  heavy_turns_proxy: number;
  blind_spots_per_heavy_turn_db: number | null;
  blind_spots_per_heavy_turn_proxy: number | null;
  // Stance-derived (only present when session_claims has rows)
  stance_available: boolean;
  asserts: number;
  hedges: number;
  speculates: number;
  truth_bearing: number;
  humility_pct: number | null;
}

const HEAVY_ASSERT_THRESHOLD = 3;

function readAssistantTurnsFromJsonl(path: string): AssistantTurn[] {
  const turns: AssistantTurn[] = [];
  const raw = readFileSync(path, "utf-8");
  for (const line of raw.split("\n")) {
    if (!line.trim()) continue;
    let entry: any;
    try {
      entry = JSON.parse(line);
    } catch {
      continue;
    }
    if (entry.type !== "assistant") continue;
    const content = entry.message?.content;
    let text = "";
    if (typeof content === "string") {
      text = content;
    } else if (Array.isArray(content)) {
      for (const block of content) {
        if (block?.type === "text" && typeof block.text === "string") {
          text += block.text + "\n";
        }
      }
    }
    text = text.trim();
    if (text) turns.push({ text });
  }
  return turns;
}

function readAssistantTurnsFromMetaTxt(path: string): AssistantTurn[] {
  // Format:
  //   ### assistant [<ts>]
  //   <body...>
  //   ### user [<ts>]   or ### assistant [<ts>]
  const raw = readFileSync(path, "utf-8");
  const turns: AssistantTurn[] = [];
  const lines = raw.split("\n");
  let inAssistant = false;
  let buf: string[] = [];
  for (const line of lines) {
    if (line.startsWith("### assistant ")) {
      if (inAssistant && buf.length) turns.push({ text: buf.join("\n").trim() });
      inAssistant = true;
      buf = [];
    } else if (line.startsWith("### user ") || line.startsWith("### tool ")) {
      if (inAssistant && buf.length) turns.push({ text: buf.join("\n").trim() });
      inAssistant = false;
      buf = [];
    } else if (inAssistant) {
      buf.push(line);
    }
  }
  if (inAssistant && buf.length) turns.push({ text: buf.join("\n").trim() });
  return turns.filter((t) => t.text.length > 0);
}

/** Crude heuristic: count sentences that look like factual claims with a named
 *  entity. Used as a proxy for ASSERT count to identify "heavy" turns when the
 *  LLM extractor wasn't run on the session. Capitalized multi-word noun-ish
 *  phrases are the proxy for a named entity. */
function namedEntitySentenceCount(text: string): number {
  const sentences = text.split(/(?<=[.!?])\s+/);
  const NE = /\b[A-Z][a-zA-Z0-9_.-]*(?:\s+[A-Z][a-zA-Z0-9_.-]*)?\b/;
  let n = 0;
  for (const s of sentences) {
    if (s.length < 20) continue;
    if (NE.test(s)) n++;
  }
  return n;
}

function inferContentTag(originPath: string, sampleText: string): string {
  const dir = originPath.toLowerCase();
  const txt = sampleText.toLowerCase().slice(0, 4000);
  // Path-based tags take precedence — they encode the project the session was for.
  if (/\/projects\/vouch\/|projects-vouch/.test(dir)) return "meta-vouch";
  if (/cardio|crforest/.test(dir)) return "research";
  if (/redacted-proj|nlpquant/.test(dir)) return "strategy";
  // Content-based for redacted-meta: distinguish strategy / research / meta-vouch
  if (/redacted-meta|inbox/.test(dir)) {
    if (/\bvouch\b/.test(txt) && /\bgate|claim|dossier|hedge|assert\b/.test(txt))
      return "meta-vouch";
    if (/\bc-index|cox|cif|vimp|tte_|fold-?\d|crforest|wanqi|hpc\b/.test(txt))
      return "research";
    return "strategy";
  }
  // Other locations
  if (/observer|claude-mem/.test(dir)) return "code";
  return "strategy";
}

function aggregateTurns(turns: AssistantTurn[]): {
  total_words: number;
  blind_spots_total: number;
  blind_spots_explicit: number;
  blind_spots_phrase: number;
  heavy_turns_proxy: number;
} {
  let total_words = 0;
  let explicit = 0;
  let phrase = 0;
  let heavy_turns_proxy = 0;
  for (const t of turns) {
    total_words += t.text.split(/\s+/).filter(Boolean).length;
    const bs = countBlindSpots(t.text);
    explicit += bs.explicit;
    phrase += bs.phrase;
    if (namedEntitySentenceCount(t.text) >= HEAVY_ASSERT_THRESHOLD) {
      heavy_turns_proxy += 1;
    }
  }
  return {
    total_words,
    blind_spots_total: explicit + phrase,
    blind_spots_explicit: explicit,
    blind_spots_phrase: phrase,
    heavy_turns_proxy,
  };
}

function listClaudeCodeJsonl(): Array<{ session_id: string; path: string; mtime: number }> {
  const root = join(homedir(), ".claude", "projects");
  const out: Array<{ session_id: string; path: string; mtime: number }> = [];
  let dirs: string[] = [];
  try {
    dirs = readdirSync(root);
  } catch {
    return out;
  }
  for (const d of dirs) {
    const full = join(root, d);
    let files: string[] = [];
    try {
      files = readdirSync(full);
    } catch {
      continue;
    }
    for (const f of files) {
      if (!f.endsWith(".jsonl")) continue;
      const p = join(full, f);
      try {
        const st = statSync(p);
        out.push({
          session_id: basename(f, ".jsonl"),
          path: p,
          mtime: st.mtimeMs,
        });
      } catch {}
    }
  }
  return out;
}

function listMetaTxt(): Array<{ session_id: string; path: string; mtime: number }> {
  const root = join(homedir(), "Projects", "meta", "sunny", "pages", "inbox", "transcripts");
  const out: Array<{ session_id: string; path: string; mtime: number }> = [];
  let files: string[] = [];
  try {
    files = readdirSync(root);
  } catch {
    return out;
  }
  for (const f of files) {
    if (!f.endsWith(".txt")) continue;
    const p = join(root, f);
    try {
      const st = statSync(p);
      out.push({ session_id: basename(f, ".txt"), path: p, mtime: st.mtimeMs });
    } catch {}
  }
  return out;
}

function rowFromTurns(
  source: SessionRow["source"],
  session_id: string,
  origin_path: string,
  turns: AssistantTurn[],
  stanceCounts: { asserts: number; hedges: number; speculates: number } | null,
  perTurnAsserts: Map<number, number> | null,
  date: string,
): SessionRow {
  const agg = aggregateTurns(turns);
  const truth = stanceCounts
    ? stanceCounts.asserts + stanceCounts.hedges + stanceCounts.speculates
    : 0;
  const hedged = stanceCounts ? stanceCounts.hedges + stanceCounts.speculates : 0;
  const humility =
    stanceCounts && truth >= 10 ? Number(((hedged / truth) * 100).toFixed(1)) : null;
  const sampleText = turns
    .slice(0, 3)
    .map((t) => t.text)
    .join("\n");
  // Real heavy-turn count from session_claims (per-turn ASSERTs >= 3).
  let heavyTurnsDb: number | null = null;
  if (perTurnAsserts && perTurnAsserts.size > 0) {
    heavyTurnsDb = 0;
    for (const n of perTurnAsserts.values()) {
      if (n >= HEAVY_ASSERT_THRESHOLD) heavyTurnsDb! += 1;
    }
  }
  return {
    source,
    session_id,
    origin_path,
    content_tag: inferContentTag(origin_path, sampleText),
    date,
    assistant_turns: turns.length,
    total_words: agg.total_words,
    blind_spots_total: agg.blind_spots_total,
    blind_spots_explicit: agg.blind_spots_explicit,
    blind_spots_phrase: agg.blind_spots_phrase,
    blind_spots_per_turn:
      turns.length > 0 ? Number((agg.blind_spots_total / turns.length).toFixed(3)) : 0,
    heavy_turns_db: heavyTurnsDb,
    heavy_turns_proxy: agg.heavy_turns_proxy,
    blind_spots_per_heavy_turn_db:
      heavyTurnsDb && heavyTurnsDb > 0
        ? Number((agg.blind_spots_total / heavyTurnsDb).toFixed(3))
        : null,
    blind_spots_per_heavy_turn_proxy:
      agg.heavy_turns_proxy > 0
        ? Number((agg.blind_spots_total / agg.heavy_turns_proxy).toFixed(3))
        : null,
    stance_available: !!stanceCounts && truth > 0,
    asserts: stanceCounts?.asserts ?? 0,
    hedges: stanceCounts?.hedges ?? 0,
    speculates: stanceCounts?.speculates ?? 0,
    truth_bearing: truth,
    humility_pct: humility,
  };
}

function main(): void {
  const rows: SessionRow[] = [];
  // 1. Claude Code .jsonl with vouch stance data — these are the gold sessions.
  const ccSessions = listClaudeCodeJsonl();
  for (const s of ccSessions) {
    // Skip synthetic bench-ctx sessions: those are vouch-internal test fixtures
    // (transcript_id literal like 'vouch-bench-ctx-xxxxxx') with no real prose.
    if (s.session_id.startsWith("vouch-bench-ctx-") || s.session_id.startsWith("vouch-diag-ctx-")) continue;
    let turns: AssistantTurn[] = [];
    try {
      turns = readAssistantTurnsFromJsonl(s.path);
    } catch {
      continue;
    }
    if (turns.length === 0) continue;
    const stance = store.getSessionFireCounts(s.session_id);
    const stanceAvail =
      stance.asserts + stance.hedges + stance.speculates > 0
        ? { asserts: stance.asserts, hedges: stance.hedges, speculates: stance.speculates }
        : null;
    const perTurn = stanceAvail ? getPerTurnAssertCounts(s.session_id) : null;
    const date = new Date(s.mtime).toISOString().slice(0, 10);
    rows.push(
      rowFromTurns(
        "claude-code-jsonl",
        s.session_id,
        s.path,
        turns,
        stanceAvail,
        perTurn,
        date,
      ),
    );
  }
  // 2. Meta vault transcripts — stance unavailable.
  const metaSessions = listMetaTxt();
  for (const s of metaSessions) {
    let turns: AssistantTurn[] = [];
    try {
      turns = readAssistantTurnsFromMetaTxt(s.path);
    } catch {
      continue;
    }
    if (turns.length === 0) continue;
    rows.push(
      rowFromTurns(
        "meta-vault-txt",
        s.session_id,
        s.path,
        turns,
        null,
        null,
        s.session_id.slice(0, 10),
      ),
    );
  }
  // Filter: skip sessions with fewer than 2 assistant turns OR < 200 words —
  // these are too small to compute meaningful per-turn metrics.
  const kept = rows.filter((r) => r.assistant_turns >= 2 && r.total_words >= 200);
  // Sort by date then turns descending
  kept.sort((a, b) => (a.date < b.date ? 1 : a.date > b.date ? -1 : b.assistant_turns - a.assistant_turns));
  for (const r of kept) console.log(JSON.stringify(r));
  console.error(`(${kept.length} sessions; ${rows.length - kept.length} skipped as too small)`);
}

main();
