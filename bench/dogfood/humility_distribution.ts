#!/usr/bin/env bun
/**
 * humility_distribution.ts — compute distributions / histograms / per-tag
 * breakdowns over the JSONL produced by humility_baseline.ts.
 *
 * Usage:
 *   ./humility_baseline.ts > /tmp/baseline.jsonl
 *   ./humility_distribution.ts /tmp/baseline.jsonl
 */
import { readFileSync } from "node:fs";

interface SessionRow {
  source: string;
  session_id: string;
  origin_path: string;
  content_tag: string;
  date: string;
  assistant_turns: number;
  total_words: number;
  blind_spots_total: number;
  blind_spots_explicit: number;
  blind_spots_phrase: number;
  blind_spots_per_turn: number;
  heavy_turns_db: number | null;
  heavy_turns_proxy: number;
  blind_spots_per_heavy_turn_db: number | null;
  blind_spots_per_heavy_turn_proxy: number | null;
  stance_available: boolean;
  asserts: number;
  hedges: number;
  speculates: number;
  truth_bearing: number;
  humility_pct: number | null;
}

function pct(arr: number[], p: number): number | null {
  if (!arr.length) return null;
  const sorted = [...arr].sort((a, b) => a - b);
  const idx = (sorted.length - 1) * p;
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return sorted[lo];
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
}

function f(n: number | null, digits = 3): string {
  if (n === null || n === undefined) return "—";
  return Number(n.toFixed(digits)).toString();
}

function asciiHistogram(
  arr: number[],
  bins: Array<{ lo: number; hi: number; label: string }>,
  barWidth = 40,
): string {
  const counts = bins.map(() => 0);
  for (const v of arr) {
    for (let i = 0; i < bins.length; i++) {
      const { lo, hi } = bins[i];
      if (v >= lo && (i === bins.length - 1 ? v <= hi : v < hi)) {
        counts[i]++;
        break;
      }
    }
  }
  const max = Math.max(1, ...counts);
  const lines: string[] = [];
  const labelW = Math.max(...bins.map((b) => b.label.length));
  for (let i = 0; i < bins.length; i++) {
    const bar = "█".repeat(Math.round((counts[i] / max) * barWidth));
    lines.push(`${bins[i].label.padEnd(labelW)} | ${bar} ${counts[i]}`);
  }
  return lines.join("\n");
}

function summary(label: string, arr: number[]): string {
  if (!arr.length) return `${label}: (empty)`;
  return [
    `${label}: N=${arr.length}`,
    `  min=${f(Math.min(...arr))}  max=${f(Math.max(...arr))}`,
    `  p10=${f(pct(arr, 0.1))}  p25=${f(pct(arr, 0.25))}  p50=${f(pct(arr, 0.5))}  p75=${f(pct(arr, 0.75))}  p90=${f(pct(arr, 0.9))}`,
    `  mean=${f(arr.reduce((a, b) => a + b, 0) / arr.length)}`,
  ].join("\n");
}

function loadRows(path: string): SessionRow[] {
  return readFileSync(path, "utf-8")
    .split("\n")
    .filter(Boolean)
    .map((line) => JSON.parse(line));
}

function main(): void {
  const path = process.argv[2] ?? "/tmp/humility-baseline.jsonl";
  const all = loadRows(path);
  // Working set: sessions of nontrivial size.
  //   ≥10 assistant turns AND ≥1000 words OR (smaller but stance-available with truth>=5)
  const ws = all.filter(
    (r) =>
      (r.assistant_turns >= 10 && r.total_words >= 1000) ||
      (r.stance_available && r.truth_bearing >= 5),
  );

  console.log(`# humility baseline summary — ${ws.length} sessions in working set\n`);
  console.log("## Working-set selection");
  console.log(`Total parsed: ${all.length}`);
  console.log(`In working set (≥10 turns & ≥1000 words OR stance-available w/ truth≥5): ${ws.length}\n`);

  // Distributions
  const bs_pt = ws.map((r) => r.blind_spots_per_turn);
  const bs_total = ws.map((r) => r.blind_spots_total);
  const heavy_db = ws.map((r) => r.heavy_turns_db).filter((v): v is number => v !== null);
  const bs_per_heavy_db = ws
    .map((r) => r.blind_spots_per_heavy_turn_db)
    .filter((v): v is number => v !== null);
  const bs_per_heavy_proxy = ws
    .map((r) => r.blind_spots_per_heavy_turn_proxy)
    .filter((v): v is number => v !== null);
  const humility = ws
    .map((r) => r.humility_pct)
    .filter((v): v is number => v !== null);

  console.log("## Distributions\n");
  console.log("```");
  console.log(summary("blind_spots_per_turn (all sessions)", bs_pt));
  console.log();
  console.log(summary("blind_spots_total (all sessions)", bs_total));
  console.log();
  console.log(summary("heavy_turns_db (DB-derived, gate sessions only)", heavy_db));
  console.log();
  console.log(summary("blind_spots_per_heavy_turn_db (gate sessions only)", bs_per_heavy_db));
  console.log();
  console.log(summary("blind_spots_per_heavy_turn_proxy (all sessions)", bs_per_heavy_proxy));
  console.log();
  console.log(summary("humility_pct (gate sessions w/ truth-bearing ≥ 10)", humility));
  console.log("```\n");

  console.log("## Histogram: blind_spots_per_turn (all sessions in working set)\n");
  console.log("```");
  console.log(
    asciiHistogram(bs_pt, [
      { lo: 0, hi: 0.001, label: "    0.000" },
      { lo: 0.001, hi: 0.01, label: "  0.001-0.01" },
      { lo: 0.01, hi: 0.025, label: "  0.01-0.025" },
      { lo: 0.025, hi: 0.05, label: "  0.025-0.05" },
      { lo: 0.05, hi: 0.1, label: "  0.05-0.10" },
      { lo: 0.1, hi: 0.25, label: "  0.10-0.25" },
      { lo: 0.25, hi: 99, label: "  ≥ 0.25" },
    ]),
  );
  console.log("```\n");

  console.log("## Histogram: humility_pct (gate sessions w/ truth-bearing ≥ 10)\n");
  console.log("```");
  console.log(
    asciiHistogram(humility, [
      { lo: 0, hi: 5, label: "  0-5%" },
      { lo: 5, hi: 10, label: "  5-10%" },
      { lo: 10, hi: 20, label: "  10-20%" },
      { lo: 20, hi: 30, label: "  20-30%" },
      { lo: 30, hi: 40, label: "  30-40%" },
      { lo: 40, hi: 50, label: "  40-50%" },
      { lo: 50, hi: 100, label: "  ≥ 50%" },
    ]),
  );
  console.log("```\n");

  // Per-tag breakdown
  console.log("## Per-content-tag breakdown\n");
  const tags = Array.from(new Set(ws.map((r) => r.content_tag))).sort();
  console.log("```");
  console.log(
    "tag           | n  | bs_per_turn (p25/p50/p75) | heavy_db (p50) | bs_per_heavy_db (p50)",
  );
  console.log(
    "--------------|----|----------------------------|-----------------|----------------------",
  );
  for (const tag of tags) {
    const tagged = ws.filter((r) => r.content_tag === tag);
    const bs = tagged.map((r) => r.blind_spots_per_turn);
    const hvd = tagged
      .map((r) => r.heavy_turns_db)
      .filter((v): v is number => v !== null);
    const bshd = tagged
      .map((r) => r.blind_spots_per_heavy_turn_db)
      .filter((v): v is number => v !== null);
    console.log(
      [
        tag.padEnd(13),
        String(tagged.length).padEnd(3),
        `${f(pct(bs, 0.25))} / ${f(pct(bs, 0.5))} / ${f(pct(bs, 0.75))}`.padEnd(27),
        `${f(pct(hvd, 0.5))} (N=${hvd.length})`.padEnd(16),
        `${f(pct(bshd, 0.5))} (N=${bshd.length})`,
      ].join(" | "),
    );
  }
  console.log("```\n");

  // Per-session table
  console.log("## Per-session table (working set, sorted by date desc)\n");
  console.log("```");
  console.log(
    "date       | source            | tag         | turns | words | bs | bs/turn | hvy_db | bs/hvy_db | hvy_px | bs/hvy_px | hum%  | sess",
  );
  console.log(
    "-----------|-------------------|-------------|-------|-------|----|---------|--------|-----------|--------|-----------|-------|-----",
  );
  for (const r of ws) {
    console.log(
      [
        r.date,
        r.source.padEnd(17),
        r.content_tag.padEnd(11),
        String(r.assistant_turns).padStart(5),
        String(r.total_words).padStart(5),
        String(r.blind_spots_total).padStart(2),
        f(r.blind_spots_per_turn).padStart(7),
        (r.heavy_turns_db === null ? "—" : String(r.heavy_turns_db)).padStart(6),
        f(r.blind_spots_per_heavy_turn_db).padStart(9),
        String(r.heavy_turns_proxy).padStart(6),
        f(r.blind_spots_per_heavy_turn_proxy).padStart(9),
        (r.humility_pct === null ? "—" : `${r.humility_pct}%`).padStart(5),
        r.session_id.slice(0, 8),
      ].join(" | "),
    );
  }
  console.log("```\n");
}

main();
