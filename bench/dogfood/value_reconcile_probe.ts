#!/usr/bin/env bun
// value_reconcile_probe.ts — first-pass NLI-free numeric-mismatch detector.
// Extracts (number, unit, context-word) triples from proposition + KB claim;
// flags pair if proposition has a number that lacks a near-match in claim
// AND there is a shared context-anchor word.

import { readFileSync, writeFileSync } from "fs";

type Variant = { fires: boolean; score: number; reason: string };
type Row = {
  ts: string;
  transcript_id: string;
  repo: string;
  entity: string;
  entity_class: string;
  proposition: string;
  claim_id: number;
  claim_text: string;
  similarity: number;
  strict: Variant;
  loose: Variant;
  broad: Variant;
};

type Num = { value: number; unit: "pct" | "raw"; raw: string; ctx: string };

// Match a number, but exclude version-shaped tokens (e.g. "0.4.0", "v3.5.2")
// and date fragments (e.g. "2026-05-10"). The negative lookahead `(?!\.\d)` rules
// out "0.4" when followed by ".0", and `(?!\s*[-/]\s*\d)` rules out dates.
const NUM_RE = /(?<![A-Za-z\d.])(\d+(?:\.\d+)?)\s*(%|pp|percentage points|percent)?(?!\.\d)(?!\s*[-/]\s*\d)(?![A-Za-z])/gi;

function extractNumbers(text: string): Num[] {
  const out: Num[] = [];
  for (const m of text.matchAll(NUM_RE)) {
    const value = parseFloat(m[1]);
    if (Number.isNaN(value)) continue;
    if (value > 10000) continue; // years, ids
    // Filter version-context: number directly preceded by "v" / "version" / package name + space + digit
    const before = text.slice(Math.max(0, (m.index ?? 0) - 12), m.index).toLowerCase();
    if (/\bv\s*$/.test(before) || /version\s*$/.test(before)) continue;
    // Filter year-shaped raw 4-digit numbers
    if (!m[2] && /^\d{4}$/.test(m[1]) && value >= 1900 && value <= 2100) continue;
    const unit = m[2] ? "pct" : "raw";
    const start = Math.max(0, (m.index ?? 0) - 40);
    const end = Math.min(text.length, (m.index ?? 0) + m[0].length + 10);
    const ctx = text.slice(start, end).toLowerCase();
    out.push({ value, unit, raw: m[0], ctx });
  }
  return out;
}

function nearMatch(a: Num, b: Num): boolean {
  if (a.unit !== b.unit) return false;
  // Tight tolerance: 0.5pp absolute for pct, 2% relative for raw.
  if (a.unit === "pct") return Math.abs(a.value - b.value) <= 0.5;
  const denom = Math.max(Math.abs(a.value), Math.abs(b.value), 1e-9);
  return Math.abs(a.value - b.value) / denom <= 0.02;
}

// shared content words (stopwords stripped) between two contexts
const STOP = new Set([
  "the","a","an","is","are","was","were","be","of","to","in","on","at","for","with","and","or",
  "this","that","these","those","by","as","it","its","from","into","onto","than","then","so",
  "x","y","but","also","just","only","very","over","under","more","less","most","each",
  "value","values","number","numbers","percent","pp","percentage","points",
]);

function contentWords(ctx: string): Set<string> {
  return new Set(
    ctx
      .split(/[^a-z0-9_-]+/i)
      .map(w => w.toLowerCase())
      .filter(w => w.length >= 3 && !STOP.has(w) && !/^\d+(\.\d+)?$/.test(w))
  );
}

function sharedAnchor(a: Num, b: Num): string[] {
  const aw = contentWords(a.ctx);
  const bw = contentWords(b.ctx);
  return [...aw].filter(w => bw.has(w));
}

type Fire = {
  entity: string;
  entity_class: string;
  proposition: string;
  claim_text: string;
  mismatched: { prop_num: string; claim_num: string; shared_anchors: string[] }[];
  strict: boolean;
  loose: boolean;
  broad: boolean;
};

const inputPath = `${process.env.HOME}/Projects/vouch/bench/dogfood/fires-judge-study-P_alpha.jsonl`;
const lines = readFileSync(inputPath, "utf8").trim().split("\n");

let total = 0;
let withNumsBoth = 0;
const fires: Fire[] = [];

for (const line of lines) {
  const r: Row = JSON.parse(line);
  total++;
  const propNums = extractNumbers(r.proposition);
  const claimNums = extractNumbers(r.claim_text);
  if (!propNums.length || !claimNums.length) continue;
  withNumsBoth++;

  // For each prop number, find the SINGLE best-anchor claim number (highest
  // shared-anchor count, same unit). Fire iff the value at that anchor mismatches.
  // This is the corrected "same metric, different value" check.
  const mismatched: Fire["mismatched"] = [];
  for (const pn of propNums) {
    const sameUnitClaimNums = claimNums.filter(cn => cn.unit === pn.unit);
    if (!sameUnitClaimNums.length) continue;
    const candidates = sameUnitClaimNums
      .map(cn => ({ cn, anchors: sharedAnchor(pn, cn) }))
      .filter(x => x.anchors.length > 0)
      .sort((a, b) => b.anchors.length - a.anchors.length);
    if (!candidates.length) continue;
    const best = candidates[0];
    if (nearMatch(pn, best.cn)) continue; // same metric, same value → OK
    mismatched.push({
      prop_num: pn.raw,
      claim_num: best.cn.raw,
      shared_anchors: best.anchors,
    });
  }

  if (mismatched.length > 0) {
    fires.push({
      entity: r.entity,
      entity_class: r.entity_class,
      proposition: r.proposition,
      claim_text: r.claim_text,
      mismatched,
      strict: r.strict.fires,
      loose: r.loose.fires,
      broad: r.broad.fires,
    });
  }
}

// Cross-tab vs strict / broad
const fireSet = new Set(fires.map(f => `${f.entity}::${f.proposition}::${f.claim_text}`));
let strictFiresInValueReconcile = 0;
let broadFiresInValueReconcile = 0;
for (const f of fires) {
  if (f.strict) strictFiresInValueReconcile++;
  if (f.broad) broadFiresInValueReconcile++;
}

const summary = {
  total_pairs: total,
  pairs_with_numbers_both_sides: withNumsBoth,
  value_reconcile_fires: fires.length,
  fires_also_strict: strictFiresInValueReconcile,
  fires_also_broad: broadFiresInValueReconcile,
  fires_unique_to_value_reconcile_vs_strict: fires.length - strictFiresInValueReconcile,
  fires_unique_to_value_reconcile_vs_broad: fires.filter(f => !f.broad).length,
  by_entity_class: Object.fromEntries(
    Object.entries(
      fires.reduce((acc, f) => {
        acc[f.entity_class] = (acc[f.entity_class] || 0) + 1;
        return acc;
      }, {} as Record<string, number>)
    ).sort((a, b) => b[1] - a[1])
  ),
};

console.log(JSON.stringify(summary, null, 2));

const outPath = `${process.env.HOME}/Projects/vouch/bench/dogfood/value-reconcile-probe-fires.jsonl`;
writeFileSync(outPath, fires.map(f => JSON.stringify(f)).join("\n") + "\n");
console.log(`\n${fires.length} fires written to ${outPath}`);
