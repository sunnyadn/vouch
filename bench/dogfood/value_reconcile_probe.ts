#!/usr/bin/env bun
// value_reconcile_probe.ts — first-pass NLI-free numeric-mismatch detector.
//
// Extracts (number, unit, context) triples from proposition + KB claim;
// fires if a prop number has no near-match in claim AND shares a contextual
// anchor word with at least one claim number (proxy for "same metric").
//
// Implements the ComprehensivenessDetector interface (defined inline in
// each detector for now — will be extracted to a shared module if a third
// detector adopts it).

import { readFileSync, writeFileSync } from "fs";
import { dirname, join } from "path";
import { fileURLToPath } from "url";

const HERE = dirname(fileURLToPath(import.meta.url));

// ──────────────────────────────────────────────────────────────────────────
// Detector interface (same shape as category_mismatch_probe.ts)
// ──────────────────────────────────────────────────────────────────────────

type DetectionInput = {
  entity: string;
  entity_class: string;
  proposition: string;
  claim_text: string;
};

type DetectionFire = {
  detector: string;
  reason: string;
  meta: Record<string, unknown>;
};

interface ComprehensivenessDetector {
  name: string;
  detect(input: DetectionInput): DetectionFire | null;
}

// ──────────────────────────────────────────────────────────────────────────
// Number extraction
// ──────────────────────────────────────────────────────────────────────────

type Num = { value: number; unit: "pct" | "raw"; raw: string; ctx: string };

const NUM_RE = /(?<![A-Za-z\d.])(\d+(?:\.\d+)?)\s*(%|pp|percentage points|percent)?(?!\.\d)(?!\s*[-/]\s*\d)(?![A-Za-z])/gi;

function extractNumbers(text: string): Num[] {
  const out: Num[] = [];
  for (const m of text.matchAll(NUM_RE)) {
    const value = parseFloat(m[1]);
    if (Number.isNaN(value)) continue;
    if (value > 10000) continue; // years, ids
    const before = text.slice(Math.max(0, (m.index ?? 0) - 12), m.index).toLowerCase();
    if (/\bv\s*$/.test(before) || /version\s*$/.test(before)) continue;
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
  if (a.unit === "pct") return Math.abs(a.value - b.value) <= 0.5;
  const denom = Math.max(Math.abs(a.value), Math.abs(b.value), 1e-9);
  return Math.abs(a.value - b.value) / denom <= 0.02;
}

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

// ──────────────────────────────────────────────────────────────────────────
// Detector impl
// ──────────────────────────────────────────────────────────────────────────

export const ValueReconcileDetector: ComprehensivenessDetector = {
  name: "value-reconcile",
  detect(input: DetectionInput): DetectionFire | null {
    const propNums = extractNumbers(input.proposition);
    const claimNums = extractNumbers(input.claim_text);
    if (!propNums.length || !claimNums.length) return null;
    const mismatched: { prop_num: string; claim_num: string; shared_anchors: string[] }[] = [];
    for (const pn of propNums) {
      const sameUnit = claimNums.filter(cn => cn.unit === pn.unit);
      if (!sameUnit.length) continue;
      const candidates = sameUnit
        .map(cn => ({ cn, anchors: sharedAnchor(pn, cn) }))
        .filter(x => x.anchors.length > 0)
        .sort((a, b) => b.anchors.length - a.anchors.length);
      if (!candidates.length) continue;
      const best = candidates[0];
      if (nearMatch(pn, best.cn)) continue;
      mismatched.push({
        prop_num: pn.raw,
        claim_num: best.cn.raw,
        shared_anchors: best.anchors,
      });
    }
    if (!mismatched.length) return null;
    return {
      detector: this.name,
      reason: `${mismatched.length} numeric value-override(s): ${mismatched.map(m => `${m.prop_num}≠${m.claim_num}`).join(", ")}`,
      meta: { mismatched },
    };
  },
};

// ──────────────────────────────────────────────────────────────────────────
// Standalone CLI
// ──────────────────────────────────────────────────────────────────────────

if (import.meta.path === Bun.main) {
  const inputPath = join(HERE, "fires-judge-study-P_alpha.jsonl");
  const lines = readFileSync(inputPath, "utf8").trim().split("\n");

  type Row = DetectionInput & {
    similarity: number;
    strict: { fires: boolean };
    loose: { fires: boolean };
    broad: { fires: boolean };
  };

  type Fire = DetectionFire & {
    entity: string;
    entity_class: string;
    proposition: string;
    claim_text: string;
    strict: boolean;
    loose: boolean;
    broad: boolean;
  };

  let total = 0;
  let withNumsBoth = 0;
  const fires: Fire[] = [];

  for (const line of lines) {
    const r: Row = JSON.parse(line);
    total++;
    const propNums = extractNumbers(r.proposition);
    const claimNums = extractNumbers(r.claim_text);
    if (propNums.length && claimNums.length) withNumsBoth++;
    const fire = ValueReconcileDetector.detect(r);
    if (fire) {
      fires.push({
        ...fire,
        entity: r.entity,
        entity_class: r.entity_class,
        proposition: r.proposition,
        claim_text: r.claim_text,
        strict: r.strict.fires,
        loose: r.loose.fires,
        broad: r.broad.fires,
      });
    }
  }

  const summary = {
    detector: ValueReconcileDetector.name,
    total_pairs: total,
    pairs_with_numbers_both_sides: withNumsBoth,
    fires: fires.length,
    fires_also_strict: fires.filter(f => f.strict).length,
    fires_also_broad: fires.filter(f => f.broad).length,
    fires_unique_vs_strict: fires.filter(f => !f.strict).length,
    fires_unique_vs_broad: fires.filter(f => !f.broad).length,
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

  const outPath = join(HERE, "value-reconcile-probe-fires.jsonl");
  writeFileSync(outPath, fires.map(f => JSON.stringify(f)).join("\n") + "\n");
  console.log(`\n${fires.length} fires written to ${outPath}`);
}
