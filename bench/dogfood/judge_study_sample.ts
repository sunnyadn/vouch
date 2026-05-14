#!/usr/bin/env bun
/**
 * judge_study_sample.ts — produce the 50-row hand-classification sample from
 * fires-judge-study-P_alpha.jsonl.
 *
 * Composition (50 pairs total):
 *   - All N broad-only fires: strict says no-fire (or score < 0.75), loose
 *     says no-fire, broad says fire. These are the candidates for the
 *     "task-definition loss" class (a refinement / partial-contradiction
 *     that strict's mutual-exclusion contract rejected).
 *   - Top up to 50 with a deterministic-shuffle sample of all-no-fire pairs
 *     (strict, loose, AND broad all say no-fire). These are the candidates
 *     for "correctly-rejected token coincidence" — with the residual
 *     hypothesis that even broad missed a real contradiction (the
 *     prompt-strictness-loss class), which is what hand reading checks.
 *
 * Output: bench/dogfood/p-alpha-judge-study-sample.tsv (gitignored; same
 *   privacy rules as fires-*.jsonl since it embeds session draft content).
 */
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const IN = join(import.meta.dir, "fires-judge-study-P_alpha.jsonl");
const OUT = join(import.meta.dir, "p-alpha-judge-study-sample.tsv");
const SAMPLE_SIZE = 50;
const FIRE_SCORE = 0.75;
const SEED = "vouch-judge-study-2026-05-14";

interface NliResult {
  fires: boolean;
  score: number;
  reason: string;
}

interface PairRow {
  ts: string;
  transcript_id: string;
  repo: string;
  entity: string;
  entity_class: string;
  proposition: string;
  claim_id: number;
  claim_text: string;
  dossier_slug: string | null;
  similarity: number;
  strict: NliResult;
  loose: NliResult;
  broad: NliResult;
}

function fires(v: NliResult): boolean {
  return v.fires && v.score >= FIRE_SCORE;
}

// Deterministic shuffle (xorshift seeded from SEED).
function hashSeed(s: string): number {
  let h = 2166136261 >>> 0;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}
function* xorshift(seed: number): Generator<number> {
  let x = seed || 1;
  while (true) {
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    yield (x >>> 0) / 0xffffffff;
  }
}
function shuffle<T>(arr: T[], seed: number): T[] {
  const rng = xorshift(seed);
  const out = [...arr];
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(rng.next().value * (i + 1));
    [out[i], out[j]] = [out[j]!, out[i]!];
  }
  return out;
}

function main() {
  const pairs: PairRow[] = readFileSync(IN, "utf8")
    .split("\n")
    .filter((l) => l.trim())
    .map((l) => JSON.parse(l) as PairRow);

  console.error(`loaded ${pairs.length} pairs`);

  const strictNo = pairs.filter((p) => !fires(p.strict));
  const broadOnly = strictNo.filter((p) => !fires(p.loose) && fires(p.broad));
  const allNo = strictNo.filter((p) => !fires(p.loose) && !fires(p.broad));

  console.error(`strict-no:      ${strictNo.length}`);
  console.error(`broad-only:     ${broadOnly.length}`);
  console.error(`all-no:         ${allNo.length}`);

  // Sample: keep all broad-only, top up with shuffled all-no.
  const broadOnlySorted = shuffle(broadOnly, hashSeed(SEED + ":broad-only"));
  const allNoShuf = shuffle(allNo, hashSeed(SEED + ":all-no"));

  const sample: Array<PairRow & { stratum: string }> = [];
  for (const p of broadOnlySorted) {
    if (sample.length >= SAMPLE_SIZE) break;
    sample.push({ ...p, stratum: "broad-only" });
  }
  for (const p of allNoShuf) {
    if (sample.length >= SAMPLE_SIZE) break;
    sample.push({ ...p, stratum: "all-no" });
  }

  // Stable order: stratum then ts.
  sample.sort((a, b) => {
    if (a.stratum !== b.stratum) return a.stratum.localeCompare(b.stratum);
    return a.ts.localeCompare(b.ts);
  });

  // Emit TSV. Tab-separated so the table renders flat in markdown / sheets.
  const header = [
    "idx",
    "stratum",
    "entity_class",
    "entity",
    "similarity",
    "strict_fires",
    "strict_score",
    "loose_fires",
    "loose_score",
    "broad_fires",
    "broad_score",
    "proposition",
    "claim_text",
    "broad_reason",
    "claim_id",
    "dossier_slug",
    "ts",
    "class_label", // hand-classify column: a=prompt-strictness, b=task-definition, c=correctly-rejected
    "notes",
  ];
  const lines = [header.join("\t")];
  sample.forEach((p, i) => {
    lines.push(
      [
        String(i + 1),
        p.stratum,
        p.entity_class,
        p.entity.replace(/\s+/g, " "),
        p.similarity.toFixed(3),
        String(p.strict.fires),
        p.strict.score.toFixed(2),
        String(p.loose.fires),
        p.loose.score.toFixed(2),
        String(p.broad.fires),
        p.broad.score.toFixed(2),
        p.proposition.replace(/\t/g, " ").replace(/\n/g, " "),
        p.claim_text.replace(/\t/g, " ").replace(/\n/g, " "),
        p.broad.reason.replace(/\t/g, " ").replace(/\n/g, " "),
        String(p.claim_id),
        p.dossier_slug || "",
        p.ts,
        "",
        "",
      ].join("\t"),
    );
  });

  writeFileSync(OUT, lines.join("\n") + "\n");
  console.error(`wrote ${sample.length} pairs to ${OUT}`);

  // Also print stratum breakdown to stderr.
  const stratumCounts = sample.reduce((acc, p) => {
    acc[p.stratum] = (acc[p.stratum] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);
  console.error(`sample composition: ${JSON.stringify(stratumCounts)}`);

  const classCounts = sample.reduce((acc, p) => {
    acc[p.entity_class] = (acc[p.entity_class] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);
  console.error(`entity_class:      ${JSON.stringify(classCounts)}`);
}

main();
