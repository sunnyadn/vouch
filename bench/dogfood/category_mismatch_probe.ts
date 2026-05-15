#!/usr/bin/env bun
// category_mismatch_probe.ts — entity-category-mismatch detector
// (NLI-free, regex over a user/agent-maintained taxonomy).
//
// Detects pairs where prop and KB claim assert different category-type
// predicates about the same entity ("X is a [dataset/package/product]"
// vs KB-attested "X is an [algorithm/lab/...]").
//
// Implements the ComprehensivenessDetector interface (defined inline,
// shared shape across detectors in this directory). Loads its taxonomy
// from comprehensiveness_taxonomy.json — that file is the agent-
// maintained source of truth, not hardcoded here.

import { readFileSync, writeFileSync } from "fs";
import { dirname, join } from "path";
import { fileURLToPath } from "url";

const HERE = dirname(fileURLToPath(import.meta.url));

// ──────────────────────────────────────────────────────────────────────────
// Detector interface (inline; will be extracted to a shared module when
// the second detector also adopts it).
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
// Config: taxonomy from JSON. User handle from env (no user-private
// literals in source — same pattern src/gate.ts §rule-1 uses).
// ──────────────────────────────────────────────────────────────────────────

const taxonomyPath = join(HERE, "comprehensiveness_taxonomy.json");
const taxonomy: { categories: Record<string, string[]> } = JSON.parse(
  readFileSync(taxonomyPath, "utf8")
);
const CATEGORIES = taxonomy.categories;

const USER_HANDLE_RAW = process.env.VOUCH_GATE_USER_HANDLE ?? "";
const USER_POSSESSIVE = USER_HANDLE_RAW ? `${USER_HANDLE_RAW.toLowerCase()}'s` : null;

// Reverse map word → canonical category
const WORD_TO_CAT: Record<string, string> = {};
for (const [cat, syns] of Object.entries(CATEGORIES)) {
  for (const s of syns) {
    if (!s.includes(" ")) WORD_TO_CAT[s.toLowerCase()] = cat;
  }
}
const MULTI_WORD: { phrase: string; cat: string }[] = [];
for (const [cat, syns] of Object.entries(CATEGORIES)) {
  for (const s of syns) {
    if (s.includes(" ")) MULTI_WORD.push({ phrase: s.toLowerCase(), cat });
  }
}

// ──────────────────────────────────────────────────────────────────────────
// Detector impl
// ──────────────────────────────────────────────────────────────────────────

function detectCategoryAssertions(text: string, entity: string): Set<string> {
  const out = new Set<string>();
  const lower = text.toLowerCase();
  const entityLower = entity.toLowerCase();

  const WORD = "[\\w./-]+";
  const SEP = "[^;!?]"; // allow . inside (file extensions, decimals)
  // Possessive determiner group — built from env-derived handle plus
  // generic determiners. No user-private literal in source.
  const possessiveOpts = ["a", "an", "the", "one", "that", "its"];
  if (USER_POSSESSIVE) possessiveOpts.push(escapeRegex(USER_POSSESSIVE));
  const DET = `(?:${possessiveOpts.join("|")})?`;

  // Pattern A: "<entity>... is/are/= [det] <noun-phrase>"
  const patA = new RegExp(
    `\\b${escapeRegex(entityLower)}\\b${SEP}{0,80}?\\b(?:is|are|=|stands\\s+for|refers\\s+to|represents)\\b\\s+${DET}\\s*(${WORD}(?:\\s+${WORD}){0,7})`,
    "gi"
  );
  for (const m of lower.matchAll(patA)) {
    addCatsFromPhrase(m[1], out);
  }

  // Pattern B: "<entity>... documented/described/classified as [det] <category>"
  const patB = new RegExp(
    `\\b${escapeRegex(entityLower)}\\b${SEP}{0,120}?\\b(?:documented|described|classified|categorized|labeled|defined|positioned|titled|presented)\\s+as\\s+${DET}\\s*(${WORD}(?:\\s+${WORD}){0,7})`,
    "gi"
  );
  for (const m of lower.matchAll(patB)) {
    addCatsFromPhrase(m[1], out);
  }

  // Pattern C: "<det> <category> <entity>" — appositive form
  // The user-possessive form is included via DET_REQ only when env handle
  // is set; otherwise this drops to generic determiners only.
  const possessiveC = USER_POSSESSIVE
    ? `(?:a|an|the|this|that|${escapeRegex(USER_POSSESSIVE)}|its|our)`
    : `(?:a|an|the|this|that|its|our)`;
  for (const cat of Object.keys(CATEGORIES)) {
    for (const syn of CATEGORIES[cat]) {
      const synLower = syn.toLowerCase();
      const patC = new RegExp(
        `\\b${possessiveC}\\s+(?:[\\w-]+\\s+){0,2}${escapeRegex(synLower)}\\s+(?:called\\s+|named\\s+)?${escapeRegex(entityLower)}\\b`,
        "i"
      );
      if (patC.test(lower)) out.add(cat);
    }
  }

  return out;
}

function escapeRegex(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function addCatsFromPhrase(phrase: string, out: Set<string>): void {
  for (const mw of MULTI_WORD) {
    if (phrase.includes(mw.phrase)) out.add(mw.cat);
  }
  for (const rawWord of phrase.split(/\s+/)) {
    const word = rawWord.toLowerCase().replace(/^[^\w]+|[^\w]+$/g, "");
    if (!word) continue;
    const cat = WORD_TO_CAT[word];
    if (cat) out.add(cat);
  }
}

export const CategoryMismatchDetector: ComprehensivenessDetector = {
  name: "category-mismatch",
  detect(input: DetectionInput): DetectionFire | null {
    const propCats = detectCategoryAssertions(input.proposition, input.entity);
    const claimCats = detectCategoryAssertions(input.claim_text, input.entity);
    if (!propCats.size || !claimCats.size) return null;
    const overlap = [...propCats].some(c => claimCats.has(c));
    if (overlap) return null;
    return {
      detector: this.name,
      reason: `prop asserts ${[...propCats].join("/")}; claim asserts ${[...claimCats].join("/")}`,
      meta: { prop_categories: [...propCats], claim_categories: [...claimCats] },
    };
  },
};

// ──────────────────────────────────────────────────────────────────────────
// Standalone CLI: run over fires-judge-study-P_alpha.jsonl
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
    broad: boolean;
  };

  let total = 0;
  let withCategory = 0;
  const fires: Fire[] = [];

  for (const line of lines) {
    const r: Row = JSON.parse(line);
    total++;
    const fire = CategoryMismatchDetector.detect(r);
    const propHasCats = detectCategoryAssertions(r.proposition, r.entity).size > 0;
    const claimHasCats = detectCategoryAssertions(r.claim_text, r.entity).size > 0;
    if (propHasCats && claimHasCats) withCategory++;
    if (fire) {
      fires.push({
        ...fire,
        entity: r.entity,
        entity_class: r.entity_class,
        proposition: r.proposition,
        claim_text: r.claim_text,
        strict: r.strict.fires,
        broad: r.broad.fires,
      });
    }
  }

  const summary = {
    detector: CategoryMismatchDetector.name,
    taxonomy_categories: Object.keys(CATEGORIES).length,
    taxonomy_version: (taxonomy as Record<string, unknown>)._version ?? "?",
    user_handle_from_env: USER_HANDLE_RAW || "<unset>",
    total_pairs: total,
    pairs_with_category_assertion_both_sides: withCategory,
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

  const outPath = join(HERE, "category-mismatch-probe-fires.jsonl");
  writeFileSync(outPath, fires.map(f => JSON.stringify(f)).join("\n") + "\n");
  console.log(`\n${fires.length} fires written to ${outPath}`);
}
