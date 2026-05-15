#!/usr/bin/env bun
// category_mismatch_probe.ts — entity-category-mismatch detector.
//
// Detects pairs where prop and KB claim assert different category-type
// predicates about the same entity ("X is a [dataset/package/product]"
// vs KB-attested "X is an [algorithm/lab/...]").
//
// Implements the ComprehensivenessDetector interface. Uses the
// `compromise` NLP library for noun-phrase extraction (head noun after
// copula / documented-as). Loads taxonomy from a JSON file pointed to
// by $VOUCH_COMPREHENSIVENESS_TAXONOMY, or falls back to the main
// comprehensiveness_taxonomy.json in this dir (which ships EMPTY —
// users supply vocabulary by either loading a sample pack from
// taxonomies/ or running discover_taxonomy.ts on their own KB).

import { readFileSync, writeFileSync, existsSync } from "fs";
import { dirname, join } from "path";
import { fileURLToPath } from "url";
import nlp from "compromise";

const HERE = dirname(fileURLToPath(import.meta.url));

// ──────────────────────────────────────────────────────────────────────────
// Detector interface
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
// Taxonomy loading: env-pointable, default empty
// ──────────────────────────────────────────────────────────────────────────

const taxonomyEnv = process.env.VOUCH_COMPREHENSIVENESS_TAXONOMY;
const taxonomyPath = taxonomyEnv
  ? (taxonomyEnv.startsWith("/") ? taxonomyEnv : join(process.cwd(), taxonomyEnv))
  : join(HERE, "comprehensiveness_taxonomy.json");
if (!existsSync(taxonomyPath)) {
  throw new Error(`Taxonomy not found at ${taxonomyPath}. Set VOUCH_COMPREHENSIVENESS_TAXONOMY or use the default empty taxonomy.`);
}
const taxonomy: { categories: Record<string, string[]>; _domain?: string; _version?: string } = JSON.parse(
  readFileSync(taxonomyPath, "utf8")
);
const CATEGORIES = taxonomy.categories;

// Reverse map: word → canonical category
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
// Noun extraction via compromise
// ──────────────────────────────────────────────────────────────────────────

function extractCategoryAssertions(text: string, entity: string): Set<string> {
  const out = new Set<string>();
  if (Object.keys(CATEGORIES).length === 0) return out; // empty taxonomy → no fires

  const doc = nlp(text);

  // Pattern A: "<entity> is/are/was/were/= [det] <NP>"
  const copulaMatch = doc.match(`(${entity}) (is|are|was|were|=)`);
  copulaMatch.forEach((m: any) => {
    const after = m.after();
    if (!after || after.length === 0) return;
    // Take the first noun phrase compromise identifies after the copula.
    // compromise's .nouns() returns noun phrases as chunks; head = last word.
    const nps = after.nouns().out("array") as string[];
    if (nps.length === 0) return;
    addCatsFromPhrase(nps[0].toLowerCase(), out);
  });

  // Pattern B: "<entity> ... documented/described/classified/titled as [det] <NP>"
  const verbMatch = doc.match(`(${entity}) .* (documented|described|classified|categorized|labeled|titled|presented) as`);
  verbMatch.forEach((m: any) => {
    const after = m.after();
    if (!after || after.length === 0) return;
    const nps = after.nouns().out("array") as string[];
    if (nps.length === 0) return;
    addCatsFromPhrase(nps[0].toLowerCase(), out);
  });

  // Pattern C: appositive — "the/a/an [adj]? <category> <entity>"
  for (const cat of Object.keys(CATEGORIES)) {
    for (const syn of CATEGORIES[cat]) {
      const synLower = syn.toLowerCase();
      // "the R package follic" / "an AI lab building..."
      const apMatch = doc.match(`(the|a|an|this|that|its|our) (#Adjective|#Noun)? ${synLower.replace(/ /g, " ")} ${entity}`);
      if (apMatch.found) out.add(cat);
    }
  }

  return out;
}

function addCatsFromPhrase(phrase: string, out: Set<string>): void {
  // Multi-word first (full phrase substring check)
  for (const mw of MULTI_WORD) {
    if (phrase.includes(mw.phrase)) {
      out.add(mw.cat);
      return; // multi-word hit dominates
    }
  }
  // Head noun = last token, stripped of punctuation
  const tokens = phrase.split(/\s+/).map(w => w.replace(/[^\w]/g, ""));
  const head = tokens[tokens.length - 1];
  if (head && WORD_TO_CAT[head]) {
    out.add(WORD_TO_CAT[head]);
  }
}

export const CategoryMismatchDetector: ComprehensivenessDetector = {
  name: "category-mismatch",
  detect(input: DetectionInput): DetectionFire | null {
    const propCats = extractCategoryAssertions(input.proposition, input.entity);
    const claimCats = extractCategoryAssertions(input.claim_text, input.entity);
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
    broad: boolean;
  };

  let total = 0;
  let withCategory = 0;
  const fires: Fire[] = [];

  for (const line of lines) {
    const r: Row = JSON.parse(line);
    total++;
    const fire = CategoryMismatchDetector.detect(r);
    const propHasCats = extractCategoryAssertions(r.proposition, r.entity).size > 0;
    const claimHasCats = extractCategoryAssertions(r.claim_text, r.entity).size > 0;
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
    taxonomy_path: taxonomyPath,
    taxonomy_domain: taxonomy._domain ?? "?",
    taxonomy_categories: Object.keys(CATEGORIES).length,
    extractor: "compromise@14",
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
