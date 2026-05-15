#!/usr/bin/env bun
// category_mismatch_probe.ts — NLI-free entity-category-mismatch detector.
// Identifies pairs where prop and claim assert different category-type
// predicates about the same entity ("X is a [dataset/package/product]" vs
// KB-attested "X is an [algorithm/lab/...]"). Identified as a residual
// (a)-class signal in the cross-validation report 2026-05-14.

import { readFileSync, writeFileSync } from "fs";

type Variant = { fires: boolean; score: number; reason: string };
type Row = {
  ts: string;
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

// Category taxonomy: each category is a set of synonyms. A pair is a
// category-mismatch fire iff prop asserts category X and claim asserts
// category Y where X != Y in the taxonomy.
const CATEGORIES: Record<string, string[]> = {
  dataset: ["dataset", "datasets", "benchmark", "benchmarks", "corpus", "corpora", "test set", "evaluation set"],
  algorithm: ["algorithm", "algorithms", "method", "methods", "metric", "metrics", "score", "procedure", "approach"],
  package: ["package", "packages", "library", "libraries", "module", "modules", "extension", "plugin"],
  product: ["product", "products", "tool", "tools", "application", "app", "platform", "service", "saas"],
  lab: ["lab", "labs", "company", "companies", "startup", "startups", "organization", "team"],
  paper: ["paper", "papers", "publication", "publications", "preprint", "study", "manuscript"],
  model: ["model", "models", "llm", "language model", "neural network"],
  evaluator: ["evaluator", "evaluators", "judge", "judges", "verifier", "scorer"],
  framework: ["framework", "frameworks", "system", "pipeline", "engine"],
};

// Reverse map: word → canonical category. Multi-word terms handled below.
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

// Detect category assertion in text relative to entity. Look for
// "<entity> is/are a/an <category>" or "<entity>: a <category>" etc.
// Returns the set of categories asserted about this entity.
function detectCategoryAssertions(text: string, entity: string): Set<string> {
  const out = new Set<string>();
  const lower = text.toLowerCase();
  const entityLower = entity.toLowerCase();

  // Sentence terminator excludes "." since file extensions / decimals
  // contain dots. Use ;!? + " . " (period with spaces) as boundaries.
  // Word capture allows hyphens, underscores, and slashes.
  const WORD = "[\\w./-]+";
  const SEP = "[^;!?]"; // allow . within window — file extensions, version dots

  // Pattern A: "<entity>... is/are/= [a|an|the] <noun-phrase up to 8 words>"
  const patA = new RegExp(
    `\\b${escapeRegex(entityLower)}\\b${SEP}{0,80}?\\b(?:is|are|=|stands\\s+for|refers\\s+to|represents)\\b\\s+(?:a|an|the|one|that|its|sunny's)?\\s*(${WORD}(?:\\s+${WORD}){0,7})`,
    "gi"
  );
  for (const m of lower.matchAll(patA)) {
    addCatsFromPhrase(m[1], out);
  }

  // Pattern B: "<entity>... documented/described/classified/labeled/titled as a <category>"
  const patB = new RegExp(
    `\\b${escapeRegex(entityLower)}\\b${SEP}{0,120}?\\b(?:documented|described|classified|categorized|labeled|defined|positioned|titled|presented)\\s+as\\s+(?:a|an|the)?\\s*(${WORD}(?:\\s+${WORD}){0,7})`,
    "gi"
  );
  for (const m of lower.matchAll(patB)) {
    addCatsFromPhrase(m[1], out);
  }

  // Pattern C: "<category>... <entity>" — appositive form, e.g. "the R package follic"
  for (const cat of Object.keys(CATEGORIES)) {
    for (const syn of CATEGORIES[cat]) {
      const synLower = syn.toLowerCase();
      const patC = new RegExp(
        `\\b(?:a|an|the|this|that|sunny's|its|our)\\s+(?:[\\w-]+\\s+){0,2}${escapeRegex(synLower)}\\s+(?:called\\s+|named\\s+)?${escapeRegex(entityLower)}\\b`,
        "i"
      );
      if (patC.test(lower)) out.add(cat);
    }
  }

  // Pattern D: "<entity> file/repo/page" — file/repo/etc. is a category implicitly
  // (handles ".Rd file is titled X as a dataset" where the "file" word itself is
  // a category-of-thing signal, but the dataset/algorithm assignment happens via
  // pattern B above on the same entity).

  return out;
}

function escapeRegex(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function addCatsFromPhrase(phrase: string, out: Set<string>): void {
  // Try multi-word phrases first
  for (const mw of MULTI_WORD) {
    if (phrase.includes(mw.phrase)) out.add(mw.cat);
  }
  // Then single words — strip leading/trailing punctuation per word
  for (const rawWord of phrase.split(/\s+/)) {
    const word = rawWord.toLowerCase().replace(/^[^\w]+|[^\w]+$/g, "");
    if (!word) continue;
    const cat = WORD_TO_CAT[word];
    if (cat) out.add(cat);
  }
}

const inputPath = `${process.env.HOME}/Projects/vouch/bench/dogfood/fires-judge-study-P_alpha.jsonl`;
const lines = readFileSync(inputPath, "utf8").trim().split("\n");

type Fire = {
  entity: string;
  entity_class: string;
  proposition: string;
  claim_text: string;
  prop_categories: string[];
  claim_categories: string[];
  strict: boolean;
  broad: boolean;
};

let total = 0;
let withCategory = 0;
const fires: Fire[] = [];

for (const line of lines) {
  const r: Row = JSON.parse(line);
  total++;
  const propCats = detectCategoryAssertions(r.proposition, r.entity);
  const claimCats = detectCategoryAssertions(r.claim_text, r.entity);
  if (!propCats.size || !claimCats.size) continue;
  withCategory++;
  // Fire iff disjoint sets (no category overlap)
  const overlap = [...propCats].some(c => claimCats.has(c));
  if (overlap) continue;
  fires.push({
    entity: r.entity,
    entity_class: r.entity_class,
    proposition: r.proposition,
    claim_text: r.claim_text,
    prop_categories: [...propCats],
    claim_categories: [...claimCats],
    strict: r.strict.fires,
    broad: r.broad.fires,
  });
}

const summary = {
  total_pairs: total,
  pairs_with_category_assertion_both_sides: withCategory,
  category_mismatch_fires: fires.length,
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

const outPath = `${process.env.HOME}/Projects/vouch/bench/dogfood/category-mismatch-probe-fires.jsonl`;
writeFileSync(outPath, fires.map(f => JSON.stringify(f)).join("\n") + "\n");
console.log(`\n${fires.length} fires written to ${outPath}`);
