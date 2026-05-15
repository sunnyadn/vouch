#!/usr/bin/env bun
// category_mismatch_probe.ts — entity-category-mismatch detector.
//
// Detects pairs where prop and KB claim assert different category-type
// predicates about the same entity ("X is a [dataset/package/product]"
// vs KB-attested "X is an [algorithm/lab/...]").
//
// Implements the ComprehensivenessDetector interface. The default
// extractor is LLM-based (vouch verifier, Gemini 3.1 Pro) — highest
// accuracy on participle clauses, quote interruptions, distant heads
// that shallow NLP misses. Falls back to compromise@14 (180KB, ~5ms/
// case) when VOUCH_CATEGORY_EXTRACTOR=compromise is set — useful for
// gate-time hot path where ~1.5s/case LLM latency is too expensive.
//
// Loads taxonomy from $VOUCH_COMPREHENSIVENESS_TAXONOMY (path) or falls
// back to comprehensiveness_taxonomy.json in this dir (ships EMPTY).

import { readFileSync, writeFileSync, existsSync } from "fs";
import { dirname, join } from "path";
import { fileURLToPath } from "url";
import nlp from "compromise";
import { generateObject } from "ai";
import { z } from "zod";
import { getLanguageModel } from "../../src/providers.ts";
import { VERIFIER_MODEL } from "../../src/config.ts";

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
  detect(input: DetectionInput): DetectionFire | Promise<DetectionFire | null> | null;
}

// ──────────────────────────────────────────────────────────────────────────
// Taxonomy loading
// ──────────────────────────────────────────────────────────────────────────

const taxonomyEnv = process.env.VOUCH_COMPREHENSIVENESS_TAXONOMY;
const taxonomyPath = taxonomyEnv
  ? (taxonomyEnv.startsWith("/") ? taxonomyEnv : join(process.cwd(), taxonomyEnv))
  : join(HERE, "comprehensiveness_taxonomy.json");
if (!existsSync(taxonomyPath)) {
  throw new Error(`Taxonomy not found at ${taxonomyPath}`);
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
// Extractor (a): compromise-based — fast, ~5ms/case, 5/8 on bakeoff
// ──────────────────────────────────────────────────────────────────────────

function extractWithCompromise(text: string, entity: string): Set<string> {
  const out = new Set<string>();
  if (Object.keys(CATEGORIES).length === 0) return out;
  const doc = nlp(text);

  const copulaMatch = doc.match(`(${entity}) (is|are|was|were|=)`);
  copulaMatch.forEach((m: any) => {
    const after = m.after();
    if (!after || after.length === 0) return;
    const nps = after.nouns().out("array") as string[];
    if (nps.length === 0) return;
    addCatsFromPhrase(nps[0].toLowerCase(), out);
  });

  const verbMatch = doc.match(`(${entity}) .* (documented|described|classified|categorized|labeled|titled|presented) as`);
  verbMatch.forEach((m: any) => {
    const after = m.after();
    if (!after || after.length === 0) return;
    const nps = after.nouns().out("array") as string[];
    if (nps.length === 0) return;
    addCatsFromPhrase(nps[0].toLowerCase(), out);
  });

  return out;
}

function addCatsFromPhrase(phrase: string, out: Set<string>): void {
  for (const mw of MULTI_WORD) {
    if (phrase.includes(mw.phrase)) {
      out.add(mw.cat);
      return;
    }
  }
  const tokens = phrase.split(/\s+/).map(w => w.replace(/[^\w]/g, ""));
  const head = tokens[tokens.length - 1];
  if (head && WORD_TO_CAT[head]) out.add(WORD_TO_CAT[head]);
}

// ──────────────────────────────────────────────────────────────────────────
// Extractor (b): LLM-based — accurate, ~1.5s/case, 6-8/8 on bakeoff
// Cache by (entity, text) since the corpus has many repeated propositions.
// ──────────────────────────────────────────────────────────────────────────

const ExtractSchema = z.object({
  asserts_category: z.boolean().describe(
    "True iff the sentence asserts what KIND/TYPE/CATEGORY the entity IS (e.g., 'X is a lab/product/dataset/library/paper'). False if the sentence only describes what X does/has, or who made X."
  ),
  category_word: z.string().nullable().describe(
    "If asserts_category=true, the single HEAD NOUN naming that category (singular, lowercase, no modifiers). null otherwise."
  ),
});

const llmCache = new Map<string, string | null>();

async function extractWithLlm(text: string, entity: string): Promise<Set<string>> {
  const out = new Set<string>();
  if (Object.keys(CATEGORIES).length === 0) return out;

  const cacheKey = `${entity}::${text}`;
  let word: string | null;
  if (llmCache.has(cacheKey)) {
    word = llmCache.get(cacheKey) ?? null;
  } else {
    try {
      const { object } = await generateObject({
        model: getLanguageModel(VERIFIER_MODEL),
        schema: ExtractSchema,
        prompt: `Sentence: ${text}\n\nEntity: ${entity}\n\nDoes this sentence assert what category/kind/type the entity ${entity} IS? If yes, what is the single head noun naming that category?`,
      });
      word = object.asserts_category
        ? (object.category_word?.toLowerCase().trim().replace(/[^\w\s-]/g, "") || null)
        : null;
    } catch (e: any) {
      console.error(`LLM extract error on ${entity}: ${e.message}`);
      word = null;
    }
    llmCache.set(cacheKey, word);
  }

  if (!word) return out;
  // Map LLM-extracted word to a canonical category in the taxonomy.
  // Direct lookup first; if not in vocabulary, the user's taxonomy
  // doesn't cover this concept yet → don't fire on it. The discover
  // tool surfaces these as candidates for taxonomy expansion.
  if (WORD_TO_CAT[word]) out.add(WORD_TO_CAT[word]);
  else for (const mw of MULTI_WORD) if (word.includes(mw.phrase) || mw.phrase.includes(word)) out.add(mw.cat);
  return out;
}

// ──────────────────────────────────────────────────────────────────────────
// Detector (dispatch on env var)
// ──────────────────────────────────────────────────────────────────────────

const EXTRACTOR = (process.env.VOUCH_CATEGORY_EXTRACTOR ?? "llm").toLowerCase();
if (EXTRACTOR !== "llm" && EXTRACTOR !== "compromise") {
  throw new Error(`Invalid VOUCH_CATEGORY_EXTRACTOR=${EXTRACTOR}; expected 'llm' or 'compromise'`);
}

async function extract(text: string, entity: string): Promise<Set<string>> {
  return EXTRACTOR === "llm" ? extractWithLlm(text, entity) : Promise.resolve(extractWithCompromise(text, entity));
}

export const CategoryMismatchDetector: ComprehensivenessDetector = {
  name: "category-mismatch",
  async detect(input: DetectionInput): Promise<DetectionFire | null> {
    const propCats = await extract(input.proposition, input.entity);
    const claimCats = await extract(input.claim_text, input.entity);
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

  const t0 = Date.now();

  // Concurrency-bounded
  const concurrency = parseInt(process.env.CONCURRENCY ?? "5", 10);
  let i = 0;
  const rows = lines.map(l => JSON.parse(l) as Row);
  total = rows.length;

  await Promise.all(
    Array.from({ length: concurrency }, async () => {
      while (i < rows.length) {
        const idx = i++;
        const r = rows[idx];
        const fire = await CategoryMismatchDetector.detect(r);
        const propHasCats = (await extract(r.proposition, r.entity)).size > 0;
        const claimHasCats = (await extract(r.claim_text, r.entity)).size > 0;
        if (propHasCats && claimHasCats) withCategory++;
        if (fire) {
          fires.push({
            ...(fire as DetectionFire),
            entity: r.entity,
            entity_class: r.entity_class,
            proposition: r.proposition,
            claim_text: r.claim_text,
            strict: r.strict.fires,
            broad: r.broad.fires,
          });
        }
        if (idx % 25 === 0 && EXTRACTOR === "llm") {
          console.error(`  [${idx}/${rows.length}] fires=${fires.length} cache=${llmCache.size}`);
        }
      }
    })
  );

  const wall = (Date.now() - t0) / 1000;

  const summary = {
    detector: CategoryMismatchDetector.name,
    extractor: EXTRACTOR,
    extractor_model: EXTRACTOR === "llm" ? VERIFIER_MODEL : "compromise@14",
    taxonomy_path: taxonomyPath,
    taxonomy_domain: taxonomy._domain ?? "?",
    taxonomy_categories: Object.keys(CATEGORIES).length,
    wall_seconds: wall,
    llm_calls: EXTRACTOR === "llm" ? llmCache.size : 0,
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
