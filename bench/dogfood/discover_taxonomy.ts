#!/usr/bin/env bun
// discover_taxonomy.ts — LLM-based taxonomy discovery for the
// comprehensiveness category detector.
//
// Scans a corpus of (entity, sentence) pairs, asks the vouch verifier LLM
// to extract the category-asserting head noun (if any) for each, and
// proposes additions to the active taxonomy.
//
// Output: a JSON proposal block with (candidate, frequency, example
// sentences, proposed category). An agent reviews and decides:
//   - add candidate to an existing category synonym list
//   - create a new top-level category
//   - decline (LLM misread / not a category)
//
// Why LLM here (not regex): the seed taxonomy work showed regex / shallow
// NLP misses real category assertions when modifiers / participial clauses
// / quoted material interrupt the surface pattern. Discover runs
// infrequently (weekly?) and quality matters more than latency.
//
// Default corpus: bench/dogfood/fires-judge-study-P_alpha.jsonl. Pass
// --kb to scan the vouch KB directly (claim_text from supported claims).
// --limit N caps the number of LLM calls (default 200; full pass ≈ $0.05).

import { readFileSync, writeFileSync, existsSync } from "fs";
import { dirname, join } from "path";
import { fileURLToPath } from "url";
import { Database } from "bun:sqlite";
import { generateObject } from "ai";
import { z } from "zod";
import { getLanguageModel } from "../../src/providers.ts";
import { VERIFIER_MODEL } from "../../src/config.ts";

const HERE = dirname(fileURLToPath(import.meta.url));

// Load current taxonomy
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
const KNOWN = new Set<string>();
for (const syns of Object.values(taxonomy.categories)) {
  for (const s of syns) KNOWN.add(s.toLowerCase());
}

const args = process.argv.slice(2);
const useKb = args.includes("--kb");
const inputIdx = args.indexOf("--input");
const inputArg = inputIdx !== -1 ? args[inputIdx + 1] : undefined;
const limitIdx = args.indexOf("--limit");
const limit = limitIdx !== -1 ? parseInt(args[limitIdx + 1], 10) : 200;
const concurrencyIdx = args.indexOf("--concurrency");
const concurrency = concurrencyIdx !== -1 ? parseInt(args[concurrencyIdx + 1], 10) : 5;

// ──────────────────────────────────────────────────────────────────────────
// Collect (entity, sentence) pairs from input
// ──────────────────────────────────────────────────────────────────────────

type Pair = { entity: string; sentence: string };
const pairs: Pair[] = [];

if (useKb) {
  const dbPath = `${process.env.HOME}/.vouch/store.db`;
  const db = new Database(dbPath, { readonly: true });
  const rows = db
    .query("SELECT primary_entity, claim_text FROM claims WHERE verdict = 'supported' AND superseded_by IS NULL")
    .all() as { primary_entity: string | null; claim_text: string }[];
  for (const r of rows) {
    if (!r.primary_entity || !r.claim_text) continue;
    // One claim may have multiple sentences; the entity is per-claim.
    for (const sentence of splitSentences(r.claim_text)) {
      pairs.push({ entity: r.primary_entity, sentence });
    }
  }
  db.close();
  console.error(`Collected ${pairs.length} (entity, sentence) pairs from KB`);
} else {
  const inputPath = inputArg ?? join(HERE, "fires-judge-study-P_alpha.jsonl");
  const lines = readFileSync(inputPath, "utf8").trim().split("\n");
  for (const line of lines) {
    const r = JSON.parse(line) as { entity?: string; proposition?: string; claim_text?: string };
    if (!r.entity) continue;
    if (r.proposition) for (const s of splitSentences(r.proposition)) pairs.push({ entity: r.entity, sentence: s });
    if (r.claim_text) for (const s of splitSentences(r.claim_text)) pairs.push({ entity: r.entity, sentence: s });
  }
  console.error(`Collected ${pairs.length} (entity, sentence) pairs from ${inputPath}`);
}

function splitSentences(text: string): string[] {
  // Coarse split on .!?  Skip very short fragments.
  return text
    .split(/(?<=[.!?])\s+/)
    .map(s => s.trim())
    .filter(s => s.length > 15 && s.length < 400);
}

// Dedupe + sample
const seen = new Set<string>();
const unique: Pair[] = [];
for (const p of pairs) {
  const k = `${p.entity}::${p.sentence}`;
  if (seen.has(k)) continue;
  seen.add(k);
  unique.push(p);
}
const sampled = unique.slice(0, limit);
console.error(`Sampling ${sampled.length} / ${unique.length} unique pairs (limit=${limit})`);

// ──────────────────────────────────────────────────────────────────────────
// LLM extraction
// ──────────────────────────────────────────────────────────────────────────

const ExtractSchema = z.object({
  asserts_category: z.boolean().describe(
    "True iff the sentence asserts what KIND/TYPE/CATEGORY the entity IS (e.g., 'X is a lab/product/dataset/library/paper'). False if the sentence only describes what X does, what X has, who made X, what's true ABOUT X."
  ),
  category_word: z.string().nullable().describe(
    "If asserts_category=true, the single HEAD NOUN that names the category (e.g., 'lab', 'product', 'library'). Use singular, lowercase, no modifiers. null otherwise."
  ),
});

async function extractCategory(p: Pair): Promise<string | null> {
  try {
    const { object } = await generateObject({
      model: getLanguageModel(VERIFIER_MODEL),
      schema: ExtractSchema,
      prompt: `Sentence: ${p.sentence}\n\nEntity: ${p.entity}\n\nDoes this sentence assert what category/kind/type the entity ${p.entity} IS? If yes, what is the single head noun naming that category?`,
    });
    if (!object.asserts_category) return null;
    const w = object.category_word?.toLowerCase().trim().replace(/[^\w\s-]/g, "");
    if (!w) return null;
    return w;
  } catch (e: any) {
    console.error(`extraction error on ${p.entity}: ${e.message}`);
    return null;
  }
}

// Run with bounded concurrency
async function mapLimited<T, R>(items: T[], fn: (item: T) => Promise<R>, conc: number): Promise<R[]> {
  const results: R[] = new Array(items.length);
  let i = 0;
  await Promise.all(
    Array.from({ length: conc }, async () => {
      while (i < items.length) {
        const idx = i++;
        results[idx] = await fn(items[idx]);
      }
    })
  );
  return results;
}

console.error(`Calling LLM (${VERIFIER_MODEL}) with concurrency=${concurrency}…`);
const t0 = Date.now();
const extracted = await mapLimited(sampled, extractCategory, concurrency);
const wall = (Date.now() - t0) / 1000;
console.error(`LLM extraction done in ${wall.toFixed(1)}s`);

// ──────────────────────────────────────────────────────────────────────────
// Aggregate candidates
// ──────────────────────────────────────────────────────────────────────────

type Candidate = {
  word: string;
  frequency: number;
  example_contexts: { entity: string; sentence: string }[];
  proposed_category: string | null;
};

const counts = new Map<string, { count: number; examples: { entity: string; sentence: string }[] }>();
for (let i = 0; i < sampled.length; i++) {
  const word = extracted[i];
  if (!word) continue;
  if (KNOWN.has(word)) continue; // already in taxonomy
  if (!counts.has(word)) counts.set(word, { count: 0, examples: [] });
  const entry = counts.get(word)!;
  entry.count++;
  if (entry.examples.length < 3) entry.examples.push(sampled[i]);
}

function proposeCategory(word: string): string | null {
  for (const [cat, syns] of Object.entries(taxonomy.categories)) {
    for (const s of syns) {
      const sLower = s.toLowerCase();
      if (word === sLower + "s" || word + "s" === sLower) return cat;
      if (word.includes(sLower) || sLower.includes(word)) {
        if (Math.abs(word.length - sLower.length) <= 3) return cat;
      }
    }
  }
  return null;
}

const MIN_FREQ = 1; // LLM is high-precision; even singletons worth surfacing
const candidates: Candidate[] = [...counts.entries()]
  .filter(([_, v]) => v.count >= MIN_FREQ)
  .map(([word, v]) => ({
    word,
    frequency: v.count,
    example_contexts: v.examples,
    proposed_category: proposeCategory(word),
  }))
  .sort((a, b) => b.frequency - a.frequency);

const proposal = {
  detector: "category-mismatch",
  taxonomy_source: taxonomyPath,
  taxonomy_domain: taxonomy._domain ?? "?",
  taxonomy_version: taxonomy._version ?? "?",
  extractor: VERIFIER_MODEL,
  input_source: useKb ? "vouch KB" : (inputArg ?? "fires-judge-study-P_alpha.jsonl"),
  sample_size: sampled.length,
  total_unique: unique.length,
  wall_seconds: wall,
  candidates_count: candidates.length,
  candidates,
};

const outPath = join(HERE, "discover-taxonomy-proposal.json");
writeFileSync(outPath, JSON.stringify(proposal, null, 2));
console.log(`${candidates.length} candidates → ${outPath}`);
console.log("\nTop 15:");
for (const c of candidates.slice(0, 15)) {
  const cat = c.proposed_category ? `→ ${c.proposed_category}` : "→ NEW CATEGORY";
  console.log(`  ${c.word.padEnd(20)} freq=${c.frequency.toString().padStart(3)}  ${cat}`);
}
