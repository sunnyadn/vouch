#!/usr/bin/env bun
// discover_taxonomy.ts — agent-driven taxonomy update for the
// comprehensiveness category detector.
//
// Scans a corpus of texts for "<entity> is a <Y>" / "<entity> ... documented
// as a <Y>" / "the <Y> <entity>" patterns, extracts the noun head Y, and
// tallies candidates that DO NOT yet appear in comprehensiveness_taxonomy.json.
//
// Output: a JSON proposal block with (candidate_word, frequency, example
// contexts). An agent (human or LLM) reviews and decides whether to:
//   - add candidate to an existing category synonym list
//   - create a new top-level category
//   - decline (noise)
//
// Default corpus: bench/dogfood/fires-judge-study-P_alpha.jsonl (propositions
// + claim texts). Pass --input <path> to scan an arbitrary JSONL where each
// row has `proposition` and/or `claim_text` strings, or use --kb to scan
// the vouch KB directly (claim_text from claims table).

import { readFileSync, writeFileSync } from "fs";
import { dirname, join } from "path";
import { fileURLToPath } from "url";
import { Database } from "bun:sqlite";

const HERE = dirname(fileURLToPath(import.meta.url));

// Load current taxonomy
const taxonomyPath = join(HERE, "comprehensiveness_taxonomy.json");
const taxonomy: { categories: Record<string, string[]> } = JSON.parse(
  readFileSync(taxonomyPath, "utf8")
);
const KNOWN = new Set<string>();
for (const syns of Object.values(taxonomy.categories)) {
  for (const s of syns) KNOWN.add(s.toLowerCase());
}

// Stop nouns — words that fit grammatically into "X is a Y" but are
// almost never category-defining.
const GENERIC = new Set([
  "thing", "things", "case", "cases", "example", "examples", "way", "ways",
  "result", "results", "kind", "type", "types", "instance", "instances",
  "part", "parts", "version", "versions", "set", "sets", "form", "forms",
  "feature", "features", "function", "functions", "value", "values",
  "name", "names", "number", "numbers", "summary", "list", "lists",
  "side", "sides", "step", "steps", "field", "fields", "process",
]);

// Function words and common modifiers that show up inside captured phrases
// when the regex over-reaches. These are never category nouns.
const NOT_A_NOUN = new Set([
  "for", "the", "that", "with", "and", "or", "from", "into", "onto", "by",
  "to", "of", "in", "on", "at", "as", "an", "if", "but", "than", "then",
  "this", "these", "those", "such", "any", "all", "some", "more", "less",
  "used", "made", "built", "shown", "given", "based", "found",
  "evaluating", "evaluated", "presented", "presenting", "running",
  "first", "second", "third", "primary", "secondary", "main", "original",
  "candidate", "choice", "current", "next", "previous", "final",
  "true", "false", "valid", "invalid",
]);

const N_RE = /\b([A-Z][\w.-]*|[a-z][\w.-]+)\b[^.;!?]{0,80}?\b(?:is|are|was|were|=)\s+(?:a|an|the|one|that)\s+([a-z][\w-]*(?:\s+[a-z][\w-]+){0,3})/g;
const PAT_B = /\b([A-Z][\w.-]*|[a-z][\w.-]+)\b[^.;!?]{0,120}?\b(?:documented|described|classified|categorized|labeled|defined|positioned|titled|presented)\s+as\s+(?:a|an|the)\s+([a-z][\w-]*(?:\s+[a-z][\w-]+){0,3})/g;

type Candidate = {
  word: string;
  frequency: number;
  example_contexts: string[];
  proposed_category: string | null; // null = needs new top-level category
};

const counts = new Map<string, { count: number; examples: string[] }>();

function harvestFromText(text: string): void {
  // Pattern A: "X is/are a Y"
  for (const m of text.matchAll(N_RE)) {
    const phrase = m[2].toLowerCase();
    harvestWordsFromPhrase(phrase, text, m.index ?? 0);
  }
  // Pattern B: "X documented as a Y"
  for (const m of text.matchAll(PAT_B)) {
    const phrase = m[2].toLowerCase();
    harvestWordsFromPhrase(phrase, text, m.index ?? 0);
  }
}

function harvestWordsFromPhrase(phrase: string, fullText: string, hitIndex: number): void {
  // Take the HEAD noun: last word of the phrase (and an optional second-to-last
  // for multi-word noun heads like "language model"). Drop modifiers entirely.
  const wordList = phrase
    .split(/\s+/)
    .map(w => w.toLowerCase().replace(/^[^\w]+|[^\w]+$/g, ""))
    .filter(Boolean);
  if (!wordList.length) return;
  const head = wordList[wordList.length - 1];
  const candidates = [head];
  if (wordList.length >= 2) {
    const bigram = `${wordList[wordList.length - 2]} ${head}`;
    candidates.push(bigram);
  }

  for (const cand of candidates) {
    if (cand.length < 3) continue;
    if (cand.split(" ").every(w => NOT_A_NOUN.has(w))) continue;
    const lastWord = cand.split(" ").pop()!;
    if (NOT_A_NOUN.has(lastWord)) continue; // head must be a noun
    if (KNOWN.has(cand)) continue;
    if (GENERIC.has(cand)) continue;
    if (/^\d/.test(cand)) continue;
    const start = Math.max(0, hitIndex - 20);
    const end = Math.min(fullText.length, hitIndex + 120);
    const ctx = fullText.slice(start, end).replace(/\s+/g, " ").trim();
    if (!counts.has(cand)) counts.set(cand, { count: 0, examples: [] });
    const entry = counts.get(cand)!;
    entry.count++;
    if (entry.examples.length < 3 && !entry.examples.some(e => e.includes(ctx.slice(0, 60)))) {
      entry.examples.push(ctx);
    }
  }
}

function proposeCategory(word: string): string | null {
  // Lightweight clustering: substring / morphological similarity to any
  // existing synonym. Falls through to null = "needs new category".
  for (const [cat, syns] of Object.entries(taxonomy.categories)) {
    for (const s of syns) {
      const sLower = s.toLowerCase();
      if (word === sLower + "s" || word + "s" === sLower) return cat;
      if (word.endsWith(sLower) || sLower.endsWith(word)) return cat;
      if (Math.abs(word.length - sLower.length) <= 2) {
        const shared = [...word].filter(c => sLower.includes(c)).length;
        if (shared >= Math.min(word.length, sLower.length) - 1) return cat;
      }
    }
  }
  return null;
}

// ──────────────────────────────────────────────────────────────────────────
// Input sources
// ──────────────────────────────────────────────────────────────────────────

const args = process.argv.slice(2);
const useKb = args.includes("--kb");
const inputArg = args[args.indexOf("--input") + 1];

if (useKb) {
  // Scan vouch KB claims directly
  const dbPath = `${process.env.HOME}/.vouch/store.db`;
  const db = new Database(dbPath, { readonly: true });
  const rows = db
    .query("SELECT claim_text FROM claims WHERE verdict = 'supported' AND superseded_by IS NULL")
    .all() as { claim_text: string }[];
  console.error(`Scanning ${rows.length} supported KB claims…`);
  for (const r of rows) if (r.claim_text) harvestFromText(r.claim_text);
  db.close();
} else {
  const inputPath = inputArg ?? join(HERE, "fires-judge-study-P_alpha.jsonl");
  console.error(`Scanning ${inputPath}…`);
  const lines = readFileSync(inputPath, "utf8").trim().split("\n");
  for (const line of lines) {
    const r = JSON.parse(line) as { proposition?: string; claim_text?: string };
    if (r.proposition) harvestFromText(r.proposition);
    if (r.claim_text) harvestFromText(r.claim_text);
  }
}

// Filter: minimum frequency 2 (drop hapax legomena)
const MIN_FREQ = 2;
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
  taxonomy_version: (taxonomy as Record<string, unknown>)._version ?? "?",
  input_source: useKb ? "vouch KB" : (inputArg ?? "fires-judge-study-P_alpha.jsonl"),
  min_frequency: MIN_FREQ,
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
