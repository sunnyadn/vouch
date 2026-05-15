#!/usr/bin/env bun
// nlp_bakeoff.ts — head-noun / category-assertion extraction bakeoff.
// Compare three approaches for category-mismatch detection's noun-extraction
// step:
//   (1) compromise — lightweight pure-JS NLP, 180KB MIT
//   (2) wink-nlp + wink-eng-lite-web-model — proper POS tagger, ~95% acc
//   (3) LLM via vouch's existing verifier (Gemini 3.1 Pro)
//
// Goal: identify, for a given (sentence, entity), the category-noun the
// sentence asserts about the entity ("X is a <Y>"). Reject cases where the
// sentence mentions X but asserts no category about it.

import nlp from "compromise";
// @ts-ignore - wink-nlp has no shipped .d.ts in this version
import winkNlp from "wink-nlp";
// @ts-ignore
import model from "wink-eng-lite-web-model";
import { generateObject } from "ai";
import { z } from "zod";
import { getLanguageModel } from "../../src/providers.ts";
import { VERIFIER_MODEL } from "../../src/config.ts";

const wink = winkNlp(model, ["pos"]);
const its = wink.its;

type TestCase = {
  sentence: string;
  entity: string;
  expected: string | null; // canonical category word, or null = should reject
  shape: string;
};

const CASES: TestCase[] = [
  { sentence: "FActScore is a pip-installable library presented at EMNLP 2023.", entity: "FActScore", expected: "library", shape: "is-a + compound modifier" },
  { sentence: "FActScore is the original implementation of an EMNLP 2023 paper on fine-grained atomic evaluation.", entity: "FActScore", expected: "paper", shape: "is-the + distant head" },
  { sentence: "Letta is an agentic memory product.", entity: "Letta", expected: "product", shape: "is-a + adjective chain" },
  { sentence: "Letta is an AI lab building persistent agents with continual learning capability.", entity: "Letta", expected: "lab", shape: "is-a + participle clause" },
  { sentence: "The R follic.Rd file is titled 'Follicular Cell Lymphoma' and is documented as a dataset.", entity: "follic", expected: "dataset", shape: "documented-as + quote interrupts" },
  { sentence: "Running the official scorer provided by the ALCE benchmark is required to obtain the citation-quality delta metric.", entity: "ALCE", expected: null, shape: "no category assertion (NEGATIVE)" },
  { sentence: "ALCE measures arm 1 only.", entity: "ALCE", expected: null, shape: "action verb, no category (NEGATIVE)" },
  { sentence: "mem0 is a system that claims to be an ultimate memory system without benchmarks.", entity: "mem0", expected: "system", shape: "is-a + relative clause" },
];

// ──────────────────────────────────────────────────────────────────────────
// (1) compromise extractor
// ──────────────────────────────────────────────────────────────────────────
function extractCompromise(sentence: string, entity: string): string | null {
  const doc = nlp(sentence);
  // Strategy: find clauses where entity is the subject. compromise has
  // .match() and .nouns().
  // Use a pattern: "<entity> #Copula [#Determiner?] [#Adjective*] (#Noun+)"
  // and capture the noun phrase.
  const m = doc.match(`${entity} (is|are|was|were) (the|a|an)? (#Noun+|#Adjective+ #Noun+)`);
  if (m.found) {
    // Get nouns in the match
    const nouns = m.nouns().out("array") as string[];
    if (nouns.length) {
      // Return the LAST noun (head) — strip plurals via nlp
      const head = nouns[nouns.length - 1].toLowerCase().split(/\s+/).pop()!;
      return nlp(head).nouns().toSingular().out("text") || head;
    }
  }
  // Fallback: "documented as <noun>"
  const m2 = doc.match(`${entity} #Verb+ (documented|described|classified|titled) as (a|an|the)? #Noun+`);
  if (m2.found) {
    const nouns = m2.nouns().out("array") as string[];
    if (nouns.length) {
      const head = nouns[nouns.length - 1].toLowerCase().split(/\s+/).pop()!;
      return nlp(head).nouns().toSingular().out("text") || head;
    }
  }
  return null;
}

// ──────────────────────────────────────────────────────────────────────────
// (2) wink-nlp extractor
// ──────────────────────────────────────────────────────────────────────────
function extractWinkNlp(sentence: string, entity: string): string | null {
  const doc = wink.readDoc(sentence);
  const tokens = doc.tokens();
  const data: { text: string; pos: string }[] = [];
  tokens.each((t: any) => {
    data.push({ text: t.out(), pos: t.out(its.pos) });
  });
  // Find entity index
  const entityIdx = data.findIndex(t => t.text.toLowerCase() === entity.toLowerCase());
  if (entityIdx === -1) return null;
  // After entity, look for copula (AUX/VERB "is/are/...") within 10 tokens,
  // then capture the next NOUN (skip DET/ADJ).
  let copulaIdx = -1;
  for (let i = entityIdx + 1; i < Math.min(data.length, entityIdx + 12); i++) {
    if (["is", "are", "was", "were"].includes(data[i].text.toLowerCase())) {
      copulaIdx = i;
      break;
    }
  }
  if (copulaIdx !== -1) {
    // Find consecutive NOUNs/ADJs after copula+determiner; head = last NOUN
    let head: string | null = null;
    let inPhrase = false;
    for (let i = copulaIdx + 1; i < Math.min(data.length, copulaIdx + 10); i++) {
      const pos = data[i].pos;
      if (["DET", "ADJ", "NOUN", "PROPN"].includes(pos)) {
        inPhrase = true;
        if (pos === "NOUN" || pos === "PROPN") head = data[i].text.toLowerCase();
      } else if (inPhrase) {
        break;
      }
    }
    if (head) return head;
  }
  // documented-as path
  for (let i = entityIdx + 1; i < Math.min(data.length, entityIdx + 25); i++) {
    if (["documented", "described", "classified", "titled"].includes(data[i].text.toLowerCase()) &&
        data[i + 1]?.text.toLowerCase() === "as") {
      // Find next noun
      for (let j = i + 2; j < Math.min(data.length, i + 8); j++) {
        if (["NOUN", "PROPN"].includes(data[j].pos)) {
          return data[j].text.toLowerCase();
        }
      }
    }
  }
  return null;
}

// ──────────────────────────────────────────────────────────────────────────
// (3) LLM extractor
// ──────────────────────────────────────────────────────────────────────────
async function extractLlm(sentence: string, entity: string): Promise<string | null> {
  const { object } = await generateObject({
    model: getLanguageModel(VERIFIER_MODEL),
    schema: z.object({
      asserts_category: z.boolean().describe("True iff the sentence asserts what KIND/TYPE/CATEGORY the entity is (e.g., 'X is a lab/product/dataset/library/paper'). False if the sentence only describes what X does, what X has, who made X, etc."),
      category_word: z.string().nullable().describe("If asserts_category=true, the single noun word that names the category (e.g., 'lab', 'product', 'library', 'paper'). Use the headword only, not modifiers. null otherwise."),
    }),
    prompt: `Sentence: ${sentence}\n\nEntity: ${entity}\n\nDoes this sentence assert what category/kind/type the entity ${entity} IS? If yes, what is the single head noun naming that category?`,
  });
  return object.asserts_category ? (object.category_word?.toLowerCase() ?? null) : null;
}

// ──────────────────────────────────────────────────────────────────────────
// Run bakeoff
// ──────────────────────────────────────────────────────────────────────────
async function main() {
  const results: any[] = [];
  for (const tc of CASES) {
    const c = extractCompromise(tc.sentence, tc.entity);
    const w = extractWinkNlp(tc.sentence, tc.entity);
    let l: string | null = null;
    try { l = await extractLlm(tc.sentence, tc.entity); } catch (e: any) { l = `ERROR: ${e.message}`; }
    const expectedCanon = tc.expected;
    results.push({
      shape: tc.shape,
      entity: tc.entity,
      expected: expectedCanon ?? "(reject)",
      compromise: c ?? "(none)",
      compromise_ok: matchesExpected(c, expectedCanon),
      wink: w ?? "(none)",
      wink_ok: matchesExpected(w, expectedCanon),
      llm: l ?? "(none)",
      llm_ok: matchesExpected(l, expectedCanon),
    });
  }
  // Tally
  const tally = {
    compromise: results.filter(r => r.compromise_ok).length,
    wink: results.filter(r => r.wink_ok).length,
    llm: results.filter(r => r.llm_ok).length,
  };
  console.table(results.map(r => ({
    shape: r.shape.slice(0, 32),
    entity: r.entity,
    expected: r.expected,
    compromise: `${r.compromise_ok ? "✓" : "✗"} ${r.compromise}`,
    wink: `${r.wink_ok ? "✓" : "✗"} ${r.wink}`,
    llm: `${r.llm_ok ? "✓" : "✗"} ${r.llm}`,
  })));
  console.log(`\nTally (correct / ${CASES.length}):`);
  console.log(`  compromise: ${tally.compromise}`);
  console.log(`  wink-nlp:   ${tally.wink}`);
  console.log(`  LLM:        ${tally.llm}`);
}

function matchesExpected(got: string | null, expected: string | null): boolean {
  if (expected === null) return got === null;
  if (got === null) return false;
  if (got.startsWith("ERROR")) return false;
  // Accept singular/plural variants and synonym overlap
  const g = got.toLowerCase().replace(/s$/, "");
  const e = expected.toLowerCase().replace(/s$/, "");
  return g === e || g.includes(e) || e.includes(g);
}

main().catch(e => { console.error(e); process.exit(1); });
