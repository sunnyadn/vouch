/** Verification-suggestion for the gate's fire message — #50 (B).
 *
 * The job: when the gate fires on an ungrounded named-entity claim, pre-build
 * a likely-correct verification command the agent can run, so the cost-cheapest
 * revise shape isn't "dodge" by default.
 *
 * Design (forced by the 2026-05-14 smoke + dialogue):
 *
 *   1. Always suggest `vouch search "<entity>"`. The user-side primitive
 *      `vouch search` (src/cli.ts:957) already does the heavy lifting —
 *      KB-first hybrid retrieval (embedding + full-text) AND a web-search
 *      fallback when the top KB hit's similarity falls below --kb-threshold.
 *      So `vouch search` IS the right surface: KB reuse when available, web
 *      lookup when not. We don't need a separate URL-guesser; suggesting an
 *      LLM-guessed top-1 web URL would push the agent back into web-fetch
 *      mode (the chaos source vouch is built to discipline) and risk
 *      seeding the next turn with a wrong-paper hallucination.
 *
 *   2. Pin `--provider` based on ID-pattern evidence in the RAW DRAFT
 *      (not the extractor's stripped proposition — by then the structural
 *      identifiers are often gone). Patterns are deterministic regex over
 *      arxiv / pubmed / DOI shapes, mapping to the providers vouch's
 *      searchers.ts:16 actually supports: pubmed | arxiv | openalex |
 *      google-scholar. OpenAlex resolves DOIs; arXiv has its own provider;
 *      PubMed for PMIDs; google-scholar is the catch-all for non-academic
 *      "scholarly" prose without an ID. No provider hint → default to web
 *      (ddg) via search's normal fallback.
 *
 * Zero new dependencies, zero latency, no LLM call, no external API.
 *
 * Why NOT also detect canonical URLs in the draft and suggest `vouch fetch`
 * directly: the URL may be for a DIFFERENT entity than the one that fired,
 * and we have no reliable way (without an LLM call) to attach URL-to-entity.
 * The search primitive picks the URL up via web-fallback anyway when the
 * entity-name query is well-formed.
 */
import type { SearchProvider } from "./searchers.ts";

/** Pinned to src/searchers.ts:16 — ["pubmed", "arxiv", "openalex",
 *  "google-scholar"]. Keep this in sync if PROVIDERS changes. */
type ProviderHint = SearchProvider;

/** Detected in the raw draft text (NOT the extractor's proposition). */
const PATTERNS: ReadonlyArray<{
  re: RegExp;
  hint: ProviderHint;
}> = [
  // arxiv: identifier patterns first (more specific), then domain mention.
  { re: /\b(?:arXiv:?\s*)\d{4}\.\d{4,5}(?:v\d+)?\b/i, hint: "arxiv" },
  { re: /\barxiv\.org\b/i, hint: "arxiv" },
  // pubmed: PMID:<digits> or pubmed domain.
  { re: /\bPMID:?\s*\d+\b/i, hint: "pubmed" },
  { re: /\bpubmed\.ncbi\.nlm\.nih\.gov\b/i, hint: "pubmed" },
  // DOI: openalex aggregates DOIs and is more open than google-scholar.
  { re: /\b10\.\d{4,9}\/[^\s)\]"<>]+/i, hint: "openalex" },
  { re: /\bdoi\.org\b/i, hint: "openalex" },
];

/** Pick the first pattern that fires on the draft, in priority order. */
export function detectProviderHint(draft: string): ProviderHint | null {
  if (!draft) return null;
  for (const { re, hint } of PATTERNS) {
    if (re.test(draft)) return hint;
  }
  return null;
}

/** Shell-quote-safe for use inside a double-quoted argument position. */
function shellQuote(s: string): string {
  return s.replace(/"/g, '\\"');
}

/** Build the suggested command. The single-line shape that follows the
 *  bullet in the gate's fire message:
 *
 *  ```
 *    • Kuhn et al: "..." (no entailment found)
 *      → suggested: vouch search "Kuhn et al" --provider arxiv
 *  ```
 */
export function suggestVerification(entity: string, draft: string): string {
  const safe = shellQuote(entity);
  const provider = detectProviderHint(draft);
  return provider
    ? `vouch search "${safe}" --provider ${provider}`
    : `vouch search "${safe}"`;
}

/** Render the indented suggestion line for formatBlockMessage. */
export function renderSuggestionLine(suggestion: string): string {
  return `      → suggested: ${suggestion}`;
}
