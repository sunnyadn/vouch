/** Quote-in-dossier matching. The anti-fabrication primitive in the new
 * architecture: a claim's quote MUST appear in the dossier vouch fetched.
 *
 * Three-tier matching:
 *   1. Exact substring match. Most fetched HTML preserves quotes verbatim.
 *   2. Whitespace + smart-quote + space-before-punctuation normalized match.
 *      Catches HTML-stripping artifacts like "engine ." vs "engine."
 *   3. Fuzzy alphanumeric-only match. Catches reformatting / dropped commas /
 *      smart-quote drift.
 *
 * Returns offset range when found, null when not. Caller decides whether to
 * reject the claim (current default: reject if not found).
 */

function normalize(s: string): string {
  return (
    s
      .replace(/[‘’‚‛]/g, "'") // smart single quotes
      .replace(/[“”„‟]/g, '"') // smart double quotes
      .replace(/[–—]/g, "-") // en-dash, em-dash
      .replace(/ /g, " ") // nbsp
      .replace(/\s+/g, " ")
      // HTML-stripping artifact: ". " becomes " . " when block tags sit between
      // a word and its punctuation. Pull punctuation back so quotes taken from
      // rendered text match.
      .replace(/\s+([,.;:!?])/g, "$1")
      .toLowerCase() // case-insensitive: sentence boundaries shift case
      .trim()
  );
}

function alphanumOnly(s: string): string {
  return s.toLowerCase().replace(/[^a-z0-9一-鿿]/g, "");
}

export interface QuoteMatch {
  found: boolean;
  start: number | null;
  end: number | null;
  matchType: "exact" | "normalized" | "fuzzy" | "none";
}

export function findQuoteInContent(quote: string, content: string): QuoteMatch {
  if (!quote.trim()) return { found: false, start: null, end: null, matchType: "none" };

  // Tier 1: exact substring
  const exactIdx = content.indexOf(quote);
  if (exactIdx !== -1) {
    return {
      found: true,
      start: exactIdx,
      end: exactIdx + quote.length,
      matchType: "exact",
    };
  }

  // Tier 2: whitespace + smart-quote + punctuation-spacing normalized
  const nQuote = normalize(quote);
  const nContent = normalize(content);
  const normIdx = nContent.indexOf(nQuote);
  if (normIdx !== -1) {
    return {
      found: true,
      start: normIdx,
      end: normIdx + nQuote.length,
      matchType: "normalized",
    };
  }

  // Tier 3: alphanumeric-only fuzzy. Require 30+ char overlap to avoid false
  // positives on common phrases.
  const aQuote = alphanumOnly(quote);
  const aContent = alphanumOnly(content);
  if (aQuote.length >= 30 && aContent.includes(aQuote)) {
    return {
      found: true,
      start: null,
      end: null,
      matchType: "fuzzy",
    };
  }

  return { found: false, start: null, end: null, matchType: "none" };
}
