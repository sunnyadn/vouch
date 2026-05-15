/** Exa web-search client for the comprehensiveness gate.
 *
 * Used by the gate to fetch top-N canonical web sources (with text
 * excerpts) when a fired proposition has no KB candidate AND no session
 * auto-ground hit — the case where the agent would otherwise silently
 * delete the claim because the fetch path is too long.
 *
 * Returns up to `numResults` (url, title, text) tuples. Caller (the gate)
 * inlines them into the fire message so the agent sees the canonical
 * source(s) inline and can reconcile in a single turn.
 *
 * Side effect: results are persisted to KB as `web/exa` dossiers so
 * future fires on the same entity can hit them via the standard
 * grounding paths (`vouch search` / `vouch fetch <url>`) without
 * re-paying for an Exa call.
 *
 * Fail-open: any error returns []. Gate code path treats empty results
 * the same as no Exa availability and falls back to the existing
 * suggestion-only fire message.
 *
 * Configured via $EXA_API_KEY. When unset, `searchWithText` returns []
 * silently (no warnings — gate continues as if Exa weren't installed).
 */

import * as store from "./store.ts";

export type ExaCandidate = {
  url: string;
  title: string;
  text: string;
};

const EXA_ENDPOINT = "https://api.exa.ai/search";
const DEFAULT_MAX_CHARS = 1200;
const DEFAULT_TIMEOUT_MS = 8000;

export async function searchWithText(
  query: string,
  opts: { numResults?: number; maxCharacters?: number; timeoutMs?: number } = {},
): Promise<ExaCandidate[]> {
  const apiKey = process.env.EXA_API_KEY;
  if (!apiKey) return [];

  const numResults = opts.numResults ?? 3;
  const maxCharacters = opts.maxCharacters ?? DEFAULT_MAX_CHARS;
  const timeoutMs = opts.timeoutMs ?? DEFAULT_TIMEOUT_MS;

  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), timeoutMs);

  try {
    const r = await fetch(EXA_ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json", "x-api-key": apiKey },
      body: JSON.stringify({
        query,
        numResults,
        type: "auto",
        contents: { text: { maxCharacters } },
      }),
      signal: ctl.signal,
    });
    if (!r.ok) return [];
    const data: any = await r.json();
    const results: any[] = Array.isArray(data?.results) ? data.results : [];
    const candidates = results.slice(0, numResults).map((x: any) => ({
      url: typeof x.url === "string" ? x.url : "",
      title: typeof x.title === "string" ? x.title : "",
      text: typeof x.text === "string" ? x.text.slice(0, maxCharacters) : "",
    })).filter(c => c.url);
    persistToKb(candidates);
    return candidates;
  } catch {
    return [];
  } finally {
    clearTimeout(timer);
  }
}

/** Persist Exa results as dossiers so subsequent gate fires can hit them
 *  via `vouch search` / KB lookup without re-paying for Exa. We write
 *  dossiers but not claims — claim extraction needs NLI verification and
 *  unverified claims would pollute the grounding path. The dossier
 *  surface lets `vouch fetch <url>` skip a network round-trip and lets
 *  future evidence-grounding cite the same source. INSERT OR REPLACE on
 *  the slugged URL makes re-fetching idempotent.
 *
 *  Fail-silent: any error skipped. KB persistence is a cache, not a
 *  correctness requirement. */
function persistToKb(candidates: ExaCandidate[]): void {
  for (const c of candidates) {
    if (!c.url || !c.text) continue;
    try {
      store.writeDossier({
        source_url: c.url,
        source_type: "web/exa",
        title: c.title || null,
        verbatim_content: c.text,
        scope: "third-party",
      });
    } catch {
      // ignore — KB write is best-effort
    }
  }
}
