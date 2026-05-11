/** Academic-citation search — thin wrapper over opencli's scholarly adapters.
 *
 * vouch's value-prop is the FBC discipline (fetch -> claim -> NLI), not a
 * search-engine surface. So citation lookup leans entirely on opencli
 * (jackwener/opencli), which already ships maintained adapters for PubMed,
 * arXiv, OpenAlex, and Google Scholar. This module spawns
 * `opencli <provider> search <query> -f json`, validates the envelope, and
 * normalizes the per-provider field names into one CitationCandidate shape.
 *
 * Hard dep: opencli on PATH. If it's missing we throw a clear, actionable
 * error — we do NOT silently fall back to a generic web search, because that
 * would defeat the point (structured citation metadata with a canonical URL is
 * exactly what makes the downstream `vouch fetch <url>` reliable).
 */

const PROVIDERS = ["pubmed", "arxiv", "openalex", "google-scholar"] as const;
export type SearchProvider = (typeof PROVIDERS)[number];

export function isSearchProvider(p: string): p is SearchProvider {
  return (PROVIDERS as readonly string[]).includes(p);
}

export const SEARCH_PROVIDERS: readonly string[] = PROVIDERS;

const OPENCLI_BIN = process.env.VOUCH_OPENCLI_BIN || "opencli";

export interface CitationCandidate {
  provider: string;
  /** Provider's own rank (1 = its best match), when available. */
  rank?: number;
  title: string;
  url: string;
  year?: string | number;
  authors?: string;
  doi?: string;
  venue?: string;
}

export class OpenCliMissingError extends Error {
  constructor() {
    super(
      `opencli is not on PATH (looked for "${OPENCLI_BIN}"). ` +
        `Install it: bun install -g @jackwener/opencli  (or npm i -g @jackwener/opencli). ` +
        `Set VOUCH_OPENCLI_BIN to override the binary name/path.`,
    );
    this.name = "OpenCliMissingError";
  }
}

let _available: boolean | null = null;
export function opencliAvailable(): boolean {
  if (_available !== null) return _available;
  try {
    const proc = Bun.spawnSync([OPENCLI_BIN, "--version"], {
      stdout: "ignore",
      stderr: "ignore",
    });
    _available = proc.exitCode === 0;
  } catch {
    _available = false;
  }
  return _available;
}

export async function searchCitations(
  provider: string,
  query: string,
  limit = 5,
): Promise<CitationCandidate[]> {
  if (!isSearchProvider(provider)) {
    throw new Error(
      `unknown citation provider "${provider}". Supported: ${PROVIDERS.join(", ")}`,
    );
  }
  if (!opencliAvailable()) throw new OpenCliMissingError();

  let stdout: string;
  let stderr: string;
  let exitCode: number;
  try {
    const proc = Bun.spawn(
      [OPENCLI_BIN, provider, "search", query, "-f", "json", "--limit", String(limit)],
      { stdout: "pipe", stderr: "pipe" },
    );
    stdout = await new Response(proc.stdout).text();
    stderr = await new Response(proc.stderr).text();
    exitCode = await proc.exited;
  } catch (e: any) {
    if (e?.code === "ENOENT") throw new OpenCliMissingError();
    throw e;
  }
  return parseSearchPayload(provider, stdout, stderr, exitCode);
}

/** Pure parsing/normalization step — split out from the subprocess call so it
 * is unit-testable without spawning opencli. Throws on any non-array / error
 * envelope; returns normalized candidates with title+url present. */
export function parseSearchPayload(
  provider: string,
  stdout: string,
  stderr: string,
  exitCode: number,
): CitationCandidate[] {
  // opencli sometimes prefixes stdout with an update-notice banner; the JSON
  // payload is the first array/object literal. Extract defensively.
  const payload = extractJsonPayload(stdout);
  if (!payload) {
    if (exitCode !== 0) {
      throw new Error(
        `opencli ${provider} search failed (exit ${exitCode}): ${
          stderr.slice(0, 400) || stdout.slice(0, 400) || "no output"
        }`,
      );
    }
    throw new Error(
      `opencli ${provider} search: no JSON payload in output: ${stdout.slice(0, 300)}`,
    );
  }

  let parsed: any;
  try {
    parsed = JSON.parse(payload);
  } catch {
    throw new Error(
      `opencli ${provider} search: could not parse JSON: ${payload.slice(0, 300)}`,
    );
  }
  if (parsed && typeof parsed === "object" && parsed.ok === false) {
    const msg = parsed.error?.message || JSON.stringify(parsed.error || parsed);
    throw new Error(`opencli ${provider} search error: ${msg}`);
  }
  if (!Array.isArray(parsed)) {
    throw new Error(
      `opencli ${provider} search: expected a JSON array, got ${typeof parsed}`,
    );
  }
  return parsed
    .map((r: any) => normalizeCandidate(provider, r))
    .filter((c) => c.title && c.url);
}

function extractJsonPayload(out: string): string | null {
  const trimmed = out.trim();
  if (!trimmed) return null;
  // Fast path: whole output is the JSON.
  if (trimmed.startsWith("[") || trimmed.startsWith("{")) return trimmed;
  // Otherwise find the first array or object literal.
  const idxArr = trimmed.indexOf("[");
  const idxObj = trimmed.indexOf("{");
  const start =
    idxArr === -1 ? idxObj : idxObj === -1 ? idxArr : Math.min(idxArr, idxObj);
  if (start === -1) return null;
  return trimmed.slice(start);
}

function normalizeCandidate(provider: string, r: any): CitationCandidate {
  return {
    provider,
    rank: typeof r?.rank === "number" ? r.rank : undefined,
    title: String(r?.title ?? "").trim(),
    url: String(r?.url ?? "").trim(),
    year: r?.year ?? r?.published ?? undefined,
    authors: r?.authors ?? r?.firstAuthor ?? undefined,
    doi: r?.doi ?? undefined,
    venue: r?.venue ?? r?.journal ?? undefined,
  };
}

/** Token-Jaccard between two short strings (titles/queries). Lowercased,
 * alphanumeric tokens, stop-short tokens dropped. Used to sanity-check that an
 * auto-picked candidate's title actually relates to the search query. */
export function titleSimilarity(a: string, b: string): number {
  const ta = titleTokens(a);
  const tb = titleTokens(b);
  if (ta.size === 0 || tb.size === 0) return 0;
  let inter = 0;
  for (const t of ta) if (tb.has(t)) inter++;
  return inter / (ta.size + tb.size - inter);
}

function titleTokens(s: string): Set<string> {
  return new Set(
    s
      .toLowerCase()
      .replace(/[^a-z0-9\s]/g, " ")
      .split(/\s+/)
      .filter((t) => t.length > 2),
  );
}
