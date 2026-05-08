import { ArxivFetcher } from "./arxiv.ts";
import { GenericFetcher } from "./generic.ts";
import type { Fetcher, FetcherResult } from "./types.ts";

export type { Fetcher, FetcherResult } from "./types.ts";

/** Fetcher routing — first matching fetcher wins. Order matters: specific
 * adapters before the generic fallback. */
const FETCHERS: Fetcher[] = [
  new ArxivFetcher(),
  // Future:
  // new GitHubFetcher(),
  // new WikipediaFetcher(),
  // new OpenCliFetcher() — twitter / reddit / hn / zhihu / bilibili
  new GenericFetcher(), // must be last (matches everything)
];

export function getFetcher(url: string, hint?: string): Fetcher {
  if (hint) {
    const f = FETCHERS.find((f) => f.name === hint);
    if (!f) {
      throw new Error(
        `unknown --fetcher hint "${hint}". Available: ${FETCHERS.map((f) => f.name).join(", ")}`,
      );
    }
    return f;
  }
  for (const f of FETCHERS) {
    if (f.matches(url)) return f;
  }
  // GenericFetcher.matches() returns true; we should never reach here.
  throw new Error(`no fetcher matched ${url} (this should be unreachable)`);
}

export async function fetchUrl(url: string, hint?: string): Promise<FetcherResult> {
  const fetcher = getFetcher(url, hint);
  return fetcher.fetch(url);
}
