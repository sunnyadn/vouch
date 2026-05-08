import { ArxivFetcher } from "./arxiv.ts";
import { GenericFetcher } from "./generic.ts";
import { OpenCliFetcher, opencliAvailable, OpenCliBridgeError } from "./opencli.ts";
import type { Fetcher, FetcherResult } from "./types.ts";

export type { Fetcher, FetcherResult } from "./types.ts";
export { OpenCliBridgeError } from "./opencli.ts";

/** Fetcher routing — first matching fetcher wins. Order matters: specific
 * adapters before the generic fallback. OpenCLI is reachable via --fetcher
 * opencli OR auto-fallback when generic content looks thin/JS-shell-only. */
const FETCHERS: Fetcher[] = [
  new ArxivFetcher(),
  new GenericFetcher(),
  // OpenCLI registered explicitly (for --fetcher hint resolution + introspection)
  // but does not match() proactively — see auto-fallback in fetchUrl.
  new OpenCliFetcher(),
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
  throw new Error(`no fetcher matched ${url} (this should be unreachable)`);
}

const THIN_CONTENT_THRESHOLD = 1500;

/** Fetch a URL with auto-fallback to OpenCLI when:
 *  (a) the primary fetcher's output is suspiciously short (likely a JS shell
 *      that didn't render server-side), AND
 *  (b) opencli is on PATH and (presumably) connected to its browser bridge.
 *  When OpenCLI itself errors with bridge-not-connected, we keep the generic
 *  result (don't fail the fetch entirely) but tag it so the caller can see
 *  the rendering may be incomplete. */
export async function fetchUrl(url: string, hint?: string): Promise<FetcherResult> {
  if (hint) {
    return getFetcher(url, hint).fetch(url);
  }
  const primary = await getFetcher(url).fetch(url);

  // Fast path: primary returned plenty of text. Done.
  if (primary.content.length >= THIN_CONTENT_THRESHOLD) return primary;

  // Auto-fallback: primary content looks thin. Try OpenCLI if available.
  if (!opencliAvailable()) {
    return {
      ...primary,
      metadata: {
        ...(primary.metadata || {}),
        thin_content_warning: `primary fetcher returned ${primary.content.length} chars; opencli not on PATH for fallback`,
      },
    };
  }

  try {
    const opencliResult = await new OpenCliFetcher().fetch(url);
    if (opencliResult.content.length > primary.content.length) {
      return {
        ...opencliResult,
        metadata: {
          ...(opencliResult.metadata || {}),
          fallback_from: primary.source_type,
          primary_content_chars: primary.content.length,
        },
      };
    }
    return primary;
  } catch (e) {
    if (e instanceof OpenCliBridgeError) {
      return {
        ...primary,
        metadata: {
          ...(primary.metadata || {}),
          thin_content_warning: `primary returned ${primary.content.length} chars; opencli fallback unavailable: ${e.message}`,
        },
      };
    }
    return {
      ...primary,
      metadata: {
        ...(primary.metadata || {}),
        thin_content_warning: `primary returned ${primary.content.length} chars; opencli fallback errored: ${e instanceof Error ? e.message.slice(0, 200) : String(e).slice(0, 200)}`,
      },
    };
  }
}
