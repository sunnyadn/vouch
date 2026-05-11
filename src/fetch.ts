/** vouch fetch — high-level: route URL to fetcher, persist as dossier with
 *  embedding. This is the primary trust-establishing step in the new
 *  architecture: claims must reference dossiers vouch fetched itself.
 */
import { fetchUrl } from "./fetchers/index.ts";
import type { FetcherResult } from "./fetchers/types.ts";
import { embedOne } from "./embedder.ts";
import * as store from "./store.ts";
import { existsSync } from "node:fs";
import { basename, resolve } from "node:path";

interface FetchResult {
  dossier_slug: string;
  source_url: string;
  source_type: string;
  title: string | null;
  publication_date: string | null;
  author_attribution: string | null;
  /** The fetched content, sliced to `contentLimit` (or the whole thing when
   *  `full` is set / the document is small). This makes `vouch fetch` a
   *  drop-in for a built-in web-fetch tool: one call yields the readable text
   *  AND persists the dossier the agent will cite. */
  content: string;
  /** Total length of the persisted dossier content (>= content.length). */
  content_chars: number;
  cached: boolean;
  fetched_at: string;
  /** Set when the content is suspiciously thin (likely JS-rendered) and the
   *  OpenCLI fallback couldn't help (extension not connected, errored, etc.).
   *  Caller should consider re-fetching with a connected browser bridge. */
  warning?: string;
  metadata?: Record<string, unknown>;
}

/** Default head-chunk size returned in `content` when neither --full nor an
 *  explicit --content-limit is given. Matches the embedding-input slice. */
export const DEFAULT_FETCH_CONTENT_CHARS = 8000;

function sliceContent(full: string, opts: { full?: boolean; contentLimit?: number }): string {
  if (opts.full) return full;
  const limit = opts.contentLimit && opts.contentLimit > 0 ? opts.contentLimit : DEFAULT_FETCH_CONTENT_CHARS;
  return full.slice(0, limit);
}

function isHttpUrl(url: string): boolean {
  try {
    const u = new URL(url);
    return u.protocol === "http:" || u.protocol === "https:";
  } catch {
    return false;
  }
}

interface LocalFileInfo {
  canonicalUrl: string;
  absolutePath: string;
}

function resolveLocalFile(url: string): LocalFileInfo | null {
  if (url.startsWith("file://")) {
    const path = url.slice("file://".length);
    const abs = resolve(path);
    if (existsSync(abs)) {
      return { canonicalUrl: `file://${abs}`, absolutePath: abs };
    }
    throw new Error(`local file not found: ${path}`);
  }
  if (!isHttpUrl(url)) {
    const abs = resolve(url);
    if (existsSync(abs)) {
      return { canonicalUrl: `file://${abs}`, absolutePath: abs };
    }
  }
  return null;
}

async function fetchLocalFile(path: string): Promise<FetcherResult> {
  const file = Bun.file(path);
  const buf = await file.arrayBuffer();
  const decoder = new TextDecoder("utf-8", { fatal: true });
  let content: string;
  try {
    content = decoder.decode(buf);
  } catch {
    throw new Error("local-file fetch supports text files; for binary sources use a URL");
  }
  return {
    content,
    title: basename(path),
    source_type: "local-file",
    publication_date: null,
    author_attribution: null,
  };
}

export async function fetchAndStore(
  url: string,
  opts: { hint?: string; forceRefetch?: boolean; full?: boolean; contentLimit?: number } = {},
): Promise<FetchResult> {
  const local = resolveLocalFile(url);
  const targetUrl = local ? local.canonicalUrl : url;

  // Cache hit: same URL fetched < 24h ago via a real fetcher (not just an
  // agent-submitted quote). Skip refetch.
  if (!opts.forceRefetch) {
    const cached = store.getRecentFetchedDossier(targetUrl, 24);
    if (cached) {
      const fullContent = cached.content || "";
      return {
        dossier_slug: cached.slug,
        source_url: cached.source_url,
        source_type: cached.source_type,
        title: cached.title,
        publication_date: cached.publication_date,
        author_attribution: cached.author_attribution,
        content: sliceContent(fullContent, opts),
        content_chars: fullContent.length,
        cached: true,
        fetched_at: cached.capture_date,
      };
    }
  }

  const result = local
    ? await fetchLocalFile(local.absolutePath)
    : await fetchUrl(url, opts.hint);

  // Embed for hybrid search. Use a representative slice — content might be
  // larger than the embedding model's input cap.
  let embedding: Float32Array | null = null;
  try {
    embedding = await embedOne(result.content.slice(0, 8000));
  } catch {
    // non-fatal — dossier persists, just won't be searchable
  }

  const slug = store.writeDossier({
    source_url: targetUrl,
    source_type: result.source_type,
    title: result.title,
    verbatim_content: result.content,
    embedding,
    publication_date: result.publication_date,
    author_attribution: result.author_attribution,
  });

  return {
    dossier_slug: slug,
    source_url: targetUrl,
    source_type: result.source_type,
    title: result.title,
    publication_date: result.publication_date,
    author_attribution: result.author_attribution,
    content: sliceContent(result.content, opts),
    content_chars: result.content.length,
    cached: false,
    fetched_at: new Date().toISOString(),
    metadata: result.metadata,
  };
}
