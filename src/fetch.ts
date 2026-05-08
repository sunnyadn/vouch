/** vouch fetch — high-level: route URL to fetcher, persist as dossier with
 *  embedding. This is the primary trust-establishing step in the new
 *  architecture: claims must reference dossiers vouch fetched itself.
 */
import { fetchUrl } from "./fetchers/index.ts";
import { embedOne } from "./embedder.ts";
import * as store from "./store.ts";

interface FetchResult {
  dossier_slug: string;
  source_url: string;
  source_type: string;
  title: string | null;
  publication_date: string | null;
  author_attribution: string | null;
  content_chars: number;
  cached: boolean;
  fetched_at: string;
}

export async function fetchAndStore(url: string, opts: { hint?: string; forceRefetch?: boolean } = {}): Promise<FetchResult> {
  // Cache hit: same URL fetched < 24h ago via a real fetcher (not just an
  // agent-submitted quote). Skip refetch.
  if (!opts.forceRefetch) {
    const cached = store.getRecentFetchedDossier(url, 24);
    if (cached) {
      return {
        dossier_slug: cached.slug,
        source_url: cached.source_url,
        source_type: cached.source_type,
        title: cached.title,
        publication_date: cached.publication_date,
        author_attribution: cached.author_attribution,
        content_chars: (cached.content || "").length,
        cached: true,
        fetched_at: cached.capture_date,
      };
    }
  }

  const result = await fetchUrl(url, opts.hint);

  // Embed for hybrid search. Use a representative slice — content might be
  // larger than the embedding model's input cap.
  let embedding: Float32Array | null = null;
  try {
    embedding = await embedOne(result.content.slice(0, 8000));
  } catch {
    // non-fatal — dossier persists, just won't be searchable
  }

  const slug = store.writeDossier({
    source_url: url,
    source_type: result.source_type,
    title: result.title,
    verbatim_content: result.content,
    embedding,
    publication_date: result.publication_date,
    author_attribution: result.author_attribution,
  });

  return {
    dossier_slug: slug,
    source_url: url,
    source_type: result.source_type,
    title: result.title,
    publication_date: result.publication_date,
    author_attribution: result.author_attribution,
    content_chars: result.content.length,
    cached: false,
    fetched_at: new Date().toISOString(),
  };
}
