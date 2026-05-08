/** Common fetcher interface. Each fetcher knows how to turn a URL into a
 * dossier-ready content blob with metadata.
 *
 * vouch's trust model: the fetcher is the trust boundary. Whatever bytes the
 * fetcher returns, vouch persists as the dossier — anti-fabrication checks
 * (quote-in-content) run against this. So fetchers should:
 *   - error loudly on failure (don't silently return placeholder text)
 *   - normalize encoding, strip boilerplate (nav/ads/script) before returning
 *   - capture publication_date / author_attribution into metadata when readily
 *     available (arxiv API, GitHub API, RSS)
 */

export interface FetcherResult {
  /** Cleaned text content (HTML stripped, whitespace normalized). This is
   *  what gets stored as dossier.content and what quote-in-content checks run
   *  against. */
  content: string;
  /** Fetcher-derived title. Falls back to URL slice if the source has none. */
  title: string;
  /** Stable label for the source category — "arxiv" | "github" | "wikipedia"
   *  | "generic" | "twitter" | etc. Used for routing and analytics. */
  source_type: string;
  /** Date the source was published / authored, when extractable. ISO 8601 (YYYY-MM-DD). */
  publication_date: string | null;
  /** Author / org attribution for the source, when extractable. */
  author_attribution: string | null;
  /** Free-form additional metadata (paper id, repo owner, etc.). */
  metadata?: Record<string, unknown>;
}

export interface Fetcher {
  /** Domain or pattern this fetcher handles. */
  readonly name: string;
  /** Returns true if this fetcher should handle the given URL. */
  matches(url: string): boolean;
  /** Fetch + normalize. Throws on hard failure (network, 4xx, 5xx). */
  fetch(url: string): Promise<FetcherResult>;
}
