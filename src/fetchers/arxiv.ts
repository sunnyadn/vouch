import { GenericFetcher, stripHtml } from "./generic.ts";
import {
  htmlToMarkdownViaMarkitdown,
  markitdownAvailable,
  stripInlineMarkdown,
} from "./markitdown.ts";
import type { Fetcher, FetcherResult } from "./types.ts";

const ARXIV_RE = /(?:^|\/\/)(?:www\.)?arxiv\.org\/(abs|pdf|html)\/(\d{4}\.\d{4,5})(v\d+)?/i;

interface ArxivMetadata {
  title: string;
  authors: string[];
  published: string | null;
}

async function fetchArxivMetadata(paperId: string): Promise<ArxivMetadata | null> {
  try {
    const resp = await fetch(
      `https://export.arxiv.org/api/query?id_list=${paperId}`,
      { headers: { "User-Agent": "vouch/0.1 (arxiv-metadata)" } },
    );
    if (!resp.ok) return null;
    const xml = await resp.text();
    const title = (xml.match(/<title[^>]*>([\s\S]*?)<\/title>/g) || [])[1]
      ?.replace(/<\/?title[^>]*>/g, "")
      .trim()
      .replace(/\s+/g, " ");
    const authors = [...xml.matchAll(/<author>\s*<name>([^<]+)<\/name>/g)].map(
      (m) => m[1]!.trim(),
    );
    const published = xml.match(/<published>([^<]+)<\/published>/)?.[1]?.slice(0, 10) || null;
    if (!title) return null;
    return { title, authors, published };
  } catch {
    return null;
  }
}

export class ArxivFetcher implements Fetcher {
  readonly name = "arxiv";

  matches(url: string): boolean {
    return ARXIV_RE.test(url);
  }

  async fetch(url: string): Promise<FetcherResult> {
    const m = url.match(ARXIV_RE);
    if (!m) throw new Error(`not an arxiv URL: ${url}`);
    const paperId = m[2]!;

    // Always fetch the HTML version — easier to parse than PDF, less truncated
    // than the abstract page. Falls back to abstract if HTML 404s (older papers).
    const htmlUrl = `https://arxiv.org/html/${paperId}`;
    const absUrl = `https://arxiv.org/abs/${paperId}`;

    let body = "";
    let actualUrl = htmlUrl;
    try {
      const r = await fetch(htmlUrl, {
        headers: { "User-Agent": "vouch/0.1 (arxiv)" },
        redirect: "follow",
      });
      if (r.ok) body = await r.text();
      else {
        // Fall back to abs page
        const r2 = await fetch(absUrl, { headers: { "User-Agent": "vouch/0.1 (arxiv)" } });
        if (!r2.ok) throw new Error(`arxiv ${absUrl}: HTTP ${r2.status}`);
        body = await r2.text();
        actualUrl = absUrl;
      }
    } catch (e) {
      throw new Error(`arxiv fetch failed for ${paperId}: ${e}`);
    }

    let text: string;
    let converter: "markitdown" | "internal";
    if (markitdownAvailable()) {
      try {
        const md = await htmlToMarkdownViaMarkitdown(body);
        text = stripInlineMarkdown(md);
        converter = "markitdown";
      } catch {
        text = stripHtml(body);
        converter = "internal";
      }
    } else {
      text = stripHtml(body);
      converter = "internal";
    }
    const meta = await fetchArxivMetadata(paperId);

    return {
      content: text.slice(0, 300_000),
      title: meta?.title || `arxiv:${paperId}`,
      source_type: "arxiv",
      publication_date: meta?.published ?? null,
      author_attribution: meta?.authors?.length
        ? meta.authors.length > 3
          ? `${meta.authors[0]} et al.`
          : meta.authors.join(", ")
        : null,
      metadata: {
        paper_id: paperId,
        fetched_url: actualUrl,
        fetched_at: new Date().toISOString(),
        text_chars: text.length,
        all_authors: meta?.authors || [],
        converter,
      },
    };
  }
}

// Re-export GenericFetcher for the index/router
export { GenericFetcher };
