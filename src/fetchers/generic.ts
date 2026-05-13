import type { Fetcher, FetcherResult } from "./types.ts";
import {
  htmlToMarkdownViaMarkitdown,
  markitdownAvailable,
  stripInlineMarkdown,
} from "./markitdown.ts";

const USER_AGENT =
  "Mozilla/5.0 (compatible; vouch/0.3) AppleWebKit/537.36";

/** HTML entity decode for the few entities we strip-and-substitute manually.
 * (Keep small — full entity table not worth carrying.) */
function decodeEntities(html: string): string {
  return html
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/&nbsp;/g, " ")
    .replace(/&apos;/g, "'")
    .replace(/&#(\d+);/g, (_, n) => String.fromCharCode(parseInt(n, 10)))
    .replace(/&#x([0-9a-fA-F]+);/g, (_, h) => String.fromCharCode(parseInt(h, 16)));
}

/** Strip HTML tags + script/style content, normalize whitespace. The verifier
 * only needs readable text — we don't need preserved formatting. */
export function stripHtml(html: string): string {
  let s = html;
  // Remove script / style / noscript / svg blocks entirely
  s = s.replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, " ");
  s = s.replace(/<style\b[^<]*(?:(?!<\/style>)<[^<]*)*<\/style>/gi, " ");
  s = s.replace(/<noscript\b[^<]*(?:(?!<\/noscript>)<[^<]*)*<\/noscript>/gi, " ");
  s = s.replace(/<svg\b[^<]*(?:(?!<\/svg>)<[^<]*)*<\/svg>/gi, " ");
  // HTML comments
  s = s.replace(/<!--[\s\S]*?-->/g, " ");
  // Block elements → newline (preserves paragraph boundaries)
  s = s.replace(/<\/(p|div|h[1-6]|li|tr|br|hr|section|article|header|footer|nav|aside|blockquote|pre|table|td|th)>/gi, "\n");
  s = s.replace(/<br\s*\/?>/gi, "\n");
  // Strip remaining tags
  s = s.replace(/<[^>]+>/g, " ");
  s = decodeEntities(s);
  // Collapse whitespace; preserve paragraph breaks
  s = s.replace(/[ \t]+/g, " ");
  s = s.replace(/\n[ \t]+/g, "\n");
  s = s.replace(/\n{3,}/g, "\n\n");
  return s.trim();
}

function extractTitle(html: string): string | null {
  const m = html.match(/<title\b[^>]*>([\s\S]*?)<\/title>/i);
  if (!m || !m[1]) return null;
  return decodeEntities(m[1]).trim().slice(0, 300);
}

function extractMeta(html: string, name: string): string | null {
  // <meta name="..." content="..."> OR <meta property="..." content="...">
  const re = new RegExp(
    `<meta[^>]+(?:name|property)=["']${name}["'][^>]+content=["']([^"']+)["']`,
    "i",
  );
  const m = html.match(re);
  return m && m[1] ? decodeEntities(m[1]).trim() : null;
}

export class GenericFetcher implements Fetcher {
  readonly name = "generic";

  matches(_url: string): boolean {
    return true; // fallback for everything
  }

  async fetch(url: string): Promise<FetcherResult> {
    const resp = await fetch(url, {
      headers: { "User-Agent": USER_AGENT, Accept: "text/html,application/xhtml+xml,*/*;q=0.8" },
      redirect: "follow",
    });
    if (!resp.ok) {
      throw new Error(`fetch ${url} failed: HTTP ${resp.status} ${resp.statusText}`);
    }
    const ctype = resp.headers.get("content-type") || "";
    const body = await resp.text();

    // PDF detection — we don't strip-tag binary; mark as unparsed and bail.
    if (ctype.includes("application/pdf") || body.startsWith("%PDF")) {
      throw new Error(
        `URL ${url} is PDF — generic fetcher can't parse PDFs. Try the HTML version (e.g. arxiv.org/html/<id>) or pre-convert.`,
      );
    }

    // JSON / plain — store as-is
    if (ctype.includes("application/json")) {
      return {
        content: body.slice(0, 200_000),
        title: url.slice(0, 200),
        source_type: "generic",
        publication_date: null,
        author_attribution: null,
        metadata: { content_type: ctype, fetched_at: new Date().toISOString() },
      };
    }
    if (ctype.includes("text/plain") || ctype.includes("text/markdown")) {
      return {
        content: body.slice(0, 200_000),
        title: extractTitle(body) || url.slice(0, 200),
        source_type: "generic",
        publication_date: null,
        author_attribution: null,
        metadata: { content_type: ctype, fetched_at: new Date().toISOString() },
      };
    }

    // HTML path
    const title = extractTitle(body);
    const pubDate =
      extractMeta(body, "article:published_time") ||
      extractMeta(body, "datePublished") ||
      extractMeta(body, "DC.date.issued") ||
      extractMeta(body, "pubdate") ||
      null;
    const author =
      extractMeta(body, "author") ||
      extractMeta(body, "article:author") ||
      extractMeta(body, "DC.creator") ||
      null;

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
    return {
      content: text.slice(0, 200_000),
      title: title || url.slice(0, 200),
      source_type: "generic",
      publication_date: pubDate ? pubDate.slice(0, 10) : null,
      author_attribution: author,
      metadata: {
        content_type: ctype,
        fetched_at: new Date().toISOString(),
        original_html_chars: body.length,
        text_chars: text.length,
        converter,
      },
    };
  }
}
