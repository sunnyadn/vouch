/** markitdown subprocess wrapper. Produces cleaner HTML→Markdown than the
 *  in-process stripHtml: preserves heading structure, link text, code blocks,
 *  and punctuation spacing.
 *
 *  Falls through to in-process stripping if markitdown isn't on PATH (so
 *  vouch still works without the optional dep).
 */
import { tmpdir } from "node:os";
import { writeFileSync, unlinkSync } from "node:fs";
import { join } from "node:path";
import { randomBytes } from "node:crypto";

let _available: boolean | null = null;

export function markitdownAvailable(): boolean {
  if (_available !== null) return _available;
  try {
    const proc = Bun.spawnSync(["markitdown", "--help"], {
      stdout: "ignore",
      stderr: "ignore",
    });
    _available = proc.exitCode === 0;
  } catch {
    _available = false;
  }
  return _available;
}

/** Strip inline Markdown syntax that breaks natural-prose quote matching.
 *  Keeps paragraph structure and list bullets, but removes `[text](url)`,
 *  `**bold**`, `*italic*`, heading markers, image refs.
 *
 *  Quotes humans write are typically rendered prose ("free and open-source
 *  database"), not raw Markdown ("[free and open-source](...) database").
 *  Without this strip the quote-in-dossier check fails on most docs. */
export function stripInlineMarkdown(md: string): string {
  let s = md;
  // Images: ![alt](url) → drop entirely
  s = s.replace(/!\[[^\]]*\]\([^)]*\)/g, "");
  // Inline links: [text](url) → text
  s = s.replace(/\[([^\]]*)\]\(([^)]*)\)/g, "$1");
  // Reference-style links: [text][id] → text
  s = s.replace(/\[([^\]]*)\]\[[^\]]*\]/g, "$1");
  // Footnote-style refs: [^1]
  s = s.replace(/\[\^[^\]]+\]/g, "");
  // Heading markers at line start: # ## ### → drop the marker
  s = s.replace(/^\s{0,3}#{1,6}\s+/gm, "");
  // Bold/italic with ** or __ — strip the markers, keep content
  s = s.replace(/\*\*([^*]+)\*\*/g, "$1");
  s = s.replace(/__([^_]+)__/g, "$1");
  // Single * / _ italics (avoid greedy across lines)
  s = s.replace(/(?<!\*)\*([^*\n]+)\*(?!\*)/g, "$1");
  s = s.replace(/(?<!_)_([^_\n]+)_(?!_)/g, "$1");
  // Inline code → keep content, drop backticks
  s = s.replace(/`([^`]+)`/g, "$1");
  // Blockquote markers ">" at line start
  s = s.replace(/^\s{0,3}>\s?/gm, "");
  // List bullet markers "*", "-", "+" at line start with space
  s = s.replace(/^\s{0,3}[-*+]\s+/gm, "");
  // Numbered list markers "1." at line start
  s = s.replace(/^\s{0,3}\d+\.\s+/gm, "");
  // Collapse runs of horizontal whitespace caused by removed inline elements
  // (e.g. an image stripped from "Before ![alt](u) after" leaves "Before  after").
  s = s.replace(/[ \t]{2,}/g, " ");
  // Collapse 3+ newlines to 2
  s = s.replace(/\n{3,}/g, "\n\n");
  // Strip residual leading/trailing whitespace per line
  s = s
    .split("\n")
    .map((l) => l.replace(/[ \t]+$/, "").replace(/^[ \t]+/, ""))
    .join("\n");
  return s.trim();
}

/** Convert raw HTML to Markdown via the markitdown CLI. Throws on failure
 *  (caller should catch and fall through to dumb stripping). */
export async function htmlToMarkdownViaMarkitdown(html: string): Promise<string> {
  // markitdown reads file most reliably; stdin sometimes mis-detects MIME.
  // Write to temp .html so the format is unambiguous.
  const tmpPath = join(tmpdir(), `vouch-${randomBytes(6).toString("hex")}.html`);
  writeFileSync(tmpPath, html, "utf-8");
  try {
    const proc = Bun.spawn(["markitdown", tmpPath], {
      stdout: "pipe",
      stderr: "pipe",
    });
    const out = await new Response(proc.stdout).text();
    const err = await new Response(proc.stderr).text();
    const code = await proc.exited;
    if (code !== 0) {
      throw new Error(`markitdown exited ${code}: ${err.slice(0, 300)}`);
    }
    return out;
  } finally {
    try {
      unlinkSync(tmpPath);
    } catch {
      // ignore cleanup failures
    }
  }
}
