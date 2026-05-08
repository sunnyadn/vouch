/** OpenCLI fetcher — wraps `opencli web read` for JS-rendered pages.
 *
 *  Why: vouch's generic httpx fetcher captures only the initial HTML response.
 *  Modern SPAs (Vercel/Next dashboards, Lemon Squeezy blogs, many vendor docs
 *  pages) render content in JS after load — generic fetcher catches the
 *  shell, OpenCLI captures the rendered DOM.
 *
 *  Usage: `vouch fetch <url> --fetcher opencli` to force, OR generic auto-
 *  falls-back to opencli when its content looks thin/shell-only.
 *
 *  Dep: OpenCLI CLI installed (sunny has it as ~/.bun/bin/opencli) AND the
 *  Chrome browser-bridge extension connected (`opencli doctor` should be
 *  green). If either is missing this fetcher throws a clear actionable error.
 */
import { existsSync } from "node:fs";
import { join } from "node:path";

import type { Fetcher, FetcherResult } from "./types.ts";
import { stripInlineMarkdown } from "./markitdown.ts";

let _available: boolean | null = null;

export function opencliAvailable(): boolean {
  if (_available !== null) return _available;
  try {
    const proc = Bun.spawnSync(["opencli", "--version"], {
      stdout: "ignore",
      stderr: "ignore",
    });
    _available = proc.exitCode === 0;
  } catch {
    _available = false;
  }
  return _available;
}

interface OpenCliReadResult {
  status?: string;
  title?: string;
  author?: string;
  publish_time?: string;
  size?: number;
  content?: string;
  output_path?: string;
  url?: string;
  error?: string;
}

async function readViaOpenCli(url: string, opts: { wait?: number } = {}): Promise<OpenCliReadResult> {
  const args = [
    "web",
    "read",
    "--url",
    url,
    "--download-images",
    "false",
    "--wait",
    String(opts.wait ?? 3),
    "--format",
    "json",
  ];
  const proc = Bun.spawn(["opencli", ...args], {
    stdout: "pipe",
    stderr: "pipe",
  });
  const stdout = await new Response(proc.stdout).text();
  const stderr = await new Response(proc.stderr).text();
  const exitCode = await proc.exited;

  if (exitCode !== 0) {
    if (stderr.includes("Browser Bridge not connected") || stderr.includes("Extension")) {
      throw new OpenCliBridgeError(
        "OpenCLI browser bridge extension is not connected. Install from " +
          "https://github.com/jackwener/opencli/releases — load unpacked in chrome://extensions, " +
          "then re-run `opencli doctor` until green.",
      );
    }
    throw new Error(`opencli web read exited ${exitCode}: ${stderr.slice(0, 400) || stdout.slice(0, 400)}`);
  }

  // Parse JSON envelope
  let parsed: any;
  try {
    parsed = JSON.parse(stdout);
  } catch (e) {
    throw new Error(`opencli web read returned non-JSON output: ${stdout.slice(0, 200)}`);
  }

  const r: OpenCliReadResult = Array.isArray(parsed) ? parsed[0] : parsed;
  if (r?.error) throw new Error(`opencli web read: ${r.error}`);
  if (r?.status && r.status !== "success" && r.status !== "ok") {
    throw new Error(`opencli web read status=${r.status}: ${JSON.stringify(r).slice(0, 300)}`);
  }
  return r || {};
}

export class OpenCliBridgeError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "OpenCliBridgeError";
  }
}

/** Read the markdown content opencli wrote to disk for this URL. opencli's
 *  web read emits the body to a file (in --output dir) and only reports
 *  metadata + the file path in the JSON envelope. */
async function readMarkdownFile(result: OpenCliReadResult): Promise<string> {
  if (result.content && result.content.length > 100) return result.content;
  const path = (result as any).output_path || (result as any).path || (result as any).file;
  if (path && existsSync(path)) {
    return await Bun.file(path).text();
  }
  // Some opencli versions write to ./web-articles/<slug>.md
  // Fallback: scan default output dir for the most recent .md file
  const defaultDir = join(process.cwd(), "web-articles");
  if (existsSync(defaultDir)) {
    const files = await Array.fromAsync(
      new Bun.Glob("*.md").scan({ cwd: defaultDir }),
    );
    if (files.length) {
      // Use most recently modified
      const stats = await Promise.all(
        files.map(async (f) => {
          const fullPath = join(defaultDir, f);
          const stat = await Bun.file(fullPath).stat();
          return { path: fullPath, mtime: stat.mtimeMs };
        }),
      );
      stats.sort((a, b) => b.mtime - a.mtime);
      return await Bun.file(stats[0]!.path).text();
    }
  }
  return "";
}

export class OpenCliFetcher implements Fetcher {
  readonly name = "opencli";

  matches(_url: string): boolean {
    // OpenCLI fetcher is opt-in via --fetcher opencli OR auto-fallback when
    // generic content looks thin. Doesn't claim to match URLs proactively.
    return false;
  }

  async fetch(url: string): Promise<FetcherResult> {
    if (!opencliAvailable()) {
      throw new Error(
        "opencli CLI is not on PATH. Install: npm install -g @jackwener/opencli",
      );
    }

    const result = await readViaOpenCli(url);
    const rawMd = await readMarkdownFile(result);
    if (!rawMd || rawMd.length < 50) {
      throw new Error(
        `opencli web read returned empty or near-empty content for ${url} (got ${rawMd.length} chars)`,
      );
    }

    const text = stripInlineMarkdown(rawMd);
    return {
      content: text.slice(0, 300_000),
      title: result.title || url.slice(0, 200),
      source_type: "opencli",
      publication_date: result.publish_time
        ? result.publish_time.slice(0, 10)
        : null,
      author_attribution: result.author || null,
      metadata: {
        fetched_via: "opencli web read",
        fetched_at: new Date().toISOString(),
        size_reported: result.size,
        text_chars: text.length,
        original_md_chars: rawMd.length,
      },
    };
  }
}
