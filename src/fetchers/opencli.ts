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
import { existsSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, isAbsolute } from "node:path";

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
  size?: string | number;
  /** OpenCLI v1.7 envelope field — relative path to saved Markdown file. */
  saved?: string;
  /** Resolved absolute path; populated by the wrapper. */
  resolvedPath?: string;
  content?: string;
  url?: string;
  error?: string;
}

async function readViaOpenCli(url: string, opts: { wait?: number } = {}): Promise<OpenCliReadResult> {
  // Use a per-call temp dir so we can find the output deterministically and
  // clean up after ourselves regardless of the caller's cwd.
  const tmpDir = mkdtempSync(join(tmpdir(), "vouch-opencli-"));
  const args = [
    "web",
    "read",
    "--url",
    url,
    "--output",
    tmpDir,
    "--download-images",
    "false",
    "--wait",
    String(opts.wait ?? 3),
    "--format",
    "json",
  ];
  const proc = Bun.spawn(["opencli", ...args], {
    cwd: tmpDir,
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
  if (r) {
    r.resolvedPath = resolveSavedPath(r.saved, tmpDir);
  }
  // Caller is responsible for reading content first, then cleaning the tmp dir.
  // We stash the path so cleanup can happen in the outer fetch() in a finally.
  (r as any)._tmpDir = tmpDir;
  return r || {};
}

function resolveSavedPath(saved: string | undefined, baseDir: string): string | undefined {
  if (!saved) return undefined;
  if (isAbsolute(saved) && existsSync(saved)) return saved;
  const rel = join(baseDir, saved);
  if (existsSync(rel)) return rel;
  // OpenCLI sometimes writes relative to its own cwd which we set to baseDir,
  // but if a future version changes the prefix we still try a few sensible
  // bases before giving up.
  const candidates = [
    join(process.cwd(), saved),
    join(baseDir, saved.replace(/^web-articles\//, "")),
  ];
  for (const c of candidates) {
    if (existsSync(c)) return c;
  }
  return undefined;
}

export class OpenCliBridgeError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "OpenCliBridgeError";
  }
}

/** Read the markdown content opencli wrote to disk for this URL. */
async function readMarkdownFile(result: OpenCliReadResult): Promise<string> {
  if (result.content && result.content.length > 100) return result.content;
  if (result.resolvedPath && existsSync(result.resolvedPath)) {
    return await Bun.file(result.resolvedPath).text();
  }
  // Last-resort: scan the tmp dir we passed as --output for the newest .md
  const tmpDir = (result as any)._tmpDir as string | undefined;
  if (tmpDir && existsSync(tmpDir)) {
    const files = await Array.fromAsync(
      new Bun.Glob("**/*.md").scan({ cwd: tmpDir }),
    );
    if (files.length) {
      const stats = await Promise.all(
        files.map(async (f) => {
          const full = join(tmpDir, f);
          const stat = await Bun.file(full).stat();
          return { path: full, mtime: stat.mtimeMs };
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
        "opencli CLI is not on PATH. Install: bun install -g @jackwener/opencli",
      );
    }

    const result = await readViaOpenCli(url);
    const tmpDir = (result as any)._tmpDir as string | undefined;
    try {
      const rawMd = await readMarkdownFile(result);
      if (!rawMd || rawMd.length < 50) {
        throw new Error(
          `opencli web read returned empty or near-empty content for ${url} (got ${rawMd.length} chars). saved=${result.saved}`,
        );
      }
      const text = stripInlineMarkdown(rawMd);
      return {
        content: text.slice(0, 300_000),
        title: result.title || url.slice(0, 200),
        source_type: "opencli",
        publication_date: result.publish_time
          ? toIsoDate(result.publish_time)
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
    } finally {
      if (tmpDir && existsSync(tmpDir)) {
        try {
          rmSync(tmpDir, { recursive: true, force: true });
        } catch {
          // ignore cleanup failures
        }
      }
    }
  }
}

function toIsoDate(s: string): string | null {
  // OpenCLI returns dates like "Apr 16, 2026" — try to coerce to YYYY-MM-DD.
  const d = new Date(s);
  if (isNaN(d.getTime())) return null;
  return d.toISOString().slice(0, 10);
}
