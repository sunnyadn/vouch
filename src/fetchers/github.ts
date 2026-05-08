import type { Fetcher, FetcherResult } from "./types.ts";
import { GenericFetcher } from "./generic.ts";

const GITHUB_RE =
  /^https?:\/\/github\.com\/([^\/?#]+)\/([^\/?#]+?)(?:\.git)?(?:\/(?:tree|blob)\/.*)?\/?(?:[?#].*)?$/i;

interface GhRepo {
  full_name: string;
  description: string | null;
  stargazers_count: number;
  forks_count: number;
  subscribers_count?: number;
  watchers_count?: number;
  open_issues_count: number;
  license: { spdx_id: string | null; name: string } | null;
  default_branch: string;
  language: string | null;
  created_at: string;
  pushed_at: string;
  size: number;
  html_url: string;
  owner: { login: string };
}

let _ghAvailable: boolean | null = null;

function ghAvailable(): boolean {
  if (_ghAvailable !== null) return _ghAvailable;
  try {
    const proc = Bun.spawnSync(["gh", "--version"], {
      stdout: "ignore",
      stderr: "ignore",
    });
    _ghAvailable = proc.exitCode === 0;
  } catch {
    _ghAvailable = false;
  }
  return _ghAvailable;
}

async function spawnGh(
  args: string[],
): Promise<{ ok: true; stdout: string } | { ok: false; reason: string }> {
  try {
    const proc = Bun.spawn(["gh", ...args], {
      stdout: "pipe",
      stderr: "pipe",
    });

    const timeoutId = setTimeout(() => {
      try {
        proc.kill("SIGTERM");
      } catch {
        /* ignore */
      }
    }, 30_000);

    const stdout = await new Response(proc.stdout).text();
    const stderr = await new Response(proc.stderr).text();
    const exitCode = await proc.exited;
    clearTimeout(timeoutId);

    if (exitCode !== 0) {
      const reason = stderr.trim() || `gh exited with code ${exitCode}`;
      return { ok: false, reason };
    }
    return { ok: true, stdout };
  } catch (e) {
    return {
      ok: false,
      reason: e instanceof Error ? e.message : String(e),
    };
  }
}

export function buildMetadataBlock(repo: GhRepo): string {
  const lines: string[] = [
    `GitHub Repository: ${repo.full_name}`,
    `Description: ${repo.description ?? "(no description)"}`,
    `Stars: ${repo.stargazers_count}`,
    `Forks: ${repo.forks_count}`,
  ];

  const watchers =
    repo.subscribers_count ?? repo.watchers_count;
  if (watchers !== undefined) {
    lines.push(`Watchers: ${watchers}`);
  }

  lines.push(
    `Open issues: ${repo.open_issues_count}`,
    `License: ${repo.license?.spdx_id ?? "none"} (${repo.license?.name ?? "no license"})`,
    `Default branch: ${repo.default_branch}`,
    `Primary language: ${repo.language ?? "(none detected)"}`,
    `Created: ${repo.created_at.slice(0, 10)}`,
    `Last pushed: ${repo.pushed_at.slice(0, 10)}`,
    `Size (KB): ${repo.size}`,
    `URL: ${repo.html_url}`,
  );

  return lines.join("\n");
}

export class GitHubFetcher implements Fetcher {
  readonly name = "github";

  matches(url: string): boolean {
    return GITHUB_RE.test(url);
  }

  async fetch(url: string): Promise<FetcherResult> {
    const m = url.match(GITHUB_RE);
    if (!m) throw new Error(`not a github repo URL: ${url}`);
    const [, owner, repo] = m;

    if (!ghAvailable()) {
      const generic = await new GenericFetcher().fetch(url);
      return {
        ...generic,
        metadata: { ...(generic.metadata || {}), gh_unavailable: true },
      };
    }

    const metaResult = await spawnGh([
      "api",
      `repos/${owner}/${repo}`,
    ]);
    if (!metaResult.ok) {
      let reason = metaResult.reason;
      if (
        reason.includes("authentication") ||
        reason.includes("auth") ||
        reason.includes("401") ||
        reason.includes("403")
      ) {
        reason +=
          " (hint: run `gh auth status` to verify GitHub CLI authentication)";
      }
      throw new Error(`gh api repos/${owner}/${repo} failed: ${reason}`);
    }
    const repoData = JSON.parse(metaResult.stdout) as GhRepo;
    const metadataBlock = buildMetadataBlock(repoData);

    let readme = "";
    const readmeResult = await spawnGh([
      "api",
      `repos/${owner}/${repo}/readme`,
    ]);
    if (readmeResult.ok) {
      const data = JSON.parse(readmeResult.stdout);
      if (data.content && data.encoding === "base64") {
        readme = Buffer.from(data.content, "base64").toString("utf-8");
      }
    }

    const content = readme
      ? `${metadataBlock}\n\n--- README ---\n\n${readme}`
      : metadataBlock;

    return {
      content: content.slice(0, 200_000),
      title: repoData.full_name,
      source_type: "github",
      publication_date: repoData.pushed_at.slice(0, 10),
      author_attribution: repoData.owner.login,
      metadata: {
        stars: repoData.stargazers_count,
        forks: repoData.forks_count,
        license: repoData.license?.spdx_id ?? null,
        html_url: repoData.html_url,
        default_branch: repoData.default_branch,
        fetched_at: new Date().toISOString(),
      },
    };
  }
}
