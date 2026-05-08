/** Fetcher router + URL-pattern tests (no network calls). */
import { describe, expect, it } from "bun:test";
import { getFetcher } from "../src/fetchers/index.ts";
import { GitHubFetcher, buildMetadataBlock } from "../src/fetchers/github.ts";
import { stripHtml } from "../src/fetchers/generic.ts";
import { stripInlineMarkdown } from "../src/fetchers/markitdown.ts";

describe("fetcher router", () => {
  it("routes arxiv.org URLs to arxiv fetcher", () => {
    expect(getFetcher("https://arxiv.org/abs/2410.05779").name).toBe("arxiv");
    expect(getFetcher("https://www.arxiv.org/pdf/2410.05779").name).toBe("arxiv");
    expect(getFetcher("http://arxiv.org/html/2410.05779v3").name).toBe("arxiv");
  });

  it("falls back to generic for non-arxiv URLs", () => {
    expect(getFetcher("https://en.wikipedia.org/wiki/SQLite").name).toBe("generic");
    expect(getFetcher("https://github.com/foo/bar").name).toBe("github");
  });

  it("respects --fetcher hint", () => {
    expect(getFetcher("https://arxiv.org/abs/0", "generic").name).toBe("generic");
  });

  it("rejects unknown hint", () => {
    expect(() => getFetcher("https://x.test", "nope")).toThrow(/unknown.*fetcher/);
  });
});

describe("stripHtml", () => {
  it("strips tags + decodes entities + collapses whitespace", () => {
    const out = stripHtml("<p>Hello <b>world</b>!</p>\n<script>alert(1)</script>");
    expect(out).not.toContain("<");
    expect(out).not.toContain("alert");
    expect(out).toContain("Hello");
    expect(out).toContain("world");
  });

  it("handles &amp; &lt; &nbsp; correctly", () => {
    const out = stripHtml("<p>A &amp; B &lt; C&nbsp;D</p>");
    expect(out).toContain("A & B < C D");
  });

  it("treats block elements as paragraph boundaries", () => {
    const out = stripHtml("<p>One.</p><p>Two.</p>");
    expect(out).toContain("One.");
    expect(out).toContain("Two.");
  });
});

describe("stripInlineMarkdown", () => {
  it("removes link syntax, keeps text", () => {
    expect(stripInlineMarkdown("Hello [world](http://x.test) today")).toBe("Hello world today");
  });

  it("removes images entirely", () => {
    expect(stripInlineMarkdown("Before ![alt text](http://x.test/img.png) after")).toBe("Before after");
  });

  it("removes heading markers", () => {
    expect(stripInlineMarkdown("# Title\n## Subtitle")).toBe("Title\nSubtitle");
  });

  it("strips bold + italic markers", () => {
    expect(stripInlineMarkdown("Some **bold** and *italic* and __bold__ text")).toBe(
      "Some bold and italic and bold text",
    );
  });

  it("strips list bullets at line start", () => {
    expect(stripInlineMarkdown("- item one\n* item two\n+ item three")).toBe("item one\nitem two\nitem three");
  });

  it("strips numbered list markers", () => {
    expect(stripInlineMarkdown("1. first\n2. second")).toBe("first\nsecond");
  });

  it("strips footnote refs", () => {
    expect(stripInlineMarkdown("Important fact[^1] here")).toBe("Important fact here");
  });
});

describe("fetcher router (github)", () => {
  it("routes github.com repo URLs to github fetcher", () => {
    expect(getFetcher("https://github.com/foo/bar").name).toBe("github");
    expect(getFetcher("https://github.com/foo/bar/").name).toBe("github");
    expect(getFetcher("https://github.com/foo/bar/tree/main").name).toBe("github");
    expect(getFetcher("https://github.com/foo/bar/blob/main/README.md").name).toBe("github");
    expect(getFetcher("https://github.com/foo/bar.git").name).toBe("github");
  });

  it("does NOT route github.com issue/PR URLs to github fetcher", () => {
    expect(getFetcher("https://github.com/foo/bar/issues/123").name).toBe("generic");
    expect(getFetcher("https://github.com/foo/bar/pull/45").name).toBe("generic");
  });

  it("does NOT route non-github URLs", () => {
    expect(getFetcher("https://gitlab.com/foo/bar").name).toBe("generic");
    expect(getFetcher("https://github.io/foo").name).toBe("generic");
  });
});

describe("buildMetadataBlock", () => {
  it("emits exactly one line per fact", () => {
    const block = buildMetadataBlock({
      full_name: "foo/bar",
      description: "test",
      stargazers_count: 100,
      forks_count: 20,
      subscribers_count: 5,
      open_issues_count: 3,
      license: { spdx_id: "MIT", name: "MIT License" },
      default_branch: "main",
      language: "TypeScript",
      created_at: "2024-01-15T00:00:00Z",
      pushed_at: "2026-04-01T00:00:00Z",
      size: 4096,
      html_url: "https://github.com/foo/bar",
      owner: { login: "foo" },
    });
    expect(block).toContain("Stars: 100");
    expect(block).toContain("Forks: 20");
    expect(block).toContain("License: MIT (MIT License)");
    expect(block).toContain("Last pushed: 2026-04-01");
    expect(block.split("\n").length).toBeGreaterThan(8);
  });

  it("handles null license + null description gracefully", () => {
    const block = buildMetadataBlock({
      full_name: "foo/bar",
      description: null,
      stargazers_count: 0,
      forks_count: 0,
      subscribers_count: 0,
      open_issues_count: 0,
      license: null,
      default_branch: "main",
      language: null,
      created_at: "2024-01-15T00:00:00Z",
      pushed_at: "2026-04-01T00:00:00Z",
      size: 0,
      html_url: "https://github.com/foo/bar",
      owner: { login: "foo" },
    });
    expect(block).toContain("License: none (no license)");
    expect(block).toContain("Description: (no description)");
  });
});
