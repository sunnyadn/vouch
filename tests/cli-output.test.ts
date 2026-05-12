/** CLI output-mode tests — human-readable default + --json round-trip. */
import { beforeAll, afterAll, describe, expect, it } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const tmp = mkdtempSync(join(tmpdir(), "vouch-cli-out-test-"));
const DB_PATH = join(tmp, "test.db");

const REPO_ROOT = join(import.meta.dir, "..");

async function run(args: string[], env?: Record<string, string>): Promise<{ stdout: string; stderr: string; exitCode: number }> {
  const proc = Bun.spawn(["bun", "run", "src/cli.ts", ...args], {
    cwd: REPO_ROOT,
    env: { ...process.env, VOUCH_DB_PATH: DB_PATH, ...env },
    stdout: "pipe",
    stderr: "pipe",
  });
  const stdout = await new Response(proc.stdout).text();
  const stderr = await new Response(proc.stderr).text();
  const exitCode = await proc.exited;
  return { stdout: stdout.trimEnd(), stderr: stderr.trimEnd(), exitCode };
}

let seededSlug = "";

beforeAll(async () => {
  // Seed via a fresh subprocess so config.ts reads VOUCH_DB_PATH correctly
  // (avoids module-cache issues when running alongside other test files).
  const code = `
    process.env.VOUCH_DB_PATH = "${DB_PATH.replace(/\\/g, "\\\\").replace(/"/g, '\\"')}";
    const store = await import("./src/store.ts");
    const slug = store.writeDossier({
      source_url: "https://example.com/test",
      source_type: "web-submitted",
      title: "Test page",
      verbatim_content: "The quick brown fox jumps over the lazy dog. This is a test dossier with enough content to be interesting for preview truncation.",
    });
    const c1 = store.recordClaim({
      dossier_slug: slug,
      claim_text: "A fox jumps over a dog",
      score: 0.95,
      status: "supported",
      claim_type: "ATOMIC",
      topic: "animals",
      author: "claude-skill",
      verification: "nli-quote",
    });
    const c2 = store.recordClaim({
      dossier_slug: slug,
      claim_text: "Foxes are agile",
      score: 0.88,
      status: "supported",
      claim_type: "INFERENCE",
      topic: "animals",
      author: "gate-harvest",
      verification: "nli-entailment",
      depends_on_ids: [c1],
    });
    store.recordClaim({
      dossier_slug: "",
      claim_text: "Maybe dogs are lazy",
      score: null,
      status: "recorded",
      claim_type: "HYPOTHESIS",
      topic: "animals",
      author: "user-edit",
      verification: "none",
    });
    store.supersedeClaim(c1, c2, "better wording");
    console.log(slug);
  `;
  const proc = Bun.spawn(["bun", "-e", code], {
    cwd: REPO_ROOT,
    env: { ...process.env, VOUCH_DB_PATH: DB_PATH },
    stdout: "pipe",
    stderr: "pipe",
  });
  const stdout = await new Response(proc.stdout).text();
  const stderr = await new Response(proc.stderr).text();
  const exitCode = await proc.exited;
  if (exitCode !== 0) throw new Error(`seed failed: ${stderr}`);
  seededSlug = stdout.trim();
});

afterAll(() => {
  try {
    rmSync(tmp, { recursive: true, force: true });
  } catch {}
});

// ---------------------------------------------------------------------------
// Representative commands — human + JSON
// ---------------------------------------------------------------------------

describe("list-claims", () => {
  it("human output contains key fields", async () => {
    const { stdout, exitCode } = await run(["list-claims"]);
    expect(exitCode).toBe(0);
    expect(stdout).toContain("#");
    expect(stdout).toContain("ATOMIC");
    expect(stdout).toContain("INFERENCE");
    expect(stdout).toContain("HYPOTHESIS");
    expect(stdout).toContain("supported");
    expect(stdout).toContain("claude-skill");
  });

  it("--json round-trips the object", async () => {
    const { stdout, exitCode } = await run(["list-claims", "--json"]);
    expect(exitCode).toBe(0);
    const parsed = JSON.parse(stdout);
    expect(Array.isArray(parsed)).toBe(true);
    expect(parsed.length).toBeGreaterThanOrEqual(3);
    const types = parsed.map((c: any) => c.claim_type);
    expect(types).toContain("ATOMIC");
    expect(types).toContain("INFERENCE");
    expect(types).toContain("HYPOTHESIS");
  });
});

describe("get-claim", () => {
  it("human output contains key fields", async () => {
    const { stdout, exitCode } = await run(["get-claim", "1"]);
    expect(exitCode).toBe(0);
    expect(stdout).toContain("#1");
    expect(stdout).toContain("ATOMIC");
    expect(stdout).toContain("supported");
    expect(stdout).toContain("A fox jumps over a dog");
  });

  it("--json round-trips the object", async () => {
    const { stdout, exitCode } = await run(["get-claim", "1", "--json"]);
    expect(exitCode).toBe(0);
    const parsed = JSON.parse(stdout);
    expect(parsed.id).toBe(1);
    expect(parsed.claim_type).toBe("ATOMIC");
    expect(parsed.status).toBe("supported");
  });
});

describe("recent", () => {
  it("human output contains header and claims", async () => {
    const { stdout, exitCode } = await run(["recent"]);
    expect(exitCode).toBe(0);
    expect(stdout).toContain("new");
    expect(stdout).toContain("supported");
    expect(stdout).toContain("ATOMIC");
  });

  it("--json round-trips the object", async () => {
    const { stdout, exitCode } = await run(["recent", "--json"]);
    expect(exitCode).toBe(0);
    const parsed = JSON.parse(stdout);
    expect(parsed.summary).toBeDefined();
    expect(Array.isArray(parsed.claims)).toBe(true);
  });
});

describe("list-dossiers", () => {
  it("human output contains key fields", async () => {
    const { stdout, exitCode } = await run(["list-dossiers"]);
    expect(exitCode).toBe(0);
    expect(stdout).toContain("evidence/web-submitted/");
    expect(stdout).toContain("web-submitted");
  });

  it("--json round-trips the object", async () => {
    const { stdout, exitCode } = await run(["list-dossiers", "--json"]);
    expect(exitCode).toBe(0);
    const parsed = JSON.parse(stdout);
    expect(Array.isArray(parsed)).toBe(true);
    expect(parsed[0].slug).toContain("evidence/web-submitted/");
  });
});

describe("list-topics", () => {
  it("human output contains key fields", async () => {
    const { stdout, exitCode } = await run(["list-topics"]);
    expect(exitCode).toBe(0);
    expect(stdout).toContain("animals");
    expect(stdout).toContain("claim");
  });

  it("--json round-trips the object", async () => {
    const { stdout, exitCode } = await run(["list-topics", "--json"]);
    expect(exitCode).toBe(0);
    const parsed = JSON.parse(stdout);
    expect(Array.isArray(parsed)).toBe(true);
    expect(parsed.some((t: any) => t.topic === "animals")).toBe(true);
  });
});

describe("get-dossier", () => {
  it("human output contains key fields", async () => {
    const { stdout, exitCode } = await run(["get-dossier", seededSlug]);
    expect(exitCode).toBe(0);
    expect(stdout).toContain("source:");
    expect(stdout).toContain("type:");
    expect(stdout).toContain("chars:");
  });

  it("--json round-trips the object", async () => {
    const { stdout, exitCode } = await run(["get-dossier", seededSlug, "--json"]);
    expect(exitCode).toBe(0);
    const parsed = JSON.parse(stdout);
    expect(parsed.slug).toBe(seededSlug);
    expect(parsed.content_total_chars).toBeGreaterThan(0);
  });
});

describe("chain", () => {
  it("human output renders a tree", async () => {
    const { stdout, exitCode } = await run(["chain", "2"]);
    expect(exitCode).toBe(0);
    expect(stdout).toContain("#2");
    expect(stdout).toContain("INFERENCE");
  });

  it("--json round-trips the object", async () => {
    const { stdout, exitCode } = await run(["chain", "2", "--json"]);
    expect(exitCode).toBe(0);
    const parsed = JSON.parse(stdout);
    expect(parsed.root).toBe(2);
    expect(parsed.node_count).toBeGreaterThanOrEqual(1);
  });
});

describe("supersede", () => {
  it("human output is a one-line confirmation", async () => {
    const { stdout, exitCode } = await run(["supersede", "1", "2", "--reason", "test reason"]);
    expect(exitCode).toBe(0);
    expect(stdout).toContain("✓ #1 superseded by #2");
    expect(stdout).toContain("test reason");
  });

  it("--json round-trips the object", async () => {
    const { stdout, exitCode } = await run(["supersede", "1", "2", "--reason", "test reason", "--json"]);
    expect(exitCode).toBe(0);
    const parsed = JSON.parse(stdout);
    expect(parsed.ok).toBe(true);
  });
});

describe("doctor", () => {
  it("human output contains check lines", async () => {
    const { stdout, exitCode } = await run(["doctor"]);
    expect(exitCode).toBe(0);
    expect(stdout).toContain("[");
    expect(stdout).toContain("db:");
  });

  it("--json round-trips the object", async () => {
    const { stdout, exitCode } = await run(["doctor", "--json"]);
    expect(exitCode).toBe(0);
    const parsed = JSON.parse(stdout);
    expect(typeof parsed.ok).toBe("boolean");
    expect(Array.isArray(parsed.checks)).toBe(true);
  });
});

describe("gate output unchanged", () => {
  it("emits compact JSON even without --json", async () => {
    const { stdout, exitCode } = await run(["gate"], { VOUCH_GATE_BYPASS: "1" });
    expect(exitCode).toBe(0);
    expect(stdout).toBe('{"blocked":false,"pairs":[]}');
  });
});

describe("VOUCH_OUTPUT=json env", () => {
  it("forces JSON output when env is set", async () => {
    const { stdout, exitCode } = await run(["list-topics"], { VOUCH_OUTPUT: "json" });
    expect(exitCode).toBe(0);
    const parsed = JSON.parse(stdout);
    expect(Array.isArray(parsed)).toBe(true);
  });
});
