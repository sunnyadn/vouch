/** claim-batch CLI error-path tests — subprocess style, no LLM calls needed. */
import { beforeAll, afterAll, describe, expect, it } from "bun:test";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const tmp = mkdtempSync(join(tmpdir(), "vouch-claim-batch-test-"));
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
  const code = `
    process.env.VOUCH_DB_PATH = "${DB_PATH.replace(/\\/g, "\\\\").replace(/"/g, '\\"')}";
    const store = await import("./src/store.ts");
    const slug = store.writeDossier({
      source_url: "https://example.com/batch-test",
      source_type: "web-submitted",
      title: "Batch test dossier",
      verbatim_content: "The quick brown fox jumps over the lazy dog. Alpha is the first letter. Beta follows alpha.",
    });
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

describe("claim-batch error paths", () => {
  it("bad quote on line 2 → batch aborted, error names line", async () => {
    const jsonlPath = join(tmp, "bad-quote.jsonl");
    writeFileSync(
      jsonlPath,
      JSON.stringify({
        text: "A fox jumps over a dog",
        claim_type: "ATOMIC",
        dossier_slug: seededSlug,
        source_quote: "The quick brown fox jumps over the lazy dog.",
      }) +
        "\n" +
        JSON.stringify({
          text: "Gamma is last",
          claim_type: "ATOMIC",
          dossier_slug: seededSlug,
          source_quote: "this quote does not exist in the dossier",
        }) +
        "\n",
    );

    const { stdout, stderr, exitCode } = await run(["claim-batch", jsonlPath]);
    expect(exitCode).toBe(1);
    const out = stdout || stderr;
    expect(out).toContain("line 2");
    expect(out).toContain("quote not found");
  });

  it("missing dossier → error names line", async () => {
    const jsonlPath = join(tmp, "missing-dossier.jsonl");
    writeFileSync(
      jsonlPath,
      JSON.stringify({
        text: "A fox jumps over a dog",
        claim_type: "ATOMIC",
        dossier_slug: "nonexistent-dossier",
        source_quote: "foo",
      }) +
        "\n",
    );

    const { stdout, stderr, exitCode } = await run(["claim-batch", jsonlPath]);
    expect(exitCode).toBe(1);
    const out = stdout || stderr;
    expect(out).toContain("line 1");
    expect(out).toContain("dossier not found");
  });

  it("unsupported claim type → error", async () => {
    const jsonlPath = join(tmp, "bad-type.jsonl");
    writeFileSync(
      jsonlPath,
      JSON.stringify({
        text: "A fox jumps over a dog",
        claim_type: "SYNTHESIS",
        dossier_slug: seededSlug,
        source_quote: "foo",
      }) +
        "\n",
    );

    const { stdout, stderr, exitCode } = await run(["claim-batch", jsonlPath]);
    expect(exitCode).toBe(1);
    const out = stdout || stderr;
    expect(out).toContain("claim-batch only supports ATOMIC");
  });

  it("invalid json → error", async () => {
    const jsonlPath = join(tmp, "bad-json.jsonl");
    writeFileSync(jsonlPath, "not json\n");

    const { stdout, stderr, exitCode } = await run(["claim-batch", jsonlPath]);
    expect(exitCode).toBe(1);
    const out = stdout || stderr;
    expect(out).toContain("line 1");
    expect(out).toContain("invalid JSON");
  });
});
