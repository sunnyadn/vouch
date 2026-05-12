/** Digest tests — pure store queries, no LLM calls. */
import { afterEach, beforeAll, describe, expect, it } from "bun:test";
import { mkdtempSync, rmSync, writeFileSync, readFileSync, appendFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const tmp = mkdtempSync(join(tmpdir(), "vouch-test-"));
process.env.VOUCH_DB_PATH = join(tmp, "test.db");

const store = await import("../src/store.ts");

afterEach(() => {
  const db = store.getDb();
  db.exec("DELETE FROM claim_dependencies; DELETE FROM claims; DELETE FROM dossiers;");
});

beforeAll(() => {
  process.on("exit", () => {
    try {
      rmSync(tmp, { recursive: true, force: true });
    } catch {}
  });
});

describe("getDigest", () => {
  it("returns claims of each type + dossier + dependency + supersede", () => {
    const slug = store.writeDossier({
      source_url: "https://example.com",
      source_type: "web-submitted",
      title: "Test",
      verbatim_content: "The quick brown fox jumps over the lazy dog.",
    });

    const atomic = store.recordClaim({
      dossier_slug: slug,
      claim_text: "A fox jumps over a dog.",
      score: 0.95,
      status: "supported",
      claim_type: "ATOMIC",
      source_passage: "The quick brown fox jumps over the lazy dog.",
      verification: "nli-quote",
    });

    const inference = store.recordClaim({
      dossier_slug: "",
      claim_text: "Some mammals are cats.",
      score: null,
      status: "recorded",
      claim_type: "INFERENCE",
      depends_on_ids: [atomic],
      author: "gate-harvest",
      verification: "tag-harvest",
    });

    const since = new Date(Date.now() - 60000).toISOString();
    const digest = store.getDigest(since);

    expect(digest.claims.length).toBe(1);
    expect(digest.claims[0]!.claim_type).toBe("ATOMIC");
    expect(digest.derived_claims.length).toBe(1);
    expect(digest.derived_claims[0]!.kind).toBe("inference-from");
    expect(digest.derived_claims[0]!.upstream_ids).toContain(atomic);
    expect(digest.dependencies.length).toBe(1);
    expect(digest.dependencies[0]!.claim_id).toBe(inference);
    expect(digest.dependencies[0]!.depends_on_ids).toContain(atomic);
    expect(digest.dossiers.length).toBe(1);
    expect(digest.dossiers[0]!.slug).toBe(slug);
  });

  it("includes multiple claim types", () => {
    const atomic = store.recordClaim({
      dossier_slug: "",
      claim_text: "Base fact",
      score: 1,
      status: "supported",
      claim_type: "ATOMIC",
      verification: "nli-quote",
    });

    const synthesis = store.recordClaim({
      dossier_slug: "",
      claim_text: "Synthesized",
      score: null,
      status: "recorded",
      claim_type: "SYNTHESIS",
      depends_on_ids: [atomic],
      author: "gate-harvest",
      verification: "tag-harvest",
    });

    const interpretation = store.recordClaim({
      dossier_slug: "",
      claim_text: "Interpreted",
      score: null,
      status: "recorded",
      claim_type: "INTERPRETATION",
      depends_on_ids: [atomic],
      author: "gate-harvest",
      verification: "tag-harvest",
    });

    const hypothesis = store.recordClaim({
      dossier_slug: "",
      claim_text: "Hypothesized",
      score: null,
      status: "recorded",
      claim_type: "HYPOTHESIS",
      author: "gate-harvest",
      verification: "tag-harvest",
    });

    const since = new Date(Date.now() - 60000).toISOString();
    const digest = store.getDigest(since);

    expect(digest.claims.length).toBe(1); // ATOMIC
    expect(digest.derived_claims.length).toBe(3);
    const kinds = digest.derived_claims.map((d) => d.kind).sort();
    expect(kinds).toEqual(["hypothesis", "interpretation", "synthesis-of"]);

    const syn = digest.derived_claims.find((d) => d.kind === "synthesis-of")!;
    expect(syn.upstream_ids).toContain(atomic);
  });

  it("includes supersedes when new claim is in window", () => {
    const old = store.recordClaim({
      dossier_slug: "",
      claim_text: "old claim",
      score: 1,
      status: "supported",
      claim_type: "ATOMIC",
    });

    const fresh = store.recordClaim({
      dossier_slug: "",
      claim_text: "new claim",
      score: 1,
      status: "supported",
      claim_type: "ATOMIC",
    });

    store.supersedeClaim(old, fresh, "better evidence");

    const since = new Date(Date.now() - 60000).toISOString();
    const digest = store.getDigest(since);
    expect(digest.supersedes.length).toBe(1);
    expect(digest.supersedes[0]!.old_id).toBe(old);
    expect(digest.supersedes[0]!.new_id).toBe(fresh);
    expect(digest.supersedes[0]!.reason).toBe("better evidence");
  });

  it("returns empty for empty window", () => {
    const since = new Date(Date.now() + 1000).toISOString();
    const digest = store.getDigest(since);
    expect(digest.claims.length).toBe(0);
    expect(digest.derived_claims.length).toBe(0);
    expect(digest.supersedes.length).toBe(0);
    expect(digest.dossiers.length).toBe(0);
    expect(digest.dependencies.length).toBe(0);
  });
});

describe("getSessionStart", () => {
  it("detects gap and returns session start", () => {
    const db = store.getDb();
    const now = Date.now();

    db.prepare(
      `INSERT INTO claims (dossier_slug, claim_text, nli_score, status, verified_at, claim_type)
       VALUES (?, ?, ?, ?, ?, ?)`,
    ).run("", "recent", 1, "supported", new Date(now).toISOString(), "ATOMIC");

    db.prepare(
      `INSERT INTO claims (dossier_slug, claim_text, nli_score, status, verified_at, claim_type)
       VALUES (?, ?, ?, ?, ?, ?)`,
    ).run("", "old", 1, "supported", new Date(now - 4 * 3600 * 1000).toISOString(), "ATOMIC");

    const start = store.getSessionStart(2);
    const startMs = new Date(start).getTime();
    expect(startMs).toBeGreaterThanOrEqual(now - 60000);
    expect(startMs).toBeLessThanOrEqual(now + 60000);
  });

  it("falls back to 24h when table is sparse", () => {
    const db = store.getDb();
    db.prepare(
      `INSERT INTO claims (dossier_slug, claim_text, nli_score, status, verified_at, claim_type)
       VALUES (?, ?, ?, ?, ?, ?)`,
    ).run("", "only", 1, "supported", new Date().toISOString(), "ATOMIC");

    const start = store.getSessionStart(2);
    const startMs = new Date(start).getTime();
    const ago = Date.now() - startMs;
    expect(ago).toBeGreaterThan(23 * 3600 * 1000);
    expect(ago).toBeLessThanOrEqual(24 * 3600 * 1000 + 60000);
  });

  it("uses dossier capture_date alongside claim verified_at", () => {
    const db = store.getDb();
    const now = Date.now();

    // Add 3 dossiers within a short window (no claims) — not sparse
    for (let i = 0; i < 3; i++) {
      db.prepare(
        `INSERT INTO dossiers (slug, source_url, source_type, capture_date, content)
         VALUES (?, ?, ?, ?, ?)`,
      ).run(`evidence/web/test-${i}`, "https://test.com", "web-submitted", new Date(now - i * 1000).toISOString(), "content");
    }

    const start = store.getSessionStart(2);
    const startMs = new Date(start).getTime();
    // Oldest dossier is ~2s ago
    expect(startMs).toBeGreaterThanOrEqual(now - 3000);
    expect(startMs).toBeLessThanOrEqual(now + 60000);
  });
});

describe("formatDigestMarkdown", () => {
  it("formats empty digest", () => {
    const digest = store.getDigest(new Date(Date.now() + 1000).toISOString());
    const md = store.formatDigestMarkdown(digest);
    expect(md.includes("(nothing entered the KB in this window)")).toBe(true);
  });

  it("includes all sections for populated digest", () => {
    const slug = store.writeDossier({
      source_url: "https://example.com",
      source_type: "web-submitted",
      verbatim_content: "test content",
    });

    store.recordClaim({
      dossier_slug: slug,
      claim_text: "Test claim with a long text that might be truncated if it exceeds the limit",
      score: 0.92,
      status: "supported",
      claim_type: "ATOMIC",
      source_passage: "test content",
      verification: "nli-quote",
    });

    const since = new Date(Date.now() - 60000).toISOString();
    const digest = store.getDigest(since);
    const md = store.formatDigestMarkdown(digest);

    expect(md.includes("## New verified claims")).toBe(true);
    expect(md.includes("score 0.92")).toBe(true);
    expect(md.includes("Dossier:")).toBe(true);
    expect(md.includes("## New dossiers")).toBe(true);
  });

  it("truncates long quotes to ~160 chars", () => {
    const slug = store.writeDossier({
      source_url: "https://example.com",
      source_type: "web-submitted",
      verbatim_content: "x",
    });

    const longQuote = "a".repeat(300);
    store.recordClaim({
      dossier_slug: slug,
      claim_text: "Short",
      score: 1,
      status: "supported",
      claim_type: "ATOMIC",
      source_passage: longQuote,
      verification: "nli-quote",
    });

    const since = new Date(Date.now() - 60000).toISOString();
    const digest = store.getDigest(since);
    const md = store.formatDigestMarkdown(digest);

    const quoteMatch = md.match(/"a{160}…"/);
    expect(quoteMatch).not.toBeNull();
  });

  it("appends to file without clobbering", () => {
    const path = join(tmp, "digest-append-test.md");
    writeFileSync(path, "# Existing content\n\n");

    const slug = store.writeDossier({
      source_url: "https://example.com",
      source_type: "web-submitted",
      verbatim_content: "test",
    });

    store.recordClaim({
      dossier_slug: slug,
      claim_text: "Test claim",
      score: 0.9,
      status: "supported",
      claim_type: "ATOMIC",
      verification: "nli-quote",
    });

    const since = new Date(Date.now() - 60000).toISOString();
    const digest = store.getDigest(since);
    const md = store.formatDigestMarkdown(digest);
    appendFileSync(path, md);

    const result = readFileSync(path, "utf8");
    expect(result.startsWith("# Existing content\n\n")).toBe(true);
    expect(result.includes("Test claim")).toBe(true);
    expect(result.includes("## New verified claims")).toBe(true);
  });
});
