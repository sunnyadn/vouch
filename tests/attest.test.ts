/** Attestation tests — mocks LLM calls, uses real SQLite store. */
import { afterEach, beforeAll, describe, expect, it, mock } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const tmp = mkdtempSync(join(tmpdir(), "vouch-test-"));
process.env.VOUCH_DB_PATH = join(tmp, "test.db");

mock.module("ai", () => ({
  generateObject: () => Promise.resolve({ object: { supported: true, score: 0.9, reason: "test" } }),
  embed: () => Promise.resolve({ embedding: [1, 0, 0, 0] }),
}));

mock.module("../src/providers.ts", () => ({
  getLanguageModel: () => ({ id: "test-model" }) as any,
  getEmbeddingModel: () => ({ id: "test-embedder" }) as any,
}));

mock.module("../src/embedder.ts", () => ({
  embedOne: () => Promise.resolve(new Float32Array([1, 0, 0, 0])),
  embedBatch: () => Promise.resolve([new Float32Array([1, 0, 0, 0])]),
}));

const store = await import("../src/store.ts");
const attest = await import("../src/attest.ts");
const submit = await import("../src/submit.ts");

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

describe("attestAndStore", () => {
  it("rejects invalid slug formats", async () => {
    const badSlugs = ["bad/slug", "bad slug", "BadSlug", "bad.slug"];
    for (const slug of badSlugs) {
      await expect(
        attest.attestAndStore({
          slug,
          content: "Valid content.",
          attribution: "tester",
        }),
      ).rejects.toThrow(/invalid slug format/);
    }
  });

  it("rejects empty content", async () => {
    await expect(
      attest.attestAndStore({
        slug: "valid-slug",
        content: "",
        attribution: "tester",
      }),
    ).rejects.toThrow(/non-empty/);
  });

  it("creates a dossier with correct fields", async () => {
    const result = await attest.attestAndStore({
      slug: "test-attestation",
      content: "This is the attested content.",
      attribution: "sunny",
      date: "2026-05-08",
    });
    expect(result.dossier_slug).toBe("evidence/attestations/test-attestation");
    expect(result.source_type).toBe("user-statement");
    expect(result.source_url).toBe("attestation://test-attestation");
    expect(result.attribution).toBe("sunny");
    expect(result.attestation_date).toBe("2026-05-08");
    expect(result.content_chars).toBe(29);

    const d = store.getDossier(result.dossier_slug);
    expect(d).not.toBeNull();
    expect(d!.source_type).toBe("user-statement");
    expect(d!.source_url).toBe("attestation://test-attestation");
    expect(d!.author_attribution).toBe("sunny");
    expect(d!.publication_date).toBe("2026-05-08");
    expect(d!.content).toBe("This is the attested content.");
    expect(d!.title).toBe("test-attestation");
  });

  it("rejects duplicate slug without force-overwrite", async () => {
    await attest.attestAndStore({
      slug: "duplicate-slug",
      content: "First version.",
      attribution: "tester",
    });
    await expect(
      attest.attestAndStore({
        slug: "duplicate-slug",
        content: "Second version.",
        attribution: "tester",
      }),
    ).rejects.toThrow(/already exists/);
  });

  it("allows force-overwrite to replace content", async () => {
    await attest.attestAndStore({
      slug: "overwrite-slug",
      content: "First version.",
      attribution: "tester",
    });
    const result = await attest.attestAndStore({
      slug: "overwrite-slug",
      content: "Second version.",
      attribution: "tester",
      forceOverwrite: true,
    });
    expect(result.content_chars).toBe(15);
    const d = store.getDossier(result.dossier_slug);
    expect(d!.content).toBe("Second version.");
  });

  it("truncates content over 200KB with warning", async () => {
    const longContent = "x".repeat(200_001);
    const stderrSpy = mock(() => {});
    const originalStderr = process.stderr.write;
    process.stderr.write = stderrSpy as any;

    const result = await attest.attestAndStore({
      slug: "long-content",
      content: longContent,
      attribution: "tester",
    });

    process.stderr.write = originalStderr;
    expect(result.content_chars).toBe(200_000);
    expect(stderrSpy).toHaveBeenCalled();
    expect(stderrSpy.mock.calls[0][0]).toContain("truncating");
  });

  it("defaults date to today when omitted", async () => {
    const today = new Date().toISOString().slice(0, 10);
    const result = await attest.attestAndStore({
      slug: "default-date",
      content: "Some content.",
      attribution: "tester",
    });
    expect(result.attestation_date).toBe(today);
  });
});

describe("claim against attestation", () => {
  it("ATOMIC claim with quote in attestation succeeds", async () => {
    const attestation = await attest.attestAndStore({
      slug: "claimable",
      content: "The quick brown fox jumps over the lazy dog.",
      attribution: "tester",
    });

    const claimResult = await submit.submitClaim({
      text: "A fox jumps over a dog.",
      claim_type: "ATOMIC",
      dossier_slug: attestation.dossier_slug,
      source_quote: "The quick brown fox jumps over the lazy dog.",
    });

    expect(claimResult.error).toBeUndefined();
    expect(claimResult.status).toBe("supported");
    expect(claimResult.quote_match).toBe("exact");
  });

  it("ATOMIC claim with forged quote fails", async () => {
    const attestation = await attest.attestAndStore({
      slug: "forged-test",
      content: "The quick brown fox jumps over the lazy dog.",
      attribution: "tester",
    });

    const claimResult = await submit.submitClaim({
      text: "Cats are the best.",
      claim_type: "ATOMIC",
      dossier_slug: attestation.dossier_slug,
      source_quote: "Cats are the best.",
    });

    expect(claimResult.error).toBeDefined();
    expect(claimResult.reason).toBe("quote-not-in-dossier");
  });

  it("INFERENCE claim depends on attestation-based ATOMIC claim", async () => {
    const attestation = await attest.attestAndStore({
      slug: "inference-base",
      content: "All cats are mammals.",
      attribution: "tester",
    });

    const atomicResult = await submit.submitClaim({
      text: "All cats are mammals.",
      claim_type: "ATOMIC",
      dossier_slug: attestation.dossier_slug,
      source_quote: "All cats are mammals.",
    });
    expect(atomicResult.claim_id).toBeDefined();

    const inferenceResult = await submit.submitClaim({
      text: "Therefore some mammals are cats.",
      claim_type: "INFERENCE",
      depends_on_ids: [atomicResult.claim_id],
    });

    expect(inferenceResult.error).toBeUndefined();
    expect(inferenceResult.status).toBe("supported");
    expect(inferenceResult.verification).toBe("nli-entailment");
  });
});

describe("attest + claim combined (cli.ts spread shape)", () => {
  // Mirrors what `vouch attest --claim "<text>"` does: attestAndStore, then
  // submitClaim against the new dossier, then flat-merge the two responses.
  // The merge itself lives in cli.ts; these tests lock in the contract that
  // the merged object surfaces both halves' fields without collision.
  //
  // We pass source_quote explicitly here so the file's NLI-shaped ai mock
  // does the right thing (autoSelectQuote needs a different mock shape).
  // In production cli.ts omits source_quote and lets auto-quote run.
  it("merges attest + claim responses with both dossier_slug and claim_id", async () => {
    const attestation = await attest.attestAndStore({
      slug: "combined-happy",
      content: "vouch takes the single active OSS-strike slot.",
      attribution: "sunny",
    });

    const claim = await submit.submitClaim({
      text: "vouch takes the single active OSS-strike slot.",
      claim_type: "ATOMIC",
      dossier_slug: attestation.dossier_slug,
      source_quote: "vouch takes the single active OSS-strike slot.",
      author: "claude-skill",
    });

    const merged = { ...attestation, ...claim };
    expect(merged.dossier_slug).toBe("evidence/attestations/combined-happy");
    expect(merged.source_url).toBe("attestation://combined-happy");
    expect(merged.attribution).toBe("sunny");
    expect(merged.claim_id).toBeDefined();
    expect(typeof merged.claim_id).toBe("number");
    expect(merged.status).toBe("supported");
    expect(merged.quote_match).toBe("exact");
  });

  it("response still carries dossier_slug when the claim is unsupported", async () => {
    // Re-mock ai's generateObject to flip the NLI verdict to unsupported.
    // bun:test's mock.module replaces the module-level mock for subsequent
    // imports; we restore the original supported=true mock at the end so
    // later tests in this file aren't affected.
    mock.module("ai", () => ({
      generateObject: () =>
        Promise.resolve({ object: { supported: false, score: 0.1, reason: "no" } }),
      embed: () => Promise.resolve({ embedding: [1, 0, 0, 0] }),
    }));

    try {
      const attestation = await attest.attestAndStore({
        slug: "combined-unsupported",
        content: "The quick brown fox jumps over the lazy dog.",
        attribution: "tester",
      });
      const claim = await submit.submitClaim({
        text: "Cats are unrelated to the content above.",
        claim_type: "ATOMIC",
        dossier_slug: attestation.dossier_slug,
        source_quote: "The quick brown fox jumps over the lazy dog.",
        author: "claude-skill",
      });

      const merged = { ...attestation, ...claim };
      // Even when NLI says unsupported, the merged response must still
      // surface dossier_slug + claim_id so callers can retry the claim text
      // without re-attesting (acceptance bullet 4).
      expect(merged.dossier_slug).toBe("evidence/attestations/combined-unsupported");
      expect(merged.claim_id).toBeDefined();
      expect(typeof merged.claim_id).toBe("number");
      expect(merged.status).toBe("unsupported");
    } finally {
      mock.module("ai", () => ({
        generateObject: () =>
          Promise.resolve({ object: { supported: true, score: 0.9, reason: "test" } }),
        embed: () => Promise.resolve({ embedding: [1, 0, 0, 0] }),
      }));
    }
  });

  it("--claim-type QUOTATION routes through the same dossier-backed path", async () => {
    const attestation = await attest.attestAndStore({
      slug: "combined-quotation",
      content: "All cats are mammals.",
      attribution: "tester",
    });

    const claim = await submit.submitClaim({
      text: "All cats are mammals.",
      claim_type: "QUOTATION",
      dossier_slug: attestation.dossier_slug,
      source_quote: "All cats are mammals.",
      author: "claude-skill",
    });

    const merged = { ...attestation, ...claim };
    expect(merged.dossier_slug).toBe("evidence/attestations/combined-quotation");
    expect(merged.claim_id).toBeDefined();
    expect(merged.status).toBe("supported");
  });
});

describe("list-dossiers includes attestations", () => {
  it("returns attestations with source_type filter", async () => {
    await attest.attestAndStore({
      slug: "listable",
      content: "Attestation content.",
      attribution: "tester",
    });
    store.writeDossier({
      source_url: "https://example.com",
      source_type: "web-submitted",
      title: "Web page",
      verbatim_content: "Web content.",
    });

    const all = store.listDossiers({ limit: 100 });
    expect(all.length).toBeGreaterThanOrEqual(2);

    const attestations = store.listDossiers({ source_type: "user-statement", limit: 100 });
    expect(attestations.length).toBe(1);
    expect(attestations[0]!.slug).toBe("evidence/attestations/listable");
    expect(attestations[0]!.source_type).toBe("user-statement");
  });
});
