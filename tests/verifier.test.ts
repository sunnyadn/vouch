/** Verifier logic tests — mocks LLM calls, uses real SQLite store. */
import { afterEach, beforeAll, describe, expect, it, mock } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const tmp = mkdtempSync(join(tmpdir(), "vouch-test-"));
process.env.VOUCH_DB_PATH = join(tmp, "test.db");

const generateObjectMock = mock(() =>
  Promise.resolve({ object: { supported: true, score: 0.9, reason: "test" } }),
);

mock.module("ai", () => ({
  generateObject: generateObjectMock,
  embed: () => Promise.resolve({ embeddings: [] }),
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
const verifier = await import("../src/verifier.ts");

afterEach(() => {
  const db = store.getDb();
  db.exec("DELETE FROM claim_dependencies; DELETE FROM claims; DELETE FROM dossiers;");
  generateObjectMock.mockClear();
});

beforeAll(() => {
  process.on("exit", () => {
    try {
      rmSync(tmp, { recursive: true, force: true });
    } catch {}
  });
});

// ---------------------------------------------------------------------------
// verifyInferenceClaim
// ---------------------------------------------------------------------------

describe("verifyInferenceClaim", () => {
  it("entailed conclusion", async () => {
    const slug = store.writeDossier({
      source_url: "https://example.com",
      source_type: "test",
      verbatim_content: "All cats are mammals.",
    });
    const upstream = store.recordClaim({
      dossier_slug: slug,
      claim_text: "All cats are mammals.",
      score: 0.95,
      status: "supported",
      claim_type: "ATOMIC",
    });

    generateObjectMock.mockImplementation(() =>
      Promise.resolve({
        object: { supported: true, score: 0.92, reason: "follows directly" },
      }),
    );

    const result = await verifier.verifyInferenceClaim("All cats are mammals.", [upstream]);
    expect(result.status).toBe("supported");
    expect(result.score).toBe(0.92);
    expect(result.source_passage).toBe("follows directly");
    expect(generateObjectMock).toHaveBeenCalledTimes(1);
    const callArgs = generateObjectMock.mock.calls[0][0];
    expect(callArgs.prompt).toContain(`[Claim ${upstream}, ATOMIC, supported]: All cats are mammals.`);
    expect(callArgs.prompt).toContain("All cats are mammals.");
  });

  it("broken chain (unsupported upstream)", async () => {
    const upstream = store.recordClaim({
      dossier_slug: "",
      claim_text: "Dogs are reptiles.",
      score: 0,
      status: "unsupported",
      claim_type: "ATOMIC",
    });

    const result = await verifier.verifyInferenceClaim("Some conclusion.", [upstream]);
    expect(result.status).toBe("unsupported");
    expect(result.source_passage).toContain("broken-chain");
    expect(result.source_passage).toContain(`${upstream} (status=unsupported)`);
    expect(generateObjectMock).toHaveBeenCalledTimes(0);
  });

  it("broken chain (superseded upstream)", async () => {
    const upstream = store.recordClaim({
      dossier_slug: "",
      claim_text: "Old claim.",
      score: 1,
      status: "supported",
      claim_type: "ATOMIC",
    });
    const fresh = store.recordClaim({
      dossier_slug: "",
      claim_text: "New claim.",
      score: 1,
      status: "supported",
      claim_type: "ATOMIC",
    });
    store.supersedeClaim(upstream, fresh, "updated");

    const result = await verifier.verifyInferenceClaim("Some conclusion.", [upstream]);
    expect(result.status).toBe("unsupported");
    expect(result.source_passage).toContain("broken-chain");
    expect(result.source_passage).toContain(`${upstream} (superseded by ${fresh})`);
    expect(generateObjectMock).toHaveBeenCalledTimes(0);
  });

  it("missing upstream", async () => {
    const result = await verifier.verifyInferenceClaim("Some conclusion.", [99999]);
    expect(result.status).toBe("insufficient");
    expect(result.source_passage).toContain("missing claim(s): 99999");
    expect(generateObjectMock).toHaveBeenCalledTimes(0);
  });
});

// ---------------------------------------------------------------------------
// verifyInterpretationClaim
// ---------------------------------------------------------------------------

describe("verifyInterpretationClaim", () => {
  it("single upstream required", async () => {
    const slug = store.writeDossier({
      source_url: "https://example.com",
      source_type: "test",
      verbatim_content: "Test.",
    });
    const a = store.recordClaim({
      dossier_slug: slug,
      claim_text: "A",
      score: 1,
      status: "supported",
      claim_type: "ATOMIC",
    });
    const b = store.recordClaim({
      dossier_slug: slug,
      claim_text: "B",
      score: 1,
      status: "supported",
      claim_type: "ATOMIC",
    });

    const result = await verifier.verifyInterpretationClaim("X", [a, b]);
    expect(result.status).toBe("insufficient");
    expect(result.source_passage).toContain("INTERPRETATION requires exactly one upstream claim");
    expect(generateObjectMock).toHaveBeenCalledTimes(0);
  });

  it("faithful reframing", async () => {
    const upstream = store.recordClaim({
      dossier_slug: "",
      claim_text: "The sky is blue.",
      score: 1,
      status: "supported",
      claim_type: "ATOMIC",
    });

    generateObjectMock.mockImplementation(() =>
      Promise.resolve({
        object: { supported: true, score: 0.95, reason: "same fact" },
      }),
    );

    const result = await verifier.verifyInterpretationClaim("The sky is blue.", [upstream]);
    expect(result.status).toBe("supported");
    expect(result.score).toBe(0.95);
    expect(generateObjectMock).toHaveBeenCalledTimes(1);
  });
});

// ---------------------------------------------------------------------------
// autoSelectQuote
// ---------------------------------------------------------------------------

describe("autoSelectQuote", () => {
  it("found", async () => {
    generateObjectMock.mockImplementation(() =>
      Promise.resolve({
        object: { found: true, quote: "the cat sat on the mat", reason: "direct support" },
      }),
    );

    const result = await verifier.autoSelectQuote(
      "A cat sat on a mat.",
      "Once upon a time the cat sat on the mat. The end.",
    );
    expect(result).not.toBeNull();
    expect(result!.quote).toBe("the cat sat on the mat");
    expect(result!.reason).toBe("direct support");
  });

  it("not found (LLM says none)", async () => {
    generateObjectMock.mockImplementation(() =>
      Promise.resolve({
        object: { found: false, quote: "", reason: "no supporting passage" },
      }),
    );

    const result = await verifier.autoSelectQuote(
      "A cat sat on a mat.",
      "Once upon a time the cat sat on the mat. The end.",
    );
    expect(result).toBeNull();
  });

  it("not found (quote not in dossier)", async () => {
    generateObjectMock.mockImplementation(() =>
      Promise.resolve({
        object: { found: true, quote: "the dog barked loudly", reason: "direct support" },
      }),
    );

    const result = await verifier.autoSelectQuote(
      "A dog barked loudly.",
      "Once upon a time the cat sat on the mat. The end.",
    );
    expect(result).toBeNull();
  });

  // SUN-58: returns dossier's first paragraph as entity-establishing prefix
  it("returns prefix from dossier first paragraph", async () => {
    generateObjectMock.mockImplementation(() =>
      Promise.resolve({
        object: { found: true, quote: "Stars: 66", reason: "stars count" },
      }),
    );

    const dossier =
      "GitHub Repository: sunnyadn/js-toml\nDescription: TOML parser\nStars: 66\nForks: 3\n\n--- README ---\n\nA TOML parser written in TS.";

    const result = await verifier.autoSelectQuote("js-toml has 66 GitHub stars", dossier);
    expect(result).not.toBeNull();
    expect(result!.quote).toBe("Stars: 66");
    expect(result!.prefix).toContain("GitHub Repository: sunnyadn/js-toml");
    expect(result!.prefix).toContain("Stars: 66");
    // Prefix is verbatim — appears at the start of the dossier
    expect(dossier.startsWith(result!.prefix)).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// extractEntityPrefix (SUN-58)
// ---------------------------------------------------------------------------

describe("extractEntityPrefix", () => {
  it("first paragraph delimited by blank line", () => {
    const dossier = "Title line\nMeta line\n\nBody starts here. More text.";
    expect(verifier.extractEntityPrefix(dossier)).toBe("Title line\nMeta line");
  });

  it("entire content if no blank line and short", () => {
    expect(verifier.extractEntityPrefix("Single short line.")).toBe("Single short line.");
  });

  it("trims leading/trailing whitespace", () => {
    expect(verifier.extractEntityPrefix("  Header\n\nbody")).toBe("Header");
  });

  it("caps long blocks at line boundaries", () => {
    const longBlock = Array.from({ length: 30 }, (_, i) => `Line ${i}: ${"x".repeat(40)}`).join("\n");
    const dossier = `${longBlock}\n\nbody`;
    const prefix = verifier.extractEntityPrefix(dossier);
    expect(prefix.length).toBeLessThanOrEqual(600);
    // Ends on a line boundary, never mid-line
    expect(prefix.endsWith("x")).toBe(true);
  });

  it("empty content returns empty", () => {
    expect(verifier.extractEntityPrefix("")).toBe("");
  });
});

// ---------------------------------------------------------------------------
// analyzeTemporalQualifier (SUN-57)
// ---------------------------------------------------------------------------

describe("analyzeTemporalQualifier", () => {
  it("no qualifier — passes claim through", () => {
    const r = verifier.analyzeTemporalQualifier("X has Y stars", {
      capture_date: "2026-05-09T10:00:00Z",
      publication_date: null,
    });
    expect(r.qualifiers).toHaveLength(0);
    expect(r.strippedClaim).toBe("X has Y stars");
    expect(r.mismatchReason).toBeUndefined();
  });

  it("'as of' matches dossier capture_date — strips qualifier", () => {
    const r = verifier.analyzeTemporalQualifier("X has Y stars as of 2026-05-09", {
      capture_date: "2026-05-09T10:00:00Z",
      publication_date: null,
    });
    expect(r.qualifiers).toHaveLength(1);
    expect(r.qualifiers[0]!.kind).toBe("absolute");
    expect(r.strippedClaim).toBe("X has Y stars");
    expect(r.mismatchReason).toBeUndefined();
  });

  it("'as of' matches dossier publication_date — strips qualifier", () => {
    const r = verifier.analyzeTemporalQualifier("X had 4 stars as of 2024-01-15", {
      capture_date: "2026-05-09T10:00:00Z",
      publication_date: "2024-01-15",
    });
    expect(r.mismatchReason).toBeUndefined();
    expect(r.strippedClaim).toBe("X had 4 stars");
  });

  it("'as of' mismatches dossier — flags mismatchReason", () => {
    const r = verifier.analyzeTemporalQualifier("X had Y stars as of 2025-01-01", {
      capture_date: "2026-05-09T10:00:00Z",
      publication_date: null,
    });
    expect(r.mismatchReason).toBeDefined();
    expect(r.mismatchReason).toContain("2025-01-01");
    expect(r.mismatchReason).toContain("2026-05-09");
  });

  it("relative 'at T+48h' qualifier — strips, no mismatch possible", () => {
    const r = verifier.analyzeTemporalQualifier("Reddit post had 120 upvotes at T+48h", {
      capture_date: "2026-05-09T10:00:00Z",
      publication_date: null,
    });
    expect(r.qualifiers).toHaveLength(1);
    expect(r.qualifiers[0]!.kind).toBe("relative");
    expect(r.strippedClaim).toBe("Reddit post had 120 upvotes");
    expect(r.mismatchReason).toBeUndefined();
  });

  it("relative 'at T-2d' qualifier — strips", () => {
    const r = verifier.analyzeTemporalQualifier("Score was 50 at T-2d", {
      capture_date: "2026-05-09T10:00:00Z",
      publication_date: null,
    });
    expect(r.qualifiers).toHaveLength(1);
    expect(r.strippedClaim).toBe("Score was 50");
  });

  it("preserves trailing punctuation when stripping", () => {
    const r = verifier.analyzeTemporalQualifier("X had Y stars as of 2026-05-09.", {
      capture_date: "2026-05-09T10:00:00Z",
      publication_date: null,
    });
    expect(r.strippedClaim).toBe("X had Y stars.");
  });

  it("'on YYYY-MM-DD' is intentionally NOT a qualifier (too ambiguous)", () => {
    const r = verifier.analyzeTemporalQualifier("Project Z was released on 2024-01-15", {
      capture_date: "2026-05-09T10:00:00Z",
      publication_date: null,
    });
    // No qualifier detected — claim passes through unchanged so NLI handles
    // the date as load-bearing factual content.
    expect(r.qualifiers).toHaveLength(0);
    expect(r.strippedClaim).toBe("Project Z was released on 2024-01-15");
    expect(r.mismatchReason).toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// verifyClaim with temporal qualifier (SUN-57 integration)
// ---------------------------------------------------------------------------

describe("verifyClaim temporal qualifier", () => {
  it("'as of' date matching dossier capture — supported via stripped NLI", async () => {
    const slug = store.writeDossier({
      source_url: "https://github.com/x/y",
      source_type: "github",
      verbatim_content: "GitHub Repository: x/y\nStars: 4\nForks: 0",
      captured_at: "2026-05-09T10:00:00Z",
    });

    generateObjectMock.mockImplementation(() =>
      Promise.resolve({
        object: { supported: true, score: 0.95, reason: "stars confirmed" },
      }),
    );

    const result = await verifier.verifyClaim("x/y has 4 stars as of 2026-05-09", slug, {
      source_quote: "Stars: 4",
    });
    expect(result.status).toBe("supported");
    // NLI receives the stripped claim, not the verbatim qualifier
    expect(generateObjectMock).toHaveBeenCalledTimes(1);
    const callArgs = generateObjectMock.mock.calls[0][0];
    expect(callArgs.prompt).toContain('CLAIM: "x/y has 4 stars"');
    expect(callArgs.prompt).not.toContain("as of 2026-05-09");
    // Stored claim_text preserves the qualifier verbatim
    const stored = store.getClaim(result.claim_id);
    expect(stored!.claim_text).toBe("x/y has 4 stars as of 2026-05-09");
  });

  it("'as of' date mismatch — short-circuits unsupported, no NLI call", async () => {
    const slug = store.writeDossier({
      source_url: "https://github.com/x/y",
      source_type: "github",
      verbatim_content: "GitHub Repository: x/y\nStars: 4",
      captured_at: "2026-05-09T10:00:00Z",
    });

    generateObjectMock.mockImplementation(() =>
      Promise.resolve({ object: { supported: true, score: 1, reason: "should not be called" } }),
    );

    const result = await verifier.verifyClaim("x/y has 4 stars as of 2025-01-01", slug, {
      source_quote: "Stars: 4",
    });
    expect(result.status).toBe("unsupported");
    expect(result.source_passage).toContain("2025-01-01");
    expect(result.source_passage).toContain("2026-05-09");
    expect(generateObjectMock).toHaveBeenCalledTimes(0);
    // Still recorded — failed attempts have audit value
    const stored = store.getClaim(result.claim_id);
    expect(stored).not.toBeNull();
    expect(stored!.status).toBe("unsupported");
    expect(stored!.claim_text).toBe("x/y has 4 stars as of 2025-01-01");
  });
});

// ---------------------------------------------------------------------------
// verifyClaimsBatch (SUN-7)
// ---------------------------------------------------------------------------

describe("verifyClaimsBatch", () => {
  it("happy path: returns results in submission order", async () => {
    generateObjectMock.mockImplementation(() =>
      Promise.resolve({
        object: {
          verdicts: [
            { idx: 0, supported: true, score: 0.95, reason: "direct match" },
            { idx: 1, supported: false, score: 0.2, reason: "no evidence" },
          ],
        },
      } as any),
    );

    const results = await verifier.verifyClaimsBatch([
      { claim_text: "A", source_passage: "A is true." },
      { claim_text: "B", source_passage: "B is false." },
    ]);

    expect(results).toHaveLength(2);
    expect(results[0]!.status).toBe("supported");
    expect(results[0]!.score).toBe(0.95);
    expect(results[1]!.status).toBe("unsupported");
    expect(results[1]!.score).toBe(0.2);
    expect(generateObjectMock).toHaveBeenCalledTimes(1);
    const callArgs = (generateObjectMock.mock.calls[0] as any)[0];
    expect(callArgs.schema).toBeDefined();
    expect(callArgs.prompt).toContain("[0] CLAIM");
    expect(callArgs.prompt).toContain("[1] CLAIM");
  });

  it("single item delegates to verifyClaimAgainstSource", async () => {
    generateObjectMock.mockImplementation(() =>
      Promise.resolve({ object: { supported: true, score: 0.9, reason: "ok" } }),
    );

    const results = await verifier.verifyClaimsBatch([
      { claim_text: "Only one", source_passage: "Only one source." },
    ]);

    expect(results).toHaveLength(1);
    expect(results[0]!.status).toBe("supported");
    // verifyClaimAgainstSource uses the single-item VerifySchema
    expect(generateObjectMock).toHaveBeenCalledTimes(1);
    const callArgs = (generateObjectMock.mock.calls[0] as any)[0];
    expect(callArgs.prompt).not.toContain("verdicts");
  });

  it("empty array returns empty", async () => {
    const results = await verifier.verifyClaimsBatch([]);
    expect(results).toHaveLength(0);
    expect(generateObjectMock).toHaveBeenCalledTimes(0);
  });

  it("idx drift (missing idx) rejects the whole batch", async () => {
    generateObjectMock.mockImplementation(() =>
      Promise.resolve({
        object: {
          verdicts: [
            { idx: 0, supported: true, score: 0.9, reason: "ok" },
            // missing idx 1
          ],
        },
      } as any),
    );

    await expect(
      verifier.verifyClaimsBatch([
        { claim_text: "A", source_passage: "A." },
        { claim_text: "B", source_passage: "B." },
      ]),
    ).rejects.toThrow("Batch verifier returned 1 verdicts for 2 claims");
  });

  it("idx drift (wrong idx) rejects the whole batch", async () => {
    generateObjectMock.mockImplementation(() =>
      Promise.resolve({
        object: {
          verdicts: [
            { idx: 0, supported: true, score: 0.9, reason: "ok" },
            { idx: 5, supported: false, score: 0.1, reason: "bad" },
          ],
        },
      } as any),
    );

    await expect(
      verifier.verifyClaimsBatch([
        { claim_text: "A", source_passage: "A." },
        { claim_text: "B", source_passage: "B." },
      ]),
    ).rejects.toThrow("idx mismatch at position 1: got idx 5");
  });

  it("idx drift (duplicate idx) rejects the whole batch", async () => {
    generateObjectMock.mockImplementation(() =>
      Promise.resolve({
        object: {
          verdicts: [
            { idx: 0, supported: true, score: 0.9, reason: "ok" },
            { idx: 0, supported: false, score: 0.1, reason: "dup" },
          ],
        },
      } as any),
    );

    await expect(
      verifier.verifyClaimsBatch([
        { claim_text: "A", source_passage: "A." },
        { claim_text: "B", source_passage: "B." },
      ]),
    ).rejects.toThrow("idx mismatch at position 1: got idx 0");
  });

  it("falls back to sequential when prompt exceeds token budget", async () => {
    generateObjectMock.mockImplementation(() =>
      Promise.resolve({
        object: { supported: true, score: 0.9, reason: "ok" },
      }),
    );

    // 6 items × 55 k chars = 330 k chars > BATCH_MAX_PROMPT_CHARS (300 k)
    const big = "x".repeat(55_000);
    const items = Array.from({ length: 6 }, (_, i) => ({
      claim_text: `Claim ${i}`,
      source_passage: big,
    }));

    const results = await verifier.verifyClaimsBatch(items);

    expect(results).toHaveLength(6);
    // Fallback calls verifyClaimAgainstSource individually (single-item schema)
    expect(generateObjectMock).toHaveBeenCalledTimes(6);
    for (let i = 0; i < 6; i++) {
      expect((generateObjectMock.mock.calls[i] as any)[0].prompt).not.toContain("verdicts");
    }
  });

  it("throws TransientVerifierError on generateObject failure", async () => {
    generateObjectMock.mockImplementation(() =>
      Promise.reject(new Error("rate limit exceeded")),
    );

    await expect(
      verifier.verifyClaimsBatch([
        { claim_text: "A", source_passage: "A." },
        { claim_text: "B", source_passage: "B." },
      ]),
    ).rejects.toBeInstanceOf(verifier.TransientVerifierError);
  });
});

// ---------------------------------------------------------------------------
// submitClaimBatch (SUN-7 integration)
// ---------------------------------------------------------------------------

const submit = await import("../src/submit.ts");

describe("submitClaimBatch", () => {
  it("happy path: 2 valid items → 2 claims persisted", async () => {
    const slug = store.writeDossier({
      source_url: "https://example.com/batch",
      source_type: "test",
      verbatim_content: "Alpha is first. Beta is second.",
    });

    generateObjectMock.mockImplementation(() =>
      Promise.resolve({
        object: {
          verdicts: [
            { idx: 0, supported: true, score: 0.95, reason: "direct" },
            { idx: 1, supported: false, score: 0.1, reason: "no match" },
          ],
        },
      } as any),
    );

    const results = await submit.submitClaimBatch([
      {
        text: "Alpha is first",
        dossier_slug: slug,
        source_quote: "Alpha is first.",
      },
      {
        text: "Gamma is third",
        dossier_slug: slug,
        source_quote: "Beta is second.",
      },
    ]);

    expect(results).toHaveLength(2);
    expect(results[0]!.status).toBe("supported");
    expect(results[0]!.claim_id).toBeGreaterThan(0);
    expect(results[1]!.status).toBe("unsupported");

    // Verify DB state
    const c1 = store.getClaim(results[0]!.claim_id);
    expect(c1).not.toBeNull();
    expect(c1!.claim_text).toBe("Alpha is first");
    const c2 = store.getClaim(results[1]!.claim_id);
    expect(c2).not.toBeNull();
    expect(c2!.claim_text).toBe("Gamma is third");
  });

  it("bad quote on item 2 → throws, nothing persisted", async () => {
    const slug = store.writeDossier({
      source_url: "https://example.com/batch2",
      source_type: "test",
      verbatim_content: "Only this content.",
    });

    // Pre-seed a claim so we can verify count after failure
    const beforeCount = store.listClaims({ dossier_slug: slug }).length;

    await expect(
      submit.submitClaimBatch([
        {
          text: "Only this content",
          dossier_slug: slug,
          source_quote: "Only this content.",
        },
        {
          text: "Fake claim",
          dossier_slug: slug,
          source_quote: "this quote does not exist",
        },
      ]),
    ).rejects.toThrow("quote not found");

    const afterCount = store.listClaims({ dossier_slug: slug }).length;
    expect(afterCount).toBe(beforeCount);
  });
});
