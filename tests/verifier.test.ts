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
