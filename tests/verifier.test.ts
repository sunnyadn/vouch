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
});
