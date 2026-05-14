/** P-α counter-evidence pull — gate looks for KB claims that CONTRADICT a
 *  pair the agent's gate would otherwise pass, and flips it to ungrounded
 *  with the counter_evidence payload attached.
 *
 *  Opt-in via VOUCH_GATE_COUNTER_EVIDENCE=1; default off.
 *
 *  Mocks the LLM to return a controlled contradicts-or-not verdict so we
 *  test the gate plumbing (search + verifyContradiction wiring + flip
 *  logic + counter_evidence attachment) without depending on real NLI.
 */
import { afterEach, beforeAll, describe, expect, it, mock } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const tmp = mkdtempSync(join(tmpdir(), "vouch-counter-test-"));
process.env.VOUCH_DB_PATH = join(tmp, "test.db");

/** Per-call mock controller. Each call to generateObject pops the next
 *  response — lets us script different LLM calls (extractor / grounding /
 *  contradiction) in sequence. */
let mockResponses: Array<any> = [];

const generateObjectMock = mock(() => {
  const next = mockResponses.shift();
  if (!next) {
    return Promise.resolve({
      object: { supported: true, score: 0.9, reason: "default-mock" },
    });
  }
  return Promise.resolve({ object: next });
});

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
const gate = await import("../src/gate.ts");

afterEach(() => {
  const db = store.getDb();
  db.exec(
    "DELETE FROM claim_dependencies; DELETE FROM claims; DELETE FROM dossiers; DELETE FROM session_claims;",
  );
  mockResponses = [];
  delete process.env.VOUCH_GATE_COUNTER_EVIDENCE;
  generateObjectMock.mockClear();
});

beforeAll(() => {
  process.on("exit", () => {
    try {
      rmSync(tmp, { recursive: true, force: true });
    } catch {}
  });
});

/** Stage a confirming claim and a contradicting claim in the KB for the
 *  same entity. The claim texts must NOT share ≥80% tokens with the
 *  test draft, else Phase 1's lexical short-circuit grounds via jaccard
 *  before Phase 2 NLI runs (and our LLM mock never gets called).
 *  Returns the dossier slug + claim ids. */
function seedConfirmAndContra(entity: string): {
  dossierSlug: string;
  confirmId: number;
  contraId: number;
} {
  const dossierSlug = store.writeDossier({
    source_url: `https://example.com/${entity}`,
    source_type: "agent-quote",
    title: `${entity} reference`,
    verbatim_content: `${entity} property documentation. Long-form text body explaining many distinct aspects of the entity in question.`,
  });
  // Both claims must reference the entity (sharesPrimaryEntity uses
  // case-insensitive substring) but should NOT share ≥80% lemma tokens
  // with the draft used in the test, so Phase 1 doesn't lexically
  // short-circuit. The mock NLI controls the actual outcome.
  const confirmId = store.recordClaim({
    dossier_slug: dossierSlug,
    claim_text: `the ${entity} project happens to operate via property aleph in mode beta.`,
    score: 0.95,
    status: "supported",
    source_passage: `confirm-quote about ${entity} happens via aleph.`,
    claim_type: "ATOMIC",
    verification: "nli-quote",
    embedding: new Float32Array([1, 0, 0, 0]),
  });
  const contraId = store.recordClaim({
    dossier_slug: dossierSlug,
    claim_text: `the ${entity} project never operates via property aleph in mode gamma.`,
    score: 0.95,
    status: "supported",
    source_passage: `contra-quote about ${entity} never via aleph.`,
    claim_type: "ATOMIC",
    verification: "nli-quote",
    embedding: new Float32Array([1, 0, 0, 0]),
  });
  return { dossierSlug, confirmId, contraId };
}

describe("P-α counter-evidence pull (#50 axis: comprehensiveness)", () => {
  // Implementation note for these tests: with mock-embedder returning the
  // same [1,0,0,0] vector for every embedOne call, every dossier-claim
  // shows up with cosine=1.0. Phase 1 hybrid search returns the seeded
  // confirm+contra claims; Phase 1 also runs an embedding-shortcut at
  // COSINE_SHORTCUT (0.95) + sharesPrimaryEntity, which short-circuits to
  // grounded WITHOUT calling Phase 2's batch NLI. So in the mock-mode
  // tests below the FIRST LLM call after extractor is verifyContradiction
  // (P-α), NOT a verdicts-batch. Mocks must reflect this ordering.

  it("default (env unset): does NOT run counter-evidence pass — pair stays grounded", async () => {
    delete process.env.VOUCH_GATE_COUNTER_EVIDENCE;
    seedConfirmAndContra("EntityFoo");

    mockResponses = [
      { pairs: [{ proposition: "EntityFoo operates via aleph mode beta.", stance: "ASSERT", entity: "EntityFoo" }] },
      // No more responses needed — Phase 1 cosine-shortcut grounds, P-α
      // is OFF, no further LLM calls.
    ];

    const verdict = await gate.runGate({
      draft: "EntityFoo operates via aleph mode beta.",
      model: "test-model",
      topK: 3,
    });

    expect(verdict.blocked).toBe(false);
    expect(verdict.pairs[0]?.grounded).toBe(true);
    expect(verdict.pairs[0]?.counter_evidence).toBeUndefined();
  });

  it("env=1: detects strong contradiction and flips pair to ungrounded with counter_evidence", async () => {
    process.env.VOUCH_GATE_COUNTER_EVIDENCE = "1";
    seedConfirmAndContra("EntityBar");

    mockResponses = [
      { pairs: [{ proposition: "EntityBar operates via aleph mode beta.", stance: "ASSERT", entity: "EntityBar" }] },
      // P-α verifyContradiction over the contra candidate (confirm is
      // skipped via matched_claim_id check):
      { contradicts: true, score: 0.9, reason: "the two claims have opposite truth values" },
    ];

    const verdict = await gate.runGate({
      draft: "EntityBar operates via aleph mode beta.",
      model: "test-model",
      topK: 3,
    });

    expect(verdict.blocked).toBe(true);
    expect(verdict.pairs[0]?.grounded).toBe(false);
    expect(verdict.pairs[0]?.counter_evidence?.length).toBeGreaterThan(0);
    expect(verdict.pairs[0]?.reason).toContain("contradicted by claim");
    expect(verdict.pairs[0]?.reason).toContain("counter-evidence requires reconciliation");
  });

  it("env=1 but contradiction score below threshold: pair stays grounded", async () => {
    process.env.VOUCH_GATE_COUNTER_EVIDENCE = "1";
    seedConfirmAndContra("EntityBaz");

    mockResponses = [
      { pairs: [{ proposition: "EntityBaz operates via aleph mode beta.", stance: "ASSERT", entity: "EntityBaz" }] },
      // verifyContradiction below COUNTER_EVIDENCE_FIRE_SCORE (0.75) → no flip
      { contradicts: true, score: 0.5, reason: "weak overlap, not strong contradiction" },
    ];

    const verdict = await gate.runGate({
      draft: "EntityBaz operates via aleph mode beta.",
      model: "test-model",
      topK: 3,
    });

    expect(verdict.blocked).toBe(false);
    expect(verdict.pairs[0]?.grounded).toBe(true);
    expect(verdict.pairs[0]?.counter_evidence ?? []).toEqual([]);
  });

  it("env=1 + verifier returns contradicts=false: pair stays grounded", async () => {
    process.env.VOUCH_GATE_COUNTER_EVIDENCE = "1";
    seedConfirmAndContra("EntityQux");

    mockResponses = [
      { pairs: [{ proposition: "EntityQux operates via aleph mode beta.", stance: "ASSERT", entity: "EntityQux" }] },
      // High score but contradicts=false → no flip
      { contradicts: false, score: 0.9, reason: "actually compatible" },
    ];

    const verdict = await gate.runGate({
      draft: "EntityQux operates via aleph mode beta.",
      model: "test-model",
      topK: 3,
    });

    expect(verdict.blocked).toBe(false);
    expect(verdict.pairs[0]?.grounded).toBe(true);
  });

  it("env=1 + pair on novel entity (no KB match): P-α has nothing to surface", async () => {
    process.env.VOUCH_GATE_COUNTER_EVIDENCE = "1";
    // Seed claims for a DIFFERENT entity so sharesPrimaryEntity fails for
    // the pair's entity. Phase 1 returns no candidates → pair ungrounded.
    // (Test wraps the "P-α only acts on grounded pairs" invariant: if no
    // KB match exists for the fired entity, P-α's branch is no-op.)
    const dossierSlug = store.writeDossier({
      source_url: "https://example.com/UnrelatedEntity",
      source_type: "agent-quote",
      title: "Unrelated reference",
      verbatim_content: "UnrelatedEntity property documentation.",
    });
    store.recordClaim({
      dossier_slug: dossierSlug,
      claim_text: "the UnrelatedEntity project operates via property aleph.",
      score: 0.95,
      status: "supported",
      source_passage: "irrelevant",
      claim_type: "ATOMIC",
      verification: "nli-quote",
      embedding: new Float32Array([1, 0, 0, 0]),
    });

    mockResponses = [
      { pairs: [{ proposition: "EntityZed operates via aleph mode beta.", stance: "ASSERT", entity: "EntityZed" }] },
      // Phase 2 might fire on the cosine-1.0 candidate (sharesPrimaryEntity
      // would fail in Phase 1 secondary filter for i>=topK, but i=0 is
      // primary and skips the entity-share check). So Phase 2 batch NLI
      // runs on the unrelated candidate and must return supported=false
      // for the pair to stay ungrounded.
      { supported: false, score: 0.2, reason: "different entity, no support" },
      // P-α shouldn't consume more. If it does, this is the verdict.
      { contradicts: true, score: 0.99, reason: "should not be called" },
    ];

    const verdict = await gate.runGate({
      draft: "EntityZed operates via aleph mode beta.",
      model: "test-model",
      topK: 3,
    });

    expect(verdict.blocked).toBe(true);
    expect(verdict.pairs[0]?.grounded).toBe(false);
    expect(verdict.pairs[0]?.counter_evidence).toBeUndefined();
    // Contradicts response should still be queued.
    expect(mockResponses.length).toBe(1);
  });
});
