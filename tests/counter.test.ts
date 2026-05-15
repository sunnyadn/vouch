/** counter.ts unit tests — focused on findCounterEvidence in isolation.
 *
 *  The gate integration with counter-evidence is covered by
 *  counter-evidence.test.ts (which exercises gate.runGate end-to-end with
 *  VOUCH_GATE_COUNTER_EVIDENCE=1). This file covers the extracted module's
 *  contract directly:
 *
 *    - entity scoping (skip claims that don't mention the entity)
 *    - status/superseded filters
 *    - excludeClaimId
 *    - maxHits cap
 *    - empty KB fail-soft
 *
 *  Mocks the LLM (verifyContradiction → generateObject) so we control which
 *  candidates "contradict" without standing up a real NLI judge.
 */
import { afterEach, beforeAll, describe, expect, it, mock } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const tmp = mkdtempSync(join(tmpdir(), "vouch-counter-unit-"));
process.env.VOUCH_DB_PATH = join(tmp, "test.db");

let mockResponses: Array<{ contradicts: boolean; score: number; reason: string }> = [];

const generateObjectMock = mock(() => {
  const next = mockResponses.shift();
  if (!next) {
    return Promise.resolve({
      object: { contradicts: false, score: 0, reason: "default-mock-no-contra" },
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
const { findCounterEvidence } = await import("../src/counter.ts");

afterEach(() => {
  const db = store.getDb();
  db.exec("DELETE FROM claim_dependencies; DELETE FROM claims; DELETE FROM dossiers;");
  mockResponses = [];
  generateObjectMock.mockClear();
});

beforeAll(() => {
  store.getDb();
});

function seedClaim(opts: {
  text: string;
  topic?: string;
  status?: string;
  superseded_by?: number | null;
  dossier_slug?: string;
}): number {
  // Match each claim's embedding so searchHybrid returns it with high cosine.
  const id = store.recordClaim({
    claim_text: opts.text,
    score: 0.9,
    claim_type: "ATOMIC",
    topic: opts.topic || "test",
    status: opts.status || "supported",
    dossier_slug: opts.dossier_slug ?? "test/dossier",
    embedding: new Float32Array([1, 0, 0, 0]),
  });
  if (opts.superseded_by != null) {
    store.getDb().prepare("UPDATE claims SET superseded_by = ? WHERE id = ?").run(opts.superseded_by, id);
  }
  return id;
}

describe("findCounterEvidence", () => {
  it("returns [] when KB is empty", async () => {
    const out = await findCounterEvidence("X is fast", "X");
    expect(out).toEqual([]);
  });

  it("returns [] when no claim mentions the entity", async () => {
    seedClaim({ text: "Something completely unrelated about Y." });
    mockResponses.push({ contradicts: true, score: 0.9, reason: "would-fire-but-entity-mismatch" });
    const out = await findCounterEvidence("X is fast", "X");
    expect(out).toEqual([]);
    // Mock should NOT have been called — the entity-mention filter
    // short-circuits before verifyContradiction.
    expect(generateObjectMock).not.toHaveBeenCalled();
  });

  it("returns the counter-claim when entity matches and verifier says contradicts", async () => {
    const id = seedClaim({ text: "X is slow.", dossier_slug: "src/x" });
    mockResponses.push({ contradicts: true, score: 0.9, reason: "X is slow contradicts X is fast" });
    const out = await findCounterEvidence("X is fast", "X");
    expect(out.length).toBe(1);
    expect(out[0]!.claim_id).toBe(id);
    expect(out[0]!.contradiction_score).toBe(0.9);
    expect(out[0]!.dossier_slug).toBe("src/x");
  });

  it("skips the excluded claim id", async () => {
    const excluded = seedClaim({ text: "X is slow." });
    mockResponses.push({ contradicts: true, score: 0.9, reason: "would-fire-but-excluded" });
    const out = await findCounterEvidence("X is fast", "X", { excludeClaimId: excluded });
    expect(out).toEqual([]);
    expect(generateObjectMock).not.toHaveBeenCalled();
  });

  it("skips superseded claims", async () => {
    seedClaim({ text: "X is slow.", superseded_by: 999 });
    mockResponses.push({ contradicts: true, score: 0.9, reason: "would-fire-but-superseded" });
    const out = await findCounterEvidence("X is fast", "X");
    expect(out).toEqual([]);
  });

  it("skips non-supported claims", async () => {
    seedClaim({ text: "X is slow.", status: "unsupported" });
    mockResponses.push({ contradicts: true, score: 0.9, reason: "would-fire-but-unsupported" });
    const out = await findCounterEvidence("X is fast", "X");
    expect(out).toEqual([]);
  });

  it("respects maxHits cap", async () => {
    seedClaim({ text: "X is slow part-1." });
    seedClaim({ text: "X is slow part-2." });
    seedClaim({ text: "X is slow part-3." });
    mockResponses.push({ contradicts: true, score: 0.9, reason: "1" });
    mockResponses.push({ contradicts: true, score: 0.9, reason: "2" });
    mockResponses.push({ contradicts: true, score: 0.9, reason: "3" });
    const out = await findCounterEvidence("X is fast", "X", { maxHits: 2 });
    expect(out.length).toBe(2);
  });

  it("filters out below-fire-score verifier verdicts", async () => {
    seedClaim({ text: "X is slow." });
    mockResponses.push({ contradicts: true, score: 0.5, reason: "below-threshold" });
    const out = await findCounterEvidence("X is fast", "X", { fireScore: 0.75 });
    expect(out).toEqual([]);
  });

  it("short-circuits when abortRef.aborted is set before the call", async () => {
    seedClaim({ text: "X is slow." });
    mockResponses.push({ contradicts: true, score: 0.9, reason: "would-fire-but-aborted" });
    const out = await findCounterEvidence("X is fast", "X", { abortRef: { aborted: true } });
    expect(out).toEqual([]);
    expect(generateObjectMock).not.toHaveBeenCalled();
  });
});
