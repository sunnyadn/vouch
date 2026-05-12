/** Stance-tighten regression spec — issue #17 items E + F.
 *
 *  The extractor is an LLM; these tests mock generateObject to return the
 *  *expected* stance and assert that the downstream gate behavior is consistent
 *  with it. This documents the contract + protects the plumbing.
 *
 *  A live-extractor block (skipped by default) can be run with:
 *    VOUCH_LIVE_EXTRACTOR=1 bun test tests/stance-tighten.test.ts
 */
import { afterEach, beforeAll, describe, expect, it, mock } from "bun:test";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { Stance } from "../src/gate.ts";

const tmp = mkdtempSync(join(tmpdir(), "vouch-stance-test-"));
process.env.VOUCH_DB_PATH = join(tmp, "test.db");

const generateObjectMock = mock(() =>
  Promise.resolve({ object: { pairs: [] } } as any),
);

mock.module("ai", () => ({
  generateObject: generateObjectMock,
  embed: () => Promise.resolve({ embeddings: [] }),
}));

mock.module("../src/providers.ts", () => ({
  getLanguageModel: () => ({ id: "test-model" }) as any,
  getEmbeddingModel: () => ({ id: "test-embedder" }) as any,
}));

const queryVec = new Float32Array([1, 0, 0, 0]);
mock.module("../src/embedder.ts", () => ({
  embedOne: () => Promise.resolve(queryVec),
  embedBatch: () => Promise.resolve([queryVec]),
}));

const store = await import("../src/store.ts");
const gate = await import("../src/gate.ts");

afterEach(() => {
  const db = store.getDb();
  db.exec("DELETE FROM claim_dependencies; DELETE FROM claims; DELETE FROM dossiers;");
  generateObjectMock.mockReset();
  generateObjectMock.mockImplementation(() =>
    Promise.resolve({ object: { pairs: [] } } as any),
  );
});

beforeAll(() => {
  process.on("exit", () => {
    try {
      rmSync(tmp, { recursive: true, force: true });
    } catch {}
  });
});

// ---------------------------------------------------------------------------
// Expected-behavior matrix (issue #17 E + F)
// ---------------------------------------------------------------------------

interface StanceExpectation {
  /** 1-based case number */
  n: number;
  /** The proposition text fed to the extractor. */
  text: string;
  /** Expected stance per the tightened taxonomy. */
  expectedStance: Stance;
  /** Category for grouping in test output. */
  category:
    | "regression-assert" // was OPINION/REFER, should be ASSERT now
    | "positive-control" // was ASSERT, must stay ASSERT
    | "genuine-opinion" // must stay OPINION
    | "pure-mention"; // must stay REFER
}

export const STANCE_EXPECTATIONS: StanceExpectation[] = [
  // Should be ASSERT (currently mis-classified as OPINION or REFER)
  { n: 1, text: "smol-toml is the de-facto standard now.", expectedStance: "ASSERT", category: "regression-assert" },
  { n: 2, text: "@iarna/toml was the canonical implementation for years.", expectedStance: "ASSERT", category: "regression-assert" },
  { n: 3, text: "litellm is the most-downloaded LLM proxy by a wide margin.", expectedStance: "ASSERT", category: "regression-assert" },
  { n: 4, text: "bitsandbytes supports int8 quantization.", expectedStance: "ASSERT", category: "regression-assert" },
  { n: 5, text: "FActScore decomposes a generation into atomic facts and scores the fraction supported.", expectedStance: "ASSERT", category: "regression-assert" },
  { n: 6, text: "MiniCheck-T5 is an 0.8B parameter model released under the MIT license.", expectedStance: "ASSERT", category: "regression-assert" },
  { n: 7, text: "SQLite is an embedded relational database engine written in C.", expectedStance: "ASSERT", category: "regression-assert" },
  { n: 8, text: "HuggingFace provides inference endpoints for a fee.", expectedStance: "ASSERT", category: "regression-assert" },

  // Should stay ASSERT (positive controls — must not regress)
  { n: 9, text: "Library X is the most actively maintained option in its ecosystem.", expectedStance: "ASSERT", category: "positive-control" },
  { n: 10, text: "vLLM added support for the Llama-3.1 405B model in version 0.5.4.", expectedStance: "ASSERT", category: "positive-control" },

  // Should stay OPINION (genuinely subjective)
  { n: 11, text: "smol-toml is the elegant one.", expectedStance: "OPINION", category: "genuine-opinion" },
  { n: 12, text: "X is the best choice for our specific case.", expectedStance: "OPINION", category: "genuine-opinion" },
  { n: 13, text: "FActScore is a cleaner fit for our decomposition step than for the end-to-end pitch, in my view.", expectedStance: "OPINION", category: "genuine-opinion" },

  // Should stay REFER (pure mention / bare naming)
  { n: 14, text: "For the floor check I'd start with LLM-AggreFact.", expectedStance: "REFER", category: "pure-mention" },
  { n: 15, text: "FEVER, ALCE, FActScore — there are several benchmarks in this space.", expectedStance: "REFER", category: "pure-mention" },
];

// ---------------------------------------------------------------------------
// Plumbing tests: mock the expected stance, assert downstream behavior
// ---------------------------------------------------------------------------

for (const exp of STANCE_EXPECTATIONS) {
  const label = `${exp.category} #${exp.n}: ${exp.expectedStance} — "${exp.text}"`;

  if (exp.expectedStance === "ASSERT") {
    it(`${label} → reaches grounding (blocked when KB empty)`, async () => {
      generateObjectMock.mockImplementationOnce(() =>
        Promise.resolve({
          object: {
            pairs: [{ entity: "TestEntity", stance: "ASSERT", proposition: exp.text }],
          },
        } as any),
      );
      const v = await gate.runGate({ draft: exp.text, model: "test" });
      expect(v.blocked).toBe(true);
      expect(v.pairs.length).toBe(1);
      expect(v.pairs[0]!.stance).toBe("ASSERT");
      expect(v.pairs[0]!.grounded).toBe(false);
      // ASSERT goes through the embed+search path → generateObject called once (extractor)
      expect(generateObjectMock).toHaveBeenCalledTimes(1);
    });
  } else {
    it(`${label} → passes through, no KB lookup`, async () => {
      generateObjectMock.mockImplementationOnce(() =>
        Promise.resolve({
          object: {
            pairs: [{ entity: "TestEntity", stance: exp.expectedStance, proposition: exp.text }],
          },
        } as any),
      );
      const v = await gate.runGate({ draft: exp.text, model: "test" });
      expect(v.blocked).toBe(false);
      expect(v.pairs.length).toBe(1);
      expect(v.pairs[0]!.stance).toBe(exp.expectedStance);
      expect(v.pairs[0]!.grounded).toBe(true);
      expect(v.pairs[0]!.reason).toContain("no fact to ground");
      // Non-ASSERT short-circuits → only extractor call
      expect(generateObjectMock).toHaveBeenCalledTimes(1);
    });
  }
}

// ---------------------------------------------------------------------------
// Live extractor tests (require Vertex credentials; skipped by default)
// ---------------------------------------------------------------------------

describe.skipIf(!process.env.VOUCH_LIVE_EXTRACTOR)("live extractor stance classification", () => {
  beforeAll(() => {
    // Bun's mock.module replaces the module globally; mock.restore() clears
    // all mocks so the real ai.generateObject is used for live tests.
    mock.restore();
  });

  for (const exp of STANCE_EXPECTATIONS) {
    it(`#${exp.n} should classify as ${exp.expectedStance}: "${exp.text}"`, async () => {
      const pairs = await gate.extractPairs(exp.text, gate.DEFAULT_GATE_MODEL);
      expect(pairs).not.toBeNull();
      const match = pairs!.find((p) =>
        exp.text.toLowerCase().includes(p.proposition.toLowerCase()) ||
        p.proposition.toLowerCase().includes(exp.text.toLowerCase()),
      );
      if (!match) {
        // If the extractor returned no matching pair, that's a failure for
        // regression cases (they *should* be extracted) but acceptable for
        // genuine-opinion / pure-mention if the extractor chose to skip.
        if (exp.category === "regression-assert" || exp.category === "positive-control") {
          throw new Error(`Expected a pair for regression/positive case, got: ${JSON.stringify(pairs)}`);
        }
        return;
      }
      expect(match.stance).toBe(exp.expectedStance);
    });
  }
});
