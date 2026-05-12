/** Hedge-propagation regression spec — issue #17-X6.
 *
 *  Block-level hedge prefixes and trailing caveats must scope ALL factual
 *  sentences in their block, not just the adjacent sentence.
 *
 *  These tests mock generateObject to return the *expected* stance and assert
 *  that the downstream gate behavior is consistent with it.
 *
 *  A live-extractor block (skipped by default) can be run with:
 *    VOUCH_LIVE_EXTRACTOR=1 bun test tests/hedge-scope.test.ts
 */
import { afterEach, beforeAll, describe, expect, it, mock } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { Stance } from "../src/gate.ts";

const tmp = mkdtempSync(join(tmpdir(), "vouch-hedge-test-"));
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
// Expected-behavior matrix (issue #17-X6)
// ---------------------------------------------------------------------------

interface HedgeExpectation {
  n: string;
  draft: string;
  /** For each expected extracted proposition, its stance. */
  expected: { proposition: string; stance: Stance }[];
  category: "block-prefix" | "trailing-caveat" | "per-clause" | "negative-control";
}

export const HEDGE_EXPECTATIONS: HedgeExpectation[] = [
  {
    n: "X1",
    category: "block-prefix",
    draft:
      "From memory, and I haven't verified any of the following:\n\n" +
      "The MiniCheck paper introduced LLM-AggreFact. " +
      "MiniCheck-FT5 has 770M parameters. " +
      "RAGTruth covers GPT-3.5, GPT-4, Llama-2 and Mistral. " +
      "FActScore decomposes generations into atomic facts.",
    expected: [
      { proposition: "The MiniCheck paper introduced LLM-AggreFact.", stance: "HEDGE" },
      { proposition: "MiniCheck-FT5 has 770M parameters.", stance: "HEDGE" },
      { proposition: "RAGTruth covers GPT-3.5, GPT-4, Llama-2 and Mistral.", stance: "HEDGE" },
      { proposition: "FActScore decomposes generations into atomic facts.", stance: "HEDGE" },
    ],
  },
  {
    n: "X2",
    category: "trailing-caveat",
    draft:
      "Vectara maintains a public hallucination leaderboard. " +
      "FaithJudge combines FaithBench and RAGTruth. " +
      "ALCE has three subsets. " +
      "MiniCheck does GPT-4-level fact-checking at lower cost.\n\n" +
      "None of those four are things I've verified this session — they're from training memory.",
    expected: [
      { proposition: "Vectara maintains a public hallucination leaderboard.", stance: "HEDGE" },
      { proposition: "FaithJudge combines FaithBench and RAGTruth.", stance: "HEDGE" },
      { proposition: "ALCE has three subsets.", stance: "HEDGE" },
      { proposition: "MiniCheck does GPT-4-level fact-checking at lower cost.", stance: "HEDGE" },
    ],
  },
  {
    n: "X3",
    category: "negative-control",
    draft:
      "smol-toml is the de-facto standard. (I haven't verified the next claim though.) " +
      "@iarna/toml has been unmaintained since 2022.",
    expected: [
      { proposition: "smol-toml is the de-facto standard.", stance: "ASSERT" },
      { proposition: "@iarna/toml has been unmaintained since 2022.", stance: "HEDGE" },
    ],
  },
  {
    n: "X4",
    category: "per-clause",
    draft:
      "From memory, and unverified — none of the following is checked:\n\n" +
      "MiniCheck does GPT-4-level fact-checking at lower cost.\n\n" +
      "Now, switching topics: RAGTruth covers GPT-3.5, GPT-4, Llama-2 and Mistral.",
    expected: [
      { proposition: "MiniCheck does GPT-4-level fact-checking at lower cost.", stance: "HEDGE" },
      { proposition: "RAGTruth covers GPT-3.5, GPT-4, Llama-2 and Mistral.", stance: "ASSERT" },
    ],
  },
  {
    n: "X5",
    category: "block-prefix",
    draft:
      "I haven't verified any of this, but from memory:\n\n" +
      "FActScore decomposes generations into atomic facts. " +
      "ALCE has three subsets.",
    expected: [
      { proposition: "FActScore decomposes generations into atomic facts.", stance: "HEDGE" },
      { proposition: "ALCE has three subsets.", stance: "HEDGE" },
    ],
  },
  {
    n: "X6",
    category: "trailing-caveat",
    draft:
      "FActScore decomposes generations into atomic facts. " +
      "ALCE has three subsets.\n\n" +
      "I should note I haven't verified either of those claims.",
    expected: [
      { proposition: "FActScore decomposes generations into atomic facts.", stance: "HEDGE" },
      { proposition: "ALCE has three subsets.", stance: "HEDGE" },
    ],
  },
];

// ---------------------------------------------------------------------------
// Plumbing tests: mock the expected stance, assert downstream behavior
// ---------------------------------------------------------------------------

for (const exp of HEDGE_EXPECTATIONS) {
  const label = `${exp.category} ${exp.n}: ${exp.expected.length} proposition(s)`;
  const hasAssert = exp.expected.some((e) => e.stance === "ASSERT");

  it(`${label} → stances correctly scoped`, async () => {
    generateObjectMock.mockImplementationOnce(() =>
      Promise.resolve({
        object: {
          pairs: exp.expected.map((e) => ({
            entity: "TestEntity",
            stance: e.stance,
            proposition: e.proposition,
          })),
        },
      } as any),
    );
    const v = await gate.runGate({ draft: exp.draft, model: "test" });
    expect(v.pairs.length).toBe(exp.expected.length);
    for (let i = 0; i < exp.expected.length; i++) {
      expect(v.pairs[i]!.stance).toBe(exp.expected[i]!.stance);
      if (exp.expected[i]!.stance === "HEDGE") {
        expect(v.pairs[i]!.grounded).toBe(true);
        expect(v.pairs[i]!.reason).toContain("no fact to ground");
      } else if (exp.expected[i]!.stance === "ASSERT") {
        // Empty KB → ungrounded, which is expected; we just verify stance.
        expect(v.pairs[i]!.grounded).toBe(false);
      }
    }
    // Blocked only if there's an ungrounded ASSERT proposition.
    expect(v.blocked).toBe(hasAssert);
    // Only extractor call when all propositions are non-ASSERT.
    if (!hasAssert) {
      expect(generateObjectMock).toHaveBeenCalledTimes(1);
    }
  });
}

// ---------------------------------------------------------------------------
// Live extractor tests (require Vertex credentials; skipped by default)
// ---------------------------------------------------------------------------

describe.skipIf(!process.env.VOUCH_LIVE_EXTRACTOR)("live extractor hedge scoping", () => {
  beforeAll(() => {
    mock.restore();
  });

  for (const exp of HEDGE_EXPECTATIONS) {
    it(`${exp.n} (${exp.category}) should scope hedges correctly`, async () => {
      const pairs = await gate.extractPairs(exp.draft, gate.DEFAULT_GATE_MODEL);
      expect(pairs).not.toBeNull();
      for (const expected of exp.expected) {
        const match = pairs!.find((p) =>
          expected.proposition.toLowerCase().includes(p.proposition.toLowerCase()) ||
          p.proposition.toLowerCase().includes(expected.proposition.toLowerCase()),
        );
        if (!match) {
          throw new Error(
            `Expected a pair for "${expected.proposition}", got: ${JSON.stringify(pairs)}`,
          );
        }
        expect(match.stance).toBe(expected.stance);
      }
    });
  }
});
