/** Workspace-classify regression spec — issue #38.
 *
 *  Broadens WORKSPACE to stop over-firing on non-third-party claims,
 *  and walks back #17-F over-correction on trivial category glosses.
 *
 *  The extractor is an LLM; these tests mock generateObject to return the
 *  *expected* stance and assert that the downstream gate behavior is consistent
 *  with it. This documents the contract + protects the plumbing.
 *
 *  A live-extractor block (skipped by default) can be run with:
 *    VOUCH_LIVE_EXTRACTOR=1 bun test tests/workspace-classify.test.ts
 */
import { afterEach, beforeAll, describe, expect, it, mock } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { Stance } from "../src/gate.ts";

const tmp = mkdtempSync(join(tmpdir(), "vouch-ws-test-"));
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
// Expected-behavior matrix (issue #38)
// ---------------------------------------------------------------------------

interface StanceExpectation {
  /** 1-based case number */
  n: number;
  /** The proposition text fed to the extractor. */
  text: string;
  /** Expected stance per the tightened taxonomy. WORKSPACE means "no triple". */
  expectedStance: Stance | "WORKSPACE";
  /** Category for grouping in test output. */
  category:
    | "workspace"
    | "refer-gloss"
    | "stay-assert"
    | "stay-opinion"
    | "stay-hedge";
}

export const STANCE_EXPECTATIONS: StanceExpectation[] = [
  // Should be WORKSPACE (currently mis-classified as ASSERT)
  { n: 1, text: "Gemini frequently emits 2–4 search_kb calls in a single turn with rephrased queries.", expectedStance: "WORKSPACE", category: "workspace" },
  { n: 2, text: "The Pro verifier has a latency of approximately 3.7 seconds per call.", expectedStance: "WORKSPACE", category: "workspace" },
  { n: 3, text: "bun test reported 168 pass and 0 fail in this environment.", expectedStance: "WORKSPACE", category: "workspace" },
  { n: 4, text: "github.com/sunnyadn/crforest auto-redirects to github.com/sunnyadn/comprisk.", expectedStance: "WORKSPACE", category: "workspace" },
  { n: 5, text: "The v3 benchmark harness uses Gemini Pro 3.1 as the generator.", expectedStance: "WORKSPACE", category: "workspace" },
  { n: 6, text: "vertex_ai/gemini-3.1-pro-preview is the verifier model in this setup.", expectedStance: "WORKSPACE", category: "workspace" },
  { n: 7, text: "Corpus draft cc-10 yielded 11 propositions.", expectedStance: "WORKSPACE", category: "workspace" },
  { n: 8, text: "Test fixture X5 covers the hedge-then-multi-claims case.", expectedStance: "WORKSPACE", category: "workspace" },

  // Should be REFER (walk-back of #17-F over-correction on trivial category glosses)
  { n: 9, text: "GitHub is a platform that hosts repositories, issues, and pull requests.", expectedStance: "REFER", category: "refer-gloss" },
  { n: 10, text: "lifelines is a Python library for survival analysis.", expectedStance: "REFER", category: "refer-gloss" },

  // Must stay ASSERT (real third-party checkable claims — controls)
  { n: 11, text: "smol-toml is the de-facto standard now.", expectedStance: "ASSERT", category: "stay-assert" },
  { n: 12, text: "vLLM added support for the Llama-3.1 405B model in version 0.5.4.", expectedStance: "ASSERT", category: "stay-assert" },
  { n: 13, text: "bitsandbytes supports int8 quantization.", expectedStance: "ASSERT", category: "stay-assert" },
  { n: 14, text: "ChatGPT bios scored ~58% on FActScore.", expectedStance: "ASSERT", category: "stay-assert" },
  { n: 15, text: "lodash-es 4.17.23 has high and medium severity advisories.", expectedStance: "ASSERT", category: "stay-assert" },

  // Must stay OPINION / HEDGE (controls — must NOT flip)
  { n: 16, text: "smol-toml is the elegant one.", expectedStance: "OPINION", category: "stay-opinion" },
  { n: 17, text: "From memory, and unverified — LLM-AggreFact aggregates 11 datasets.", expectedStance: "HEDGE", category: "stay-hedge" },
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
      expect(generateObjectMock).toHaveBeenCalledTimes(1);
    });
  } else if (exp.expectedStance === "WORKSPACE") {
    it(`${label} → extractor returns no pairs → not blocked`, async () => {
      generateObjectMock.mockImplementationOnce(() =>
        Promise.resolve({ object: { pairs: [] } } as any),
      );
      const v = await gate.runGate({ draft: exp.text, model: "test" });
      expect(v.blocked).toBe(false);
      expect(v.pairs.length).toBe(0);
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
      expect(v.pairs[0]!.stance).toBe(exp.expectedStance as Stance);
      expect(v.pairs[0]!.grounded).toBe(true);
      expect(v.pairs[0]!.reason).toContain("no fact to ground");
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
      if (exp.expectedStance === "WORKSPACE") {
        // WORKSPACE means no matching pair should be returned.
        const match = pairs?.find((p) =>
          exp.text.toLowerCase().includes(p.proposition.toLowerCase()) ||
          p.proposition.toLowerCase().includes(exp.text.toLowerCase()),
        );
        if (match) {
          throw new Error(`Expected no pair for WORKSPACE case, got: ${JSON.stringify(pairs)}`);
        }
        return;
      }
      expect(pairs).not.toBeNull();
      const match = pairs!.find((p) =>
        exp.text.toLowerCase().includes(p.proposition.toLowerCase()) ||
        p.proposition.toLowerCase().includes(exp.text.toLowerCase()),
      );
      if (!match) {
        // For ASSERT controls, the extractor MUST return a pair.
        if (exp.category === "stay-assert") {
          throw new Error(`Expected a pair for ASSERT control case, got: ${JSON.stringify(pairs)}`);
        }
        return;
      }
      expect(match.stance).toBe(exp.expectedStance as Stance);
    });
  }
});
