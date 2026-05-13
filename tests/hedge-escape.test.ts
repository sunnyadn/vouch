/** HEDGE escape-hatch regression spec — issue #42.
 *
 *  Deterministic post-filter that escalates HEDGE → ASSERT when the sentence is
 *  a fact-shape claim with a trailing parenthetical/comma caveat.
 */
import { afterEach, beforeAll, describe, expect, it, mock } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

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
// Escalate expectations matrix
// ---------------------------------------------------------------------------

interface HedgeEscapeExpectation {
  proposition: string;
  entity: string;
  draft: string;
  expectEscalate: boolean;
}

const HEDGE_ESCAPE_EXPECTATIONS: HedgeEscapeExpectation[] = [
  // ESCALATE (must fire when ungrounded)
  {
    proposition: "Cole & Hernán (2008) defines the ESS-stable gmin clip method for IPCW weights",
    entity: "Cole & Hernán (2008)",
    draft: "Cole & Hernán (2008) defines the ESS-stable gmin clip method for IPCW weights (unverified, from training memory).",
    expectEscalate: true,
  },
  {
    proposition: "Ishwaran (2008) introduced random survival forests in Annals of Applied Statistics 2(3):841-860",
    entity: "Ishwaran (2008)",
    draft: "Ishwaran (2008) introduced random survival forests in Annals of Applied Statistics 2(3):841-860 (training memory, not verified).",
    expectEscalate: true,
  },
  {
    proposition: "survC1 is Uno's R package",
    entity: "survC1",
    draft: "survC1 is Uno's R package (named, not run here).",
    expectEscalate: true,
  },
  {
    proposition: "the gmin auto picker uses Cole-Hernán ESS stability",
    entity: "Cole-Hernán",
    draft: "the gmin auto picker uses Cole-Hernán ESS stability, unverified from training memory.",
    expectEscalate: true,
  },

  // KEEP HEDGE (must NOT escalate)
  {
    proposition: "Cole & Hernán published in 2008",
    entity: "Cole & Hernán",
    draft: "I think Cole & Hernán published in 2008.",
    expectEscalate: false,
  },
  {
    proposition: "survC1 is Uno's package",
    entity: "survC1",
    draft: "Is survC1 actually Uno's package?",
    expectEscalate: false,
  },
  {
    proposition: "the year was 2008 or 2009",
    entity: "the year",
    draft: "I am not sure whether the year was 2008 or 2009.",
    expectEscalate: false,
  },
  {
    proposition: "the result is TBD",
    entity: "the result",
    draft: "The result is TBD.",
    expectEscalate: false,
  },
  {
    proposition: "this might be Ishwaran's paper",
    entity: "Ishwaran",
    draft: "Probably this is Ishwaran's paper, from training memory.",
    expectEscalate: false,
  },
];

for (const exp of HEDGE_ESCAPE_EXPECTATIONS) {
  it(`escalateHedgeAssertions — ${exp.expectEscalate ? "escalate" : "keep"}: "${exp.proposition}"`, () => {
    const result = gate.escalateHedgeAssertions(
      [{ proposition: exp.proposition, stance: "HEDGE", entity: exp.entity }],
      exp.draft,
    );
    const p = result[0]!;
    if (exp.expectEscalate) {
      expect(p.stance).toBe("ASSERT");
      expect(p.escalatedFromHedge).toBe(true);
    } else {
      expect(p.stance).toBe("HEDGE");
      expect(p.escalatedFromHedge).toBeUndefined();
    }
  });
}

// ---------------------------------------------------------------------------
// runGate-level integration tests
// ---------------------------------------------------------------------------

describe("runGate integration with hedge-escape post-filter", () => {
  it("HEDGE matching a BLOCK pattern → escalated → blocked when KB empty", async () => {
    generateObjectMock.mockImplementationOnce(() =>
      Promise.resolve({
        object: {
          pairs: [
            {
              entity: "Cole & Hernán (2008)",
              stance: "HEDGE",
              proposition: "Cole & Hernán (2008) defines the ESS-stable gmin clip method for IPCW weights",
            },
          ],
        },
      } as any),
    );
    const v = await gate.runGate({
      draft: "Cole & Hernán (2008) defines the ESS-stable gmin clip method for IPCW weights (unverified, from training memory).",
      model: "test",
    });
    expect(v.blocked).toBe(true);
    expect(v.pairs.length).toBe(1);
    expect(v.pairs[0]!.stance).toBe("ASSERT");
    expect(v.pairs[0]!.escalatedFromHedge).toBe(true);
    expect(v.pairs[0]!.grounded).toBe(false);
  });

  it("HEDGE matching a KEEP pattern → stays HEDGE → not blocked", async () => {
    generateObjectMock.mockImplementationOnce(() =>
      Promise.resolve({
        object: {
          pairs: [
            {
              entity: "Ishwaran",
              stance: "HEDGE",
              proposition: "this might be Ishwaran's paper",
            },
          ],
        },
      } as any),
    );
    const v = await gate.runGate({
      draft: "Probably this is Ishwaran's paper, from training memory.",
      model: "test",
    });
    expect(v.blocked).toBe(false);
    expect(v.pairs.length).toBe(1);
    expect(v.pairs[0]!.stance).toBe("HEDGE");
    expect(v.pairs[0]!.escalatedFromHedge).toBeUndefined();
    expect(v.pairs[0]!.grounded).toBe(true);
  });
});
