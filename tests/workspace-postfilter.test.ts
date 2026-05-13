/** Deterministic workspace-meta post-filter regression spec — issue #40.
 *
 *  The post-filter is deterministic (no LLM), so it's directly testable.
 *  It re-classifies obvious workspace-meta ASSERTs → WORKSPACE before grounding.
 */
import { afterEach, beforeAll, describe, expect, it, mock } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const tmp = mkdtempSync(join(tmpdir(), "vouch-wspf-test-"));
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
// Reclassify expectations matrix
// ---------------------------------------------------------------------------

interface ReclassifyExpectation {
  proposition: string;
  entity: string;
  draft?: string;
  expectDowngrade: boolean;
  rule?: 1 | 2 | 3;
}

const RECLASSIFY_EXPECTATIONS: ReclassifyExpectation[] = [
  // downgrade (rule 1) — workspace-project self-reference / publication-state / asset locator
  {
    // github.com/sunnyadn/ locator
    proposition: "github.com/sunnyadn/crforest auto-redirects to github.com/sunnyadn/comprisk",
    entity: "crforest",
    expectDowngrade: true,
    rule: 1,
  },
  {
    // publication state of user's own package
    proposition: "comprisk 0.3.0 does not exist on PyPI",
    entity: "comprisk",
    expectDowngrade: true,
    rule: 1,
  },
  {
    // self-referential ownership: "my vouch"
    proposition: "my vouch CLI has a claim-batch subcommand",
    entity: "vouch",
    expectDowngrade: true,
    rule: 1,
  },
  {
    // self-referential ownership: "X is my project"
    proposition: "vouch is my project for fact verification",
    entity: "vouch",
    expectDowngrade: true,
    rule: 1,
  },
  // NO downgrade (rule 1 narrowed 2026-05-13): bare assertions about a
  // workspace-project's behavior/API/features are NOT self-referential and
  // must still be gate-checked against the KB. If KB lacks coverage, fire
  // (correct — the agent should look at the repo before claiming).
  {
    proposition: "the vouch CLI has a claim-batch subcommand",
    entity: "vouch",
    expectDowngrade: false, // bare assertion about own project — must be checked
  },
  {
    proposition: "crforest 0.4.0 supports clustered standard errors",
    entity: "crforest",
    expectDowngrade: false, // bare API claim about own project — must be checked
  },
  {
    proposition: "comprisk uses lifelines 0.27 as a dependency",
    entity: "comprisk",
    expectDowngrade: false, // bare claim about own project's deps — must be checked
  },

  // downgrade (rule 2) — agent-machinery phrasings
  {
    proposition: "The default Pro verifier has a latency of approximately 3.7 seconds per call",
    entity: "the Pro verifier",
    expectDowngrade: true,
    rule: 2,
  },
  {
    proposition: "vertex_ai/gemini-3.1-pro-preview is the verifier model in this setup",
    entity: "vertex_ai/gemini-3.1-pro-preview",
    expectDowngrade: true,
    rule: 2,
  },
  {
    proposition: "the v3 benchmark harness uses Gemini Pro 3.1 as the generator",
    entity: "the v3 harness",
    expectDowngrade: true,
    rule: 2,
  },
  {
    proposition: "corpus draft cc-10 yielded 11 propositions",
    entity: "cc-10",
    expectDowngrade: true,
    rule: 2,
  },

  // downgrade (rule 3) — mention-not-use: an ≥8-token slice of the proposition
  // appears inside a real double-quote / backtick / blockquote region of the draft
  {
    proposition: "smol-toml is the de-facto standard now in the JS ecosystem",
    entity: "smol-toml",
    draft:
      'I should note the gate fired on "smol-toml is the de-facto standard now in the JS ecosystem" — that is a quoted claim being discussed, not an assertion I am making.',
    expectDowngrade: true,
    rule: 3,
  },
  {
    proposition: "FActScore decomposes a generation into atomic facts and scores the supported fraction",
    entity: "FActScore",
    draft:
      "> FActScore decomposes a generation into atomic facts and scores the supported fraction\n\n(quoting the FActScore README above; I'm not asserting this myself)",
    expectDowngrade: true,
    rule: 3,
  },

  // Note: Rule 3b (entity-in-quoted-region with invented predicate) was
  // shipped in d85eae7 and reverted after the #35 freeze showed recall
  // 100% → 69% — the contiguous-match check on the predicate body was too
  // strict against extractor-canonicalized propositions. The four
  // "expectDowngrade: true (rule 3 entity-in-quoted-region)" cases from that
  // PR are removed here; they will return with a jaccard-based redesign.

  // NO downgrade (controls — must stay ASSERT)
  {
    // contraction apostrophes ("I'd", "it's") must NOT be treated as quote
    // delimiters — this draft has no real quoted region
    proposition: "smol-toml is the de-facto standard now",
    entity: "smol-toml",
    draft:
      "I'd reach for smol-toml — it's the de-facto standard now, it's what most new projects pull in, and it doesn't pull a parser-generator into your bundle.",
    expectDowngrade: false,
  },
  {
    proposition: "vLLM added support for the Llama-3.1 405B model in version 0.5.4",
    entity: "vLLM",
    expectDowngrade: false,
  },
  {
    proposition: "bitsandbytes supports int8 quantization",
    entity: "bitsandbytes",
    expectDowngrade: false,
  },
  {
    proposition: "ChatGPT bios scored approximately 58% on FActScore",
    entity: "ChatGPT",
    expectDowngrade: false,
  },
  {
    proposition: "lodash-es 4.17.23 has high and medium severity advisories",
    entity: "lodash-es",
    expectDowngrade: false,
  },
  {
    proposition: "GPT-4 has approximately 1.7 trillion parameters",
    entity: "GPT-4",
    expectDowngrade: false,
  },
  {
    proposition: "Llama-3.1 405B was released in 2024",
    entity: "Llama-3.1 405B",
    expectDowngrade: false,
  },
  {
    proposition: "comprisk uses lifelines 0.27 as a dependency",
    entity: "lifelines",
    expectDowngrade: false,
  },

  // NO downgrade (rule 1 narrowed): bare `entity=vouch` no longer wins.
  // Even though the entity is a workspace project and the draft mentions it
  // in inline-code, the proposition "vouch is a verifier" is a fact-shape
  // claim about its identity — should be ground-checked against the KB
  // (which has many vouch claims). If KB grounds, pass; if not, fire.
  {
    proposition: "vouch is a verifier",
    entity: "vouch",
    draft: "`vouch` was mentioned in the prior turn.",
    expectDowngrade: false,
  },
];

for (const exp of RECLASSIFY_EXPECTATIONS) {
  const label = `${exp.expectDowngrade ? "downgrade" : "keep"}${exp.rule ? ` (rule ${exp.rule})` : ""}: "${exp.proposition}"`;
  it(`reclassifyWorkspaceMeta — ${label}`, () => {
    const draft = exp.draft ?? "";
    const result = gate.reclassifyWorkspaceMeta(
      [{ proposition: exp.proposition, stance: "ASSERT", entity: exp.entity }],
      draft,
    );
    const p = result[0]!;
    if (exp.expectDowngrade) {
      expect(p.stance).toBe("WORKSPACE");
      expect(p.reclassifiedRule).toBe(exp.rule);
    } else {
      expect(p.stance).toBe("ASSERT");
      expect(p.reclassifiedRule).toBeUndefined();
    }
  });
}

// ---------------------------------------------------------------------------
// runGate-level integration tests
// ---------------------------------------------------------------------------

describe("runGate integration with post-filter", () => {
  it("ASSERT that matches rule 2 → downgraded → not blocked", async () => {
    generateObjectMock.mockImplementationOnce(() =>
      Promise.resolve({
        object: {
          pairs: [
            {
              entity: "vertex_ai/gemini-3.1-pro-preview",
              stance: "ASSERT",
              proposition: "vertex_ai/gemini-3.1-pro-preview is the verifier model in this setup",
            },
          ],
        },
      } as any),
    );
    const v = await gate.runGate({
      draft: "vertex_ai/gemini-3.1-pro-preview is the verifier model in this setup",
      model: "test",
    });
    expect(v.blocked).toBe(false);
    expect(v.pairs.length).toBe(1);
    expect(v.pairs[0]!.stance).toBe("WORKSPACE");
    expect(v.pairs[0]!.grounded).toBe(true);
    expect(v.pairs[0]!.reason).toContain("reclassified WORKSPACE by deterministic post-filter (rule 2)");
    expect(generateObjectMock).toHaveBeenCalledTimes(1);
  });

  it("ASSERT that does NOT match any rule → reaches grounding (blocked when KB empty)", async () => {
    generateObjectMock.mockImplementationOnce(() =>
      Promise.resolve({
        object: {
          pairs: [
            {
              entity: "FEVER",
              stance: "ASSERT",
              proposition: "FEVER has 185,445 claims.",
            },
          ],
        },
      } as any),
    );
    const v = await gate.runGate({
      draft: "FEVER has 185,445 claims.",
      model: "test",
    });
    expect(v.blocked).toBe(true);
    expect(v.pairs.length).toBe(1);
    expect(v.pairs[0]!.stance).toBe("ASSERT");
    expect(v.pairs[0]!.grounded).toBe(false);
  });

  // Rule 3b runGate integration test removed alongside its post-filter (see
  // gate.ts note about d85eae7 recall regression).
});
