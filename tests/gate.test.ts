/** Gate logic tests — mocks LLM + embedder, uses real SQLite store. */
import { afterEach, beforeAll, describe, expect, it, mock } from "bun:test";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const tmp = mkdtempSync(join(tmpdir(), "vouch-gate-test-"));
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
  generateObjectMock.mockClear();
  delete process.env.VOUCH_GATE_BYPASS;
});

beforeAll(() => {
  process.on("exit", () => {
    try {
      rmSync(tmp, { recursive: true, force: true });
    } catch {}
  });
});

// ---------------------------------------------------------------------------
// runGate — the 3 prototype fixtures
// ---------------------------------------------------------------------------

describe("runGate fixtures", () => {
  it("WORKSPACE: extractor returns no pairs → not blocked", async () => {
    generateObjectMock.mockImplementationOnce(() =>
      Promise.resolve({ object: { pairs: [] } } as any),
    );
    const v = await gate.runGate({
      draft: "Looking at your vault now. Will start with the strategy doc.",
      model: "vertex_ai/test",
    });
    expect(v.blocked).toBe(false);
    expect(v.pairs.length).toBe(0);
  });

  it("PASS: hedged draft → extractor skips → not blocked", async () => {
    // Real classifier would skip hedged claims; mock simulates that.
    generateObjectMock.mockImplementationOnce(() =>
      Promise.resolve({ object: { pairs: [] } } as any),
    );
    const v = await gate.runGate({
      draft:
        "FEVER has roughly 185k claims (unverified, from training memory — let me verify).",
      model: "vertex_ai/test",
    });
    expect(v.blocked).toBe(false);
    expect(v.pairs.length).toBe(0);
  });

  it("BLOCK: named-entity assertion, KB has nothing → blocked", async () => {
    generateObjectMock.mockImplementationOnce(() =>
      Promise.resolve({
        object: {
          pairs: [
            { entity: "FEVER", stance: "ASSERT", proposition: "FEVER has 185,445 claims." },
            { entity: "MiniCheck-7B", stance: "ASSERT", proposition: "MiniCheck-7B beats GPT-4o on FACTBENCH." },
          ],
        },
      } as any),
    );
    const v = await gate.runGate({
      draft: "FEVER has 185,445 claims. MiniCheck-7B beats GPT-4o on FACTBENCH.",
      model: "vertex_ai/test",
    });
    expect(v.blocked).toBe(true);
    expect(v.pairs.length).toBe(2);
    expect(v.pairs.every((p) => !p.grounded)).toBe(true);
    expect(v.pairs.every((p) => p.stance === "ASSERT")).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// Grounding via KB
// ---------------------------------------------------------------------------

describe("runGate grounding", () => {
  it("supported KB claim entails assertion → grounded → not blocked", async () => {
    const slug = store.writeDossier({
      source_url: "https://aclanthology.org/N18-1074/",
      source_type: "test",
      verbatim_content: "FEVER consists of 185,445 claims classified as Supported, Refuted or NotEnoughInfo.",
    });
    const cid = store.recordClaim({
      dossier_slug: slug,
      claim_text: "FEVER consists of 185,445 claims.",
      score: 0.95,
      status: "supported",
      claim_type: "ATOMIC",
      embedding: queryVec,
      source_offset_start: 0,
      source_offset_end: 90,
    });

    generateObjectMock.mockImplementationOnce(() =>
      Promise.resolve({
        object: { pairs: [{ entity: "FEVER", stance: "ASSERT", proposition: "FEVER has 185,445 claims." }] },
      } as any),
    );
    generateObjectMock.mockImplementationOnce(() =>
      Promise.resolve({
        object: { supported: true, score: 0.92, reason: "matches stored claim" },
      } as any),
    );

    const v = await gate.runGate({
      draft: "FEVER has 185,445 claims.",
      model: "vertex_ai/test",
    });
    expect(v.blocked).toBe(false);
    expect(v.pairs[0]!.grounded).toBe(true);
    expect(v.pairs[0]!.matched_claim_id).toBe(cid);
  });

  it("KB hit unsupported → not grounded → blocked", async () => {
    const slug = store.writeDossier({
      source_url: "https://example.com",
      source_type: "test",
      verbatim_content: "FEVER is a fact-verification dataset.",
    });
    store.recordClaim({
      dossier_slug: slug,
      claim_text: "FEVER has 999,999 claims.",
      score: 0.1,
      status: "unsupported",
      claim_type: "ATOMIC",
      embedding: queryVec,
    });

    generateObjectMock.mockImplementationOnce(() =>
      Promise.resolve({
        object: { pairs: [{ entity: "FEVER", stance: "ASSERT", proposition: "FEVER has 185,445 claims." }] },
      } as any),
    );

    const v = await gate.runGate({
      draft: "FEVER has 185,445 claims.",
      model: "vertex_ai/test",
    });
    expect(v.blocked).toBe(true);
    expect(v.pairs[0]!.grounded).toBe(false);
  });

  it("superseded supported claim is skipped", async () => {
    const slug = store.writeDossier({
      source_url: "https://example.com",
      source_type: "test",
      verbatim_content: "FEVER consists of 185,445 claims.",
    });
    const oldId = store.recordClaim({
      dossier_slug: slug,
      claim_text: "FEVER consists of 185,445 claims.",
      score: 0.95,
      status: "supported",
      claim_type: "ATOMIC",
      embedding: queryVec,
    });
    const newId = store.recordClaim({
      dossier_slug: slug,
      claim_text: "FEVER consists of 185,445 claims (v2).",
      score: 0.95,
      status: "supported",
      claim_type: "ATOMIC",
      embedding: new Float32Array([0, 1, 0, 0]),
    });
    store.supersedeClaim(oldId, newId, "updated");

    generateObjectMock.mockImplementationOnce(() =>
      Promise.resolve({
        object: { pairs: [{ entity: "FEVER", stance: "ASSERT", proposition: "FEVER has 185,445 claims." }] },
      } as any),
    );
    // Even if the verifier WOULD say supported, the gate should never call it
    // for a superseded candidate. So we don't queue a verifier mock here —
    // an extra generateObject call would surface as an undefined object.

    const v = await gate.runGate({
      draft: "FEVER has 185,445 claims.",
      model: "vertex_ai/test",
    });
    expect(v.blocked).toBe(true);
    expect(v.pairs[0]!.grounded).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// Stance dispatch — issue #1: gate fires only on ASSERT
// ---------------------------------------------------------------------------

describe("runGate stance dispatch", () => {
  it("HEDGE proposition → not blocked, no KB lookup", async () => {
    generateObjectMock.mockImplementationOnce(() =>
      Promise.resolve({
        object: {
          pairs: [
            { entity: "FEVER", stance: "HEDGE", proposition: "FEVER has 185,445 claims." },
          ],
        },
      } as any),
    );
    const v = await gate.runGate({
      draft: "FEVER has 185,445 claims (unverified, from training memory).",
      model: "vertex_ai/test",
    });
    expect(v.blocked).toBe(false);
    expect(v.pairs[0]!.grounded).toBe(true);
    expect(v.pairs[0]!.stance).toBe("HEDGE");
    // Only the extractor call — no embedder/verifier dispatch for non-ASSERT
    expect(generateObjectMock).toHaveBeenCalledTimes(1);
  });

  it("SPECULATE / hypothetical → not blocked", async () => {
    generateObjectMock.mockImplementationOnce(() =>
      Promise.resolve({
        object: {
          pairs: [
            { entity: "FEVER", stance: "SPECULATE", proposition: "FEVER might have around 200k claims." },
          ],
        },
      } as any),
    );
    const v = await gate.runGate({
      draft: "If FEVER is comparable to similar datasets, it might have around 200k claims.",
      model: "vertex_ai/test",
    });
    expect(v.blocked).toBe(false);
    expect(v.pairs[0]!.grounded).toBe(true);
    expect(v.pairs[0]!.stance).toBe("SPECULATE");
  });

  it("RETRACT re-mentions entity → not blocked", async () => {
    generateObjectMock.mockImplementationOnce(() =>
      Promise.resolve({
        object: {
          pairs: [
            { entity: "FEVER", stance: "RETRACT", proposition: "Retracting earlier claim about FEVER's size." },
          ],
        },
      } as any),
    );
    const v = await gate.runGate({
      draft: "Retracting my earlier claim about FEVER's size — I shouldn't have stated it.",
      model: "vertex_ai/test",
    });
    expect(v.blocked).toBe(false);
    expect(v.pairs[0]!.grounded).toBe(true);
    expect(v.pairs[0]!.stance).toBe("RETRACT");
  });

  it("COMPARE: entity is comparison topic → not blocked", async () => {
    generateObjectMock.mockImplementationOnce(() =>
      Promise.resolve({
        object: {
          pairs: [
            { entity: "FEVER", stance: "COMPARE", proposition: "FEVER is one of the datasets being evaluated." },
            { entity: "HaluBench", stance: "COMPARE", proposition: "HaluBench is one of the datasets being evaluated." },
          ],
        },
      } as any),
    );
    const v = await gate.runGate({
      draft: "We evaluate vouch against FEVER vs HaluBench.",
      model: "vertex_ai/test",
    });
    expect(v.blocked).toBe(false);
    expect(v.pairs.every((p) => p.grounded)).toBe(true);
  });

  it("NEGATE: explicit denial → not blocked", async () => {
    generateObjectMock.mockImplementationOnce(() =>
      Promise.resolve({
        object: {
          pairs: [
            { entity: "FEVER", stance: "NEGATE", proposition: "FEVER does not have 999,999 claims." },
          ],
        },
      } as any),
    );
    const v = await gate.runGate({
      draft: "FEVER does not have 999,999 claims.",
      model: "vertex_ai/test",
    });
    expect(v.blocked).toBe(false);
    expect(v.pairs[0]!.grounded).toBe(true);
  });

  it("META: reflective reference to prior claim → not blocked", async () => {
    generateObjectMock.mockImplementationOnce(() =>
      Promise.resolve({
        object: {
          pairs: [
            { entity: "FEVER", stance: "META", proposition: "Earlier I said FEVER has 185,445 claims." },
          ],
        },
      } as any),
    );
    const v = await gate.runGate({
      draft: "Earlier I said FEVER has 185,445 claims (claim_id: 42).",
      model: "vertex_ai/test",
    });
    expect(v.blocked).toBe(false);
    expect(v.pairs[0]!.grounded).toBe(true);
  });

  it("REFER: name as label only → not blocked", async () => {
    generateObjectMock.mockImplementationOnce(() =>
      Promise.resolve({
        object: {
          pairs: [
            { entity: "FEVER", stance: "REFER", proposition: "See also: FEVER." },
          ],
        },
      } as any),
    );
    const v = await gate.runGate({
      draft: "See also: FEVER.",
      model: "vertex_ai/test",
    });
    expect(v.blocked).toBe(false);
    expect(v.pairs[0]!.grounded).toBe(true);
  });

  it("acceptance: transcript with only HEDGE + RETRACT → gate fires zero times", async () => {
    // Hedge spiral / retraction transcript — extractor labels stance correctly,
    // gate must not block. Issue #1 acceptance criterion.
    generateObjectMock.mockImplementationOnce(() =>
      Promise.resolve({
        object: {
          pairs: [
            { entity: "FEVER", stance: "HEDGE", proposition: "FEVER has roughly 185k claims." },
            { entity: "FEVER", stance: "RETRACT", proposition: "Retracting earlier claim about FEVER." },
            { entity: "MiniCheck-7B", stance: "HEDGE", proposition: "MiniCheck-7B is around 7B params." },
          ],
        },
      } as any),
    );
    const v = await gate.runGate({
      draft:
        "FEVER has roughly 185k claims (unverified). Retracting earlier claim about FEVER. " +
        "MiniCheck-7B is around 7B params (from training memory).",
      model: "vertex_ai/test",
    });
    expect(v.blocked).toBe(false);
    expect(v.pairs.length).toBe(3);
    expect(v.pairs.every((p) => p.grounded)).toBe(true);
    // Stance values preserved in audit output
    expect(v.pairs.map((p) => p.stance).sort()).toEqual(["HEDGE", "HEDGE", "RETRACT"]);
  });

  it("mixed: ASSERT (ungrounded) + HEDGE → blocked only on ASSERT", async () => {
    generateObjectMock.mockImplementationOnce(() =>
      Promise.resolve({
        object: {
          pairs: [
            { entity: "FEVER", stance: "ASSERT", proposition: "FEVER has 185,445 claims." },
            { entity: "MiniCheck-7B", stance: "HEDGE", proposition: "MiniCheck-7B is 7B params." },
          ],
        },
      } as any),
    );
    const v = await gate.runGate({
      draft: "FEVER has 185,445 claims. MiniCheck-7B is 7B params (unverified).",
      model: "vertex_ai/test",
    });
    expect(v.blocked).toBe(true);
    const ungrounded = v.pairs.filter((p) => !p.grounded);
    expect(ungrounded.length).toBe(1);
    expect(ungrounded[0]!.stance).toBe("ASSERT");
    expect(ungrounded[0]!.entity).toBe("FEVER");
  });
});

// ---------------------------------------------------------------------------
// runGateCli — CLI dispatch
// ---------------------------------------------------------------------------

describe("runGateCli", () => {
  it("bypass env → exit 0 (never even calls extractor)", async () => {
    process.env.VOUCH_GATE_BYPASS = "1";
    const r = await gate.runGateCli({
      draft: "FEVER has 185k claims.",
      model: "test",
      strict: true,
      bypassEnv: "VOUCH_GATE_BYPASS",
    });
    expect(r.exitCode).toBe(0);
    expect(generateObjectMock).toHaveBeenCalledTimes(0);
  });

  it("strict + ungrounded → exit 2 + block message", async () => {
    generateObjectMock.mockImplementationOnce(() =>
      Promise.resolve({
        object: { pairs: [{ entity: "X", stance: "ASSERT", proposition: "X has 100 features." }] },
      } as any),
    );
    const r = await gate.runGateCli({
      draft: "X has 100 features.",
      model: "test",
      strict: true,
      bypassEnv: "VOUCH_GATE_BYPASS",
    });
    expect(r.exitCode).toBe(2);
    expect(r.message).toContain("vouch-gate");
    expect(r.message).toContain("X");
  });

  it("advisory + ungrounded → exit 0 with advisory message", async () => {
    generateObjectMock.mockImplementationOnce(() =>
      Promise.resolve({
        object: { pairs: [{ entity: "X", stance: "ASSERT", proposition: "X has 100 features." }] },
      } as any),
    );
    const r = await gate.runGateCli({
      draft: "X has 100 features.",
      model: "test",
      strict: false,
      bypassEnv: "VOUCH_GATE_BYPASS",
    });
    expect(r.exitCode).toBe(0);
    expect(r.message).toContain("advisory");
  });

  it("missing transcript path → fail-open exit 0", async () => {
    const r = await gate.runGateCli({
      transcriptPath: "/definitely/does/not/exist.jsonl",
      model: "test",
      strict: true,
      bypassEnv: "VOUCH_GATE_BYPASS",
    });
    expect(r.exitCode).toBe(0);
    expect(generateObjectMock).toHaveBeenCalledTimes(0);
  });

  it("hookPayload → derives transcript_path → reads last assistant text", async () => {
    const path = join(tmp, "transcript.jsonl");
    writeFileSync(
      path,
      [
        JSON.stringify({ type: "user", message: { content: "hi" } }),
        JSON.stringify({
          type: "assistant",
          message: { content: [{ type: "text", text: "FEVER has 185,445 claims." }] },
        }),
      ].join("\n"),
    );
    generateObjectMock.mockImplementationOnce(() =>
      Promise.resolve({
        object: { pairs: [{ entity: "FEVER", stance: "ASSERT", proposition: "FEVER has 185,445 claims." }] },
      } as any),
    );
    const r = await gate.runGateCli({
      hookPayload: { transcript_path: path },
      model: "test",
      strict: true,
      bypassEnv: "VOUCH_GATE_BYPASS",
    });
    expect(r.exitCode).toBe(2);
    expect(r.verdict.pairs[0]!.entity).toBe("FEVER");
    expect(r.verdict.pairs[0]!.stance).toBe("ASSERT");
  });

  it("classifier failure → fail-open exit 0", async () => {
    generateObjectMock.mockImplementationOnce(() => Promise.reject(new Error("boom")));
    const r = await gate.runGateCli({
      draft: "FEVER has 185k claims.",
      model: "test",
      strict: true,
      bypassEnv: "VOUCH_GATE_BYPASS",
    });
    expect(r.exitCode).toBe(0);
    expect(r.verdict.classifier_error).toBeDefined();
  });

  it("empty draft → fail-open exit 0", async () => {
    const r = await gate.runGateCli({
      draft: "   \n  ",
      model: "test",
      strict: true,
      bypassEnv: "VOUCH_GATE_BYPASS",
    });
    expect(r.exitCode).toBe(0);
    expect(generateObjectMock).toHaveBeenCalledTimes(0);
  });
});

// ---------------------------------------------------------------------------
// lastAssistantText — transcript JSONL parsing
// ---------------------------------------------------------------------------

describe("lastAssistantText", () => {
  it("plain string content", () => {
    const path = join(tmp, "t1.jsonl");
    writeFileSync(
      path,
      JSON.stringify({ type: "assistant", message: { content: "hello" } }) + "\n",
    );
    expect(gate.lastAssistantText(path)).toBe("hello");
  });

  it("array content with multiple text blocks", () => {
    const path = join(tmp, "t2.jsonl");
    writeFileSync(
      path,
      JSON.stringify({
        type: "assistant",
        message: {
          content: [
            { type: "text", text: "line1" },
            { type: "tool_use", name: "Read" },
            { type: "text", text: "line2" },
          ],
        },
      }) + "\n",
    );
    expect(gate.lastAssistantText(path)).toBe("line1\nline2");
  });

  it("ignores corrupt JSONL lines", () => {
    const path = join(tmp, "t3.jsonl");
    writeFileSync(
      path,
      "not json\n" +
        JSON.stringify({ type: "assistant", message: { content: "ok" } }) +
        "\n",
    );
    expect(gate.lastAssistantText(path)).toBe("ok");
  });

  it("returns empty when no assistant entry", () => {
    const path = join(tmp, "t4.jsonl");
    writeFileSync(
      path,
      JSON.stringify({ type: "user", message: { content: "hi" } }) + "\n",
    );
    expect(gate.lastAssistantText(path)).toBe("");
  });

  // SUN-62: only consider the most recent assistant turn's direct text.
  // Tool-result content from prior user turns must not leak into the draft;
  // a tool-only most-recent assistant turn must return empty (no walk-back).

  it("excludes tool_result content from prior user turns", () => {
    const path = join(tmp, "t5.jsonl");
    writeFileSync(
      path,
      [
        JSON.stringify({
          type: "assistant",
          message: {
            content: [
              {
                type: "tool_use",
                id: "tu_1",
                name: "mcp__linear-server__save_issue",
                input: { description: "Patronus Lynx is an 8B-param model with 87.7% on HaluBench." },
              },
            ],
          },
        }),
        JSON.stringify({
          type: "user",
          message: {
            content: [
              {
                tool_use_id: "tu_1",
                type: "tool_result",
                content: '{"id":"SUN-XX","description":"Patronus Lynx is an 8B-param model..."}',
              },
            ],
          },
        }),
        JSON.stringify({
          type: "assistant",
          message: { content: [{ type: "text", text: "Hello" }] },
        }),
      ].join("\n"),
    );
    expect(gate.lastAssistantText(path)).toBe("Hello");
  });

  it("most-recent assistant turn is tool-only → returns empty (no walk-back)", () => {
    const path = join(tmp, "t6.jsonl");
    writeFileSync(
      path,
      [
        JSON.stringify({
          type: "assistant",
          message: {
            content: [
              { type: "text", text: "FEVER has 185,445 claims." },
            ],
          },
        }),
        JSON.stringify({
          type: "user",
          message: { content: [{ tool_use_id: "tu_x", type: "tool_result", content: "ok" }] },
        }),
        JSON.stringify({
          type: "assistant",
          message: {
            content: [
              { type: "tool_use", id: "tu_2", name: "Bash", input: { command: "ls" } },
            ],
          },
        }),
      ].join("\n"),
    );
    expect(gate.lastAssistantText(path)).toBe("");
  });

  it("skips sidechain assistant events", () => {
    const path = join(tmp, "t7.jsonl");
    writeFileSync(
      path,
      [
        JSON.stringify({
          type: "assistant",
          message: { content: [{ type: "text", text: "main thread answer" }] },
        }),
        JSON.stringify({
          type: "assistant",
          isSidechain: true,
          message: { content: [{ type: "text", text: "subagent chatter" }] },
        }),
      ].join("\n"),
    );
    expect(gate.lastAssistantText(path)).toBe("main thread answer");
  });
});
