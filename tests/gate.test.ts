/** Gate logic tests — mocks LLM + embedder, uses real SQLite store. */
import { afterEach, beforeAll, describe, expect, it, mock } from "bun:test";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { homedir, tmpdir } from "node:os";
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
  // mockReset clears queued mockImplementationOnce as well as call history,
  // preventing leaks across tests that don't end up consuming their queued mock.
  generateObjectMock.mockReset();
  generateObjectMock.mockImplementation(() =>
    Promise.resolve({ object: { pairs: [] } } as any),
  );
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

  it("near-verbatim restate of a supported claim → grounded via lexical fast-path (no NLI call)", async () => {
    const slug = store.writeDossier({
      source_url: "https://example.com",
      source_type: "test",
      verbatim_content: "FEVER consists of 185,445 claims classified as Supported, Refuted or NotEnoughInfo.",
    });
    const cid = store.recordClaim({
      dossier_slug: slug,
      claim_text: "FEVER consists of 185445 claims",
      score: 0.95,
      status: "supported",
      claim_type: "ATOMIC",
      embedding: queryVec,
    });

    // Only the extractor mock is queued. If the gate falls through to an NLI
    // round-trip, generateObject is called a second time and returns the
    // default { pairs: [] }, which would NOT be "supported" → test would fail.
    generateObjectMock.mockImplementationOnce(() =>
      Promise.resolve({
        object: { pairs: [{ entity: "FEVER", stance: "ASSERT", proposition: "FEVER consists of 185445 claims" }] },
      } as any),
    );

    const v = await gate.runGate({
      draft: "FEVER consists of 185,445 claims.",
      model: "vertex_ai/test",
    });
    expect(v.blocked).toBe(false);
    expect(v.pairs[0]!.grounded).toBe(true);
    expect(v.pairs[0]!.matched_claim_id).toBe(cid);
    expect(generateObjectMock).toHaveBeenCalledTimes(1);
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

  it("OPINION: value/normative judgment → not blocked, no KB lookup", async () => {
    generateObjectMock.mockImplementationOnce(() =>
      Promise.resolve({
        object: {
          pairs: [
            { entity: "FEVER", stance: "OPINION", proposition: "FEVER is the best dataset for this." },
          ],
        },
      } as any),
    );
    const v = await gate.runGate({
      draft: "FEVER is the best dataset for this comparison.",
      model: "vertex_ai/test",
    });
    expect(v.blocked).toBe(false);
    expect(v.pairs[0]!.grounded).toBe(true);
    expect(v.pairs[0]!.stance).toBe("OPINION");
    expect(generateObjectMock).toHaveBeenCalledTimes(1);
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
          timestamp: new Date().toISOString(),
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

// ---------------------------------------------------------------------------
// readLatestAssistantTurn — freshness guard against transcript-flush race
// ---------------------------------------------------------------------------

describe("readLatestAssistantTurn", () => {
  it("returns immediately as fresh when latest event has a recent timestamp", async () => {
    const path = join(tmp, "fresh1.jsonl");
    const nowIso = new Date().toISOString();
    writeFileSync(
      path,
      JSON.stringify({
        type: "assistant",
        timestamp: nowIso,
        message: { content: [{ type: "text", text: "hi" }] },
      }) + "\n",
    );
    const start = Date.now();
    const turn = await gate.readLatestAssistantTurn(path, {
      intervalMs: 20,
      maxWaitMs: 500,
      freshThresholdMs: 10000,
    });
    expect(turn.text).toBe("hi");
    expect(turn.isFresh).toBe(true);
    expect(Date.now() - start).toBeLessThan(100);
  });

  it("waits, then sees a freshly-appended turn (race fix)", async () => {
    const path = join(tmp, "fresh2.jsonl");
    // Simulate the staleness scenario: file currently has only an OLD
    // turn whose timestamp is too old to be the just-finished turn.
    const oldIso = new Date(Date.now() - 60000).toISOString(); // 60s ago
    writeFileSync(
      path,
      JSON.stringify({
        type: "assistant",
        timestamp: oldIso,
        message: { content: [{ type: "text", text: "OLD turn" }] },
      }) + "\n",
    );

    // 100ms later, simulate the transcript flush of the just-finished turn.
    setTimeout(() => {
      const fs = require("node:fs");
      fs.appendFileSync(
        path,
        JSON.stringify({
          type: "assistant",
          timestamp: new Date().toISOString(),
          message: { content: [{ type: "text", text: "NEW TURN" }] },
        }) + "\n",
      );
    }, 100);

    const turn = await gate.readLatestAssistantTurn(path, {
      intervalMs: 30,
      maxWaitMs: 800,
      freshThresholdMs: 10000,
    });
    expect(turn.text).toBe("NEW TURN");
    expect(turn.isFresh).toBe(true);
  });

  it("marks stale when latest event stays too old past maxWaitMs", async () => {
    const path = join(tmp, "fresh3.jsonl");
    const oldIso = new Date(Date.now() - 60000).toISOString();
    writeFileSync(
      path,
      JSON.stringify({
        type: "assistant",
        timestamp: oldIso,
        message: { content: [{ type: "text", text: "stale Claude claim" }] },
      }) + "\n",
    );
    const start = Date.now();
    const turn = await gate.readLatestAssistantTurn(path, {
      intervalMs: 30,
      maxWaitMs: 200,
      freshThresholdMs: 10000,
    });
    expect(turn.isFresh).toBe(false);
    expect(turn.text).toBe("stale Claude claim");
    expect(Date.now() - start).toBeGreaterThanOrEqual(200);
  });

  it("returns silently with empty + stale when file has no assistant events", async () => {
    const path = join(tmp, "fresh4.jsonl");
    writeFileSync(path, '{"type":"user","message":{"content":"hi"}}\n');
    const turn = await gate.readLatestAssistantTurn(path, {
      intervalMs: 20,
      maxWaitMs: 100,
      freshThresholdMs: 10000,
    });
    expect(turn.isFresh).toBe(false);
    expect(turn.text).toBe("");
  });

  it("skips sidechain events when finding latest", async () => {
    const path = join(tmp, "fresh5.jsonl");
    const nowIso = new Date().toISOString();
    writeFileSync(
      path,
      [
        JSON.stringify({
          type: "assistant",
          timestamp: nowIso,
          message: { content: [{ type: "text", text: "main" }] },
        }),
        JSON.stringify({
          type: "assistant",
          isSidechain: true,
          timestamp: nowIso,
          message: { content: [{ type: "text", text: "subagent" }] },
        }),
      ].join("\n"),
    );
    const turn = await gate.readLatestAssistantTurn(path, {
      intervalMs: 20,
      maxWaitMs: 200,
      freshThresholdMs: 10000,
    });
    expect(turn.text).toBe("main");
    expect(turn.isFresh).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// runGateCli — race-fix integration
// ---------------------------------------------------------------------------

describe("runGateCli race-fix integration", () => {
  it("stale transcript → fail-open (exit 0, not blocked, never calls extractor)", async () => {
    const path = join(tmp, "race1.jsonl");
    // Latest assistant event is too old to be the just-finished turn:
    // staleness signal that should trigger fail-open.
    const oldIso = new Date(Date.now() - 60000).toISOString();
    writeFileSync(
      path,
      JSON.stringify({
        type: "assistant",
        timestamp: oldIso,
        message: {
          content: [{ type: "text", text: "Claude has 999 features." }],
        },
      }) + "\n",
    );

    // If the extractor were called, it would fire ASSERT → block.
    generateObjectMock.mockImplementationOnce(() =>
      Promise.resolve({
        object: {
          pairs: [
            {
              entity: "Claude",
              stance: "ASSERT",
              proposition: "Claude has 999 features.",
            },
          ],
        },
      } as any),
    );

    const r = await gate.runGateCli({
      transcriptPath: path,
      model: "vertex_ai/test",
      strict: true,
      bypassEnv: "VOUCH_GATE_BYPASS",
      // Tight timing so the test stays fast
    } as any);

    expect(r.exitCode).toBe(0);
    expect(r.verdict.blocked).toBe(false);
    // Extractor must NOT be called for stale transcripts
    expect(generateObjectMock).toHaveBeenCalledTimes(0);
  });

  it("fresh transcript with ungrounded ASSERT → blocked (existing behavior preserved)", async () => {
    const path = join(tmp, "race2.jsonl");
    const nowIso = new Date().toISOString();
    writeFileSync(
      path,
      JSON.stringify({
        type: "assistant",
        timestamp: nowIso,
        message: {
          content: [{ type: "text", text: "FEVER has 185,445 claims." }],
        },
      }) + "\n",
    );

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

    const r = await gate.runGateCli({
      transcriptPath: path,
      model: "vertex_ai/test",
      strict: true,
      bypassEnv: "VOUCH_GATE_BYPASS",
    } as any);

    expect(r.exitCode).toBe(2);
    expect(r.verdict.blocked).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// safeFileReadCommand — Bash single-file-read recognition (issue #22)
// ---------------------------------------------------------------------------

describe("safeFileReadCommand", () => {
  it("accepts plain cat / head / tail of a single file", () => {
    expect(gate.safeFileReadCommand("cat /repo/notes.md")).toBe("/repo/notes.md");
    expect(gate.safeFileReadCommand("cat ./relative/path.yaml")).toBe("./relative/path.yaml");
    expect(gate.safeFileReadCommand("  cat   spaced.txt  ")).toBe("spaced.txt");
    expect(gate.safeFileReadCommand("head -n 20 src/gate.ts")).toBe("src/gate.ts");
    expect(gate.safeFileReadCommand("head -n20 src/gate.ts")).toBe("src/gate.ts");
    expect(gate.safeFileReadCommand("head -20 file")).toBe("file");
    expect(gate.safeFileReadCommand("head -c 500 file")).toBe("file");
    expect(gate.safeFileReadCommand("tail file")).toBe("file");
    expect(gate.safeFileReadCommand("tail -n +5 CHANGELOG.md")).toBe("CHANGELOG.md");
    expect(gate.safeFileReadCommand("/bin/cat /etc/hosts")).toBe("/etc/hosts");
    expect(gate.safeFileReadCommand('cat "spaced name.md"')).toBe("spaced name.md");
    expect(gate.safeFileReadCommand("cat 'quoted.txt'")).toBe("quoted.txt");
    expect(gate.safeFileReadCommand("cat -- -dashfile")).toBe("-dashfile");
  });

  it("accepts a `< file` input redirect with no positional arg", () => {
    expect(gate.safeFileReadCommand("cat < notes.md")).toBe("notes.md");
    expect(gate.safeFileReadCommand("cat<notes.md")).toBe("notes.md");
    expect(gate.safeFileReadCommand("head -n 5 < file.txt")).toBe("file.txt");
  });

  it("expands a leading ~", () => {
    expect(gate.safeFileReadCommand("cat ~/.vouch/.env")).toBe(join(homedir(), ".vouch/.env"));
    expect(gate.safeFileReadCommand("cat ~")).toBe(homedir());
  });

  it("rejects pipes, substitutions, globs, multi-file, redirects, and non-read commands", () => {
    for (const c of [
      "cat a.md | grep foo",
      "cat $(ls)",
      "cat `ls`",
      "cat *.md",
      "cat a.md b.md",
      "cat a.md && echo done",
      "cat a.md; cat b.md",
      "cat a.md > out.txt",
      "cat a.md >> out.txt",
      "cat file 2>/dev/null",
      "cat $HOME/.env",
      "cat {a,b}.md",
      "grep -r foo .",
      "find . -name '*.md' -exec cat {} ;",
      "python foo.py",
      "jq . data.json",
      "ls -la",
      "cat -n file", // -n transforms output (line numbers)
      "cat -A file",
      "tail -f log", // follow, not a one-shot read
      "cat -", // stdin
      "cat /dev/stdin",
      "cat /dev/fd/0",
      "cat < a < b",
      "cat a.md < b.md", // redirect + positional → ambiguous
      "",
      "cat", // no file (reads stdin)
    ]) {
      expect(gate.safeFileReadCommand(c)).toBeNull();
    }
  });
});

// ---------------------------------------------------------------------------
// parseSessionSources — transcript tool_result extraction (issues #21, #22)
// ---------------------------------------------------------------------------

describe("parseSessionSources", () => {
  it("extracts Read / WebFetch / WebSearch content; strips Read line numbers", () => {
    const path = join(tmp, "ps1.jsonl");
    writeFileSync(
      path,
      [
        JSON.stringify({
          type: "assistant",
          message: { content: [{ type: "tool_use", id: "tu_read", name: "Read", input: { file_path: "/repo/notes.md" } }] },
        }),
        JSON.stringify({
          type: "user",
          message: { content: [{ type: "tool_result", tool_use_id: "tu_read", content: "     1\t# Notes\n     2\tFoo benchmark has 42 tasks." }] },
        }),
        JSON.stringify({
          type: "assistant",
          message: { content: [{ type: "tool_use", id: "tu_fetch", name: "WebFetch", input: { url: "https://example.com/foo", prompt: "..." } }] },
        }),
        JSON.stringify({
          type: "user",
          message: { content: [{ type: "tool_result", tool_use_id: "tu_fetch", content: [{ type: "text", text: "Foo benchmark page: 42 tasks total." }] }] },
        }),
        JSON.stringify({
          type: "assistant",
          message: { content: [{ type: "tool_use", id: "tu_search", name: "WebSearch", input: { query: "foo benchmark size" } }] },
        }),
        JSON.stringify({
          type: "user",
          message: { content: [{ type: "tool_result", tool_use_id: "tu_search", content: "Results: Foo benchmark — 42 tasks ..." }] },
        }),
      ].join("\n"),
    );
    const got = gate.parseSessionSources(path);
    expect(got.length).toBe(3);
    const byTool = Object.fromEntries(got.map((s) => [s.tool, s]));
    expect(byTool.Read!.uri).toBe("/repo/notes.md");
    expect(byTool.Read!.content).toBe("# Notes\nFoo benchmark has 42 tasks.");
    expect(byTool.WebFetch!.uri).toBe("https://example.com/foo");
    expect(byTool.WebFetch!.content).toContain("42 tasks total");
    expect(byTool.WebSearch!.uri).toBe("websearch:foo benchmark size");
  });

  it("ignores non-source tools, error results, and sidechain events", () => {
    const path = join(tmp, "ps2.jsonl");
    writeFileSync(
      path,
      [
        JSON.stringify({
          type: "assistant",
          message: {
            content: [
              { type: "tool_use", id: "tu_bash", name: "Bash", input: { command: "ls" } },
              { type: "tool_use", id: "tu_read_err", name: "Read", input: { file_path: "/missing" } },
            ],
          },
        }),
        JSON.stringify({
          type: "user",
          message: {
            content: [
              { type: "tool_result", tool_use_id: "tu_bash", content: "a.txt b.txt" },
              { type: "tool_result", tool_use_id: "tu_read_err", is_error: true, content: "ENOENT" },
            ],
          },
        }),
        JSON.stringify({
          type: "assistant",
          isSidechain: true,
          message: { content: [{ type: "tool_use", id: "tu_sub", name: "Read", input: { file_path: "/sub/file" } }] },
        }),
        JSON.stringify({
          type: "user",
          isSidechain: true,
          message: { content: [{ type: "tool_result", tool_use_id: "tu_sub", content: "     1\tsubagent read" }] },
        }),
      ].join("\n"),
    );
    expect(gate.parseSessionSources(path)).toEqual([]);
  });

  it("returns [] for a non-Claude-Code file or a missing file", () => {
    const path = join(tmp, "ps3.txt");
    writeFileSync(path, "just some text\nnot jsonl at all\n");
    expect(gate.parseSessionSources(path)).toEqual([]);
    expect(gate.parseSessionSources(join(tmp, "no-such-file.jsonl"))).toEqual([]);
  });

  it("picks up Bash single-file reads (cat/head); ignores pipes/multi-file/other commands; no cat -n stripping", () => {
    const path = join(tmp, "ps5.jsonl");
    writeFileSync(
      path,
      [
        // cat F → picked up verbatim (the `␣␣1\t…` text below must NOT be stripped — that's Read-tool-only)
        JSON.stringify({ type: "assistant", message: { content: [{ type: "tool_use", id: "tu_cat", name: "Bash", input: { command: "cat /repo/META.md" } }] } }),
        JSON.stringify({ type: "user", message: { content: [{ type: "tool_result", tool_use_id: "tu_cat", content: "Foo benchmark has 42 tasks.\n     1\tnot a line-number prefix" }] } }),
        // head -n N F → picked up
        JSON.stringify({ type: "assistant", message: { content: [{ type: "tool_use", id: "tu_head", name: "Bash", input: { command: "head -n 5 ./config.yaml" } }] } }),
        JSON.stringify({ type: "user", message: { content: [{ type: "tool_result", tool_use_id: "tu_head", content: "model: test" }] } }),
        // pipe → not a session source
        JSON.stringify({ type: "assistant", message: { content: [{ type: "tool_use", id: "tu_pipe", name: "Bash", input: { command: "cat /repo/big.md | grep Foo" } }] } }),
        JSON.stringify({ type: "user", message: { content: [{ type: "tool_result", tool_use_id: "tu_pipe", content: "Foo benchmark line" }] } }),
        // multi-file → not a session source
        JSON.stringify({ type: "assistant", message: { content: [{ type: "tool_use", id: "tu_multi", name: "Bash", input: { command: "cat a.md b.md" } }] } }),
        JSON.stringify({ type: "user", message: { content: [{ type: "tool_result", tool_use_id: "tu_multi", content: "merged content" }] } }),
        // git log → not a session source
        JSON.stringify({ type: "assistant", message: { content: [{ type: "tool_use", id: "tu_git", name: "Bash", input: { command: "git log --oneline -5" } }] } }),
        JSON.stringify({ type: "user", message: { content: [{ type: "tool_result", tool_use_id: "tu_git", content: "abc123 commit" }] } }),
      ].join("\n"),
    );
    const got = gate.parseSessionSources(path);
    expect(got.length).toBe(2);
    expect(got.every((s) => s.tool === "Bash")).toBe(true);
    const byUri = Object.fromEntries(got.map((s) => [s.uri, s]));
    expect(byUri["/repo/META.md"]!.content).toBe("Foo benchmark has 42 tasks.\n     1\tnot a line-number prefix");
    expect(byUri["./config.yaml"]!.content).toBe("model: test");
  });

  it("dedups by (tool|uri), keeping the freshest content", () => {
    const path = join(tmp, "ps4.jsonl");
    writeFileSync(
      path,
      [
        JSON.stringify({ type: "assistant", message: { content: [{ type: "tool_use", id: "tu_a", name: "Read", input: { file_path: "/f" } }] } }),
        JSON.stringify({ type: "user", message: { content: [{ type: "tool_result", tool_use_id: "tu_a", content: "     1\told version" }] } }),
        JSON.stringify({ type: "assistant", message: { content: [{ type: "tool_use", id: "tu_b", name: "Read", input: { file_path: "/f" } }] } }),
        JSON.stringify({ type: "user", message: { content: [{ type: "tool_result", tool_use_id: "tu_b", content: "     1\tnew version" }] } }),
      ].join("\n"),
    );
    const got = gate.parseSessionSources(path);
    expect(got.length).toBe(1);
    expect(got[0]!.content).toBe("new version");
  });
});

// ---------------------------------------------------------------------------
// runGate — session-evidence auto-grounding (issue #21)
// ---------------------------------------------------------------------------

describe("runGate session-evidence auto-grounding", () => {
  function transcriptWithSource(toolName: string, id: string, input: any, resultContent: any): string {
    return [
      JSON.stringify({ type: "assistant", message: { content: [{ type: "tool_use", id, name: toolName, input }] } }),
      JSON.stringify({ type: "user", message: { content: [{ type: "tool_result", tool_use_id: id, content: resultContent }] } }),
    ].join("\n");
  }

  it("ungrounded ASSERT entailed by a session Read → auto-grounds, records dossier+claim, not blocked", async () => {
    const path = join(tmp, "ag1.jsonl");
    writeFileSync(
      path,
      transcriptWithSource("Read", "tu1", { file_path: "/repo/META.md" }, "     1\tFoo benchmark consists of 42 evaluation tasks across 6 domains."),
    );
    generateObjectMock.mockImplementationOnce(() =>
      Promise.resolve({
        object: { pairs: [{ entity: "Foo benchmark", stance: "ASSERT", proposition: "Foo benchmark has 42 evaluation tasks." }] },
      } as any),
    );
    // KB empty → checkGrounding makes no NLI call; next mock is the session NLI.
    generateObjectMock.mockImplementationOnce(() =>
      Promise.resolve({ object: { supported: true, score: 0.91, reason: "source states 42 tasks" } } as any),
    );

    const v = await gate.runGate({
      draft: "Foo benchmark has 42 evaluation tasks.",
      model: "vertex_ai/test",
      sessionTranscriptPath: path,
    });
    expect(v.blocked).toBe(false);
    expect(v.pairs[0]!.grounded).toBe(true);
    expect(v.pairs[0]!.auto_grounded).toBe(true);
    expect(v.pairs[0]!.session_sources_checked).toBe(1);
    const cid = v.pairs[0]!.matched_claim_id!;
    const claim = store.getClaim(cid)!;
    expect(claim.status).toBe("supported");
    expect(claim.verification).toBe("nli-session");
    const dossier = store.getDossier(claim.dossier_slug)!;
    expect(dossier.scope).toBe("workspace");
    expect(dossier.source_type).toBe("session-read");
    expect(dossier.source_url).toBe("/repo/META.md");
    expect(generateObjectMock).toHaveBeenCalledTimes(2);
  });

  it("session WebFetch entailment → scope third-party, source_type session-webfetch", async () => {
    const path = join(tmp, "ag2.jsonl");
    writeFileSync(
      path,
      transcriptWithSource("WebFetch", "tu1", { url: "https://example.org/bar" }, [{ type: "text", text: "Bar dataset: 1,000 labeled examples." }]),
    );
    generateObjectMock.mockImplementationOnce(() =>
      Promise.resolve({ object: { pairs: [{ entity: "Bar dataset", stance: "ASSERT", proposition: "Bar dataset has 1,000 labeled examples." }] } } as any),
    );
    generateObjectMock.mockImplementationOnce(() => Promise.resolve({ object: { supported: true, score: 0.88, reason: "stated" } } as any));
    const v = await gate.runGate({ draft: "Bar dataset has 1,000 labeled examples.", model: "t", sessionTranscriptPath: path });
    expect(v.blocked).toBe(false);
    const dossier = store.getDossier(store.getClaim(v.pairs[0]!.matched_claim_id!)!.dossier_slug)!;
    expect(dossier.scope).toBe("third-party");
    expect(dossier.source_type).toBe("session-webfetch");
    expect(dossier.source_url).toBe("https://example.org/bar");
  });

  it("ungrounded ASSERT entailed by a session Bash `cat` → auto-grounds, scope workspace, source_type session-bash", async () => {
    const path = join(tmp, "ag-bash.jsonl");
    writeFileSync(
      path,
      transcriptWithSource("Bash", "tu1", { command: "cat /repo/META.md" }, "Foo benchmark consists of 42 evaluation tasks across 6 domains."),
    );
    generateObjectMock.mockImplementationOnce(() =>
      Promise.resolve({ object: { pairs: [{ entity: "Foo benchmark", stance: "ASSERT", proposition: "Foo benchmark has 42 evaluation tasks." }] } } as any),
    );
    generateObjectMock.mockImplementationOnce(() => Promise.resolve({ object: { supported: true, score: 0.9, reason: "stated" } } as any));
    const v = await gate.runGate({ draft: "Foo benchmark has 42 evaluation tasks.", model: "t", sessionTranscriptPath: path });
    expect(v.blocked).toBe(false);
    expect(v.pairs[0]!.auto_grounded).toBe(true);
    expect(v.pairs[0]!.session_sources_checked).toBe(1);
    const claim = store.getClaim(v.pairs[0]!.matched_claim_id!)!;
    expect(claim.verification).toBe("nli-session");
    const dossier = store.getDossier(claim.dossier_slug)!;
    expect(dossier.scope).toBe("workspace");
    expect(dossier.source_type).toBe("session-bash");
    expect(dossier.source_url).toBe("/repo/META.md");
  });

  it("entity appears only in piped Bash output → still blocked (a pipe is not a session source, no session NLI)", async () => {
    const path = join(tmp, "ag-bash-pipe.jsonl");
    writeFileSync(
      path,
      transcriptWithSource("Bash", "tu1", { command: "cat /repo/big.md | grep -i foo" }, "Foo benchmark has 42 evaluation tasks."),
    );
    generateObjectMock.mockImplementationOnce(() =>
      Promise.resolve({ object: { pairs: [{ entity: "Foo benchmark", stance: "ASSERT", proposition: "Foo benchmark has 42 evaluation tasks." }] } } as any),
    );
    const v = await gate.runGate({ draft: "Foo benchmark has 42 evaluation tasks.", model: "t", sessionTranscriptPath: path });
    expect(v.blocked).toBe(true);
    expect(v.pairs[0]!.session_sources_checked).toBeUndefined(); // parseSessionSources returned [] → loop skipped
    expect(generateObjectMock).toHaveBeenCalledTimes(1);
    expect(store.listDossiers().length).toBe(0);
  });

  it("session source mentions the entity but does not entail → still blocked, session_sources_checked recorded", async () => {
    const path = join(tmp, "ag3.jsonl");
    writeFileSync(path, transcriptWithSource("Read", "tu1", { file_path: "/repo/x.md" }, "     1\tFoo benchmark is a popular evaluation suite."));
    generateObjectMock.mockImplementationOnce(() =>
      Promise.resolve({ object: { pairs: [{ entity: "Foo benchmark", stance: "ASSERT", proposition: "Foo benchmark has 42 evaluation tasks." }] } } as any),
    );
    generateObjectMock.mockImplementationOnce(() => Promise.resolve({ object: { supported: false, score: 0.1, reason: "no task count in source" } } as any));
    const v = await gate.runGate({ draft: "Foo benchmark has 42 evaluation tasks.", model: "t", sessionTranscriptPath: path });
    expect(v.blocked).toBe(true);
    expect(v.pairs[0]!.grounded).toBe(false);
    expect(v.pairs[0]!.session_sources_checked).toBe(1);
    expect(v.pairs[0]!.auto_grounded).toBeUndefined();
    // No dossier/claim should have been recorded for a non-entailing source.
    expect(store.listDossiers().length).toBe(0);
  });

  it("no session source mentions the entity → no NLI attempt, blocked with session_sources_checked=0", async () => {
    const path = join(tmp, "ag4.jsonl");
    writeFileSync(path, transcriptWithSource("Read", "tu1", { file_path: "/repo/unrelated.md" }, "     1\tThis file is about something else entirely."));
    generateObjectMock.mockImplementationOnce(() =>
      Promise.resolve({ object: { pairs: [{ entity: "Foo benchmark", stance: "ASSERT", proposition: "Foo benchmark has 42 evaluation tasks." }] } } as any),
    );
    const v = await gate.runGate({ draft: "Foo benchmark has 42 evaluation tasks.", model: "t", sessionTranscriptPath: path });
    expect(v.blocked).toBe(true);
    expect(v.pairs[0]!.session_sources_checked).toBe(0);
    expect(generateObjectMock).toHaveBeenCalledTimes(1);
  });

  it("no sessionTranscriptPath → unchanged behavior, no session fields", async () => {
    generateObjectMock.mockImplementationOnce(() =>
      Promise.resolve({ object: { pairs: [{ entity: "Foo benchmark", stance: "ASSERT", proposition: "Foo benchmark has 42 tasks." }] } } as any),
    );
    const v = await gate.runGate({ draft: "Foo benchmark has 42 tasks.", model: "t" });
    expect(v.blocked).toBe(true);
    expect(v.pairs[0]!.session_sources_checked).toBeUndefined();
    expect(generateObjectMock).toHaveBeenCalledTimes(1);
  });

  it("only the agent's own draft text, no tool sources → does not auto-ground", async () => {
    // Transcript with the assistant turn only (no tool_result events): the
    // proposition is "supported" by nothing the agent retrieved via a tool.
    const path = join(tmp, "ag5.jsonl");
    writeFileSync(
      path,
      JSON.stringify({ type: "assistant", message: { content: [{ type: "text", text: "Foo benchmark has 42 tasks." }] } }) + "\n",
    );
    generateObjectMock.mockImplementationOnce(() =>
      Promise.resolve({ object: { pairs: [{ entity: "Foo benchmark", stance: "ASSERT", proposition: "Foo benchmark has 42 tasks." }] } } as any),
    );
    const v = await gate.runGate({ draft: "Foo benchmark has 42 tasks.", model: "t", sessionTranscriptPath: path });
    expect(v.blocked).toBe(true);
    expect(v.pairs[0]!.session_sources_checked).toBeUndefined(); // parseSessionSources returned [] → loop skipped
    expect(generateObjectMock).toHaveBeenCalledTimes(1);
  });
});

// ---------------------------------------------------------------------------
// runGateCli — session-evidence auto-grounding integration (issue #21)
// ---------------------------------------------------------------------------

describe("runGateCli session-evidence integration", () => {
  it("transcript with a prior tool_result → auto-grounds the just-emitted draft, exit 0 + visible message", async () => {
    const path = join(tmp, "cli-ag.jsonl");
    const nowIso = new Date().toISOString();
    writeFileSync(
      path,
      [
        JSON.stringify({ type: "assistant", message: { content: [{ type: "tool_use", id: "tu1", name: "Read", input: { file_path: "/repo/META.md" } }] } }),
        JSON.stringify({ type: "user", message: { content: [{ type: "tool_result", tool_use_id: "tu1", content: "     1\tQux corpus contains 7,532 documents." }] } }),
        JSON.stringify({ type: "assistant", timestamp: nowIso, message: { content: [{ type: "text", text: "The Qux corpus contains 7,532 documents." }] } }),
      ].join("\n"),
    );
    generateObjectMock.mockImplementationOnce(() =>
      Promise.resolve({ object: { pairs: [{ entity: "Qux corpus", stance: "ASSERT", proposition: "The Qux corpus contains 7,532 documents." }] } } as any),
    );
    generateObjectMock.mockImplementationOnce(() => Promise.resolve({ object: { supported: true, score: 0.93, reason: "stated verbatim" } } as any));
    const r = await gate.runGateCli({ transcriptPath: path, model: "vertex_ai/test", strict: true, bypassEnv: "VOUCH_GATE_BYPASS" });
    expect(r.exitCode).toBe(0);
    expect(r.verdict.blocked).toBe(false);
    expect(r.verdict.pairs[0]!.auto_grounded).toBe(true);
    expect(r.message).toContain("auto-grounded");
    expect(r.message).toContain("verified:");
  });

  it("strict block message reports how many session sources were checked", async () => {
    const path = join(tmp, "cli-ag-block.jsonl");
    const nowIso = new Date().toISOString();
    writeFileSync(
      path,
      [
        JSON.stringify({ type: "assistant", message: { content: [{ type: "tool_use", id: "tu1", name: "WebFetch", input: { url: "https://example.org/q" } }] } }),
        JSON.stringify({ type: "user", message: { content: [{ type: "tool_result", tool_use_id: "tu1", content: "Quux is a widely used tool." }] } }),
        JSON.stringify({ type: "assistant", timestamp: nowIso, message: { content: [{ type: "text", text: "Quux has 12,345 users." }] } }),
      ].join("\n"),
    );
    generateObjectMock.mockImplementationOnce(() =>
      Promise.resolve({ object: { pairs: [{ entity: "Quux", stance: "ASSERT", proposition: "Quux has 12,345 users." }] } } as any),
    );
    generateObjectMock.mockImplementationOnce(() => Promise.resolve({ object: { supported: false, score: 0.05, reason: "no user count" } } as any));
    const r = await gate.runGateCli({ transcriptPath: path, model: "vertex_ai/test", strict: true, bypassEnv: "VOUCH_GATE_BYPASS" });
    expect(r.exitCode).toBe(2);
    expect(r.message).toContain("checked 1 session source(s), none entailed");
  });
});

// ---------------------------------------------------------------------------
// parseDerivedTags (issue #23)
// ---------------------------------------------------------------------------

describe("parseDerivedTags", () => {
  it("parses each tag form with ids, segment, and a `; score:` override", () => {
    const draft =
      "GraphRAG uses community detection over a knowledge graph [verified: 5]. " +
      "Given GraphRAG's high indexing cost and LightRAG's lower token usage, LightRAG is the more defensible default for cost-constrained SaaS [inference-from: 5, 6; score: 0.85]. " +
      "GraphRAG and LightRAG both target RAG quality [synthesis-of: 5, 6]. " +
      "Restating: LightRAG keeps retrieval cheap [interpretation: 6]. " +
      "It might also help latency-sensitive workloads [hypothesis].";
    const tags = gate.parseDerivedTags(draft);
    expect(tags.map((t) => t.kind)).toEqual([
      "verified",
      "inference-from",
      "synthesis-of",
      "interpretation",
      "hypothesis",
    ]);
    const inf = tags[1]!;
    expect(inf.ids).toEqual([5, 6]);
    expect(inf.softScore).toBe(0.85);
    expect(inf.segment).toBe(
      "Given GraphRAG's high indexing cost and LightRAG's lower token usage, LightRAG is the more defensible default for cost-constrained SaaS",
    );
    expect(tags[3]!.ids).toEqual([6]);
    expect(tags[3]!.softScore).toBeNull();
    expect(tags[4]!.ids).toEqual([]);
    expect(tags[4]!.segment).toBe("It might also help latency-sensitive workloads");
  });

  it("accepts bare [inference: ids] / [synthesis: ids] aliases and [hypothesis; score: N]", () => {
    const tags = gate.parseDerivedTags("X deduces Y [inference: 9]. X plus Z give W [synthesis: 9, 10]. Maybe Q [hypothesis; score: 0.3].");
    expect(tags[0]!.kind).toBe("inference-from");
    expect(tags[0]!.ids).toEqual([9]);
    expect(tags[1]!.kind).toBe("synthesis-of");
    expect(tags[1]!.ids).toEqual([9, 10]);
    expect(tags[2]!.kind).toBe("hypothesis");
    expect(tags[2]!.ids).toEqual([]);
    expect(tags[2]!.softScore).toBe(0.3);
  });

  it("does not split a segment on a mid-sentence abbreviation, but does at a real sentence end", () => {
    expect(gate.parseDerivedTags("Per the paper, e.g. Table 2, method X beats Y by 3 points [inference-from: 1].")[0]!.segment).toBe(
      "Per the paper, e.g. Table 2, method X beats Y by 3 points",
    );
    expect(gate.parseDerivedTags("Here is meta prose. We then set up X. Given the data, X wins [inference-from: 1].")[0]!.segment).toBe(
      "Given the data, X wins",
    );
  });

  it("returns [] for untagged prose", () => {
    expect(gate.parseDerivedTags("Just a normal sentence with no tags at all.")).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// runGate — tagged-derived-claim harvesting (issue #23)
// ---------------------------------------------------------------------------

describe("runGate tagged-derived-claim harvest", () => {
  function atomic(text: string, topic?: string): number {
    const slug = store.writeDossier({ source_url: "https://x/" + text.slice(0, 8), source_type: "test", verbatim_content: text + " — full source text." });
    return store.recordClaim({
      dossier_slug: slug,
      claim_text: text,
      score: 1,
      status: "supported",
      claim_type: "ATOMIC",
      topic: topic ?? null,
      embedding: queryVec,
    });
  }

  it("passing draft → harvests INFERENCE/SYNTHESIS/INTERPRETATION/HYPOTHESIS with right deps + soft_score; dedups on re-emit", async () => {
    const c1 = atomic("LightRAG retrieval token cost is lower than GraphRAG", "rag");
    const c2 = atomic("Microsoft's GraphRAG README warns it is expensive to index", "rag");
    // No extractor mock queued → default returns { pairs: [] } → gate passes.
    const draft =
      `LightRAG retrieval token cost is lower than GraphRAG [verified: ${c1}]. ` +
      `Given LightRAG's lower retrieval cost and GraphRAG's own 'expensive' warning, LightRAG is the more defensible default for cost-constrained SaaS [inference-from: ${c1}, ${c2}; score: 0.85]. ` +
      `LightRAG and GraphRAG both implement graph-structured RAG [synthesis-of: ${c1}, ${c2}]. ` +
      `Restating: GraphRAG's indexing step is the costly part [interpretation: ${c2}]. ` +
      `It may also matter for latency-sensitive setups [hypothesis].`;
    const v = await gate.runGate({ draft, model: "t" });
    expect(v.blocked).toBe(false);
    expect(v.harvest).toBeDefined();
    const filed = v.harvest!.filed;
    expect(filed.map((f) => f.claim_type).sort()).toEqual(["HYPOTHESIS", "INFERENCE", "INTERPRETATION", "SYNTHESIS"]);

    const inf = filed.find((f) => f.claim_type === "INFERENCE")!;
    expect([...inf.depends_on].sort()).toEqual([c1, c2].sort());
    expect(inf.soft_score).toBe(0.85);
    const infClaim = store.getClaim(inf.claim_id)!;
    expect(infClaim.claim_type).toBe("INFERENCE");
    expect(infClaim.status).toBe("recorded");
    expect(infClaim.verification).toBe("tag-harvest");
    expect(infClaim.topic).toBe("rag"); // inherited — both upstreams share it
    expect(infClaim.depends_on.map((d) => d.depends_on_id).sort()).toEqual([c1, c2].sort());

    const syn = filed.find((f) => f.claim_type === "SYNTHESIS")!;
    expect([...syn.depends_on].sort()).toEqual([c1, c2].sort());
    expect(store.getClaim(syn.claim_id)!.depends_on[0]!.dependency_type).toBe("support");

    const interp = filed.find((f) => f.claim_type === "INTERPRETATION")!;
    expect(interp.depends_on).toEqual([c2]);

    const hyp = filed.find((f) => f.claim_type === "HYPOTHESIS")!;
    expect(hyp.depends_on).toEqual([]);
    expect(hyp.soft_score).toBe(0.4);
    expect(store.getClaim(hyp.claim_id)!.author).toBe("gate-harvest");

    // [verified: c1] never creates a claim — only the 4 derived ones were filed.
    const allClaims = store.listClaims({ limit: 100 });
    expect(allClaims.filter((c) => c.author === "gate-harvest").length).toBe(4);

    // Re-emit the same draft → nothing new filed; the 4 derived are reported as skipped.
    const v2 = await gate.runGate({ draft, model: "t" });
    expect(v2.blocked).toBe(false);
    expect(v2.harvest!.filed.length).toBe(0);
    expect(v2.harvest!.skipped.length).toBe(4);
    expect(store.listClaims({ limit: 100 }).filter((c) => c.author === "gate-harvest").length).toBe(4);
  });

  it("flags a dangling [verified: id], and a missing-upstream [inference-from:]; does not file a claim with no resolvable deps", async () => {
    const v = await gate.runGate({
      draft: "Foo is great [verified: 99999]. Therefore bar [inference-from: 99998].",
      model: "t",
    });
    expect(v.blocked).toBe(false);
    expect(v.harvest!.filed.length).toBe(0);
    expect(v.harvest!.flags.some((f) => f.includes("99999") && f.includes("not in the KB"))).toBe(true);
    expect(v.harvest!.flags.some((f) => f.includes("99998") && f.includes("dropped from depends_on"))).toBe(true);
    expect(v.harvest!.flags.some((f) => f.includes("inference-from") && f.includes("not filed"))).toBe(true);
  });

  it("[verified: id] pointing at an unsupported claim is flagged (not re-filed)", async () => {
    const slug = store.writeDossier({ source_url: "https://u", source_type: "test", verbatim_content: "blah blah" });
    const cid = store.recordClaim({ dossier_slug: slug, claim_text: "Unsupported thing", score: 0, status: "unsupported", claim_type: "ATOMIC", embedding: queryVec });
    const v = await gate.runGate({ draft: `Unsupported thing [verified: ${cid}].`, model: "t" });
    expect(v.harvest!.filed.length).toBe(0);
    expect(v.harvest!.flags.some((f) => f.includes(`${cid}`) && f.includes("status=unsupported"))).toBe(true);
  });

  it("a blocked draft does NOT harvest", async () => {
    const c1 = atomic("Some upstream claim");
    generateObjectMock.mockImplementationOnce(() =>
      Promise.resolve({ object: { pairs: [{ entity: "Widget", stance: "ASSERT", proposition: "Widget has 9000 features." }] } } as any),
    );
    const v = await gate.runGate({
      draft: `Widget has 9000 features. Therefore widgets win [inference-from: ${c1}].`,
      model: "t",
    });
    expect(v.blocked).toBe(true);
    expect(v.harvest).toBeUndefined();
    expect(store.listClaims({ claim_type: "INFERENCE", limit: 10 }).length).toBe(0);
  });

  it("untagged passing draft → no harvest field", async () => {
    const v = await gate.runGate({ draft: "Looking at the vault now. Nothing to ground here.", model: "t" });
    expect(v.blocked).toBe(false);
    expect(v.harvest).toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// runGateCli — harvest integration (issue #23)
// ---------------------------------------------------------------------------

describe("runGateCli harvest integration", () => {
  it("passing transcript with derived tags → exit 0, claims recorded, visible harvest message", async () => {
    const slug = store.writeDossier({ source_url: "https://h", source_type: "test", verbatim_content: "Premise text here." });
    const c1 = store.recordClaim({ dossier_slug: slug, claim_text: "Premise about Frobnicator", score: 1, status: "supported", claim_type: "ATOMIC", topic: "frob", embedding: queryVec });
    const path = join(tmp, "cli-harvest.jsonl");
    const nowIso = new Date().toISOString();
    writeFileSync(
      path,
      [
        JSON.stringify({ type: "user", message: { content: "do it" } }),
        JSON.stringify({
          type: "assistant",
          timestamp: nowIso,
          message: { content: [{ type: "text", text: `Premise about Frobnicator [verified: ${c1}]. Given the premise, the Frobnicator approach is the safer default here [inference-from: ${c1}].` }] },
        }),
      ].join("\n"),
    );
    // No extractor mock queued → default { pairs: [] } → gate passes.
    const r = await gate.runGateCli({ transcriptPath: path, model: "vertex_ai/test", strict: true, bypassEnv: "VOUCH_GATE_BYPASS" });
    expect(r.exitCode).toBe(0);
    expect(r.verdict.blocked).toBe(false);
    expect(r.verdict.harvest!.filed.length).toBe(1);
    expect(r.verdict.harvest!.filed[0]!.claim_type).toBe("INFERENCE");
    expect(r.message).toContain("harvested 1 derived claim");
    const filedClaim = store.getClaim(r.verdict.harvest!.filed[0]!.claim_id)!;
    expect(filedClaim.topic).toBe("frob");
    expect(filedClaim.depends_on[0]!.depends_on_id).toBe(c1);
  });
});
