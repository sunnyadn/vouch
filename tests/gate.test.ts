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
            { entity: "FEVER", assertion: "FEVER has 185,445 claims." },
            { entity: "MiniCheck-7B", assertion: "MiniCheck-7B beats GPT-4o on FACTBENCH." },
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
        object: { pairs: [{ entity: "FEVER", assertion: "FEVER has 185,445 claims." }] },
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
        object: { pairs: [{ entity: "FEVER", assertion: "FEVER has 185,445 claims." }] },
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
        object: { pairs: [{ entity: "FEVER", assertion: "FEVER has 185,445 claims." }] },
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
        object: { pairs: [{ entity: "X", assertion: "X has 100 features." }] },
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
        object: { pairs: [{ entity: "X", assertion: "X has 100 features." }] },
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
        object: { pairs: [{ entity: "FEVER", assertion: "FEVER has 185,445 claims." }] },
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
});
