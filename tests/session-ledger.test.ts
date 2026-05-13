/** Session ledger + cross-turn contradiction-fire regression spec — issue #43.
 *
 *  Tests `applySessionLedger` (RETRACT auto-mark + contradiction-fire via
 *  inverse-NLI + ledger writes) and the underlying store helpers
 *  (`recordSessionClaim`, `findSessionContradictionCandidates`,
 *  `markSessionClaimRetracted`, `markSessionClaimSuperseded`).
 */
import { afterEach, beforeAll, describe, expect, it, mock } from "bun:test";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const tmp = mkdtempSync(join(tmpdir(), "vouch-sledger-test-"));
process.env.VOUCH_DB_PATH = join(tmp, "test.db");

// `generateObject` is shared by extractPairs, verifyClaim*, verifyContradiction.
// Use a stateful mock-impl queue: each call shifts the next response.
const responseQueue: any[] = [];
const generateObjectMock = mock(() => {
  const next = responseQueue.shift();
  return Promise.resolve(next ?? { object: {} } as any);
});
mock.module("ai", () => ({
  generateObject: generateObjectMock,
  embed: () => Promise.resolve({ embeddings: [] }),
}));

mock.module("../src/providers.ts", () => ({
  getLanguageModel: () => ({ id: "test-model" }) as any,
  getEmbeddingModel: () => ({ id: "test-embedder" }) as any,
}));

// Each proposition gets a unique embedding so cosine distinguishes them. The
// gate-side cosine candidate filter uses a threshold; we make embeddings
// near-identical when we want a match, orthogonal when we don't.
const embedMap = new Map<string, Float32Array>();
function setEmbed(text: string, vec: number[]) {
  embedMap.set(text, new Float32Array(vec));
}
mock.module("../src/embedder.ts", () => ({
  embedOne: (text: string) =>
    Promise.resolve(embedMap.get(text) ?? new Float32Array([1, 0, 0, 0])),
  embedBatch: (texts: string[]) =>
    Promise.resolve(texts.map((t) => embedMap.get(t) ?? new Float32Array([1, 0, 0, 0]))),
}));

const store = await import("../src/store.ts");
const gate = await import("../src/gate.ts");

afterEach(() => {
  const db = store.getDb();
  db.exec("DELETE FROM session_claims;");
  responseQueue.length = 0;
  generateObjectMock.mockClear();
  embedMap.clear();
});

beforeAll(() => {
  process.on("exit", () => {
    try {
      rmSync(tmp, { recursive: true, force: true });
    } catch {}
  });
});

// ---------------------------------------------------------------------------
// Helpers — make GroundedPair fixtures
// ---------------------------------------------------------------------------

function pair(opts: {
  entity: string;
  stance: string;
  proposition: string;
  grounded?: boolean;
}): any {
  return {
    entity: opts.entity,
    stance: opts.stance,
    proposition: opts.proposition,
    grounded: opts.grounded ?? true,
    matched_claim_id: null,
    reason: "test fixture",
  };
}

function makeTranscript(id: string): string {
  const path = join(tmp, `${id}.jsonl`);
  writeFileSync(path, ""); // contents don't matter — only the basename is used
  return path;
}

function queueContradictionVerdict(contradicts: boolean, score = 0.95, reason = "test") {
  responseQueue.push({ object: { contradicts, score, reason } });
}

// ---------------------------------------------------------------------------
// Store-level unit tests
// ---------------------------------------------------------------------------

describe("store: session_claims schema + helpers", () => {
  it("getNextSessionTurnIdx — empty transcript returns 0", () => {
    expect(store.getNextSessionTurnIdx("tid-empty")).toBe(0);
  });

  it("recordSessionClaim + listSessionClaims round-trip", () => {
    store.recordSessionClaim({
      transcript_id: "tid-1",
      turn_idx: 0,
      claim_idx: 0,
      proposition: "rfsrc has 4 stars",
      entity: "rfsrc",
      stance: "ASSERT",
      verdict: "grounded",
      reason: "ok",
      embedding: new Float32Array([1, 0, 0]),
    });
    const rows = store.listSessionClaims("tid-1");
    expect(rows.length).toBe(1);
    expect(rows[0]!.proposition).toBe("rfsrc has 4 stars");
    expect(rows[0]!.retracted).toBe(0);
    expect(rows[0]!.superseded_by_turn).toBeNull();
  });

  it("getNextSessionTurnIdx — after recording turn 0 returns 1, then 2", () => {
    store.recordSessionClaim({
      transcript_id: "tid-seq",
      turn_idx: 0,
      claim_idx: 0,
      proposition: "p0",
      entity: "e",
      stance: "ASSERT",
      verdict: "grounded",
    });
    expect(store.getNextSessionTurnIdx("tid-seq")).toBe(1);
    store.recordSessionClaim({
      transcript_id: "tid-seq",
      turn_idx: 1,
      claim_idx: 0,
      proposition: "p1",
      entity: "e",
      stance: "ASSERT",
      verdict: "grounded",
    });
    expect(store.getNextSessionTurnIdx("tid-seq")).toBe(2);
  });

  it("markSessionClaimRetracted — flips retracted to 1 and appends reason", () => {
    store.recordSessionClaim({
      transcript_id: "tid-r",
      turn_idx: 0,
      claim_idx: 0,
      proposition: "x is y",
      entity: "x",
      stance: "ASSERT",
      verdict: "grounded",
      reason: "initial",
    });
    expect(store.markSessionClaimRetracted("tid-r", 0, 0, "user pushed back")).toBe(true);
    const rows = store.listSessionClaims("tid-r", { include_retracted: true });
    expect(rows[0]!.retracted).toBe(1);
    expect(rows[0]!.reason).toContain("retracted: user pushed back");
  });

  it("listSessionClaims — default hides retracted; include_retracted shows them", () => {
    store.recordSessionClaim({
      transcript_id: "tid-h",
      turn_idx: 0,
      claim_idx: 0,
      proposition: "p0",
      entity: "e",
      stance: "ASSERT",
      verdict: "grounded",
    });
    store.markSessionClaimRetracted("tid-h", 0, 0, "test");
    expect(store.listSessionClaims("tid-h").length).toBe(0);
    expect(store.listSessionClaims("tid-h", { include_retracted: true }).length).toBe(1);
  });

  it("markSessionClaimSuperseded — sets old row's superseded_by_turn/claim", () => {
    store.recordSessionClaim({
      transcript_id: "tid-s",
      turn_idx: 0,
      claim_idx: 0,
      proposition: "old",
      entity: "e",
      stance: "ASSERT",
      verdict: "grounded",
    });
    store.recordSessionClaim({
      transcript_id: "tid-s",
      turn_idx: 1,
      claim_idx: 0,
      proposition: "new",
      entity: "e",
      stance: "ASSERT",
      verdict: "grounded",
    });
    expect(store.markSessionClaimSuperseded("tid-s", 0, 0, 1, 0)).toBe(true);
    const old = store.getSessionClaim("tid-s", 0, 0);
    expect(old!.superseded_by_turn).toBe(1);
    expect(old!.superseded_by_claim).toBe(0);
    // only_active hides superseded
    const active = store.listSessionClaims("tid-s", { only_active: true });
    expect(active.length).toBe(1);
    expect(active[0]!.turn_idx).toBe(1);
  });

  it("findSessionContradictionCandidates — returns near-cosine prior rows, sorted desc", () => {
    store.recordSessionClaim({
      transcript_id: "tid-c",
      turn_idx: 0,
      claim_idx: 0,
      proposition: "matching prior",
      entity: "x",
      stance: "ASSERT",
      verdict: "grounded",
      embedding: new Float32Array([1, 0, 0]),
    });
    store.recordSessionClaim({
      transcript_id: "tid-c",
      turn_idx: 1,
      claim_idx: 0,
      proposition: "orthogonal prior",
      entity: "y",
      stance: "ASSERT",
      verdict: "grounded",
      embedding: new Float32Array([0, 1, 0]),
    });
    const hits = store.findSessionContradictionCandidates(
      "tid-c",
      new Float32Array([1, 0, 0]),
      { topK: 5, minCos: 0.5 },
    );
    expect(hits.length).toBe(1);
    expect(hits[0]!.row.proposition).toBe("matching prior");
    expect(hits[0]!.similarity).toBeGreaterThan(0.99);
  });

  it("findSessionContradictionCandidates — skips WORKSPACE / REFER / reclassified rows", () => {
    for (const stance of ["WORKSPACE", "REFER"] as const) {
      store.recordSessionClaim({
        transcript_id: "tid-skip",
        turn_idx: 0,
        claim_idx: 0,
        proposition: `${stance} row`,
        entity: "x",
        stance,
        verdict: "grounded",
        embedding: new Float32Array([1, 0, 0]),
      });
      store.recordSessionClaim({
        transcript_id: "tid-skip",
        turn_idx: 1,
        claim_idx: 0,
        proposition: "reclassified row",
        entity: "x",
        stance: "ASSERT",
        verdict: "reclassified",
        embedding: new Float32Array([1, 0, 0]),
      });
      const hits = store.findSessionContradictionCandidates(
        "tid-skip",
        new Float32Array([1, 0, 0]),
      );
      expect(hits.length).toBe(0);
      store.getDb().exec("DELETE FROM session_claims WHERE transcript_id = 'tid-skip';");
    }
  });
});

// ---------------------------------------------------------------------------
// applySessionLedger — integration tests
// ---------------------------------------------------------------------------

describe("applySessionLedger — RETRACT auto-mark", () => {
  it("RETRACT pair in this turn marks matching prior ASSERT as retracted", async () => {
    const path = makeTranscript("ret-tid-1");
    const tid = "ret-tid-1";

    // Seed a prior ASSERT at turn 0.
    setEmbed("rfsrc treats status==2 as censoring", [1, 0, 0]);
    store.recordSessionClaim({
      transcript_id: tid,
      turn_idx: 0,
      claim_idx: 0,
      proposition: "rfsrc treats status==2 as censoring",
      entity: "rfsrc",
      stance: "ASSERT",
      verdict: "grounded",
      embedding: new Float32Array([1, 0, 0]),
    });

    // Current turn carries a RETRACT pair entity-matching rfsrc.
    setEmbed("Retracting earlier claim about rfsrc", [0, 1, 0]);
    const turnPairs = [
      pair({
        entity: "rfsrc",
        stance: "RETRACT",
        proposition: "Retracting earlier claim about rfsrc",
      }),
    ];

    const out = await gate.applySessionLedger(path, turnPairs);
    expect(out.contradictionFire).toBe(false);

    // Prior row should now be retracted.
    const prior = store.getSessionClaim(tid, 0, 0);
    expect(prior!.retracted).toBe(1);
    expect(prior!.reason).toContain("retracted: RETRACT in turn 1");
  });
});

describe("applySessionLedger — contradiction-fire", () => {
  it("turn-2 ASSERT contradicting turn-1 ASSERT (verifier says contradicts) fires", async () => {
    const path = makeTranscript("con-tid-1");
    const tid = "con-tid-1";

    // Seed prior ASSERT at turn 0.
    store.recordSessionClaim({
      transcript_id: tid,
      turn_idx: 0,
      claim_idx: 0,
      proposition: "rfsrc treats status==2 as censoring; comprisk does not",
      entity: "rfsrc",
      stance: "ASSERT",
      verdict: "grounded",
      embedding: new Float32Array([1, 0, 0]),
    });

    // Current-turn ASSERT — embed is near-identical so it surfaces as a
    // candidate. Verifier mock will say it contradicts.
    setEmbed(
      "both rfsrc and comprisk treat status==2 as censoring",
      [1, 0, 0],
    );
    queueContradictionVerdict(true, 0.95, "explicit reversal of asymmetry");

    const turnPairs = [
      pair({
        entity: "rfsrc",
        stance: "ASSERT",
        proposition: "both rfsrc and comprisk treat status==2 as censoring",
      }),
    ];

    const out = await gate.applySessionLedger(path, turnPairs);
    expect(out.contradictionFire).toBe(true);
    expect(out.pairs[0]!.grounded).toBe(false);
    expect(out.pairs[0]!.contradicts_session).toBeDefined();
    expect(out.pairs[0]!.contradicts_session!.old_turn).toBe(0);
    expect(out.pairs[0]!.contradicts_session!.old_claim).toBe(0);
    expect(out.pairs[0]!.reason).toContain("contradicts prior session turn 0");

    // Ledger should now have both turn 0 and turn 1 rows; turn 1 verdict is
    // "contradicted".
    const all = store.listSessionClaims(tid);
    expect(all.length).toBe(2);
    expect(all[1]!.verdict).toBe("contradicted");
  });

  it("turn-2 ASSERT that does NOT contradict (verifier returns false) passes through", async () => {
    const path = makeTranscript("ok-tid");
    const tid = "ok-tid";

    store.recordSessionClaim({
      transcript_id: tid,
      turn_idx: 0,
      claim_idx: 0,
      proposition: "rfsrc has 4 stars",
      entity: "rfsrc",
      stance: "ASSERT",
      verdict: "grounded",
      embedding: new Float32Array([1, 0, 0]),
    });

    setEmbed("rfsrc has 4 stars as of 2026-05-09", [1, 0, 0]);
    queueContradictionVerdict(false, 0.1, "refinement, not contradiction");

    const turnPairs = [
      pair({
        entity: "rfsrc",
        stance: "ASSERT",
        proposition: "rfsrc has 4 stars as of 2026-05-09",
      }),
    ];

    const out = await gate.applySessionLedger(path, turnPairs);
    expect(out.contradictionFire).toBe(false);
    expect(out.pairs[0]!.grounded).toBe(true);
  });

  it("verifier returns contradicts=true but below fire threshold → no fire", async () => {
    const path = makeTranscript("low-conf-tid");
    const tid = "low-conf-tid";

    store.recordSessionClaim({
      transcript_id: tid,
      turn_idx: 0,
      claim_idx: 0,
      proposition: "rfsrc has 4 stars",
      entity: "rfsrc",
      stance: "ASSERT",
      verdict: "grounded",
      embedding: new Float32Array([1, 0, 0]),
    });

    setEmbed("rfsrc maybe has 5 stars", [1, 0, 0]);
    queueContradictionVerdict(true, 0.4, "weak");

    const turnPairs = [
      pair({
        entity: "rfsrc",
        stance: "ASSERT",
        proposition: "rfsrc maybe has 5 stars",
      }),
    ];

    const out = await gate.applySessionLedger(path, turnPairs);
    expect(out.contradictionFire).toBe(false);
    expect(out.pairs[0]!.grounded).toBe(true);
  });

  it("contradiction-fire is scoped to same transcript_id (no cross-talk)", async () => {
    const pathA = makeTranscript("tid-A");
    store.recordSessionClaim({
      transcript_id: "tid-A",
      turn_idx: 0,
      claim_idx: 0,
      proposition: "X is Y",
      entity: "X",
      stance: "ASSERT",
      verdict: "grounded",
      embedding: new Float32Array([1, 0, 0]),
    });

    const pathB = makeTranscript("tid-B");
    setEmbed("X is not Y", [1, 0, 0]);
    // No contradiction verdict queued — if the check ran, it would crash on
    // an undefined response object. So if this test passes, the cross-talk
    // didn't happen.
    const turnPairs = [
      pair({
        entity: "X",
        stance: "ASSERT",
        proposition: "X is not Y",
      }),
    ];

    const out = await gate.applySessionLedger(pathB, turnPairs);
    expect(out.contradictionFire).toBe(false);
    expect(out.pairs[0]!.grounded).toBe(true);
  });
});

describe("applySessionLedger — ledger writes", () => {
  it("non-ASSERT pairs are persisted with their final verdict", async () => {
    const path = makeTranscript("write-tid");
    const tid = "write-tid";

    setEmbed("This setup uses gemini-3.1-pro as verifier", [1, 0, 0]);
    const turnPairs = [
      {
        entity: "gemini-3.1-pro",
        stance: "WORKSPACE",
        proposition: "This setup uses gemini-3.1-pro as verifier",
        grounded: true,
        matched_claim_id: null,
        reason: "reclassified WORKSPACE by deterministic post-filter (rule 2)",
        reclassifiedRule: 2,
      } as any,
    ];

    await gate.applySessionLedger(path, turnPairs);
    const rows = store.listSessionClaims(tid);
    expect(rows.length).toBe(1);
    expect(rows[0]!.stance).toBe("WORKSPACE");
    expect(rows[0]!.verdict).toBe("reclassified");
  });

  it("escalated-from-HEDGE pair persisted with verdict='escalated'", async () => {
    const path = makeTranscript("esc-tid");
    const tid = "esc-tid";

    setEmbed("Cole & Hernán (2008) defines the gmin clip", [1, 0, 0]);
    const turnPairs = [
      {
        entity: "Cole & Hernán (2008)",
        stance: "ASSERT",
        proposition: "Cole & Hernán (2008) defines the gmin clip",
        grounded: false,
        matched_claim_id: null,
        reason: "no KB match",
        escalatedFromHedge: true,
      } as any,
    ];

    // No contradiction candidates exist → just persists.
    await gate.applySessionLedger(path, turnPairs);
    const rows = store.listSessionClaims(tid);
    expect(rows.length).toBe(1);
    expect(rows[0]!.verdict).toBe("escalated");
  });
});

describe("transcriptIdFromPath", () => {
  it("strips .jsonl suffix and directory", () => {
    expect(gate.transcriptIdFromPath("/foo/bar/abc-123.jsonl")).toBe("abc-123");
    expect(gate.transcriptIdFromPath("abc-123.jsonl")).toBe("abc-123");
    expect(gate.transcriptIdFromPath("/x/y/z")).toBe("z");
  });
});
