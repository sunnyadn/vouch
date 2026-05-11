/** Pure-store tests (no LLM calls). Verifies SQLite operations end-to-end. */
import { afterEach, beforeAll, describe, expect, it } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

// Set DB path BEFORE importing store, since DB_PATH is read at module load.
const tmp = mkdtempSync(join(tmpdir(), "vouch-test-"));
process.env.VOUCH_DB_PATH = join(tmp, "test.db");

const store = await import("../src/store.ts");

afterEach(() => {
  // Clean state between tests by truncating tables (keeps schema)
  const db = store.getDb();
  db.exec("DELETE FROM claim_dependencies; DELETE FROM claims; DELETE FROM dossiers;");
});

describe("store", () => {
  it("writes + reads a dossier", () => {
    const slug = store.writeDossier({
      source_url: "https://example.com/x",
      source_type: "web-submitted",
      title: "Test page",
      verbatim_content: "The answer is 42.",
    });
    expect(slug).toContain("evidence/web-submitted/");
    const d = store.getDossier(slug);
    expect(d?.title).toBe("Test page");
    expect(d?.content).toBe("The answer is 42.");
  });

  it("URL lookup tolerates http/https + arxiv version", () => {
    store.writeDossier({
      source_url: "https://arxiv.org/abs/2410.05779",
      source_type: "arxiv",
      title: "LightRAG",
      verbatim_content: "Some content.",
    });
    expect(store.getDossierByUrl("https://arxiv.org/abs/2410.05779v3")?.title).toBe("LightRAG");
    expect(store.getDossierByUrl("http://arxiv.org/abs/2410.05779")?.title).toBe("LightRAG");
  });

  it("records ATOMIC claim with offsets + attribution", () => {
    const slug = store.writeDossier({
      source_url: "https://x.test",
      source_type: "web-submitted",
      title: "T",
      verbatim_content: "Quote here.",
    });
    const id = store.recordClaim({
      dossier_slug: slug,
      claim_text: "Test claim",
      score: 0.9,
      status: "supported",
      claim_type: "ATOMIC",
      topic: "test",
      author: "tester",
      attribution: "Anon",
      source_offset_start: 0,
      source_offset_end: 11,
    });
    const c = store.getClaim(id);
    expect(c?.status).toBe("supported");
    expect(c?.source_offset_end).toBe(11);
    expect(c?.attribution).toBe("Anon");
  });

  it("walks dependency DAG", () => {
    const slug = store.writeDossier({
      source_url: "https://y.test",
      source_type: "web-submitted",
      verbatim_content: "x",
    });
    const a = store.recordClaim({
      dossier_slug: slug,
      claim_text: "atomic",
      score: 1,
      status: "supported",
      claim_type: "ATOMIC",
    });
    const b = store.recordClaim({
      dossier_slug: "",
      claim_text: "inference",
      score: 1,
      status: "supported",
      claim_type: "INFERENCE",
      depends_on_ids: [a],
    });
    const chain = store.getClaimChain(b);
    expect(chain.node_count).toBe(2);
    expect(chain.edges).toHaveLength(1);
    expect(chain.edges[0]).toMatchObject({ from: b, to: a, type: "inference" });
  });

  it("supersede preserves audit trail", () => {
    const old = store.recordClaim({
      dossier_slug: "",
      claim_text: "old",
      score: 1,
      status: "supported",
      claim_type: "HYPOTHESIS",
    });
    const fresh = store.recordClaim({
      dossier_slug: "",
      claim_text: "new",
      score: 1,
      status: "supported",
      claim_type: "HYPOTHESIS",
    });
    expect(store.supersedeClaim(old, fresh, "test reason")).toBe(true);
    const c = store.getClaim(old);
    expect(c?.superseded_by).toBe(fresh);
    expect(c?.supersede_reason).toBe("test reason");
  });

  it("supersede rejects non-existent IDs", () => {
    expect(store.supersedeClaim(9999, 9998, "x")).toBe(false);
  });

  it("listClaims filters by topic + status", () => {
    store.recordClaim({
      dossier_slug: "",
      claim_text: "a",
      score: 1,
      status: "supported",
      topic: "X",
    });
    store.recordClaim({
      dossier_slug: "",
      claim_text: "b",
      score: 0,
      status: "unsupported",
      topic: "X",
    });
    store.recordClaim({
      dossier_slug: "",
      claim_text: "c",
      score: 1,
      status: "supported",
      topic: "Y",
    });
    expect(store.listClaims({ topic: "X" })).toHaveLength(2);
    expect(store.listClaims({ topic: "X", status: "supported" })).toHaveLength(1);
    expect(store.listClaims({ contains: "b" })).toHaveLength(1);
  });

  it("listTopics aggregates counts", () => {
    store.recordClaim({
      dossier_slug: "",
      claim_text: "a",
      score: 1,
      status: "supported",
      topic: "X",
    });
    store.recordClaim({
      dossier_slug: "",
      claim_text: "b",
      score: 0,
      status: "unsupported",
      topic: "X",
    });
    const topics = store.listTopics();
    const x = topics.find((t) => t.topic === "X")!;
    expect(x.n_claims).toBe(2);
    expect(x.n_supported).toBe(1);
  });

  it("listClaims filters by author, verification, depends_on, newest-first", () => {
    const slug = store.writeDossier({ source_url: "https://f.test", source_type: "web-submitted", verbatim_content: "foo" });
    const c1 = store.recordClaim({ dossier_slug: slug, claim_text: "alpha", score: 1, status: "supported", claim_type: "ATOMIC", author: "claude-skill", verification: "nli-quote" });
    const c2 = store.recordClaim({ dossier_slug: slug, claim_text: "beta", score: 1, status: "supported", claim_type: "INFERENCE", author: "gate-harvest", verification: "tag-harvest", depends_on_ids: [c1] });
    store.recordClaim({ dossier_slug: slug, claim_text: "gamma", score: 0, status: "unsupported", claim_type: "ATOMIC", author: "claude-skill", verification: "nli-quote" });

    expect(store.listClaims({ author: "claude-skill" })).toHaveLength(2);
    expect(store.listClaims({ verification: "tag-harvest" })).toHaveLength(1);
    expect(store.listClaims({ depends_on_id: c1 })).toHaveLength(1);
    expect(store.listClaims({ depends_on_id: c1 })[0]!.id).toBe(c2);

    // newest-first orders by verified_at DESC, id DESC tie-breaker
    const newestFirst = store.listClaims({ newestFirst: true, limit: 10 });
    expect(newestFirst[0]!.claim_text).toBe("gamma");
    expect(newestFirst[newestFirst.length - 1]!.claim_text).toBe("alpha");
  });

  it("listClaims filters by since", () => {
    const slug = store.writeDossier({ source_url: "https://since.test", source_type: "web-submitted", verbatim_content: "x" });
    const before = new Date().toISOString();
    store.recordClaim({ dossier_slug: slug, claim_text: "old", score: 1, status: "supported" });
    store.recordClaim({ dossier_slug: slug, claim_text: "new1", score: 1, status: "supported" });
    store.recordClaim({ dossier_slug: slug, claim_text: "new2", score: 1, status: "supported" });
    const results = store.listClaims({ since: before });
    expect(results).toHaveLength(3);
    expect(results.map((c) => c.claim_text).sort()).toEqual(["new1", "new2", "old"]);
  });

  it("listClaims composes multiple filters", () => {
    const slug = store.writeDossier({ source_url: "https://comp.test", source_type: "web-submitted", verbatim_content: "x" });
    store.recordClaim({ dossier_slug: slug, claim_text: "a", score: 1, status: "supported", claim_type: "ATOMIC", author: "claude-skill", verification: "nli-quote" });
    store.recordClaim({ dossier_slug: slug, claim_text: "b", score: 1, status: "supported", claim_type: "INFERENCE", author: "gate-harvest", verification: "tag-harvest" });
    store.recordClaim({ dossier_slug: slug, claim_text: "c", score: 1, status: "supported", claim_type: "ATOMIC", author: "gate-harvest", verification: "tag-harvest" });
    const results = store.listClaims({ author: "gate-harvest", claim_type: "INFERENCE", verification: "tag-harvest" });
    expect(results).toHaveLength(1);
    expect(results[0]!.claim_text).toBe("b");
  });

  it("listRecentClaimsSummary returns correct aggregates", () => {
    const slug = store.writeDossier({ source_url: "https://r.test", source_type: "web-submitted", verbatim_content: "x" });
    const sinceIso = new Date().toISOString();
    store.recordClaim({ dossier_slug: slug, claim_text: "s1", score: 1, status: "supported", author: "claude-skill" });
    store.recordClaim({ dossier_slug: slug, claim_text: "s2", score: 0, status: "unsupported", author: "claude-skill" });
    store.recordClaim({ dossier_slug: "", claim_text: "s3", score: null, status: "recorded", author: "gate-harvest" });
    const summary = store.listRecentClaimsSummary(sinceIso);
    expect(summary.total).toBe(3);
    expect(summary.supported).toBe(1);
    expect(summary.unsupported).toBe(1);
    expect(summary.recorded).toBe(1);
    expect(summary.dossiers).toBe(1);
    expect(summary.authorBreakdown["claude-skill"]).toBe(2);
    expect(summary.authorBreakdown["gate-harvest"]).toBe(1);
  });

  it("hybrid search finds closest claim/dossier by cosine", () => {
    // Hand-craft normalized 4D embeddings: query is closer to dossier1
    const e = (xs: number[]) => {
      const norm = Math.sqrt(xs.reduce((s, x) => s + x * x, 0));
      return new Float32Array(xs.map((x) => x / norm));
    };
    store.writeDossier({
      source_url: "https://a",
      source_type: "x",
      title: "doss1",
      verbatim_content: "alpha content",
      embedding: e([1, 0, 0, 0]),
    });
    store.writeDossier({
      source_url: "https://b",
      source_type: "x",
      title: "doss2",
      verbatim_content: "beta content",
      embedding: e([0, 1, 0, 0]),
    });
    store.recordClaim({
      dossier_slug: "",
      claim_text: "claim alpha",
      score: 1,
      status: "supported",
      embedding: e([0.9, 0.1, 0, 0]),
    });
    const hits = store.searchHybrid(e([1, 0, 0, 0]), 5);
    expect(hits[0]!.kind === "dossier" || hits[0]!.kind === "claim").toBe(true);
    expect(hits[0]!.text).toContain("alpha");
  });
});

// Cleanup tmp dir on exit (one-time, no per-test churn beyond truncation)
beforeAll(() => {
  process.on("exit", () => {
    try {
      rmSync(tmp, { recursive: true, force: true });
    } catch {}
  });
});
