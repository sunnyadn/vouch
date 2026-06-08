import { describe, expect, test } from "bun:test";
import type { ExtractOwnWork, OwnWorkClaim } from "../src/core/extractor.ts";
import { looksLikeOwnWorkClaim, proseGateV2 } from "../src/core/prose-gate.ts";

// fake extractors (the LLM seam) — deterministic
const extractNone: ExtractOwnWork = async () => [];
function extractFixed(...claims: OwnWorkClaim[]): ExtractOwnWork {
  return async () => claims;
}
function recording(): { fn: ExtractOwnWork; calls: number } {
  const box: { fn: ExtractOwnWork; calls: number } = {
    calls: 0,
    fn: async () => {
      box.calls++;
      return [];
    },
  };
  return box;
}

describe("looksLikeOwnWorkClaim (cheap pre-filter)", () => {
  test("fires on test counts", () => {
    expect(looksLikeOwnWorkClaim("all 251 tests pass")).toBe(true);
    expect(looksLikeOwnWorkClaim("251 pass / 0 fail")).toBe(true);
  });
  test("fires on build/lint/tsc results", () => {
    expect(looksLikeOwnWorkClaim("the build succeeded, biome clean")).toBe(true);
    expect(looksLikeOwnWorkClaim("0 tsc errors")).toBe(true);
  });
  test("fires on commit-hash claims", () => {
    expect(looksLikeOwnWorkClaim("committed as 692b824")).toBe(true);
  });
  test("fires on first-person result verbs", () => {
    expect(looksLikeOwnWorkClaim("I verified the suite is green")).toBe(true);
    expect(looksLikeOwnWorkClaim("it returned exit 0")).toBe(true);
  });
  test("stays silent on plain conversational text (no LLM call wasted)", () => {
    expect(looksLikeOwnWorkClaim("Which timing should we use? I haven't decided.")).toBe(false);
    expect(looksLikeOwnWorkClaim("Next I'll read the extractor then design the surface.")).toBe(
      false,
    );
  });
});

describe("proseGateV2", () => {
  test("no own-work shape -> no extractor call, clean", async () => {
    const r = recording();
    const v = await proseGateV2("just a plan, nothing asserted", r.fn);
    expect(v.fires).toBe(false);
    expect(v.untagged).toEqual([]);
    expect(r.calls).toBe(0); // pre-filter skipped the LLM entirely
  });

  test("shape present but extractor finds nothing -> clean (fail-soft)", async () => {
    const v = await proseGateV2("all 251 tests pass", extractNone);
    expect(v.fires).toBe(false);
    expect(v.untagged).toEqual([]);
  });

  test("untagged own-work claim -> fires", async () => {
    const v = await proseGateV2(
      "all 251 tests pass",
      extractFixed({ claim: "all 251 tests pass", kind: "test-result" }),
    );
    expect(v.fires).toBe(true);
    expect(v.untagged).toHaveLength(1);
    expect(v.untagged[0]?.kind).toBe("test-result");
  });

  test("claim already covered by an [ev:] tag -> not double-flagged", async () => {
    const v = await proseGateV2(
      "all 251 tests pass [ev: ev_run]",
      extractFixed({ claim: "all 251 tests pass", kind: "test-result" }),
    );
    expect(v.fires).toBe(false);
    expect(v.untagged).toEqual([]);
  });

  test("mixed: one tagged claim handled by v1, one untagged surfaced by v2", async () => {
    const draft = "all 251 tests pass [ev: ev_run] and the build succeeded";
    const v = await proseGateV2(
      draft,
      extractFixed(
        { claim: "all 251 tests pass", kind: "test-result" },
        { claim: "the build succeeded", kind: "build-result" },
      ),
    );
    expect(v.fires).toBe(true);
    expect(v.untagged).toHaveLength(1);
    expect(v.untagged[0]?.claim).toBe("the build succeeded");
  });

  test("the dominant real fabrication (false-green) fires", async () => {
    // committed-false-green from the corpus, untagged
    const v = await proseGateV2(
      "251 full-suite pass / 0 fail, 0 new tsc errors — committing now.",
      extractFixed(
        { claim: "251 full-suite pass / 0 fail", kind: "test-result" },
        { claim: "0 new tsc errors", kind: "build-result" },
      ),
    );
    expect(v.fires).toBe(true);
    expect(v.untagged.length).toBeGreaterThanOrEqual(1);
  });
});
