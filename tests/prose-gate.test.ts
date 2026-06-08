// Gate-LOGIC tests for proseGateV1 — deterministic, no API, runs in CI.
//
// These lock the decision machinery (tag parsing, dangling-ref fire,
// unsupported fire, supported clean, untagged clean, fail-open on verifier
// error, multi-tag) using CANNED fake NLI verifiers. They do NOT assert that a
// real model judges entailment correctly on the corpus fixtures — that is the
// live eval's job (bench/prose-gate/eval.ts, real Anthropic). Keeping the two
// separate avoids a fake-green: a proxy NLI here would only test the proxy.

import { describe, expect, test } from "bun:test";
import type { NliVerdict, NliVerify } from "../src/core/nli.ts";
import { parseTaggedClaims, proseGateV1 } from "../src/core/prose-gate.ts";

const supported: NliVerify = async () => ({ supported: true, score: 0.9, reason: "entails" });
const unsupported: NliVerify = async () => ({
  supported: false,
  score: 0.9,
  reason: "number not in source",
});
const thrower = (msg: string): NliVerify => {
  return async () => {
    throw new Error(msg);
  };
};
/** records every (claim, source) it is asked to judge */
function recordingNli(verdict: NliVerdict): { verify: NliVerify; calls: Array<[string, string]> } {
  const calls: Array<[string, string]> = [];
  const verify: NliVerify = async (claim, source) => {
    calls.push([claim, source]);
    return verdict;
  };
  return { verify, calls };
}

describe("parseTaggedClaims", () => {
  test("extracts claim text + id, tag stripped", () => {
    const c = parseTaggedClaims("Full suite green — 209 pass / 0 fail. [ev: ev_run]");
    expect(c).toHaveLength(1);
    expect(c[0]?.evidenceId).toBe("ev_run");
    expect(c[0]?.claim).toBe("Full suite green — 209 pass / 0 fail.");
  });

  test("untagged prose yields no claims", () => {
    expect(parseTaggedClaims("The A/B ran on sonnet, no doubt about it.")).toEqual([]);
  });

  test("multiple tags each capture their preceding segment", () => {
    const c = parseTaggedClaims("A is true. [ev: e1] B is also true. [ev: e2]");
    expect(c.map((x) => x.evidenceId)).toEqual(["e1", "e2"]);
    expect(c[1]?.claim).toBe("B is also true.");
  });

  test("empty segment before a tag is skipped", () => {
    expect(parseTaggedClaims("[ev: e1] real claim [ev: e2]")).toHaveLength(1);
  });
});

describe("proseGateV1 decision logic", () => {
  test("tagged claim whose evidence does NOT support it → fires", async () => {
    const v = await proseGateV1(
      { draft: "209 pass. [ev: ev_run]", evidence: { ev_run: "208 pass" } },
      unsupported,
    );
    expect(v.fires).toBe(true);
    expect(v.reasons[0]).toContain("ev_run");
  });

  test("tagged claim whose evidence supports it → clean", async () => {
    const v = await proseGateV1(
      { draft: "208 pass. [ev: ev_run]", evidence: { ev_run: "208 pass" } },
      supported,
    );
    expect(v.fires).toBe(false);
    expect(v.reasons).toEqual([]);
  });

  test("dangling/fabricated evidence id → fires WITHOUT calling NLI", async () => {
    const { verify, calls } = recordingNli({ supported: true, score: 1, reason: "" });
    const v = await proseGateV1({ draft: "claim. [ev: ev_missing]", evidence: {} }, verify);
    expect(v.fires).toBe(true);
    expect(v.reasons[0]).toContain("not in the evidence set");
    expect(calls).toHaveLength(0); // no NLI round-trip for a dangling ref
  });

  test("untagged draft → clean, NLI never called", async () => {
    const { verify, calls } = recordingNli({ supported: false, score: 1, reason: "" });
    const v = await proseGateV1({ draft: "bare unverified claim", evidence: {} }, verify);
    expect(v.fires).toBe(false);
    expect(calls).toHaveLength(0);
  });

  test("transient verifier error → fail-open (no fire), note recorded", async () => {
    const v = await proseGateV1(
      { draft: "claim. [ev: e1]", evidence: { e1: "src" } },
      thrower("HTTP 429 rate_limit"),
    );
    expect(v.fires).toBe(false);
    expect(v.notes[0]).toContain("transient");
  });

  test("content verifier error → fail-open (no fire), note recorded", async () => {
    const v = await proseGateV1(
      { draft: "claim. [ev: e1]", evidence: { e1: "src" } },
      thrower("malformed JSON from model"),
    );
    expect(v.fires).toBe(false);
    expect(v.notes[0]).toContain("content");
  });

  test("multiple tags: fires if ANY tagged claim is unsupported", async () => {
    let n = 0;
    const mixed: NliVerify = async () => {
      n += 1;
      return n === 1
        ? { supported: true, score: 1, reason: "ok" }
        : { supported: false, score: 1, reason: "no" };
    };
    const v = await proseGateV1(
      { draft: "good. [ev: e1] bad. [ev: e2]", evidence: { e1: "x", e2: "y" } },
      mixed,
    );
    expect(v.fires).toBe(true);
    expect(v.reasons).toHaveLength(1);
    expect(v.reasons[0]).toContain("e2");
  });
});
