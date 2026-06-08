import { describe, expect, test } from "bun:test";
import {
  type ContradictionJudge,
  type ContradictionVerdict,
  contradictionGate,
  deterministicCountContradiction,
  matchClaimToRun,
  parseRunPassCount,
  parseTestPassClaims,
  type RunRow,
} from "../src/core/contradiction.ts";

const runs = (...cmds: [string, string][]): RunRow[] =>
  cmds.map(([command, stdout]) => ({ command, stdout }));

describe("deterministic count contradiction (the airtight path)", () => {
  test("parseTestPassClaims: only counts bound to tests+pass", () => {
    expect(parseTestPassClaims("7 tests pass")).toEqual([7]);
    expect(parseTestPassClaims("7 passing tests")).toEqual([7]);
    expect(parseTestPassClaims("7 tests are passing")).toEqual([7]);
    expect(parseTestPassClaims("suite 7/9 pass")).toEqual([7]);
    // conservative: NOT bound to "tests" / "pass" → empty (defer to the LLM)
    expect(parseTestPassClaims("all 7 passing")).toEqual([]); // no "tests"
    expect(parseTestPassClaims("added 7 tests")).toEqual([]);
    expect(parseTestPassClaims("fixed 7 bugs")).toEqual([]);
    expect(parseTestPassClaims("bump to v7")).toEqual([]);
  });
  test("review false-block #1/#3: no wildcard bridge", () => {
    expect(parseTestPassClaims("fix: 7 tests still failing, rest green — WIP")).toEqual([]);
    expect(parseTestPassClaims("test: added 12 tests; the 264-test suite is green")).toEqual([]);
  });
  test("review false-block #2: a non-test leading number can't hijack", () => {
    // only the test-qualified "264 tests pass" is captured, not "2 passing"
    expect(parseTestPassClaims("2 passing runs later, all 264 tests pass")).toEqual([264]);
  });
  test("parseRunPassCount: MAX (full suite), never a trailing partial, never fail", () => {
    expect(parseRunPassCount("3 pass\n0 fail")).toBe(3);
    expect(parseRunPassCount("264 passed")).toBe(264);
    expect(parseRunPassCount(" 250 pass\n 1 fail")).toBe(250);
    // review false-block #4: a compound run's trailing partial must NOT win
    expect(parseRunPassCount("264 pass\n0 fail\n3 pass\n0 fail")).toBe(264);
    expect(parseRunPassCount("nothing here")).toBeNull();
  });
  test("fires only when EVERY claim differs from the actual", () => {
    expect(deterministicCountContradiction("7 tests passing", "3 pass\n0 fail")).toEqual({
      contradicted: true,
      claimed: 7,
      actual: 3,
    });
    expect(deterministicCountContradiction("3 tests passing", "3 pass")).toEqual({
      contradicted: false,
      claimed: 3,
      actual: 3,
    });
    // #2/#3: a message whose count DOES match the run never blocks
    expect(
      deterministicCountContradiction("2 passing runs later, all 264 tests pass", "264 pass"),
    ).toEqual({ contradicted: false, claimed: 264, actual: 264 });
    // #4: full-suite claim vs a compound run whose partial tail is smaller
    expect(
      deterministicCountContradiction("all 264 tests pass", "264 pass\n0 fail\n3 pass\n0 fail"),
    ).toEqual({ contradicted: false, claimed: 264, actual: 264 });
    // #1: an honest WIP message with no pass-claim defers (null), never blocks
    expect(
      deterministicCountContradiction("7 tests still failing, rest green", "257 pass"),
    ).toBeNull();
    expect(deterministicCountContradiction("feat: add widget", "3 pass")).toBeNull();
    expect(deterministicCountContradiction("7 tests passing", "build ok")).toBeNull(); // no actual
  });
});

const judge = (label: ContradictionVerdict["label"]): ContradictionJudge => {
  return async () => ({ label, score: 0.9, reason: label });
};
const throwingJudge: ContradictionJudge = async () => {
  throw new Error("infra down");
};

describe("matchClaimToRun (deterministic command-kind filter)", () => {
  test("test-result matches a test command, NOT an unrelated git/ls row", () => {
    const rs = runs(["git status", "On branch main"], ["bun test", "264 pass"], ["ls -la", "x"]);
    expect(matchClaimToRun("test-result", rs)?.command).toBe("bun test");
  });

  test("test-result does NOT match when only unrelated rows exist (no false pair)", () => {
    const rs = runs(["git status", "On branch main"], ["ls -la", "files"]);
    expect(matchClaimToRun("test-result", rs)).toBeNull();
  });

  test("build-result matches tsc / biome", () => {
    expect(matchClaimToRun("build-result", runs(["bunx tsc --noEmit", "0 errors"]))?.command).toBe(
      "bunx tsc --noEmit",
    );
    expect(matchClaimToRun("build-result", runs(["biome check", "clean"]))?.command).toBe(
      "biome check",
    );
  });

  test("git-fact matches a git command", () => {
    expect(matchClaimToRun("git-fact", runs(["git rm --cached x", "did not match"]))?.command).toBe(
      "git rm --cached x",
    );
  });

  test("runtime-fact / other-ownwork have no deterministic kind -> null (degrade to advise)", () => {
    const rs = runs(["curl https://x", "HTTP 429"]);
    expect(matchClaimToRun("runtime-fact", rs)).toBeNull();
    expect(matchClaimToRun("other-ownwork", rs)).toBeNull();
  });

  test("empty runs -> null", () => {
    expect(matchClaimToRun("test-result", [])).toBeNull();
  });

  test("most-recent matching run wins", () => {
    const rs = runs(["bun test", "1 fail"], ["bun test", "0 fail"]);
    expect(matchClaimToRun("test-result", rs)?.stdout).toBe("0 fail");
  });
});

describe("contradictionGate (fires ONLY on matched + contradicted)", () => {
  test("matched run + contradicted -> FIRES (would block)", async () => {
    const r = await contradictionGate(
      "264 tests pass",
      "test-result",
      runs(["bun test", "2 failed"]),
      judge("contradicted"),
    );
    expect(r.fires).toBe(true);
    expect(r.label).toBe("contradicted");
    expect(r.matchedCommand).toBe("bun test");
  });

  test("matched run + supported (truthful) -> does NOT fire", async () => {
    const r = await contradictionGate(
      "264 tests pass",
      "test-result",
      runs(["bun test", "264 pass 0 fail"]),
      judge("supported"),
    );
    expect(r.fires).toBe(false);
    expect(r.label).toBe("supported");
  });

  test("matched run + neutral (insufficient) -> does NOT fire", async () => {
    const r = await contradictionGate(
      "264 tests pass",
      "test-result",
      runs(["bun test", "^C interrupted"]),
      judge("neutral"),
    );
    expect(r.fires).toBe(false);
    expect(r.label).toBe("neutral");
  });

  test("NO matching run -> does NOT fire (advise, never block)", async () => {
    // judge would say 'contradicted' but it must never be consulted: no run of the kind
    const r = await contradictionGate(
      "all tests pass",
      "test-result",
      runs(["git status", "clean"]),
      judge("contradicted"),
    );
    expect(r.fires).toBe(false);
    expect(r.label).toBe("no-match");
    expect(r.matchedCommand).toBeUndefined();
  });

  test("judge error -> fail-open (no block)", async () => {
    const r = await contradictionGate(
      "264 tests pass",
      "test-result",
      runs(["bun test", "2 failed"]),
      throwingJudge,
    );
    expect(r.fires).toBe(false);
    expect(r.label).toBe("neutral");
  });

  test("the unrelated-row trap: a test claim is never blocked by a git row", async () => {
    // even with a 'contradicted'-returning judge, the matcher refuses to pair a
    // test claim with a git row, so no block — the contradicted!=unsupported guard.
    const r = await contradictionGate(
      "264 tests pass",
      "test-result",
      runs(["git status", "On branch main, nothing to commit"]),
      judge("contradicted"),
    );
    expect(r.fires).toBe(false);
    expect(r.label).toBe("no-match");
  });
});
