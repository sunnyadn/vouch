import { describe, expect, test } from "bun:test";
import { extractLastAssistantText } from "../src/core/prose-stop.ts";

// JSONL transcript line builder
function asst(text: string): string {
  return JSON.stringify({
    type: "assistant",
    message: { role: "assistant", content: [{ type: "text", text }] },
  });
}
function user(text: string): string {
  return JSON.stringify({
    type: "user",
    message: { role: "user", content: [{ type: "text", text }] },
  });
}
function asstToolUse(name: string): string {
  return JSON.stringify({
    type: "assistant",
    message: { role: "assistant", content: [{ type: "tool_use", name, input: {} }] },
  });
}

describe("extractLastAssistantText", () => {
  test("returns the last assistant text turn", () => {
    const jsonl = [asst("first"), user("a question"), asst("second [ev: ev_x]")].join("\n");
    expect(extractLastAssistantText(jsonl)).toBe("second [ev: ev_x]");
  });

  test("concatenates multiple text blocks in one assistant message", () => {
    const rec = JSON.stringify({
      message: {
        role: "assistant",
        content: [
          { type: "text", text: "a" },
          { type: "text", text: "b" },
        ],
      },
    });
    expect(extractLastAssistantText(rec)).toBe("ab");
  });

  test("a trailing tool-use-only assistant record does not blank the real draft", () => {
    const jsonl = [asst("the real draft [ev: ev_1]"), asstToolUse("Bash")].join("\n");
    expect(extractLastAssistantText(jsonl)).toBe("the real draft [ev: ev_1]");
  });

  test("blank and malformed lines are skipped, never throws", () => {
    const jsonl = ["", "   ", "{not json", asst("ok"), "garbage}"].join("\n");
    expect(extractLastAssistantText(jsonl)).toBe("ok");
  });

  test("empty transcript -> empty string", () => {
    expect(extractLastAssistantText("")).toBe("");
  });

  test("only user turns -> empty string", () => {
    expect(extractLastAssistantText([user("hi"), user("again")].join("\n"))).toBe("");
  });
});

// ---- tool-call-arg surface ---------------------------------------------------

import { extractClaimArgsFromCommand, extractToolCallClaimText } from "../src/core/prose-stop.ts";

function asstBash(command: string): string {
  return JSON.stringify({
    type: "assistant",
    message: {
      role: "assistant",
      content: [{ type: "tool_use", name: "Bash", input: { command } }],
    },
  });
}
function asstToolInput(name: string, input: Record<string, unknown>): string {
  return JSON.stringify({
    type: "assistant",
    message: { role: "assistant", content: [{ type: "tool_use", name, input }] },
  });
}
function toolResult(text: string): string {
  return JSON.stringify({
    type: "user",
    message: { role: "user", content: [{ type: "tool_result", tool_use_id: "t", content: text }] },
  });
}

describe("extractClaimArgsFromCommand", () => {
  test("git commit -m double-quoted", () => {
    expect(extractClaimArgsFromCommand('git commit -m "feat: 264 tests pass"')).toEqual([
      "feat: 264 tests pass",
    ]);
  });
  test("--note single-quoted", () => {
    expect(extractClaimArgsFromCommand("vouch branch enter x --note 'build succeeded'")).toEqual([
      "build succeeded",
    ]);
  });
  test("--statement and --summary both captured", () => {
    const out = extractClaimArgsFromCommand(
      'vouch finding add --statement "all green" --summary "251 pass"',
    );
    expect(out).toContain("all green");
    expect(out).toContain("251 pass");
  });
  test("flag=value form (bare)", () => {
    expect(extractClaimArgsFromCommand("cmd --reason=done-here")).toEqual(["done-here"]);
  });
  test('flag"value" with no space', () => {
    expect(extractClaimArgsFromCommand('git commit -m"0 tsc errors"')).toEqual(["0 tsc errors"]);
  });
  test("repeated -m captured both", () => {
    expect(extractClaimArgsFromCommand('git commit -m "a" -m "b"')).toEqual(["a", "b"]);
  });
  test("-m is NOT matched inside --message-like prefixes", () => {
    expect(extractClaimArgsFromCommand('git commit --message "once"')).toEqual(["once"]);
  });
  test("escaped quote inside a double-quoted message", () => {
    expect(extractClaimArgsFromCommand('git commit -m "say \\"hi\\""')).toEqual(['say "hi"']);
  });
  test("no claim flags -> empty", () => {
    expect(extractClaimArgsFromCommand("bun test > /tmp/x.txt 2>&1")).toEqual([]);
  });
});

describe("extractToolCallClaimText", () => {
  test("pulls the commit message out of a Bash tool call", () => {
    const jsonl = [user("ship it"), asstBash('git commit -m "feat: 264 tests pass"')].join("\n");
    expect(extractToolCallClaimText(jsonl)).toContain("feat: 264 tests pass");
  });

  test("scopes to the current turn — a prior turn's commit is dropped", () => {
    const jsonl = [
      user("turn 1"),
      asstBash('git commit -m "old: 99 tests pass"'),
      user("turn 2"),
      asstBash("ls -la"),
    ].join("\n");
    expect(extractToolCallClaimText(jsonl)).toBe("");
  });

  test("tool_result records do NOT reset the window (multi-roundtrip turn)", () => {
    const jsonl = [
      user("do the work"),
      asstBash("bun test"),
      toolResult("251 pass 0 fail"),
      asstBash('git commit -m "build succeeded"'),
    ].join("\n");
    expect(extractToolCallClaimText(jsonl)).toContain("build succeeded");
  });

  test("structured note/statement fields on a non-Bash tool are captured", () => {
    const jsonl = [user("p"), asstToolInput("SomeTool", { statement: "all green, 0 errors" })].join(
      "\n",
    );
    expect(extractToolCallClaimText(jsonl)).toContain("all green, 0 errors");
  });

  test("Edit/Write tool calls contribute nothing (no claim flags)", () => {
    const jsonl = [
      user("p"),
      asstToolInput("Edit", { file_path: "a.ts", old_string: "x", new_string: "264 tests pass" }),
    ].join("\n");
    expect(extractToolCallClaimText(jsonl)).toBe("");
  });

  test("malformed lines are skipped, never throws", () => {
    const jsonl = ["{bad", asstBash('git commit -m "ok 1 pass"'), "junk}"].join("\n");
    expect(extractToolCallClaimText(jsonl)).toContain("ok 1 pass");
  });
});

// ---- scoping regressions ----------------------------------------------------

function metaUser(text: string): string {
  return JSON.stringify({ type: "user", isMeta: true, message: { role: "user", content: text } });
}
function sidechainUser(text: string): string {
  return JSON.stringify({
    type: "user",
    isSidechain: true,
    message: { role: "user", content: text },
  });
}
function combinedUser(resultText: string, promptText: string): string {
  return JSON.stringify({
    type: "user",
    message: {
      role: "user",
      content: [
        { type: "tool_result", tool_use_id: "t", content: resultText },
        { type: "text", text: promptText },
      ],
    },
  });
}

describe("scoping: meta / sidechain records do not drop the turn", () => {
  test("an isMeta record mid-turn does NOT drop the just-made commit", () => {
    const jsonl = [
      user("do the work"),
      asstBash('git commit -m "feat: 264 tests pass"'),
      toolResult("264 pass"),
      metaUser("Stop hook feedback"),
      asst("done."),
    ].join("\n");
    expect(extractToolCallClaimText(jsonl)).toContain("feat: 264 tests pass");
  });

  test("an isSidechain record does NOT reset the window", () => {
    const jsonl = [
      user("do the work"),
      asstBash('git commit -m "feat: 264 tests pass"'),
      sidechainUser("a subagent prompt"),
      asstBash("ls"),
    ].join("\n");
    expect(extractToolCallClaimText(jsonl)).toContain("feat: 264 tests pass");
  });
});

describe("scoping: combined text+tool_result record IS a boundary", () => {
  test("a queued prompt bundled with a tool_result resets the window", () => {
    const jsonl = [
      user("turn 1"),
      asstBash('git commit -m "old: 99 tests pass"'),
      combinedUser("99 pass", "turn 2 — now do something else"),
      asstBash("ls -la"),
    ].join("\n");
    expect(extractToolCallClaimText(jsonl)).toBe("");
  });

  test("a pure tool_result array still does NOT reset (multi-roundtrip)", () => {
    const jsonl = [
      user("do it"),
      asstBash('git commit -m "build succeeded"'),
      toolResult("ok"),
      asstBash("ls"),
    ].join("\n");
    expect(extractToolCallClaimText(jsonl)).toContain("build succeeded");
  });
});

describe("command scoping: only git commit / vouch are harvested", () => {
  test("non-claim-authoring -m is NOT harvested", () => {
    const stash = [user("p"), asstBash('git stash push -m "WIP: 0 fail snapshot"')].join("\n");
    const grep = [user("p"), asstBash("grep -m 5 pattern run.log")].join("\n");
    const curl = [user("p"), asstBash('curl -m 30 https://x --data "ran 5 tests"')].join("\n");
    expect(extractToolCallClaimText(stash)).toBe("");
    expect(extractToolCallClaimText(grep)).toBe("");
    expect(extractToolCallClaimText(curl)).toBe("");
  });

  test("git commit and vouch commands ARE harvested", () => {
    const commit = [user("p"), asstBash('git commit -m "264 tests pass"')].join("\n");
    const xt = [user("p"), asstBash('vouch finding add --statement "all green"')].join("\n");
    const cli = [user("p"), asstBash('bun src/cli.ts branch enter x --note "0 fail"')].join("\n");
    expect(extractToolCallClaimText(commit)).toContain("264 tests pass");
    expect(extractToolCallClaimText(xt)).toContain("all green");
    expect(extractToolCallClaimText(cli)).toContain("0 fail");
  });
});

describe("parser hardening", () => {
  test("a flag name inside an already-quoted value is not re-scanned", () => {
    expect(extractClaimArgsFromCommand('git commit -m "fix --note handling"')).toEqual([
      "fix --note handling",
    ]);
    expect(
      extractClaimArgsFromCommand(`git commit -m "docs: why --note '99 tests pass' is risky"`),
    ).toEqual(["docs: why --note '99 tests pass' is risky"]);
  });

  test("bare value stops at a shell metacharacter", () => {
    expect(extractClaimArgsFromCommand("vouch x --reason done>/tmp/out.txt")).toEqual(["done"]);
    expect(extractClaimArgsFromCommand("vouch x --reason done|tee")).toEqual(["done"]);
  });
});

describe("cross-arg blending", () => {
  test("a count in one arg cannot bind to a result-noun in another", () => {
    const jsonl = [
      user("p"),
      asstBash("vouch finding add --summary 'bumped to 5'"),
      asstBash("vouch branch enter x --note 'tests pass'"),
    ].join("\n");
    const out = extractToolCallClaimText(jsonl);
    expect(out).toContain("bumped to 5");
    expect(out).toContain("tests pass");
    expect(/\b5\s*tests\b/.test(out)).toBe(false);
  });
});

// ---- proseCommitGate --------------------------------------------------------

import type { ContradictionJudge } from "../src/core/contradiction.ts";
import { type CommitRunsProvider, proseCommitGate } from "../src/core/prose-stop.ts";

const judgeC: ContradictionJudge = async () => ({
  label: "contradicted",
  score: 0.9,
  reason: "run shows 2 failed",
});
const judgeS: ContradictionJudge = async () => ({
  label: "supported",
  score: 0.9,
  reason: "matches",
});
const provide =
  (...rows: { command: string; stdout: string; id?: string }[]): CommitRunsProvider =>
  async () =>
    rows;

describe("proseCommitGate (PreToolUse commit BLOCK)", () => {
  test("non-commit command -> no block, extractor never called", async () => {
    let called = 0;
    const r = await proseCommitGate({
      command: "ls -la",
      extract: async () => {
        called++;
        return [];
      },
      judge: judgeC,
      runsProvider: provide({ command: "bun test", stdout: "2 failed" }),
    });
    expect(r.blocks).toBe(false);
    expect(called).toBe(0);
  });

  test("commit with no own-work shape -> no block, extractor never called", async () => {
    let called = 0;
    const r = await proseCommitGate({
      command: 'git commit -m "feat: add widget loader"',
      extract: async () => {
        called++;
        return [];
      },
      judge: judgeC,
      runsProvider: provide({ command: "bun test", stdout: "2 failed" }),
    });
    expect(r.blocks).toBe(false);
    expect(called).toBe(0);
  });

  test("a contradicted claim -> BLOCKS and names the contradicting evidence", async () => {
    const r = await proseCommitGate({
      command: 'git commit -m "feat: 264 tests pass"',
      extract: async () => [{ claim: "264 tests pass", kind: "test-result" as const }],
      judge: judgeC,
      runsProvider: provide({ command: "bun test", stdout: "2 failed", id: "ev_run1" }),
    });
    expect(r.blocks).toBe(true);
    expect(r.fired[0]?.evidenceId).toBe("ev_run1");
    expect(r.message).toContain("BLOCK");
    expect(r.message).toContain("ev_run1");
  });

  test("truthful claim (recorded run supports it) -> no block", async () => {
    const r = await proseCommitGate({
      command: 'git commit -m "feat: 264 tests pass"',
      extract: async () => [{ claim: "264 tests pass", kind: "test-result" as const }],
      judge: judgeS,
      runsProvider: provide({ command: "bun test", stdout: "264 pass 0 fail", id: "ev_run1" }),
    });
    expect(r.blocks).toBe(false);
  });

  test("no recorded run of the kind -> no block", async () => {
    let consulted = 0;
    const r = await proseCommitGate({
      command: 'git commit -m "all 264 tests pass"',
      extract: async () => [{ claim: "264 tests pass", kind: "test-result" as const }],
      judge: async () => {
        consulted++;
        return { label: "contradicted", score: 1, reason: "x" };
      },
      runsProvider: provide(),
    });
    expect(r.blocks).toBe(false);
    expect(consulted).toBe(0);
  });

  test("only an unrelated run -> no block", async () => {
    const r = await proseCommitGate({
      command: 'git commit -m "264 tests pass"',
      extract: async () => [{ claim: "264 tests pass", kind: "test-result" as const }],
      judge: judgeC,
      runsProvider: provide({ command: "git status", stdout: "clean", id: "ev_g" }),
    });
    expect(r.blocks).toBe(false);
  });

  test("extractor returns nothing -> no block", async () => {
    const r = await proseCommitGate({
      command: 'git commit -m "264 tests pass"',
      extract: async () => [],
      judge: judgeC,
      runsProvider: provide({ command: "bun test", stdout: "2 failed" }),
    });
    expect(r.blocks).toBe(false);
  });

  test("DETERMINISTIC: a contradicted count blocks WITHOUT consulting the judge", async () => {
    let judged = 0;
    const r = await proseCommitGate({
      command: 'git commit -m "feat: 7 tests passing"',
      extract: async () => [],
      judge: async () => {
        judged++;
        return { label: "neutral", score: 1, reason: "must not be consulted" };
      },
      runsProvider: provide({ command: "bun test", stdout: "3 pass\n0 fail", id: "ev_run" }),
    });
    expect(r.blocks).toBe(true);
    expect(judged).toBe(0);
    expect(r.fired[0]?.reason).toContain("deterministic");
    expect(r.fired[0]?.evidenceId).toBe("ev_run");
  });

  test("DETERMINISTIC: a matching count does NOT false-fire", async () => {
    let judged = 0;
    const r = await proseCommitGate({
      command: 'git commit -m "feat: 3 tests passing"',
      extract: async () => [],
      judge: async () => {
        judged++;
        return { label: "contradicted", score: 1, reason: "x" };
      },
      runsProvider: provide({ command: "bun test", stdout: "3 pass\n0 fail", id: "ev_run" }),
    });
    expect(r.blocks).toBe(false);
    expect(judged).toBe(0);
  });
});
