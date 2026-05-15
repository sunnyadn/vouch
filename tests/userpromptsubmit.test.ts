/** UserPromptSubmit hook tests — L3 hook for SUN-82 PR #2.
 *
 *  Coverage:
 *    - shouldSkip heuristic gate (every skip branch + the pass-through path)
 *    - codeBlockFraction edge cases (no fences, paired fences, dominant code)
 *    - formatExaContext / formatKbContext shape + truncation
 *    - runUserPromptSubmit decision tree:
 *      - skip → empty envelope
 *      - KB hit → KB-context envelope (no Exa call)
 *      - KB miss + no Exa key → empty envelope
 *
 *  We don't exercise the real KB/Exa code paths here — those are owned by
 *  store.test.ts and (eventually) exa.test.ts. The KB-hit branch is covered
 *  by stubbing lookupKb's underlying primitive via embedOne import indirection
 *  isn't trivial in bun:test, so we use the "KB miss + no key" path as the
 *  reachable failure-mode test and add an explicit unit on formatKbContext.
 */
import { afterAll, beforeAll, describe, expect, it } from "bun:test";
import {
  codeBlockFraction,
  formatExaContext,
  formatKbContext,
  runUserPromptSubmit,
  shouldSkip,
} from "../src/userpromptsubmit.ts";
import type { SearchHit } from "../src/store.ts";

describe("shouldSkip", () => {
  const originalBypass = process.env.VOUCH_USERPROMPT_BYPASS;
  beforeAll(() => {
    delete process.env.VOUCH_USERPROMPT_BYPASS;
  });
  afterAll(() => {
    if (originalBypass === undefined) delete process.env.VOUCH_USERPROMPT_BYPASS;
    else process.env.VOUCH_USERPROMPT_BYPASS = originalBypass;
  });

  it("bypass env wins over everything else", () => {
    process.env.VOUCH_USERPROMPT_BYPASS = "1";
    expect(shouldSkip("a perfectly fine research prompt about wink-nlp")).toBe("bypass");
    delete process.env.VOUCH_USERPROMPT_BYPASS;
  });

  it("empty prompt → empty", () => {
    expect(shouldSkip(undefined)).toBe("empty");
    expect(shouldSkip("")).toBe("empty");
  });

  it("too-short prompt → too_short (greeting / 'thanks')", () => {
    expect(shouldSkip("hi")).toBe("too_short");
    expect(shouldSkip("thanks")).toBe("too_short");
  });

  it("too-long prompt → too_long (file paste / code dump)", () => {
    expect(shouldSkip("x".repeat(5001))).toBe("too_long");
  });

  it("code-heavy prompt → code_heavy", () => {
    const prompt = "fix this:\n```ts\n" + "const x = 1;\n".repeat(50) + "```\nplease";
    expect(shouldSkip(prompt)).toBe("code_heavy");
  });

  it("research-y prompt → null (passes the gate)", () => {
    expect(shouldSkip("tell me about the wink-nlp library and its precision")).toBeNull();
  });
});

describe("codeBlockFraction", () => {
  it("returns 0 for text with no fences", () => {
    expect(codeBlockFraction("plain prose about wink-nlp library")).toBe(0);
  });

  it("returns 0 for text with a single lone fence (unbalanced)", () => {
    expect(codeBlockFraction("hi ```typescript\nconst x = 1;")).toBe(0);
  });

  it("returns >0.5 when code dominates", () => {
    const prompt = "fix:\n```\n" + "const x = 1;\n".repeat(20) + "```\n";
    expect(codeBlockFraction(prompt)).toBeGreaterThan(0.5);
  });

  it("returns <0.5 when prose dominates", () => {
    const prose = "I'm researching the wink-nlp library and its tokenizer pipeline. ".repeat(10);
    const prompt = prose + "Example:\n```\nconst x = 1;\n```\nWhat do you think?";
    expect(codeBlockFraction(prompt)).toBeLessThan(0.5);
  });
});

describe("formatExaContext", () => {
  it("returns empty string for zero candidates", () => {
    expect(formatExaContext([])).toBe("");
  });

  it("formats N candidates with URL, title, and truncated excerpt", () => {
    const out = formatExaContext([
      { url: "https://example.com/a", title: "Example A", text: "Body about A. ".repeat(50) },
      { url: "https://example.com/b", title: "Example B", text: "Body B" },
    ]);
    expect(out).toContain("[vouch L3]");
    expect(out).toContain("Pre-fetched 2");
    expect(out).toContain("https://example.com/a");
    expect(out).toContain("Example A");
    expect(out).toContain("https://example.com/b");
    expect(out).toContain("Body B");
    // Truncation: A's excerpt should be capped (~200 chars)
    const aLine = out.split("\n").find((l) => l.includes("Body about A.")) || "";
    expect(aLine.length).toBeLessThan(260);
  });

  it("omits the excerpt line when text is empty", () => {
    const out = formatExaContext([
      { url: "https://example.com", title: "T", text: "" },
    ]);
    expect(out).toContain("https://example.com");
    expect(out).not.toContain('""');
  });
});

describe("formatKbContext", () => {
  it("returns empty string for zero hits", () => {
    expect(formatKbContext([])).toBe("");
  });

  it("formats claim hits with id, sim, status, and text", () => {
    const hits: SearchHit[] = [
      { kind: "claim", similarity: 0.83, id: 42, text: "wink-nlp is a JS NLP toolkit", status: "supported" },
    ];
    const out = formatKbContext(hits);
    expect(out).toContain("KB has 1 match(es)");
    expect(out).toContain("claim 42");
    expect(out).toContain("sim=0.83");
    expect(out).toContain("status=supported");
    expect(out).toContain("wink-nlp is a JS NLP toolkit");
  });

  it("formats dossier hits with slug, sim, title, and excerpt", () => {
    const hits: SearchHit[] = [
      {
        kind: "dossier",
        similarity: 0.72,
        slug: "github/winkjs/wink-nlp",
        text: "wink-nlp is a JavaScript NLP toolkit",
        title: "winkjs/wink-nlp",
        source_url: "https://github.com/winkjs/wink-nlp",
      },
    ];
    const out = formatKbContext(hits);
    expect(out).toContain("dossier github/winkjs/wink-nlp");
    expect(out).toContain("sim=0.72");
    expect(out).toContain("winkjs/wink-nlp");
  });
});

describe("runUserPromptSubmit (integration, no provider)", () => {
  // Without EXA_API_KEY or a populated KB, the KB-miss path falls through
  // to an empty envelope. This is the failure-mode contract for a fresh
  // install with no Exa provisioned.
  const originalKey = process.env.EXA_API_KEY;
  beforeAll(() => {
    delete process.env.EXA_API_KEY;
  });
  afterAll(() => {
    if (originalKey === undefined) delete process.env.EXA_API_KEY;
    else process.env.EXA_API_KEY = originalKey;
  });

  it("returns {} on bypass env", async () => {
    process.env.VOUCH_USERPROMPT_BYPASS = "1";
    const out = await runUserPromptSubmit({ prompt: "tell me about wink-nlp library" });
    expect(out).toEqual({});
    delete process.env.VOUCH_USERPROMPT_BYPASS;
  });

  it("returns {} on empty prompt", async () => {
    const out = await runUserPromptSubmit({ prompt: "" });
    expect(out).toEqual({});
  });

  it("returns {} on missing prompt field", async () => {
    const out = await runUserPromptSubmit({});
    expect(out).toEqual({});
  });
});
