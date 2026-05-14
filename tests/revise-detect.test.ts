/** #50 (A) Stage 2 detection — classifyReviseAction pinpoints how a turn's
 *  draft addressed (or didn't address) a prior-turn fire's entity. */
import { describe, expect, it } from "bun:test";
import { classifyReviseAction } from "../src/gate.ts";

function pair(entity: string, proposition = `${entity} does X.`, stance = "ASSERT" as const) {
  return { entity, proposition, stance };
}

describe("classifyReviseAction", () => {
  it("returns null for empty entity", () => {
    expect(classifyReviseAction("", "anything", [])).toBeNull();
  });

  describe("fetch (tool call referencing entity)", () => {
    it("vouch fetch <url> with entity in same line", () => {
      const draft = "OK, verifying: `vouch fetch https://arxiv.org/abs/2302.09664` for Kuhn et al.";
      expect(classifyReviseAction("Kuhn et al", draft, [])).toBe("fetch");
    });

    it("vouch claim with entity in same line", () => {
      const draft = "Filed: `vouch claim \"ALCE is a citation eval bench\" --dossier ...`";
      expect(classifyReviseAction("ALCE", draft, [])).toBe("fetch");
    });

    it("WebFetch on a URL near entity reference", () => {
      const draft = "I'll WebFetch https://github.com/marimo-team/marimo to confirm marimo's stars.";
      expect(classifyReviseAction("marimo", draft, [])).toBe("fetch");
    });

    it("gh api near entity", () => {
      const draft = "Let me run `gh api repos/anthropics/claude-code` to check Claude Code's stars.";
      expect(classifyReviseAction("Claude Code", draft, [])).toBe("fetch");
    });

    it("does NOT trigger on tool call far from entity (>300 chars)", () => {
      const filler = "a".repeat(400);
      const draft = `vouch fetch https://example.com\n${filler}\nALCE is a benchmark`;
      // Tool match at start, entity 400+ chars later → outside proximity window.
      // Should NOT classify as fetch; falls to 'remove' (entity not in pairs).
      expect(classifyReviseAction("ALCE", draft, [])).toBe("remove");
    });
  });

  describe("hedge (uncertainty tag adjacent to entity)", () => {
    it("(unverified) tag near entity", () => {
      const draft = "Marimo has ~21k stars (unverified, from training memory).";
      expect(classifyReviseAction("Marimo", draft, [])).toBe("hedge");
    });

    it("'from training memory' tag near entity", () => {
      const draft = "I think TreeSHAP runs in O(TLD^2) — from training memory, not verified.";
      expect(classifyReviseAction("TreeSHAP", draft, [])).toBe("hedge");
    });

    it("'haven't verified' phrasing near entity", () => {
      const draft = "ALCE recall is around 35% but I haven't verified the exact number.";
      expect(classifyReviseAction("ALCE", draft, [])).toBe("hedge");
    });

    it("'not yet checked' near entity", () => {
      const draft = "MiniCheck-T5 outperforms on FActScore — not yet checked against the paper.";
      expect(classifyReviseAction("MiniCheck-T5", draft, [])).toBe("hedge");
    });

    it("does NOT match hedge token far from entity", () => {
      const filler = "x".repeat(500);
      const draft = `Some unrelated unverified statement.\n${filler}\nALCE is a benchmark.`;
      // Hedge token at start, ALCE 500+ chars later → out of window.
      expect(classifyReviseAction("ALCE", draft, [])).toBe("remove");
    });
  });

  describe("remove (entity absent from current extracted pairs)", () => {
    it("entity dropped from claim — no tool call, no hedge", () => {
      const draft = "The benchmark suite measures various properties of agents.";
      expect(classifyReviseAction("ALCE", draft, [pair("benchmark", "benchmark measures things")])).toBe("remove");
    });

    it("empty pairs + no signal → remove (trivially absent)", () => {
      expect(classifyReviseAction("ALCE", "Some text without the entity name.", [])).toBe("remove");
    });
  });

  describe("null (entity still asserted with no fetch, no hedge)", () => {
    it("entity reappears in new ASSERT with no resolution → null", () => {
      const draft = "ALCE achieves state-of-the-art on citation recall.";
      // Entity is in a new pair, no hedge token, no tool call → null = unaddressed
      expect(
        classifyReviseAction("ALCE", draft, [
          pair("ALCE", "ALCE achieves SOTA on citation recall."),
        ]),
      ).toBeNull();
    });
  });

  describe("priority order: fetch beats hedge beats remove", () => {
    it("both fetch AND hedge → fetch wins (more decisive)", () => {
      const draft = "Running `vouch fetch arxiv.org/abs/2302.09664` for Kuhn (unverified for now).";
      expect(classifyReviseAction("Kuhn", draft, [])).toBe("fetch");
    });

    it("hedge present + entity in extracted pair → hedge wins (not remove)", () => {
      const draft = "ALCE recall is 35% (unverified, from training memory).";
      expect(
        classifyReviseAction("ALCE", draft, [
          pair("ALCE", "ALCE recall is 35%"),
        ]),
      ).toBe("hedge");
    });
  });

  describe("case-insensitive matching", () => {
    it("hedge tag with uppercase Unverified", () => {
      const draft = "marimo has 21k stars (Unverified, from memory).";
      expect(classifyReviseAction("marimo", draft, [])).toBe("hedge");
    });

    it("tool call with mixed case (gh API)", () => {
      const draft = "I'll Vouch Fetch the page for marimo to ground this.";
      expect(classifyReviseAction("marimo", draft, [])).toBe("fetch");
    });
  });
});
