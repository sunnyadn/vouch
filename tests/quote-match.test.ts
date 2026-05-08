/** Pure logic tests for quote-in-dossier matching. */
import { describe, expect, it } from "bun:test";
import { findQuoteInContent } from "../src/quote-match.ts";

describe("findQuoteInContent", () => {
  it("exact substring → exact match with offsets", () => {
    const content = "Once upon a time the cat sat on the mat. The end.";
    const quote = "the cat sat on the mat";
    const r = findQuoteInContent(quote, content);
    expect(r.found).toBe(true);
    expect(r.matchType).toBe("exact");
    expect(r.start).toBe(content.indexOf(quote));
    expect(r.end).toBe(content.indexOf(quote) + quote.length);
  });

  it("smart-quote difference → normalized match", () => {
    const r = findQuoteInContent("It's the cat's hat", "yesterday it’s the cat’s hat sat there");
    expect(r.found).toBe(true);
    expect(r.matchType).toBe("normalized");
  });

  it("space-before-punctuation artifact → normalized match", () => {
    // mimic HTML strip output where </span> inserts spaces
    const content = "engine written in C . It is a library . It serves...";
    const r = findQuoteInContent("engine written in C. It is a library.", content);
    expect(r.found).toBe(true);
    expect(r.matchType).toBe("normalized");
  });

  it("dropped comma → fuzzy match (≥30 char overlap)", () => {
    const content = "the quick brown fox jumps over the lazy dog at noon, every day";
    const quote = "the quick brown fox jumps over the lazy dog at noon every day";
    const r = findQuoteInContent(quote, content);
    expect(r.found).toBe(true);
    expect(r.matchType).toBe("fuzzy");
  });

  it("fabricated quote → not found", () => {
    const content = "Once upon a time there was a cat who liked yarn very much.";
    const r = findQuoteInContent("the cat owned a smartphone and a tesla", content);
    expect(r.found).toBe(false);
    expect(r.matchType).toBe("none");
  });

  it("empty quote → not found", () => {
    const r = findQuoteInContent("", "any content here");
    expect(r.found).toBe(false);
  });

  it("short fuzzy candidate (<30 chars alphanumeric) → not promoted", () => {
    // Common fragments shouldn't false-positive
    const r = findQuoteInContent("the cat", "Once upon a time something else entirely happened.");
    expect(r.found).toBe(false);
  });
});
