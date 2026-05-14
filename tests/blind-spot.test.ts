/** P-γ.5 blind-spot enumeration detector — counts explicit [gap: …]
 *  markers AND natural-language gap phrases. Pure function over draft
 *  text; no DB / LLM dependency. */
import { describe, expect, it } from "bun:test";
import { countBlindSpots } from "../src/gate.ts";

describe("countBlindSpots — explicit [gap: ...] markers", () => {
  it("single bracket marker", () => {
    expect(countBlindSpots("[gap: didn't check API latency]")).toEqual({
      explicit: 1,
      phrase: 0,
    });
  });

  it("paren marker", () => {
    expect(countBlindSpots("(gap: didn't check cost)")).toEqual({
      explicit: 1,
      phrase: 0,
    });
  });

  it("multiple markers, mixed brackets", () => {
    const draft = `Some text. [gap: latency unknown] more text (gap: cost unknown) end.`;
    expect(countBlindSpots(draft)).toEqual({ explicit: 2, phrase: 0 });
  });

  it("case-insensitive", () => {
    expect(countBlindSpots("[GAP: x] [Gap: y]")).toEqual({ explicit: 2, phrase: 0 });
  });

  it("ignores non-gap markers", () => {
    const draft = "[verified: x] [unverified: y] [note: z]";
    expect(countBlindSpots(draft)).toEqual({ explicit: 0, phrase: 0 });
  });
});

describe("countBlindSpots — natural phrases", () => {
  it("'I didn't check X'", () => {
    expect(countBlindSpots("I didn't check the latency.")).toEqual({
      explicit: 0,
      phrase: 1,
    });
  });

  it("'I haven't verified X'", () => {
    expect(countBlindSpots("I haven't verified the recall number.")).toEqual({
      explicit: 0,
      phrase: 1,
    });
  });

  it("'I don't know if X'", () => {
    expect(countBlindSpots("I don't know if it handles unicode.")).toEqual({
      explicit: 0,
      phrase: 1,
    });
  });

  it("'I'm not sure about X'", () => {
    expect(countBlindSpots("I'm not sure about the threshold value.")).toEqual({
      explicit: 0,
      phrase: 1,
    });
  });

  it("'worth checking X'", () => {
    expect(countBlindSpots("Worth checking whether v2.1 fixed this.")).toEqual({
      explicit: 0,
      phrase: 1,
    });
  });

  it("'I should also check X'", () => {
    expect(countBlindSpots("I should also check the GitHub issue tracker.")).toEqual({
      explicit: 0,
      phrase: 1,
    });
  });

  it("'open question'", () => {
    expect(countBlindSpots("That's an open question — needs more data.")).toEqual({
      explicit: 0,
      phrase: 1,
    });
  });

  it("multiple natural phrases", () => {
    const draft = `I didn't check the perf. I'm not sure about cost. Worth checking docs.`;
    const result = countBlindSpots(draft);
    expect(result.phrase).toBeGreaterThanOrEqual(3);
  });
});

describe("countBlindSpots — combined", () => {
  it("explicit + natural mixed", () => {
    const draft = `Filed claim X. [gap: API rate limits unknown] Also, I didn't check the docs.`;
    expect(countBlindSpots(draft)).toEqual({ explicit: 1, phrase: 1 });
  });

  it("empty draft", () => {
    expect(countBlindSpots("")).toEqual({ explicit: 0, phrase: 0 });
  });

  it("draft with no gap signals", () => {
    expect(countBlindSpots("X is Y. A is B. Done.")).toEqual({
      explicit: 0,
      phrase: 0,
    });
  });
});
