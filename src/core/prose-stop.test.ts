import { expect, test } from "bun:test";
import { extractLastAssistantText, isHarnessError } from "./prose-stop.ts";

const asst = (text: string): string =>
  JSON.stringify({ type: "assistant", message: { role: "assistant", content: [{ type: "text", text }] } });

test("isHarnessError flags API/rate-limit/abort strings, not genuine prose", () => {
  expect(isHarnessError("API Error: Server is temporarily limiting requests · Rate limited")).toBe(true);
  expect(isHarnessError("Request was aborted.")).toBe(true);
  expect(isHarnessError("429 Too Many Requests")).toBe(true);
  expect(isHarnessError("(eval):1: command not found: timeout")).toBe(true);
  // genuine agent prose must NOT be flagged
  expect(isHarnessError("The tests pass and I committed the fix.")).toBe(false);
  expect(isHarnessError("I'll verify the rate limit handling in the code.")).toBe(false); // mentions 'rate limit' mid-sentence
});

test("isHarnessError does NOT skip a genuine draft that STARTS with an error keyword (recall hole)", () => {
  // The closed recall hole (falsification-tested 2026-06-24): a loose prefix match dropped these
  // genuine drafts from review entirely. They start with an error keyword but continue as narrative,
  // so the tightened matcher (keyword must be followed by error-shaped continuation) keeps them reviewed.
  expect(isHarnessError("API error handling: I added a retry wrapper and 6/6 tests pass.")).toBe(false);
  expect(isHarnessError("Rate limit handling is now implemented with backoff.")).toBe(false);
  expect(isHarnessError("429 handling: I added exponential backoff; verified in tests.")).toBe(false);
  expect(isHarnessError("The API returned 429 for my request, but I retried and it succeeded.")).toBe(false);
  // …while real harness errors of each class still skip (the #5 cry-wolf must not reopen):
  expect(isHarnessError("503 Service Unavailable")).toBe(true);
  expect(isHarnessError("rate limit exceeded")).toBe(true);
  expect(isHarnessError("Request timed out.")).toBe(true);
});

test("extractLastAssistantText skips a harness-error draft and keeps the last GENUINE response", () => {
  // The eye-problem #5 case: an API error recorded as the final assistant turn would otherwise be
  // reviewed as the agent's 'fabricated' claim. The reviewer must instead see the prior real response.
  const transcript = [
    asst("Here is my analysis: the fix works and tests pass."),
    asst("API Error: Server is temporarily limiting requests (not your usage limit) · Rate limited"),
  ].join("\n");
  expect(extractLastAssistantText(transcript)).toBe("Here is my analysis: the fix works and tests pass.");
});

test("extractLastAssistantText returns the genuine final response when there is no error", () => {
  const transcript = [asst("first answer"), asst("final answer")].join("\n");
  expect(extractLastAssistantText(transcript)).toBe("final answer");
});
