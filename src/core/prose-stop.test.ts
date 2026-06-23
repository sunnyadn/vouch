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
