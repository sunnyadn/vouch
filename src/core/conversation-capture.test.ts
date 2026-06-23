import { test, expect } from "bun:test";
import {
  extractAssistantMessages,
  extractPriorVerdicts,
  extractUserMessages,
  searchConversation,
} from "./conversation-capture.ts";

// Build a transcript JSONL line for a vouch Stop-hook verdict (the real shape: type=attachment,
// command="vouch stop review", verdict text JSON-nested in attachment.stdout).
const verdictLine = (text: string): string =>
  JSON.stringify({
    type: "attachment",
    attachment: {
      type: "hook_success",
      command: "vouch stop review",
      stdout: JSON.stringify({ hookSpecificOutput: { hookEventName: "Stop", additionalContext: text } }),
    },
  });
const userLine = (text: string): string =>
  JSON.stringify({ type: "user", message: { role: "user", content: text } });
const assistantLine = (text: string): string =>
  JSON.stringify({ type: "assistant", message: { role: "assistant", content: [{ type: "text", text }] } });
const assistantToolLine = (): string =>
  JSON.stringify({
    type: "assistant",
    message: { role: "assistant", content: [{ type: "tool_use", id: "t1", name: "Bash", input: {} }] },
  });

test("extractPriorVerdicts pulls verdict text from the attachment channel (not user turns)", () => {
  const transcript = [
    userLine("fix the bug"),
    verdictLine("⛔ vouch reviewer (BLOCK): fabrication detected: claim X has no evidence"),
    verdictLine("⚠ vouch reviewer (advise): research concern: claim Y under-researched"),
  ].join("\n");
  const verdicts = extractPriorVerdicts(transcript);
  expect(verdicts.length).toBe(2);
  expect(verdicts[0]).toContain("BLOCK");
  expect(verdicts[1]).toContain("advise");
});

test("extractPriorVerdicts skips the fail-open health note (not a verdict to ground against)", () => {
  const transcript = [
    verdictLine("⚠ vouch reviewer unavailable — it failed open and reviewed nothing this turn."),
    verdictLine("⛔ vouch reviewer (BLOCK): real verdict here"),
  ].join("\n");
  const verdicts = extractPriorVerdicts(transcript);
  expect(verdicts.length).toBe(1); // the unavailable note is dropped
  expect(verdicts[0]).toContain("real verdict");
});

test("extractPriorVerdicts ignores non-vouch attachments and malformed stdout", () => {
  const transcript = [
    JSON.stringify({ type: "attachment", attachment: { command: "other hook", stdout: "{}" } }),
    JSON.stringify({ type: "attachment", attachment: { command: "vouch stop review", stdout: "not json" } }),
    verdictLine("⛔ vouch reviewer (BLOCK): the only real one"),
  ].join("\n");
  const verdicts = extractPriorVerdicts(transcript);
  expect(verdicts.length).toBe(1);
  expect(verdicts[0]).toContain("the only real one");
});

test("extractUserMessages and extractPriorVerdicts read DISJOINT channels from one transcript", () => {
  const transcript = [userLine("do the thing"), verdictLine("⛔ vouch reviewer (BLOCK): nope")].join("\n");
  expect(extractUserMessages(transcript)).toEqual(["do the thing"]);
  expect(extractPriorVerdicts(transcript).length).toBe(1);
});

test("extractAssistantMessages returns prior prose and DROPS the current draft (last assistant turn)", () => {
  const transcript = [
    userLine("question 1"),
    assistantLine("earlier answer A"),
    userLine("question 2"),
    assistantLine("earlier answer B"),
    assistantLine("THE CURRENT DRAFT under review"),
  ].join("\n");
  const msgs = extractAssistantMessages(transcript);
  expect(msgs).toEqual(["earlier answer A", "earlier answer B"]); // draft dropped, prose oldest→newest
});

test("extractAssistantMessages skips tool-use-only assistant turns (no prose to ground)", () => {
  const transcript = [
    assistantLine("prose turn one"),
    assistantToolLine(), // tool_use only → no text → skipped
    assistantLine("prose turn two"),
    assistantLine("the draft"),
  ].join("\n");
  expect(extractAssistantMessages(transcript)).toEqual(["prose turn one", "prose turn two"]);
});

test("extractAssistantMessages ignores user turns and verdict attachments", () => {
  const transcript = [
    userLine("not me"),
    assistantLine("my only prior prose"),
    verdictLine("⛔ vouch reviewer (BLOCK): not assistant prose"),
    assistantLine("the draft"),
  ].join("\n");
  expect(extractAssistantMessages(transcript)).toEqual(["my only prior prose"]);
});

test("searchConversation reaches an aged-out assistant turn the prompt window would drop", () => {
  // The referenced prose is the OLDEST of many turns — exactly what a recent-N prompt window drops.
  const assistantMessages = [
    "I will first confirm the architecture before proposing", // the aged-out reference target
    ...Array.from({ length: 20 }, (_, i) => `later filler turn ${i}`),
  ];
  const hit = searchConversation("first confirm the architecture", { assistantMessages });
  expect(hit).toContain("YOUR OWN PRIOR RESPONSES");
  expect(hit).toContain("first confirm the architecture");
  // existence-only labelling survives into the query result
  expect(hit).toContain("CONVERSATION-LAYER matches");
});

test("searchConversation returns empty when nothing matches or no layers supplied", () => {
  expect(searchConversation("nope", {})).toBe("");
  expect(searchConversation("absent", { userMessages: ["hello"] })).toBe("");
});

test("searchConversation labels each layer distinctly", () => {
  const hit = searchConversation("merge", {
    userMessages: ["please merge it"],
    priorVerdicts: ["BLOCK: do not merge unverified"],
  });
  expect(hit).toContain("USER MESSAGES");
  expect(hit).toContain("YOUR PRIOR VERDICTS");
});
