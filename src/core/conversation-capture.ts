// Conversation-layer evidence for the reviewer — the OTHER half of "what happened this session".
//
// evidence-capture.ts records only the PostToolUse TOOL trace (Bash/Read/Edit/Web I/O). It does
// NOT record the conversation: the user's messages, the gate's own prior advise/block output, or
// the agent's prior turns. So when the agent's response references that layer ("you asked me to X",
// "as the user said"), the reviewer queries query_history, finds nothing — chat is not in the tool
// trace — and fires passive-fabrication. That recursive cry-wolf on recap prose is the documented
// blind spot (handoff 2026-06-21, line 34), and it is a SCOPE gap, not a model failure: the full
// transcript is already loaded at the Stop hook (dispatch.ts: `await Bun.file(transcript_path).text()`),
// it just isn't surfaced to the reviewer.
//
// This module surfaces the SAFE, high-value slice of that layer — the GENUINE USER messages — so a
// claim like "the user asked me to merge" becomes verifiable and a misquote of the user becomes
// catchable. Scope is deliberately narrow:
//   - USER messages only. Prior ASSISTANT turns are NOT included: feeding the agent's own earlier
//     prose back as "evidence" risks circular grounding (turn N citing turn N-1's unverified claim).
//   - Provenance verified on a live transcript (2026-06-22): a genuine user turn is `type==="user"
//     && message.role==="user"` whose content is a string or carries a text block. Machine injections
//     live in `type=attachment`/`type=system` (role=undefined); tool I/O lives in `type=user role=user`
//     records whose content is a `tool_result` block — both correctly excluded by this filter.
//   - KNOWN GAP (v1): an interrupt message the user sends mid-tool-execution is stored in the
//     `type=queue-operation`/`type=attachment` channel (same channel as machinery), NOT as a user
//     record — so it is NOT captured here. Closing that needs decoding the version-specific (CC
//     v2.1.x) `attachment` schema and separating user-interrupt from machinery; deferred to v2.

interface TranscriptBlock {
  type?: string;
  text?: string;
}
interface TranscriptRecord {
  type?: string;
  message?: { role?: string; content?: unknown };
}

// Plain text of a user record's content. A genuine user turn is a bare string or an array with text
// blocks; a tool_result-bearing record is tool I/O (not a user message) → "" so the caller drops it.
function userText(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  const blocks = content as TranscriptBlock[];
  if (blocks.some((b) => b?.type === "tool_result")) return ""; // tool I/O, not a user message
  return blocks
    .filter((b) => b?.type === "text" && typeof b.text === "string")
    .map((b) => b.text as string)
    .join("\n")
    .trim();
}

// Strip the harness wrappers Claude Code threads through user-role records so a real prompt isn't
// confused with machinery: <system-reminder> blocks, slash-command echoes, local-command stdout.
// If nothing human-authored remains, it was not a real user message.
function stripHarness(text: string): string {
  return text
    .replace(/<system-reminder>[\s\S]*?<\/system-reminder>/g, "")
    .replace(/<local-command-[\s\S]*?<\/local-command-[^>]*>/g, "")
    .replace(/<command-(name|message|args)>[\s\S]*?<\/command-(name|message|args)>/g, "")
    .trim();
}

export interface ConversationOpts {
  max?: number; // keep only the most recent N user turns (newest last)
  perMsg?: number; // cap each message's length
}

// Parse a transcript JSONL string into the GENUINE user messages, oldest→newest.
export function extractUserMessages(transcriptText: string, opts?: ConversationOpts): string[] {
  const perMsg = opts?.perMsg ?? 2000;
  const out: string[] = [];
  for (const line of transcriptText.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    let rec: TranscriptRecord;
    try {
      rec = JSON.parse(trimmed) as TranscriptRecord;
    } catch {
      continue;
    }
    if (rec.type !== "user" || rec.message?.role !== "user") continue;
    const raw = userText(rec.message.content);
    if (!raw) continue;
    const text = stripHarness(raw);
    if (!text) continue; // pure harness wrapper, no human content
    out.push(text.length > perMsg ? `${text.slice(0, perMsg)}…[truncated]` : text);
  }
  const max = opts?.max ?? 30;
  return out.slice(-max);
}

// Format the user messages as a context block for the reviewer prompt. Empty string when there are
// none (so the caller can keep the prompt byte-identical to the no-conversation path).
export function formatUserMessages(msgs: string[]): string {
  if (!msgs.length) return "";
  const body = msgs.map((m, i) => `  [${i + 1}] ${m.replace(/\n/g, "\n      ")}`).join("\n");
  return `\n\nUSER MESSAGES THIS SESSION (the user's actual words, oldest→newest):\n${body}`;
}

// Self-test: `bun src/core/conversation-capture.ts <transcript.jsonl>`
if (import.meta.main) {
  const path =
    process.argv[2] ??
    (() => {
      throw new Error("usage: bun conversation-capture.ts <transcript.jsonl>");
    })();
  const msgs = extractUserMessages(await Bun.file(path).text());
  console.log(`extracted ${msgs.length} genuine user message(s):\n`);
  for (const [i, m] of msgs.entries()) console.log(`  [${i}] ${m.replace(/\n/g, " ⏎ ").slice(0, 120)}`);
}
