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
  // Hook output is threaded back as a `type:"attachment"` record (NOT a user/assistant turn). The
  // Stop reviewer's own verdict lands here: attachment.command==="vouch stop review", and the
  // verdict text is JSON in attachment.stdout → hookSpecificOutput.additionalContext. (Verified on
  // a live transcript 2026-06-23: the `vouch reviewer (BLOCK)` records are type=attachment, not user.)
  attachment?: { type?: string; command?: string; stdout?: string };
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

// The gate's OWN prior verdicts this session. These are INDEPENDENT evidence (the gate authored
// them, not the agent), so grounding a "this was flagged before" / "that warning was a false
// positive" reference against them is NOT circular. They live in `type:"attachment"` records
// (command "vouch stop review"), with the verdict text as JSON in `attachment.stdout` under
// hookSpecificOutput.additionalContext — a channel stripHarness() never sees.
export function extractPriorVerdicts(transcriptText: string, opts?: ConversationOpts): string[] {
  const perMsg = opts?.perMsg ?? 1500;
  const out: string[] = [];
  for (const line of transcriptText.split("\n")) {
    const t = line.trim();
    if (!t) continue;
    let rec: TranscriptRecord;
    try {
      rec = JSON.parse(t) as TranscriptRecord;
    } catch {
      continue;
    }
    if (rec.type !== "attachment" || rec.attachment?.command !== "vouch stop review") continue;
    const stdout = rec.attachment?.stdout;
    if (typeof stdout !== "string" || !stdout) continue;
    let ctx = "";
    try {
      const parsed = JSON.parse(stdout) as { hookSpecificOutput?: { additionalContext?: unknown } };
      const ac = parsed.hookSpecificOutput?.additionalContext;
      if (typeof ac === "string") ctx = ac.trim();
    } catch {
      continue; // a malformed hook payload is not a verdict we can surface
    }
    if (!ctx) continue;
    if (/reviewer unavailable/i.test(ctx)) continue; // fail-open health note, not a verdict to ground against
    out.push(ctx.length > perMsg ? `${ctx.slice(0, perMsg)}…[truncated]` : ctx);
  }
  return out.slice(-(opts?.max ?? 20));
}

// Format prior verdicts as a context block. The framing is deliberate: these ground a back-reference
// to the gate's own earlier output; they are NOT a reason to re-fire the same objection reflexively.
export function formatPriorVerdicts(verdicts: string[]): string {
  if (!verdicts.length) return "";
  const body = verdicts.map((v, i) => `  [${i + 1}] ${v.replace(/\n/g, "\n      ")}`).join("\n");
  return `\n\nYOUR PRIOR VERDICTS THIS SESSION (the gate's OWN earlier output, oldest→newest — use ONLY to ground a reference TO a prior verdict, e.g. "this was already flagged"/"that warning was a false positive"; do NOT re-raise a past objection just because it appears here):\n${body}`;
}

// The agent's OWN PRIOR responses this session. Unlike user messages / prior verdicts, this layer is
// NOT independent evidence — it is the agent's own prose, so it is circular to ground a claim's TRUTH
// against it. We surface it anyway, scoped to ONE non-circular use: verifying that the agent ACTUALLY
// SAID something it now back-references ("as I said", "I noted earlier", "I already explained"). Without
// this the reviewer can't see the chat layer at all, so a faithful self-reference fires passive-
// fabrication ("no evidence the agent said X") — the documented blind spot (this session: 3 such fires
// on "让我先确认", a temporal self-narration, and a repeat). The anti-laundering rule lives in the
// reviewer prompt (CONVERSATION_CLAUSE): a prior response proves only WHAT WAS SAID, never that its
// content is TRUE — a factual/own-work claim is still audited in the tool trace even if asserted before.
//
// The CURRENT response (the action under review) is the LAST assistant turn in the transcript at Stop
// time (it's already written). We DROP it — feeding the draft back as its own "evidence" is exactly the
// circularity we're guarding against; only EARLIER turns are surfaced.
export function extractAssistantMessages(transcriptText: string, opts?: ConversationOpts): string[] {
  const perMsg = opts?.perMsg ?? 1500;
  const all: string[] = [];
  for (const line of transcriptText.split("\n")) {
    const t = line.trim();
    if (!t) continue;
    let rec: TranscriptRecord;
    try {
      rec = JSON.parse(t) as TranscriptRecord;
    } catch {
      continue;
    }
    if (rec.message?.role !== "assistant" || !Array.isArray(rec.message.content)) continue;
    const blocks = rec.message.content as TranscriptBlock[];
    const text = blocks
      .filter((b) => b?.type === "text" && typeof b.text === "string")
      .map((b) => b.text as string)
      .join("\n")
      .trim();
    if (!text) continue; // a tool-use-only assistant turn carries no prose to ground a reference to
    all.push(text.length > perMsg ? `${text.slice(0, perMsg)}…[truncated]` : text);
  }
  all.pop(); // drop the current draft (the action under review) — surfacing it back is circular
  return all.slice(-(opts?.max ?? 15));
}

// Format the agent's prior responses. The framing is the load-bearing anti-laundering guard: these
// ground ONLY the existence of an utterance, never the truth of its content.
export function formatAssistantMessages(msgs: string[]): string {
  if (!msgs.length) return "";
  const body = msgs.map((m, i) => `  [${i + 1}] ${m.replace(/\n/g, "\n      ")}`).join("\n");
  return `\n\nYOUR OWN PRIOR RESPONSES THIS SESSION (the agent's earlier prose, oldest→newest — use ONLY to verify the agent ACTUALLY SAID something it now references, e.g. "as I said"/"I noted earlier". A prior response proves WHAT WAS SAID, NOT that its content is true: a factual or own-work claim must STILL be checked in the tool trace even if asserted here — do NOT credit a claim merely because the agent said it before):\n${body}`;
}

// Search the FULL conversation layer (user messages, the agent's own prior responses, the gate's
// prior verdicts) for `pattern` — the conversation-layer analog of queryHistory over the tool trace.
// WHY: the prompt only surfaces a recent WINDOW of each layer (cost), so a self-reference to prose
// from EARLIER in a long session ages out of the window and re-fires the very cry-wolf this is meant
// to kill (measured: a referenced turn at index 4 of 30 fell outside a 15-turn window). Mirrors the
// events/searchEvents decoupling (fix #1, a4f8cf4): window the prompt, but let the reviewer REACH the
// whole layer on demand. Each layer is LABELLED so the anti-laundering scoping (assistant prose =
// existence-only) survives into the query result. Returns "" when nothing matches or no layers given.
export interface ConversationLayers {
  userMessages?: string[];
  assistantMessages?: string[];
  priorVerdicts?: string[];
}
export function searchConversation(pattern: string, layers: ConversationLayers, perHit = 600): string {
  let re: RegExp;
  try {
    re = new RegExp(pattern, "i");
  } catch {
    re = new RegExp(pattern.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "i"); // literal fallback
  }
  const sections: string[] = [];
  const scan = (label: string, msgs: string[] | undefined) => {
    if (!msgs?.length) return;
    const hits = msgs.filter((m) => re.test(m));
    if (!hits.length) return;
    const body = hits
      .map((m) => {
        if (m.length <= perHit) return `  • ${m.replace(/\n/g, " ")}`;
        const idx = m.match(re)?.index ?? 0;
        const start = Math.max(0, idx - Math.floor(perHit / 3));
        return `  • …${m.slice(start, start + perHit).replace(/\n/g, " ")}…`;
      })
      .join("\n");
    sections.push(`${label} (${hits.length} match${hits.length > 1 ? "es" : ""}):\n${body}`);
  };
  scan("USER MESSAGES", layers.userMessages);
  // existence-only: a match here proves the agent SAID it, NOT that it is true (see CONVERSATION_CLAUSE)
  scan("YOUR OWN PRIOR RESPONSES", layers.assistantMessages);
  scan("YOUR PRIOR VERDICTS", layers.priorVerdicts);
  if (!sections.length) return "";
  return `\n---\nCONVERSATION-LAYER matches (not the tool trace):\n${sections.join("\n")}`;
}

// Search the RAW transcript (every record type: user, assistant, system, attachment) for `pattern`.
// This is the root fix for the reviewer's worst blindness: query_history searches only the TOOL TRACE
// (PostToolUse events), but a huge share of what the agent references — the gate's OWN prior verdicts,
// system-reminders, the agent's prose — lives in `type=system`/`type=attachment`/assistant records that
// the tool trace NEVER contains. Measured this session: 220 "vouch reviewer" verdict records existed in
// the transcript (66 system + 126 attachment + ...) while the reviewer fired `active-fabrication` on them
// as "non-existent events", because its search couldn't reach that layer. This searches it directly, so
// "did the agent/gate/system actually say X" is answerable regardless of which layer X lives in.
//
// SEARCH-ONLY (like the events/searchEvents split): the raw transcript is searched on demand and only
// matched snippets are returned — the full text never enters the prompt, so cost stays bounded. Additive
// evidence → can only REDUCE false fabrication-fires (referent now findable); it does not touch the
// firing criteria or the tool-trace path, so it is recall-safe by construction.
export function searchTranscript(transcriptText: string, pattern: string, opts?: { max?: number; perHit?: number }): string {
  const max = opts?.max ?? 12;
  const perHit = opts?.perHit ?? 500;
  let re: RegExp;
  try {
    re = new RegExp(pattern, "i");
  } catch {
    re = new RegExp(pattern.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "i"); // literal fallback
  }
  const hits: string[] = [];
  for (const line of transcriptText.split("\n")) {
    const t = line.trim();
    if (!t) continue;
    let rec: TranscriptRecord & { content?: unknown };
    try {
      rec = JSON.parse(t) as TranscriptRecord & { content?: unknown };
    } catch {
      continue;
    }
    // Pull the human-readable text for this record, by type.
    let text = "";
    const c = rec.message?.content;
    if (typeof c === "string") text = c;
    else if (Array.isArray(c)) {
      text = (c as TranscriptBlock[])
        .filter((b) => b?.type === "text" && typeof b.text === "string")
        .map((b) => b.text as string)
        .join("\n");
    }
    if (rec.attachment?.stdout) text += `\n${rec.attachment.stdout}`;
    if (typeof rec.content === "string") text += `\n${rec.content}`; // type=system records carry text here
    if (!text || !re.test(text)) continue;
    const role = rec.message?.role ?? rec.attachment?.type ?? rec.type ?? "?";
    const idx = text.match(re)?.index ?? 0;
    const start = Math.max(0, idx - Math.floor(perHit / 3));
    const snippet = text.slice(start, start + perHit).replace(/\s+/g, " ").trim();
    hits.push(`  • [${rec.type}/${role}] ${start > 0 ? "…" : ""}${snippet}${start + perHit < text.length ? "…" : ""}`);
  }
  if (!hits.length) return "";
  // existence-only: a match in an ASSISTANT record proves the agent SAID it, not that it is true.
  return `\n---\nTRANSCRIPT matches (conversation/system layer — NOT the tool trace; an assistant-record match proves WHAT WAS SAID, not that it is true):\n${hits.slice(-max).join("\n")}`;
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
