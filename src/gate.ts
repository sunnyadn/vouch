/** Stop-hook confabulation gate.
 *
 * Pipeline:
 *   1. Read the last assistant text from a Claude Code transcript.
 *   2. Use a fast LLM to extract {proposition, stance, entity} triples — every
 *      proposition the draft makes about a NAMED EXTERNAL ENTITY, labelled
 *      with its stance (ASSERT, OPINION, HEDGE, SPECULATE, NEGATE, COMPARE,
 *      META, RETRACT, REFER). Workspace context and common knowledge are
 *      excluded at extraction.
 *   3. For each ASSERT triple: hybrid-search vouch's claim KB (on the
 *      proposition, not the bare entity); a supported claim with high lexical
 *      overlap short-circuits as grounded; otherwise run NLI (proposition vs
 *      claim_text + dossier source quote). Grounded if ANY supported claim
 *      entails the proposition. Non-ASSERT triples (OPINION / HEDGE / … )
 *      short-circuit as grounded — there is no checkable fact to verify.
 *   4. Block (exit 2) if any ASSERT triple is ungrounded; otherwise pass.
 *
 * Fail-open: classifier/network/transient errors → exit 0, never block.
 *
 * Why proposition+stance, not entity+assertion (issue #1):
 *   The earlier extractor unit was "named-entity reference + inferred
 *   assertion". Hedge-spirals, hypotheticals, retractions, and comparison
 *   topics all leaked through because the entity name was always present in
 *   the prose. Forcing the extractor to commit to an explicit stance per
 *   proposition makes "this is not an assertion" a first-class output rather
 *   than an implicit prompt-side filter.
 *
 * Session-evidence auto-grounding (issue #21):
 *   When an ASSERT proposition isn't grounded by the KB, the gate — before
 *   blocking — scans the transcript's `tool_result` events for content the
 *   agent demonstrably retrieved THIS SESSION via `Read` / `WebFetch` /
 *   `WebSearch`. If one of those entity-mentioning results NLI-entails the
 *   proposition, the gate snapshots it as a dossier, files the claim against
 *   it, and passes — turning a false-positive fire into a recorded grounding
 *   instead of training the agent to dodge. Lazy (only on a fire), bounded
 *   (entity-matching results only, K-capped NLI, size-capped per result), and
 *   it ONLY trusts content that came back from a tool — never the assistant's
 *   own asserted prose. Anything not retrieved via a tool still blocks exactly
 *   as before.
 */
import { existsSync, readFileSync } from "node:fs";
import { generateObject } from "ai";
import { z } from "zod";

import { MAX_SOURCE_CHARS } from "./config.ts";
import { getLanguageModel } from "./providers.ts";
import { embedOne } from "./embedder.ts";
import { classifyError, verifyClaimAgainstSource } from "./verifier.ts";
import * as store from "./store.ts";

export const DEFAULT_GATE_MODEL =
  process.env.VOUCH_GATE_MODEL || "vertex_ai/gemini-3.1-flash-lite";

export const STANCES = [
  "ASSERT",
  "OPINION",
  "HEDGE",
  "SPECULATE",
  "NEGATE",
  "COMPARE",
  "META",
  "RETRACT",
  "REFER",
] as const;
export type Stance = (typeof STANCES)[number];

const StanceEnum = z.enum(STANCES);

const ExtractSchema = z.object({
  pairs: z
    .array(
      z.object({
        proposition: z.string(),
        stance: StanceEnum,
        entity: z.string(),
      }),
    )
    .max(20),
});

function buildExtractPrompt(draft: string): string {
  const projectsEnv = (process.env.VOUCH_GATE_WORKSPACE_PROJECTS || "").trim();
  const projectsLine = projectsEnv
    ? `\n      Known workspace projects for this installation (claims about their command surface, flags, file paths, test counts, build state, internal architecture, or runtime state are workspace, not external): ${projectsEnv}.`
    : "";
  return `You are a fact-grounding gate. Extract every PROPOSITION the draft makes about a NAMED EXTERNAL ENTITY and label its STANCE. The unit is "proposition with stance" — a draft can mention an entity name without asserting any fact about it; those cases get a non-ASSERT stance, not silently dropped and not over-extracted.

A NAMED EXTERNAL ENTITY is a third-party dataset, paper, product, library, model, company, person, or benchmark whose properties live OUTSIDE the assistant's workspace.

For each proposition, return { proposition, entity, stance }:
  - proposition: 1-sentence verbatim or close paraphrase
  - entity: short canonical name
  - stance — exactly one of:
      ASSERT    — declarative, checkable factual claim about a measurable property ("X has 100 features", "X beats Y by 3 points", "X was released in 2023")
      OPINION   — evaluative or normative judgment, not a measurable fact ("X is the best Y", "X is as valuable as Y", "X matters here", "X is overkill", "X is the right choice", "X's approach is cleaner") — including value-comparisons ("X is better than Y" with no metric)
      HEDGE     — assertion + caveat in the same sentence/clause: "(unverified)", "from training memory", "without verifying", "I haven't verified", "let me verify", "凭印象", or equivalent
      SPECULATE — hypothetical / conditional / modal ("X might do Y", "if X then Y")
      NEGATE    — explicit denial ("X does not support Y")
      COMPARE   — entity is the comparison topic without an asserted outcome ("we evaluated against X", "X vs Y")
      META      — reflective reference to a prior claim ("earlier I said X is Y")
      RETRACT   — explicit cancellation ("retracting earlier claim about X")
      REFER     — name as label only ("see also X"), OR a one-line gloss of what an entity/tool/skill IS at the level a directory listing would give ("kimi-task is a task-dispatch skill", "X is a CLI for Y") — provided no quantitative or specific factual property is asserted

DECISION RULES:
  - Hedge tokens in the same sentence/clause → stance is HEDGE, even if surface looks declarative. Hedge wins over ASSERT.
  - Evaluative/normative wording (best/worst/right/wrong/valuable/sufficient/overkill/cleaner/better-without-a-metric) → stance is OPINION, not ASSERT. "Is X good enough" is OPINION; "does X score ≥ 0.8" is ASSERT.
  - A claim about the model currently running this session (its capabilities, context budget, speed, resource use while doing the present work) → that is meta-commentary about the conversation → WORKSPACE, do not return a triple. The running model is not a third-party external entity here.
  - Composition claims: a proposition of the form "<the workspace project> {uses | defaults to | depends on | integrates | ships with | wraps | is built on} <third-party X>" is a claim about the workspace project's own code/config — verifiable by reading its source — NOT a claim about X. Skip it (WORKSPACE), even though X is a third-party name. Contrast: "<X> {has | returns | requires | scored} ..." (a property of X itself) is a real ASSERT about X.
  - "X vs Y" without an outcome → both X and Y are COMPARE; with a measured outcome ("X beat Y by N") → ASSERT; with an unmeasured value-outcome ("X is better than Y") → OPINION.
  - Retraction sentences re-mention the entity by necessity → stance is RETRACT, never ASSERT.
  - Annotations like "(claim N)", "(claim_id: N)", "(supported NLI)" → stance is META (already-grounded handle).

EXCLUDE entirely (do NOT return any triple):
  - Textbook background. Bar: "would a textbook in the relevant field state this without citation?" — if yes, skip. Method-of-X descriptions ("X is a method/model/test for Y", "X decomposes Y into Z") and well-known algorithm / statistical-test / mathematical-primitive names are textbook.
  - WORKSPACE — when in doubt, treat as workspace and skip:
      (a) The assistant's own actions, plans, recommendations, framing, or meta-commentary about the conversation.
      (b) Anything internal to a project the assistant maintains / authors / dogfoods — its command surface, flags, file paths, test counts, build state, runtime state, internal architecture, feature-support / roadmap claims, AND its own prompts / taxonomies / test fixtures (e.g., named fixtures like X1, X4, X5, C1) / source code / gate or extractor outputs. Even when X is itself a third-party methodology, claims about whether the project supports / lacks / will support it are workspace.${projectsLine}
      (c) Anything observed via a tool call earlier in this session (Bash output, file contents, git log/diff, test results, HTTP responses).
      (d) Forward-looking or proposed entities framed as not-yet-existing ("I'll file ISSUE-X", "a planned feature would ...").
      (e) Internal issue-tracker IDs (Linear / Jira / GitHub) and their described scope when filing / summarizing / proposing them.

If nothing qualifies, return { pairs: [] }.

Draft:
<<<
${draft}
>>>`;
}

export interface ExtractedPair {
  proposition: string;
  stance: Stance;
  entity: string;
}

export interface GroundedPair extends ExtractedPair {
  grounded: boolean;
  matched_claim_id: number | null;
  reason: string;
  /** Set when the gate fired on this pair and then scanned session evidence
   *  (tool_result events) for the entity: number of candidate session sources
   *  NLI-checked. 0 = none matched the entity. Absent when no session
   *  transcript was available or the pair was already grounded. */
  session_sources_checked?: number;
  /** True when this pair was resolved by auto-grounding against a source the
   *  agent retrieved this session (Read / WebFetch / WebSearch). */
  auto_grounded?: boolean;
}

export interface GateVerdict {
  blocked: boolean;
  pairs: GroundedPair[];
  classifier_error?: string;
}

export async function extractPairs(
  draft: string,
  model: string,
): Promise<ExtractedPair[] | null> {
  const prompt = buildExtractPrompt(draft.slice(-MAX_SOURCE_CHARS));
  try {
    const { object } = await generateObject({
      model: getLanguageModel(model),
      schema: ExtractSchema,
      prompt,
      temperature: 0,
    });
    return object.pairs;
  } catch {
    return null;
  }
}

/** Normalized token set for cheap lexical overlap checks. */
function tokenSet(s: string): Set<string> {
  return new Set(
    s
      .toLowerCase()
      .replace(/[^a-z0-9\s]/g, " ")
      .split(/\s+/)
      .filter((t) => t.length > 1),
  );
}

function jaccard(a: Set<string>, b: Set<string>): number {
  if (a.size === 0 || b.size === 0) return 0;
  let inter = 0;
  for (const t of a) if (b.has(t)) inter++;
  return inter / (a.size + b.size - inter);
}

/** Alphanumeric-only fold (ASCII + CJK), for entity-mention matching that
 *  tolerates spacing/punctuation drift ("MiniCheck-7B" vs "MiniCheck 7B"). */
function alphanumOnly(s: string): string {
  return s.toLowerCase().replace(/[^a-z0-9一-鿿]/g, "");
}

export async function checkGrounding(
  pair: ExtractedPair,
  topK = 8,
): Promise<GroundedPair> {
  // Retrieve on the PROPOSITION, not the bare entity name. The proposition is
  // far more discriminative — embedding "ALCE" pulls every ALCE claim and the
  // top-K cut then drops the one that actually entails; embedding the
  // proposition ranks the entailing claim first. Falls back to entity if the
  // proposition somehow embeds to nothing useful (kept implicit — both go
  // through the same hybrid scan).
  let queryEmb: Float32Array;
  try {
    queryEmb = await embedOne(`${pair.entity}. ${pair.proposition}`);
  } catch (e: any) {
    return {
      ...pair,
      grounded: false,
      matched_claim_id: null,
      reason: `embed-failed: ${(e?.message || String(e)).slice(0, 200)}`,
    };
  }

  const hits = store.searchHybrid(queryEmb, topK).filter((h) => h.kind === "claim");
  const propTokens = tokenSet(pair.proposition);

  for (const h of hits) {
    if (h.id == null) continue;
    const claim = store.getClaim(h.id);
    if (!claim) continue;
    if (claim.status !== "supported") continue;
    if (claim.superseded_by != null) continue;

    // Fast path: the draft is restating a claim already filed and supported.
    // High lexical overlap with a supported claim's text → grounded without an
    // NLI round-trip. (Anti-demoralizer: "I grounded this, why are you firing".)
    if (jaccard(propTokens, tokenSet(claim.claim_text)) >= 0.8) {
      return {
        ...pair,
        grounded: true,
        matched_claim_id: claim.id,
        reason: `restates supported claim ${claim.id} (lexical overlap)`,
      };
    }

    let quote = "";
    if (claim.dossier_slug) {
      const dossier = store.getDossier(claim.dossier_slug);
      if (dossier) {
        const content = dossier.content || "";
        if (
          claim.source_offset_start != null &&
          claim.source_offset_end != null &&
          claim.source_offset_end > claim.source_offset_start
        ) {
          quote = content.slice(claim.source_offset_start, claim.source_offset_end);
        }
      }
    }

    const source = quote ? `${claim.claim_text}\n\n${quote}` : claim.claim_text;
    const verdict = await verifyClaimAgainstSource(pair.proposition, source);
    if (verdict.status === "supported") {
      return {
        ...pair,
        grounded: true,
        matched_claim_id: claim.id,
        reason: `entailed by claim ${claim.id} (score=${verdict.score.toFixed(2)})`,
      };
    }
  }

  return {
    ...pair,
    grounded: false,
    matched_claim_id: null,
    reason: hits.length
      ? `${hits.length} candidate(s) found but none entailed the proposition`
      : "no candidate claim in KB",
  };
}

// ---------------------------------------------------------------------------
// Session-evidence auto-grounding (issue #21)
//
// Before the gate blocks an ungrounded ASSERT, it checks whether the agent
// already retrieved a source supporting the proposition THIS SESSION — via a
// `Read` (local file), `WebFetch` (page content), or `WebSearch` (result
// snippets). Those live in the transcript's `tool_result` events. If one
// entity-mentioning result NLI-entails the proposition, we snapshot it as a
// dossier, file the claim against it, and pass.
//
// We trust ONLY content that came back from a tool — never the assistant's own
// asserted prose. WebFetch content is model-extracted (not raw HTML), so a
// failed NLI there is the cue to do a real `vouch fetch <url>`.
// ---------------------------------------------------------------------------

export interface SessionSource {
  tool: "Read" | "WebFetch" | "WebSearch";
  /** Source identifier: a file path (Read), URL (WebFetch), or `websearch:<query>`. */
  uri: string;
  content: string;
}

const SESSION_SOURCE_TOOLS = new Set(["Read", "WebFetch", "WebSearch"]);
/** Max candidate session sources NLI-checked per ungrounded pair. */
const SESSION_AUTOGROUND_K = 3;
/** Most-recent unique session sources retained from a transcript. */
const SESSION_SOURCES_MAX = 50;
const TOOL_PRIORITY: Record<string, number> = { Read: 0, WebFetch: 1, WebSearch: 2 };

function toolResultText(content: unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .filter((b: any) => typeof b === "string" || b?.type === "text")
      .map((b: any) => (typeof b === "string" ? b : b?.text ?? ""))
      .join("\n");
  }
  return "";
}

/** Claude Code `Read` results are `cat -n` formatted ("␣␣␣␣␣1\tcontent").
 *  Strip the line-number prefix so the snapshot is a clean copy of the file. */
function stripLineNumbers(text: string): string {
  return text
    .split("\n")
    .map((l) => {
      const m = l.match(/^\s*\d+\t([\s\S]*)$/);
      return m ? m[1]! : l;
    })
    .join("\n");
}

/** Parse a Claude Code transcript JSONL for content the agent retrieved via
 *  Read / WebFetch / WebSearch this session. Returns `[]` for anything that
 *  doesn't look like a Claude Code transcript (the "fall through to current
 *  behavior" path — a thin adapter for other harnesses can be added later). */
export function parseSessionSources(transcriptPath: string): SessionSource[] {
  let raw: string;
  try {
    raw = readFileSync(transcriptPath, "utf8");
  } catch {
    return [];
  }
  const lines = raw.split("\n").filter((l) => l.trim());
  if (!lines.length) return [];

  const events: any[] = [];
  // Pass 1: tool_use id → {name, input} from main-thread assistant events.
  const toolUseById = new Map<string, { name: string; input: any }>();
  for (const line of lines) {
    let ev: any;
    try {
      ev = JSON.parse(line);
    } catch {
      continue;
    }
    events.push(ev);
    if (ev?.type !== "assistant" || ev?.isSidechain) continue;
    const c = ev?.message?.content;
    if (!Array.isArray(c)) continue;
    for (const b of c) {
      if (b?.type === "tool_use" && typeof b.id === "string") {
        toolUseById.set(b.id, { name: String(b.name ?? ""), input: b.input ?? {} });
      }
    }
  }
  if (!events.some((e) => typeof e?.type === "string")) return []; // not a CC transcript

  // Pass 2: tool_result blocks in main-thread user events.
  const out: SessionSource[] = [];
  for (const ev of events) {
    if (ev?.type !== "user" || ev?.isSidechain) continue;
    const c = ev?.message?.content;
    if (!Array.isArray(c)) continue;
    for (const b of c) {
      if (b?.type !== "tool_result" || typeof b.tool_use_id !== "string" || b.is_error) continue;
      const tu = toolUseById.get(b.tool_use_id);
      if (!tu || !SESSION_SOURCE_TOOLS.has(tu.name)) continue;
      let text = toolResultText(b.content).trim();
      if (!text && ev.toolUseResult) {
        const r = ev.toolUseResult;
        if (tu.name === "Read") text = String(r?.file?.content ?? "").trim();
        else if (tu.name === "WebFetch") text = String(r?.result ?? "").trim();
      }
      if (!text) continue;
      let uri = "";
      if (tu.name === "Read") uri = String(tu.input?.file_path ?? "");
      else if (tu.name === "WebFetch") uri = String(tu.input?.url ?? "");
      else if (tu.name === "WebSearch") uri = "websearch:" + String(tu.input?.query ?? "");
      if (!uri || uri === "websearch:") continue;
      if (tu.name === "Read") text = stripLineNumbers(text);
      if (text.length > MAX_SOURCE_CHARS) text = text.slice(0, MAX_SOURCE_CHARS);
      out.push({ tool: tu.name as SessionSource["tool"], uri, content: text });
    }
  }
  // Dedup by (tool|uri), keeping the freshest occurrence; cap total.
  const byKey = new Map<string, SessionSource>();
  for (const s of out) byKey.set(`${s.tool}|${s.uri}`, s);
  return [...byKey.values()].slice(-SESSION_SOURCES_MAX);
}

function sessionSourceTitle(src: SessionSource): string {
  if (src.tool === "Read") {
    const p = src.uri.replace(/^file:\/\//, "");
    const i = Math.max(p.lastIndexOf("/"), p.lastIndexOf("\\"));
    return i >= 0 ? p.slice(i + 1) : p;
  }
  if (src.tool === "WebSearch") return src.uri.replace(/^websearch:/, "web search: ");
  return src.uri;
}

type AutoGroundOutcome =
  | { grounded: true; pair: GroundedPair }
  | { grounded: false; checked: number };

/** Try to ground `pair` against one of the session sources. May throw a
 *  TransientVerifierError (caller fail-opens, as with the KB check). */
async function autoGroundPair(
  pair: GroundedPair,
  sources: SessionSource[],
): Promise<AutoGroundOutcome> {
  const ent = pair.entity.toLowerCase().trim();
  const entAlnum = alphanumOnly(pair.entity);
  const matching = sources
    .filter((s) => {
      const lc = s.content.toLowerCase();
      if (ent && lc.includes(ent)) return true;
      return entAlnum.length >= 4 && alphanumOnly(s.content).includes(entAlnum);
    })
    .sort((a, b) => (TOOL_PRIORITY[a.tool] ?? 9) - (TOOL_PRIORITY[b.tool] ?? 9))
    .slice(0, SESSION_AUTOGROUND_K);
  if (!matching.length) return { grounded: false, checked: 0 };

  let n = 0;
  for (const src of matching) {
    n++;
    const slice = src.content.slice(0, MAX_SOURCE_CHARS);
    const verdict = await verifyClaimAgainstSource(pair.proposition, slice);
    if (verdict.status !== "supported") continue;

    // Entailed by a tool-retrieved source. Snapshot it as a dossier and file
    // the claim against it. The quote-in-dossier invariant holds trivially:
    // the "quote" is the retrieved content itself, which NLI just entailed —
    // the anti-fabrication primitive (a claim must trace to content vouch can
    // see) is satisfied by construction.
    const scope = src.tool === "Read" ? "workspace" : "third-party";
    let dossierEmb: Float32Array | null = null;
    try {
      dossierEmb = await embedOne(slice.slice(0, 8000));
    } catch {
      // non-fatal — dossier persists, just won't be hybrid-searchable
    }
    const slug = store.writeDossier({
      source_url: src.uri,
      source_type: `session-${src.tool.toLowerCase()}`,
      title: sessionSourceTitle(src),
      verbatim_content: slice,
      embedding: dossierEmb,
      scope,
    });
    let claimEmb: Float32Array | null = null;
    try {
      claimEmb = await embedOne(pair.proposition);
    } catch {
      // non-fatal
    }
    const cid = store.recordClaim({
      dossier_slug: slug,
      claim_text: pair.proposition,
      score: verdict.score,
      status: verdict.status,
      source_passage: verdict.source_passage,
      claim_type: "ATOMIC",
      source_offset_start: 0,
      source_offset_end: slice.length,
      embedding: claimEmb,
      verification: "nli-session",
    });
    return {
      grounded: true,
      pair: {
        ...pair,
        grounded: true,
        matched_claim_id: cid,
        auto_grounded: true,
        session_sources_checked: n,
        reason: `auto-grounded from session ${src.tool} of ${src.uri} (claim ${cid}, score=${verdict.score.toFixed(2)})`,
      },
    };
  }
  return { grounded: false, checked: n };
}

export async function runGate(opts: {
  draft: string;
  model: string;
  topK?: number;
  /** Claude Code transcript path; if present and the gate fires, session-
   *  retrieved sources are scanned for auto-grounding before blocking. */
  sessionTranscriptPath?: string;
}): Promise<GateVerdict> {
  const extracted = await extractPairs(opts.draft, opts.model);
  if (extracted === null) {
    return { blocked: false, pairs: [], classifier_error: "extractor failed" };
  }
  if (!extracted.length) {
    return { blocked: false, pairs: [] };
  }
  const checked: GroundedPair[] = [];
  for (const p of extracted) {
    if (p.stance !== "ASSERT") {
      checked.push({
        ...p,
        grounded: true,
        matched_claim_id: null,
        reason: `stance=${p.stance} — no fact to ground`,
      });
      continue;
    }
    try {
      checked.push(await checkGrounding(p, opts.topK));
    } catch (e) {
      const transient = classifyError(e);
      return {
        blocked: false,
        pairs: checked,
        classifier_error:
          transient?.message ||
          (e instanceof Error ? e.message : String(e)).slice(0, 200),
      };
    }
  }

  // Lazy: only parse the transcript / run session NLI when the gate would
  // otherwise fire on something.
  const ungroundedIdx = checked.flatMap((p, i) => (p.grounded ? [] : [i]));
  if (ungroundedIdx.length && opts.sessionTranscriptPath) {
    let sources: SessionSource[] = [];
    try {
      sources = parseSessionSources(opts.sessionTranscriptPath);
    } catch {
      sources = [];
    }
    if (sources.length) {
      for (const i of ungroundedIdx) {
        try {
          const res = await autoGroundPair(checked[i]!, sources);
          checked[i] = res.grounded
            ? res.pair
            : { ...checked[i]!, session_sources_checked: res.checked };
        } catch (e) {
          const transient = classifyError(e);
          return {
            blocked: false,
            pairs: checked,
            classifier_error:
              transient?.message ||
              (e instanceof Error ? e.message : String(e)).slice(0, 200),
          };
        }
      }
    }
  }

  const ungrounded = checked.filter((p) => !p.grounded);
  return { blocked: ungrounded.length > 0, pairs: checked };
}

/**
 * Read the draft text the assistant just emitted to the user, from the most
 * recent main-thread `type: "assistant"` event in a Claude Code transcript.
 *
 * Strict semantics (SUN-62):
 *   - Only direct `text` content blocks of the most recent assistant event
 *     count. Tool-use input strings (filtered by `b.type === "text"`) and
 *     tool-result content (which lives in `type: "user"` events) are
 *     excluded by construction.
 *   - If that most recent assistant event has zero text content (the turn
 *     was purely tool-use), return empty — DO NOT walk back to an earlier
 *     assistant turn. Earlier-turn text is no longer the user-visible draft.
 *   - Sidechain (subagent) assistant events are skipped.
 *
 * `VOUCH_GATE_DEBUG=1` prints the extracted draft to stderr.
 */
export function lastAssistantText(transcriptPath: string): string {
  const raw = readFileSync(transcriptPath, "utf8").trim();
  if (!raw) return "";
  const lines = raw.split("\n");
  for (let i = lines.length - 1; i >= 0; i--) {
    let ev: any;
    try {
      ev = JSON.parse(lines[i]!);
    } catch {
      continue;
    }
    if (ev?.type !== "assistant") continue;
    if (ev?.isSidechain) continue;
    const c = ev?.message?.content;
    let text = "";
    if (typeof c === "string") {
      text = c;
    } else if (Array.isArray(c)) {
      text = c
        .filter((b: any) => b?.type === "text")
        .map((b: any) => b?.text ?? "")
        .join("\n");
    }
    text = text.trim();
    if (process.env.VOUCH_GATE_DEBUG === "1") {
      const preview = text.length > 500 ? text.slice(0, 500) + "…" : text;
      process.stderr.write(
        `[vouch-gate-debug] last assistant text (${text.length} chars): ${JSON.stringify(preview)}\n`,
      );
    }
    return text;
  }
  if (process.env.VOUCH_GATE_DEBUG === "1") {
    process.stderr.write(
      `[vouch-gate-debug] no assistant event found in transcript\n`,
    );
  }
  return "";
}

/**
 * Read the most-recent assistant text from the transcript, with a freshness
 * guard against transcript-flush race conditions.
 *
 * Race fix: Claude Code's Stop hook can fire before the just-finished
 * assistant turn is appended to the transcript JSONL. If the gate reads at
 * that moment, the "latest" assistant event in the file is actually the
 * PREVIOUS turn — any factual claim there gets re-flagged even though the
 * user-visible draft no longer contains it.
 *
 * Strategy: poll until the latest assistant event in the transcript has a
 * timestamp within `freshThresholdMs` of now. If after `maxWaitMs` no fresh
 * event has appeared, mark the read as stale. Caller (runGateCli) uses the
 * stale flag to fail-open rather than block on phantom claims.
 *
 * Defaults: poll every 50ms up to 1500ms; 10s freshness threshold (Stop
 * hook always fires right after an assistant turn ends, so the latest
 * event MUST be very recent — if it isn't, the transcript hasn't caught up).
 */
export interface AssistantTurnRead {
  text: string;
  timestamp: string | null;
  isFresh: boolean;
}

function readLastAssistantEvent(
  transcriptPath: string,
): { text: string; timestamp: string | null } {
  let raw: string;
  try {
    raw = readFileSync(transcriptPath, "utf8").trim();
  } catch {
    return { text: "", timestamp: null };
  }
  if (!raw) return { text: "", timestamp: null };
  const lines = raw.split("\n");
  for (let i = lines.length - 1; i >= 0; i--) {
    let ev: any;
    try {
      ev = JSON.parse(lines[i]!);
    } catch {
      continue;
    }
    if (ev?.type !== "assistant") continue;
    if (ev?.isSidechain) continue;
    const c = ev?.message?.content;
    let text = "";
    if (typeof c === "string") {
      text = c;
    } else if (Array.isArray(c)) {
      text = c
        .filter((b: any) => b?.type === "text")
        .map((b: any) => b?.text ?? "")
        .join("\n");
    }
    return { text: text.trim(), timestamp: typeof ev.timestamp === "string" ? ev.timestamp : null };
  }
  return { text: "", timestamp: null };
}

export async function readLatestAssistantTurn(
  transcriptPath: string,
  opts: { maxWaitMs?: number; intervalMs?: number; freshThresholdMs?: number } = {},
): Promise<AssistantTurnRead> {
  const maxWaitMs = opts.maxWaitMs ?? 1500;
  const intervalMs = opts.intervalMs ?? 50;
  const freshThresholdMs = opts.freshThresholdMs ?? 10000;
  const start = Date.now();
  let last: { text: string; timestamp: string | null } = { text: "", timestamp: null };

  while (true) {
    last = readLastAssistantEvent(transcriptPath);
    if (last.timestamp) {
      const ts = Date.parse(last.timestamp);
      if (!Number.isNaN(ts)) {
        const age = Date.now() - ts;
        if (age >= 0 && age < freshThresholdMs) {
          if (process.env.VOUCH_GATE_DEBUG === "1") {
            process.stderr.write(
              `[vouch-gate-debug] latest assistant event is ${age}ms old; fresh\n`,
            );
          }
          return { ...last, isFresh: true };
        }
      }
    }
    if (Date.now() - start >= maxWaitMs) break;
    await new Promise((r) => setTimeout(r, intervalMs));
  }

  if (process.env.VOUCH_GATE_DEBUG === "1") {
    const age = last.timestamp
      ? `${Date.now() - Date.parse(last.timestamp)}ms`
      : "unknown";
    process.stderr.write(
      `[vouch-gate-debug] no fresh assistant event after ${maxWaitMs}ms (latest age: ${age}); marking stale → fail-open\n`,
    );
  }
  return { ...last, isFresh: false };
}

export function readStdinJson(): any {
  try {
    return JSON.parse(readFileSync(0, "utf8"));
  } catch {
    return {};
  }
}

export interface GateRunOptions {
  /** Path to a Claude Code transcript JSONL. */
  transcriptPath?: string;
  /** Stop-hook payload JSON read from stdin. Used to derive transcriptPath. */
  hookPayload?: any;
  /** Direct draft text — bypasses transcript reading (for tests / piping). */
  draft?: string;
  model: string;
  topK?: number;
  strict: boolean;
  /** Env var name; if set to "1" the gate exits 0 immediately. */
  bypassEnv: string;
}

export interface GateRunResult {
  verdict: GateVerdict;
  exitCode: 0 | 2;
  /** Human-readable block message, written to stderr by the CLI. */
  message?: string;
}

export async function runGateCli(opts: GateRunOptions): Promise<GateRunResult> {
  if (opts.bypassEnv && process.env[opts.bypassEnv] === "1") {
    return { verdict: { blocked: false, pairs: [] }, exitCode: 0 };
  }

  let draft = opts.draft;
  let transcriptPath = opts.transcriptPath;
  if (!draft && !transcriptPath && opts.hookPayload) {
    if (typeof opts.hookPayload.transcript_path === "string") {
      transcriptPath = opts.hookPayload.transcript_path;
    }
  }

  const transcriptAvailable = !!transcriptPath && existsSync(transcriptPath);
  if (!draft && transcriptPath) {
    if (!transcriptAvailable) {
      return { verdict: { blocked: false, pairs: [] }, exitCode: 0 };
    }
    try {
      const turn = await readLatestAssistantTurn(transcriptPath);
      if (!turn.isFresh) {
        // Transcript-flush race: the just-finished turn is not yet in the
        // file (or no recent turn exists). Fail-open rather than block on
        // potentially-stale prior-turn content.
        return { verdict: { blocked: false, pairs: [] }, exitCode: 0 };
      }
      draft = turn.text;
    } catch {
      return { verdict: { blocked: false, pairs: [] }, exitCode: 0 };
    }
  }

  if (!draft?.trim()) {
    return { verdict: { blocked: false, pairs: [] }, exitCode: 0 };
  }

  const verdict = await runGate({
    draft,
    model: opts.model,
    topK: opts.topK,
    sessionTranscriptPath: transcriptAvailable ? transcriptPath : undefined,
  });
  if (!verdict.blocked) {
    const autoGrounded = verdict.pairs.filter((p) => p.auto_grounded);
    return {
      verdict,
      exitCode: 0,
      ...(autoGrounded.length ? { message: formatAutoGroundMessage(autoGrounded) } : {}),
    };
  }
  if (!opts.strict) {
    return { verdict, exitCode: 0, message: formatBlockMessage(verdict, true) };
  }
  return { verdict, exitCode: 2, message: formatBlockMessage(verdict, false) };
}

function formatAutoGroundMessage(autoGrounded: GroundedPair[]): string {
  const lines = autoGrounded.map(
    (p) => `  • [verified: ${p.matched_claim_id}] ${p.entity} — ${p.reason}`,
  );
  return (
    `[vouch-gate] auto-grounded ${autoGrounded.length} claim(s) from source(s) you ` +
    `retrieved this session — recorded in the KB:\n${lines.join("\n")}\n`
  );
}

function formatBlockMessage(verdict: GateVerdict, advisory: boolean): string {
  const ungrounded = verdict.pairs.filter((p) => !p.grounded);
  const lines = ungrounded.map((p) => {
    const checked =
      p.session_sources_checked && p.session_sources_checked > 0
        ? ` — checked ${p.session_sources_checked} session source(s), none entailed`
        : "";
    return `  • ${p.entity}: "${p.proposition.slice(0, 200)}" (${p.reason})${checked}`;
  });
  const anySessionChecked = ungrounded.some((p) => (p.session_sources_checked ?? 0) > 0);
  const autoGrounded = verdict.pairs.filter((p) => p.auto_grounded);
  const agNote = autoGrounded.length
    ? `\n(also auto-grounded from session source(s): ${autoGrounded
        .map((p) => `[verified: ${p.matched_claim_id}] ${p.entity}`)
        .join(", ")})`
    : "";
  const header = advisory
    ? `[vouch-gate advisory] Ungrounded named-entity claim(s) in draft:`
    : `[vouch-gate] Detected ungrounded factual claim(s) about named external entities.`;
  const fetchHint = anySessionChecked
    ? ` — for a WebFetch result that didn't entail, \`vouch fetch <url>\` pulls the raw page (WebFetch returns model-extracted text)`
    : "";
  const guidance = advisory
    ? ""
    : `\nBefore answering, ground each claim:\n` +
      `  • vouch search "<keyword>" — check the KB\n` +
      `  • vouch fetch <url> — pull the source${fetchHint}\n` +
      `  • vouch claim "<text>" --type ATOMIC --dossier <slug> --source-quote "..."\n` +
      `Or hedge explicitly with "(unverified, from training memory)" near the claim.\n`;
  return `${header}\n${lines.join("\n")}${agNote}${guidance ? "\n" + guidance : "\n"}`;
}
