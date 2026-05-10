/** Stop-hook confabulation gate.
 *
 * Pipeline:
 *   1. Read the last assistant text from a Claude Code transcript.
 *   2. Use a fast LLM to extract {proposition, stance, entity} triples — every
 *      proposition the draft makes about a NAMED EXTERNAL ENTITY, labelled
 *      with its stance (ASSERT, HEDGE, SPECULATE, NEGATE, COMPARE, META,
 *      RETRACT, REFER). Workspace context and common knowledge are excluded
 *      at extraction.
 *   3. For each ASSERT triple: hybrid-search vouch's claim KB; for each
 *      supported candidate run NLI (proposition vs claim_text + dossier
 *      source quote). Grounded if ANY supported claim entails the proposition.
 *      Non-ASSERT triples short-circuit as grounded — there is no fact to
 *      verify against a source.
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
  return `You are a fact-grounding gate. The assistant has just produced a draft response. Your job is to extract every PROPOSITION the draft makes about a NAMED EXTERNAL ENTITY and label its STANCE.

The unit of analysis is "proposition with stance", NOT "entity reference". A draft can mention an entity name without making any factual assertion about it — those cases must be labelled accordingly, not treated as ASSERT.

A NAMED EXTERNAL ENTITY is a third-party dataset, paper, product, library, model, company, person, or benchmark whose properties live outside the assistant's workspace.

For each proposition, return:
  - proposition: 1-sentence verbatim or close paraphrase of what the draft says
  - entity: short canonical name of the entity the proposition is about
  - stance: exactly one of:
      ASSERT    — declarative factual claim ("X has 100 features", "X beats Y on Z by 3 points")
      HEDGE     — assertion paired with an explicit caveat in the same sentence/clause: "(unverified)", "from training memory", "without verifying", "I haven't verified", "let me verify", "凭印象", or equivalent
      SPECULATE — hypothetical, conditional, or modal ("X might do Y", "if X then Y", "X would probably ...")
      NEGATE    — explicit denial ("X does not support Y", "X is not a Z")
      COMPARE   — entity is the comparison topic, no factual outcome asserted ("we evaluated against X", "comparing X vs Y vs Z")
      META      — reflective reference to a prior claim ("earlier I said X is Y", "the claim about X above")
      RETRACT   — explicit cancellation of a prior claim ("retracting earlier claim about X", "ignore my earlier point about X")
      REFER     — name used as a label only, no proposition attached ("see also X", "thanks to X")

DECISION RULES:
  - If a hedge token appears in the same sentence or clause as the assertion about the entity, the stance is HEDGE — even if the surface form looks declarative. Hedge wins over ASSERT.
  - For "X vs Y" patterns: if the sentence names what is being compared without asserting the outcome, both X and Y are COMPARE. If the sentence asserts an outcome ("X beat Y by 3 points"), that is ASSERT about X.
  - A retraction sentence re-mentions the entity by necessity. The stance is RETRACT, never ASSERT, regardless of how the entity is described inside the retraction.
  - Annotations like "(claim N)", "(vouch claim N)", "(claim_id: N)", "(claim_ids: A,B,C)", "(supported NLI)" are explicit grounding handles — treat the same as a hedge token: stance becomes META (the assertion is bookkept against a verified claim_id, not a fresh assertion to ground).

EXCLUDE entirely (do NOT return any triple):
  - Generic common knowledge / textbook background. The bar is "would a textbook in the relevant field state this without citation?" — if yes, skip. Examples that must skip: "Fine-Gray is a statistical model for competing risks", "Fourier transform decomposes a signal into frequencies", "Cox regression models hazard ratios", "Gray test compares cumulative incidence functions". Method-of-X descriptions ("X is a method for Y", "X is a model used for Y", "X is a statistical test for Y") are textbook background.
  - Workspace context — when in doubt, treat as workspace and skip:
      (a) The assistant's own actions, plans, recommendations, framing, or meta-commentary about the conversation itself.
      (b) Properties of any project the assistant is acting as maintainer / author / dogfooder of — its command surface, flags, file paths, function names, build artifacts, test counts, commit hashes, internal architecture, current runtime state, OR FEATURE-SUPPORT / ROADMAP claims about that project ("X is supported in our toolkit", "we have built-in support for Y", "Z is on the roadmap", "our library covers W", "feature V is missing in our package"). Even when X / Y / Z is itself a third-party methodology or external entity, a claim about whether the user's project supports / will support / lacks it is workspace, not external.${projectsLine}
      (c) Anything the assistant plausibly observed via a tool call earlier in the same session (Bash command output, file contents read, git log/diff/show output, test results, HTTP responses, database queries). The transcript itself is the source of those — vouch is not the right gate.
      (d) Forward-looking, hypothetical, or proposed entities that the assistant frames as not-yet-existing ("I'll file ISSUE-X", "a proposed feature would ...", "the planned issue tracks ...").
      (e) Internal issue-tracker IDs (Linear / Jira / GitHub issue numbers) and their described scope when the assistant is filing, summarizing, or proposing them — workspace coordination, not external claims.

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

export async function checkGrounding(
  pair: ExtractedPair,
  topK = 5,
): Promise<GroundedPair> {
  let queryEmb: Float32Array;
  try {
    queryEmb = await embedOne(pair.entity);
  } catch (e: any) {
    return {
      ...pair,
      grounded: false,
      matched_claim_id: null,
      reason: `embed-failed: ${(e?.message || String(e)).slice(0, 200)}`,
    };
  }

  const hits = store.searchHybrid(queryEmb, topK).filter((h) => h.kind === "claim");

  for (const h of hits) {
    if (h.id == null) continue;
    const claim = store.getClaim(h.id);
    if (!claim) continue;
    if (claim.status !== "supported") continue;
    if (claim.superseded_by != null) continue;

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

export async function runGate(opts: {
  draft: string;
  model: string;
  topK?: number;
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

  if (!draft && transcriptPath) {
    if (!existsSync(transcriptPath)) {
      return { verdict: { blocked: false, pairs: [] }, exitCode: 0 };
    }
    try {
      draft = lastAssistantText(transcriptPath);
    } catch {
      return { verdict: { blocked: false, pairs: [] }, exitCode: 0 };
    }
  }

  if (!draft?.trim()) {
    return { verdict: { blocked: false, pairs: [] }, exitCode: 0 };
  }

  const verdict = await runGate({ draft, model: opts.model, topK: opts.topK });
  if (!verdict.blocked) {
    return { verdict, exitCode: 0 };
  }
  if (!opts.strict) {
    return { verdict, exitCode: 0, message: formatBlockMessage(verdict, true) };
  }
  return { verdict, exitCode: 2, message: formatBlockMessage(verdict, false) };
}

function formatBlockMessage(verdict: GateVerdict, advisory: boolean): string {
  const ungrounded = verdict.pairs.filter((p) => !p.grounded);
  const lines = ungrounded.map(
    (p) => `  • ${p.entity}: "${p.proposition.slice(0, 200)}" (${p.reason})`,
  );
  const header = advisory
    ? `[vouch-gate advisory] Ungrounded named-entity claim(s) in draft:`
    : `[vouch-gate] Detected ungrounded factual claim(s) about named external entities.`;
  const guidance = advisory
    ? ""
    : `\nBefore answering, ground each claim:\n` +
      `  • vouch search "<keyword>" — check the KB\n` +
      `  • vouch fetch <url> — pull the source\n` +
      `  • vouch claim "<text>" --type ATOMIC --dossier <slug> --source-quote "..."\n` +
      `Or hedge explicitly with "(unverified, from training memory)" near the claim.\n`;
  return `${header}\n${lines.join("\n")}${guidance ? "\n" + guidance : "\n"}`;
}
