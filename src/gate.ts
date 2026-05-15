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
 * Session-evidence auto-grounding (issues #21, #22):
 *   When an ASSERT proposition isn't grounded by the KB, the gate — before
 *   blocking — scans the transcript's `tool_result` events for content the
 *   agent demonstrably retrieved THIS SESSION: via the `Read` / `WebFetch` /
 *   `WebSearch` tools, or via `Bash` when the command is an unambiguous
 *   single-file read (`cat F`, `head … F`, `tail … F`, `< F`) — agents `cat`
 *   files at least as often as they use the `Read` tool. If one of those
 *   entity-mentioning results NLI-entails the proposition, the gate snapshots
 *   it as a dossier, files the claim against it, and passes — turning a
 *   false-positive fire into a recorded grounding instead of training the agent
 *   to dodge. Lazy (only on a fire), bounded (entity-matching results only,
 *   K-capped NLI, size-capped per result), and it ONLY trusts content that came
 *   back from a tool — never the assistant's own asserted prose. Bash output
 *   that pipes / globs / merges files / runs another program is opaque and is
 *   NOT ingested (those claims block as normal); a path mis-parse can only make
 *   the dossier's provenance metadata less precise, never produce a wrong
 *   grounding, because NLI still gates it. Anything not retrieved via a tool
 *   still blocks exactly as before.
 *
 * Tagged-derived-claim harvesting (issue #23):
 *   Two passes share the hook — block-check first; if the draft PASSES, the
 *   gate then harvests the *derived* claims the skill's tag table already
 *   makes the agent write inline: `[inference-from: <ids>]` → INFERENCE,
 *   `[synthesis-of: <ids>]` → SYNTHESIS, `[interpretation: <id>]` →
 *   INTERPRETATION, `[hypothesis]` → HYPOTHESIS (a `; score: <0..1>` suffix
 *   overrides the default soft_score). `[verified: <id>]` never files anything
 *   — it only flags a dangling / unsupported id. These are the agent's own
 *   deductions / paraphrases, not source-entailments, so there is NO NLI step
 *   (a Stop hook firing on every passing turn can't afford one, and the only
 *   error mode — citing the wrong upstream id — is a tagging mistake regardless
 *   of who runs the command); they're recorded with status `recorded` and
 *   `verification: tag-harvest` so the audit trail shows they were auto-harvested
 *   and the grounding check (which requires status=supported) won't treat them
 *   as evidence. Dedup on (claim_text, claim_type, depends_on-set) — re-emitting
 *   a draft after a block doesn't double-file. Untagged prose is never touched.
 */
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, appendFileSync, readFileSync } from "node:fs";
import { homedir } from "node:os";
import { basename, dirname, join } from "node:path";
import { generateObject } from "ai";
import { z } from "zod";

import { DB_PATH, MAX_SOURCE_CHARS } from "./config.ts";
import { getLanguageModel } from "./providers.ts";
import { embedOne } from "./embedder.ts";
import { classifyError, verifyClaimAgainstSource, verifyClaimsBatch, verifyContradiction } from "./verifier.ts";
import { findCounterEvidence, COUNTER_EVIDENCE_MAX_HITS } from "./counter.ts";
import { safeHarvest, type HarvestResult } from "./harvest.ts";
import * as store from "./store.ts";
import { suggestVerification, renderSuggestionLine } from "./suggest.ts";
import { searchWithText, type ExaCandidate } from "./exa.ts";
import type { ClaimType } from "./types.ts";

export const DEFAULT_GATE_MODEL =
  process.env.VOUCH_GATE_MODEL || "vertex_ai/gemini-3.1-flash-lite";

// ANSI helpers. Honor https://no-color.org/ via NO_COLOR env. The Stop
// hook's output is piped to Claude Code, which renders ANSI escapes in
// its UI; agents reading the feedback see the bold-red entity name as
// visual focus. Without TTY detection because the pipe context isn't a
// TTY but the rendering layer still handles escapes — so we default-ON.
const VOUCH_NO_COLOR = !!process.env.NO_COLOR;
function boldRed(s: string): string {
  return VOUCH_NO_COLOR ? s : `\x1b[1;31m${s}\x1b[0m`;
}

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
  "WORKSPACE",
] as const;
export type Stance = (typeof STANCES)[number];

// The extractor prompt instructs the LLM to skip workspace items (return no
// triple), so WORKSPACE is not a valid extractor output — only a post-filter
// downgrade target.
const StanceEnum = z.enum([
  "ASSERT",
  "OPINION",
  "HEDGE",
  "SPECULATE",
  "NEGATE",
  "COMPARE",
  "META",
  "RETRACT",
  "REFER",
]);

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
      OPINION   — genuinely subjective evaluation where there is no empirical fact to check: "X is elegant", "X is the best choice for our use case", "X is overrated", "I'd reach for Y here", aesthetic or preference judgments. NOT ranking/standing claims that are empirically verifiable ("X is the de-facto standard", "X is the canonical implementation", "X is state-of-the-art on Y", "X is the most-downloaded") — those are ASSERT.
      HEDGE     — assertion + caveat in the same sentence/clause: "(unverified)", "from training memory", "without verifying", "I haven't verified", "let me verify", "凭印象", or equivalent. Also: a block-level hedge prefix or trailing caveat that scopes an entire block of factual sentences (see DECISION RULES below).
      SPECULATE — hypothetical / conditional / modal ("X might do Y", "if X then Y")
      NEGATE    — explicit denial ("X does not support Y")
      COMPARE   — entity is the comparison topic without an asserted outcome ("we evaluated against X", "X vs Y")
      META      — reflective reference to a prior claim ("earlier I said X is Y")
      RETRACT   — explicit cancellation ("retracting earlier claim about X")
      REFER     — pure mention / bare naming with no verifiable predicate attached: "I'd start with X", "see the FEVER paper", "FEVER, ALCE, FActScore — all benchmarks in this space" (used as a bare list). Boundary with ASSERT: a generic category gloss ("X is a [language] library for [domain]", "X is a [tool/benchmark/framework]") stays REFER because it carries no specific, non-obvious, checkable predicate. The moment a specific verifiable predicate is attached ("FEVER is used for fact verification against Wikipedia", "ALCE measures citation quality on open-generation tasks", "X requires Python 3.11+", "X was released in 2021", "X is the first library to do Y", "X scored 58% on benchmark Y"), it's ASSERT, not REFER.

DECISION RULES:
  - Hedge tokens in the same sentence/clause → stance is HEDGE, even if surface looks declarative. Hedge wins over ASSERT.
  - Block-level hedge scoping: a sentence that is a pure caveat about a group of claims ("From memory, and I haven't verified any of the following", "None of those four are things I've verified this session — they're from training memory", "the above is unverified / not checked / from memory", "none of these are verified") scopes ALL factual sentences in its block.
      • Leading block hedge: scope forward to every factual sentence after it, until a clear topic change or paragraph break.
      • Trailing block caveat: scope backward to every factual sentence in the same block.
      • When in doubt whether a caveat is block-level, treat it as scoping the block — a false HEDGE is far less harmful than a false ASSERT-and-fire on something the author explicitly flagged unverified.
  - Evaluative/normative wording → stance is OPINION only when there is no empirical fact to check. "Is X good enough for us" is OPINION; "does X score ≥ 0.8" is ASSERT. Ranking/standing claims dressed in evaluative phrasing ("the de-facto standard", "the canonical implementation", "the most-downloaded", "state-of-the-art on Y") are checkable empirical assertions → ASSERT. On close calls between ASSERT and OPINION/REFER, choose ASSERT — grounding will pass it through cheaply if it is in fact grounded.
  - A claim about the model currently running this session (its capabilities, context budget, speed, resource use while doing the present work) → that is meta-commentary about the conversation → WORKSPACE, do not return a triple. The running model is not a third-party external entity here.
  - A sentence whose predicate is an action performed by the agent / assistant / "this session" in the present work — closed / opened / merged / reopened / ran / executed / pushed / committed / filed / created / dispatched / edited / wrote / dossiered / fetched an issue, PR, commit, branch, file, command, Linear issue, etc. — the named entities it mentions (GitHub, a repo name, an issue/PR number, a commit hash, a CLI tool name like \`gh\`/\`git\`/\`bun\`, a file path, a Linear ID) are context of that agent action, not the subject of a verifiable third-party claim → WORKSPACE, do not return a triple. The agent's own session actions are not third-party entities here.
  - Agent-observations about the agent's own tools, loop, or runtime behavior in the present work — e.g., "Gemini emits 2–4 search calls a turn", "the verifier took ~3.7s per call", "the search returned 3 hits" — describe the agent's own run, not a third-party entity → WORKSPACE, do not return a triple.
  - Harness, configuration, and fixture statements — e.g., "the v3 benchmark harness uses Gemini Pro 3.1 as the generator", "vertex_ai/gemini-3.1-pro-preview is the verifier model in this setup", "fixture X5 covers the hedge case", "test suite T runs in CI" — describe the agent's own setup or test artifacts, not a third-party claim → WORKSPACE, do not return a triple.
  - Command or tool output reported verbatim — e.g., "\`bun test\` reported 168 pass and 0 fail", "the build emitted 251 modules", "\`git log\` shows 3 commits" — the output is observational data from the agent's own environment, not an assertion about a third-party entity → WORKSPACE, do not return a triple.
  - Assets the user maintains or authors — repositories under the user's own GitHub handle, packages the user maintains, and the user's own eval/corpus files (e.g., "corpus draft cc-10 yielded 11 propositions", "the labels.jsonl"). Claims about these assets describe the user's own work, not a third-party entity → WORKSPACE, do not return a triple. Contrast: a claim about a package the user does NOT maintain (e.g., lodash-es, bitsandbytes, vLLM) is a real ASSERT about a third-party entity. When in genuine doubt between WORKSPACE and ASSERT, prefer ASSERT.
  - Mention-not-use: a sentence that quotes a claim in order to discuss it ("the fire on 'X is Y' was a good catch", "your earlier claim that 'Z requires W'") is not asserting the quoted proposition itself → WORKSPACE, do not return a triple for the quoted claim. If the surrounding sentence makes its own assertion about the entity, that surrounding assertion is evaluated separately.
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

EXTRACTIVE FRAMING IS NOT A WORKSPACE SIGNAL (#48):
  A draft framed as "Based on the text provided, X is Y" / "According to the passage, X is Y" / "The text states that X is Y" still makes a verifiable factual claim about the named entity X. DO extract it as ASSERT (or whichever stance fits the claim shape) — the downstream grounding step will verify whether the cited source actually entails the predicate. The extractive framing is the agent's CLAIM of source, not proof that the source supports it. Do NOT skip the proposition merely because the agent attributed it to a provided passage.

  Counter-example to internalize:
    DRAFT: "Based on the text provided, the answer is the p-adic norm gets smaller when a number is prime."
    EXTRACT: { entity: "p-adic norm", stance: "ASSERT", proposition: "The p-adic norm gets smaller when a number is prime." }
    Reasoning: the draft asserts a specific property of a named entity. Even though framed as "based on the text", the proposition is a checkable claim about p-adic-norm + prime — and grounding will verify whether the source actually says that (it doesn't — the source says "multiplied by p"). DO extract.

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
  /** If set by reclassifyWorkspaceMeta, the rule number (1-3) that caused the
   *  downgrade from ASSERT to WORKSPACE. */
  reclassifiedRule?: number;
  /** If set by escalateHedgeAssertions, this pair was originally stance=HEDGE
   *  but was escalated to ASSERT because its sentence is a fact-shape claim
   *  carrying a trailing caveat like "(unverified, from training memory)". */
  escalatedFromHedge?: boolean;
  /** If set by collapseSofterStancesToAssert (env-gated experiment), this
   *  pair was originally a "softer" assertion stance (HEDGE/OPINION/
   *  SPECULATE/NEGATE/COMPARE) that was collapsed to ASSERT under the
   *  single-principle rule. */
  stanceCollapsedFrom?: string;
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
  /** True when this pair was resolved by auto-grounding against a session
   *  source (tool-result OR user-prompt). Distinguish persistence behavior
   *  via auto_grounded_kind (#46). */
  auto_grounded?: boolean;
  /** Discriminator for `auto_grounded`: `"tool-result"` (Read / WebFetch /
   *  WebSearch / Bash) writes a dossier + records the claim in vouch's KB —
   *  the agent retrieved this and it is auditable as evidence. `"user-prompt"`
   *  matches user-pasted context as a faithfulness-source (no KB write); the
   *  agent's proposition is consistent with what the user provided, but vouch
   *  did not verify the user's source. Trust tier is the user's responsibility,
   *  not vouch's (#46). */
  auto_grounded_kind?: "tool-result" | "user-prompt";
  /** When the gate fired and a session source had lexical overlap with the
   *  proposition but did NOT entail it, this carries the best-matching source
   *  so the CLI can suggest `vouch attest --from-session-tool`. */
  hint?: { tool_use_id: string; uri: string; overlap: number };
  /** Set when the session-ledger contradiction check (#43) found a prior
   *  ASSERT in this transcript that semantically denies the new one. The
   *  pair is flipped to grounded:false and the gate fires; the message
   *  guides the agent to retract or supersede. */
  contradicts_session?: {
    old_turn: number;
    old_claim: number;
    old_proposition: string;
    score: number;
    reason: string;
  };
  /** Top KB candidates that the gate's grounding step considered but that
   *  NLI did NOT entail. Surfaced in the fire message (#50 E) so the agent
   *  sees what the KB has on this entity in-line — no need to re-run
   *  `vouch search`. Capped at 2 to keep the fire scannable. Sorted by
   *  NLI score descending (closest-miss first — most likely fixable via
   *  rephrasing or claim-against-existing-dossier). */
  kb_candidates?: Array<{
    claim_id: number;
    claim_text: string;
    dossier_slug: string | null;
    nli_reason: string;
    nli_score: number;
  }>;
  /** Counter-evidence: KB claims that CONTRADICT this proposition with
   *  high-confidence `verifyContradiction` score. Surfaced in the fire
   *  message even when the pair was otherwise grounded — agent must
   *  reconcile, supersede, or hedge before passing. Only populated when
   *  VOUCH_GATE_COUNTER_EVIDENCE=1. */
  counter_evidence?: Array<{
    claim_id: number;
    claim_text: string;
    dossier_slug: string | null;
    contradiction_score: number;
    contradiction_reason: string;
  }>;
}

export interface GateVerdict {
  blocked: boolean;
  pairs: GroundedPair[];
  classifier_error?: string;
  /** Derived claims auto-harvested from the draft's tags. Only populated on a
   *  PASSING draft (block-check first; if passing, harvest) and only when there
   *  was something to report. See `harvestDerivedClaims`. */
  harvest?: HarvestResult;
  /** Revise-check: how this turn's draft addressed prior-turn fires
   *  flagged `awaiting_revise=1`. Per-action counts plus the list of
   *  entities still unaddressed (the candidate-dodge surface).
   *  `suspectedSilentRephrase` is a sub-count of would-be `remove`
   *  outcomes that look like the entity was excised but the underlying
   *  claim re-asserted in different wording (cosine ≥
   *  SILENT_REPHRASE_COSINE between fired and a new proposition). */
  reviseCheck?: {
    addressedFetch: number;
    addressedHedge: number;
    addressedRemove: number;
    suspectedSilentRephrase: number;
    stillUnaddressed: Array<{ entity: string; turn_idx: number; proposition: string }>;
  };
  /** True when the watchdog budget was exceeded before all propositions could
   *  be checked. The pairs array contains only the propositions processed so
   *  far; remaining propositions were not checked. */
  incomplete?: boolean;
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

// ---------------------------------------------------------------------------
// Deterministic workspace-meta post-filter (issue #40)
// ---------------------------------------------------------------------------

/** Workspace projects come ENTIRELY from `VOUCH_GATE_WORKSPACE_PROJECTS`
 *  (comma-separated). No hardcoded defaults — historic base set
 *  `[vouch, comprisk, crforest, js-toml, redacted-proj]` was the original
 *  author's projects and constituted user-private bakeage in shared source
 *  (cleaned up 2026-05-13). Empty env → no workspace projects → Rule 1
 *  self-reference checks are no-ops. */
function getWorkspaceProjects(): Set<string> {
  return new Set(
    (process.env.VOUCH_GATE_WORKSPACE_PROJECTS || "")
      .split(",")
      .map((s) => s.trim().toLowerCase())
      .filter(Boolean),
  );
}

/** Optional user/org handle (e.g. GitHub login) for owner-derived self-
 *  reference patterns: `<handle>'s <proj>` and `github.com/<handle>/`.
 *  Env unset → those patterns are no-ops. */
function getUserHandle(): string | null {
  const h = (process.env.VOUCH_GATE_USER_HANDLE || "").trim().toLowerCase();
  return h || null;
}

function escapeRegex(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/** Lowercase, strip punctuation, collapse whitespace. */
function normTokens(s: string): string {
  return s
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

/** Extract quoted / blockquoted text from the draft for mention-not-use
 *  (rule 3).  Each entry is already token-normalised. */
function extractQuotedRegions(draft: string): string[] {
  const lc = draft.toLowerCase();
  const out: string[] = [];
  let m: RegExpExecArray | null;

  // IMPORTANT — match each quoted span CORRECTLY first, then length-filter.
  // A naive `/"([^"]{10,})"/g` would, when it hits a short span (e.g. an inline
  // `code` shorter than the minimum), *skip* that span and instead match the
  // prose between two short spans (` ` text ` ` → captures "text"). So: match
  // single-line `"..."` / `` `...` `` spans (content can't span a newline or
  // contain the delimiter → adjacent delimiters pair correctly), THEN keep the
  // ones ≥10 chars. Also deliberately NOT matching single-quoted regions
  // (`'...'`) — English contraction apostrophes ("it's", "I'd") would be
  // treated as delimiters; single-quote quotation in agent prose is rare.
  const dqRe = /"([^"\n]+)"/g;
  while ((m = dqRe.exec(lc))) if (m[1]!.length >= 10) out.push(normTokens(m[1]!));

  const btRe = /`([^`\n]+)`/g;
  while ((m = btRe.exec(lc))) if (m[1]!.length >= 10) out.push(normTokens(m[1]!));

  const bqRe = /^>\s*(.+)$/gm;
  while ((m = bqRe.exec(lc))) out.push(normTokens(m[1]!));

  return out;
}

/** Deterministic post-filter that re-classifies obvious workspace-meta ASSERTs
 *  to WORKSPACE before grounding.  Recall-biased: when no rule cleanly matches,
 *  the pair is left as ASSERT. */
export function reclassifyWorkspaceMeta(
  pairs: ExtractedPair[],
  draft: string,
): ExtractedPair[] {
  const projects = getWorkspaceProjects();
  const quoted = extractQuotedRegions(draft);

  return pairs.map((pair) => {
    if (pair.stance !== "ASSERT") return pair;

    const prop = pair.proposition;
    const lcProp = prop.toLowerCase();
    const entity = pair.entity
      .toLowerCase()
      .trim()
      .replace(/^[^a-z0-9]+|[^a-z0-9]+$/g, "");

    // ---- Rule 1 — workspace-project SELF-REFERENCE only ------------------
    // Cleaned up 2026-05-13 (two passes):
    //   (i)  Narrowed from "any entity ∈ projects → WORKSPACE" (silenced
    //        ALL behavior/API claims about user's own projects) to a small
    //        set of self-reference / publication-state / asset-locator
    //        patterns.
    //   (ii) Further removed user-private bakeage: dropped hardcoded
    //        `github.com/sunnyadn/`, the hardcoded `sunny'?s` in self-ref
    //        regexes, AND the PyPI / version-publication-state predicates.
    //        Publication-state of a package IS a fact the agent can be
    //        wrong about (e.g. "comprisk 0.3.0 does not exist on PyPI" is
    //        verifiable via `pip index versions comprisk`); it should flow
    //        to grounding, not get silently silenced.
    // What remains is the unambiguous workspace-meta core: explicit
    // self-reference where the speaker claims ownership/authorship of the
    // project. Plus a handle-derived `<handle>'s <proj>` / `github.com/
    // <handle>/` variant gated by VOUCH_GATE_USER_HANDLE (env unset → off).
    const handle = getUserHandle();
    if (handle) {
      const escHandle = escapeRegex(handle);
      if (new RegExp(`\\bgithub\\.com/${escHandle}/`).test(lcProp)) {
        return { ...pair, stance: "WORKSPACE", reclassifiedRule: 1 };
      }
    }
    for (const proj of projects) {
      const esc = escapeRegex(proj);
      // Generic self-reference (works for any user — no name baked in):
      //   `(my|our) <proj>`
      //   `I (built|wrote|maintain|created|own|made|started|developed|authored) <proj>`
      //   `<proj> is (my|our) (project|package|library|cli|tool|repo)`
      if (
        new RegExp(`\\b(my|our) ${esc}\\b`).test(lcProp) ||
        new RegExp(`\\bi (built|wrote|maintain|created|own|made|started|developed|authored) ${esc}\\b`).test(lcProp) ||
        new RegExp(`\\b${esc} is (my|our) (project|package|library|cli|tool|repo)\\b`).test(lcProp)
      ) {
        return { ...pair, stance: "WORKSPACE", reclassifiedRule: 1 };
      }
      // Handle-derived self-reference: `<handle>'s <proj>` /
      // `<proj> is <handle>'s (project|...)`. Gated by VOUCH_GATE_USER_HANDLE.
      if (handle) {
        const escHandle = escapeRegex(handle);
        if (
          new RegExp(`\\b${escHandle}'?s ${esc}\\b`).test(lcProp) ||
          new RegExp(`\\b${esc} is ${escHandle}'?s (project|package|library|cli|tool|repo)\\b`).test(lcProp)
        ) {
          return { ...pair, stance: "WORKSPACE", reclassifiedRule: 1 };
        }
      }
    }
    // Note: bare `entity ∈ projects`, `<proj> on PyPI`, `github.com/<handle>/`
    // (without env), and version-publication-state assertions are all NO
    // LONGER downgrade triggers. They flow to grounding — KB has the source
    // of truth (e.g. `comprisk PyPI downloads`), or the gate fires and the
    // agent must check the repo / package registry before claiming.

    // ---- Rule 2 — agent-machinery phrasings -------------------------------
    const r2a =
      /^the (default )?(pro |flash |pro-preview |flash-lite )?(verifier|extractor|generator|judge|gate|hook|harness|corpus|eval|dry.?run|benchmark harness|v\d+ (benchmark )?(impl(ementation)?|harness|config))\b/;
    if (r2a.test(lcProp)) {
      return { ...pair, stance: "WORKSPACE", reclassifiedRule: 2 };
    }

    const r2b1 =
      /\b(vertex_ai\/|gemini-3\.\d|gemini-\d|gpt-\d|claude-[\w.-]+)[\w.\/-]* is the (verifier|extractor|generator|judge|extractor model|verifier model) model?\b/;
    const r2b2 =
      /\bis the (verifier|extractor|generator|judge) model (in|for) (this|the) (setup|harness|run|installation|config(uration)?|benchmark)\b/;
    if (r2b1.test(lcProp) || r2b2.test(lcProp)) {
      return { ...pair, stance: "WORKSPACE", reclassifiedRule: 2 };
    }

    const r2cMachinery =
      /\bthis (setup|session|installation|benchmark|harness|run|environment|config(uration)?)\b/;
    const r2cSubject =
      /^(this (setup|session|installation|benchmark|harness|run|environment|config(uration)?)\b|the .*? in this (setup|session|installation|benchmark|harness|run|environment|config(uration)?)\b)/;
    if (r2cMachinery.test(lcProp) && r2cSubject.test(lcProp)) {
      return { ...pair, stance: "WORKSPACE", reclassifiedRule: 2 };
    }

    const r2dArtifact =
      /\b(cc-\d+|adv-\d+|labels\.jsonl|frozen_extractions|kb-snapshot|the gate-recall corpus)\b/;
    if (r2dArtifact.test(lcProp.slice(0, 50))) {
      return { ...pair, stance: "WORKSPACE", reclassifiedRule: 2 };
    }

    const r2e =
      /^(claude|this model|the (running )?model)\b.*(context (budget|window)|tokens? per|speed|latency|resource use).*( this (turn|session|run)| while )/;
    if (r2e.test(lcProp)) {
      return { ...pair, stance: "WORKSPACE", reclassifiedRule: 2 };
    }

    // ---- Rule 3 — mention-not-use -----------------------------------------
    // Require an 8-token contiguous overlap inside an actual quoted region
    // (double-quote / backtick / blockquote — NOT single-quote, see
    // extractQuotedRegions). Conservative: a false-keep (the prop fires even
    // though it was being quoted) is cheaper than a false-downgrade (a real
    // assertion gets suppressed because some nearby text was quoted).
    const RULE3_MIN_TOKENS = 8;
    const normProp = normTokens(prop);
    const propTokens = normProp.split(" ");
    if (propTokens.length >= RULE3_MIN_TOKENS) {
      for (let i = 0; i <= propTokens.length - RULE3_MIN_TOKENS; i++) {
        const slice = propTokens.slice(i, i + RULE3_MIN_TOKENS).join(" ");
        for (const quote of quoted) {
          if (quote.includes(slice)) {
            return { ...pair, stance: "WORKSPACE", reclassifiedRule: 3 };
          }
        }
      }
    }

    // NOTE: Rule 3b (entity-in-quoted-region with invented predicate) was
    // shipped in d85eae7 and reverted here after the #35 freeze showed it
    // dropped recall 100% → 69% on the standing corpus. The regression came
    // from requiring the predicate body's content tokens to appear
    // CONTIGUOUSLY in non-quoted prose — but the LLM extractor reorders /
    // simplifies / paraphrases the proposition, so legitimate ASSERTs like
    // `\`lodash-es\` 4.17.23 has high and medium severity alerts` were
    // downgraded because the canonicalized proposition wasn't a contiguous
    // slice of the original draft. Re-design needed (jaccard / minimum
    // overlap percentage with the entity-stripped body, not contiguous
    // match) before re-introducing. Tracked in #44 part 3 redesign.

    return pair;
  });
}

// ---------------------------------------------------------------------------
// Deterministic HEDGE escape-hatch post-filter (issue #42)
// ---------------------------------------------------------------------------

const HEDGE_ESCAPE_BLOCK_PATTERNS = [
  // trailing parenthetical caveat
  /\(\s*unverified\b[^)]*\)\s*\.?\s*$/i,
  /\(\s*(?:from\s+)?(?:training|my)\s+memory[^)]*\)\s*\.?\s*$/i,
  /\(\s*training\s+memory[^)]*\)\s*\.?\s*$/i,
  /\(\s*named\s*,\s*not\s+(?:run|tested|verified|checked)\s+here\s*\)\s*\.?\s*$/i,
  /\(\s*not\s+(?:verified|checked)\b[^)]*\)\s*\.?\s*$/i,
  // trailing comma-separated caveat (no parens)
  /,\s+(?:unverified|from\s+(?:training|my)\s+memory|training\s+memory)\b[^.]*\.?\s*$/i,
  // " — unverified" em-dash trailer
  /\s+[—-]\s+(?:unverified|from\s+(?:training|my)\s+memory)\b[^.]*\.?\s*$/i,
];

const HEDGE_KEEP_PATTERNS = [
  // leading epistemic hedge
  /^\s*(?:i\s+(?:think|believe|suspect|guess)|afaik|probably|maybe|perhaps)\b/i,
  // question form
  /\?\s*$/,
  // explicit uncertainty about identity
  /^\s*i\s+(?:am\s+not|'m\s+not|do\s+not|don'?t)\s+(?:sure|know)\b/i,
  // placeholder/TBD
  /\b(?:TBD|placeholder|TK|FIXME)\b/i,
];

/** Best-effort locate the sentence in `draft` that contains `proposition`. */
function findSentenceContaining(draft: string, proposition: string): string | null {
  const normProp = normTokens(proposition);
  const sentences = draft
    .replace(/\n+/g, " ")
    .split(/(?<=[.!?])\s+/)
    .map((s) => s.trim())
    .filter(Boolean);
  for (const sent of sentences) {
    if (normTokens(sent).includes(normProp)) {
      return sent;
    }
  }
  // Fallback: try the whole draft
  if (normTokens(draft).includes(normProp)) {
    return draft.trim();
  }
  return null;
}

/** Stance-collapse experiment (env-gated by VOUCH_GATE_STANCE_COLLAPSE=1).
 *  Treats the entire family of "softer" assertion stances as ASSERT-equivalent:
 *  HEDGE / OPINION / SPECULATE / NEGATE / COMPARE → ASSERT.
 *  Rationale: each softer stance has historically been a leak path (4a/4b/4c
 *  dodging). Collapsing them to ASSERT means the gate's rule reduces to one
 *  principle — any predicate on a named external entity needs a source —
 *  rather than enumerating escape patterns per stance. WORKSPACE, REFER,
 *  META, RETRACT are preserved (they're not "softer assertions" but
 *  semantically different categories).
 *
 *  Default off so existing behavior is preserved; flip when measurement
 *  confirms the collapse is a net win on the benchmark. */
export function collapseSofterStancesToAssert(pairs: ExtractedPair[]): ExtractedPair[] {
  if (process.env.VOUCH_GATE_STANCE_COLLAPSE !== "1") return pairs;
  const SOFTER = new Set(["HEDGE", "OPINION", "SPECULATE", "NEGATE", "COMPARE"]);
  return pairs.map((pair) =>
    SOFTER.has(pair.stance) ? { ...pair, stance: "ASSERT", stanceCollapsedFrom: pair.stance } : pair,
  );
}

/** Deterministic post-filter that escalates HEDGE → ASSERT when the sentence is
 *  a fact-shape claim carrying a trailing parenthetical/comma caveat.
 *  Recall-biased: when the sentence cannot be located, or no rule matches,
 *  the pair is left as HEDGE. */
export function escalateHedgeAssertions(
  pairs: ExtractedPair[],
  draft: string,
): ExtractedPair[] {
  return pairs.map((pair) => {
    if (pair.stance !== "HEDGE") return pair;
    const sentence = findSentenceContaining(draft, pair.proposition);
    if (!sentence) return pair;
    const isBlock = HEDGE_ESCAPE_BLOCK_PATTERNS.some((re) => re.test(sentence));
    const isKeep = HEDGE_KEEP_PATTERNS.some((re) => re.test(sentence));
    if (isBlock && !isKeep) {
      return { ...pair, stance: "ASSERT", escalatedFromHedge: true };
    }
    return pair;
  });
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

/** True when `text` mentions the same primary entity as the extracted pair.
 *  Used to guard high-confidence embedding shortcuts so we don't ground a
 *  claim about a different entity just because the sentence structure is
 *  similar. */
function sharesPrimaryEntity(pair: ExtractedPair, text: string): boolean {
  const ent = pair.entity.toLowerCase().trim();
  if (ent && text.toLowerCase().includes(ent)) return true;
  const entAlnum = alphanumOnly(pair.entity);
  return entAlnum.length >= 3 && alphanumOnly(text).includes(entAlnum);
}

const WIDER_TOPK = 8;
const COSINE_SHORTCUT = 0.95;
const COSINE_SECONDARY = 0.92;

/** Cosine threshold above which a NEW proposition this turn is treated as
 *  semantically the same claim as a prior fired one — used to distinguish
 *  legitimate topic-drop from "entity removed but claim re-asserted in
 *  different wording" (the silent-rephrase dodge shape). */
const SILENT_REPHRASE_COSINE = 0.85;

/** Per-turn over-confidence advisory thresholds. */
const OVER_CONFIDENT_ASSERT_FLOOR = 3;     // ≥ N asserts this turn triggers the flag check
const HUMILITY_DISPLAY_MIN_TRUTH = 10;     // floor for showing humility ratio (avoid noisy small-N)

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

  const hits = store.searchHybrid(queryEmb, WIDER_TOPK).filter((h) => h.kind === "claim");
  const propTokens = tokenSet(pair.proposition);

  for (let i = 0; i < hits.length; i++) {
    const h = hits[i]!;
    if (h.id == null) continue;
    const claim = store.getClaim(h.id);
    if (!claim) continue;
    if (claim.status !== "supported") continue;
    if (claim.superseded_by != null) continue;

    // Fast path 1: the draft is restating a claim already filed and supported.
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

    // Fast path 2: high-confidence embedding match catches paraphrases the
    // lexical fast-path misses (issue #37). Restricted to primary pool — the
    // wider pool only gets the lower-threshold secondary pass (NLI, not shortcut).
    if (i < topK && h.similarity >= COSINE_SHORTCUT && sharesPrimaryEntity(pair, claim.claim_text)) {
      return {
        ...pair,
        grounded: true,
        matched_claim_id: claim.id,
        reason: `restates supported claim ${claim.id} (embedding cosine=${h.similarity.toFixed(3)})`,
      };
    }

    // Only run NLI on primary pool (topK) or high-cosine secondary candidates.
    if (i >= topK && !(h.similarity >= COSINE_SECONDARY && sharesPrimaryEntity(pair, claim.claim_text))) {
      continue;
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

/** Run up to `limit` async tasks concurrently, returning results in order. */
async function concurrencyLimit<T>(tasks: (() => Promise<T>)[], limit: number): Promise<T[]> {
  const results: T[] = new Array(tasks.length);
  let index = 0;
  async function worker() {
    while (index < tasks.length) {
      const i = index++;
      results[i] = await tasks[i]!();
    }
  }
  const workers = Array(Math.min(limit, tasks.length))
    .fill(0)
    .map(worker);
  await Promise.all(workers);
  return results;
}

interface NliCandidate {
  pairIdx: number;
  claimId: number;
  proposition: string;
  source: string;
}

/** Batch-capable grounding for multiple ASSERT propositions.
 *
 *  Pipeline:
 *    1. Parallel embed+search+lexical-fast-path for all propositions.
 *    2. Collect candidates that need NLI into a flat list.
 *    3. Run one (or a few token-budget-sized) `verifyClaimsBatch` call(s).
 *    4. Map results back. A proposition is grounded if ANY candidate is supported.
 *
 *  Abort handling: checks `abortRef.aborted` before the NLI batch. If already
 *  aborted, returns existing lexical matches and leaves remaining propositions
 *  ungrounded — caller sets `incomplete`.
 */
async function batchGroundAssertions(
  pairs: ExtractedPair[],
  topK: number,
  abortRef?: { aborted: boolean },
): Promise<{ results: GroundedPair[]; incomplete: boolean }> {
  const results: GroundedPair[] = pairs.map((p) => ({
    ...p,
    grounded: false,
    matched_claim_id: null,
    reason: "no candidate claim in KB",
  }));
  if (!pairs.length) return { results, incomplete: false };

  const candidates: NliCandidate[] = [];

  // Phase 1: parallel embed + search + lexical fast-path + embedding shortcut
  await Promise.all(
    pairs.map(async (pair, idx) => {
      let queryEmb: Float32Array;
      try {
        queryEmb = await embedOne(`${pair.entity}. ${pair.proposition}`);
      } catch (e: any) {
        results[idx]!.reason = `embed-failed: ${(e?.message || String(e)).slice(0, 200)}`;
        return;
      }

      // Wider retrieval pool (cheap) — NLI only runs on topK + high-cosine extras.
      const hits = store.searchHybrid(queryEmb, WIDER_TOPK).filter((h) => h.kind === "claim");
      const propTokens = tokenSet(pair.proposition);

      for (let i = 0; i < hits.length; i++) {
        const h = hits[i]!;
        if (h.id == null) continue;
        const claim = store.getClaim(h.id);
        if (!claim) continue;
        if (claim.status !== "supported") continue;
        if (claim.superseded_by != null) continue;

        if (jaccard(propTokens, tokenSet(claim.claim_text)) >= 0.8) {
          results[idx] = {
            ...pair,
            grounded: true,
            matched_claim_id: claim.id,
            reason: `restates supported claim ${claim.id} (lexical overlap)`,
          };
          return; // pair resolved — no NLI needed
        }

        // High-confidence embedding shortcut for paraphrases the lexical path
        // misses (issue #37). Restricted to primary pool.
        if (i < topK && h.similarity >= COSINE_SHORTCUT && sharesPrimaryEntity(pair, claim.claim_text)) {
          results[idx] = {
            ...pair,
            grounded: true,
            matched_claim_id: claim.id,
            reason: `restates supported claim ${claim.id} (embedding cosine=${h.similarity.toFixed(3)})`,
          };
          return;
        }

        // Only add to NLI batch if in primary pool or a high-cosine secondary.
        if (i >= topK && !(h.similarity >= COSINE_SECONDARY && sharesPrimaryEntity(pair, claim.claim_text))) {
          continue;
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
        candidates.push({
          pairIdx: idx,
          claimId: claim.id,
          proposition: pair.proposition,
          source,
        });
      }

      if (hits.length && !results[idx]!.grounded) {
        results[idx]!.reason = `${hits.length} candidate(s) found but none entailed the proposition`;
      }
    }),
  );

  // Phase 2: batched NLI for remaining candidates
  if (candidates.length && !abortRef?.aborted) {
    const batchItems = candidates.map((c) => ({
      claim_text: c.proposition,
      source_passage: c.source,
    }));

    try {
      const batchResults = await verifyClaimsBatch(batchItems);
      // Collect non-entailing candidates per pair so the fire message
      // can surface them inline. Buffer here, prune+sort once after the
      // batch settles (a candidate that DID entail can short-circuit its
      // pair to grounded:true without polluting kb_candidates).
      const candidateBuffer = new Map<number, GroundedPair["kb_candidates"]>();
      for (let i = 0; i < candidates.length; i++) {
        const c = candidates[i]!;
        const r = batchResults[i]!;
        if (r.status === "supported" && !results[c.pairIdx]!.grounded) {
          results[c.pairIdx] = {
            ...pairs[c.pairIdx]!,
            grounded: true,
            matched_claim_id: c.claimId,
            reason: `entailed by claim ${c.claimId} (score=${r.score.toFixed(2)})`,
          };
          continue;
        }
        // Non-supported — capture for inline-KB-candidate surfacing IF this pair is
        // still ungrounded at the end. Note: source_passage carries the
        // verifier's natural-language reason for unsupported claims (the
        // post-#49 invariant: only `supported` paths store verbatim quote).
        const claim = store.getClaim(c.claimId);
        if (!claim) continue;
        // Surface filter: the candidate's claim text must actually mention
        // the firing entity. Primary-pool hits skip the sharesPrimaryEntity
        // gate during NLI (so we find genuine entailments via paraphrase),
        // but for DISPLAY we only want candidates the agent can actually
        // act on — showing "CoBuddy has 131k context" as a KB candidate for
        // a claim about "BridgeNet" is noise, not signal.
        if (!sharesPrimaryEntity(pairs[c.pairIdx]!, claim.claim_text)) continue;
        if (!candidateBuffer.has(c.pairIdx)) candidateBuffer.set(c.pairIdx, []);
        candidateBuffer.get(c.pairIdx)!.push({
          claim_id: c.claimId,
          claim_text: claim.claim_text,
          dossier_slug: claim.dossier_slug || null,
          nli_reason: r.source_passage || "(no reason returned)",
          nli_score: r.score,
        });
      }
      // Sort each pair's buffer by NLI score desc, keep top 2. Attach only
      // when the pair stayed ungrounded (didn't get entailed by another
      // candidate in the same batch).
      for (const [pairIdx, buf] of candidateBuffer.entries()) {
        if (results[pairIdx]!.grounded) continue;
        buf!.sort((a, b) => b.nli_score - a.nli_score);
        results[pairIdx]!.kb_candidates = buf!.slice(0, 2);
      }
    } catch (e) {
      // Transient/system error on the batch — re-throw so runGate can fail-open.
      throw e;
    }
  }

  // Counter-evidence pull: for each grounded pair, also check whether the KB
  // contains a STRONG CONTRADICTING claim on the same entity. Confirming
  // evidence alone is insufficient — if a comparably-supported counter-claim
  // exists, flip to ungrounded so the agent must reconcile, supersede, or
  // hedge. Opt-in because each grounded pair adds an embed + search +
  // up to TOPK verifyContradiction calls (~5-10s/turn). Implementation in
  // src/counter.ts so the same pull is reachable as the `vouch counter`
  // agent-callable CLI verb.
  if (process.env.VOUCH_GATE_COUNTER_EVIDENCE === "1" && !abortRef?.aborted) {
    const groundedIdx = results.flatMap((p, i) => (p.grounded ? [i] : []));
    if (groundedIdx.length) {
      await Promise.all(
        groundedIdx.map(async (idx) => {
          if (abortRef?.aborted) return;
          const pair = results[idx]!;
          const counter = await findCounterEvidence(pair.proposition, pair.entity, {
            excludeClaimId: pair.matched_claim_id ?? undefined,
            maxHits: COUNTER_EVIDENCE_MAX_HITS,
            abortRef,
          });
          if (counter.length > 0) {
            results[idx] = {
              ...pair,
              grounded: false,
              reason:
                `${pair.reason} BUT contradicted by claim ${counter[0]!.claim_id} (score=${counter[0]!.contradiction_score.toFixed(2)}) — counter-evidence requires reconciliation`,
              counter_evidence: counter,
            };
          }
        }),
      );
    }
  }

  return { results, incomplete: !!abortRef?.aborted };
}

// ---------------------------------------------------------------------------
// Session-evidence auto-grounding (issues #21, #22, #39)
//
// Before the gate blocks an ungrounded ASSERT, it checks whether the agent
// already retrieved a source supporting the proposition THIS SESSION — via a
// `Read` (local file), `WebFetch` (page content), `WebSearch` (result
// snippets), or a `Bash` command whose output is NOT model-authored (denylist:
// `echo`, `printf`, `yes`, `:`, `true`, heredocs, and scripting one-liners
// with quoted literal args). Those live in the transcript's `tool_result`
// events. If one entity-mentioning result NLI-entails the proposition, we
// snapshot it as a dossier, file the claim against it, and pass.
//
// We trust ONLY content that came back from a tool — never the assistant's own
// asserted prose. WebFetch content is model-extracted (not raw HTML), so a
// failed NLI there is the cue to do a real `vouch fetch <url>`. Bash output
// that is a verbatim file read (`cat`/`head`/`tail`) gets the file path as its
// URI; everything else gets an honest `session-bash:<cmd>` provenance.
// ---------------------------------------------------------------------------

export interface SessionSource {
  /** `Bash` = any non-model-authored command (denylist, not allowlist).
   *  `UserPrompt` = verbatim text the user typed/pasted into a prior user turn,
   *  used as a faithfulness-check source (NOT world-truth grounding). Distinct
   *  trust tier — see autoGroundPair's UserPrompt branch (no dossier write). */
  tool: "Read" | "WebFetch" | "WebSearch" | "Bash" | "UserPrompt";
  /** Source identifier: a file path (Read / Bash cat-head-tail), `session-bash:<cmd>`
   *  for other Bash commands, URL (WebFetch), `websearch:<query>`, or
   *  `user-prompt:event-<idx>` for UserPrompt sources. */
  uri: string;
  content: string;
  /** The `tool_use_id` from the transcript's `tool_result` block. */
  tool_use_id?: string;
}

const SESSION_SOURCE_TOOLS = new Set(["Read", "WebFetch", "WebSearch", "Bash"]);
/** Max candidate session sources NLI-checked per ungrounded pair. */
const SESSION_AUTOGROUND_K = 3;
/** Most-recent unique session sources retained from a transcript.
 *  If 100 proves insufficient, a relevance-ranked retention (keep sources most
 *  lexically similar to the draft's propositions) is the smarter follow-up. */
const SESSION_SOURCES_MAX = 100;
/** Per-message cap on UserPrompt source content. Large pastes get truncated;
 *  the head of the prompt is what the agent most likely keyed off, and the
 *  NLI window itself caps at SESSION_NLI_MAX_CHARS. */
const USER_PROMPT_MAX_CHARS = 12000;
/** Floor below which a user-typed turn is ignored as a faithfulness source
 *  (e.g. "ok", "可以", "继续" — chit-chat, not paste-context). */
const USER_PROMPT_MIN_CHARS = 80;
const TOOL_PRIORITY: Record<string, number> = { Read: 0, Bash: 0, WebFetch: 1, WebSearch: 2, UserPrompt: 3 };

// --- Bash file-read recognition (issue #22) --------------------------------
//
// Agents read file content via `Bash` (`cat F`, `head -n N F`, `tail F`, `< F`)
// at least as often as via the `Read` tool, and that path was invisible to the
// auto-grounding parser — a claim about something the agent just `cat`-ed would
// still fire. Only an UNAMBIGUOUS single-file read counts: `cat`/`head`/`tail`
// of exactly one file (optionally via a `< F` input redirect). Anything that
// pipes (`cat x | grep y`), substitutes (`$(...)`, backticks), globs
// (`cat *.md`), merges (`cat a b`), or runs another program is opaque and is
// rejected — those claims block as normal. Risk is bounded by NLI still running
// after: a path mis-parse only makes the recorded `source_url` less precise, it
// can never produce a grounding that the content doesn't actually entail.

const HOME = homedir();

/** Expand a leading `~` / `~/…` — the only tilde form bash expands at word
 *  start; other uses are literal and left alone. */
function expandTilde(p: string): string {
  if (p === "~") return HOME;
  if (p.startsWith("~/")) return join(HOME, p.slice(2));
  return p;
}

/** Tokenize a shell command into words, honoring `'…'` / `"…"` quoting and `\`
 *  escapes. Returns `null` the moment it sees any construct meaning the command
 *  does more than read one file verbatim: pipes, `;`, `&`, command / process /
 *  arithmetic substitution, parameter expansion, output redirects, globs, brace
 *  expansion, `!`, `#`. A bare `<` is emitted as its own token (the one
 *  redirect we understand). */
function tokenizeShellRead(cmd: string): string[] | null {
  const tokens: string[] = [];
  let cur = "";
  let started = false; // current token has begun (so we keep an empty "" arg)
  let inSingle = false;
  let inDouble = false;
  const flush = () => {
    if (started) tokens.push(cur);
    cur = "";
    started = false;
  };
  for (let i = 0; i < cmd.length; i++) {
    const ch = cmd[i]!;
    if (inSingle) {
      if (ch === "'") inSingle = false;
      else cur += ch;
      started = true;
      continue;
    }
    if (inDouble) {
      if (ch === '"') inDouble = false;
      else if (ch === "$" || ch === "`") return null; // expansion inside ""
      else if (ch === "\\" && i + 1 < cmd.length && '"\\$`'.includes(cmd[i + 1]!)) cur += cmd[++i]!;
      else cur += ch;
      started = true;
      continue;
    }
    if (/\s/.test(ch)) { flush(); continue; }
    switch (ch) {
      case "'": inSingle = true; started = true; continue;
      case '"': inDouble = true; started = true; continue;
      case "\\":
        if (i + 1 < cmd.length) { cur += cmd[++i]!; started = true; }
        continue;
      case "<":
        flush();
        tokens.push("<");
        continue;
      case "|": case ";": case "&": case "$": case "`":
      case "(": case ")": case "{": case "}":
      case "*": case "?": case "[": case "]":
      case ">": case "#": case "!":
        return null;
      default:
        cur += ch;
        started = true;
    }
  }
  if (inSingle || inDouble) return null;
  flush();
  return tokens;
}

/** A `head`/`tail` option flag we can safely skip — the output then stays a
 *  prefix/suffix of the file, still useful to NLI. */
function isHeadTailFlag(tok: string): boolean {
  return (
    /^-\d+[bkmcKMG]?$/.test(tok) ||                          // -10  -10k
    /^-[nc]\d+[bkmKMG]?$/.test(tok) ||                       // -n10 -c20k
    /^-[ncqvz]+$/.test(tok) ||                               // -n -c -q -v -z (and bundled)
    /^--lines(=.*)?$/.test(tok) ||
    /^--bytes(=.*)?$/.test(tok) ||
    /^--(quiet|silent|verbose|zero-terminated)$/.test(tok)
  );
}
/** Value-bearing forms — the count comes from the following token. */
const FLAG_TAKES_NEXT_COUNT = /^(-[nc]|--lines|--bytes)$/;
const COUNT_VALUE = /^[+-]?\d+[bkmcKMG]?$/;

/** If `cmd` is an unambiguous single-file read (`cat F`, `head … F`, `tail … F`,
 *  optionally with a `< F` input redirect), return the (tilde-expanded) path —
 *  the Bash result text IS that file's content (or a prefix/suffix of it).
 *  Everything else (pipes, substitutions, globs, multiple files, output
 *  redirects, any other command, stdin / fd pseudo-files) → `null`. */
export function safeFileReadCommand(cmd: string): string | null {
  const tokens = tokenizeShellRead((cmd || "").trim());
  if (!tokens || tokens.length < 2) return null;
  let name = tokens[0]!;
  const slash = Math.max(name.lastIndexOf("/"), name.lastIndexOf("\\"));
  if (slash >= 0) name = name.slice(slash + 1);
  if (name !== "cat" && name !== "head" && name !== "tail") return null;

  let positional: string | null = null;
  let redirect: string | null = null;
  let endOfOptions = false;
  for (let i = 1; i < tokens.length; i++) {
    const t = tokens[i]!;
    if (t === "<") {
      const target = tokens[++i];
      if (!target || target === "<" || redirect) return null;
      redirect = target;
      continue;
    }
    if (!endOfOptions && t === "--") { endOfOptions = true; continue; }
    if (!endOfOptions && t.length > 1 && t.startsWith("-")) {
      if (name === "cat") return null; // any `cat` flag (-n, -A, -b, …) transforms output
      if (!isHeadTailFlag(t)) return null; // unrecognized head/tail flag (e.g. tail -f)
      if (FLAG_TAKES_NEXT_COUNT.test(t) && COUNT_VALUE.test(tokens[i + 1] ?? "")) i++; // consume count
      continue;
    }
    if (positional) return null; // a second file → ambiguous concat
    positional = t;
  }
  if (redirect && positional) return null; // both → ambiguous
  const path = redirect ?? positional;
  if (!path || path === "-" || path === "/dev/stdin") return null;
  if (/^\/(dev\/fd|proc\/self\/fd)\//.test(path)) return null;
  return expandTilde(path);
}

/** True when a Bash command's PRIMARY output is text the model typed, not bytes
 *  a system produced. We look only at the command up to the first `&&`, `;`, `|`,
 *  or `>` — a trailing model-authored command (`cat realfile && echo done`) does
 *  not poison a source whose primary output is a real file read. */
function isModelAuthoredOutputCommand(cmd: string): boolean {
  const trimmed = (cmd || "").trim();
  if (!trimmed) return false;

  // Primary command: up to the first control operator.
  const controlRe = /&&|;|\||>/;
  const primary = trimmed.split(controlRe)[0]!.trim();
  if (!primary) return false;

  // First token (strip leading path)
  let firstToken = primary.match(/^(\S+)/)?.[1] ?? "";
  const slash = Math.max(firstToken.lastIndexOf("/"), firstToken.lastIndexOf("\\"));
  if (slash >= 0) firstToken = firstToken.slice(slash + 1);

  // Direct model-authored output commands
  if (["echo", "printf", "yes", ":", "true"].includes(firstToken)) return true;

  // Scripting one-liners with a quoted -c / -e / -pe arg
  if (["python", "python3", "node", "ruby", "perl", "bun", "deno"].includes(firstToken)) {
    const args = primary.split(/\s+/).slice(1);
    for (let i = 0; i < args.length; i++) {
      if (args[i]!.match(/^-[ce]$/) || args[i] === "-pe") {
        const next = args[i + 1];
        if (next && (next.startsWith("'") || next.startsWith('"'))) return true;
      }
    }
  }

  // Heredoc with a closing delimiter — the body is model-authored text
  const heredocMatch = trimmed.match(/<<['"](\w+)['"]/) || trimmed.match(/<<(\w+)/);
  if (heredocMatch) {
    const delimiter = heredocMatch[1]!;
    const endPattern = new RegExp(`\\n${delimiter}(?:\\s|$)`);
    if (endPattern.test(trimmed)) return true;
  }

  return false;
}

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
 *  Read / WebFetch / WebSearch / a Bash single-file read this session. Returns
 *  `[]` for anything that doesn't look like a Claude Code transcript (the "fall
 *  through to current behavior" path — a thin adapter for other harnesses can
 *  be added later). */
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
  // Pass 1: tool_use id → {name, input} from assistant events (main-thread + sidechain).
  const toolUseById = new Map<string, { name: string; input: any }>();
  for (const line of lines) {
    let ev: any;
    try {
      ev = JSON.parse(line);
    } catch {
      continue;
    }
    events.push(ev);
    if (ev?.type !== "assistant") continue;
    const c = ev?.message?.content;
    if (!Array.isArray(c)) continue;
    for (const b of c) {
      if (b?.type === "tool_use" && typeof b.id === "string") {
        toolUseById.set(b.id, { name: String(b.name ?? ""), input: b.input ?? {} });
      }
    }
  }
  if (!events.some((e) => typeof e?.type === "string")) return []; // not a CC transcript

  // Pass 2: tool_result blocks AND verbatim user-typed text blocks in user
  // events (main-thread + sidechain). User-typed text becomes a "UserPrompt"
  // session source — a faithfulness check pool (#46), not world-truth KB.
  const out: SessionSource[] = [];
  for (let evIdx = 0; evIdx < events.length; evIdx++) {
    const ev = events[evIdx];
    if (ev?.type !== "user") continue;
    const c = ev?.message?.content;
    // Handle bare-string content (some transcripts wrap message.content as a
    // string for simple user-typed messages).
    if (typeof c === "string") {
      const text = c.trim();
      if (text.length >= USER_PROMPT_MIN_CHARS) {
        out.push({
          tool: "UserPrompt",
          uri: `user-prompt:event-${evIdx}`,
          content: text.length > USER_PROMPT_MAX_CHARS ? text.slice(0, USER_PROMPT_MAX_CHARS) : text,
          tool_use_id: `user-prompt-${evIdx}`,
        });
      }
      continue;
    }
    if (!Array.isArray(c)) continue;
    for (const b of c) {
      // ---- UserPrompt branch — verbatim user-typed text ---------------
      if (b?.type === "text" && typeof b.text === "string") {
        const text = b.text.trim();
        if (text.length < USER_PROMPT_MIN_CHARS) continue;
        out.push({
          tool: "UserPrompt",
          uri: `user-prompt:event-${evIdx}`,
          content: text.length > USER_PROMPT_MAX_CHARS ? text.slice(0, USER_PROMPT_MAX_CHARS) : text,
          tool_use_id: `user-prompt-${evIdx}`,
        });
        continue;
      }
      if (b?.type !== "tool_result" || typeof b.tool_use_id !== "string" || b.is_error) continue;
      const tu = toolUseById.get(b.tool_use_id);
      if (!tu || !SESSION_SOURCE_TOOLS.has(tu.name)) continue;
      let bashPath: string | null = null;
      let bashCmd = "";
      if (tu.name === "Bash") {
        bashCmd = String(tu.input?.command ?? "");
        if (isModelAuthoredOutputCommand(bashCmd)) continue;
        bashPath = safeFileReadCommand(bashCmd);
        // If the full command fails safeFileReadCommand but the primary portion
        // (before && or ;) is a clean read, trust the primary (e.g.
        // `cat file && echo done`). We do NOT do this for pipes or redirects,
        // because the output is transformed, not verbatim file content.
        if (!bashPath) {
          const primary = bashCmd.split(/&&|;\s*/)[0]!.trim();
          if (primary !== bashCmd) {
            bashPath = safeFileReadCommand(primary);
          }
        }
      }
      let text = toolResultText(b.content).trim();
      if (!text && ev.toolUseResult) {
        const r = ev.toolUseResult;
        if (tu.name === "Read") text = String(r?.file?.content ?? "").trim();
        else if (tu.name === "WebFetch") text = String(r?.result ?? "").trim();
        else if (tu.name === "Bash") text = String(r?.stdout ?? "").trim();
      }
      if (!text) continue;
      let uri = "";
      if (tu.name === "Read") uri = String(tu.input?.file_path ?? "");
      else if (tu.name === "WebFetch") uri = String(tu.input?.url ?? "");
      else if (tu.name === "WebSearch") uri = "websearch:" + String(tu.input?.query ?? "");
      else if (tu.name === "Bash") {
        if (bashPath) {
          uri = bashPath;
        } else {
          uri = "session-bash: " + bashCmd.replace(/\s+/g, " ").slice(0, 200);
        }
      }
      if (!uri || uri === "websearch:") continue;
      if (tu.name === "Read") text = stripLineNumbers(text); // Bash output is raw — no `cat -n` to strip
      if (text.length > MAX_SOURCE_CHARS) text = text.slice(0, MAX_SOURCE_CHARS);
      out.push({ tool: tu.name as SessionSource["tool"], uri, content: text, tool_use_id: b.tool_use_id });
    }
  }
  // Dedup by (tool|uri), keeping the freshest occurrence; cap total.
  const byKey = new Map<string, SessionSource>();
  for (const s of out) byKey.set(`${s.tool}|${s.uri}`, s);
  return [...byKey.values()].slice(-SESSION_SOURCES_MAX);
}

/** Look up a single session source by its `tool_use_id` without deduping.
 *  Returns `null` if the transcript is unreadable or the id is not found. */
export function findSessionSourceByToolUseId(
  transcriptPath: string,
  toolUseId: string,
): SessionSource | null {
  let raw: string;
  try {
    raw = readFileSync(transcriptPath, "utf8");
  } catch {
    return null;
  }
  const lines = raw.split("\n").filter((l) => l.trim());
  if (!lines.length) return null;

  const toolUseById = new Map<string, { name: string; input: any }>();
  const events: any[] = [];
  for (const line of lines) {
    let ev: any;
    try {
      ev = JSON.parse(line);
    } catch {
      continue;
    }
    events.push(ev);
    if (ev?.type !== "assistant") continue;
    const c = ev?.message?.content;
    if (!Array.isArray(c)) continue;
    for (const b of c) {
      if (b?.type === "tool_use" && typeof b.id === "string") {
        toolUseById.set(b.id, { name: String(b.name ?? ""), input: b.input ?? {} });
      }
    }
  }
  if (!events.some((e) => typeof e?.type === "string")) return null;

  for (const ev of events) {
    if (ev?.type !== "user") continue;
    const c = ev?.message?.content;
    if (!Array.isArray(c)) continue;
    for (const b of c) {
      if (b?.type !== "tool_result" || b.tool_use_id !== toolUseId || b.is_error) continue;
      const tu = toolUseById.get(b.tool_use_id);
      if (!tu || !SESSION_SOURCE_TOOLS.has(tu.name)) continue;
      let bashPath: string | null = null;
      let bashCmd = "";
      if (tu.name === "Bash") {
        bashCmd = String(tu.input?.command ?? "");
        if (isModelAuthoredOutputCommand(bashCmd)) continue;
        bashPath = safeFileReadCommand(bashCmd);
        if (!bashPath) {
          const primary = bashCmd.split(/&&|;\s*/)[0]!.trim();
          if (primary !== bashCmd) {
            bashPath = safeFileReadCommand(primary);
          }
        }
      }
      let text = toolResultText(b.content).trim();
      if (!text && ev.toolUseResult) {
        const r = ev.toolUseResult;
        if (tu.name === "Read") text = String(r?.file?.content ?? "").trim();
        else if (tu.name === "WebFetch") text = String(r?.result ?? "").trim();
        else if (tu.name === "Bash") text = String(r?.stdout ?? "").trim();
      }
      if (!text) continue;
      let uri = "";
      if (tu.name === "Read") uri = String(tu.input?.file_path ?? "");
      else if (tu.name === "WebFetch") uri = String(tu.input?.url ?? "");
      else if (tu.name === "WebSearch") uri = "websearch:" + String(tu.input?.query ?? "");
      else if (tu.name === "Bash") {
        if (bashPath) {
          uri = bashPath;
        } else {
          uri = "session-bash: " + bashCmd.replace(/\s+/g, " ").slice(0, 200);
        }
      }
      if (!uri || uri === "websearch:") continue;
      if (tu.name === "Read") text = stripLineNumbers(text);
      if (text.length > MAX_SOURCE_CHARS) text = text.slice(0, MAX_SOURCE_CHARS);
      return { tool: tu.name as SessionSource["tool"], uri, content: text, tool_use_id: b.tool_use_id };
    }
  }
  return null;
}

function sessionSourceTitle(src: SessionSource): string {
  if (src.tool === "Read" || src.tool === "Bash") {
    const p = src.uri.replace(/^file:\/\//, "");
    const i = Math.max(p.lastIndexOf("/"), p.lastIndexOf("\\"));
    return i >= 0 ? p.slice(i + 1) : p;
  }
  if (src.tool === "WebSearch") return src.uri.replace(/^websearch:/, "web search: ");
  return src.uri;
}

// ---------------------------------------------------------------------------
// Relevant-chunk slicing for large session sources (issue #39)
// ---------------------------------------------------------------------------

const SESSION_CHUNK_THRESHOLD = 8000;
const SESSION_CHUNK_MAX_SIZE = 2000;
const SESSION_CHUNK_OVERLAP = 300;
const SESSION_LEXICAL_TOPK = 5;
const SESSION_NLI_TOPK = 3;
const SESSION_NLI_MAX_CHARS = 6000;

function splitIntoChunks(text: string): string[] {
  const paragraphs = text.split(/\n\s*\n/);
  const chunks: string[] = [];
  for (const p of paragraphs) {
    if (!p.trim()) continue;
    if (p.length <= SESSION_CHUNK_MAX_SIZE) {
      chunks.push(p);
    } else {
      const step = SESSION_CHUNK_MAX_SIZE - SESSION_CHUNK_OVERLAP;
      for (let i = 0; i < p.length; i += step) {
        chunks.push(p.slice(i, i + SESSION_CHUNK_MAX_SIZE));
      }
    }
  }
  return chunks;
}

function rankChunksLexical(chunks: string[], proposition: string, entity: string): string[] {
  const propTokens = tokenSet(proposition);
  const ent = entity.toLowerCase().trim();
  const scored = chunks.map((chunk) => ({
    chunk,
    score: jaccard(propTokens, tokenSet(chunk)) + (ent && chunk.toLowerCase().includes(ent) ? 0.3 : 0),
  }));
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, SESSION_LEXICAL_TOPK).map((s) => s.chunk);
}

function selectRelevantSlice(src: SessionSource, pair: ExtractedPair): { slice: string; reason: string } {
  if (src.content.length <= SESSION_CHUNK_THRESHOLD) {
    return { slice: src.content, reason: "full content" };
  }

  const chunks = splitIntoChunks(src.content);
  if (chunks.length === 0) {
    return { slice: src.content.slice(0, MAX_SOURCE_CHARS), reason: "head" };
  }

  const topChunks = rankChunksLexical(chunks, pair.proposition, pair.entity);
  if (topChunks.length === 0) {
    return { slice: src.content.slice(0, MAX_SOURCE_CHARS), reason: "head" };
  }

  const selected: string[] = [];
  let total = 0;
  for (const c of topChunks.slice(0, SESSION_NLI_TOPK)) {
    if (total + c.length + 5 > SESSION_NLI_MAX_CHARS) break;
    selected.push(c);
    total += c.length + 5;
  }
  if (selected.length === 0) {
    selected.push(topChunks[0]!.slice(0, SESSION_NLI_MAX_CHARS));
  }

  return { slice: selected.join("\n---\n"), reason: `chunked top-${selected.length}` };
}

type AutoGroundOutcome =
  | { grounded: true; pair: GroundedPair }
  | { grounded: false; checked: number; hint?: { tool_use_id: string; uri: string; overlap: number } };

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
    const { slice, reason: sliceReason } = selectRelevantSlice(src, pair);
    const verdict = await verifyClaimAgainstSource(pair.proposition, slice);
    if (verdict.status !== "supported") continue;

    // UserPrompt branch (#46): the user pasted this content. We've verified
    // the agent's claim is FAITHFUL to that paste — not that the paste itself
    // is world-truth. Pass the gate, but DO NOT persist as a dossier (the
    // source isn't vouch-verified; promoting user prompts to KB would let
    // adversarial / mistaken / LLM-generated pastes ground future claims).
    // The agent took responsibility for the paste; vouch reports the
    // faithfulness check and leaves world-truth verification to the user's
    // own `vouch fetch` / `vouch claim` workflow.
    if (src.tool === "UserPrompt") {
      return {
        grounded: true,
        pair: {
          ...pair,
          grounded: true,
          matched_claim_id: null,
          auto_grounded: true,
          auto_grounded_kind: "user-prompt",
          session_sources_checked: n,
          reason: `auto-grounded against user-provided context (${sliceReason}, score=${verdict.score.toFixed(2)}) — faithfulness-checked, not KB-verified`,
        },
      };
    }

    // Entailed by a tool-retrieved source. Snapshot it as a dossier and file
    // the claim against it. The quote-in-dossier invariant holds trivially:
    // the "quote" is the retrieved content itself, which NLI just entailed —
    // the anti-fabrication primitive (a claim must trace to content vouch can
    // see) is satisfied by construction.
    const scope = src.tool === "Read" || src.tool === "Bash" ? "workspace" : "third-party";
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
        auto_grounded_kind: "tool-result",
        session_sources_checked: n,
        reason: `auto-grounded from session ${src.tool} of ${src.uri} (${sliceReason}, score=${verdict.score.toFixed(2)})`,
      },
    };
  }

  // Compute the best lexical overlap among ALL sources for the breadcrumb hint.
  const propTokens = tokenSet(pair.proposition);
  let bestHint: { tool_use_id: string; uri: string; overlap: number } | undefined;
  for (const s of sources) {
    const overlap = jaccard(propTokens, tokenSet(s.content));
    if (!bestHint || overlap > bestHint.overlap) {
      bestHint = { tool_use_id: s.tool_use_id!, uri: s.uri, overlap };
    }
  }
  if (bestHint) {
    const entMentioned = bestHint.uri.toLowerCase().includes(pair.entity.toLowerCase()) ||
      sources.find((s) => s.tool_use_id === bestHint!.tool_use_id)?.content.toLowerCase().includes(pair.entity.toLowerCase());
    if (bestHint.overlap < 0.15 && !entMentioned) {
      bestHint = undefined;
    }
  }
  return { grounded: false, checked: n, ...(bestHint ? { hint: bestHint } : {}) };
}

/** Wrap a failure-prone embed error message exactly as the inline catch blocks
 *  in `runGate` did before the single-exit refactor. */
function classifierErrorMessage(e: unknown): string {
  const transient = classifyError(e);
  return transient?.message || (e instanceof Error ? e.message : String(e)).slice(0, 200);
}

// ---------------------------------------------------------------------------
// Session ledger — cross-turn self-consistency (issue #43)
// ---------------------------------------------------------------------------

/** Derive the session transcript_id (basename minus .jsonl) used as the ledger
 *  partition key. */
export function transcriptIdFromPath(p: string): string {
  return basename(p).replace(/\.jsonl$/, "");
}

const SESSION_CONTRADICTION_TOPK = 5;
const SESSION_CONTRADICTION_MIN_COS = 0.55;
const SESSION_CONTRADICTION_FIRE_SCORE = 0.75;

interface SessionLedgerOutcome {
  /** New pair state, possibly with grounded flipped to false because of
   *  contradiction or with a "retraction-of-self" status. */
  pair: GroundedPair;
  /** True iff this pair's verdict triggers blocking on top of the existing
   *  grounding pass. */
  contradictionFire: boolean;
}

/** Heuristic entity-match for RETRACT auto-mark. Entities are loose strings;
 *  do a normalized contains check both ways. */
// Blind-spot enumeration: counts of explicit `[gap: ...]` markers + natural-
// language uncertainty phrases the agent surfaced this turn. Distinct from
// HEDGE-tag-near-entity (per-claim) — this is per-turn "did you enumerate
// what you DON'T know?". Visibility-only.

const GAP_MARKER_RE = /[\[(]gap:\s*([^\])]+)[\])]/gi;
const GAP_PHRASE_RES: RegExp[] = [
  /\b(?:i|we)\s+(?:didn['']?t|did\s+not)\s+(?:check|verify|test|measure|run|audit)\b/gi,
  /\b(?:i|we)\s+(?:haven['']?t|have\s+not)\s+(?:checked|verified|tested|measured|run|audited)\b/gi,
  /\b(?:i|we)\s+(?:don['']?t|do\s+not)\s+know\s+(?:if|whether|how|why|what|the)\b/gi,
  /\b(?:i'?m|we'?re|i\s+am|we\s+are)\s+not\s+sure\s+(?:about|if|whether|how)\b/gi,
  /\b(?:i|we)\s+should\s+(?:also\s+)?(?:check|verify|look\s+at|audit|investigate)\b/gi,
  /\bworth\s+(?:checking|verifying|investigating|auditing)\b/gi,
  /\b(?:open|unanswered)\s+question\b/gi,
];

export function countBlindSpots(draft: string): { explicit: number; phrase: number } {
  if (!draft) return { explicit: 0, phrase: 0 };
  const explicit = (draft.match(GAP_MARKER_RE) || []).length;
  let phrase = 0;
  for (const re of GAP_PHRASE_RES) {
    phrase += (draft.match(re) || []).length;
  }
  return { explicit, phrase };
}

function entitiesMatch(a: string, b: string): boolean {
  const na = a.toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
  const nb = b.toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
  if (!na || !nb) return false;
  if (na === nb) return true;
  return na.includes(nb) || nb.includes(na);
}

// Revise-action classifier: given a fired-but-unaddressed entity and this
// turn's draft + extracted pairs, return which of three productive revise
// shapes the agent took, or null:
//   'fetch'  — agent invoked a verification tool referencing the entity
//   'hedge'  — explicit uncertainty tag adjacent to the entity
//   'remove' — entity absent from any new extracted proposition
//   null     — none of the above (still a candidate dodge)
//
// Deterministic regex + windowed proximity. Lean toward false NEGATIVES
// (over-flag dodge) so the advisory stays visible until a real productive
// revise happens.

const TOOL_CALL_RE =
  /\b(vouch\s+(?:fetch|claim|search|attest)|WebFetch|WebSearch|gh\s+api|gh\s+\w+\s+view|git\s+show|curl\s+-[sLI]?\s|npm\s+view|pip\s+show)\b/gi;

const HEDGE_TAG_RE =
  /(\((?:un|not\s+)?verified|from\s+(?:training\s+)?memory|haven['']?t\s+verified|not\s+yet\s+(?:verified|checked|grounded)|unsubstantiated|to\s+be\s+verified)/gi;

const PROXIMITY_WINDOW = 300;

/** Find all index positions in `text` where the regex matches, ignoring
 *  zero-width matches and resetting the regex's lastIndex between calls. */
function findAllMatchIndices(re: RegExp, text: string): number[] {
  const indices: number[] = [];
  if (!text) return indices;
  const r = new RegExp(re.source, re.flags.includes("g") ? re.flags : re.flags + "g");
  let m: RegExpExecArray | null;
  while ((m = r.exec(text)) !== null) {
    indices.push(m.index);
    if (m.index === r.lastIndex) r.lastIndex++; // guard zero-width
  }
  return indices;
}

/** True iff `entity` (case- and punct-folded) appears within
 *  PROXIMITY_WINDOW chars of any anchor in `anchorIndices`. */
function entityNearAnchor(
  entity: string,
  text: string,
  anchorIndices: number[],
): boolean {
  if (!anchorIndices.length) return false;
  const ent = entity.toLowerCase().trim();
  if (!ent) return false;
  const lower = text.toLowerCase();
  for (const a of anchorIndices) {
    const start = Math.max(0, a - PROXIMITY_WINDOW);
    const end = Math.min(text.length, a + PROXIMITY_WINDOW);
    if (lower.slice(start, end).includes(ent)) return true;
  }
  return false;
}

export type ReviseAction = "fetch" | "hedge" | "remove";

/** Classify how (if at all) the current turn's draft addresses a prior-turn
 *  fire on `entity`. `currentExtractedPairs` is this turn's extractor output
 *  (used for 'remove' detection: entity absent from all new propositions). */
export function classifyReviseAction(
  entity: string,
  draft: string,
  currentExtractedPairs: ExtractedPair[],
): ReviseAction | null {
  if (!entity) return null;

  // (a) Tool-call referencing the entity. Most specific path → check first.
  const toolMatches = findAllMatchIndices(TOOL_CALL_RE, draft);
  if (toolMatches.length && entityNearAnchor(entity, draft, toolMatches)) {
    return "fetch";
  }

  // (b) Hedge tag adjacent to the entity. Acceptable provenance disclosure.
  const hedgeMatches = findAllMatchIndices(HEDGE_TAG_RE, draft);
  if (hedgeMatches.length && entityNearAnchor(entity, draft, hedgeMatches)) {
    return "hedge";
  }

  // (c) Removal — entity not present in any new extracted proposition this
  // turn. NOTE: this is the most contested shape — silent removal is the
  // (4b) dodge per #50. We classify it as 'remove' (action taken) but Stage
  // 3's enforcement message will explicitly call out silent-rephrase as
  // dodge-shaped. This classifier just records the action class; the
  // silent-rephrase detector upstream distinguishes the two via cosine.
  const entityPresent = currentExtractedPairs.some((p) => entitiesMatch(p.entity, entity));
  if (!entityPresent) return "remove";

  // Entity reappears in a new claim with no hedge / no tool — unaddressed.
  return null;
}

/** Per-turn session-ledger pass:
 *  1. RETRACT-stance pairs in this turn → mark matching prior ledger entries as retracted.
 *  2. For each new ASSERT (or escalated ASSERT) → embed, find cosine candidates, run
 *     inverse-NLI; if contradicts a prior active entry, flip the pair to ungrounded.
 *  3. Persist all final pair states to the ledger.
 *  Returns the (possibly contradiction-flipped) pairs and whether any contradiction
 *  fired on this turn. */
export async function applySessionLedger(
  transcriptPath: string,
  pairs: GroundedPair[],
  abortRef?: { aborted: boolean },
  /** Current turn's raw draft. Required for the revise-action check —
   *  classification of tool-call / hedge-tag / removal against prior-turn
   *  fires flagged `awaiting_revise=1`. Omit to skip the check entirely
   *  (legacy behavior, used by tests that don't supply a draft). */
  draft?: string,
): Promise<{
  pairs: GroundedPair[];
  contradictionFire: boolean;
  reviseCheck?: {
    addressedFetch: number;
    addressedHedge: number;
    addressedRemove: number;
    suspectedSilentRephrase: number;
    stillUnaddressed: Array<{ entity: string; turn_idx: number; proposition: string }>;
  };
}> {
  const transcript_id = transcriptIdFromPath(transcriptPath);
  const turn_idx = store.getNextSessionTurnIdx(transcript_id);

  // ----- 0. Revise-action check on prior fires.
  // Runs BEFORE this turn's pairs are processed, so the awaiting-set reflects
  // the state going INTO this turn. Results feed an advisory line rendered
  // in formatSessionSummary.
  let reviseCheck: {
    addressedFetch: number;
    addressedHedge: number;
    addressedRemove: number;
    /** Sub-count of `remove` outcomes: how many had a NEW high-cosine
     *  proposition in this turn (suspected silent-rephrase
     *  dodge)? When >0 the advisory line names them. */
    suspectedSilentRephrase: number;
    stillUnaddressed: Array<{ entity: string; turn_idx: number; proposition: string }>;
  } | undefined;
  if (draft) {
    const awaiting = store.listAwaitingReviseClaims(transcript_id);
    if (awaiting.length) {
      let addressedFetch = 0;
      let addressedHedge = 0;
      let addressedRemove = 0;
      let suspectedSilentRephrase = 0;
      const stillUnaddressed: Array<{ entity: string; turn_idx: number; proposition: string }> = [];
      // Lazy cache: new-pair embeddings, populated the first time a
      // `remove` outcome needs the silent-rephrase cosine check and
      // reused across remaining awaiting rows.
      const newPairEmbeddings: Array<{ idx: number; emb: Float32Array }> = [];
      for (const row of awaiting) {
        // Skip entries from THIS same turn (they're being processed now, not
        // "prior-turn fires"). Compare turn_idx.
        if (row.turn_idx >= turn_idx) continue;
        const via = classifyReviseAction(row.entity, draft, pairs);
        if (via === "fetch") {
          addressedFetch++;
          store.markAddressedAwaiting(transcript_id, row.turn_idx, row.claim_idx, "fetch", turn_idx);
        } else if (via === "hedge") {
          addressedHedge++;
          store.markAddressedAwaiting(transcript_id, row.turn_idx, row.claim_idx, "hedge", turn_idx);
        } else if (via === "remove") {
          // Distinguish legitimate topic-drop from silent-rephrase: if any
          // new pair this turn has a proposition semantically near the
          // fired one (cosine ≥ SILENT_REPHRASE_COSINE), the agent kept
          // the claim but excised the entity name. Compare against the
          // fired proposition's stored embedding.
          let suspectedRephrase = false;
          if (row.embedding && pairs.length) {
            // Lazy-embed once per turn in parallel; cache reused across
            // subsequent `remove` checks in the same turn.
            if (newPairEmbeddings.length === 0) {
              const embedded = await Promise.all(
                pairs.map(async (p, pi) => {
                  try {
                    return { idx: pi, emb: await embedOne(p.proposition) };
                  } catch {
                    return null;
                  }
                }),
              );
              for (const e of embedded) if (e) newPairEmbeddings.push(e);
            }
            for (const { emb } of newPairEmbeddings) {
              if (store.cosine(row.embedding, emb) >= SILENT_REPHRASE_COSINE) {
                suspectedRephrase = true;
                break;
              }
            }
          }
          if (suspectedRephrase) {
            // Treat as still-unaddressed: the claim's content survived
            // even though the entity name didn't.
            suspectedSilentRephrase++;
            stillUnaddressed.push({
              entity: row.entity,
              turn_idx: row.turn_idx,
              proposition: row.proposition,
            });
            // Don't clear awaiting_revise yet — agent needs to address
            // the underlying claim, not just rename the entity.
          } else {
            addressedRemove++;
            store.markAddressedAwaiting(transcript_id, row.turn_idx, row.claim_idx, "remove", turn_idx);
          }
        } else {
          stillUnaddressed.push({
            entity: row.entity,
            turn_idx: row.turn_idx,
            proposition: row.proposition,
          });
        }
      }
      reviseCheck = { addressedFetch, addressedHedge, addressedRemove, suspectedSilentRephrase, stillUnaddressed };
    }
  }

  // ----- 1. RETRACT-stance pairs in this turn → mark prior matches retracted.
  for (const p of pairs) {
    if (p.stance !== "RETRACT") continue;
    const priors = store.listSessionClaims(transcript_id, { only_active: true });
    for (const r of priors) {
      if (entitiesMatch(r.entity, p.entity)) {
        store.markSessionClaimRetracted(
          transcript_id,
          r.turn_idx,
          r.claim_idx,
          `RETRACT in turn ${turn_idx}: "${p.proposition.slice(0, 120)}"`,
        );
      }
    }
  }

  // ----- 2. Contradiction check on new ASSERTs (and escalated-from-HEDGE).
  let contradictionFire = false;
  const outPairs: GroundedPair[] = [];
  for (let claim_idx = 0; claim_idx < pairs.length; claim_idx++) {
    const p = pairs[claim_idx]!;
    // Only check claims the agent is actually asserting. Skip RETRACT,
    // WORKSPACE, REFER, OPINION, HEDGE (stays HEDGE — #42 already escalated
    // any HEDGE-with-trailing-caveat to ASSERT before grounding).
    if (p.stance !== "ASSERT") {
      outPairs.push(p);
      continue;
    }
    if (abortRef?.aborted) {
      outPairs.push(p);
      continue;
    }
    let queryEmb: Float32Array | null = null;
    try {
      queryEmb = await embedOne(p.proposition);
    } catch {
      // embed failure — fall through, no contradiction check possible.
    }
    if (!queryEmb) {
      outPairs.push(p);
      continue;
    }
    const candidates = store.findSessionContradictionCandidates(transcript_id, queryEmb, {
      topK: SESSION_CONTRADICTION_TOPK,
      minCos: SESSION_CONTRADICTION_MIN_COS,
    });
    let flippedPair: GroundedPair = p;
    for (const cand of candidates) {
      if (abortRef?.aborted) break;
      try {
        const verdict = await verifyContradiction(p.proposition, cand.row.proposition);
        if (verdict.contradicts && verdict.score >= SESSION_CONTRADICTION_FIRE_SCORE) {
          flippedPair = {
            ...p,
            grounded: false,
            reason:
              `contradicts prior session turn ${cand.row.turn_idx} claim ${cand.row.claim_idx}: ` +
              `"${cand.row.proposition.slice(0, 140)}" — ${verdict.reason}`,
            contradicts_session: {
              old_turn: cand.row.turn_idx,
              old_claim: cand.row.claim_idx,
              old_proposition: cand.row.proposition,
              score: verdict.score,
              reason: verdict.reason,
            },
          };
          contradictionFire = true;
          break;
        }
      } catch {
        // Verifier transient error — skip this candidate (recall-biased).
        continue;
      }
    }
    outPairs.push(flippedPair);
  }

  // ----- 3. Persist final pair states to the ledger.
  for (let claim_idx = 0; claim_idx < outPairs.length; claim_idx++) {
    const p = outPairs[claim_idx]!;
    let emb: Float32Array | null = null;
    try {
      emb = await embedOne(p.proposition);
    } catch {
      emb = null;
    }
    const verdict = p.contradicts_session
      ? "contradicted"
      : p.escalatedFromHedge
      ? "escalated"
      : p.reclassifiedRule
      ? "reclassified"
      : p.grounded
      ? "grounded"
      : "ungrounded";
    store.recordSessionClaim({
      transcript_id,
      turn_idx,
      claim_idx,
      proposition: p.proposition,
      entity: p.entity ?? "",
      stance: p.stance,
      verdict,
      reason: p.reason,
      embedding: emb,
    });
  }

  return { pairs: outPairs, contradictionFire, reviseCheck };
}

export async function runGate(opts: {
  draft: string;
  model: string;
  topK?: number;
  /** Claude Code transcript path; if present and the gate fires, session-
   *  retrieved sources are scanned for auto-grounding before blocking. */
  sessionTranscriptPath?: string;
  /** Pre-extracted pairs (runGateCli calls extractPairs itself to print the
   *  extraction breadcrumb). If omitted, runGate extracts itself. */
  extractedPairs?: ExtractedPair[] | null;
  /** Shared abort flag; when true, runGate stops further work and marks the
   *  verdict as incomplete. */
  abortRef?: { aborted: boolean };
}): Promise<GateVerdict & { incomplete?: boolean }> {
  let pairs: GroundedPair[] = [];
  let classifierError: string | undefined;
  let blocked = false;
  let incomplete = false;
  let reviseCheck:
    | {
        addressedFetch: number;
        addressedHedge: number;
        addressedRemove: number;
        stillUnaddressed: Array<{ entity: string; turn_idx: number; proposition: string }>;
      }
    | undefined;

  const extracted = opts.extractedPairs !== undefined ? opts.extractedPairs : await extractPairs(opts.draft, opts.model);
  const pairsToCheck = extracted
    ? collapseSofterStancesToAssert(
        escalateHedgeAssertions(reclassifyWorkspaceMeta(extracted, opts.draft), opts.draft),
      )
    : null;
  if (extracted === null) {
    classifierError = "extractor failed";
  } else if (pairsToCheck && pairsToCheck.length) {
    const checked: GroundedPair[] = new Array(pairsToCheck.length);
    let errored = false;

    // Separate ASSERT from non-ASSERT to preserve output ordering.
    const assertIndices: number[] = [];
    for (let i = 0; i < pairsToCheck.length; i++) {
      const p = pairsToCheck[i]!;
      if (p.stance !== "ASSERT") {
        const reason = p.reclassifiedRule
          ? `reclassified WORKSPACE by deterministic post-filter (rule ${p.reclassifiedRule})`
          : `stance=${p.stance} — no fact to ground`;
        checked[i] = {
          ...p,
          grounded: true,
          matched_claim_id: null,
          reason,
        };
      } else {
        assertIndices.push(i);
      }
    }

    if (opts.abortRef?.aborted) {
      incomplete = true;
    } else if (assertIndices.length) {
      try {
        const assertPairs = assertIndices.map((i) => pairsToCheck[i]!);
        const { results: groundedAssertions, incomplete: batchIncomplete } =
          await batchGroundAssertions(assertPairs, opts.topK ?? 3, opts.abortRef);
        if (batchIncomplete) incomplete = true;
        for (let j = 0; j < assertIndices.length; j++) {
          checked[assertIndices[j]!] = groundedAssertions[j]!;
        }
      } catch (e) {
        classifierError = classifierErrorMessage(e);
        errored = true;
      }
    }

    // Lazy: only parse the transcript / run session NLI when the gate would
    // otherwise fire on something.
    if (!errored && !incomplete) {
      const ungroundedIdx = checked.flatMap((p, i) => (p.grounded ? [] : [i]));
      if (ungroundedIdx.length && opts.sessionTranscriptPath) {
        let sources: SessionSource[] = [];
        try {
          sources = parseSessionSources(opts.sessionTranscriptPath);
        } catch {
          sources = [];
        }
        if (sources.length) {
          const tasks = ungroundedIdx.map((i) => async () => {
            if (opts.abortRef?.aborted) {
              return { aborted: true as const, idx: i };
            }
            try {
              const res = await autoGroundPair(checked[i]!, sources);
              return { res, idx: i };
            } catch (e) {
              return { error: e as unknown, idx: i };
            }
          });
          const agResults = await concurrencyLimit(tasks, 6);
          for (const r of agResults) {
            if ("aborted" in r) {
              incomplete = true;
              continue;
            }
            if ("error" in r) {
              classifierError = classifierErrorMessage(r.error);
              errored = true;
              break;
            }
            checked[r.idx] = r.res.grounded
              ? r.res.pair
              : { ...checked[r.idx]!, session_sources_checked: r.res.checked, ...(r.res.hint ? { hint: r.res.hint } : {}) };
          }
        }
      }
    }

    pairs = checked.filter((p): p is GroundedPair => !!p);

    // Session-ledger pass (#43): RETRACT auto-mark, contradiction-fire vs
    // prior-turn ASSERTs in this transcript, persist this turn's pairs.
    // Only runs when we have a transcript to key the ledger by, and only
    // when the per-pair grounding loop wasn't truncated (errored/incomplete);
    // a truncated grounding pass would write incomplete verdicts.
    if (!errored && !incomplete && opts.sessionTranscriptPath && pairs.length) {
      try {
        const ledger = await applySessionLedger(
          opts.sessionTranscriptPath,
          pairs,
          opts.abortRef,
          opts.draft,
        );
        pairs = ledger.pairs;
        reviseCheck = ledger.reviseCheck;
      } catch (e) {
        // Ledger errors fail OPEN — never let a session-DB hiccup block.
        classifierError = classifierErrorMessage(e);
      }
    }

    // A verifier error fails the gate OPEN (blocked stays false) — a transient
    // system fault carries no signal about the (claim, source) pair.
    if (!errored && !incomplete) blocked = pairs.some((p) => !p.grounded);
  }
  // extracted.length === 0 → nothing to ground → blocked stays false.

  // Zero-proposition turns need their own revise-check pass — the
  // standard branch above only runs when pairs.length > 0. Silent-
  // rephrase dodges typically produce drafts with NO new claims, so
  // "entity not in new pairs" IS the removal signal. Skip silently
  // if applySessionLedger fails (advisory).
  if (
    !classifierError &&
    !incomplete &&
    opts.sessionTranscriptPath &&
    (extracted?.length ?? 0) === 0 &&
    !reviseCheck
  ) {
    try {
      const ledger = await applySessionLedger(
        opts.sessionTranscriptPath,
        [],
        opts.abortRef,
        opts.draft,
      );
      reviseCheck = ledger.reviseCheck;
    } catch {
      // advisory only
    }
  }

  // Second pass: a passing draft gets its tagged derived claims harvested.
  const harvest = (!blocked && !incomplete) ? await safeHarvest(opts.draft) : undefined;

  return {
    blocked,
    pairs,
    ...(classifierError ? { classifier_error: classifierError } : {}),
    ...(harvest ? { harvest } : {}),
    ...(incomplete ? { incomplete: true } : {}),
    ...(reviseCheck ? { reviseCheck } : {}),
  };
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

/** Derive the session start timestamp from a Claude Code transcript's first
 *  event. Returns `null` if the file is unreadable or contains no timestamped
 *  events. */
export function getFirstEventTimestamp(transcriptPath: string): string | null {
  try {
    const raw = readFileSync(transcriptPath, "utf8").trim();
    if (!raw) return null;
    let firstTs: string | null = null;
    for (const line of raw.split("\n")) {
      if (!line.trim()) continue;
      let ev: any;
      try {
        ev = JSON.parse(line);
      } catch {
        continue;
      }
      if (typeof ev?.timestamp === "string") {
        if (!firstTs || ev.timestamp < firstTs) {
          firstTs = ev.timestamp;
        }
      }
    }
    return firstTs;
  } catch {
    return null;
  }
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
  /** Claude Code transcript JSONL to use as session evidence for auto-grounding.
   *  When set, this path is passed to runGate as sessionTranscriptPath regardless
   *  of whether draft or transcriptPath is used to obtain the draft text. */
  sessionContext?: string;
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
  /** True when the gate exited early due to budget exceeded or uncaught error. */
  incomplete?: boolean;
}

function sha256Prefix(text: string, len = 12): string {
  return createHash("sha256").update(text).digest("hex").slice(0, len);
}

interface AuditEntry {
  ts: string;
  pid: number;
  draft_sha256: string;
  n_propositions: number;
  n_checked: number;
  n_grounded: number;
  n_ungrounded: number;
  verdict: "pass" | "block" | "incomplete" | "error";
  wall_ms: number;
  exit_code: number;
  mode: "strict" | "advisory";
  error?: string;
}

function writeGateLog(entry: AuditEntry) {
  const path = process.env.VOUCH_GATE_LOG || join(dirname(DB_PATH), "gate.log");
  try {
    mkdirSync(dirname(path), { recursive: true });
    appendFileSync(path, JSON.stringify(entry) + "\n");
  } catch (e: any) {
    process.stderr.write(`[vouch-gate] (could not write gate.log: ${e?.message || String(e)})\n`);
  }
}

/** Single-line per-session fire summary appended to every Stop hook's
 *  stderr. Counts come from store.session_claims (gate-time verdicts), so
 *  this is the *intent* signal — the agent attempted N claims, M fired,
 *  K got grounded one way or another. The cross-turn revise outcome
 *  (verified / hedged / dodged) is NOT classified here; that's a downstream
 *  human/judge pass via classify_fires.ts. This line surfaces the raw
 *  pressure: a fire count that climbs without a matching grounded count
 *  is the dodge-accumulation signal the user can act on. */
/** Escalated block message: rendered when this turn passed its own gate
 *  but the revise-check detected unaddressed prior-turn fires (opt-in via
 *  VOUCH_GATE_ESCALATE_UNADDRESSED). Names the lingering entities and
 *  points at the productive paths. */
function formatEscalatedBlockMessage(verdict: GateVerdict, draft: string): string {
  const lines: string[] = [];
  lines.push(
    `[vouch-gate] Detected unaddressed prior-turn fires — your previous draft fired on entit(y|ies) below, but this revise neither verified, hedged, nor removed them. (Enabled by VOUCH_GATE_ESCALATE_UNADDRESSED=1.)`,
  );
  for (const u of verdict.reviseCheck?.stillUnaddressed ?? []) {
    lines.push(
      `  • ${boldRed(u.entity)} (fired turn ${u.turn_idx}): "${u.proposition.slice(0, 200)}"`,
    );
    lines.push(`      → ${suggestVerification(u.entity, draft)}`);
  }
  lines.push("");
  lines.push(
    `To pass, the revise must contain ONE of these for EACH entity above (in preference order):`,
  );
  lines.push(
    `  • verify — \`vouch fetch\` / \`vouch claim\` / \`vouch search\` / \`WebFetch\` / \`gh api\` tool call referencing it`,
  );
  lines.push(
    `  • hedge — \`(unverified, from training memory)\` tag adjacent to the entity`,
  );
  lines.push(
    `  • remove — only if the entity is genuinely not load-bearing. Silent delete hides the gap.`,
  );
  lines.push("");
  return lines.join("\n") + "\n";
}

function formatSessionSummary(
  transcript_id: string,
  reviseCheck?: GateVerdict["reviseCheck"],
  blindSpotsThisTurn?: { explicit: number; phrase: number },
  thisTurnStances?: { asserts: number; hedges: number; speculates: number },
): string {
  const c = store.getSessionFireCounts(transcript_id);
  if (c.total === 0) return "";
  const unresolved = c.ungrounded;
  const resolvedNotes: string[] = [];
  if (c.grounded > 0) resolvedNotes.push(`${c.grounded} grounded`);
  if (c.reclassified > 0) resolvedNotes.push(`${c.reclassified} reclassified`);
  if (c.retracted > 0) resolvedNotes.push(`${c.retracted} retracted`);
  const resolvedStr = resolvedNotes.length ? ` · ${resolvedNotes.join(" / ")}` : "";

  // Render the revise-check breakdown: (a) cleared-this-turn counts,
  // (b) suspected silent-rephrase count, (c) still-unaddressed list.
  // Advisory only — escalation lives in runGateCli behind the opt-in
  // env flag.
  let reviseCheckLines = "";
  if (reviseCheck) {
    const cleared =
      reviseCheck.addressedFetch + reviseCheck.addressedHedge + reviseCheck.addressedRemove;
    const clearedParts: string[] = [];
    if (reviseCheck.addressedFetch > 0) clearedParts.push(`${reviseCheck.addressedFetch} fetch`);
    if (reviseCheck.addressedHedge > 0) clearedParts.push(`${reviseCheck.addressedHedge} hedge`);
    if (reviseCheck.addressedRemove > 0) clearedParts.push(`${reviseCheck.addressedRemove} remove`);
    if (cleared > 0) {
      reviseCheckLines +=
        `[vouch-gate] revise check: ${cleared} prior fire(s) addressed this turn (${clearedParts.join(" / ")})\n`;
    }
    if (reviseCheck.suspectedSilentRephrase > 0) {
      reviseCheckLines +=
        `⚠ [vouch-gate] silent-rephrase suspected: ${reviseCheck.suspectedSilentRephrase} prior fire(s) had the entity removed but the underlying claim recurred at high cosine (≥${SILENT_REPHRASE_COSINE}) in this turn — kept on the awaiting backlog\n`;
    }
    if (reviseCheck.stillUnaddressed.length > 0) {
      const entityList = reviseCheck.stillUnaddressed
        .slice(0, 5)
        .map((u) => `"${u.entity}" (turn ${u.turn_idx})`)
        .join(", ");
      const more =
        reviseCheck.stillUnaddressed.length > 5
          ? ` + ${reviseCheck.stillUnaddressed.length - 5} more`
          : "";
      reviseCheckLines +=
        `⚠ [vouch-gate] revise check: ${reviseCheck.stillUnaddressed.length} prior fire(s) still unaddressed in this turn — ${entityList}${more}\n` +
        `   (advisory: this revise neither verified, hedged, nor removed these entities — candidate dodge surface)\n`;
    }
  }

  // Backlog: total awaiting_revise (entries not yet addressed in any
  // later turn). With the revise-action classifier active this number
  // should drop turn-over-turn as detection clears entries.
  const awaitingLine =
    c.awaiting_revise > 0
      ? `[vouch-gate] revise backlog: ${c.awaiting_revise} fired entit${c.awaiting_revise === 1 ? "y" : "ies"} from prior turns awaiting revise verification\n`
      : "";

  // Humility ratio: explicit-uncertainty stances (HEDGE + SPECULATE) over
  // all truth-bearing stances (ASSERT + HEDGE + SPECULATE). Diagnostic
  // signal — falling rate flags drift toward over-confidence, chronically
  // high rate flags over-hedging. HUMILITY_DISPLAY_MIN_TRUTH avoids
  // noisy small-N display.
  const truthBearing = c.asserts + c.hedges + c.speculates;
  let humilityLine = "";
  if (truthBearing >= HUMILITY_DISPLAY_MIN_TRUTH) {
    const hedged = c.hedges + c.speculates;
    const rate = ((hedged / truthBearing) * 100).toFixed(1);
    humilityLine =
      `[vouch-gate] humility: ${hedged}/${truthBearing} = ${rate}% explicit-uncertainty (${c.asserts} assert / ${c.hedges} hedge / ${c.speculates} speculate)\n`;
  }

  // Per-turn blind-spot count. Always renders when blindSpotsThisTurn
  // is provided — signal is per-turn, not session-cumulative.
  let blindSpotLine = "";
  if (blindSpotsThisTurn) {
    const total = blindSpotsThisTurn.explicit + blindSpotsThisTurn.phrase;
    const breakdown =
      blindSpotsThisTurn.explicit > 0 || blindSpotsThisTurn.phrase > 0
        ? ` (${blindSpotsThisTurn.explicit} explicit [gap:] / ${blindSpotsThisTurn.phrase} natural phrase)`
        : "";
    blindSpotLine =
      `[vouch-gate] blind-spots this turn: ${total} enumerated${breakdown}\n`;
  }

  // Over-confidence advisory: ≥ OVER_CONFIDENT_ASSERT_FLOOR confident
  // ASSERTs this turn AND 0 hedge/speculate stances AND 0 blind-spots
  // enumerated → emit a single advisory line. Advisory only — exit code
  // unchanged.
  let overConfidentLine = "";
  if (thisTurnStances && blindSpotsThisTurn) {
    const blindTotal = blindSpotsThisTurn.explicit + blindSpotsThisTurn.phrase;
    const uncertainCount = thisTurnStances.hedges + thisTurnStances.speculates;
    if (
      thisTurnStances.asserts >= OVER_CONFIDENT_ASSERT_FLOOR &&
      uncertainCount === 0 &&
      blindTotal === 0
    ) {
      overConfidentLine =
        `⚠ [vouch-gate] over-confidence flag: ${thisTurnStances.asserts} confident ASSERTs this turn with 0 hedges/speculates and 0 blind-spots enumerated. Consider hedging at least one uncertain claim or adding [gap: <specific thing you didn't check>].\n`;
    }
  }

  return (
    `[vouch-gate] session so far: ${c.total} claim(s) seen · ${unresolved} fire(s) unresolved${resolvedStr}\n` +
    reviseCheckLines +
    awaitingLine +
    humilityLine +
    blindSpotLine +
    overConfidentLine
  );
}

function formatDeltaMessage(
  delta: store.KbDelta,
  unsupportedSkipped: number,
  advisories: number,
): string {
  const total = delta.atomicSupported + delta.derivedHarvested + delta.dossiersSnapshotted + unsupportedSkipped + advisories;
  if (total === 0) {
    return "[vouch-gate] turn Δ: (nothing entered the KB this turn)\n";
  }
  const kbParts = [
    `${delta.atomicSupported} ATOMIC supported (auto-grounded)`,
    `${delta.derivedHarvested} derived (harvested)`,
    `${delta.dossiersSnapshotted} dossier(s) snapshotted`,
  ];
  return (
    `[vouch-gate] turn Δ: ${kbParts.join(" / ")} · ${unsupportedSkipped} unsupported attempt${unsupportedSkipped === 1 ? "" : "s"} skipped · ${advisories} ${advisories === 1 ? "advisory" : "advisories"}\n`
  );
}

export async function runGateCli(opts: GateRunOptions): Promise<GateRunResult & { incomplete?: boolean }> {
  // Default 25 s so the watchdog wins the race against Claude Code's 30 s
  // Stop-hook timeout. If you raise this, raise the hook timeout too.
  const budgetMs = parseInt(process.env.VOUCH_GATE_BUDGET_MS || "25000", 10);
  const failMode = process.env.VOUCH_GATE_FAILMODE || "warn";
  const mode = opts.strict ? "strict" : "advisory";
  const wallStart = Date.now();

  let draft = opts.draft;
  let transcriptPath = opts.transcriptPath;
  let turnAnchor = new Date().toISOString();
  let extractedPairs: ExtractedPair[] | null = null;
  let verdict: GateVerdict & { incomplete?: boolean } = { blocked: false, pairs: [] };
  let exitCode: 0 | 2 = 0;
  let message: string | undefined;
  let errorMessage: string | undefined;
  let incomplete = false;

  process.stderr.write(`[vouch-gate] start pid=${process.pid} budget=${budgetMs / 1000}s\n`);

  const abortRef = { aborted: false };
  const timer = setTimeout(() => {
    abortRef.aborted = true;
  }, budgetMs);

  try {
    if (opts.bypassEnv && process.env[opts.bypassEnv] === "1") {
      message = formatDeltaMessage({ atomicSupported: 0, derivedHarvested: 0, dossiersSnapshotted: 0 }, 0, 0);
    } else {
      if (!draft && !transcriptPath && opts.hookPayload) {
        if (typeof opts.hookPayload.transcript_path === "string") {
          transcriptPath = opts.hookPayload.transcript_path;
        }
      }

      const transcriptAvailable = !!transcriptPath && existsSync(transcriptPath);
      if (!draft && transcriptPath) {
        if (!transcriptAvailable) {
          message = formatDeltaMessage({ atomicSupported: 0, derivedHarvested: 0, dossiersSnapshotted: 0 }, 0, 0);
        } else {
          try {
            const turn = await readLatestAssistantTurn(transcriptPath);
            if (!turn.isFresh) {
              // Transcript-flush race: the just-finished turn is not yet in the
              // file (or no recent turn exists). Fail-open rather than block on
              // potentially-stale prior-turn content.
              message = formatDeltaMessage({ atomicSupported: 0, derivedHarvested: 0, dossiersSnapshotted: 0 }, 0, 0);
            } else {
              draft = turn.text;
              if (turn.timestamp) turnAnchor = turn.timestamp;
            }
          } catch {
            message = formatDeltaMessage({ atomicSupported: 0, derivedHarvested: 0, dossiersSnapshotted: 0 }, 0, 0);
          }
        }
      }

      if (!message && !draft?.trim()) {
        message = formatDeltaMessage({ atomicSupported: 0, derivedHarvested: 0, dossiersSnapshotted: 0 }, 0, 0);
      }

      if (!message) {
        extractedPairs = await extractPairs(draft!, opts.model);
        if (extractedPairs !== null) {
          process.stderr.write(`[vouch-gate] extracted ${extractedPairs.length} proposition(s); grounding…\n`);
        }

        verdict = await runGate({
          draft: draft!,
          model: opts.model,
          topK: opts.topK,
          sessionTranscriptPath: transcriptAvailable ? transcriptPath : opts.sessionContext,
          extractedPairs,
          abortRef,
        });

        if (verdict.incomplete) {
          const elapsedS = (Date.now() - wallStart) / 1000;
          const nPropositions = extractedPairs?.length ?? 0;
          const nChecked = verdict.pairs.length;
          const nUnchecked = nPropositions - nChecked;
          process.stderr.write(
            `⚠ [vouch-gate] BUDGET EXCEEDED after ${elapsedS.toFixed(1)}s — ${nUnchecked} of ${nPropositions} proposition(s) unchecked; turn NOT fully gated\n`,
          );
          exitCode = failMode === "block" ? 2 : 0;
          incomplete = true;
        } else {
          const delta = store.getKbDelta(turnAnchor);
          const unsupportedSkipped = verdict.pairs.filter(
            (p) => p.stance === "ASSERT" && !p.grounded && !p.auto_grounded && (p.session_sources_checked ?? 0) > 0,
          ).length;
          const advisories = verdict.harvest?.flags.length ?? 0;
          const deltaMsg = formatDeltaMessage(delta, unsupportedSkipped, advisories);

          // Per-session fire counter (P1 of #50 work). Visible-to-user
          // dodge-pressure signal: how many fires has THIS session accumulated
          // and how many got resolved (grounded / autoground / reclassified).
          // Surfacing this puts the dodge-rate trend in the user's face every
          // turn — the user-mediated forcing function (today's session proved
          // this works when active; this makes it systematic instead of ad-hoc).
          const sessionTid = transcriptAvailable
            ? transcriptIdFromPath(transcriptPath!)
            : opts.sessionContext
              ? transcriptIdFromPath(opts.sessionContext)
              : null;
          // Count blind-spot markers in this turn's draft.
          const blindSpotsThisTurn = draft ? countBlindSpots(draft) : { explicit: 0, phrase: 0 };
          // Count THIS turn's stance distribution for the over-confidence
          // flag (separate from session-level humility ratio).
          const thisTurnStances = {
            asserts: verdict.pairs.filter((p) => p.stance === "ASSERT").length,
            hedges: verdict.pairs.filter((p) => p.stance === "HEDGE").length,
            speculates: verdict.pairs.filter((p) => p.stance === "SPECULATE").length,
          };
          // Web-counter-evidence suggestion: when the KB counter-check ran
          // but found nothing for any grounded pair, prompt for a manual
          // web counter-search. The 44% KB-miss rate on the dogfood
          // baseline says most relevant counter-evidence isn't in KB.
          // Suppressed when KB counter-evidence WAS found (the inline
          // counter-evidence block carries the load there).
          const paEnabled = process.env.VOUCH_GATE_COUNTER_EVIDENCE === "1";
          const groundedAsserts = verdict.pairs.filter(
            (p) => p.stance === "ASSERT" && p.grounded,
          );
          const anyKbCounter = verdict.pairs.some(
            (p) => (p.counter_evidence?.length ?? 0) > 0,
          );
          const paWebSuggestion =
            paEnabled && groundedAsserts.length > 0 && !anyKbCounter
              ? `[vouch-gate] ${groundedAsserts.length} grounded ASSERT(s) this turn cleared the KB counter-evidence check with no contradictions found. The KB lacks a counter for ~44% of dodge-fire entities on the latest dogfood baseline — for load-bearing claims, consider \`vouch search "<entity> limitations" --provider ddg\` to check the web for contradicting sources.\n`
              : "";
          const sessionMsg = sessionTid
            ? formatSessionSummary(sessionTid, verdict.reviseCheck, blindSpotsThisTurn, thisTurnStances) + paWebSuggestion
            : paWebSuggestion;

          // Opt-in escalation: when VOUCH_GATE_ESCALATE_UNADDRESSED=1 and
          // the revise-check reports stillUnaddressed prior-turn fires
          // and we're in strict mode, escalate to a block even if THIS
          // turn's draft didn't independently fire. Forcing function: an
          // agent cannot pass a turn while leaving the prior fire's
          // entity asserted-but-ungrounded. Default OFF — visibility-only
          // is enough for user-in-loop sessions.
          const escalateOn = process.env.VOUCH_GATE_ESCALATE_UNADDRESSED === "1";
          const escalatedFromUnaddressed =
            escalateOn &&
            opts.strict &&
            !verdict.blocked &&
            (verdict.reviseCheck?.stillUnaddressed.length ?? 0) > 0;
          const effectiveBlocked = verdict.blocked || escalatedFromUnaddressed;

          // Pre-fetch Exa candidates for any ungrounded ASSERT with no
          // KB candidate. Fail-open (empty Map when EXA_API_KEY unset or
          // network error). Only worth the call when we're going to emit
          // a fire message (effectiveBlocked) — skip on advisory-pass.
          const exaByEntity = effectiveBlocked
            ? await fetchExaForUngrounded(verdict)
            : undefined;

          if (!effectiveBlocked) {
            const autoGrounded = verdict.pairs.filter((p) => p.auto_grounded);
            const msgs: string[] = [deltaMsg];
            if (autoGrounded.length) msgs.push(formatAutoGroundMessage(autoGrounded));
            if (verdict.harvest) {
              const hm = formatHarvestMessage(verdict.harvest);
              if (hm) msgs.push(hm);
            }
            if (sessionMsg) msgs.push(sessionMsg);
            message = msgs.join("");
            exitCode = 0;
          } else if (!opts.strict) {
            message = formatBlockMessage(verdict, true, draft || "", exaByEntity) + deltaMsg + sessionMsg;
            exitCode = 0;
          } else if (escalatedFromUnaddressed && !verdict.blocked) {
            // Escalation: this turn passed its OWN gate
            // but a prior turn's fire is still unaddressed. Block with an
            // escalation message that names the lingering entities.
            message = formatEscalatedBlockMessage(verdict, draft || "") + deltaMsg + sessionMsg;
            exitCode = 2;
          } else {
            message = formatBlockMessage(verdict, false, draft || "", exaByEntity) + deltaMsg + sessionMsg;
            exitCode = 2;
          }
        }
      }
    }
  } catch (e) {
    errorMessage = e instanceof Error ? e.message : String(e);
    process.stderr.write(`⚠ [vouch-gate] ERROR: ${errorMessage} — turn NOT gated\n`);
    exitCode = failMode === "block" ? 2 : 0;
    incomplete = true;
    verdict = { blocked: false, pairs: [] };
  } finally {
    clearTimeout(timer);
  }

  const nPropositions = extractedPairs?.length ?? verdict.pairs.length;
  const nChecked = verdict.pairs.length;
  const nGrounded = verdict.pairs.filter((p) => p.grounded).length;
  const nUngrounded = nChecked - nGrounded;

  let verdictStr: "pass" | "block" | "incomplete" | "error";
  if (errorMessage) verdictStr = "error";
  else if (incomplete) verdictStr = "incomplete";
  else if (verdict.blocked && opts.strict) verdictStr = "block";
  else verdictStr = "pass";

  const auditEntry: AuditEntry = {
    ts: new Date().toISOString(),
    pid: process.pid,
    draft_sha256: sha256Prefix(draft || ""),
    n_propositions: nPropositions,
    n_checked: nChecked,
    n_grounded: nGrounded,
    n_ungrounded: nUngrounded,
    verdict: verdictStr,
    wall_ms: Date.now() - wallStart,
    exit_code: exitCode,
    mode,
    ...(errorMessage ? { error: errorMessage } : {}),
  };

  writeGateLog(auditEntry);

  const result: GateRunResult & { incomplete?: boolean } = { verdict, exitCode, message };
  if (incomplete) result.incomplete = true;
  return result;
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

function formatHarvestMessage(h: HarvestResult): string {
  const out: string[] = [];
  if (h.filed.length) {
    const lines = h.filed.map((f) => {
      const dep = f.depends_on.length ? ` ←(${f.depends_on.join(", ")})` : "";
      return `  • [${f.claim_type}] claim ${f.claim_id}${dep}: "${f.claim_text.slice(0, 160)}"`;
    });
    out.push(
      `[vouch-gate] harvested ${h.filed.length} derived claim(s) from your draft's tags — recorded in the KB:\n${lines.join("\n")}\n`,
    );
  }
  if (h.skipped.length) {
    out.push(
      `[vouch-gate] ${h.skipped.length} tagged segment(s) already in the KB (not re-filed): ${h.skipped
        .map((s) => `claim ${s.claim_id}`)
        .join(", ")}\n`,
    );
  }
  if (h.flags.length) {
    out.push(`[vouch-gate] tag advisories:\n${h.flags.map((f) => `  • ${f}`).join("\n")}\n`);
  }
  return out.join("");
}

/** Pre-fetch Exa candidates for any ungrounded ASSERT that has no KB
 *  candidate and no counter-evidence — the case where the agent
 *  historically silent-deletes because the verify path is too long.
 *  Inlining the canonical source(s) collapses fetch+rephrase from two
 *  turns to one (3/8 in dogfood simulation 2026-05-15). Parallel, fail-
 *  open: missing EXA_API_KEY or network error → empty Map, no fire. */
async function fetchExaForUngrounded(verdict: GateVerdict): Promise<Map<string, ExaCandidate[]>> {
  const targets = verdict.pairs.filter(
    (p) => !p.grounded && !p.auto_grounded && p.stance === "ASSERT" &&
           (!p.kb_candidates || p.kb_candidates.length === 0) &&
           (!p.counter_evidence || p.counter_evidence.length === 0)
  );
  if (!targets.length) return new Map();
  const results = await Promise.all(
    targets.map(async (p) => {
      const candidates = await searchWithText(p.proposition, { numResults: 3 });
      return [p.entity, candidates] as const;
    })
  );
  return new Map(results);
}

function formatBlockMessage(verdict: GateVerdict, advisory: boolean, draft: string, exaByEntity?: Map<string, ExaCandidate[]>): string {
  const ungrounded = verdict.pairs.filter((p) => !p.grounded);
  const lines: string[] = [];
  for (const p of ungrounded) {
    const checked =
      p.session_sources_checked && p.session_sources_checked > 0
        ? ` — checked ${p.session_sources_checked} session source(s), none entailed`
        : "";
    lines.push(`  • ${boldRed(p.entity)}: "${p.proposition.slice(0, 200)}" (${p.reason})${checked}`);
    if (p.hint) {
      const safeProp = p.proposition.replace(/"/g, '\\"');
      lines.push(
        `      — looks like a session observation (you ran \`${p.hint.uri}\` this session): ` +
          `\`vouch attest --from-session-tool ${p.hint.tool_use_id} --stance observation --claim "${safeProp}"\` to record it`,
      );
      continue;
    }
    // Surface counter-evidence FIRST when present — it's the strongest
    // signal that the claim needs work. Even with confirming evidence, a
    // strong counter forces a reconciliation choice. Then inline KB
    // nearest candidates: the gate's own grounding step already considered
    // these and the agent doesn't need to re-run `vouch search` to find
    // them. Render before any `vouch search` suggestion so "re-claim
    // against existing dossier" is offered before "fetch a fresh source".
    if (p.counter_evidence && p.counter_evidence.length > 0) {
      lines.push(`      ⚠ counter-evidence in KB — this claim has BOTH supporting AND contradicting prior claims:`);
      for (const c of p.counter_evidence) {
        const truncatedClaim = c.claim_text.length > 160
          ? c.claim_text.slice(0, 160) + "…"
          : c.claim_text;
        const dossierLine = c.dossier_slug
          ? `\n         dossier: ${c.dossier_slug}`
          : "";
        lines.push(
          `        [claim ${c.claim_id}, contra=${c.contradiction_score.toFixed(2)}] "${truncatedClaim}"${dossierLine}\n         why contradicts: ${c.contradiction_reason.slice(0, 200)}`,
        );
      }
      lines.push(
        `      → reconcile: edit the claim to acknowledge the disagreement (e.g. "X is widely held; Y is the counter-position from <source>")`,
      );
      lines.push(
        `      → or supersede: if the new claim is correct, file with --supersedes ${p.counter_evidence[0]!.claim_id} citing why`,
      );
      lines.push(
        `      → or hedge: "(unverified, sources disagree)" near the entity if you can't decide right now`,
      );
    }
    if (p.kb_candidates && p.kb_candidates.length > 0) {
      lines.push(`      KB nearest:`);
      for (const c of p.kb_candidates) {
        const truncatedClaim = c.claim_text.length > 160
          ? c.claim_text.slice(0, 160) + "…"
          : c.claim_text;
        const dossierLine = c.dossier_slug
          ? `\n         dossier: ${c.dossier_slug}`
          : "";
        const reasonLine = c.nli_reason
          ? `\n         NLI gap: ${c.nli_reason.slice(0, 200)}`
          : "";
        lines.push(
          `        [claim ${c.claim_id}, NLI=${c.nli_score.toFixed(2)}] "${truncatedClaim}"${dossierLine}${reasonLine}`,
        );
      }
      // Show both productive paths the agent can choose between.
      const closestSlug = p.kb_candidates[0]!.dossier_slug;
      if (closestSlug) {
        const safeProp = p.proposition.replace(/"/g, '\\"');
        lines.push(
          `      → if rephrasing the claim closes the gap: vouch claim "${safeProp}" --type ATOMIC --dossier ${closestSlug}`,
        );
      }
      lines.push(`      → for a fresh source: ${suggestVerification(p.entity, draft)}`);
    } else if (!p.counter_evidence?.length) {
      // No KB candidates AND no counter-evidence. Phase 1 hybrid search
      // returned nothing relevant. If Exa results were pre-fetched (via
      // fetchExaForUngrounded), inline them so the agent sees canonical
      // source(s) up-front — converts the historical 2-turn (fetch then
      // rephrase) sequence into a 1-turn rephrase in 3/8 of dogfood
      // simulation cases (2026-05-15). Without Exa results: fall back to
      // suggestion-only.
      const exaCandidates = exaByEntity?.get(p.entity);
      if (exaCandidates && exaCandidates.length > 0) {
        lines.push(`      Fresh web sources for ${p.entity}:`);
        for (let i = 0; i < exaCandidates.length; i++) {
          const c = exaCandidates[i]!;
          const excerpt = c.text.slice(0, 250).replace(/\s+/g, " ").trim();
          lines.push(`        [${i + 1}] ${c.url}`);
          if (excerpt) lines.push(`            "${excerpt}…"`);
        }
        lines.push(`      → reconcile against the source(s) above, or \`vouch fetch <url>\` for the full page`);
      } else {
        lines.push(renderSuggestionLine(suggestVerification(p.entity, draft)));
      }
    }
  }
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
    : `\nVerify; don't delete. The source exists.\n` +
      `  • vouch search "<entity>"\n` +
      `  • vouch fetch <url>${fetchHint}\n` +
      `  • vouch claim "<text>" --dossier <slug>\n` +
      `If the source contradicts your draft, rephrase. Silent delete hides what you didn't check. Hedge "(unverified, from training memory)" only if no source can verify.\n`;
  return `${header}\n${lines.join("\n")}${agNote}${guidance ? "\n" + guidance : "\n"}`;
}
