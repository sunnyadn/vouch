/** L3 UserPromptSubmit hook — pre-prompt augmentation.
 *
 * Fires once per user turn (BEFORE the agent drafts), reads the user prompt
 * from the CC hook stdin payload, and injects pre-prompt context.
 *
 * Two orthogonal context streams compose into one additionalContext block:
 *
 *   (a) Prompt-driven sources: KB-first lookup → Exa fallback. Mirrors
 *       L5's gate.ts contract (`fetchExaForUngrounded` only fires on
 *       `!p.kb_candidates`):
 *
 *         1. Embed prompt → searchHybrid against KB claims + dossiers
 *         2. If best similarity ≥ KB_HIT_THRESHOLD: inject KB matches,
 *            do NOT call Exa (avoid redundant fetch + dossier pollution)
 *         3. Else: call Exa, persist new web/exa dossiers, inject those
 *
 *   (b) Session humility nudge: read session_claims ledger via
 *       transcript_path, compute (HEDGE + SPECULATE) / (ASSERT + HEDGE +
 *       SPECULATE) for the session-so-far. Surface only when the ratio
 *       is below the healthy band (default 10%); above-band sessions
 *       don't need a nudge. Independent of the current prompt — surfaces
 *       even on heuristic-skipped prompts (the agent's disposition
 *       carries across turns regardless of whether THIS prompt triggers
 *       a lookup).
 *
 * Migrated from L5 fire-time path (gate.ts → fetchExaForUngrounded). The L5
 * path remains live in parallel for the 2-week observation window per the
 * spec's rollback policy — see docs/specs/2026-05-15-vouch-layer-architecture.md.
 *
 * Heuristic gate (skip the embed + Exa work entirely):
 *   1. Prompt < 10 chars (greeting / "thanks")
 *   2. Prompt > 5000 chars (code paste / file dump)
 *   3. ≥ 50% of non-whitespace chars are inside fenced code blocks
 *   4. VOUCH_USERPROMPT_BYPASS=1 (operator opt-out)
 *
 * Note: EXA_API_KEY is no longer part of the skip gate. With a populated
 * KB, useful pre-prompt context can come from the KB alone. Exa is the
 * fallback for KB-miss prompts — if it's unset, the KB-miss path returns
 * an empty envelope and the agent proceeds without pre-prompt context.
 *
 * Fail-open contract: any error returns exit 0 with empty additionalContext.
 * The hook is advisory — never blocks the user's prompt. Budget capped via
 * VOUCH_USERPROMPT_BUDGET_MS (default 8000 ≈ Exa client timeout).
 */

import { embedOne } from "./embedder.ts";
import { searchWithText, type ExaCandidate } from "./exa.ts";
import * as store from "./store.ts";

const DEFAULT_BUDGET_MS = 8000;
const MIN_PROMPT_CHARS = 10;
const MAX_PROMPT_CHARS = 5000;
const CODE_BLOCK_FRACTION_MAX = 0.5;
const NUM_RESULTS = 3;
const EXCERPT_CHARS = 200;
/** Minimum session-claim count before showing the humility ratio. Below
 *  this the ratio is noisy small-N (a single hedge in a 2-claim session
 *  isn't a real signal). Smaller than the gate's display floor of 10
 *  because pre-prompt nudging is cheap and starts paying off earlier. */
const HUMILITY_MIN_TRUTH = 3;
const HUMILITY_TARGET_LOW = 10;
const HUMILITY_TARGET_HIGH = 25;
/** Cosine threshold above which we consider the KB to "have" the entity.
 *  Embeddings are l2-normalized so this is dot-product. Picked
 *  conservatively at 0.65 — below this, the embedder is reaching for any
 *  topical hit and Exa's web ranking will likely produce something more
 *  relevant. Tune via dogfood once L3 has fire-rate data. */
const KB_HIT_THRESHOLD = 0.65;
const KB_TOPK = 3;

export type UserPromptSubmitInput = {
  prompt?: string;
  session_id?: string;
  transcript_path?: string;
  hook_event_name?: string;
};

export type UserPromptSubmitOutput = {
  hookSpecificOutput?: {
    hookEventName: "UserPromptSubmit";
    additionalContext?: string;
  };
};

export type SkipReason =
  | "empty"
  | "too_short"
  | "too_long"
  | "code_heavy"
  | "bypass";

export function shouldSkip(prompt: string | undefined): SkipReason | null {
  if (process.env.VOUCH_USERPROMPT_BYPASS === "1") return "bypass";
  if (!prompt) return "empty";
  const trimmed = prompt.trim();
  if (trimmed.length < MIN_PROMPT_CHARS) return "too_short";
  if (trimmed.length > MAX_PROMPT_CHARS) return "too_long";
  if (codeBlockFraction(trimmed) >= CODE_BLOCK_FRACTION_MAX) return "code_heavy";
  return null;
}

/** Fraction of non-whitespace chars enclosed in ``` fences. Used to skip
 *  prompts that are predominantly pasted code (rare to want web search on
 *  raw code content). Pairs of ``` define a block; lone trailing ``` is
 *  ignored.  */
export function codeBlockFraction(text: string): number {
  const fences = [...text.matchAll(/```/g)].map((m) => m.index!);
  if (fences.length < 2) return 0;
  const totalNonWs = text.replace(/\s/g, "").length;
  if (!totalNonWs) return 0;
  let insideNonWs = 0;
  for (let i = 0; i + 1 < fences.length; i += 2) {
    insideNonWs += text.slice(fences[i]!, fences[i + 1]!).replace(/\s/g, "").length;
  }
  return insideNonWs / totalNonWs;
}

/** Format Exa candidates into a structured context block. Kept terse — the
 *  agent's context is a finite resource and additionalContext rides with
 *  every subsequent message in the turn. */
export function formatExaContext(candidates: ExaCandidate[]): string {
  if (!candidates.length) return "";
  const lines: string[] = [];
  lines.push(`[vouch context] Pre-fetched ${candidates.length} fresh web source(s) for this turn (no confident KB hit) — persisted as web/exa dossiers:`);
  for (let i = 0; i < candidates.length; i++) {
    const c = candidates[i]!;
    const excerpt = c.text.slice(0, EXCERPT_CHARS).replace(/\s+/g, " ").trim();
    lines.push(`  [${i + 1}] ${c.url}`);
    if (c.title) lines.push(`      ${c.title.slice(0, 160)}`);
    if (excerpt) lines.push(`      "${excerpt}…"`);
  }
  lines.push(`Verify before asserting; cite via \`vouch claim "<quote>" --dossier <slug>\` if you use them. Hedge "(unverified, from training memory)" only when no source above can support your claim.`);
  return lines.join("\n");
}

/** Format KB matches into a structured context block. Distinguishes claims
 *  (verified statements with NLI history) from dossiers (raw source text).
 *  We show the top-N matches above threshold so the agent can ground its
 *  draft against them in the same turn. */
export function formatKbContext(hits: store.SearchHit[]): string {
  if (!hits.length) return "";
  const lines: string[] = [];
  lines.push(`[vouch context] KB has ${hits.length} match(es) likely relevant to this turn — cite these instead of refetching:`);
  for (let i = 0; i < hits.length; i++) {
    const h = hits[i]!;
    const sim = h.similarity.toFixed(2);
    const excerpt = (h.text || "").slice(0, EXCERPT_CHARS).replace(/\s+/g, " ").trim();
    if (h.kind === "claim") {
      lines.push(`  [${i + 1}] claim ${h.id} (sim=${sim}, status=${h.status || "?"}): "${excerpt}"`);
    } else {
      lines.push(`  [${i + 1}] dossier ${h.slug} (sim=${sim}): ${h.title || h.source_url || ""}`);
      if (excerpt) lines.push(`      "${excerpt}…"`);
    }
  }
  lines.push(`Inspect a hit: \`vouch get-claim <id>\` / \`vouch get-dossier <slug>\`. Cite via \`vouch claim "<quote>" --dossier <slug>\`.`);
  return lines.join("\n");
}

/** Format the session-so-far humility ratio as a single-line context
 *  nudge. Renders only when the session has at least HUMILITY_MIN_TRUTH
 *  truth-bearing claims, and only when the ratio is below the healthy
 *  target band. Above-target / in-band sessions don't need a nudge;
 *  emitting one anyway would be noise. */
export function formatHumilityContext(c: {
  asserts: number;
  hedges: number;
  speculates: number;
}): string {
  const truthBearing = c.asserts + c.hedges + c.speculates;
  if (truthBearing < HUMILITY_MIN_TRUTH) return "";
  const uncertain = c.hedges + c.speculates;
  const ratePct = (uncertain / truthBearing) * 100;
  if (ratePct >= HUMILITY_TARGET_LOW) return "";
  const rateStr = ratePct.toFixed(1);
  return (
    `[vouch context] Session-so-far humility: ${uncertain}/${truthBearing} = ${rateStr}% explicit-uncertainty ` +
    `(${c.asserts} assert / ${c.hedges} hedge / ${c.speculates} speculate). ` +
    `Healthy band ${HUMILITY_TARGET_LOW}-${HUMILITY_TARGET_HIGH}%. ` +
    `Hedge load-bearing claims you can't fully verify, or surface a specific \`[gap: <facet>]\` this turn.`
  );
}

/** Read session humility counts from the session_claims ledger. Returns
 *  zero counts on any read failure or missing transcript_path — caller
 *  treats that the same as a fresh session (no nudge). */
export function lookupHumility(transcript_path: string | undefined): {
  asserts: number;
  hedges: number;
  speculates: number;
} {
  const empty = { asserts: 0, hedges: 0, speculates: 0 };
  if (!transcript_path) return empty;
  try {
    const transcript_id = transcript_path.split("/").pop()?.replace(/\.jsonl$/, "") || "";
    if (!transcript_id) return empty;
    const c = store.getSessionFireCounts(transcript_id);
    return { asserts: c.asserts, hedges: c.hedges, speculates: c.speculates };
  } catch {
    return empty;
  }
}

/** KB-first lookup. Embeds the prompt, runs hybrid search, returns hits
 *  above the configured threshold (capped at KB_TOPK). Empty array on any
 *  failure — caller falls through to Exa. */
export async function lookupKb(prompt: string): Promise<store.SearchHit[]> {
  try {
    const emb = await embedOne(prompt);
    const all = store.searchHybrid(emb, KB_TOPK);
    return all.filter((h) => h.similarity >= KB_HIT_THRESHOLD);
  } catch {
    return [];
  }
}

export async function runUserPromptSubmit(input: UserPromptSubmitInput): Promise<UserPromptSubmitOutput> {
  const skip = shouldSkip(input.prompt);
  if (skip) {
    // Humility nudge is independent of the prompt-driven KB/Exa lookup and
    // worth surfacing even on skipped prompts — it's about the agent's
    // session disposition, not the current prompt's content.
    return wrap(formatHumilityContext(lookupHumility(input.transcript_path)));
  }

  const prompt = input.prompt!.trim();
  const budgetMs = parseInt(process.env.VOUCH_USERPROMPT_BUDGET_MS || "", 10) || DEFAULT_BUDGET_MS;
  const humilityLine = formatHumilityContext(lookupHumility(input.transcript_path));

  // KB-first. The embed call is the cost of the lookup; it's also the cost
  // of writing dossiers, so a populated KB is a free hit-rate boost.
  const kbHits = await Promise.race([
    lookupKb(prompt),
    new Promise<store.SearchHit[]>((resolve) => setTimeout(() => resolve([]), budgetMs)),
  ]);
  if (kbHits.length > 0) {
    return wrap(joinContext(formatKbContext(kbHits), humilityLine));
  }

  // KB miss → fall through to Exa. searchWithText fail-opens internally:
  // missing EXA_API_KEY / network error → []. We pass numResults + timeout
  // so Exa can't outlive the outer wall budget.
  if (!process.env.EXA_API_KEY) return wrap(humilityLine);
  const candidates = await Promise.race([
    searchWithText(prompt, { numResults: NUM_RESULTS, timeoutMs: budgetMs }).catch(() => [] as ExaCandidate[]),
    new Promise<ExaCandidate[]>((resolve) => setTimeout(() => resolve([]), budgetMs + 500)),
  ]);
  return wrap(joinContext(formatExaContext(candidates), humilityLine));
}

function joinContext(...blocks: string[]): string {
  return blocks.filter((b) => b).join("\n\n");
}

function wrap(additionalContext: string): UserPromptSubmitOutput {
  if (!additionalContext) return {};
  return {
    hookSpecificOutput: {
      hookEventName: "UserPromptSubmit",
      additionalContext,
    },
  };
}
