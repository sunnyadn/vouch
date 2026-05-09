/** Stop-hook confabulation gate.
 *
 * Pipeline:
 *   1. Read the last assistant text from a Claude Code transcript.
 *   2. Use a fast LLM to extract {entity, assertion} pairs that are factual
 *      claims about specific named external entities AND lack an explicit
 *      "(unverified)" hedge nearby.
 *   3. For each pair: hybrid-search vouch's claim KB; for each supported
 *      candidate run NLI (assertion vs claim_text + dossier source quote).
 *      Grounded if ANY supported claim entails the assertion.
 *   4. Block (exit 2) if any pair is ungrounded; otherwise pass.
 *
 * Fail-open: classifier/network/transient errors → exit 0, never block.
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

const ExtractSchema = z.object({
  pairs: z
    .array(
      z.object({
        entity: z.string(),
        assertion: z.string(),
      }),
    )
    .max(20),
});

function buildExtractPrompt(draft: string): string {
  const projectsEnv = (process.env.VOUCH_GATE_WORKSPACE_PROJECTS || "").trim();
  const projectsLine = projectsEnv
    ? `\n      Known workspace projects for this installation (claims about their command surface, flags, file paths, test counts, build state, internal architecture, or runtime state are workspace, not external): ${projectsEnv}.`
    : "";
  return `You are a fact-grounding gate. The assistant has just produced a draft response. Extract every assertion that REQUIRES vouch verification.

INCLUDE an assertion when ALL of these hold:
  - It states a SPECIFIC factual property (number, version, capability, comparison, pricing, attribution, perf, organizational fact) of a NAMED EXTERNAL ENTITY (third-party dataset, paper, product, library, model, company, person, benchmark)
  - It is NOT hedged near the claim with "(unverified)", "from training memory", "without verifying", "I haven't verified", "let me verify", "I'm going to verify now", "凭印象", or equivalent caveat
  - It is NOT generic common knowledge or domain-textbook background. Generic common knowledge includes: language features, OS basics, well-known algorithm / methodology / statistical-test names with method-of-X descriptions ("X is a method for Y" / "X is a model used for Y" / "X is a statistical test for Y" / "X decomposes Y into Z"), classic papers' broad categories, mathematical primitives. The bar is "would a textbook in the relevant field state this without citation?" — if yes, skip. (Examples that should skip: "Fine-Gray is a statistical model for competing risks", "Fourier transform decomposes a signal into frequencies", "Cox regression models hazard ratios", "Gray test compares cumulative incidence functions".)
  - It is NOT workspace context. Workspace covers ALL of these — when in doubt, treat as workspace and skip:
      (a) The assistant's own actions, plans, recommendations, framing, or meta-commentary about the conversation itself.
      (b) Properties of any project the assistant is acting as maintainer / author / dogfooder of — its command surface, flags, file paths, function names, build artifacts, test counts, commit hashes, internal architecture, current runtime state, OR FEATURE-SUPPORT / ROADMAP claims about that project ("X is supported in our toolkit", "we have built-in support for Y", "Z is on the roadmap", "our library covers W", "feature V is missing in our package"). Even when X / Y / Z is itself a third-party methodology or external entity, a claim about whether the user's project supports / will support / lacks it is workspace, not external.${projectsLine}
      (c) Anything the assistant plausibly observed via a tool call earlier in the same session (Bash command output, file contents read, git log/diff/show output, test results, HTTP responses, database queries). The transcript itself is the source of those — vouch is not the right gate.
      (d) Forward-looking, hypothetical, or proposed entities that the assistant frames as not-yet-existing ("I'll file ISSUE-X", "a proposed feature would ...", "the planned issue tracks ..."). Nothing external to verify against until the entity actually exists.
      (e) Internal issue-tracker IDs (Linear / Jira / GitHub issue numbers) and their described scope when the assistant is filing, summarizing, or proposing them — these are workspace coordination, not external claims.

ALSO SKIP entities annotated as already-grounded by the vouch KB. If a named entity is annotated within a few words by "(claim N)", "(vouch claim N)", "(claim_id: N)", "(claim_ids: A,B,C)", or "supported (NLI)" / "(supported NLI)", the assistant has already grounded that entity through the vouch KB — DO NOT extract a pair for it. The annotation IS the verification handle; downstream DB-lookup by entity name is redundant and produces false positives because the lookup-key is the entity, not the claim_id. Treat such annotations the same as "(unverified)" hedge tokens for the purpose of suppression: the entity's properties have an explicit grounding signal, so skip.

For each, return { entity: short canonical name, assertion: 1-sentence paraphrase or verbatim of the claim }.

If nothing qualifies, return { pairs: [] }.

Draft:
<<<
${draft}
>>>`;
}

export interface ExtractedPair {
  entity: string;
  assertion: string;
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
    const verdict = await verifyClaimAgainstSource(pair.assertion, source);
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
      ? `${hits.length} candidate(s) found but none entailed the assertion`
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
    (p) => `  • ${p.entity}: "${p.assertion.slice(0, 200)}" (${p.reason})`,
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
