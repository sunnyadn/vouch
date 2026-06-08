// v1 prose-gate — absorbed (adapted, not copied) from vouch's Stop-hook claim
// gate, REPOINTED at vouch's own evidence graph so it covers the own-work-fact
// quadrant vouch deliberately skips. v1 scope, locked 2026-05-30:
//
//   - SCOPE: own-work claims only (test counts, results, commits, the running
//     model, own tool output) — NOT third-party entities (that would resurrect
//     vouch's dropped fetch/KB/embedding substack).
//   - RESOLVER: explicit-tag. The agent tags a factual claim with an evidence
//     id (`[ev: <id>]`); the gate (a) fires if the cited id is not in the
//     evidence set (dangling / fabricated ref — cf. vouch's unknownIdRefs), and
//     (b) NLI-checks that the cited evidence actually entails the claim, firing
//     on unsupported (false attribution). It does NOT catch bare untagged lies
//     — that is the v2 extractor's job (see bench/prose-gate/README.md).
//   - TIMING: Stop = advise, conclude = block. Enforced by the hook wiring, not
//     by this function.
//   - FAIL-OPEN: a verifier error (transient or content) never fires — infra
//     failure must not block the agent. The reason is recorded so the trail
//     shows the check was skipped, not silently passed.
//
// The NLI verifier is INJECTED (NliVerify) so this decision logic is unit-tested
// deterministically (tests/prose-gate.test.ts) while the eval drives the real
// Anthropic judge (bench/prose-gate/eval.ts).

import { classifyError, type NliVerify } from "./nli.ts";

export interface ProseGateInput {
  /** the agent's outgoing draft text; may carry `[ev: <id>]` tags */
  draft: string;
  /** evidence id -> the grounding content the cited claim is checked against */
  evidence: Record<string, string>;
}

export interface ProseGateVerdict {
  /** true = a tagged claim is ungrounded or mis-attributed (gate should fire) */
  fires: boolean;
  /** one human-readable reason per fired claim */
  reasons: string[];
  /** non-firing notes (fail-open skips) — kept for the audit trail */
  notes: string[];
}

export interface TaggedClaim {
  /** the claim text the tag annotates (tag stripped, trimmed) */
  claim: string;
  /** the cited evidence id */
  evidenceId: string;
}

// Tag source. Do NOT share one global-flagged RegExp across an .exec() while-loop
// that also calls String.replace() with the same instance: .replace() resets the
// shared lastIndex to 0, so .exec() re-finds the first match forever — a real
// infinite loop that hung every test + eval on any tagged draft (proven by
// bounded repro: matchedIndex stayed 11, lastIndex reset to 0 every iteration).
// matchAll with a FRESH per-call regex owns its own iterator and can't be clobbered.
const EV_TAG_SRC = "\\[ev:\\s*([a-zA-Z0-9_]+)\\]";

/** Parse `[ev: <id>]` tags. Each tag's claim is the draft text from the end of
 *  the previous tag (or start) up to this tag, trimmed. The inter-tag slice
 *  cannot itself contain a tag, so no tag-stripping is needed.
 *  v1 limitation: a multi-sentence segment before a tag is sent whole to NLI;
 *  finer claim segmentation is a later refinement. */
export function parseTaggedClaims(draft: string): TaggedClaim[] {
  const out: TaggedClaim[] = [];
  let lastEnd = 0;
  for (const m of draft.matchAll(new RegExp(EV_TAG_SRC, "g"))) {
    const claim = draft.slice(lastEnd, m.index).trim();
    if (claim.length > 0) out.push({ claim, evidenceId: m[1]! });
    lastEnd = m.index + m[0].length;
  }
  return out;
}

export async function proseGateV1(
  input: ProseGateInput,
  verify: NliVerify,
): Promise<ProseGateVerdict> {
  const reasons: string[] = [];
  const notes: string[] = [];

  for (const { claim, evidenceId } of parseTaggedClaims(input.draft)) {
    const source = input.evidence[evidenceId];
    if (source === undefined) {
      // dangling / fabricated evidence ref — fire without an NLI round-trip.
      reasons.push(`cites [ev: ${evidenceId}] which is not in the evidence set`);
      continue;
    }
    try {
      const v = await verify(claim, source);
      if (!v.supported) {
        reasons.push(
          `[ev: ${evidenceId}] does not support "${claim}" — ${v.reason || "unsupported"}`,
        );
      }
    } catch (e) {
      const kind = classifyError(e);
      const msg = e instanceof Error ? e.message : String(e);
      // Fail open: never block on verifier failure. Record it.
      notes.push(`verifier ${kind} error on [ev: ${evidenceId}] — not blocked (${msg})`);
    }
  }

  return { fires: reasons.length > 0, reasons, notes };
}

// ===========================================================================
// v2 — UNtagged own-work claim detection (force-tag)
// ===========================================================================
//
// v1 only checks claims the agent explicitly tagged `[ev: id]`. No agent tags
// its own lie, so v1 caught 0 of this session's real fabrications. v2 closes
// that: an LLM extractor (extractOwnWork) surfaces UNtagged own-work
// result-claims (test counts, build/commit/runtime facts) so they can be
// flagged as unverified. v2 does NOT auto-ground them (NLI against an 800-row
// evidence set is expensive + weak); it composes with v1 — v2 catches the
// OMISSION (untagged result-claim), v1 then catches the MISATTRIBUTION once the
// agent adds the tag the advice asks for.
//
// Proven: bench/prose-gate/extractor-probe.ts caught 6/6 untagged fabrications,
// 0 FP on 4 controls (ev_kih7m90b6e).

import type { ExtractOwnWork, OwnWorkClaim } from "./extractor.ts";

export interface ProseGateV2Verdict {
  /** true = ≥1 untagged own-work claim surfaced (advise the agent to ground it) */
  fires: boolean;
  /** the untagged own-work claims (not already covered by an [ev:] tag) */
  untagged: OwnWorkClaim[];
  /** fail-soft notes (e.g. extractor returned nothing / errored) */
  notes: string[];
}

// Cheap deterministic PRE-FILTER: only spend an LLM extractor call when the
// draft actually looks like it carries an own-work result-claim. Mirrors v1's
// "free on untagged drafts" cost profile — a plain conversational turn pays
// nothing. Broad by design (a false-negative just skips one advise; the
// extractor itself is the precision layer). Covers the high-frequency
// fabrication shapes: counts+result-words, build/lint/tsc, commit-hash, and
// first-person result verbs.
const OWN_WORK_SHAPE =
  /\b\d+\s*(?:pass(?:e[ds])?|fail(?:e[ds])?|tests?|errors?)\b|\b(?:all green|0 fail|tsc|biome (?:clean|pass)|build (?:succeed|passed|failed|green)|compiles?\b)|\bcommit(?:ted)?\b[^.]*\b[0-9a-f]{7,40}\b|\b(?:I (?:ran|verified|tested|confirmed)|it (?:returned|printed|passed))\b/i;

/** Does the draft look like it asserts an own-work result? Cheap gate before
 *  the LLM extractor — skip the call entirely when false. */
export function looksLikeOwnWorkClaim(draft: string): boolean {
  return OWN_WORK_SHAPE.test(draft);
}

/** True when an extracted claim is already covered by an `[ev: id]` tag in the
 *  draft — i.e. its text falls within a tagged segment, so v1 already handles
 *  it and v2 must not double-flag. Normalized, containment-based (conservative:
 *  prefer NOT re-flagging a claim the agent did tag). */
function isCoveredByTag(claim: string, taggedSegments: string[]): boolean {
  const norm = (s: string) => s.toLowerCase().replace(/\s+/g, " ").trim();
  const c = norm(claim);
  if (!c) return true;
  return taggedSegments.some((seg) => {
    const s = norm(seg);
    return s.includes(c) || c.includes(s);
  });
}

/**
 * v2 gate: surface UNtagged own-work claims. Pure-ish — the LLM call is the
 * injected `extract`. Pre-filtered: returns clean immediately when the draft
 * carries no own-work-result shape (no LLM call). Claims already covered by an
 * `[ev:]` tag are dropped (v1 owns those). Fail-soft: an empty extraction is a
 * clean pass, never an error.
 */
export async function proseGateV2(
  draft: string,
  extract: ExtractOwnWork,
): Promise<ProseGateV2Verdict> {
  if (!looksLikeOwnWorkClaim(draft)) {
    return { fires: false, untagged: [], notes: [] };
  }
  const claims = await extract(draft);
  if (claims.length === 0) {
    return { fires: false, untagged: [], notes: [] };
  }
  const taggedSegments = parseTaggedClaims(draft).map((t) => t.claim);
  const untagged = claims.filter((c) => !isCoveredByTag(c.claim, taggedSegments));
  return { fires: untagged.length > 0, untagged, notes: [] };
}
