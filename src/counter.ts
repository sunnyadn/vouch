/** L4 counter-evidence verb — KB-wide contradiction lookup for a proposition.
 *
 * Two consumers:
 *
 *   1. Stop-hook gate (src/gate.ts), when VOUCH_GATE_COUNTER_EVIDENCE=1.
 *      Gate runs this on every grounded pair and flips grounded → ungrounded
 *      when a strong counter-claim exists on the same entity.
 *
 *   2. Agent-callable CLI: `vouch counter "<proposition>" [--entity X]`.
 *      Lets an agent adversarially probe the KB before committing to a
 *      claim ("does my KB already have a contradicting fact on this?").
 *
 * Single source of truth. Both call sites use findCounterEvidence(). gate.ts
 * wires the per-pair mutation; the CLI prints JSON.
 *
 * Tuning: thresholds mirror the session-contradiction path so the two
 * contradiction-detection surfaces (KB-wide here, same-session in gate.ts)
 * stay aligned. Default-off in gate (each grounded pair adds an embed +
 * search + up to TOPK verifyContradiction LLM calls, ~5–10s/turn).
 */

import { embedOne } from "./embedder.ts";
import * as store from "./store.ts";
import { verifyContradiction } from "./verifier.ts";

export const COUNTER_EVIDENCE_TOPK = 5;
export const COUNTER_EVIDENCE_MIN_COS = 0.55;
export const COUNTER_EVIDENCE_FIRE_SCORE = 0.75;
export const COUNTER_EVIDENCE_MAX_HITS = 2;

export type CounterClaim = {
  claim_id: number;
  claim_text: string;
  dossier_slug: string | null;
  contradiction_score: number;
  contradiction_reason: string;
};

export type FindCounterOpts = {
  topK?: number;
  minCos?: number;
  fireScore?: number;
  /** Claim id to exclude from results (e.g., the entailing match in the
   *  gate's grounding pass — don't compare a claim against itself). */
  excludeClaimId?: number;
  /** Stop after this many counter-hits. Defaults to COUNTER_EVIDENCE_MAX_HITS. */
  maxHits?: number;
  /** Cooperative cancellation: caller can set abortRef.aborted = true to
   *  short-circuit mid-flight (between LLM calls). */
  abortRef?: { aborted: boolean };
};

/** Entity-mention check. True when `text` mentions `entity` either as a
 *  direct substring (case-insensitive) or as an alphanumeric-folded
 *  substring with ≥3 chars. Mirrors gate.ts's sharesPrimaryEntity but
 *  takes the entity as a string instead of an ExtractedPair, so the
 *  module stays decoupled from gate.ts's type surface. */
function entityInText(entity: string, text: string): boolean {
  const ent = entity.toLowerCase().trim();
  if (ent && text.toLowerCase().includes(ent)) return true;
  const fold = (s: string) => s.toLowerCase().replace(/[^a-z0-9一-鿿]/g, "");
  const entAlnum = fold(entity);
  return entAlnum.length >= 3 && fold(text).includes(entAlnum);
}

/** Pull KB claims on the same entity that contradict the proposition.
 *  Returns an array of counter-claims (empty if none). Pure function — no
 *  side effects, caller decides what to do with the result.
 *
 *  Pipeline:
 *    1. Embed `<entity>. <proposition>`
 *    2. searchHybrid topK; filter to claims only
 *    3. For each hit: skip if low similarity, excluded id, unsupported,
 *       superseded, or not entity-mentioning
 *    4. verifyContradiction(proposition, candidate) — LLM NLI
 *    5. Keep if contradicts && score ≥ fireScore. Stop at maxHits.
 *
 *  Fail-soft: embed failures return []; verify failures skip that
 *  candidate but continue. Throws only if the caller's abort is hit.   */
export async function findCounterEvidence(
  proposition: string,
  entity: string,
  opts: FindCounterOpts = {},
): Promise<CounterClaim[]> {
  const topK = opts.topK ?? COUNTER_EVIDENCE_TOPK;
  const minCos = opts.minCos ?? COUNTER_EVIDENCE_MIN_COS;
  const fireScore = opts.fireScore ?? COUNTER_EVIDENCE_FIRE_SCORE;
  const maxHits = opts.maxHits ?? COUNTER_EVIDENCE_MAX_HITS;
  const abortRef = opts.abortRef;

  if (abortRef?.aborted) return [];

  let queryEmb: Float32Array;
  try {
    queryEmb = await embedOne(`${entity}. ${proposition}`);
  } catch {
    return [];
  }

  const hits = store.searchHybrid(queryEmb, topK).filter((h) => h.kind === "claim");
  const out: CounterClaim[] = [];
  for (const h of hits) {
    if (abortRef?.aborted) break;
    if (h.id == null) continue;
    if (h.similarity < minCos) continue;
    if (opts.excludeClaimId != null && h.id === opts.excludeClaimId) continue;
    const claim = store.getClaim(h.id);
    if (!claim) continue;
    if (claim.status !== "supported") continue;
    if (claim.superseded_by != null) continue;
    if (!entityInText(entity, claim.claim_text)) continue;
    try {
      const verdict = await verifyContradiction(proposition, claim.claim_text);
      if (verdict.contradicts && verdict.score >= fireScore) {
        out.push({
          claim_id: claim.id,
          claim_text: claim.claim_text,
          dossier_slug: claim.dossier_slug || null,
          contradiction_score: verdict.score,
          contradiction_reason: verdict.reason,
        });
      }
    } catch {
      // transient — skip candidate, continue
    }
    if (out.length >= maxHits) break;
  }
  return out;
}
