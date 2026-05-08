import { generateObject } from "ai";
import { z } from "zod";

import { MAX_SOURCE_CHARS, VERIFIER_MODEL } from "./config.ts";
import { getLanguageModel } from "./providers.ts";
import { embedBatch } from "./embedder.ts";
import * as store from "./store.ts";
import { findQuoteInContent } from "./quote-match.ts";
import type { Claim, ClaimDependency, ClaimType, VerifyResult } from "./types.ts";

/** Thrown when the verifier failed for a system reason (auth expired, network
 *  unreachable, provider quota) — NOT for an evidence outcome. Caller must
 *  bail without recording the claim, since these errors are not informative
 *  about the (claim, source) pair. */
export class TransientVerifierError extends Error {
  constructor(public kind: "auth" | "network" | "quota" | "unknown", message: string, public hint?: string) {
    super(message);
    this.name = "TransientVerifierError";
  }
}

export function classifyError(e: unknown): TransientVerifierError | null {
  const msg = (e instanceof Error ? e.message : String(e)) || "";
  const lower = msg.toLowerCase();
  if (
    lower.includes("invalid_rapt") ||
    lower.includes("reauth") ||
    lower.includes("invalid_grant") ||
    lower.includes("could not load the default credentials") ||
    lower.includes("unauthenticated") ||
    /\b401\b/.test(msg)
  ) {
    return new TransientVerifierError(
      "auth",
      "Vertex / provider auth expired or missing.",
      "Refresh credentials. For Vertex: run `gcloud auth application-default login`. For OpenAI/Anthropic: check the API key env var.",
    );
  }
  if (lower.includes("rate limit") || /\b429\b/.test(msg) || lower.includes("resource_exhausted")) {
    return new TransientVerifierError("quota", `Provider rate limit / quota: ${msg.slice(0, 200)}`);
  }
  if (lower.includes("econnrefused") || lower.includes("etimedout") || lower.includes("enetunreach")) {
    return new TransientVerifierError("network", `Network unreachable: ${msg.slice(0, 200)}`);
  }
  return null;
}

const CHUNK_SIZE = 1500;
const CHUNK_OVERLAP = 300;
const RETRIEVAL_K = 3;

const VerifySchema = z.object({
  supported: z.boolean(),
  score: z.number().min(0).max(1),
  reason: z.string().max(500),
});

const VERIFIER_PROMPT_TEMPLATE = `You verify factual claims against source text.

CLAIM: "{CLAIM}"

SOURCE:
---
{SOURCE}
---

Question: Does the SOURCE genuinely support the CLAIM?

Rules:
- Be LENIENT on phrasing — same factual content with different words IS supported.
- Be STRICT on facts — every entity, number, dataset, baseline, or causal relationship in the claim must trace to the source.
- ABSENCE CLAIMS ("source does not mention X" / "X is missing from the paper"):
    supported = source genuinely lacks X.
    unsupported = source actually contains X (claim is wrong about absence).
- TABLE-LOOKUP CLAIMS ("system A scored Y% on dataset Z"):
    supported requires the cell at the right row × column.
    A correct number at the wrong position FAILS.
- Reject "almost-true" claims that overstate, generalize, or paraphrase beyond what the source says.

Return your verdict as JSON: { supported, score (0..1 confidence), reason (one sentence) }.`;

function chunkText(text: string): string[] {
  const out: string[] = [];
  let body = text;
  if (body.startsWith("REPO METADATA")) {
    const idx = body.indexOf("--- README ---");
    if (idx > 0) {
      out.push(body.slice(0, idx).trim());
      body = body.slice(idx + "--- README ---".length).trim();
    }
  }
  if (body.length <= CHUNK_SIZE) {
    if (body) out.push(body);
    return out.length ? out : [""];
  }
  const step = CHUNK_SIZE - CHUNK_OVERLAP;
  for (let i = 0; i < body.length; i += step) {
    const c = body.slice(i, i + CHUNK_SIZE);
    if (c) out.push(c);
  }
  return out;
}

async function retrieveRelevant(content: string, claim: string, k = RETRIEVAL_K): Promise<string> {
  const chunks = chunkText(content);
  if (chunks.length <= k) return chunks.join("\n\n---\n\n");
  const embs = await embedBatch([...chunks, claim]);
  const claimEmb = embs[embs.length - 1]!;
  const sims = embs.slice(0, -1).map((e, i) => ({
    i,
    sim: cosine(e, claimEmb),
  }));
  sims.sort((a, b) => b.sim - a.sim);
  const chosen = sims.slice(0, k).map((s) => s.i).sort((a, b) => a - b);
  return chosen.map((i) => chunks[i]!).join("\n\n---\n\n");
}

function cosine(a: Float32Array, b: Float32Array): number {
  const n = Math.min(a.length, b.length);
  let dot = 0;
  for (let i = 0; i < n; i++) dot += a[i]! * b[i]!;
  return dot;
}

export async function verifyClaimAgainstSource(
  claim: string,
  source: string,
): Promise<VerifyResult> {
  const truncated = source.slice(0, MAX_SOURCE_CHARS);
  const prompt = VERIFIER_PROMPT_TEMPLATE.replace("{CLAIM}", claim).replace(
    "{SOURCE}",
    truncated,
  );
  try {
    const { object } = await generateObject({
      model: getLanguageModel(VERIFIER_MODEL),
      schema: VerifySchema,
      prompt,
      temperature: 0.0,
    });
    return {
      status: object.supported ? "supported" : "unsupported",
      score: object.score,
      source_passage: object.reason,
      verifier: VERIFIER_MODEL,
    };
  } catch (e: any) {
    // Re-throw transient/system errors so the caller can bail without
    // recording a phantom claim. Only "verifier returned an answer that
    // wasn't parseable" or other content-level issues fall through to the
    // insufficient-status path.
    const transient = classifyError(e);
    if (transient) throw transient;
    return {
      status: "insufficient",
      score: 0,
      source_passage: `verifier error: ${e?.message || String(e)}`.slice(0, 300),
      verifier: `${VERIFIER_MODEL}-error`,
    };
  }
}

export async function verifyClaim(
  claim: string,
  dossierSlug: string,
  opts: {
    topic?: string;
    author?: string;
    claim_type?: ClaimType;
    attribution?: string;
    /** Verbatim quote — when provided (the strict-mode path post-`vouch fetch`),
     *  NLI runs against THIS, not against retrieved chunks of the full dossier.
     *  The caller has already verified that this quote appears in the dossier. */
    source_quote?: string;
    source_offset_start?: number | null;
    source_offset_end?: number | null;
  } = {},
): Promise<VerifyResult & { claim_id: number }> {
  const dossier = store.getDossier(dossierSlug);
  if (!dossier) {
    return {
      status: "insufficient",
      score: 0,
      source_passage: "dossier not found",
      verifier: VERIFIER_MODEL,
      claim_id: 0,
    };
  }
  const content = dossier.content || "";

  let source: string;
  if (opts.source_quote) {
    source = opts.source_quote;
  } else if (content.length < 50) {
    return {
      status: "insufficient",
      score: 0,
      source_passage: content.slice(0, 500),
      verifier: VERIFIER_MODEL,
      claim_id: 0,
    };
  } else if (content.length <= CHUNK_SIZE * 2) {
    source = content;
  } else {
    source = await retrieveRelevant(content, claim);
  }
  const result = await verifyClaimAgainstSource(claim, source);

  // Attribution priority: explicit --attribution arg > dossier.author_attribution.
  // Do NOT auto-fill from dossier.title — title often defaults to the URL when
  // --source-title isn't passed, which is a poor "attribution" string.
  const attribution = opts.attribution || dossier.author_attribution || null;

  // Quote position: prefer caller-provided offsets (computed from quote-match);
  // fall back to null. The legacy "whole quote-only dossier" heuristic is gone.
  const offsetStart = opts.source_offset_start ?? null;
  const offsetEnd = opts.source_offset_end ?? null;

  // Embed the claim text so /search_kb can index it
  let claimEmbedding: Float32Array | null = null;
  try {
    const embs = await embedBatch([claim]);
    claimEmbedding = embs[0] || null;
  } catch {
    // Embedding failure is non-fatal — claim still gets stored, just won't be searchable
  }

  const cid = store.recordClaim({
    dossier_slug: dossierSlug,
    claim_text: claim,
    score: result.score,
    status: result.status,
    source_passage: result.source_passage,
    claim_type: opts.claim_type || "ATOMIC",
    topic: opts.topic ?? null,
    author: opts.author ?? null,
    attribution,
    source_offset_start: offsetStart,
    source_offset_end: offsetEnd,
    embedding: claimEmbedding,
    verification: "nli-quote",
  });

  return { ...result, claim_id: cid };
}

// ---------------------------------------------------------------------------
// INFERENCE / INTERPRETATION verifiers (v0.2)
// ---------------------------------------------------------------------------

const INFERENCE_PROMPT_TEMPLATE = `You verify whether a CONCLUSION follows logically from a set of PREMISES.

PREMISES:
{PREMISES_BLOCK}

CONCLUSION: "{CLAIM}"

Question: Does the CONCLUSION follow strictly from the PREMISES?

Rules:
- supported = the conclusion follows from the premises alone, without needing
  unstated assumptions, world knowledge, or facts not in the premises.
- unsupported = the conclusion needs unstated premises, OR overreaches what
  the premises actually say, OR doesn't follow at all.
- A conclusion that is *probably true* in the world but does NOT follow
  strictly from the premises = unsupported.
- A conclusion that adds new factual content not in the premises = unsupported.

Return: { supported, score (0..1 confidence), reason (one sentence) }.`;

const INTERPRETATION_PROMPT_TEMPLATE = `You verify whether a NEW STATEMENT is a faithful reframing of a SOURCE CLAIM.

SOURCE CLAIM:
{SOURCE_BLOCK}

NEW STATEMENT: "{CLAIM}"

Question: Is the NEW STATEMENT a faithful reframing of the SOURCE CLAIM?

Rules:
- supported = the new statement says the same factual content with different
  words. Paraphrase, summary, or restatement — but no new facts.
- unsupported = the new statement adds factual content not in the source,
  OR drops factual qualifiers that change the meaning, OR generalizes beyond
  the source's actual scope.
- "Same fact, different words" = supported.
- "Source said X applies to Y; reframing says X applies broadly" = unsupported.

Return: { supported, score (0..1 confidence), reason (one sentence) }.`;

export async function verifyInferenceClaim(
  claim: string,
  upstreamClaimIds: number[],
): Promise<VerifyResult> {
  const upstreams: (Claim & { depends_on: ClaimDependency[] })[] = [];
  const missing: number[] = [];
  for (const id of upstreamClaimIds) {
    const c = store.getClaim(id);
    if (!c) missing.push(id);
    else upstreams.push(c);
  }
  if (missing.length) {
    return {
      status: "insufficient",
      score: 0,
      source_passage: `depends_on references missing claim(s): ${missing.join(", ")}`,
      verifier: VERIFIER_MODEL,
    };
  }

  const broken: string[] = [];
  for (const c of upstreams) {
    if (c.status === "unsupported") {
      broken.push(`${c.id} (status=unsupported)`);
    }
    if (c.superseded_by != null) {
      broken.push(`${c.id} (superseded by ${c.superseded_by})`);
    }
  }
  if (broken.length) {
    return {
      status: "unsupported",
      score: 0,
      source_passage: `depends on broken-chain claim(s): ${broken.join(", ")}`,
      verifier: VERIFIER_MODEL,
    };
  }

  const premisesBlock = upstreams
    .map((c) => `[Claim ${c.id}, ${c.claim_type}, ${c.status}]: ${c.claim_text}`)
    .join("\n");

  const prompt = INFERENCE_PROMPT_TEMPLATE.replace("{PREMISES_BLOCK}", premisesBlock).replace(
    "{CLAIM}",
    claim,
  );

  try {
    const { object } = await generateObject({
      model: getLanguageModel(VERIFIER_MODEL),
      schema: VerifySchema,
      prompt,
      temperature: 0.0,
    });
    return {
      status: object.supported ? "supported" : "unsupported",
      score: object.score,
      source_passage: object.reason,
      verifier: VERIFIER_MODEL,
    };
  } catch (e: any) {
    const transient = classifyError(e);
    if (transient) throw transient;
    return {
      status: "insufficient",
      score: 0,
      source_passage: `verifier error: ${e?.message || String(e)}`.slice(0, 300),
      verifier: `${VERIFIER_MODEL}-error`,
    };
  }
}

export async function verifyInterpretationClaim(
  claim: string,
  upstreamClaimIds: number[],
): Promise<VerifyResult> {
  if (upstreamClaimIds.length !== 1) {
    return {
      status: "insufficient",
      score: 0,
      source_passage: `INTERPRETATION requires exactly one upstream claim, got ${upstreamClaimIds.length}`,
      verifier: VERIFIER_MODEL,
    };
  }
  const upstreamClaimId = upstreamClaimIds[0]!;
  const c = store.getClaim(upstreamClaimId);
  if (!c) {
    return {
      status: "insufficient",
      score: 0,
      source_passage: `depends_on references missing claim: ${upstreamClaimId}`,
      verifier: VERIFIER_MODEL,
    };
  }
  if (c.status === "unsupported") {
    return {
      status: "unsupported",
      score: 0,
      source_passage: `depends on broken-chain claim: ${c.id} (status=unsupported)`,
      verifier: VERIFIER_MODEL,
    };
  }
  if (c.superseded_by != null) {
    return {
      status: "unsupported",
      score: 0,
      source_passage: `depends on broken-chain claim: ${c.id} (superseded by ${c.superseded_by})`,
      verifier: VERIFIER_MODEL,
    };
  }

  const sourceBlock = `[Claim ${c.id}, ${c.claim_type}, ${c.status}]: ${c.claim_text}`;
  const prompt = INTERPRETATION_PROMPT_TEMPLATE.replace("{SOURCE_BLOCK}", sourceBlock).replace(
    "{CLAIM}",
    claim,
  );

  try {
    const { object } = await generateObject({
      model: getLanguageModel(VERIFIER_MODEL),
      schema: VerifySchema,
      prompt,
      temperature: 0.0,
    });
    return {
      status: object.supported ? "supported" : "unsupported",
      score: object.score,
      source_passage: object.reason,
      verifier: VERIFIER_MODEL,
    };
  } catch (e: any) {
    const transient = classifyError(e);
    if (transient) throw transient;
    return {
      status: "insufficient",
      score: 0,
      source_passage: `verifier error: ${e?.message || String(e)}`.slice(0, 300),
      verifier: `${VERIFIER_MODEL}-error`,
    };
  }
}

// ---------------------------------------------------------------------------
// Auto-quote selector (v0.2)
// ---------------------------------------------------------------------------

const AutoQuoteSchema = z.object({
  found: z.boolean(),
  quote: z.string(),
  reason: z.string().max(500),
});

const AUTOQUOTE_PROMPT_TEMPLATE = `You select supporting evidence for a CLAIM from a SOURCE DOCUMENT.

CLAIM: "{CLAIM}"

SOURCE:
---
{SOURCE}
---

Task: find the 1–3 sentence verbatim quote from SOURCE that most directly supports CLAIM. The quote MUST appear verbatim in the SOURCE — do not paraphrase, summarize, or invent. If no passage in SOURCE genuinely supports the claim, return { found: false, reason: "..." }.

Return JSON: { found, quote (verbatim string from source, or empty), reason }.`;

export async function autoSelectQuote(
  claim: string,
  dossierContent: string,
): Promise<{ quote: string; reason: string } | null> {
  const truncated = dossierContent.slice(0, MAX_SOURCE_CHARS);
  const prompt = AUTOQUOTE_PROMPT_TEMPLATE.replace("{CLAIM}", claim).replace(
    "{SOURCE}",
    truncated,
  );
  try {
    const { object } = await generateObject({
      model: getLanguageModel(VERIFIER_MODEL),
      schema: AutoQuoteSchema,
      prompt,
      temperature: 0.0,
    });
    if (!object.found) return null;
    const match = findQuoteInContent(object.quote, dossierContent);
    if (!match.found) return null;
    return { quote: object.quote, reason: object.reason };
  } catch (e: any) {
    const transient = classifyError(e);
    if (transient) throw transient;
    return null;
  }
}
