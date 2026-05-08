import { generateObject } from "ai";
import { z } from "zod";

import { MAX_SOURCE_CHARS, VERIFIER_MODEL } from "./config.ts";
import { getLanguageModel } from "./providers.ts";
import { embedBatch } from "./embedder.ts";
import * as store from "./store.ts";
import type { ClaimType, VerifyResult } from "./types.ts";

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
  });

  return { ...result, claim_id: cid };
}
