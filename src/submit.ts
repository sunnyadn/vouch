/** High-level claim submission — strict mode.
 *
 * ATOMIC / QUOTATION / SYNTHESIS claims must reference dossiers vouch
 * fetched itself (`vouch fetch <url>` first). The submitted quote is
 * verified against the dossier content (anti-fabrication primitive),
 * THEN handed to the verifier for NLI.
 *
 * INFERENCE / INTERPRETATION need only depends_on_ids — they're derivations
 * over already-stored claims, no source step.
 *
 * HYPOTHESIS is recorded as-is (explicitly unverified speculation).
 */
import * as store from "./store.ts";
import { embedBatch, embedOne } from "./embedder.ts";
import {
  TransientVerifierError,
  verifyClaim,
  verifyClaimsBatch,
  verifyInferenceClaim,
  verifyInterpretationClaim,
  autoSelectQuote,
} from "./verifier.ts";
import { findQuoteInContent } from "./quote-match.ts";
import type { ClaimType, SubmitClaimRequest } from "./types.ts";

interface RejectedSubmission {
  error: string;
  reason: "missing-dossier" | "quote-not-in-dossier" | "missing-deps" | "missing-deps-claims" | "missing-source" | "bad-deps" | "bad-claim-type" | "empty-text";
  detail?: unknown;
}

export async function submitClaim(req: SubmitClaimRequest): Promise<any> {
  if (!req.text?.trim()) return errOut("claim text is empty", "empty-text");

  const ct = req.claim_type;

  if (ct === "ATOMIC" || ct === "QUOTATION") {
    return submitAtomic(req);
  }
  if (ct === "SYNTHESIS") {
    return submitSynthesis(req);
  }
  if (ct === "INFERENCE" || ct === "INTERPRETATION") {
    return submitDerived(req, ct);
  }
  if (ct === "HYPOTHESIS") {
    return submitHypothesis(req);
  }
  return errOut(`unknown claim_type: ${ct}`, "bad-claim-type");
}

// ---------------------------------------------------------------------------
// ATOMIC / QUOTATION
// ---------------------------------------------------------------------------

async function submitAtomic(req: SubmitClaimRequest): Promise<any> {
  if (!req.dossier_slug) {
    return errOut(
      `${req.claim_type} requires --dossier <slug>. Run \`vouch fetch <url>\` first to ingest the source, then claim against the returned slug.`,
      "missing-dossier",
    );
  }

  const dossier = store.getDossier(req.dossier_slug);
  if (!dossier) {
    return errOut(`dossier not found: ${req.dossier_slug}`, "missing-dossier");
  }

  const content = dossier.content || "";

  let combinedQuote: string;
  let offsetStart: number | null;
  let offsetEnd: number | null;
  let matchType: "exact" | "normalized" | "fuzzy" | "none";
  let autoSelected = false;

  if (req.source_quote) {
    const match = findQuoteInContent(req.source_quote, content);
    if (!match.found) {
      return errOut(
        `quote not found in dossier "${req.dossier_slug}". The submitted quote must appear in the dossier content vouch fetched. Re-check the source page or fetch a different one.`,
        "quote-not-in-dossier",
        { quote_preview: req.source_quote.slice(0, 200), dossier_chars: content.length },
      );
    }
    combinedQuote = req.source_quote;
    offsetStart = match.start;
    offsetEnd = match.end;
    matchType = match.matchType;
  } else {
    // No --source-quote → auto-select. The --auto-quote flag is retained as a
    // no-op for backwards-compatible scripts. Friction reduction per #18: the
    // common case (claim text already entailed by the dossier) should not
    // require the caller to re-paste a contiguous slice of the dossier.
    const auto = await autoSelectQuote(req.text, content);
    if (!auto) {
      return errOut(
        `auto-quote: no supporting passage found in dossier "${req.dossier_slug}".`,
        "quote-not-in-dossier",
      );
    }
    const passageMatch = findQuoteInContent(auto.quote, content);
    if (!passageMatch.found) {
      // Defensive — autoSelectQuote already verified, but recompute for offsets.
      return errOut(
        `auto-quote: passage not found in dossier "${req.dossier_slug}".`,
        "quote-not-in-dossier",
      );
    }

    // SUN-58: sandwich picked passage with entity-establishing prefix so NLI
    // sees the dossier's subject, not just the bare data row. Verify the
    // prefix is verbatim from the dossier; drop it if not.
    let prefix = auto.prefix;
    if (prefix) {
      const prefixMatch = findQuoteInContent(prefix, content);
      if (!prefixMatch.found) prefix = "";
    }

    if (prefix && prefix.includes(auto.quote)) {
      // Picked passage is already inside the prefix (small dossier / GitHub
      // metadata block). Use prefix alone — no need to duplicate.
      const m = findQuoteInContent(prefix, content);
      combinedQuote = prefix;
      offsetStart = m.start;
      offsetEnd = m.end;
      matchType = m.matchType;
    } else if (prefix) {
      // Two non-contiguous pieces from the dossier — combined quote is not a
      // single substring, so offsets can't represent it. The full text lives
      // in source_quote; gate.ts falls back to that when offsets are null.
      combinedQuote = `${prefix}\n\n${auto.quote}`;
      offsetStart = null;
      offsetEnd = null;
      matchType = passageMatch.matchType;
    } else {
      combinedQuote = auto.quote;
      offsetStart = passageMatch.start;
      offsetEnd = passageMatch.end;
      matchType = passageMatch.matchType;
    }
    autoSelected = true;
  }

  const result = await verifyClaim(req.text, req.dossier_slug, {
    topic: req.topic,
    author: req.author,
    claim_type: req.claim_type,
    attribution: req.attribution,
    source_quote: combinedQuote,
    source_offset_start: offsetStart,
    source_offset_end: offsetEnd,
  });
  return {
    ...result,
    dossier_slug: req.dossier_slug,
    source_url: dossier.source_url,
    quote_match: matchType,
    ...(autoSelected ? { metadata: { auto_selected_quote: true } } : {}),
  };
}

// ---------------------------------------------------------------------------
// SYNTHESIS
// ---------------------------------------------------------------------------

async function submitSynthesis(req: SubmitClaimRequest): Promise<any> {
  // SYNTHESIS uses an array of {dossier_slug, quote} pairs in --sources
  // (not URLs anymore — must be already-fetched dossiers).
  if (!req.sources || req.sources.length < 2) {
    return errOut(
      "SYNTHESIS requires --sources [{\"dossier_slug\":\"...\",\"quote\":\"...\"}, ...] with ≥2 entries (dossiers must be pre-fetched via `vouch fetch`).",
      "missing-source",
    );
  }
  const checks = [];
  for (const s of req.sources) {
    if (!s.dossier_slug) {
      return errOut(
        "SYNTHESIS source missing dossier_slug. Each source must reference a pre-fetched dossier.",
        "missing-dossier",
      );
    }
    const d = store.getDossier(s.dossier_slug);
    if (!d) return errOut(`dossier not found: ${s.dossier_slug}`, "missing-dossier");
    const m = findQuoteInContent(s.quote, d.content || "");
    if (!m.found) {
      return errOut(
        `quote not found in dossier "${s.dossier_slug}".`,
        "quote-not-in-dossier",
        { dossier: s.dossier_slug, quote_preview: s.quote.slice(0, 200) },
      );
    }
    checks.push({ slug: s.dossier_slug, quote: s.quote, match: m });
  }

  const results = await Promise.all(
    checks.map((c) =>
      verifyClaim(req.text, c.slug, {
        topic: req.topic,
        author: req.author,
        claim_type: "SYNTHESIS" as ClaimType,
        attribution: req.attribution,
        source_quote: c.quote,
        source_offset_start: c.match.start,
        source_offset_end: c.match.end,
      }),
    ),
  );
  const best = results.reduce((a, b) => (b.score > a.score ? b : a));
  return {
    ...best,
    source_slugs: checks.map((c) => c.slug),
    per_source_scores: checks.map((c, i) => ({
      slug: c.slug,
      score: results[i]!.score,
      status: results[i]!.status,
    })),
  };
}

// ---------------------------------------------------------------------------
// INFERENCE / INTERPRETATION
// ---------------------------------------------------------------------------

async function submitDerived(req: SubmitClaimRequest, ct: "INFERENCE" | "INTERPRETATION"): Promise<any> {
  if (!req.depends_on_ids?.length) {
    return errOut(`${ct} requires --depends-on <ids> (≥1 upstream claim).`, "missing-deps");
  }

  if (ct === "INTERPRETATION" && req.depends_on_ids.length !== 1) {
    return errOut(
      `INTERPRETATION requires exactly one upstream claim, got ${req.depends_on_ids.length}`,
      "bad-deps",
    );
  }

  const missing = req.depends_on_ids.filter((id) => !store.getClaim(id));
  if (missing.length) {
    return errOut(
      `depends_on_ids reference missing claim(s): ${missing.join(", ")}`,
      "missing-deps-claims",
    );
  }

  const claimEmb = await embedClaim(req.text);

  if (ct === "INFERENCE") {
    const verdict = await verifyInferenceClaim(req.text, req.depends_on_ids);
    const cid = store.recordClaim({
      dossier_slug: "",
      claim_text: req.text,
      score: verdict.score,
      status: verdict.status,
      source_passage: verdict.source_passage,
      claim_type: ct,
      topic: req.topic ?? null,
      author: req.author ?? null,
      attribution: req.attribution ?? null,
      soft_score: req.soft_score ?? null,
      depends_on_ids: req.depends_on_ids,
      dependency_type: "inference",
      embedding: claimEmb,
      verification: "nli-entailment",
    });
    return { claim_id: cid, status: verdict.status, claim_type: ct, source_passage: verdict.source_passage, verification: "nli-entailment" };
  }

  // INTERPRETATION
  const verdict = await verifyInterpretationClaim(req.text, req.depends_on_ids);
  const cid = store.recordClaim({
    dossier_slug: "",
    claim_text: req.text,
    score: verdict.score,
    status: verdict.status,
    source_passage: verdict.source_passage,
    claim_type: ct,
    topic: req.topic ?? null,
    author: req.author ?? null,
    attribution: req.attribution ?? null,
    soft_score: req.soft_score ?? null,
    depends_on_ids: req.depends_on_ids,
    dependency_type: "inference",
    embedding: claimEmb,
    verification: "nli-reframing",
  });
  return { claim_id: cid, status: verdict.status, claim_type: ct, source_passage: verdict.source_passage, verification: "nli-reframing" };
}

// ---------------------------------------------------------------------------
// HYPOTHESIS
// ---------------------------------------------------------------------------

async function submitHypothesis(req: SubmitClaimRequest): Promise<any> {
  const claimEmb = await embedClaim(req.text);
  const cid = store.recordClaim({
    dossier_slug: "",
    claim_text: req.text,
    score: null,
    status: "recorded",
    claim_type: "HYPOTHESIS",
    topic: req.topic ?? null,
    author: req.author ?? null,
    attribution: req.attribution ?? null,
    soft_score: req.soft_score ?? null,
    depends_on_ids: req.depends_on_ids,
    dependency_type: "support",
    embedding: claimEmb,
    verification: "none",
  });
  return { claim_id: cid, status: "recorded", claim_type: "HYPOTHESIS", nli_score: null, verification: "none", soft_score: req.soft_score ?? null };
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

async function embedClaim(text: string): Promise<Float32Array | null> {
  try {
    return await embedOne(text);
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------------------
// Batch ATOMIC submission
// ---------------------------------------------------------------------------

export interface BatchSubmissionItem {
  text: string;
  dossier_slug: string;
  source_quote: string;
  topic?: string;
  attribution?: string;
  author?: string;
}

export async function submitClaimBatch(items: BatchSubmissionItem[]): Promise<any[]> {
  if (items.length === 0) return [];

  // Step 1: validate all quotes (fail fast)
  const validated: Array<{
    text: string;
    dossier_slug: string;
    source_passage: string;
    topic?: string;
    attribution?: string;
    author?: string;
    offsetStart: number | null;
    offsetEnd: number | null;
    matchType: "exact" | "normalized" | "fuzzy";
    dossier: ReturnType<typeof store.getDossier>;
  }> = [];

  for (let i = 0; i < items.length; i++) {
    const item = items[i]!;
    const dossier = store.getDossier(item.dossier_slug);
    if (!dossier) {
      throw new Error(`line ${i + 1}: dossier not found: ${item.dossier_slug}`);
    }
    const match = findQuoteInContent(item.source_quote, dossier.content || "");
    if (!match.found) {
      throw new Error(
        `line ${i + 1}: quote not found in dossier "${item.dossier_slug}"`,
      );
    }
    validated.push({
      text: item.text,
      dossier_slug: item.dossier_slug,
      source_passage: item.source_quote,
      topic: item.topic,
      attribution: item.attribution,
      author: item.author,
      offsetStart: match.start,
      offsetEnd: match.end,
      matchType: match.matchType as "exact" | "normalized" | "fuzzy",
      dossier,
    });
  }

  // Step 2: batch NLI (all-or-nothing — throws on any LLM error)
  const batchItems = validated.map((v) => ({
    claim_text: v.text,
    source_passage: v.source_passage,
  }));

  const results = await verifyClaimsBatch(batchItems);

  // Step 3: embed all claim texts in one call
  let allEmbeddings: Float32Array[] = [];
  try {
    allEmbeddings = await embedBatch(validated.map((v) => v.text));
  } catch {
    // Embedding failure is non-fatal — claims still record, just won't be searchable
  }

  // Step 4: persist each claim
  const out: any[] = [];
  for (let i = 0; i < validated.length; i++) {
    const v = validated[i]!;
    const r = results[i]!;
    const attribution = v.attribution || v.dossier?.author_attribution || null;
    const claimEmbedding = allEmbeddings[i] || null;

    const cid = store.recordClaim({
      dossier_slug: v.dossier_slug,
      claim_text: v.text,
      score: r.score,
      status: r.status,
      source_passage: r.source_passage,
      claim_type: "ATOMIC",
      topic: v.topic ?? null,
      author: v.author ?? null,
      attribution,
      source_offset_start: v.offsetStart,
      source_offset_end: v.offsetEnd,
      embedding: claimEmbedding,
      verification: "nli-quote",
    });

    out.push({
      ...r,
      claim_id: cid,
      dossier_slug: v.dossier_slug,
      quote_match: v.matchType,
    });
  }

  return out;
}

function errOut(message: string, reason: RejectedSubmission["reason"], detail?: unknown) {
  return { error: message, reason, ...(detail ? { detail } : {}) };
}
