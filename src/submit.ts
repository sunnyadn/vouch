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
import { embedOne } from "./embedder.ts";
import { TransientVerifierError, verifyClaim } from "./verifier.ts";
import { findQuoteInContent } from "./quote-match.ts";
import type { ClaimType, SubmitClaimRequest } from "./types.ts";

interface RejectedSubmission {
  error: string;
  reason: "missing-dossier" | "quote-not-in-dossier" | "missing-deps" | "missing-deps-claims" | "missing-source" | "bad-claim-type" | "empty-text";
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
  if (!req.source_quote) {
    return errOut(
      `${req.claim_type} requires --source-quote (the verbatim 1–3 sentences supporting the claim).`,
      "missing-source",
    );
  }

  const dossier = store.getDossier(req.dossier_slug);
  if (!dossier) {
    return errOut(`dossier not found: ${req.dossier_slug}`, "missing-dossier");
  }

  const match = findQuoteInContent(req.source_quote, dossier.content || "");
  if (!match.found) {
    return errOut(
      `quote not found in dossier "${req.dossier_slug}". The submitted quote must appear in the dossier content vouch fetched. Re-check the source page or fetch a different one.`,
      "quote-not-in-dossier",
      { quote_preview: req.source_quote.slice(0, 200), dossier_chars: (dossier.content || "").length },
    );
  }

  const result = await verifyClaim(req.text, req.dossier_slug, {
    topic: req.topic,
    author: req.author,
    claim_type: req.claim_type,
    attribution: req.attribution,
    source_quote: req.source_quote,
    source_offset_start: match.start,
    source_offset_end: match.end,
  });
  return {
    ...result,
    dossier_slug: req.dossier_slug,
    source_url: dossier.source_url,
    quote_match: match.matchType,
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
  const missing = req.depends_on_ids.filter((id) => !store.getClaim(id));
  if (missing.length) {
    return errOut(
      `depends_on_ids reference missing claim(s): ${missing.join(", ")}`,
      "missing-deps-claims",
    );
  }
  const claimEmb = await embedClaim(req.text);
  const cid = store.recordClaim({
    dossier_slug: "",
    claim_text: req.text,
    score: 1.0,
    status: "supported",
    claim_type: ct,
    topic: req.topic ?? null,
    author: req.author ?? null,
    attribution: req.attribution ?? null,
    soft_score: req.soft_score ?? null,
    depends_on_ids: req.depends_on_ids,
    dependency_type: "inference",
    embedding: claimEmb,
  });
  return { claim_id: cid, status: "supported", claim_type: ct };
}

// ---------------------------------------------------------------------------
// HYPOTHESIS
// ---------------------------------------------------------------------------

async function submitHypothesis(req: SubmitClaimRequest): Promise<any> {
  const claimEmb = await embedClaim(req.text);
  const cid = store.recordClaim({
    dossier_slug: "",
    claim_text: req.text,
    score: 1.0,
    status: "supported",
    claim_type: "HYPOTHESIS",
    topic: req.topic ?? null,
    author: req.author ?? null,
    attribution: req.attribution ?? null,
    soft_score: req.soft_score ?? null,
    depends_on_ids: req.depends_on_ids,
    dependency_type: "support",
    embedding: claimEmb,
  });
  return { claim_id: cid, status: "supported", claim_type: "HYPOTHESIS" };
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

function errOut(message: string, reason: RejectedSubmission["reason"], detail?: unknown) {
  return { error: message, reason, ...(detail ? { detail } : {}) };
}
