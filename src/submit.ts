/** High-level claim submission — the main entry point used by CLI commands.
 * Mirrors the v0.8 /claim FastAPI endpoint logic from fbc-poc.
 */
import * as store from "./store.ts";
import { embedOne } from "./embedder.ts";
import { verifyClaim } from "./verifier.ts";
import type { SubmitClaimRequest, ClaimType } from "./types.ts";

async function ensureDossierFromQuote(
  url: string,
  quote: string,
  title?: string,
  publication_date?: string,
  author_attribution?: string,
): Promise<string> {
  const cached = store.getDossierByUrl(url);
  if (cached) return cached.slug;

  // Embed the quote so dossiers are searchable in /search_kb
  let embedding: Float32Array | null = null;
  try {
    embedding = await embedOne(quote);
  } catch {
    // non-fatal
  }

  return store.writeDossier({
    source_url: url,
    source_type: "web-submitted",
    title: title || url.slice(0, 100),
    verbatim_content: quote,
    embedding,
    publication_date: publication_date ?? null,
    author_attribution: author_attribution ?? null,
  });
}

export async function submitClaim(req: SubmitClaimRequest): Promise<any> {
  if (!req.text?.trim()) return { error: "claim text is empty" };

  const ct = req.claim_type;

  if (ct === "ATOMIC" || ct === "QUOTATION") {
    if (!req.source_url || !req.source_quote) {
      return { error: `${ct} requires source_url + source_quote` };
    }
    const slug = await ensureDossierFromQuote(
      req.source_url,
      req.source_quote,
      req.source_title,
      req.publication_date,
      req.author_attribution,
    );
    const result = await verifyClaim(req.text, slug, {
      topic: req.topic,
      author: req.author,
      claim_type: ct,
    });
    return { ...result, dossier_slug: slug, source_url: req.source_url };
  }

  if (ct === "SYNTHESIS") {
    if (!req.sources || req.sources.length < 2) {
      return { error: "SYNTHESIS requires sources[] with ≥2 entries" };
    }
    const slugs = await Promise.all(
      req.sources.map((s) => ensureDossierFromQuote(s.url, s.quote, s.title)),
    );
    const results = await Promise.all(
      slugs.map((slug) =>
        verifyClaim(req.text, slug, {
          topic: req.topic,
          author: req.author,
          claim_type: "SYNTHESIS" as ClaimType,
        }),
      ),
    );
    const best = results.reduce((a, b) => (b.score > a.score ? b : a));
    return {
      ...best,
      source_slugs: slugs,
      per_source_scores: slugs.map((s, i) => ({
        slug: s,
        score: results[i]!.score,
        status: results[i]!.status,
      })),
    };
  }

  if (ct === "INFERENCE" || ct === "INTERPRETATION") {
    if (!req.depends_on_ids?.length) {
      return { error: `${ct} requires depends_on_ids[] with ≥1 ID` };
    }
    const missing = req.depends_on_ids.filter((id) => !store.getClaim(id));
    if (missing.length) {
      return { error: `depends_on_ids reference missing claim(s): ${missing.join(", ")}` };
    }
    let claimEmb: Float32Array | null = null;
    try {
      claimEmb = await embedOne(req.text);
    } catch {
      // non-fatal
    }
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

  if (ct === "HYPOTHESIS") {
    let claimEmb: Float32Array | null = null;
    try {
      claimEmb = await embedOne(req.text);
    } catch {
      // non-fatal
    }
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

  return { error: `unknown claim_type: ${ct}` };
}
