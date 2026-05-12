/** vouch attest — create a user-attested dossier without HTTP fetch.
 *  The user takes responsibility for the content; downstream claims still go
 *  through quote-in-dossier + NLI checks normally.
 */
import { embedOne } from "./embedder.ts";
import * as store from "./store.ts";

interface AttestResult {
  dossier_slug: string;
  source_type: string;
  source_url: string;
  attribution: string;
  attestation_date: string;
  content_chars: number;
  stored_at: string;
}

export async function attestAndStore(opts: {
  slug: string;
  content: string;
  attribution: string;
  date?: string;
  topic?: string;
  forceOverwrite?: boolean;
  /** Override the default `attestation://<slug>` source_url. */
  source_url?: string;
  /** Override the default `user-statement` source_type. */
  source_type?: string;
  /** Override the default `attested` scope. */
  scope?: string;
}): Promise<AttestResult> {
  // 1. Validate slug format
  if (!/^[a-z0-9_-]+$/.test(opts.slug)) {
    throw new Error(
      `invalid slug format: must match /^[a-z0-9_-]+$/, got "${opts.slug}"`,
    );
  }
  const fullSlug = `evidence/attestations/${opts.slug}`;

  // 2. Check for existing attestation
  const existing = store.getDossier(fullSlug);
  if (existing && !opts.forceOverwrite) {
    throw new Error(
      `attestation already exists at ${fullSlug}. ` +
        `Use --force-overwrite to replace, or pick a different slug.`,
    );
  }

  // 3. Validate content
  if (!opts.content || opts.content.length === 0) {
    throw new Error(
      "--content (or --content-file) is required and must be non-empty",
    );
  }
  let content = opts.content;
  if (content.length > 200_000) {
    process.stderr.write(
      `warning: content exceeds 200KB, truncating from ${content.length} to 200000 chars\n`,
    );
    content = content.slice(0, 200_000);
  }

  // 4. Embed
  let embedding: Float32Array | null = null;
  try {
    embedding = await embedOne(content.slice(0, 8000));
  } catch {
    // non-fatal; dossier persists, just won't be searchable
  }

  // 5. Persist via store.writeDossier (same primitive as fetched)
  const date = opts.date || new Date().toISOString().slice(0, 10);
  const sourceUrl = opts.source_url ?? `attestation://${opts.slug}`;
  const sourceType = opts.source_type ?? "user-statement";
  const slug = store.writeDossier({
    source_url: sourceUrl,
    source_type: sourceType,
    title: opts.slug,
    verbatim_content: content,
    embedding,
    publication_date: date,
    author_attribution: opts.attribution,
    slug: fullSlug,
    scope: opts.scope ?? "attested",
  });

  return {
    dossier_slug: slug,
    source_type: sourceType,
    source_url: sourceUrl,
    attribution: opts.attribution,
    attestation_date: date,
    content_chars: content.length,
    stored_at: new Date().toISOString(),
  };
}
