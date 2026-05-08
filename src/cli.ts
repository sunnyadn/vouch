#!/usr/bin/env bun
/** vouch — verified-claim KB CLI.
 *
 * No daemon, no HTTP. Each command opens SQLite directly and calls the
 * configured LiteLLM-style provider for verification + embedding.
 *
 * Output: JSON by default (parseable by Claude). --pretty for human inspection.
 */
import { Command } from "commander";

import * as store from "./store.ts";
import { submitClaim } from "./submit.ts";
import { embedOne } from "./embedder.ts";
import type { ClaimType } from "./types.ts";

const program = new Command()
  .name("vouch")
  .description(
    "Verified-claim KB CLI — Fetch Before Claim (FBC) pattern. " +
      "Submit claims with sources, get NLI verification, build a queryable provenance graph.",
  )
  .version("0.1.0")
  .option("--pretty", "Pretty-print JSON output");

function emit(obj: unknown) {
  const pretty = program.opts().pretty;
  if (pretty) console.log(JSON.stringify(obj, null, 2));
  else console.log(JSON.stringify(obj));
}

function fail(msg: string, code = 1): never {
  console.error(`vouch: ${msg}`);
  process.exit(code);
}

// ---------- claim ----------
program
  .command("claim <text>")
  .description("Submit a claim with sources for NLI verification + storage")
  .requiredOption(
    "-t, --type <type>",
    "ATOMIC | SYNTHESIS | INFERENCE | INTERPRETATION | HYPOTHESIS | QUOTATION",
  )
  .option("--topic <topic>")
  .option("--attribution <attribution>", "e.g. 'Guo et al. 2024'")
  .option("--author <author>", "claude-skill | user-edit | etc", "claude-skill")
  .option("--source-url <url>", "ATOMIC/QUOTATION single source URL")
  .option("--source-quote <quote>", "Verbatim 1-3 sentence quote from the source")
  .option("--source-title <title>")
  .option("--publication-date <date>", "YYYY-MM-DD source publication date")
  .option("--author-attribution <name>", "Source author/org (e.g. 'Guo et al.')")
  .option(
    "--sources <json>",
    'SYNTHESIS multi-source JSON: \'[{"url":"...","quote":"..."}]\'',
  )
  .option("--depends-on <ids>", "INFERENCE/INTERPRETATION upstream IDs, comma-separated")
  .option("--soft-score <num>", "0..1 confidence for HYPOTHESIS-like claims", parseFloat)
  .action(async (text: string, opts: any) => {
    const type = opts.type as ClaimType;
    const validTypes = [
      "ATOMIC",
      "SYNTHESIS",
      "INFERENCE",
      "INTERPRETATION",
      "HYPOTHESIS",
      "QUOTATION",
    ];
    if (!validTypes.includes(type)) {
      fail(`invalid --type "${type}". Must be one of: ${validTypes.join(", ")}`);
    }
    const dependsOn: number[] | undefined = opts.dependsOn
      ? String(opts.dependsOn)
          .split(",")
          .map((s) => parseInt(s.trim(), 10))
          .filter((n) => !isNaN(n))
      : undefined;
    const sources = opts.sources ? JSON.parse(opts.sources) : undefined;
    const result = await submitClaim({
      text,
      claim_type: type,
      topic: opts.topic,
      attribution: opts.attribution,
      author: opts.author,
      source_url: opts.sourceUrl,
      source_quote: opts.sourceQuote,
      source_title: opts.sourceTitle,
      publication_date: opts.publicationDate,
      author_attribution: opts.authorAttribution,
      sources,
      depends_on_ids: dependsOn,
      soft_score: opts.softScore,
    });
    emit(result);
  });

// ---------- list-claims ----------
program
  .command("list-claims")
  .description("Browse claims with filters")
  .option("--topic <topic>")
  .option("--status <status>", "supported | unsupported | insufficient")
  .option("--dossier-slug <slug>")
  .option("--claim-type <type>")
  .option("--contains <substring>", "LIKE substring match on claim_text")
  .option("--limit <n>", "default 50", (v) => parseInt(v, 10), 50)
  .action((opts: any) => {
    emit(
      store.listClaims({
        topic: opts.topic,
        status: opts.status,
        dossier_slug: opts.dossierSlug,
        claim_type: opts.claimType,
        contains: opts.contains,
        limit: opts.limit,
      }),
    );
  });

// ---------- get-claim ----------
program
  .command("get-claim <id>")
  .description("Get full claim by ID")
  .action((id: string) => {
    const c = store.getClaim(parseInt(id, 10));
    if (!c) fail(`claim ${id} not found`);
    emit(c);
  });

// ---------- chain ----------
program
  .command("chain <id>")
  .description("Walk dependency DAG from a claim")
  .option("--max-depth <n>", "default 6", (v) => parseInt(v, 10), 6)
  .action((id: string, opts: any) => {
    emit(store.getClaimChain(parseInt(id, 10), opts.maxDepth));
  });

// ---------- list-topics ----------
program
  .command("list-topics")
  .description("Distinct topics with claim counts")
  .action(() => emit(store.listTopics()));

// ---------- supersede ----------
program
  .command("supersede <oldId> <newId>")
  .description("Mark old claim as superseded by new claim (audit trail preserved)")
  .requiredOption("-r, --reason <reason>")
  .action((oldId: string, newId: string, opts: any) => {
    const ok = store.supersedeClaim(
      parseInt(oldId, 10),
      parseInt(newId, 10),
      opts.reason,
    );
    emit({ ok });
  });

// ---------- list-dossiers ----------
program
  .command("list-dossiers")
  .description("List captured dossiers")
  .option("--source-type <type>")
  .option("--limit <n>", "default 50", (v) => parseInt(v, 10), 50)
  .action((opts: any) => {
    emit(store.listDossiers({ source_type: opts.sourceType, limit: opts.limit }));
  });

// ---------- get-dossier ----------
program
  .command("get-dossier <slug>")
  .description("Get dossier with content preview")
  .action((slug: string) => {
    const d = store.getDossier(slug);
    if (!d) fail(`dossier ${slug} not found`);
    const out = {
      slug: d!.slug,
      title: d!.title,
      source_url: d!.source_url,
      source_type: d!.source_type,
      capture_date: d!.capture_date,
      source_hash: d!.source_hash,
      publication_date: d!.publication_date,
      author_attribution: d!.author_attribution,
      content_preview: (d!.content || "").slice(0, 4000),
      content_total_chars: (d!.content || "").length,
    };
    emit(out);
  });

// ---------- search ----------
program
  .command("search <query>")
  .description("Hybrid search across claims + dossiers (cosine over embeddings)")
  .option("--top-k <n>", "default 10", (v) => parseInt(v, 10), 10)
  .action(async (query: string, opts: any) => {
    const qEmb = await embedOne(query);
    emit(store.searchHybrid(qEmb, opts.topK));
  });

// ---------- health ----------
program
  .command("health")
  .description("Verify config + DB + provider auth")
  .action(async () => {
    const out: any = {
      ok: true,
      db_path: process.env.VOUCH_DB_PATH || "~/.vouch/store.db",
      verifier_model: process.env.VOUCH_VERIFIER_MODEL || "vertex_ai/gemini-3.1-pro-preview",
      embedder_model: process.env.VOUCH_EMBEDDER_MODEL || "vertex_ai/text-embedding-005",
    };
    try {
      store.getDb().prepare("SELECT 1").get();
      out.db_ok = true;
    } catch (e: any) {
      out.db_ok = false;
      out.db_error = e.message;
      out.ok = false;
    }
    emit(out);
  });

await program.parseAsync(process.argv);
