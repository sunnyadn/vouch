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
import { fetchAndStore } from "./fetch.ts";
import { attestAndStore } from "./attest.ts";
import { TransientVerifierError } from "./verifier.ts";
import { DEFAULT_GATE_MODEL, readStdinJson, runGateCli } from "./gate.ts";
import type { ClaimType } from "./types.ts";

const program = new Command()
  .name("vouch")
  .description(
    "Verified-claim KB CLI — Fetch Before Claim (FBC) pattern. " +
      "Submit claims with sources, get NLI verification, build a queryable provenance graph.",
  )
  .version("0.3.0")
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

// ---------- fetch ----------
program
  .command("fetch <url>")
  .description(
    "Fetch a URL and persist it as a dossier (the trust-establishing step). " +
      "Subsequent `vouch claim` references the returned dossier_slug.",
  )
  .option("--fetcher <name>", "Force a specific fetcher (arxiv | generic)")
  .option("--force-refetch", "Skip 24h cache and re-fetch even if dossier exists")
  .action(async (url: string, opts: any) => {
    try {
      const result = await fetchAndStore(url, {
        hint: opts.fetcher,
        forceRefetch: opts.forceRefetch,
      });
      emit(result);
    } catch (e: any) {
      fail(`fetch failed: ${e?.message || String(e)}`);
    }
  });

// ---------- attest ----------
program
  .command("attest")
  .description(
    "Create a user-attested dossier. The user takes responsibility for the content; " +
      "vouch does not fetch or independently verify it. Downstream claims still verify " +
      "via quote-in-dossier + NLI.",
  )
  .requiredOption("--slug <slug>", "stable slug; lowercase + dashes/underscores only")
  .option("--content <text>", "attested content (inline)")
  .option("--content-file <path>", "attested content (from file)")
  .requiredOption("--attribution <name>", "who attests (e.g., 'sunny')")
  .option("--date <YYYY-MM-DD>", "attestation date; defaults to today UTC")
  .option("--topic <topic>", "searchability tag")
  .option("--force-overwrite", "replace existing attestation at same slug")
  .action(async (opts: any) => {
    let content = opts.content;
    if (!content && opts.contentFile) {
      content = await Bun.file(opts.contentFile).text();
    }
    if (!content) {
      console.error(
        JSON.stringify({
          error: "--content or --content-file required",
          reason: "missing-content",
        }),
      );
      process.exit(2);
    }
    try {
      const result = await attestAndStore({
        slug: opts.slug,
        content,
        attribution: opts.attribution,
        date: opts.date,
        topic: opts.topic,
        forceOverwrite: opts.forceOverwrite,
      });
      emit(result);
    } catch (e: any) {
      console.error(
        JSON.stringify({ error: e.message, reason: "attest-failed" }),
      );
      process.exit(1);
    }
  });

// ---------- claim ----------
program
  .command("claim <text>")
  .description(
    "Submit a claim. Strict mode: ATOMIC/QUOTATION/SYNTHESIS need a dossier from `vouch fetch`. " +
      "INFERENCE/INTERPRETATION need --depends-on. HYPOTHESIS is freeform.",
  )
  .requiredOption(
    "-t, --type <type>",
    "ATOMIC | SYNTHESIS | INFERENCE | INTERPRETATION | HYPOTHESIS | QUOTATION",
  )
  .option("--topic <topic>")
  .option("--attribution <attribution>", "Override the dossier-derived attribution, e.g. 'Guo et al. 2024'")
  .option("--author <author>", "claude-skill | user-edit | etc", "claude-skill")
  .option("--dossier <slug>", "ATOMIC/QUOTATION: slug from prior `vouch fetch`")
  .option("--source-quote <quote>", "ATOMIC/QUOTATION: verbatim 1-3 sentence quote (must appear in the dossier)")
  .option("--auto-quote", "ATOMIC only: vouch picks the best supporting passage from the dossier instead of requiring --source-quote")
  .option(
    "--sources <json>",
    'SYNTHESIS multi-source JSON: \'[{"dossier_slug":"...","quote":"..."}, ...]\' (≥2 entries; dossiers must be pre-fetched)',
  )
  .option("--depends-on <ids>", "INFERENCE/INTERPRETATION upstream claim IDs, comma-separated")
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
    try {
      const result = await submitClaim({
        text,
        claim_type: type,
        topic: opts.topic,
        attribution: opts.attribution,
        author: opts.author,
        dossier_slug: opts.dossier,
        source_quote: opts.sourceQuote,
        auto_quote: opts.autoQuote,
        sources,
        depends_on_ids: dependsOn,
        soft_score: opts.softScore,
      });
      emit(result);
    } catch (e: any) {
      if (e instanceof TransientVerifierError) {
        // Surface transient/system errors clearly. NOT recorded in the KB —
        // these don't carry information about the (claim, source) pair.
        emit({
          error: e.message,
          kind: e.kind,
          hint: e.hint,
          recorded: false,
        });
        process.exit(2);
      }
      throw e;
    }
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
  .description("Get dossier with content (default: 4000-char preview; --full for entire content)")
  .option("--full", "Return entire dossier content (may be very large for full papers)")
  .option("--offset <n>", "Start of content slice (with --limit)", (v) => parseInt(v, 10))
  .option("--limit <n>", "Length of content slice (with --offset)", (v) => parseInt(v, 10))
  .action((slug: string, opts: any) => {
    const d = store.getDossier(slug);
    if (!d) fail(`dossier ${slug} not found`);
    const fullContent = d!.content || "";
    let content: string;
    if (opts.full) content = fullContent;
    else if (typeof opts.offset === "number" || typeof opts.limit === "number") {
      const off = opts.offset || 0;
      const lim = opts.limit || 4000;
      content = fullContent.slice(off, off + lim);
    } else {
      content = fullContent.slice(0, 4000);
    }
    const out = {
      slug: d!.slug,
      title: d!.title,
      source_url: d!.source_url,
      source_type: d!.source_type,
      capture_date: d!.capture_date,
      source_hash: d!.source_hash,
      publication_date: d!.publication_date,
      author_attribution: d!.author_attribution,
      content,
      content_total_chars: fullContent.length,
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

// ---------- gate ----------
program
  .command("gate")
  .description(
    "Stop-hook gate: scan the last assistant message in a transcript for ungrounded " +
      "named-entity factual claims and block (exit 2) if any aren't entailed by a " +
      "supported vouch claim. Designed for Claude Code Stop hook integration.",
  )
  .option("--transcript-stdin", "Read Stop-hook payload JSON from stdin and derive transcript_path from it")
  .option("--transcript-path <path>", "Read the transcript directly from this path (overrides stdin)")
  .option("--draft <text>", "Use this text as the draft instead of reading a transcript")
  .option("--strict", "exit 2 on ungrounded claims (default)")
  .option("--advisory", "exit 0 + stderr warning only; never block")
  .option("--bypass-env <name>", "env var that disables the gate when set to '1'", "VOUCH_GATE_BYPASS")
  .option("--model <id>", "extractor model (LiteLLM-style)", DEFAULT_GATE_MODEL)
  .option("--top-k <n>", "candidate claims fetched per entity", (v) => parseInt(v, 10), 5)
  .action(async (opts: any) => {
    let transcriptPath: string | undefined = opts.transcriptPath;
    let hookPayload: any | undefined;
    if (!transcriptPath && opts.transcriptStdin) {
      hookPayload = readStdinJson();
    }
    const strict = !opts.advisory;
    const result = await runGateCli({
      transcriptPath,
      hookPayload,
      draft: opts.draft,
      model: opts.model,
      topK: opts.topK,
      strict,
      bypassEnv: opts.bypassEnv,
    });
    if (result.message) process.stderr.write(result.message);
    emit({
      blocked: result.verdict.blocked,
      pairs: result.verdict.pairs,
      ...(result.verdict.classifier_error
        ? { classifier_error: result.verdict.classifier_error }
        : {}),
    });
    process.exit(result.exitCode);
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
