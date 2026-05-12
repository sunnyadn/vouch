#!/usr/bin/env bun
/** vouch — verified-claim KB CLI.
 *
 * No daemon, no HTTP. Each command opens SQLite directly and calls the
 * configured LiteLLM-style provider for verification + embedding.
 *
 * Output: JSON by default (parseable by Claude). --pretty for human inspection.
 */
import { Command } from "commander";
import { mkdirSync, appendFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";

import * as store from "./store.ts";
import { submitClaim } from "./submit.ts";
import { embedOne } from "./embedder.ts";
import { fetchAndStore } from "./fetch.ts";
import { attestAndStore } from "./attest.ts";
import { TransientVerifierError } from "./verifier.ts";
import { DEFAULT_GATE_MODEL, getFirstEventTimestamp, readStdinJson, runGateCli } from "./gate.ts";
import { runDoctor } from "./doctor.ts";
import {
  SEARCH_PROVIDERS,
  searchCitations,
  ddgSearch,
  isSearchProvider,
  WebSearchError,
  type CitationCandidate,
  type WebResult,
} from "./searchers.ts";
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

/** Parse `--since` values: relative (`1h`, `12h`, `1d`, `3d`, `1w`) or absolute
 *  (`2026-05-11`, `2026-05-11T05:00`). Returns an ISO string. Throws on
 *  unparseable input so the caller can fail loudly. */
function parseSince(input: string): string {
  const rel = input.match(/^(\d+)([hdw])$/i);
  if (rel) {
    const n = parseInt(rel[1]!, 10);
    const unit = rel[2]!.toLowerCase();
    const ms = unit === "h" ? n * 3600_000 : unit === "d" ? n * 86400_000 : n * 7 * 86400_000;
    return new Date(Date.now() - ms).toISOString();
  }
  const abs = new Date(input);
  if (Number.isNaN(abs.getTime())) {
    throw new Error(`unrecognized --since value "${input}". Use relative (1h, 12h, 1d, 3d, 1w) or absolute (YYYY-MM-DD, YYYY-MM-DDTHH:mm).`);
  }
  return abs.toISOString();
}

// ---------- fetch ----------
program
  .command("fetch <url>")
  .description(
    "Fetch a URL and persist it as a dossier (the trust-establishing step), " +
      "returning the readable content so this is a drop-in for a web-fetch " +
      "tool. Subsequent `vouch claim --dossier <slug>` cites it (quote " +
      "auto-selected from the dossier).",
  )
  .option("--fetcher <name>", "Force a specific fetcher (arxiv | generic)")
  .option("--force-refetch", "Skip 24h cache and re-fetch even if dossier exists")
  .option("--full", "Return the entire content (default: first ~8000 chars)")
  .option("--content-limit <n>", "Return the first N chars of content", (v) => parseInt(v, 10))
  .action(async (url: string, opts: any) => {
    try {
      const result = await fetchAndStore(url, {
        hint: opts.fetcher,
        forceRefetch: opts.forceRefetch,
        full: opts.full,
        contentLimit: opts.contentLimit,
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
      "via quote-in-dossier + NLI. Pass --claim to file a representative claim against " +
      "the new dossier in the same call (auto-quote, defaults to ATOMIC).",
  )
  .requiredOption("--slug <slug>", "stable slug; lowercase + dashes/underscores only")
  .option("--content <text>", "attested content (inline)")
  .option("--content-file <path>", "attested content (from file)")
  .requiredOption("--attribution <name>", "who attests (e.g., 'sunny')")
  .option("--date <YYYY-MM-DD>", "attestation date; defaults to today UTC")
  .option("--topic <topic>", "searchability tag")
  .option("--force-overwrite", "replace existing attestation at same slug")
  .option(
    "--claim <text>",
    "file a representative claim against the new dossier in the same call (auto-quotes from content)",
  )
  .option("--claim-type <type>", "claim type when --claim is set: ATOMIC (default) | QUOTATION", "ATOMIC")
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

    // Validate --claim-type up front so we don't attest and then bail. Only the
    // dossier-backed types make sense here — SYNTHESIS needs --sources,
    // INFERENCE/INTERPRETATION need --depends-on, HYPOTHESIS ignores dossiers.
    if (opts.claim) {
      const allowed = ["ATOMIC", "QUOTATION"];
      if (!allowed.includes(opts.claimType)) {
        fail(
          `invalid --claim-type "${opts.claimType}" for attest --claim. ` +
            `Use one of: ${allowed.join(", ")}. (SYNTHESIS/INFERENCE/INTERPRETATION/HYPOTHESIS ` +
            `don't fit the attest-and-claim shape — file them separately with \`vouch claim\`.)`,
        );
      }
    }

    let attestResult: Awaited<ReturnType<typeof attestAndStore>>;
    try {
      attestResult = await attestAndStore({
        slug: opts.slug,
        content,
        attribution: opts.attribution,
        date: opts.date,
        topic: opts.topic,
        forceOverwrite: opts.forceOverwrite,
      });
    } catch (e: any) {
      console.error(
        JSON.stringify({ error: e.message, reason: "attest-failed" }),
      );
      process.exit(1);
    }

    if (!opts.claim) {
      emit(attestResult);
      return;
    }

    // Attestation persisted; now file the representative claim against it.
    // Auto-quote (no --source-quote) is the default — most attest-and-claim
    // cases have the claim text already entailed by the content. If NLI says
    // unsupported, the response still carries the dossier_slug so the caller
    // can retry with reworded text without re-attesting.
    try {
      const claimResult = await submitClaim({
        text: opts.claim,
        claim_type: opts.claimType as ClaimType,
        topic: opts.topic,
        author: "claude-skill",
        dossier_slug: attestResult.dossier_slug,
      });
      // Flat-merge: attest's dossier_slug/source_url match the claim's
      // (same dossier), so no field collisions. Claim fields are the
      // additive payload (claim_id, status, score, ...).
      emit({ ...attestResult, ...claimResult });
    } catch (e: any) {
      if (e instanceof TransientVerifierError) {
        // Attestation succeeded, verifier was transiently unreachable. Carry
        // dossier_slug so the caller can retry the claim without re-attesting.
        emit({
          error: e.message,
          kind: e.kind,
          hint: e.hint,
          recorded: false,
          dossier_slug: attestResult.dossier_slug,
        });
        process.exit(2);
      }
      throw e;
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
  .option("--source-quote <quote>", "ATOMIC/QUOTATION: verbatim 1-3 sentence quote (must appear in the dossier). If omitted, vouch auto-selects the best supporting passage.")
  .option("--auto-quote", "[deprecated] no-op; auto-selection is now the default when --source-quote is omitted")
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
  .option("--author <name>", "filter by author (e.g. claude-skill, gate-harvest)")
  .option("--verification <kind>", "filter by verification tag (e.g. tag-harvest, nli-session)")
  .option("--depends-on <id>", "claims whose dependency graph includes this claim id")
  .option("--since <when>", "relative (1h, 12h, 1d, 3d, 1w) or absolute (2026-05-11, 2026-05-11T05:00)")
  .option("--newest-first", "order by verified_at DESC instead of id DESC")
  .option("--limit <n>", "default 50", (v) => parseInt(v, 10), 50)
  .action((opts: any) => {
    let sinceIso: string | undefined;
    if (opts.since) {
      try {
        sinceIso = parseSince(opts.since);
      } catch (e: any) {
        fail(e.message);
      }
    }
    emit(
      store.listClaims({
        topic: opts.topic,
        status: opts.status,
        dossier_slug: opts.dossierSlug,
        claim_type: opts.claimType,
        contains: opts.contains,
        author: opts.author,
        verification: opts.verification,
        depends_on_id: opts.dependsOn ? parseInt(opts.dependsOn, 10) : undefined,
        since: sinceIso,
        newestFirst: opts.newestFirst,
        limit: opts.limit,
      }),
    );
  });

// ---------- recent ----------
program
  .command("recent")
  .description("Recency-ordered view of claims with a summary header (convenience wrapper around list-claims --since --newest-first)")
  .option("--since <when>", "default 1d; relative (1h, 12h, 1d, 3d, 1w) or absolute")
  .option("--limit <n>", "default 20", (v) => parseInt(v, 10), 20)
  .action((opts: any) => {
    const sinceRaw = opts.since || "1d";
    let sinceIso: string;
    try {
      sinceIso = parseSince(sinceRaw);
    } catch (e: any) {
      fail(e.message);
    }
    const claims = store.listClaims({ since: sinceIso, newestFirst: true, limit: opts.limit });
    const summary = store.listRecentClaimsSummary(sinceIso);
    const authorParts = Object.entries(summary.authorBreakdown).map(([a, n]) => `${a}=${n}`);
    const header = `${summary.total} new (${summary.supported} supported / ${summary.unsupported} unsupported / ${summary.recorded} recorded-derived) across ${summary.dossiers} dossier(s) — author breakdown: ${authorParts.join(" ")}`;
    emit({ summary: { ...summary, header }, claims });
  });

// ---------- digest ----------
program
  .command("digest")
  .description(
    "Content-bearing summary of KB mutations in a time window (the session-level complement to the per-turn turn Δ line)",
  )
  .option("--since <when>", "relative (1h, 12h, 1d, 3d, 1w), absolute ISO, or 'session'")
  .option("--transcript-path <path>", "Transcript path (used with --since session)")
  .option("--transcript-stdin", "Read Stop-hook payload from stdin (used with --since session)")
  .option("--write [path]", "Append digest as Markdown to path (default ~/.vouch/sessions/<date>.md)")
  .action(async (opts: any) => {
    const gapHours = parseInt(process.env.VOUCH_SESSION_GAP_HOURS || "2", 10);
    let since: string;

    if (!opts.since) {
      since = store.getSessionStart(gapHours);
    } else if (opts.since === "session") {
      let transcriptPath: string | undefined = opts.transcriptPath;
      if (!transcriptPath && opts.transcriptStdin) {
        const payload = readStdinJson();
        if (typeof payload?.transcript_path === "string") {
          transcriptPath = payload.transcript_path;
        }
      }
      if (transcriptPath) {
        const ts = getFirstEventTimestamp(transcriptPath);
        if (ts) {
          since = ts;
        } else {
          since = store.getSessionStart(gapHours);
        }
      } else {
        since = store.getSessionStart(gapHours);
      }
    } else {
      try {
        since = parseSince(opts.since);
      } catch (e: any) {
        fail(e.message);
      }
    }

    const digest = store.getDigest(since);
    const isEmpty =
      digest.claims.length === 0 &&
      digest.derived_claims.length === 0 &&
      digest.supersedes.length === 0 &&
      digest.dossiers.length === 0 &&
      digest.dependencies.length === 0;

    let out: any = digest;
    if (isEmpty) {
      out = { ...digest, message: "(nothing entered the KB in this window)" };
    }

    if (opts.write !== undefined) {
      const md = store.formatDigestMarkdown(digest);
      const defaultPath = join(homedir(), ".vouch", "sessions", `${new Date().toISOString().slice(0, 10)}.md`);
      const outPath = typeof opts.write === "string" && opts.write ? opts.write : defaultPath;
      mkdirSync(dirname(outPath), { recursive: true });
      appendFileSync(outPath, md + "\n");
      emit({ ...out, written_to: outPath });
    } else {
      emit(out);
    }
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
  .description(
    "Find a source for a claim. KB-first: returns dossiers + claims you already " +
      "have (reuse them, don't re-fetch). If the KB has no strong match, falls " +
      "through to a web search — DuckDuckGo by default (general), or an academic " +
      "index with --provider (" + SEARCH_PROVIDERS.join(" | ") + "). " +
      "Pick a web result and `vouch fetch <url>` it, then `vouch claim`.",
  )
  .option("--top-k <n>", "KB hits to return", (v) => parseInt(v, 10), 10)
  .option(
    "--provider <name>",
    `web fallback provider: ddg (default, general) | ${SEARCH_PROVIDERS.join(" | ")} (academic)`,
  )
  .option("--limit <n>", "web candidates to return", (v) => parseInt(v, 10), 5)
  .option(
    "--kb-threshold <f>",
    "top-KB-hit similarity at/above which the KB is 'sufficient' (skip web)",
    parseFloat,
    0.7,
  )
  .option("--kb-only", "never web-search, even when the KB is thin")
  .option("--web-only", "skip the KB, go straight to web search")
  .action(async (query: string, opts: any) => {
    const provider: string = opts.provider || "ddg";
    if (provider !== "ddg" && !isSearchProvider(provider)) {
      fail(`unknown --provider "${provider}". Use: ddg, ${SEARCH_PROVIDERS.join(", ")}`);
    }

    let kb: ReturnType<typeof store.searchHybrid> = [];
    let kbSufficient = false;
    if (!opts.webOnly) {
      try {
        const qEmb = await embedOne(query);
        kb = store.searchHybrid(qEmb, opts.topK);
      } catch (e: any) {
        // embed failure shouldn't sink the whole search — fall through to web
        kb = [];
      }
      const top = kb[0];
      kbSufficient = !!top && typeof top.similarity === "number" && top.similarity >= opts.kbThreshold;
    }

    let web: (WebResult | CitationCandidate)[] | null = null;
    let webError: string | undefined;
    const doWeb = opts.webOnly || (!opts.kbOnly && !kbSufficient);
    if (doWeb) {
      try {
        web =
          provider === "ddg"
            ? await ddgSearch(query, opts.limit)
            : await searchCitations(provider, query, opts.limit);
      } catch (e: any) {
        web = null;
        webError =
          e instanceof WebSearchError ? e.message : e?.message || String(e);
      }
    }

    emit({
      query,
      kb_sufficient: kbSufficient,
      kb,
      web,
      web_provider: doWeb ? provider : null,
      ...(webError ? { web_error: webError } : {}),
      providers_available: ["ddg", ...SEARCH_PROVIDERS],
      ...(!doWeb && !kbSufficient
        ? {
            hint:
              "KB had no strong match and --kb-only suppressed web search. Drop --kb-only (auto-uses DuckDuckGo) or pass --provider <openalex|pubmed|arxiv|google-scholar> for scholarly sources, then `vouch fetch <url>` the right hit.",
          }
        : {}),
    });
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
  .option("--top-k <n>", "candidate claims fetched per proposition", (v) => parseInt(v, 10), 8)
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
      ...(result.verdict.harvest ? { harvest: result.verdict.harvest } : {}),
      ...(result.verdict.classifier_error
        ? { classifier_error: result.verdict.classifier_error }
        : {}),
    });
    process.exit(result.exitCode);
  });

// ---------- doctor ----------
program
  .command("doctor")
  .description(
    "Diagnose config, env vars, DB connectivity, and Claude Code Stop-hook installation. " +
      "Pure local checks — no API calls. Reports OK / WARN / FAIL per check with fix hints.",
  )
  .action(async () => {
    const report = runDoctor();
    emit(report);
    if (!report.ok) process.exit(1);
  });

await program.parseAsync(process.argv);
