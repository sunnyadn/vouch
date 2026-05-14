#!/usr/bin/env bun
/** vouch — verified-claim KB CLI.
 *
 * No daemon, no HTTP. Each command opens SQLite directly and calls the
 * configured LiteLLM-style provider for verification + embedding.
 *
 * Output: human-readable by default. --json for machine-parseable output.
 * (vouch gate always prints prose for the Stop hook.)
 */
import { Command } from "commander";
import { mkdirSync, appendFileSync, readdirSync, statSync } from "node:fs";
import type { Stats } from "node:fs";
import { homedir } from "node:os";
import { basename, dirname, join } from "node:path";

import * as store from "./store.ts";
import { submitClaim, submitClaimBatch } from "./submit.ts";
import { embedOne } from "./embedder.ts";
import { fetchAndStore } from "./fetch.ts";
import { attestAndStore } from "./attest.ts";
import { TransientVerifierError } from "./verifier.ts";
import { DEFAULT_GATE_MODEL, findSessionSourceByToolUseId, getFirstEventTimestamp, readStdinJson, runGateCli } from "./gate.ts";
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
  .option("--json", "Emit machine-parseable JSON output");

function emit(obj: unknown, humanRenderer?: (obj: any) => string) {
  const jsonMode = program.opts().json || process.env.VOUCH_OUTPUT === "json";
  if (jsonMode) {
    console.log(JSON.stringify(obj));
  } else if (humanRenderer) {
    console.log(humanRenderer(obj));
  } else {
    console.log(JSON.stringify(obj, null, 2));
  }
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

// ---------------------------------------------------------------------------
// Human-render helpers
// ---------------------------------------------------------------------------

function truncate(s: string | null | undefined, n: number): string {
  if (!s || s.length <= n) return s || "";
  return s.slice(0, n - 1) + "…";
}

function renderClaimsTable(claims: any[]): string {
  if (!claims.length) return "(no claims)";
  const wId = Math.max(3, ...claims.map((c) => String(c.id).length));
  const wType = Math.max(4, ...claims.map((c) => (c.claim_type || "—").length));
  const wStatus = Math.max(6, ...claims.map((c) => (c.status || "—").length));
  const wAuthor = Math.max(6, ...claims.map((c) => (c.author || "—").length));
  const wDate = 20;
  const header = `${"#".padStart(wId)}  ${"TYPE".padEnd(wType)}  ${"STATUS".padEnd(wStatus)}  SCORE  ${"AUTHOR".padEnd(wAuthor)}  ${"VERIFIED_AT".padEnd(wDate)}  CLAIM`;
  const lines = claims.map((c) => {
    const id = String(c.id).padStart(wId);
    const type = (c.claim_type || "—").padEnd(wType);
    const status = (c.status || "—").padEnd(wStatus);
    const score = c.nli_score != null ? c.nli_score.toFixed(2).padStart(5) : "  —  ";
    const author = (c.author || "—").padEnd(wAuthor);
    const date = (c.verified_at || "—").slice(0, 20).padEnd(wDate);
    const text = truncate(c.claim_text, 70);
    return `${id}  ${type}  ${status}  ${score}  ${author}  ${date}  ${text}`;
  });
  return [header, ...lines].join("\n");
}

function renderRecent(obj: any): string {
  if (!obj.claims?.length) return obj.summary?.header || "(no recent claims)";
  return `${obj.summary.header}\n${renderClaimsTable(obj.claims)}`;
}

function renderListClaims(claims: any[]): string {
  if (!claims.length) return "(no claims match the filters)";
  return renderClaimsTable(claims);
}

function renderListDossiers(dossiers: any[]): string {
  if (!dossiers.length) return "(no dossiers)";
  const wSlug = Math.max(4, ...dossiers.map((d) => d.slug.length));
  const wType = Math.max(11, ...dossiers.map((d) => (d.source_type || "—").length));
  const header = `${"SLUG".padEnd(wSlug)}  ${"SOURCE_TYPE".padEnd(wType)}  CAPTURE_DATE  CHARS`;
  const lines = dossiers.map((d) => {
    const slug = d.slug.padEnd(wSlug);
    const type = (d.source_type || "—").padEnd(wType);
    const date = (d.capture_date || "—").slice(0, 10);
    const chars = String(d.content_len ?? d.content_chars ?? d.content_total_chars ?? 0);
    return `${slug}  ${type}  ${date}  ${chars}`;
  });
  return [header, ...lines].join("\n");
}

function renderListTopics(topics: any[]): string {
  if (!topics.length) return "(no topics)";
  const sorted = [...topics].sort((a, b) => b.n_claims - a.n_claims);
  return sorted.map((t) => `${t.topic}  (${t.n_claims} claim${t.n_claims === 1 ? "" : "s"})`).join("\n");
}

function renderGetClaim(c: any): string {
  const lines: string[] = [];
  const nli = c.nli_score != null ? `NLI ${c.nli_score.toFixed(2)}` : "NLI —";
  const verif = c.verification ? `, ${c.verification}` : "";
  lines.push(`#${c.id}  ${c.claim_type || "UNKNOWN"}  ${c.status}${c.nli_score != null ? ` (${nli}${verif})` : ""}`);
  lines.push(`  "${c.claim_text}"`);
  lines.push(`  author:   ${c.author || "—"}        verified: ${c.verified_at || "—"}`);
  if (c.dossier_slug) {
    lines.push(`  dossier:  ${c.dossier_slug}`);
  }
  if (c.source_passage) {
    lines.push(`  quote:    "${truncate(c.source_passage, 200)}"`);
  }
  if (c.depends_on?.length) {
    const deps = c.depends_on.map((d: any) => {
      const depClaim = store.getClaim(d.depends_on_id);
      const depText = depClaim ? `("${truncate(depClaim.claim_text, 60)}")` : "";
      return `#${d.depends_on_id} ${depText}`;
    });
    lines.push(`  depends on: ${deps.join(", ")}`);
  }
  if (c.superseded_by) {
    lines.push(`  superseded by: #${c.superseded_by}  ("${truncate(c.supersede_reason || "no reason given", 60)}")`);
  }
  return lines.join("\n");
}

function renderGetDossier(d: any): string {
  const lines: string[] = [];
  lines.push(`${d.slug}`);
  lines.push(`  source:   ${d.source_url || "—"}`);
  lines.push(`  type:     ${d.source_type || "—"}`);
  lines.push(`  captured: ${d.capture_date || "—"}`);
  lines.push(`  chars:    ${d.content_total_chars}`);
  const previewLen = 600;
  const content = d.content || "";
  if (content.length > previewLen) {
    lines.push(`\n${content.slice(0, previewLen)}… [+${content.length - previewLen} more chars — vouch get-dossier ${d.slug} --full]`);
  } else {
    lines.push(`\n${content}`);
  }
  const citing = store.listClaims({ dossier_slug: d.slug, limit: 20 });
  if (citing.length) {
    lines.push(`\nCited by ${citing.length} claim(s):`);
    for (const c of citing) {
      lines.push(`  #${c.id}  ${truncate(c.claim_text, 70)}`);
    }
  }
  return lines.join("\n");
}

function renderChain(obj: any): string {
  const { root, nodes, edges } = obj;
  if (!nodes[root]) return `(claim #${root} not found)`;
  const adj = new Map<number, number[]>();
  for (const e of edges) {
    if (!adj.has(e.from)) adj.set(e.from, []);
    adj.get(e.from)!.push(e.to);
  }
  const lines: string[] = [];
  const visited = new Set<number>();
  function dfs(id: number, prefix: string) {
    if (visited.has(id)) {
      lines.push(`${prefix}#${id}  (already shown above)`);
      return;
    }
    visited.add(id);
    const node = nodes[id];
    const type = node?.claim_type || "UNKNOWN";
    const text = truncate(node?.claim_text || "", 70);
    const dossier = node?.dossier_slug ? `  (dossier ${node.dossier_slug})` : "";
    lines.push(`${prefix}#${id}  ${type}  "${text}"${dossier}`);
    const children = adj.get(id) || [];
    for (let i = 0; i < children.length; i++) {
      const isLast = i === children.length - 1;
      const nextPrefix = prefix + (isLast ? "   " : "│  ");
      dfs(children[i]!, nextPrefix);
    }
  }
  dfs(root, "");
  return lines.join("\n");
}

function renderSearch(obj: any): string {
  const lines: string[] = [];
  lines.push(`query: ${obj.query}`);
  lines.push(`kb_sufficient: ${obj.kb_sufficient ? "yes" : "no"}`);
  if (obj.web_provider) lines.push(`web_provider: ${obj.web_provider}`);
  let rank = 1;
  for (const hit of obj.kb || []) {
    const id = hit.kind === "claim" ? `#${hit.id}` : hit.slug;
    const text = truncate(hit.text, 120);
    lines.push(`${rank}. [kb]  ${id}  — ${text}`);
    rank++;
  }
  for (const w of obj.web || []) {
    const title = (w as any).title || "—";
    const snippet = truncate((w as any).snippet || "", 120);
    const url = (w as any).url || "";
    lines.push(`${rank}. [web]  ${title}  — ${snippet}  (${url})`);
    rank++;
  }
  if (obj.web_error) lines.push(`web error: ${obj.web_error}`);
  if (obj.hint) lines.push(`hint: ${obj.hint}`);
  return lines.join("\n");
}

function renderClaimResult(r: any): string {
  if (r.error) {
    return `✗ claim NOT recorded — ${r.reason}: ${r.error}`;
  }
  if (r.claim_id) {
    const type = r.claim_type || "ATOMIC";
    const score = r.nli_score != null ? r.nli_score.toFixed(2) : r.score != null ? r.score.toFixed(2) : "—";
    const verifier = r.verification || r.verifier || "—";
    const dossier = r.dossier_slug ? `, dossier ${r.dossier_slug}` : "";
    const quote = r.source_passage ? `, quote "${truncate(r.source_passage, 80)}"` : "";
    return `✓ claim #${r.claim_id} recorded — ${type}, NLI ${score} (${verifier})${dossier}${quote}`;
  }
  return JSON.stringify(r, null, 2);
}

function renderClaimBatch(obj: any): string {
  if (obj.error) {
    return `✗ batch NOT recorded — ${obj.error}`;
  }
  const lines: string[] = [];
  for (let i = 0; i < obj.results.length; i++) {
    const r = obj.results[i];
    const status = r.supported ? "supported" : "unsupported";
    const score = r.score != null ? r.score.toFixed(2) : "—";
    lines.push(`${i} → claim #${r.claim_id} — ${status} (${score})`);
  }
  lines.push(`${obj.results.length} claim${obj.results.length === 1 ? "" : "s"} recorded`);
  return lines.join("\n");
}

function renderAttestResult(r: any): string {
  return `✓ attested ${r.dossier_slug} (${r.content_chars} chars) — ${r.attribution}`;
}

function renderAttestWithClaim(r: any): string {
  const attestLine = `✓ attested ${r.dossier_slug} (${r.content_chars} chars) — ${r.attribution}`;
  const claimLine = renderClaimResult(r);
  return `${attestLine}\n${claimLine}`;
}

function renderFetchResult(r: any): string {
  const cached = r.cached ? " [cached]" : "";
  return `✓ fetched ${r.source_url} → dossier ${r.dossier_slug} (${r.content_chars} chars)${cached}`;
}

function renderSupersede(oldId: number, newId: number, reason: string): (r: any) => string {
  return () => `✓ #${oldId} superseded by #${newId} — "${truncate(reason, 80)}"`;
}

function renderDoctor(report: any): string {
  const lines = report.checks.map((c: any) => {
    const status = c.status.toUpperCase();
    const fix = c.fix ? `  → ${c.fix}` : "";
    return `[${status}] ${c.name}: ${c.detail}${fix}`;
  });
  if (!report.ok) lines.push("\nSome checks failed. Fix the FAIL items above.");
  return lines.join("\n");
}

function renderDigest(obj: any): string {
  const md = store.formatDigestMarkdown(obj);
  if (obj.written_to) {
    return `✓ digest appended to ${obj.written_to}\n\n${md}`;
  }
  return md;
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
      emit(result, renderFetchResult);
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
      "the new dossier in the same call (auto-quote, defaults to ATOMIC). " +
      "Use --from-session-tool to record a session tool_result as a dossier.",
  )
  .option("--slug <slug>", "stable slug; lowercase + dashes/underscores only (auto-derived with --from-session-tool)")
  .option("--content <text>", "attested content (inline)")
  .option("--content-file <path>", "attested content (from file)")
  .option("--attribution <name>", "who attests (a name or handle); defaults to session-<tool> with --from-session-tool")
  .option("--date <YYYY-MM-DD>", "attestation date; defaults to today UTC")
  .option("--topic <topic>", "searchability tag")
  .option("--force-overwrite", "replace existing attestation at same slug")
  .option("--from-session-tool <id>", "use a session tool_result as the attested content (requires --session-context or --transcript-stdin)")
  .option("--transcript-stdin", "read Stop-hook payload JSON from stdin to derive transcript path (used with --from-session-tool)")
  .option("--session-context <path>", "Claude Code transcript JSONL to read tool_results from (used with --from-session-tool)")
  .option("--stance <stance>", "observation | inference | hypothesis | placeholder (default observation; only with --from-session-tool)", "observation")
  .option("--depends-on <ids>", "upstream claim IDs for --stance inference, comma-separated")
  .option(
    "--claim <text>",
    "file a representative claim against the new dossier in the same call (auto-quotes from content)",
  )
  .option("--claim-type <type>", "claim type when --claim is set: ATOMIC (default) | QUOTATION", "ATOMIC")
  .action(async (opts: any) => {
    let content: string | undefined;
    let slug: string;
    let attribution: string;
    let sourceUrl: string | undefined;
    let sourceType: string | undefined;
    let scope: string | undefined;

    if (opts.fromSessionTool) {
      let transcriptPath: string | undefined = opts.sessionContext;
      if (!transcriptPath && opts.transcriptStdin) {
        const payload = readStdinJson();
        if (typeof payload?.transcript_path === "string") {
          transcriptPath = payload.transcript_path;
        }
      }
      if (!transcriptPath) {
        fail("--from-session-tool requires --session-context or --transcript-stdin");
      }

      const src = findSessionSourceByToolUseId(transcriptPath, opts.fromSessionTool);
      if (!src) {
        fail(`tool_result with tool_use_id "${opts.fromSessionTool}" not found in transcript`);
      }

      content = src.content;
      const toolLower = src.tool.toLowerCase();
      sourceUrl = src.uri;
      sourceType = `session-${toolLower}`;
      scope = src.tool === "Read" || src.tool === "Bash" ? "workspace" : "third-party";
      attribution = opts.attribution || `session-${toolLower}`;

      if (!opts.slug) {
        const sanitized = src.uri
          .toLowerCase()
          .replace(/^https?:\/\//, "")
          .replace(/[^a-z0-9]+/g, "-")
          .replace(/^-+|-+$/g, "")
          .slice(0, 50);
        const date = new Date().toISOString().slice(0, 10);
        slug = `session-${toolLower}-${sanitized}-${date}`;
      } else {
        slug = opts.slug;
      }
    } else {
      if (opts.stance && opts.stance !== "observation") {
        process.stderr.write("warning: --stance is only meaningful with --from-session-tool; ignoring\n");
      }
      content = opts.content;
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
      if (!opts.slug) {
        fail("--slug is required");
      }
      if (!opts.attribution) {
        fail("--attribution is required");
      }
      slug = opts.slug;
      attribution = opts.attribution;
    }

    let claimType: ClaimType | undefined;
    if (opts.claim) {
      if (opts.fromSessionTool) {
        if (opts.stance === "placeholder") {
          console.error(
            JSON.stringify({
              error: "a placeholder / TODO isn't a claim — file it once it's measured",
              reason: "placeholder-refused",
            }),
          );
          process.exit(1);
        }
        if (opts.stance === "qualified-inference") {
          console.error(
            JSON.stringify({
              error: "not yet supported — use --stance observation then `vouch claim --type INTERPRETATION --depends-on <id>` separately",
              reason: "qualified-inference-deferred",
            }),
          );
          process.exit(1);
        }
        const stanceMap: Record<string, ClaimType> = {
          observation: "ATOMIC",
          inference: "INFERENCE",
          hypothesis: "HYPOTHESIS",
        };
        claimType = stanceMap[opts.stance];
        if (!claimType) {
          fail(`invalid --stance "${opts.stance}"`);
        }
        if (claimType === "INFERENCE" && !opts.dependsOn) {
          fail("--stance inference requires --depends-on");
        }
        if (opts.claimType && opts.claimType !== claimType) {
          process.stderr.write(
            `warning: --claim-type (${opts.claimType}) conflicts with --stance (${opts.stance}); --stance wins\n`,
          );
        }
      } else {
        const allowed = ["ATOMIC", "QUOTATION"];
        if (!allowed.includes(opts.claimType)) {
          fail(
            `invalid --claim-type "${opts.claimType}" for attest --claim. ` +
              `Use one of: ${allowed.join(", ")}. (SYNTHESIS/INFERENCE/INTERPRETATION/HYPOTHESIS ` +
              `don't fit the attest-and-claim shape — file them separately with \`vouch claim\`.)`,
          );
        }
        claimType = opts.claimType;
      }
    } else if (opts.fromSessionTool && opts.stance !== "observation") {
      process.stderr.write("note: --stance is ignored when --claim is not set\n");
    }

    let attestResult: Awaited<ReturnType<typeof attestAndStore>>;
    try {
      attestResult = await attestAndStore({
        slug,
        content: content!,
        attribution,
        date: opts.date,
        topic: opts.topic,
        forceOverwrite: opts.forceOverwrite,
        source_url: sourceUrl,
        source_type: sourceType,
        scope,
      });
    } catch (e: any) {
      console.error(
        JSON.stringify({ error: e.message, reason: "attest-failed" }),
      );
      process.exit(1);
    }

    if (!opts.claim) {
      emit(attestResult, renderAttestResult);
      return;
    }

    try {
      const dependsOn: number[] | undefined = opts.dependsOn
        ? String(opts.dependsOn)
            .split(",")
            .map((s: string) => parseInt(s.trim(), 10))
            .filter((n: number) => !isNaN(n))
        : undefined;
      const claimResult = await submitClaim({
        text: opts.claim,
        claim_type: claimType!,
        topic: opts.topic,
        author: "claude-skill",
        dossier_slug: attestResult.dossier_slug,
        depends_on_ids: dependsOn,
      });
      emit({ ...attestResult, ...claimResult }, renderAttestWithClaim);
    } catch (e: any) {
      if (e instanceof TransientVerifierError) {
        emit({
          error: e.message,
          kind: e.kind,
          hint: e.hint,
          recorded: false,
          dossier_slug: attestResult.dossier_slug,
        }, renderClaimResult);
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
      emit(result, renderClaimResult);
    } catch (e: any) {
      if (e instanceof TransientVerifierError) {
        emit({
          error: e.message,
          kind: e.kind,
          hint: e.hint,
          recorded: false,
        }, renderClaimResult);
        process.exit(2);
      }
      throw e;
    }
  });

// ---------- claim-batch ----------
program
  .command("claim-batch <jsonlPath>")
  .description(
    "Submit multiple ATOMIC claims in one LLM round-trip. " +
      "JSONL lines: {\"text\":\"...\",\"claim_type\":\"ATOMIC\",\"dossier_slug\":\"...\",\"source_quote\":\"...\",\"topic\":\"...\",\"attribution\":\"...\"}",
  )
  .action(async (jsonlPath: string) => {
    let lines: string[];
    try {
      const text = await Bun.file(jsonlPath).text();
      lines = text.split("\n").filter((l) => l.trim());
    } catch (e: any) {
      fail(`cannot read ${jsonlPath}: ${e?.message || String(e)}`);
    }

    const items: Array<{
      text: string;
      dossier_slug: string;
      source_quote: string;
      topic?: string;
      attribution?: string;
      author?: string;
    }> = [];

    for (let i = 0; i < lines.length; i++) {
      let parsed: any;
      try {
        parsed = JSON.parse(lines[i]!);
      } catch {
        fail(`line ${i + 1}: invalid JSON`);
      }
      if (!parsed.text?.trim()) fail(`line ${i + 1}: missing "text"`);
      if (!parsed.dossier_slug) fail(`line ${i + 1}: missing "dossier_slug"`);
      if (!parsed.source_quote?.trim()) fail(`line ${i + 1}: missing "source_quote"`);
      const ct = parsed.claim_type || "ATOMIC";
      if (ct !== "ATOMIC") {
        fail(`line ${i + 1}: claim-batch only supports ATOMIC, got "${ct}"`);
      }
      items.push({
        text: parsed.text,
        dossier_slug: parsed.dossier_slug,
        source_quote: parsed.source_quote,
        topic: parsed.topic,
        attribution: parsed.attribution,
        author: parsed.author || "claude-skill",
      });
    }

    try {
      const results = await submitClaimBatch(items);
      const out = {
        results: results.map((r, idx) => ({
          idx,
          claim_id: r.claim_id,
          supported: r.status === "supported",
          score: r.score,
          reason: r.source_passage,
        })),
      };
      emit(out, renderClaimBatch);
    } catch (e: any) {
      if (e instanceof TransientVerifierError) {
        emit({
          error: e.message,
          kind: e.kind,
          hint: e.hint,
          recorded: false,
        }, renderClaimBatch);
        process.exit(2);
      }
      emit({ error: e?.message || String(e) }, renderClaimBatch);
      process.exit(1);
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
      renderListClaims,
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
    emit({ summary: { ...summary, header }, claims }, renderRecent);
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
      emit({ ...out, written_to: outPath }, renderDigest);
    } else {
      emit(out, renderDigest);
    }
  });

// ---------- get-claim ----------
program
  .command("get-claim <id>")
  .description("Get full claim by ID")
  .action((id: string) => {
    const c = store.getClaim(parseInt(id, 10));
    if (!c) fail(`claim ${id} not found`);
    emit(c, renderGetClaim);
  });

// ---------- chain ----------
program
  .command("chain <id>")
  .description("Walk dependency DAG from a claim")
  .option("--max-depth <n>", "default 6", (v) => parseInt(v, 10), 6)
  .action((id: string, opts: any) => {
    emit(store.getClaimChain(parseInt(id, 10), opts.maxDepth), renderChain);
  });

// ---------- list-topics ----------
program
  .command("list-topics")
  .description("Distinct topics with claim counts")
  .action(() => emit(store.listTopics(), renderListTopics));

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
    emit({ ok }, renderSupersede(parseInt(oldId, 10), parseInt(newId, 10), opts.reason));
  });

// ---------- session show (issue #43) ----------
program
  .command("session")
  .description("Inspect the per-transcript session-claim ledger")
  .argument("<subcommand>", 'subcommand: "show" | "status"')
  .argument("[transcriptId]", "transcript_id (auto-detected from most-recently-modified Claude Code transcript if omitted)")
  .option("--include-retracted", "Include retracted rows in the listing (show subcommand only)")
  .option("--only-active", "Only show rows that are neither retracted nor superseded (show subcommand only)")
  .option("--format <fmt>", 'output format: "text" (default) or "json"', "text")
  .action((subcommand: string, transcriptId: string | undefined, opts: any) => {
    if (subcommand !== "show" && subcommand !== "status") {
      console.error(`unknown session subcommand: ${subcommand} (expected: show | status)`);
      process.exit(2);
    }
    // Auto-detect transcript_id if omitted: most-recently-modified .jsonl
    // under any of the user's Claude Code project dirs.
    if (!transcriptId) {
      try {
        const root = join(homedir(), ".claude", "projects");
        const dirs = readdirSync(root);
        let bestPath: string | null = null;
        let bestMtime = 0;
        for (const d of dirs) {
          const fullDir = join(root, d);
          let stat: Stats;
          try {
            stat = statSync(fullDir);
          } catch {
            continue;
          }
          if (!stat.isDirectory()) continue;
          let files: string[];
          try {
            files = readdirSync(fullDir);
          } catch {
            continue;
          }
          for (const f of files) {
            if (!f.endsWith(".jsonl")) continue;
            const fullPath = join(fullDir, f);
            try {
              const fstat = statSync(fullPath);
              if (fstat.mtimeMs > bestMtime) {
                bestMtime = fstat.mtimeMs;
                bestPath = fullPath;
              }
            } catch {}
          }
        }
        if (!bestPath) {
          console.error("could not auto-detect transcript_id (no .jsonl files under ~/.claude/projects/)");
          process.exit(2);
        }
        transcriptId = basename(bestPath, ".jsonl");
        console.error(`(auto-detected transcript_id: ${transcriptId})`);
      } catch (e: any) {
        console.error(`auto-detect failed: ${e?.message || e}`);
        process.exit(2);
      }
    }

    if (subcommand === "status") {
      const c = store.getSessionFireCounts(transcriptId);
      if (c.total === 0) {
        console.log(`(no session claims for transcript_id ${transcriptId})`);
        return;
      }
      if (opts.format === "json") {
        const escalate = process.env.VOUCH_GATE_ESCALATE_UNADDRESSED === "1";
        const counter = process.env.VOUCH_GATE_COUNTER_EVIDENCE === "1";
        process.stdout.write(JSON.stringify({
          transcript_id: transcriptId,
          ...c,
          truth_bearing: c.asserts + c.hedges + c.speculates,
          humility_pct: c.asserts + c.hedges + c.speculates > 0
            ? Number((((c.hedges + c.speculates) / (c.asserts + c.hedges + c.speculates)) * 100).toFixed(1))
            : null,
          env: {
            VOUCH_GATE_ESCALATE_UNADDRESSED: escalate,
            VOUCH_GATE_COUNTER_EVIDENCE: counter,
          },
        }, null, 2) + "\n");
        return;
      }
      // Text format — formatted like the gate's stderr lines but as a
      // standalone diagnostic.
      const truth = c.asserts + c.hedges + c.speculates;
      const hed = c.hedges + c.speculates;
      const humility = truth > 0 ? `${((hed / truth) * 100).toFixed(1)}%` : "—";
      const escalate = process.env.VOUCH_GATE_ESCALATE_UNADDRESSED === "1";
      const counter = process.env.VOUCH_GATE_COUNTER_EVIDENCE === "1";
      console.log(`Session status: ${transcriptId}`);
      console.log(``);
      console.log(`  Total claims seen:   ${c.total}`);
      console.log(`  Verdict breakdown:`);
      console.log(`    grounded         ${c.grounded.toString().padStart(4)}`);
      console.log(`    ungrounded       ${c.ungrounded.toString().padStart(4)}  (fires)`);
      console.log(`    reclassified     ${c.reclassified.toString().padStart(4)}`);
      console.log(`    retracted        ${c.retracted.toString().padStart(4)}`);
      console.log(``);
      console.log(`  Stance breakdown (truth-bearing):`);
      console.log(`    ASSERT           ${c.asserts.toString().padStart(4)}`);
      console.log(`    HEDGE            ${c.hedges.toString().padStart(4)}`);
      console.log(`    SPECULATE        ${c.speculates.toString().padStart(4)}`);
      console.log(`    Humility ratio   ${hed}/${truth} = ${humility} explicit-uncertainty`);
      console.log(``);
      console.log(`  Revise backlog:      ${c.awaiting_revise}  (fired entities awaiting verification — #50 A Stage 1)`);
      console.log(``);
      console.log(`  Strict env state:`);
      console.log(`    VOUCH_GATE_ESCALATE_UNADDRESSED  ${escalate ? "ON" : "OFF (default)"}`);
      console.log(`    VOUCH_GATE_COUNTER_EVIDENCE      ${counter ? "ON" : "OFF (default)"}`);
      return;
    }

    // subcommand === "show"
    const rows = store.listSessionClaims(transcriptId, {
      include_retracted: !!opts.includeRetracted,
      only_active: !!opts.onlyActive,
    });
    if (opts.format === "json") {
      // Strip embeddings for compact JSON output.
      const stripped = rows.map((r) => {
        const { embedding: _e, ...rest } = r as any;
        return rest;
      });
      process.stdout.write(JSON.stringify(stripped, null, 2) + "\n");
      return;
    }
    if (!rows.length) {
      console.log(`(no session claims for transcript_id ${transcriptId})`);
      return;
    }
    console.log(`Session ledger for ${transcriptId} — ${rows.length} claim(s):\n`);
    for (const r of rows) {
      const flags: string[] = [];
      if (r.retracted) flags.push("RETRACTED");
      if (r.superseded_by_turn !== null) {
        flags.push(`superseded by turn ${r.superseded_by_turn} claim ${r.superseded_by_claim}`);
      }
      const flagStr = flags.length ? ` [${flags.join(", ")}]` : "";
      console.log(`  turn ${r.turn_idx} claim ${r.claim_idx}  stance=${r.stance}  verdict=${r.verdict}${flagStr}`);
      console.log(`    entity: ${r.entity}`);
      console.log(`    "${r.proposition}"`);
      if (r.reason) console.log(`    reason: ${r.reason}`);
      console.log(`    ts: ${r.ts}\n`);
    }
  });

// ---------- list-dossiers ----------
program
  .command("list-dossiers")
  .description("List captured dossiers")
  .option("--source-type <type>")
  .option("--limit <n>", "default 50", (v) => parseInt(v, 10), 50)
  .action((opts: any) => {
    emit(store.listDossiers({ source_type: opts.sourceType, limit: opts.limit }), renderListDossiers);
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
    emit(out, renderGetDossier);
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
    }, renderSearch);
  });

// ---------- gate ----------
program
  .command("gate")
  .description(
    "Stop-hook gate: scan the last assistant message in a transcript for ungrounded " +
      "named-entity factual claims and block (exit 2) if any aren't entailed by a " +
      "supported vouch claim. Designed for Claude Code Stop hook integration. " +
      "Env: VOUCH_GATE_BUDGET_MS=ms (default 25000, keep below hook timeout), " +
      "VOUCH_GATE_FAILMODE=warn|block (default warn), " +
      "VOUCH_GATE_LOG=path (default <db-dir>/gate.log).",
  )
  .option("--transcript-stdin", "Read Stop-hook payload JSON from stdin and derive transcript_path from it")
  .option("--transcript-path <path>", "Read the transcript directly from this path (overrides stdin)")
  .option("--draft <text>", "Use this text as the draft instead of reading a transcript")
  .option("--session-context <path>", "Claude Code transcript JSONL to use as session evidence for auto-grounding (used with --draft; also overrides --transcript-path/-stdin for sources)")
  .option("--strict", "exit 2 on ungrounded claims (default)")
  .option("--advisory", "exit 0 + stderr warning only; never block")
  .option("--bypass-env <name>", "env var that disables the gate when set to '1'", "VOUCH_GATE_BYPASS")
  .option("--model <id>", "extractor model (LiteLLM-style)", DEFAULT_GATE_MODEL)
  .option("--top-k <n>", "candidate claims fetched per proposition (gate defaults to 3 for latency; override with care)", (v) => parseInt(v, 10), 3)
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
      sessionContext: opts.sessionContext,
      model: opts.model,
      topK: opts.topK,
      strict,
      bypassEnv: opts.bypassEnv,
    });
    if (result.message) process.stderr.write(result.message);
    // Gate output is intentionally untouched — byte-identical compact JSON for Stop-hook consumers.
    console.log(JSON.stringify({
      blocked: result.verdict.blocked,
      pairs: result.verdict.pairs,
      ...(result.verdict.harvest ? { harvest: result.verdict.harvest } : {}),
      ...(result.verdict.classifier_error
        ? { classifier_error: result.verdict.classifier_error }
        : {}),
    }));
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
    emit(report, renderDoctor);
    if (!report.ok) process.exit(1);
  });

await program.parseAsync(process.argv);
