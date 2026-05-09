# CLAUDE.md — vouch

A single-binary verified-claim KB CLI implementing the **Fetch Before Claim
(FBC)** pattern. Bun + bun:sqlite + Vercel AI SDK; ~3000 LOC across `src/`.
Pre-launch, zero external users. Designed to be invoked by LLM agents (e.g.
Claude Code skills) that want source-traceable factual statements.

Read [`README.md`](README.md) for the user-facing surface (commands, claim
types, threat model). This file covers what an agent maintaining vouch needs
to know to not break things.

## Architecture invariants

These define what vouch IS. Don't violate without an explicit issue.

- **No daemon, no HTTP server.** Each `vouch <cmd>` invocation opens SQLite,
  calls the configured provider, prints JSON, exits. Cold start ~0.5s. If a
  feature seems to need a daemon, the feature is wrong — find a CLI shape.
- **Single binary.** `bun run build` → `dist/vouch`. No JS/TS distribution,
  no npm publish, no Docker image. Users `ln -s dist/vouch ~/.local/bin/`.
- **JSON to stdout is the contract.** Agents parse it. Errors to stderr.
  `--pretty` reformats for humans only — never change the JSON shape based
  on `--pretty`. Exit codes are part of the contract: `0` ok, `1` user/usage
  error, `2` transient/system error (auth/network/quota — caller should
  retry, not record).
- **Fetch is the trust boundary.** `vouch fetch` does the HTTP itself and
  persists the dossier. `vouch claim` then verifies a quote against that
  dossier. `vouch attest` is the user-attested escape hatch (no fetch,
  user takes responsibility). Anything that lets an agent supply
  unfetched/unattested "source content" defeats the entire product —
  reject such PRs.
- **Quote-in-dossier check is non-negotiable for ATOMIC/QUOTATION/SYNTHESIS.**
  See `src/quote-match.ts` (3-tier: exact → normalized → fuzzy alphanumeric).
  This is the anti-fabrication primitive. NLI runs after, not instead.
- **Provider strings are LiteLLM-style** (`vertex_ai/gemini-3.1-pro-preview`,
  `openai/gpt-4o`, `anthropic/claude-sonnet-4-6`). Anything Vercel AI SDK
  supports works without code changes; see `src/providers.ts`.

## Non-goals (explicit)

Past discussion has settled these. Don't relitigate without an issue.

- Cryptographic attestation / notarization (out of scope per README threat
  model — vouch makes sloppy claims hard, not malicious claims impossible).
- Multi-user, SaaS, hosted tier, web UI.
- Adversarial defense against URL spoofing or verifier gaming.
- Migration tooling, semver ceremony, deprecation cycles. Pre-launch, zero
  users — change the code shape directly. Schema changes are additive ALTER
  with `try { ... } catch (duplicate column) {}` for idempotency (see
  `src/store.ts` for the pattern).

## Error semantics

`TransientVerifierError` (in `src/verifier.ts`, `kind: auth | network | quota
| unknown`) is for **system** failures and must NOT be recorded in the KB —
they carry no information about the (claim, source) pair. CLI surfaces them
as `{error, kind, hint, recorded: false}` with exit 2.

`unsupported` / `insufficient` ARE evidence outcomes — record them. The KB's
value is the audit trail, including failed attempts.

Don't add try/catch that swallows transient errors silently — they need to
surface so the caller can retry or fix auth.

## File layout

Flat. Don't reorganize.

- `src/cli.ts` — argparse + JSON emit + exit-code surface (single file)
- `src/submit.ts` — claim submission dispatch by type
- `src/store.ts` — bun:sqlite schema + queries (single source of truth for DB)
- `src/verifier.ts` — NLI judge + chunk retrieval + transient-error classifier
- `src/quote-match.ts` — anti-fabrication quote-in-dossier check
- `src/fetch.ts` + `src/fetchers/` — URL → dossier (arxiv, github, generic,
  opencli fallback for JS-rendered, markitdown for PDF/office)
- `src/attest.ts` — user-attested dossiers (no HTTP)
- `src/embedder.ts`, `src/providers.ts` — Vercel AI SDK glue
- `src/types.ts`, `src/config.ts` — shared

Tests in `tests/` mirror `src/` modules. Run with `bun test`.

## Configuration

Env loaded from `~/.vouch/.env` first, then `cwd/.env`. The compiled binary
runs from arbitrary cwd (often the meta vault), so user config MUST live at
`~/.vouch/.env`, not a project `.env`. See `src/config.ts`.

DB path: `~/.vouch/store.db` by default; override with `VOUCH_DB_PATH`.

## Working on this codebase

- **Issues come from Linear (SUN-XX).** Pick up an issue, work the acceptance
  list, commit. Issue + this file should be enough context — if not, the
  issue is under-specified.
- **Wider context lives in the meta vault** at `~/Projects/meta/sunny/`
  (private). Decisions about vouch's direction (launch criteria, scope) are
  filed there. Don't pull meta context speculatively — the issue should cite
  it if relevant.
- **Commit style** mirrors the existing log: `v0.x.x: ...` for releases,
  `feat(...)`, `fix(...)`, `chore: ...` for incremental work. Single `main`
  branch, no release branches.
- **Test before commit:** `bun test` (full suite, ~seconds). `bun run
  typecheck` for tsc.
- **Write loudly, fail loudly.** No silent error swallowing, no defensive
  try/catch around things that "shouldn't happen." Trust internal calls;
  validate at the CLI boundary.
- **Keep it small.** 2963 LOC today. New features justify their LOC. Prefer
  fewer abstractions; this is not a framework.

## Downstream awareness

Vouch is used by:
- The meta vault's evidence + decisions convention (see meta vault's
  `CLAUDE.md` §"Evidence dossiers"). The vault writes `vouch_claims:
  [...]` arrays in dossier frontmatter — depends on stable JSON output
  shape from `vouch claim` and `vouch attest`.
- Claude Code skill at `~/.claude/skills/vouch/`. Skill invokes the binary
  via Bash; depends on JSON contract and exit codes.

Breaking the JSON shape or exit-code semantics is a downstream-breaking
change — surface it in the commit message and check the consuming surfaces.
