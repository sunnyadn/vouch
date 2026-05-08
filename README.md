# vouch

A verified-claim KB CLI — the **Fetch Before Claim (FBC)** pattern as a single
binary. Submit factual claims with their source quotes, get NLI verification,
build a queryable provenance graph with dependency chains and supersede history.

Designed for use by LLM agents (e.g. as a Claude Code skill) that want
source-traceable factual statements without inventing a citation system from
scratch.

## What it does

```
You (or an LLM) say:                    vouch responds:
─────────────────────                   ─────────────────────
"Here's a claim + the verbatim          ✓ supported (score 0.92)  → claim_id 868
 quote that supports it"                ✗ unsupported (score 0.05) → reason
                                        + persisted with full provenance
```

A KB grows over time. Claims have types (ATOMIC / SYNTHESIS / INFERENCE /
HYPOTHESIS), depend on other claims via a DAG, can be superseded with audit
trail, and are searchable by hybrid embedding + keyword.

## Install

Requires [Bun](https://bun.sh) ≥ 1.3.

```bash
git clone https://github.com/<you>/vouch ~/Projects/vouch
cd ~/Projects/vouch
bun install
bun run build              # produces dist/vouch (single binary, ~59MB)
ln -sf $PWD/dist/vouch ~/.local/bin/vouch
vouch --version
```

## Configure

Auth + provider config via env (see `.env.example`):

```bash
# Verifier — anything Vercel AI SDK supports. LiteLLM-style provider/model strings.
VOUCH_VERIFIER_MODEL=vertex_ai/gemini-3.1-pro-preview
# VOUCH_VERIFIER_MODEL=openai/gpt-4o
# VOUCH_VERIFIER_MODEL=anthropic/claude-sonnet-4-6

# Embedder — same convention.
VOUCH_EMBEDDER_MODEL=vertex_ai/text-embedding-005
# VOUCH_EMBEDDER_MODEL=openai/text-embedding-3-small

# Vertex (default) — auth via `gcloud auth application-default login`
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=global
```

Defaults assume Vertex Gemini + Vertex `text-embedding-005`. Switch providers
without code changes.

## Quick start

```bash
# Submit a claim with its source quote
vouch claim "LightRAG outperforms GraphRAG on Agriculture, CS, Legal datasets" \
  --type ATOMIC \
  --source-url https://arxiv.org/abs/2410.05779 \
  --source-quote "On the Agriculture, CS, and Legal datasets, LightRAG shows a clear advantage, significantly surpassing GraphRAG." \
  --topic rag-systems \
  --attribution "Guo et al. 2024"
# → {"status":"supported","score":1,"claim_id":1,...}

# List what's in the KB
vouch list-claims --topic rag-systems
vouch list-topics

# Build derived claims with dependency chain
vouch claim "LightRAG is competitive with GraphRAG on UltraDomain" \
  --type INFERENCE --depends-on 1 --topic rag-systems --soft-score 0.8

# Walk the dependency DAG
vouch chain 2 --pretty

# Hybrid search across claims + dossier quotes
vouch search "which RAG system wins" --top-k 5 --pretty

# Self-correct: supersede an earlier claim (preserves audit trail)
vouch supersede 2 4 --reason "INFERENCE was overreach: 3 datasets ≠ 'competitive overall'"
```

Default DB at `~/.vouch/store.db` (SQLite). Override with `VOUCH_DB_PATH`.

## Claim types

| Type | When to use | Required fields |
|---|---|---|
| `ATOMIC` | Direct fact from one source | `--source-url` + `--source-quote` |
| `QUOTATION` | Verbatim quote from one source | `--source-url` + `--source-quote` |
| `SYNTHESIS` | Cross-source statement | `--sources` (JSON array, ≥2) |
| `INFERENCE` | Logical deduction from KB claims | `--depends-on <ids>` |
| `INTERPRETATION` | Reframing a single claim | `--depends-on <id>` |
| `HYPOTHESIS` | Speculation; recorded but not verified | (no source needed) |

ATOMIC / QUOTATION / SYNTHESIS hit the verifier (NLI judge). INFERENCE /
INTERPRETATION / HYPOTHESIS skip NLI but require explicit dependencies or
honest `--soft-score`.

## Architecture

```
CLI (vouch)
 ├─ bun:sqlite  →  ~/.vouch/store.db  (claims + dossiers + dependency DAG)
 ├─ Vercel AI SDK
 │   ├─ verifier  →  generateObject({ schema: { supported, score, reason } })
 │   └─ embedder  →  embed/embedMany
 └─ commander    →  argparse + JSON output
```

No daemon. No HTTP server. Each invocation opens SQLite, calls the configured
provider, returns JSON. Cold start ~0.5s.

## The "Fetch Before Claim" pattern

The principle: every factual claim in an LLM's output must trace to a verbatim
source quote that *the LLM itself fetched* — not invented from latent knowledge,
not hallucinated as a "probably-correct citation". `vouch` enforces this by
making the source quote part of the claim's primary key:

- ATOMIC claim without `--source-quote` → rejected
- Quote that doesn't actually support the claim → NLI marks `unsupported`, claim still stored but flagged
- Derived claim without `--depends-on` → rejected for INFERENCE/INTERPRETATION
- Self-correction → `supersede` with reason; both versions remain visible

The result: a KB where every "supported" claim has a recoverable provenance
trail down to a verbatim source, and every derivation is explicit.

## License

MIT.
