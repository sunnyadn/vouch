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
git clone https://github.com/sunnyadn/vouch
cd vouch
bun install
bun run build              # produces dist/vouch (single binary, ~59MB)
# Symlink the binary into a directory on your PATH, e.g.:
ln -sf "$PWD/dist/vouch" ~/.local/bin/vouch
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

# Vertex (default) — auth via service-account JSON key
GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa-key.json
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=global
```

Defaults assume Vertex Gemini + Vertex `text-embedding-005`. Switch providers
without code changes.

After setting env vars, run `vouch doctor` to verify provider credentials,
DB connectivity, and (optionally) the Claude Code Stop-hook installation —
all pure-local checks, no API calls. Exits 1 on any failure with actionable
fix hints, so it's safe to drop into CI / setup scripts.

## Quick start

vouch is a **two-phase tool**: first fetch a source (vouch does the fetch),
then make claims against the resulting dossier.

```bash
# Phase 1 — vouch fetches the URL and persists the full content as a dossier
vouch fetch https://arxiv.org/abs/2410.05779
# → {"dossier_slug":"evidence/arxiv/...","title":"LightRAG: Simple and Fast...","content_chars":68151,...}

# Read the dossier text vouch saved (use this to find quotes that will match)
vouch get-dossier evidence/arxiv/... --full

# Phase 2 — submit a claim against the dossier with a verbatim quote
vouch claim "LightRAG outperforms GraphRAG on Agriculture, CS, Legal datasets" \
  --type ATOMIC \
  --dossier evidence/arxiv/... \
  --source-quote "On the Agriculture, CS, and Legal datasets, LightRAG shows a clear advantage..." \
  --topic rag-systems \
  --attribution "Guo et al. 2024"
# → {"status":"supported","score":1,"claim_id":1,"quote_match":"exact",...}

# List what's in the KB
vouch list-claims --topic rag-systems
vouch list-topics

# Build derived claims with dependency chain (no source step needed)
vouch claim "LightRAG is competitive with GraphRAG on UltraDomain" \
  --type INFERENCE --depends-on 1 --topic rag-systems --soft-score 0.8

# Walk the dependency DAG
vouch chain 2 --pretty

# Hybrid search across claims + dossier content
vouch search "which RAG system wins" --top-k 5 --pretty

# Self-correct: supersede an earlier claim (preserves audit trail)
vouch supersede 2 4 --reason "INFERENCE was overreach: 3 datasets ≠ 'competitive overall'"
```

Default DB at `~/.vouch/store.db` (SQLite). Override with `VOUCH_DB_PATH`.

## Claude Code integration (optional)

`vouch gate` is a Stop-hook subcommand that scans the last assistant message
for ungrounded factual claims about named external entities and prompts the
agent to either ground them (via `vouch fetch` + `vouch claim`) or hedge
explicitly. This is what turns vouch from a passive KB into an active
confabulation gate during a Claude Code session.

To enable, merge this block into your Claude Code settings (typically
`~/.claude/settings.json` for user-level, or `.claude/settings.json` inside
a project for project-level):

```json
{
  "hooks": {
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "vouch gate --transcript-stdin --strict",
            "timeout": 30,
            "statusMessage": "vouch gate"
          }
        ]
      }
    ]
  }
}
```

If the file already has a `hooks.Stop` array, append the inner `hooks` entry
to the existing one rather than replacing the whole block. The `vouch` binary
must be on PATH (see Install). Drop `--strict` for advisory-only mode (gate
warns but does not block the assistant turn).

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
source quote that *vouch itself fetched* — not invented from latent knowledge,
not hallucinated as a "probably-correct citation", not a quote the agent
copied from a different summary. `vouch` enforces this in two layers:

1. **Trust-establishing fetch.** `vouch fetch <url>` is the primary entry
   point. vouch — not the agent — performs the HTTP request and persists the
   resulting content (markitdown-converted when available, in-process strip
   as fallback). The dossier is the canonical record of what was at the URL
   at fetch time.

2. **Quote-in-dossier check + NLI.** `vouch claim` rejects on two layers:
   - The submitted quote must appear in the fetched dossier content (after
     light normalization for whitespace, smart quotes, punctuation spacing).
     Forged or paraphrased quotes are rejected before reaching the verifier.
   - An LLM-as-judge (default Vertex Gemini 3.1 Pro) then verifies the
     claim is supported by the quote. NLI is strict — overshooting the
     quote ("most widely deployed" claimed from a quote that says only
     "is widely deployed") gets rejected.

3. **Self-correction.** `vouch supersede` records corrections with reason;
   both old and new claims stay queryable. The KB grows an audit trail.

## Threat model

vouch is honest about what it does and doesn't guarantee:

| Threat | Caught by vouch? |
|---|---|
| Agent invents a quote that isn't in the source | ✅ quote-in-dossier check |
| Agent paraphrases instead of quoting verbatim | ✅ quote-in-dossier check (with light normalization) |
| Agent submits a quote that doesn't actually support its claim | ✅ NLI judge |
| Agent claims more than the quote says (overreach) | ✅ NLI judge — strict by design |
| Agent submits an absent claim ("source does not mention X") incorrectly | ✅ NLI judge with absence prompt |
| Agent submits a real quote out of context | ⚠️ partial — frontier verifier sometimes catches, sometimes not |
| Agent fabricates the URL itself (URL doesn't exist or doesn't say what's claimed) | ❌ vouch fetch errors on bad URLs but trusts whatever returns |
| Agent intentionally games the verifier with trivial-but-supported claims | ❌ no defense; relies on visible audit trail + supersede |
| Source page changes after fetch | ⚠️ source_hash records the fetched state; drift detectable on re-fetch |
| Verifier model itself hallucinates a "supported" verdict | ⚠️ choose stronger verifier; rotate occasionally for high-stakes claims |

**Trust assumption**: vouch's trust boundary is the fetcher. If `vouch fetch
<url>` returns content X, vouch treats X as canonical for that URL at that
time. URL spoofing (fake URL → real-looking content via a controlled server)
is out of scope.

**Non-goal**: cryptographic attestation. vouch is a *discipline-imposing*
record-keeping system, not a notarized fact archive. It makes sloppy claims
hard, not malicious claims impossible. For adversarial scenarios needing
cryptographic guarantees, layer in archive.org submission or third-party
attestation — out of vouch's scope.

## License

MIT.
