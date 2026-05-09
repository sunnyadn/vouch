# vouch

A confabulation gate for LLM agents. When an agent's draft reply contains
an ungrounded factual claim about a named external entity, vouch's Stop
hook blocks the turn and pushes the agent into a **Fetch-Before-Claim**
loop: the agent points vouch at a source URL, vouch fetches and persists
it as a dossier, the agent picks a verbatim quote from that dossier, and
vouch's NLI judge verifies the quote actually supports the claim.

Three pieces, one binary:

- **CLI** (`vouch fetch`, `vouch claim`, ...) — the trust-establishing
  fetcher + NLI verifier + persistent KB.
- **Skill** (`skills/vouch/SKILL.md`) — drops into `~/.claude/skills/` to
  teach the agent the Fetch-Before-Claim workflow.
- **Stop hook** (`vouch gate`) — runs after every assistant turn, scans
  the draft for ungrounded claims, and blocks until each one is grounded
  or hedged.

Without the hook, vouch is an opt-in citation system. With the hook, it's
an enforced one.

## What it does

```
1. Assistant emits draft                  2. Stop hook intercepts
   ─────────                                 ─────────
   "Foo is the leading                     → BLOCKED — ungrounded claim
    framework for X."                        about "Foo" (no candidate
                                             claim in KB)

3. Agent (skill loaded) runs the loop     4. Hook re-checks, unblocks
   ─────────                                 ─────────
   $ vouch search "Foo X"                  → All factual claims now
   $ vouch fetch https://.../foo-docs        carry [verified: id] tags
   $ vouch get-dossier <slug> --full         or are hedged
   $ vouch claim "..." \                   → turn ships
       --source-quote "..."
                          → claim_id 178
```

## Install

vouch ships as three pieces. Install the CLI first; the skill and Stop
hook are optional but together turn vouch from a passive KB into an
enforced gate.

Requires [Bun](https://bun.sh) ≥ 1.3.

### 1. CLI binary

```bash
git clone https://github.com/sunnyadn/vouch
cd vouch
bun install
bun run build              # produces dist/vouch (single binary, ~59MB)
ln -sf "$PWD/dist/vouch" ~/.local/bin/vouch
vouch --version
```

### 2. Claude Code skill

Drop the bundled skill into your Claude Code skills directory:

```bash
cp -r skills/vouch ~/.claude/skills/
```

This loads the agent-side workflow (fetch → read dossier → submit claim
→ tag synthesis). Without it, agents need to be told the loop manually
each session.

### 3. Stop hook

Merge this block into `~/.claude/settings.json` (user-level), or into
`.claude/settings.json` inside a project for project-level scope:

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

If the file already has a `hooks.Stop` array, append the inner `hooks`
entry rather than replacing the block. Drop `--strict` for advisory mode
(gate warns but does not block the turn).

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

Defaults assume Vertex Gemini + Vertex `text-embedding-005`. Switch
providers without code changes.

After setting env vars, run `vouch doctor` to verify provider credentials,
DB connectivity, and Stop-hook installation — all pure-local checks, no
API calls. Exits 1 on any failure with actionable fix hints, so it drops
cleanly into CI / setup scripts.

## How agents use it

When the Stop hook detects an ungrounded factual claim in the assistant's
draft, it injects feedback like:

```
[vouch-gate] Detected ungrounded factual claim(s) about named external entities.
  • <Entity>: "<the unverified statement>"
    (no candidate claim in KB)

Before answering, ground each claim:
  • vouch search "<keyword>" — check the KB
  • vouch fetch <url> — pull the source
  • vouch claim "<text>" --type ATOMIC --dossier <slug> --source-quote "..."
Or hedge explicitly with "(unverified, from training memory)" near the claim.
```

The agent (with the skill loaded) then runs the loop:

1. **Search the KB first** — `vouch search "<keyword>"`. If a relevant
   verified claim already exists, cite its `claim_id` and skip ahead.
2. **Otherwise fetch the source** — `vouch fetch <url>` returns a
   `dossier_slug`. vouch performs the HTTP request itself; the agent's
   own WebFetch is not the trust boundary.
3. **Read what vouch saved** — `vouch get-dossier <slug> --full`. The
   quote must come from this dossier text, not the agent's own read of
   the URL — HTML strippers differ and the quote-in-dossier check
   rejects mismatches.
4. **Submit the claim** — `vouch claim "<text>" --type ATOMIC --dossier
   <slug> --source-quote "<verbatim>"`. Returns `claim_id`.
5. **Re-emit the response** — tag the grounded claim with
   `[verified: <claim_id>]`, or hedge with `(unverified, from training
   memory)` for claims the agent chose not to ground. Hook re-checks; if
   every factual claim is tagged or hedged, the turn unblocks.

The full agent instructions live in `skills/vouch/SKILL.md` — the
canonical source for the workflow including INFERENCE / SYNTHESIS rules,
supersede semantics, output tagging conventions, and anti-patterns.

## CLI reference

vouch is agent-driven by design, but the CLI is fully usable from a shell
for one-off claims, debugging, or KB inspection.

```bash
# Fetch (vouch performs the HTTP request itself — this is the trust boundary)
vouch fetch https://arxiv.org/abs/2410.05779
# → {"dossier_slug":"evidence/arxiv/...","content_chars":68151,...}

# Read what vouch saved
vouch get-dossier <slug> --full
vouch get-dossier <slug> --offset 8000 --limit 4000

# Submit a claim
vouch claim "<text>" --type ATOMIC \
  --dossier <slug> \
  --source-quote "<verbatim 1–3 sentences from the dossier>" \
  --topic <topic> --attribution "<authors / org>"
# → {"status":"supported","score":1,"claim_id":1,"quote_match":"exact"}

# Build derived claims (no source step — depends on existing KB claim_ids)
vouch claim "<text>" --type INFERENCE --depends-on 1,2 --topic <topic> --soft-score 0.8

# Browse + search
vouch list-topics
vouch list-claims --topic <X> --status supported
vouch search "<query>" --top-k 5 --pretty
vouch chain <id>                  # walk dependency DAG

# Self-correct (audit trail preserved)
vouch supersede <old_id> <new_id> --reason "<why>"

# Diagnostics
vouch doctor
```

See `vouch <command> --help` for full flags.

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

No daemon. No HTTP server. Each invocation opens SQLite, calls the
configured provider, returns JSON. Cold start ~0.5s.

## The "Fetch Before Claim" pattern

The principle: every factual claim in an LLM's output must trace to a
verbatim source quote that *vouch itself fetched* — not invented from
latent knowledge, not hallucinated as a "probably-correct citation", not
copied from a summary. vouch enforces this in three layers:

1. **Trust-establishing fetch.** `vouch fetch <url>` is the primary entry
   point. vouch — not the agent — performs the HTTP request and persists
   the resulting content (markitdown-converted when available, in-process
   strip as fallback). The dossier is the canonical record of what was at
   the URL at fetch time.

2. **Quote-in-dossier check + NLI.** `vouch claim` rejects on two layers:
   - The submitted quote must appear in the fetched dossier content
     (after light normalization for whitespace, smart quotes, punctuation
     spacing). Forged or paraphrased quotes are rejected before reaching
     the verifier.
   - An LLM-as-judge (default Vertex Gemini 3.1 Pro) then verifies the
     claim is supported by the quote. NLI is strict — overshooting the
     quote ("most widely deployed" claimed from a quote that says only
     "is widely deployed") gets rejected.

3. **Self-correction.** `vouch supersede` records corrections with
   reason; both old and new claims stay queryable. The KB grows an audit
   trail.

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

**Trust assumption**: vouch's trust boundary is the fetcher. If `vouch
fetch <url>` returns content X, vouch treats X as canonical for that URL
at that time. URL spoofing (fake URL → real-looking content via a
controlled server) is out of scope.

**Non-goal**: cryptographic attestation. vouch is a *discipline-imposing*
record-keeping system, not a notarized fact archive. It makes sloppy
claims hard, not malicious claims impossible. For adversarial scenarios
needing cryptographic guarantees, layer in archive.org submission or
third-party attestation — out of vouch's scope.

## License

MIT.
