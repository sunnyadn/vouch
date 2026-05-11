---
name: vouch
description: |
  vouch is a strict claim verifier + persistent KB for LLM agents. You do the
  research with your NATIVE tools (Read / WebFetch / WebSearch / `cat`); vouch's
  Stop-hook (`vouch gate`) then checks any ungrounded named-entity factual claim
  in your draft. If you already retrieved a supporting source THIS SESSION, the
  gate finds it in the transcript, snapshots it as a dossier, files the claim,
  and passes — you'll see `[verified: <id>] auto-grounded from session`, and you
  do nothing. You run vouch commands by hand only when:
    (a) the gate fires on something you have NOT retrieved (training memory, or a
        source you never actually read): `vouch search "<q>"` (does the KB
        already have it?) → `vouch fetch <url>` (vouch does the HTTP + persists
        the dossier) → read what vouch saved → `vouch claim "<text>" --type
        ATOMIC --dossier <slug>` (quote auto-selected; vouch runs NLI). Or hedge
        `(unverified, from training memory)`.
    (b) you're DELIBERATELY building source-traceable KB (decision memo, dossier
        work where the user wants claim_ids): same `vouch search → fetch →
        claim` loop, run proactively.
    (c) inferences from KB claims: `vouch claim "..." --type INFERENCE
        --depends-on <ids>`; corrections: `vouch supersede <old> <new>`.
  Use your native WebSearch/WebFetch freely — the gate captures whatever you
  read. `vouch search` is for reusing the KB (dedup against dossiers/claims you
  already have), not as your web-search tool.

  Triggers: research-grounded writing, decision memos, comparative analyses,
  any synthesis where the user wants source-traceable factual statements. Or
  when user explicitly says "use vouch" / "vouch verify this" / "build kb on X".

  Skip when: casual chat, code help, real-time data ("current btc price"), or
  questions where source-grounding adds friction without value.
---

# vouch — Claude-driven verification

## Mental model — research natively; the gate verifies + remembers

You don't route research *through* vouch. Use your normal tools — `Read`,
`WebFetch`, `WebSearch`, `cat` — and vouch sits behind them:

- **The Stop-hook (`vouch gate`)** scans your draft for ungrounded named-entity
  factual claims. If a claim fires and you **already pulled a supporting source
  this session** (a file you `Read`/`cat`-ed, a page you `WebFetch`-ed, results
  from `WebSearch`), the gate finds it in the transcript, snapshots it as a
  dossier, files the claim, and passes — you'll see `[verified: <id>]
  auto-grounded from session ...`. **You do nothing** for that case; the source
  you already read *is* the grounding.
- If a claim fires and **no session source supports it** — you're citing
  something you never actually retrieved, or it's straight from training memory.
  *Then* you act: `vouch search "<q>"` (does the KB already have it?) → `vouch
  fetch <url>` (vouch retrieves + persists it) → read what vouch saved → `vouch
  claim "<text>" --dossier <slug>`. Or hedge `(unverified, from training
  memory)`. The gate's block message tells you how many session sources it
  checked and, for a `WebFetch` result that didn't entail, suggests `vouch fetch
  <url>` for the raw page (WebFetch returns model-extracted text, which can miss
  the supporting span).
- You also run the `search → fetch → claim` loop **proactively** when you're
  deliberately building source-traceable KB (decision memo, dossier work where
  the user wants `claim_ids`) rather than waiting for the gate to fire.
- `vouch search` is for **reusing the KB** — "do I already have a dossier/claim
  on this?" Use your native `WebSearch` to *discover* new web sources; use
  `vouch search` to dedup against what vouch already knows.

When you *do* call `vouch claim` by hand against a `vouch fetch` dossier, quote
from what vouch saved (`vouch get-dossier <slug>`), not from your own read of
the URL — vouch's stripped text can differ from a built-in fetcher's, and the
quote-in-dossier check rejects mismatches. (`--source-quote` is optional anyway:
omit it and vouch auto-selects a supporting passage. Pass it only to pin one.)
Using your native `WebFetch` for *research* is fine — it's only the manual
`--source-quote` path that needs the quote to come from vouch's dossier text.

## Smoke test before first use

```bash
vouch health
# → {"ok":true,"db_path":"...","verifier_model":"vertex_ai/gemini-3.1-pro-preview","db_ok":true}
```

If `vouch` not on PATH: install per `~/Projects/vouch/README.md`.

## Commands

```bash
# Find a source — KB-first, web-fallback (DuckDuckGo by default; --provider
# openalex|pubmed|arxiv|google-scholar for academic). Reuses dossiers/claims
# you already have, so check this BEFORE fetching anything.
vouch search "<question>" [--provider <p>] [--limit 5] [--kb-only] [--web-only]
# → {kb_sufficient, kb:[...hits...], web:[{title,url,snippet}]|null, web_provider, ...}

# Fetch — vouch does the HTTP request itself (trust boundary) AND returns the
# content, so this is your web-fetch tool. The dossier persists as a side effect.
vouch fetch <url> [--fetcher arxiv|generic] [--force-refetch] [--full | --content-limit N]
# → {dossier_slug, content:"<first ~8000 chars>", content_chars, title, ...}
vouch get-dossier <slug> --offset 8000 --limit 4000   # re-read a later window

# Claim (against an already-fetched dossier). --source-quote OPTIONAL — omit it
# and vouch auto-selects the supporting passage from the dossier.
vouch claim "<text>" --type ATOMIC \
  --dossier <slug> \
  [--source-quote "<verbatim 1–3 sentences from the dossier>"] \
  --topic <topic> \
  --attribution "<authors / org>"

# SYNTHESIS — cross-source statement (≥2 dossiers)
vouch claim "<text>" --type SYNTHESIS \
  --sources '[{"dossier_slug":"...","quote":"..."},{"dossier_slug":"...","quote":"..."}]' \
  --topic <topic>

# INFERENCE / INTERPRETATION — built on existing KB claims (no source step)
vouch claim "<text>" --type INFERENCE --depends-on 854,856 --topic <topic> --soft-score 0.7

# HYPOTHESIS — recorded but unverified
vouch claim "<text>" --type HYPOTHESIS --topic <topic> --soft-score 0.4

# Read existing KB
vouch list-topics
vouch list-claims --topic <X> --status supported --contains <kw>
vouch get-claim <id>
vouch chain <id>                     # walk dependency DAG
vouch list-dossiers

# Correct yourself
vouch supersede <old_id> <new_id> --reason "<why old was wrong>"

# Pretty-print any command
vouch --pretty <command> ...
```

## Workflow — the `search → fetch → claim` loop

Run this loop when the gate fired on a claim you *haven't* already retrieved a
source for, OR when you're proactively building source-traceable KB. If you
already read the source this session, the gate auto-grounds it for you — skip
to Step 4 (tagging) and just cite the auto-grounded `claim_id` it printed.

### Step 1 — `vouch search` (KB-first, web-fallback in one call)

```bash
vouch search "<your question>"
# kb_sufficient: true  → the KB already covers this. Read the kb hits
#                        (vouch get-claim <id> / get-dossier <slug>), cite the
#                        existing claim_ids, synthesize. Skip to Step 4 or done.
# kb_sufficient: false → web[] holds candidate URLs. Pick the right one → Step 2.
#                        For a scholarly source pass --provider openalex|pubmed|
#                        arxiv|google-scholar (the DuckDuckGo default is general).
```

`vouch list-topics` / `vouch list-claims --topic <X> --contains <kw>` are still
useful for browsing, but `vouch search` is the one move that both dedups
against the KB and finds new sources.

### Step 2 — `vouch fetch` the chosen URL

```bash
vouch fetch https://arxiv.org/abs/2410.05779
# → {"dossier_slug":"evidence/arxiv/...", "content":"<first ~8000 chars>", "content_chars":68151, ...}
vouch fetch <url> --full          # entire content if the head chunk isn't enough
```

The `content` in the result IS the page text — read it directly, no separate
call needed. For very long sources, `vouch get-dossier <slug> --offset N --limit M`
pages through later. vouch picks the right fetcher (arxiv → HTML version +
arxiv-API metadata; generic fallback otherwise).

You can also reach this dossier from a source you already pulled with your
native `WebFetch` this session — the gate snapshots it automatically when it
auto-grounds a claim. `vouch fetch` here is for the case where you're citing a
source you *haven't* retrieved (or want vouch's canonical copy for a manual
`--source-quote`). If `vouch fetch` errors (auth wall, JS-heavy site, PDF
without HTML), surface the limit to the user — don't fabricate, and don't
quietly fall back to a built-in fetcher and then `vouch claim --source-quote`
against text vouch never saw.

### Step 3 — Submit claims (quote auto-selected)

```bash
vouch claim "LightRAG outperforms GraphRAG on Agriculture, CS, Legal datasets" \
  --type ATOMIC \
  --dossier evidence/arxiv/... \
  --topic rag-systems \
  --attribution "Guo et al. 2024"
# → {"status":"supported","score":1,"claim_id":12,"quote_match":"...","metadata":{"auto_selected_quote":true}}
#
# Pass --source-quote "<verbatim 1–3 sentences from the dossier>" only when you
# want to pin a specific passage rather than let vouch pick.
```

**If `error: quote-not-in-dossier`** (only happens when you passed
`--source-quote`): the quote you pinned isn't in vouch's stripped text. Drop
`--source-quote` (let vouch auto-pick), or re-read the dossier `content` and
copy the EXACT phrasing, or the dossier may not cover what you want — fetch a
different URL.

**If `error: quote-not-in-dossier` with `auto_selected`** / `auto-quote: no
supporting passage found`: the dossier genuinely contains nothing that
entails the claim — the source doesn't support it. Fetch a different source or
drop the claim.

**If `status: unsupported` (NLI rejected)**: the quote IS in the dossier but
doesn't actually support your claim. Tighten the claim, pass a stronger
`--source-quote`, or drop the claim. NLI is strict by design — claims that
overshoot their quote are caught here.

### Step 4 — Tag the synthesis

When drafting your response to the user, every factual segment carries a tag:

| Tag | When | Constraints |
|---|---|---|
| `[verified: <id>]` | Direct fact from a verified ATOMIC claim | cite claim_id |
| `[synthesis-of: <id1>, <id2>]` | Cross-source statement | cite all source IDs |
| `[inference-from: <id1>, ...]` | Conclusion deduced from verified claims | scenario-anchored OR paper-derived only |
| `[interpretation: <id>]` | Reframing without adding content | cite the single claim |
| `[hypothesis]` | Speculation you can't fully justify | explicit |

### CRITICAL — when can you use INFERENCE / SYNTHESIS?

Only in two cases:

1. **SCENARIO-ANCHORED** — directly answers the user's specific question,
   every step deduces from verified atomic claims.
2. **PAPER-DERIVED** — paraphrases an inference the paper authors themselves
   wrote (cite their dossier; attribution to authors).

If neither: **DOWNGRADE TO HYPOTHESIS**. Free-form "interesting connection"
is HYPOTHESIS, not INFERENCE.

### Step 5 — Heuristic for adversarial decisions

When the user asks "X vs Y, which?", the most credible evidence is **a quote
where the loser admits the weakness in their own docs**. A vendor admitting
"X is expensive" in their own README has more leverage than any third-party
benchmark, because vendors bias toward overselling — self-admissions are
load-bearing.

So: when answering "X vs Y", spend fetch budget finding self-incriminating
quotes from each side, not balanced comparison documents. One self-admission
quote per option > five neutral-comparison quotes.

### Step 6 — Record derived claims back

For each `[inference-from: ...]` / `[synthesis-of: ...]` / `[hypothesis]`
segment, file it back so future sessions can build on it:

```bash
vouch claim "Given LightRAG's lower retrieval token cost and the Microsoft README's own 'expensive' warning, LightRAG is the more defensible default for cost-constrained SaaS." \
  --type INFERENCE \
  --topic rag-systems \
  --depends-on 854,856,858 \
  --soft-score 0.8 \
  --attribution claude-skill
```

DO NOT post `[verified]` / `[interpretation]` segments unless the
interpretation is a substantive reframing worth re-using.

### Step 7 — Self-audit before responding

Scan your draft:

- Any `[verified]` claim has a matching claim_id from KB?
- Any `[inference]` is genuinely (1) scenario-anchored or (2) paper-derived?
- Any over-reach? Mark as `[hypothesis]` instead.

If you find an old claim was wrong:

```bash
# Submit corrected claim → returns new_id
vouch claim "<corrected>" --type HYPOTHESIS --topic <X> --soft-score 0.5

# Mark old as superseded
vouch supersede 864 870 --reason "On audit: this overreached because [explain]."
```

`vouch list-claims` shows superseded claims with `superseded_by` +
`supersede_reason`. The KB records its own corrections — don't delete past
mistakes, supersede them.

## Output format expected by user

Markdown response:
1. Direct answer / synthesis to user's question
2. Each factual segment carries a tag from the table above
3. Brief footer:
   ```
   ──────────────────
   vouch: filed N atomic claims, M derivations under topic "<X>".
   New claim_ids: [868, 869, 870, ...]
   ```

## Anti-patterns (don't do these)

- ❌ Don't `vouch claim --source-quote "..."` from your own WebFetch read —
  a manual quote must come from vouch's dossier text (`vouch get-dossier`), or
  omit `--source-quote` and let vouch auto-select. (Using native WebFetch for
  *research* is fine — the gate auto-grounds from it.)
- ❌ Don't skip `vouch search` when you're about to fetch/build — it's KB-first,
  it tells you whether you already have a dossier/claim to reuse
- ❌ Don't tag `[verified]` for things you derived but didn't fetch
- ❌ Don't tag `[inference]` for "interesting connections" — those are HYPOTHESIS
- ❌ Don't fabricate claim_ids — they must come from `vouch list-claims` output
- ❌ Don't auto-supersede other agents' claims without strong justification
- ❌ Don't pass a paraphrased `--source-quote` — it must be verbatim from the
  dossier (or omit it and let vouch auto-select)

## Anti-pattern: silent rewriting

If you find a wrong claim in KB, NEVER:
- Edit it directly
- Just write a new one without linking
- Pretend the old one didn't exist

DO:
- `vouch claim "<corrected>" ...` → returns new_id
- `vouch supersede <old> <new> --reason ...`
- Both versions stay queryable; the audit trail is the value.

This is the core differentiation: vouch's KB records its own corrections.
