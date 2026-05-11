---
name: vouch
description: |
  Use vouch as a strict claim verifier + persistent KB while YOU (Claude) do
  the research and decision-making. vouch is a CLI that fetches sources
  itself (the trust-establishing step), persists them as dossiers, then
  verifies claims you make against those dossiers via NLI.

  Do your research THROUGH vouch — the dossier a claim cites is a byproduct
  of reading the source, not a separate step:
    1. `vouch search "<question>"` — KB-first (reuses dossiers/claims you
       have); web-fallback (DuckDuckGo, or --provider for academic) when thin
    2. `vouch fetch <url>` — vouch does the HTTP request AND returns the
       content (drop-in for a web-fetch tool); the dossier persists as a side
       effect. Don't use a built-in WebFetch for anything you'll cite.
    3. `vouch claim "<text>" --type ATOMIC --dossier <slug>` — quote
       auto-selected from the dossier (pass --source-quote only to pin one);
       vouch runs NLI
    4. For inferences from KB claims: `vouch claim "..." --type INFERENCE
       --depends-on <ids>`
    5. Use claim_ids when responding to user

  Triggers: research-grounded writing, decision memos, comparative analyses,
  any synthesis where the user wants source-traceable factual statements. Or
  when user explicitly says "use vouch" / "vouch verify this" / "build kb on X".

  Skip when: casual chat, code help, real-time data ("current btc price"), or
  questions where source-grounding adds friction without value.
---

# vouch — Claude-driven verification

## Mental model — vouch IS your research loop

```
Your natural loop:        →  routes through vouch:           vouch's job:
─────────────────            ──────────────────────          ────────────
search for a source          vouch search "<q>"              KB-first; web-fallback
read the source              vouch fetch <url>               fetch + persist + RETURN content
reason, write a sentence     vouch claim "<text>" --dossier  auto-pick quote, run NLI
classify each statement      (you set --type)                store result + dependency DAG
correct yourself             vouch supersede                 audit trail preserved
```

Do your research THROUGH vouch, not alongside it. `vouch fetch` returns the
page content directly (head chunk, or `--full`) — it is a drop-in for a
built-in web-fetch tool, and the side effect is that the dossier the agent
cites is a byproduct of the read, not a separate ceremony. **For anything you
will cite, `vouch fetch` it — don't use a built-in WebFetch** (its stripped
text diverges from vouch's, which breaks the quote check). Built-in WebSearch
is fine for *discovering* candidate URLs when `vouch search`'s web fallback is
thin, but the URL you settle on goes through `vouch fetch`.

`--source-quote` on `vouch claim` is optional: omit it and vouch auto-selects
a supporting passage from the dossier. Pass it only to pin a specific quote.

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

## Workflow

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

**Don't use a built-in WebFetch for anything you'll cite** — vouch's stripped
text diverges from a built-in fetcher's, which breaks the quote check. Built-in
WebSearch is OK only to *discover* candidate URLs when `vouch search`'s web
fallback came up thin; the URL you settle on still goes through `vouch fetch`.
If `vouch fetch` errors (auth wall, JS-heavy site, PDF without HTML), surface
the limit to the user — don't fabricate, and don't quietly fall back to a
built-in fetcher and then claim against it.

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

- ❌ Don't use a built-in WebFetch for a URL you'll cite — `vouch fetch` it
  (returns the content AND persists the dossier); a built-in fetcher's
  stripped text diverges from vouch's and breaks the quote check
- ❌ Don't skip `vouch search` and fetch blindly — search is KB-first, it
  tells you whether you already have a dossier/claim to reuse
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
