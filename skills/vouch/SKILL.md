---
name: vouch
description: |
  Use vouch as a strict claim verifier + persistent KB while YOU (Claude) do
  the research and decision-making. vouch is a CLI that fetches sources
  itself (the trust-establishing step), persists them as dossiers, then
  verifies claims you make against those dossiers via NLI.

  YOU drive the workflow:
    1. Decide whether the KB already has relevant claims (`vouch list-claims`,
       `vouch search`)
    2. If gap exists, point vouch at the source: `vouch fetch <url>` →
       returns a dossier_slug
    3. Read what vouch fetched: `vouch get-dossier <slug> --full` (or paged)
    4. Pick a verbatim 1–3 sentence quote FROM THE DOSSIER TEXT (not from
       your own WebFetch — vouch's stripped text may differ)
    5. `vouch claim "<text>" --type ATOMIC --dossier <slug> --source-quote "..."`
       — vouch verifies the quote is in the dossier, then runs NLI
    6. For inferences from KB claims: `vouch claim "..." --type INFERENCE
       --depends-on <ids>`
    7. Use claim_ids when responding to user

  Triggers: research-grounded writing, decision memos, comparative analyses,
  any synthesis where the user wants source-traceable factual statements. Or
  when user explicitly says "use vouch" / "vouch verify this" / "build kb on X".

  Skip when: casual chat, code help, real-time data ("current btc price"), or
  questions where source-grounding adds friction without value.
---

# vouch — Claude-driven verification

## Mental model

```
You are the agent. vouch is a tool with two phases:

  Your job:                          vouch's job:
  ─────────                          ────────────
  · pick what to research            · fetch the URL itself
  · ask vouch to fetch the URL       · persist full content as a dossier
  · read the dossier vouch saved     · check your quote is in the dossier
  · pick a verbatim quote from it    · run NLI on (claim, quote)
  · classify each statement          · store result + dependency DAG
  · own inference logic              · supersede on correction
```

**Critical**: the source quote you submit must appear in the dossier vouch
fetched. If you WebFetch a URL yourself and submit a quote from your own
read, the text will likely diverge from what vouch stored (different
HTML→text strippers). Always read `vouch get-dossier --full` first and pick
the quote from there.

## Smoke test before first use

```bash
vouch doctor
# → {"ok":true,"checks":[{"name":"db","status":"ok",...},...]}
```

If `vouch` not on PATH or any check is `fail`: see the vouch repo README
for install + provider-credential setup.

## Commands

```bash
# Phase 1: fetch (trust-establishing)
vouch fetch <url> [--fetcher arxiv|generic] [--force-refetch]
# → {dossier_slug, source_type, title, publication_date, author_attribution,
#    content_chars, cached, fetched_at}

# Read what vouch saved
vouch get-dossier <slug>             # 4000-char preview
vouch get-dossier <slug> --full      # full content
vouch get-dossier <slug> --offset 8000 --limit 4000  # specific window

# Phase 2: claim (against an already-fetched dossier)
vouch claim "<text>" --type ATOMIC \
  --dossier <slug> \
  --source-quote "<verbatim 1–3 sentences from the dossier>" \
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
vouch search "<query>" --top-k 8     # hybrid: claims + dossiers
vouch list-dossiers

# Correct yourself
vouch supersede <old_id> <new_id> --reason "<why old was wrong>"

# Pretty-print any command
vouch --pretty <command> ...
```

## Workflow

### Step 1 — Check KB before fetching

```bash
vouch list-topics
vouch list-claims --topic <existing> --contains <keyword>
vouch search "<your question>"
# If KB already covers this, read the existing claims, synthesize, skip Step 2.
```

### Step 2 — Fetch (only if KB gap)

```bash
vouch fetch https://arxiv.org/abs/2410.05779
# → {"dossier_slug":"evidence/arxiv/...", "content_chars":68151, ...}
```

vouch picks the right fetcher (arxiv special case, generic fallback). For
arxiv URLs, vouch fetches the HTML version (cleaner than PDF/abstract) and
populates publication_date + author_attribution from the arxiv API.

**Fetch tool selection:**
1. **`vouch fetch`** — primary. Public, static, or arxiv-style URLs.
2. **WebFetch / OpenCLI for general research** — to discover URLs worth
   submitting to vouch. Then `vouch fetch` the chosen ones.
3. If `vouch fetch` errors (auth wall, JS-heavy site, PDF without HTML
   alternative), surface the limit to the user; don't fabricate.

### Step 3 — Read the dossier

```bash
vouch get-dossier <slug> --full | jq -r .content
# Search for relevant section, e.g. with grep / a sub-agent / your own read.
```

For long dossiers (papers, READMEs), use `--offset / --limit` to page through.

### Step 4 — Submit claims

```bash
vouch claim "LightRAG outperforms GraphRAG on Agriculture, CS, Legal datasets" \
  --type ATOMIC \
  --dossier evidence/arxiv/... \
  --source-quote "On the Agriculture, CS, and Legal datasets, LightRAG shows a clear advantage..." \
  --topic rag-systems \
  --attribution "Guo et al. 2024"
# → {"status":"supported","score":1,"claim_id":12,"quote_match":"exact"}
```

**If `error: quote-not-in-dossier`**: your quote does not appear in vouch's
stripped text. Either:
- Re-read `vouch get-dossier --full` and copy the EXACT phrasing
- Pick a different quote from the dossier
- The dossier may not actually cover what you want — fetch a different URL

**If `status: unsupported` (NLI rejected)**: the quote IS in the dossier but
doesn't actually support your claim. Tighten the claim, find a stronger
quote, or drop the claim. NLI is strict by design — claims that overshoot
their quote are caught here.

### Step 5 — Tag the synthesis

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

### Step 6 — Heuristic for adversarial decisions

When the user asks "X vs Y, which?", the most credible evidence is **a quote
where the loser admits the weakness in their own docs**. A vendor admitting
"X is expensive" in their own README has more leverage than any third-party
benchmark, because vendors bias toward overselling — self-admissions are
load-bearing.

So: when answering "X vs Y", spend fetch budget finding self-incriminating
quotes from each side, not balanced comparison documents. One self-admission
quote per option > five neutral-comparison quotes.

### Step 7 — Record derived claims back

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

### Step 8 — Self-audit before responding

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

- ❌ Don't WebFetch a URL and submit a quote from your own read — submit
  via `vouch fetch` then quote from `vouch get-dossier --full`
- ❌ Don't ask vouch to do web search (you do it via WebSearch)
- ❌ Don't tag `[verified]` for things you derived but didn't fetch
- ❌ Don't tag `[inference]` for "interesting connections" — those are HYPOTHESIS
- ❌ Don't fabricate claim_ids — they must come from `vouch list-claims` output
- ❌ Don't auto-supersede other agents' claims without strong justification
- ❌ Don't paraphrase the source quote — it must be verbatim from the dossier

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
