---
name: vouch
description: |
  vouch is a strict claim verifier + persistent KB. You research with your
  native tools (Read / WebFetch / WebSearch / `cat`); vouch's Stop-hook (`vouch
  gate`) checks ungrounded named-entity claims in your draft, auto-grounds any
  whose source you already retrieved this session, and harvests your tagged
  `[inference-from:]` / `[synthesis-of:]` / `[interpretation:]` / `[hypothesis]`
  segments back as derived claims. If it fires on something you haven't
  retrieved: `vouch fetch <url>` → `vouch claim "<text>" --dossier <slug>`, or
  hedge `(unverified, from training memory)`. Corrections: `vouch supersede`.

  Triggers: research-grounded writing, decision memos, comparative analyses, any
  synthesis where the user wants source-traceable factual statements; or when
  the user says "use vouch" / "vouch verify this" / "build kb on X".

  Skip when: casual chat, code help, real-time data, source-grounding adds
  friction without value.
---

# vouch — Claude-driven verification

## Mental model

Research natively. Use your normal `Read` / `WebFetch` / `WebSearch` / `cat`;
vouch sits behind them as a Stop-hook that does two passes on every draft:

1. **Block-check.** Ungrounded ASSERT propositions about named external
   entities fire. Before blocking, the gate scans your session's `tool_result`
   events — if a file you `Read`/`cat`-ed, a page you `WebFetch`-ed, or a
   `WebSearch` result entails the proposition (same NLI judge), it snapshots
   the content as a dossier, files the claim, and passes. You'll see
   `[verified: <id>] auto-grounded from session …`. For claims with **no
   session source**: `vouch search "<q>"` → `vouch fetch <url>` → `vouch claim
   "…" --dossier <slug>`, or hedge `(unverified, from training memory)`.
2. **Harvest.** On a passing draft, each `[inference-from:]` / `[synthesis-of:]`
   / `[interpretation:]` / `[hypothesis]` segment is filed as the matching
   claim type with the cited ids as `depends_on`. You do NOT run `vouch claim
   --type INFERENCE …` by hand for these — tag and write self-contained.

`vouch search` is for **reusing the KB** ("do I already have a dossier/claim?").
Use your native `WebSearch` to discover *new* URLs; settle on one with
`vouch fetch`. Don't use a native fetcher for anything you'll cite with a
manual `--source-quote` — its stripped text diverges from vouch's and breaks
the quote check (`--source-quote` is optional anyway; omit it and vouch
auto-selects from the dossier).

## Commands

```bash
# Find a source — KB-first, web-fallback. DuckDuckGo by default; --provider
# openalex|pubmed|arxiv|google-scholar for academic.
vouch search "<question>" [--provider <p>] [--limit 5] [--kb-only|--web-only]

# Fetch — vouch does the HTTP itself (trust boundary) AND returns the content.
# This is your web-fetch tool; the dossier persists as a side effect.
vouch fetch <url> [--fetcher arxiv|generic] [--force-refetch] [--full | --content-limit N]
vouch get-dossier <slug> --offset 8000 --limit 4000   # re-read a later window

# Claim (against an already-fetched dossier). --source-quote OPTIONAL — omit it
# and vouch auto-selects the supporting passage.
vouch claim "<text>" --type ATOMIC --dossier <slug> \
  [--source-quote "<verbatim 1–3 sentences>"] --topic <topic> \
  --attribution "<authors / org>"

# SYNTHESIS — cross-source statement (≥2 dossiers)
vouch claim "<text>" --type SYNTHESIS --topic <topic> \
  --sources '[{"dossier_slug":"…","quote":"…"},{"dossier_slug":"…","quote":"…"}]'

# INFERENCE / INTERPRETATION / HYPOTHESIS — derived claims. Usually you DON'T
# run these by hand: tag the segment and the gate harvests on a passing draft.
# This form is for filing a derivation outside a tagged draft.
vouch claim "<text>" --type INFERENCE --depends-on 854,856 --topic <topic> --soft-score 0.7
vouch claim "<text>" --type HYPOTHESIS --topic <topic> --soft-score 0.4

# Read / correct the KB
vouch list-topics
vouch list-claims --topic <X> --status supported --contains <kw>
vouch get-claim <id>
vouch chain <id>                                     # walk dependency DAG
vouch list-dossiers
vouch supersede <old_id> <new_id> --reason "<why old was wrong>"

vouch --pretty <command> ...                          # human-readable JSON
```

`vouch claim` errors and what to do:
- `quote-not-in-dossier` (you passed `--source-quote`): drop it (let vouch
  auto-pick), or copy EXACT phrasing from `vouch get-dossier <slug>`.
- `quote-not-in-dossier` with `auto_selected` (or `auto-quote: no supporting
  passage`): the dossier genuinely doesn't support the claim — fetch a
  different source or drop the claim.
- `status: unsupported` (NLI rejected): the quote is in the dossier but
  doesn't actually support your claim. Tighten the claim, pass a stronger
  `--source-quote`, or drop it.

## Tagging

Every factual segment in your response to the user carries a tag:

| Tag | When | Constraints |
|---|---|---|
| `[verified: <id>]` | Direct fact from a verified ATOMIC claim | cite claim_id |
| `[synthesis-of: <id1>, <id2>]` | Cross-source statement | cite all source IDs |
| `[inference-from: <id1>, …]` | Conclusion deduced from verified claims | scenario-anchored OR paper-derived only |
| `[interpretation: <id>]` | Substantive reframing of one claim | cite the single claim |
| `[hypothesis]` | Speculation you can't fully justify | explicit |

The gate harvests the four derived forms on a passing draft — `claim_text` is
your segment **verbatim**, `depends_on` is the cited ids, `status` is
`recorded`, and re-emitting deduplicates. `; score: 0.85` inside the tag
overrides the default `soft_score`. `[verified: <id>]` never files anything;
a dangling / non-`supported` id is flagged. Because the segment becomes the
`claim_text`:

- **No bare back-references.** Replace "this means…" / "the above implies…"
  with the actual subject so the claim reads standalone.
- **Carry the qualifier.** "for cost-constrained SaaS", "on Agriculture" —
  dropping it overreaches.

### CRITICAL — when can you tag `[inference-from:]` / `[synthesis-of:]`?

Only when one of:

1. **SCENARIO-ANCHORED** — directly answers the user's specific question,
   every step deduces from verified atomic claims.
2. **PAPER-DERIVED** — paraphrases an inference the paper authors themselves
   wrote (cite their dossier; attribution to authors).

If neither: **DOWNGRADE TO `[hypothesis]`.** Free-form "interesting connection"
is HYPOTHESIS, not INFERENCE.

When the user asks "X vs Y, which?", the most credible evidence is a quote
where the *loser* admits the weakness in their own docs — self-admissions
have more leverage than third-party benchmarks. Spend fetch budget there.

## Output

Markdown response with tags on every factual segment, plus a brief footer:

```
──────────────────
vouch: filed N atomic claims under topic "<X>". New claim_ids: [868, 869, …]
(derived [inference-from]/[synthesis-of]/[interpretation]/[hypothesis] segments
are harvested by the gate from this draft.)
```

## Corrections

If an old claim was wrong: `vouch claim "<corrected>" …` → `new_id`, then
`vouch supersede <old> <new> --reason "<why>"`. Both versions stay queryable;
the audit trail is the value. Never silently rewrite — vouch's KB records
its own corrections.

## Anti-patterns

- ❌ Don't pass `--source-quote "…"` from your own `WebFetch` read — a manual
  quote must come from `vouch get-dossier` (or omit `--source-quote` entirely
  and let vouch auto-select). Using native `WebFetch` for *research* is fine —
  the gate auto-grounds from it.
- ❌ Don't skip `vouch search` when you're about to fetch/build — it dedups
  against dossiers/claims you already have.
- ❌ Don't tag `[verified]` for things you derived but didn't fetch.
- ❌ Don't tag `[inference]` for "interesting connections" — those are
  `[hypothesis]`.
- ❌ Don't write a tagged derived segment with a bare "this" / "the above" /
  "it" — the gate files it verbatim as the `claim_text`.
- ❌ Don't re-run `vouch claim --type INFERENCE …` for a segment you already
  tagged — same-text+deps dedups, but a slightly-reworded duplicate doesn't.
- ❌ Don't fabricate claim_ids — they come from `vouch list-claims` /
  `vouch search` output.
- ❌ Don't auto-supersede other agents' claims without strong justification.
