# humility-baseline — historical distribution of humility-ratio + blind-spot signals

**Date:** 2026-05-14.
**Issue:** [#52](https://github.com/sunnyadn/vouch/issues/52).
**Scope:** offline replay of the humility-ratio + blind-spot parsers shipped
in commits `7b3418b` (P-γ Stage 1), `c91a33b` (P-γ Stage 2 over-confidence
flag), `5d72fc4` (P-γ.5 blind-spot enumeration) across all historical
sessions reachable from this machine.

## Verdict

**The N=1 provisional range from `PRIMITIVES-2026-05-14.md` is unrepresentative.**
Across 246 historical sessions, median blind-spots-per-turn is 0; only the
post-2026-05-13 sessions surface gaps at non-trivial rates — pre-convention
agents almost never enumerated what they didn't know in natural prose.
Humility-ratio has only N=2 sessions with enough stance data to compute
(21.2% and 48.3%) — already wider than the provisional 10–25% band, with
the high value coming from a meta-vouch discussion that *should* hedge
heavily. The 10–25% / ≥1-per-heavy-turn range describes an aspiration the
parsers ship to *create*, not a baseline the population currently occupies.

## TL;DR for the dashboard

| metric | provisional (N=1) | empirical (N≥10) | revised |
|---|---|---|---|
| humility_pct healthy band | 10–25% | N=2 observations: {21.2%, 48.3%} | **HOLD** — collect more, expand to ~15–40% pending data |
| over-confidence floor | ≤5% | no observations ≤5% in usable stance data | unchanged for now |
| over-hedging ceiling | ≥40% | 1/2 observed above this | **40% is reachable on legitimate meta-discussion content; should be a *flag-and-explain*, not a *block*** |
| blind_spots_per_turn healthy | not specified | p50=0, p75=0.01, p90=0.032 | populations-of-record are at 0 — any non-zero is signal |
| blind_spots_per_heavy_turn (DB) | ≥1 | N=2 observations: {0, 3.25} | provisional threshold is right *direction* but a single-session observation either way is too noisy |

## Methodology

The two parsers under test:

- `getSessionFireCounts(transcript_id)` (`src/store.ts:987`) — SQL over
  `session_claims.stance`. Reports `humility_pct = (HEDGE+SPECULATE) /
  (ASSERT+HEDGE+SPECULATE)`. Stance is LLM-extracted **at gate time** —
  cannot be rerun offline without re-invoking the extractor across every
  draft. So humility is only available for sessions vouch actually
  processed.

- `countBlindSpots(draft)` (`src/gate.ts:2111`) — pure regex (1 explicit
  `[gap: …]` marker pattern + 7 natural-language phrase patterns). Runs
  offline on any text.

Two input sources:

1. **Claude Code .jsonl transcripts** at `~/.claude/projects/<dir>/*.jsonl`
   — 310 transcripts found. Assistant-turn text extracted via the per-turn
   `message.content[].text` blocks. Stance pulled from `session_claims`
   when present (3 transcripts).

2. **Meta vault `.txt` transcripts** at `~/Projects/meta/sunny/pages/inbox/
   transcripts/*.txt` — 27 transcripts; assistant turns parsed by `###
   assistant […]` headers. Stance unavailable (predate vouch).

3. **Excluded:** synthetic `vouch-bench-ctx-*` rows in `session_claims`
   (vouch-internal test fixtures, no real prose) — these are the bench
   harness's stub sessions used during gate-logic regression testing.

**Working set:** 246 sessions with ≥10 assistant turns AND ≥1000 words
(or stance-available with ≥5 truth-bearing claims). Sessions below this
floor are too small for stable per-turn ratios. See
[`humility_baseline.ts`](humility_baseline.ts) for the aggregator and
[`humility_distribution.ts`](humility_distribution.ts) for the
distribution / histogram code. Raw output checked in as
[`humility-baseline-2026-05-14.jsonl`](humility-baseline-2026-05-14.jsonl)
(regenerate via `bun run bench/dogfood/humility_baseline.ts >
bench/dogfood/humility-baseline-2026-05-14.jsonl`).

### Heavy-turn count: two definitions

The provisional threshold from PRIMITIVES says "≥1 blind-spot per heavy-claim
turn (3+ ASSERTs)". "ASSERTs per turn" is per-turn data the **LLM extractor**
produces; we have it from `session_claims.turn_idx` **only for the 3 sessions
vouch processed**. For other sessions we approximate via
`namedEntitySentenceCount ≥ 3` per assistant message — but this proxy is
much *looser* than the real gate definition (capitalized-noun-phrase
sentences are common; ≥3-ASSERT-extracted turns are rare).

We report both in the per-session table (`hvy_db` vs `hvy_px`) and treat them
as distinct metrics. The proxy is reported for completeness and to give a
crude population baseline, but **only the `_db` numbers should drive
threshold decisions**.

## Distributions (working set: N=246)

```
blind_spots_per_turn (all sessions): N=246
  min=0  max=0.4
  p10=0  p25=0  p50=0  p75=0.01  p90=0.032
  mean=0.011

blind_spots_total (all sessions): N=246
  min=0  max=13
  p10=0  p25=0  p50=0  p75=1  p90=2
  mean=0.667

heavy_turns_db (DB-derived, gate sessions only): N=4
  min=0  max=4   p50=1.5

blind_spots_per_heavy_turn_db (gate sessions only): N=2 usable
  values = { 3e60f5cf: 3.25,  9606fa26: 0 }

blind_spots_per_heavy_turn_proxy (all sessions): N=246
  min=0  max=0.571  p50=0  p75=0.016  p90=0.065

humility_pct (gate sessions w/ truth-bearing ≥ 10): N=2 usable
  values = { 3e60f5cf: 21.2%,  9606fa26: 48.3% }
```

### Histogram: blind_spots_per_turn (N=246)

```
    0.000    | ████████████████████████████████████████ 170
  0.001-0.01 | ███ 13
  0.01-0.025 | ███████ 28
  0.025-0.05 | ████ 19
  0.05-0.10  | ████ 15
  0.10-0.25  |  0
  ≥ 0.25     |  1
```

**Reading:** 170 of 246 sessions (69%) surface **zero** blind-spots across
the entire session. 28 more sessions surface fewer than 1 per 40 turns. Only
1 session exceeds 0.25 (session `16f288ca`, May 6, research, 10 turns w/ 4
gap phrases — 0.4 bs/turn, atypical).

This shape is the headline: **pre-2026-05-13, agents did not enumerate gaps
in natural prose.** The blind-spot parser ships with 7 natural-phrase regexes
("I haven't verified", "worth checking", "open question", …) and these
phrases existed before the convention — they just weren't deployed
frequently. The `[gap: …]` explicit-marker pattern was introduced 2026-05-14
(commit `5d72fc4`) and only sessions after that authored explicit markers
(13 explicit markers in the 539-turn 3e60f5cf ship session; none elsewhere).

### Pre- vs post-convention split

| split | N | sessions w/ ≥1 blind-spot | p90 bs/turn | max bs/turn |
|---|---|---|---|---|
| before 2026-05-13 | 241 | 73 (30%) | 0.033 | 0.4 |
| 2026-05-13 onward | 5 | 3 (60%) | 0.024 | 0.031 |

Post-convention sample is too small to show a behavior shift yet, but the
proportion-of-sessions-with-any-gap roughly doubled. This is the
pre-/post-ship gap the dogfood loop needs to widen.

### Histogram: humility_pct (N=2 — both points plotted)

```
  0-5%   |
  5-10%  |
  10-20% |
  20-30% | ████ 1  (3e60f5cf, 21.2%, May 14 ship session)
  30-40% |
  40-50% | ████ 1  (9606fa26, 48.3%, May 14 meta-vouch chat)
  ≥ 50%  |
```

The `9606fa26` outlier is informative: that session was a discussion
*about* vouch's primitives, where every other sentence is reasonably
hedged because the subject is unfinished work. **High humility on a
meta-vouch session is appropriate, not pathological** — which is the
exact reason the README threat model says to avoid blocking on humility
percentiles.

## Per-content-tag breakdown (working set)

The provisional threshold was set on a meta-vouch session; the issue called
out that content type matters. Empirical per-tag percentiles:

```
tag        | n   | bs_pt p25 | bs_pt p50 | bs_pt p75 | bs_pt p90 | bs_pt max | hvy_px p50
-----------|-----|-----------|-----------|-----------|-----------|-----------|-----------
code       | 69  |     0     |     0     |   0.011   |   0.037   |   0.095   |     31
meta-vouch | 12  |     0     |     0     |   0.024   |   0.054   |   0.070   |     13
research   | 97  |     0     |     0     |   0.011   |   0.029   |   0.400   |     20
strategy   | 68  |     0     |     0     |     0     |   0.017   |   0.083   |     16
```

**Reading:**

- **`meta-vouch`** sessions have the highest p75 and p90 — agents talking
  *about* a verification tool naturally enumerate more uncertainty. Highest
  among the four tags. Consistent with the original provisional being
  drawn from this category.
- **`code`** sessions have surprisingly comparable p75 — code edits include
  "I haven't tested", "worth checking the type of…", "open question
  whether…". Same regex catches.
- **`research`** has the widest spread — one outlier (`16f288ca`) at 0.4
  bs/turn pushes the max, but p50/p75 are similar to code.
- **`strategy`** is the *lowest* — p75 is 0. Strategy/decision drafts
  hedge inline ("might", "perhaps") rather than enumerating gaps explicitly;
  the gap regex is biased toward verification-style language.

**Implication for per-tag thresholds:**

The per-tag p75 ranges from 0 (strategy) to 0.024 (meta-vouch) — a 2.4×
spread. Setting a single global "healthy ≥ X bs/turn" threshold will
under-fit one tag and over-fit another. Either:
- accept that the threshold is meta-vouch-tuned and noise on others, or
- ship a `--content-tag` flag and tag-aware thresholds.

(Recommendation: keep visibility-only for now; bake tag-aware enforcement
only after one more measurement cycle, per Layer 1 / Layer 2 / Layer 3
discipline from `PHASE1-RESULTS.md`.)

## Per-session table (working set, top by date, all sessions w/ stance)

The full 246-row table is in
[`humility-baseline-2026-05-14.distribution.txt`](humility-baseline-2026-05-14.distribution.txt)
(regen via `bun run bench/dogfood/humility_distribution.ts
bench/dogfood/humility-baseline-2026-05-14.jsonl`). Sample below shows:
all 4 stance-available sessions + top 18 by recency to demonstrate the
per-session shape.

| date | source | tag | turns | words | bs | bs/turn | hvy_db | bs/hvy_db | hvy_px | bs/hvy_px | hum% | sess |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-05-14 | jsonl | meta-vouch | 539 | 63910 | 13 | 0.024 | **4** | **3.25** | 162 | 0.080 | **21.2%** | 3e60f5cf |
| 2026-05-14 | jsonl | research | 65 | 6200 | 2 | 0.031 | 0 | — | 28 | 0.071 | — | 363d4d47 |
| 2026-05-14 | jsonl | meta-vouch | 50 | 4937 | 0 | 0 | **3** | **0** | 13 | 0 | **48.3%** | 9606fa26 |
| 2026-05-14 | jsonl | meta-vouch | 28 | 1888 | 0 | 0 | 0 | — | 5 | 0 | — | 7843c561 |
| 2026-05-13 | jsonl | research | 142 | 17478 | 1 | 0.007 | — | — | 36 | 0.028 | — | 1213c543 |
| 2026-05-12 | jsonl | meta-vouch | 32 | 4744 | 0 | 0 | — | — | 8 | 0 | — | 425c4170 |
| 2026-05-12 | jsonl | code | 15 | 5197 | 0 | 0 | — | — | 15 | 0 | — | 4dad5b46 |
| 2026-05-11 | jsonl | meta-vouch | 270 | 19461 | 0 | 0 | — | — | 64 | 0 | — | 203e37a5 |
| 2026-05-11 | jsonl | meta-vouch | 43 | 14174 | 3 | 0.070 | — | — | 40 | 0.075 | — | f8b3b913 |
| 2026-05-10 | jsonl | meta-vouch | 92 | 6740 | 5 | 0.054 | — | — | 23 | 0.217 | — | 56875d7b |
| 2026-05-10 | jsonl | meta-vouch | 33 | 2058 | 2 | 0.061 | — | — | 7 | 0.286 | — | 47471b9f |
| 2026-05-09 | jsonl | strategy | 120 | 9455 | 0 | 0 | — | — | 30 | 0 | — | f48166d8 |
| 2026-05-08 | jsonl | research | 532 | 52480 | 1 | 0.002 | — | — | 133 | 0.008 | — | 2dc0f73d |
| 2026-05-06 | jsonl | research | 10 | 1495 | 4 | 0.400 | — | — | 7 | 0.571 | — | 16f288ca |
| 2026-05-05 | jsonl | code | 21 | 5981 | 2 | 0.095 | — | — | 18 | 0.111 | — | 6eeba5e1 |
| 2026-05-04 | jsonl | code | 239 | 61860 | 3 | 0.013 | — | — | 210 | 0.014 | — | 3d62c7d3 |
| 2026-05-01 | txt | research | 122 | 14956 | 4 | 0.033 | — | — | 52 | 0.077 | — | 2026-05-01 |
| 2026-05-01 | txt | research | 45 | 5331 | 2 | 0.044 | — | — | 26 | 0.077 | — | 2026-05-01 |
| 2026-05-01 | txt | research | 8 | 1818 | 2 | 0.250 | — | — | 5 | 0.400 | — | 2026-05-01 |
| 2026-04-30 | txt | research | 116 | 15179 | 3 | 0.026 | — | — | 79 | 0.038 | — | 2026-04-30 |
| 2026-04-30 | txt | research | 101 | 13978 | 7 | 0.069 | — | — | 59 | 0.119 | — | 2026-04-30 |
| 2026-04-29 | txt | research | 109 | 13579 | 0 | 0 | — | — | 79 | 0 | — | 2026-04-29 |

Bolded cells in the top row are the stance-available data points.

### Reading the stance-available rows

- **3e60f5cf (May 14 ship session, 539 turns):** humility 21.2% sits inside
  the provisional band. blind-spots-per-heavy-turn-DB = 3.25 → comfortably
  above the ≥1 threshold. This is the session the provisional was tuned on
  — it passes itself by construction. The proxy heavy-turn count (162) vs
  DB heavy-turn count (4) shows how loose the proxy is: a 40× ratio.
- **9606fa26 (May 14 meta-vouch chat, 50 turns):** humility 48.3% — *above*
  the provisional 40% over-hedging ceiling. But zero `[gap:]` markers and
  zero natural-phrase gaps even though the agent hedged heavily. This is
  the legitimate over-hedging shape: the topic itself is unfinished, so
  every claim is rightly qualified — but the agent isn't separately
  enumerating *what they didn't check*. **The two humility signals
  (per-claim hedging vs per-turn gap enumeration) are genuinely
  independent, as the P-γ.5 commit message asserts.**
- **363d4d47 (May 14 research, 65 turns):** truth-bearing = 5 — too few
  stances to compute a meaningful humility ratio (the parser uses ≥10
  threshold). Useful for blind-spot data only.
- **7843c561:** truth-bearing = 2 — same.

## Revised thresholds (proposed)

Empirical p25–p75 as "healthy" would set the blind-spot band at **0
bs/turn**, which is useless: the population is currently at zero. The
parsers are *visibility-only* primitives — they exist to surface the
metric and let *user pushback* be the forcing function (per the P-γ Stage
1 commit message). So:

- **blind_spots_per_turn:** keep the metric visible; don't enforce a
  floor on it. The pre-/post-convention split is the right framing for
  measuring whether the parser *changes behavior*, not whether
  individual sessions hit a threshold. Forward-looking metric: % of
  weekly sessions with ≥1 explicit `[gap:]` marker should rise from the
  current 1/246 (just 3e60f5cf) toward 50%+ by 2026-06-14 if the
  convention takes hold.

- **humility_pct band:** the N=1 provisional 10–25% was anchored on one
  session. The N=2 sample already busts the ceiling. Widen to **10–40%
  as the visibility-only band, and 40–60% as a soft "over-hedging,
  inspect" zone** (not a hard cap). Below 5% remains the
  over-confidence flag (no empirical contradiction).

- **blind_spots_per_heavy_turn_db ≥ 1:** the N=2 sample (0, 3.25) is too
  small to confirm or refute. Defensible as a visibility-only flag for
  now; revisit after one more measurement cycle.

- **Over-confidence flag (P-γ Stage 2):** the trigger condition (3+
  ASSERTs this turn AND 0 hedges AND 0 gaps) reflects per-turn structure
  not session-aggregate; this measurement doesn't speak to its calibration.
  Per-turn data would need a separate pass that joins per-turn ASSERT
  counts with per-turn draft text — out of scope for this baseline.

## Caveats (own these in the verdict)

1. **N=2 for the actual humility distribution.** All historical
   pre-2026-05-12 sessions have no stance data because vouch didn't run
   over them. Adding meaningful breadth requires either (a) re-running the
   LLM extractor on every assistant message across the 27 meta-vault
   transcripts and 280+ Claude Code transcripts (cost: ~$10-30, ~30
   minutes) or (b) waiting for forward-looking dogfood data to accumulate.

2. **Proxy heavy-turn count is much looser than the gate's real
   definition.** `namedEntitySentenceCount ≥ 3` fires on most technical
   paragraphs — the DB/proxy ratio in the ship session is 4 vs 162. Any
   per-heavy-turn threshold should be evaluated on `_db` data, not `_proxy`,
   even though the proxy gives more population coverage.

3. **Content tagging is heuristic.** Path patterns + content keywords. A
   strategy session that quotes a lot of code will be tagged "code"; a
   meta-vouch session that doesn't mention "gate/claim/dossier" might miss
   the tag. The per-tag breakdown should be read as directional, not
   precise.

4. **Pre-convention transcripts predate the `[gap: …]` convention.** The
   regex was *introduced* on 2026-05-14. Historical data measures only
   natural-phrase gaps, not the explicit-marker convention. A baseline
   measured against a metric whose enabling convention didn't exist yet
   is necessarily a lower-bound, not a steady-state.

5. **"Per-turn" semantics differ between data sources.** For Claude Code
   `.jsonl`, an assistant turn is a `type=assistant` message — a single
   gate fire can be spread across multiple consecutive messages (tool
   calls + final text). For meta-vault `.txt`, an assistant turn is one
   `### assistant […]` block. Per-turn rates are not perfectly comparable
   across sources; aggregate session-level rates are.

## Next steps (out of scope here)

- Once 1–2 more weeks of post-convention sessions accumulate, redo this
  measurement and check whether the post-2026-05-13 lift (1/5 → 50%+
  goal) is materializing.
- If running the extractor offline on the meta vault transcripts is
  desired (cost: ~$10-30), wire a `bench/dogfood/extract_stances.ts` that
  invokes the extractor prompt over each turn's draft text. That would
  give N≈30 humility data points instead of N=2. Filed informally; not
  prioritized unless N=2 → N=4 over the next 2 weeks remains insufficient.

## One-line verdict

> The provisional 10–25% humility / ≥1-per-heavy-turn ranges were derived
> from N=1 and are not representative: pre-convention sessions surface
> zero blind-spots at the median across 246 sessions, and the two
> humility-eligible data points (21.2%, 48.3%) already span outside the
> provisional band — keep the parsers visibility-only and re-measure in
> 2 weeks before tightening any threshold into enforcement.
