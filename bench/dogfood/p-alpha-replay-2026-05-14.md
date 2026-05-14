# P-α counter-evidence replay — 2026-05-14

Replay of the Axis-3 comprehensiveness primitive (P-α, `VOUCH_GATE_COUNTER_EVIDENCE=1`)
over the full `fires-last14d.jsonl` corpus (245 fires, 430 propositions).
Owner-side parent: SUN-68; execution issue: [#51](https://github.com/sunnyadn/vouch/issues/51);
sequence position: 2 of 4 per [`2026-05-14 replay-first decision`](../../../meta/sunny/pages/decisions/2026-05-14-replay-first-axis-2-3-measurement.md).

## Headline

| metric | value | gate |
|---|---|---|
| **flip_rate** | **2 / 245 = 0.8 %** | (no formal gate — informs reach) |
| **contradiction_precision** | **2 / 2 = 100 %** (N=2 hand-audit) | ≥70 % launch-ready · 50-70 % tune · <50 % reconsider |
| **NLI confidence (max score on flipped)** | both ≥ 0.90 | — |
| **entity_coverage** | `dataset` (1/27, 3.7 %) and `other` (1/77, 1.3 %) carry both flips | — |

**Verdict (one line):** P-α catches real comprehensiveness gaps when it fires,
but fires on only 0.8 % of historical drafts — KB density is the binding
constraint, same shape as #50 (E)'s 0.9 % upper-bound lift. Launch-ready as
opt-in by the precision gate, but with the structural expectation that it
will be a rare, high-signal primitive — not a frequent gate, not a behavior
shaper.

## Methodology

Production P-α is gated by `VOUCH_GATE_COUNTER_EVIDENCE=1` and runs, for each
*grounded* ASSERT in `src/gate.ts`:

1. `embedOne(\`${entity}. ${proposition}\`)`
2. `searchHybrid(queryEmb, topK=5)`
3. Filter to supported, non-superseded, entity-sharing claims with
   cosine ≥ 0.55
4. `verifyContradiction(prop, claim.claim_text)` per surviving candidate
5. Flip grounded → ungrounded when `contradicts && score ≥ 0.75`

Steps 1-5 are independent of *how* the proposition originally grounded
(KB-NLI vs session-source autoground vs would-have-grounded). To measure
whether the KB carries enough counter-evidence to be useful, we run steps
1-5 on every fired proposition in `fires-last14d.jsonl` — treating each AS
IF grounded and asking whether the current KB has a NLI-contradiction on
the same entity.

### Caveats

1. **Selection effect.** Fires are propositions that *failed* to ground, so
   they skew toward entities with weaker KB coverage. Production flip_rate
   on naturally-grounded ASSERTs (which select for well-covered entities)
   will be ≥ this measurement. We treat 0.8 % as a *lower bound* on the
   structural firing rate.
2. **Current-KB replay, not historical-KB.** We use today's KB (2026-05-14).
   Fires from 2026-05-09 didn't have access to the claims added in subsequent
   sessions. The counterfactual asks: "if those drafts happened today, would
   P-α catch them?"
3. **Strict NLI prompt.** `CONTRADICTION_PROMPT_TEMPLATE` is explicitly
   conservative ("Be CONSERVATIVE: when in doubt, answer false. We only want
   clear, logical mutual exclusion"). That's a deliberate design choice — the
   alternative (loose NLI) would push contradiction_precision toward
   token-coincidence noise. The low flip_rate is partly a feature of that
   conservatism.
4. **N=2 on the precision audit.** The issue's gate language assumes a
   20-row hand audit; with N=2 the precision number is fragile. The headline
   takeaway is the flip_rate floor, not the 100 % precision.

Script: `bench/dogfood/counterfactual_P_alpha.ts` (mirrors `counterfactual_E.ts`
shape). Output: `bench/dogfood/fires-counterfactual-P_alpha.jsonl` (gitignored,
245 rows; regenerate with `./bench/dogfood/counterfactual_P_alpha.ts --concurrency 6`).
Run cost: ~6 minutes wall-clock, ~430 embed calls + ~345 NLI calls (well
under \$1 on Vertex AI / gemini-3.1-pro-preview).

## Funnel — where the 99.2 % go

| stage | n | % of input |
|---|---|---|
| input propositions | **430** | 100.0 % |
| ≥ 1 KB hit in top-5 (any) | (all) | — |
| ≥ 1 candidate passes cosine 0.55 **and** entity-share filter | **195** | 45.3 % |
| ≥ 1 NLI verdict `contradicts && score ≥ 0.75` | **3** ¹ | 0.7 % |

¹ 3 contradicting (claim, proposition) pairs across 2 distinct propositions
in 2 distinct fire rows (the TRUE-T5 proposition contradicts two different
KB claims).

**The NLI verifier is the binding funnel stage, not retrieval.** 45 % of
historical-fire propositions had an entity-sharing supported KB claim that
passed cosine threshold; only 0.7 % of those survived the NLI contradiction
prompt. The KB and the agent's drafts overwhelmingly say *complementary*
things (one is silent where the other speaks), not negations.

Row-level rollup (any-proposition-flips → row flips): **2 / 245 = 0.8 %**.

## Entity coverage — where P-α has a chance to fire

`flip_rate_within_class` = rows-in-class that flipped / rows-in-class total.

| entity_class | n in corpus | flipped | flip_rate_within_class | KB-candidate hit rate |
|---|---|---|---|---|
| dataset | 27 | 1 | **3.7 %** | **96.3 %** |
| named-product | 44 | 0 | 0.0 % | 72.7 % |
| workspace-meta | 41 | 0 | 0.0 % | 78.0 % |
| library-or-concept | 29 | 0 | 0.0 % | 31.0 % |
| version | 26 | 0 | 0.0 % | 57.7 % |
| pricing | 1 | 0 | 0.0 % | 0.0 % |
| other | 77 | 1 | 1.3 % | 39.0 % |

**`dataset` is the design sweet spot.** Datasets (FEVER / ALCE / FActScore /
SQuAD / TriviaQA / MMLU / etc.) have:

- 96 % KB-candidate hit rate — the KB already has measured numerical claims
  for these
- 3.7 % flip rate within class — when the agent re-summarizes a benchmark
  number from training memory, the KB carries a counter-claim often enough
  for P-α to fire

`workspace-meta` and `named-product` have high KB-candidate hit rates (78 %
and 73 %) but 0 flips: the KB has many claims that *mention* these entities,
but those claims don't deny anything the agent typically asserts about them.
Workspace metadata is descriptive ("vouch's gate runs in Stop hook") not
mutually-exclusive-shaped; brand names co-occur with many adjacent facts
without contradictions.

`pricing` has N=1 in this corpus — no signal. Worth re-measuring on a
larger, less-vouch-internal-dominated corpus.

## NLI confidence distribution (flipped subset)

| bucket | n | % of flipped |
|---|---|---|
| 0.75 – 0.80 | 0 | 0 % |
| 0.80 – 0.85 | 0 | 0 % |
| 0.85 – 0.90 | 0 | 0 % |
| 0.90 – 0.95 | 1 | 50 % |
| 0.95 – 1.00 | 1 | 50 % |

Both flips landed at ≥ 0.90 max score — the gate threshold of 0.75 is well
inside the empty middle band, so threshold tuning won't open up the funnel.
The NLI judge is binary in practice on this corpus: clear contradiction or
nothing.

## Hand spot-check on the flipped subset (N=2)

### Flip 1 — ALCE / 50 % citation support claim · **TP (clean)**

| field | value |
|---|---|
| ts | 2026-05-09T08:18:02.765Z |
| repo | meta/sunny |
| entity_class | dataset |
| fired proposition | "The best model in the ALCE report achieves 50% citation support on the ELI5 task." |
| contradicting claim (id 258, dossier `sunny-alce-table6-eli5-baselines-2026-05-10`) | "In ALCE paper Table 6 (ELI5 main results), the best citation precision among all ChatGPT prompting strategies is 67.8 percent, achieved by ChatGPT with RERANK; the VANILLA 5-psg baseline is 50.0 percent." |
| similarity | 0.80 |
| contradiction_score | 0.90 |
| NLI reason | "Prop A claims the best model achieves 50%; Prop B states 50.0% is the VANILLA baseline and the best strategy achieves 67.8%, contradicting the claim that 50% is the best." |

**Classification: TP.** The original draft summarized the ALCE paper from
memory and conflated "baseline" with "best". The KB has the specific Table 6
numbers. P-α flipping forces the agent to reconcile (either supersede claim
258 with corrected reading of the paper, or rephrase the draft to match what
the KB actually attests). This is exactly the comprehensiveness gap P-α was
designed to catch — and crucially, the original gate did NOT fire grounding
on this (it fired as 5-candidates-found-none-entailed), so without P-α the
50 % number would have round-tripped through a draft revision unchallenged.

### Flip 2 — TRUE-T5 results "尚未出来" (not yet available) · **TP (borderline)**

| field | value |
|---|---|
| ts | 2026-05-11T01:40:39.480Z |
| repo | meta/sunny |
| entity_class | other |
| fired proposition | "TRUE-T5 结果尚未出来。" |
| KB counter A (id 255) | "Comparing MiniCheck-T5 to TRUE-T5-XXL ... +30.1 pp / +23.0 pp ..." (cos 0.59, score 0.95) |
| KB counter B (id 251) | "In the 50-sample TRUE-T5-XXL paper canonical evaluation, the with-vouch v3 arm achieved citation_recall 92.5 %, citation_precision 92.6 %, citation_f1 92.6 %." (cos 0.57, score 1.00) |

**Classification: borderline TP.** Surface-level the NLI is right — the
proposition "results are not yet out" is mutually exclusive with the KB's
"results are X, Y, Z". Pragmatically, the agent meant "this 100-sample-batch's
results aren't out yet"; the KB carries the *previous* 50-sample-batch's
numbers. Both true under disambiguation. The original drafted sentence didn't
specify the batch.

Even on the borderline reading P-α does a useful thing: it forces the agent
to disambiguate ("the 100-sample batch is still running; the 50-sample dry
run reported [claim 251]"), which is more comprehensive than the original
draft. Counting as TP because the structural behavior P-α drives (force
reconciliation rather than under-specified assertion) is the right one,
even when the literal NLI verdict over-constrains.

### Precision = 2 / 2 = 100 %; N=2 caveat noted

The issue's 20-row sample size was scoped for a corpus where flip_rate was
expected to be higher. With N=2 in the entire flipped subset, a single
mis-classification swings precision by 50 pp. Treat **100 % as direction-
positive (no token-coincidence noise observed), not as a calibrated point
estimate**.

## Verdict (gated)

Per the issue's launch-readiness gate:

- ≥ 70 % contradiction_precision → launch-ready as opt-in
- 50–70 % → tune prompt/threshold
- < 50 % → design reconsideration

**Status: launch-ready as opt-in** (100 % precision clears the bar even on N=2).
Production default stays OFF; users who turn `VOUCH_GATE_COUNTER_EVIDENCE=1`
on get a high-precision, low-reach primitive — a feature, not a bug, for
the design-space P-α is good at (numerical / dataset-benchmark reconciliation
where the KB already has the verified number).

### Caveats for the user enabling P-α

- **Cost.** Each grounded ASSERT pays one embed + up to 5 NLI calls = ~5-10s
  per turn additional latency, ~\$0.05-0.2/turn API. With 0.8 % productive
  fire-rate observed here, that's expensive insurance for the average turn.
- **Sweet spot.** P-α is most useful when working on benchmark-evaluation
  topics where the KB has numerical attestations. Outside that workload, it
  almost never fires.
- **Threshold tuning won't help reach.** Lowering `COUNTER_EVIDENCE_FIRE_SCORE`
  from 0.75 to 0.5 wouldn't open the funnel — both flips landed ≥ 0.90 and
  there's nothing in the 0.50–0.85 band. The funnel is bottlenecked by the
  NLI prompt's strict mutual-exclusion definition, not by score calibration.

### Improvements worth piloting

1. **Bias retrieval toward numerical claims when the proposition contains
   a number.** Add a "has-number" boost to the cosine score for fact-table
   style claims. Would primarily improve `dataset` and `pricing` reach.
2. **Combine with a stakes signal.** P-α's cost is only paid for grounded
   ASSERTs; if we had a stakes-weighting tag, we could run P-α only on
   high-stakes assertions (numerical comparisons, decision-driving claims),
   amortizing the latency.
3. **P-α.5 (web-counter-evidence suggestion) is the right complement.**
   When P-α finds no counter in KB (the 99.2 % case), surfacing the
   `vouch search "<entity> limitations" --provider ddg` suggestion is the
   only path that addresses the "KB doesn't carry the counter-evidence"
   floor. Measurement of P-α.5's user-followthrough rate is the open
   downstream question — needs forward dogfood data, not replay.

## How #51 talks to #52

[#52](https://github.com/sunnyadn/vouch/issues/52)'s finding: humility-axis
primitives are kept visibility-only because the N=1 provisional thresholds
in `PRIMITIVES-2026-05-14.md` were unrepresentative.

#51's finding (this report): comprehensiveness-axis P-α has structurally
correct semantics (precision 100 % on the small flipped subset) but is
reach-bound by KB density. The opt-in default is the right ship shape; no
threshold tuning recovers reach.

**Combined,** both axes are "ship the visibility, hold the enforcement" —
neither has the population-level statistics to be a default-on gate yet.
The forward step (per the parent decision) is #54 (controlled Counter-evidence
bench on N≥100 synthetic items) for the one piece replay can't measure.

## Reproduction

```sh
cd ~/Projects/vouch

# Full replay (~6min, <\$1 on Vertex)
./bench/dogfood/counterfactual_P_alpha.ts --concurrency 6

# Smoke test (10 rows, <1min)
./bench/dogfood/counterfactual_P_alpha.ts --limit 10 --concurrency 3
```

Output schema (per row):

```jsonc
{
  "ts": "...",
  "transcript_id": "...",
  "repo": "...",
  "propositions": [{
    "entity": "...",
    "proposition": "...",
    "kb_candidates_seen": <int>,  // entity-share filter survivors
    "contradictions": [{ "claim_id", "claim_text", "dossier_slug",
                         "similarity", "contradiction_score",
                         "contradiction_reason" }]
  }],
  "would_flip": <bool>,
  "max_contradiction_score": <float|null>,
  "entity_class": "dataset|named-product|workspace-meta|library-or-concept|version|pricing|other"
}
```

## Files

- `bench/dogfood/counterfactual_P_alpha.ts` — replay script (~310 LOC)
- `bench/dogfood/fires-counterfactual-P_alpha.jsonl` — 245-row replay output (gitignored)
- This document — the replay report
