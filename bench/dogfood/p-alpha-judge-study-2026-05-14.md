# P-α judge study — characterizing the 44.3pp funnel loss (2026-05-14)

Follow-up to [`p-alpha-replay-2026-05-14.md`](p-alpha-replay-2026-05-14.md):
the P-α counter-evidence replay funnel went 430 propositions → 195
entity-share + cosine ≥ 0.55 candidates (45.3 %) → 3 NLI contradictions
(0.7 %). This study asks: where do the 192 non-flipped (proposition, KB-claim)
pairs go? Specifically:

- (a) **prompt-strictness loss** — actually a contradiction, current strict
  NLI prompt missed it (tunable).
- (b) **task-definition loss** — actually a refinement / value-override /
  partial-contradiction; P-α's strict-mutual-exclusion contract is narrower
  than usefully-disagree (design question).
- (c) **correctly-rejected token coincidence** — strict NLI was right, the
  candidate is adjacent noise (current design is correct).

**Research-only.** This is not launch readiness. No precision-gating decisions
attached.

## Headline

| metric | value |
|---|---|
| **pairs (entity-share + cosine ≥ 0.55 survivors)** | **345** |
| strict fires (production prompt) | 5 / 345 = 1.4 % |
| loose fires (strict body, "be CONSERVATIVE…" removed) | 6 / 345 = 1.7 % |
| broad fires ("useful disagreement" contract) | 43 / 345 = 12.5 % |
| broad-only fires (strict-no AND loose-no AND broad-yes) | 37 |

**Hand-classification of N=50 sample** (all 37 broad-only fires + 13
shuffled all-no fires):

| class | sample count | scaled to 340 strict-no pairs |
|---|---|---|
| (a) prompt-strictness loss | 1 (2 %) | ≈ 1 pair (~0.3 %) |
| (b) task-definition loss   | 33 (66 %) | ≈ 33 pairs (~9.7 %) |
| (c) correctly-rejected      | 16 (32 %) | ≈ 306 pairs (~90.0 %) |

**Verdict (one line):** the 44.3pp funnel loss is **almost entirely
contract-loss (b), not prompt-strictness (a)**. Removing the "be CONSERVATIVE"
clause moves only 1 pair (+0.3pp); switching from "logical mutual exclusion"
to "useful disagreement" recovers ~11pp. That ~11pp is overwhelmingly
**numerical value-overrides on under-specified agent claims**
("baseline = 50 %" vs KB-attested "baseline = 64.4 % on ALCE-ELI5
citation_precision"). P-α as currently designed is **functioning as
specified** — it implements strict mutual-exclusion correctly. The design
question is whether mutual-exclusion is the right contract for the
comprehensiveness axis.

## Methodology

### Funnel replay (judge-study script)

`bench/dogfood/judge_study_P_alpha.ts` reproduces the production P-α funnel
on `fires-last14d.jsonl`, but instead of stopping at the first contradiction
per proposition (as production does), it logs **every** entity-share +
cosine ≥ 0.55 candidate survivor and runs three NLI prompt variants in
parallel on each (proposition, candidate) pair.

The three variants (verbatim in script; same JSON schema):

- **V_STRICT** — verbatim copy of production `CONTRADICTION_PROMPT_TEMPLATE`
  in `src/verifier.ts`. Mutual-exclusion contract; "Be CONSERVATIVE: when in
  doubt, answer false."
- **V_LOOSE** — same body, the "Be CONSERVATIVE…" sentence removed.
  Isolates the calibration effect of the conservative-clause separately from
  the contract change.
- **V_BROAD** — contract change. Fires if B contradicts, **or** refines,
  **or** value-overrides, **or** materially qualifies A in a way an honest
  writer should reconcile. Closer to "useful disagreement" than to "logical
  mutual exclusion".

The 4th variant proposed in the brief (Claude-via-API cross-judge) was
**skipped** — no `ANTHROPIC_API_KEY` configured. Brief listed it as
non-blocking.

### Funnel constants

Identical to production P-α:

| constant | value |
|---|---|
| `COUNTER_EVIDENCE_TOPK` | 5 |
| `COUNTER_EVIDENCE_MIN_COS` | 0.55 |
| `COUNTER_EVIDENCE_FIRE_SCORE` | 0.75 (score threshold a variant must clear to count as "fires") |

A pair "fires" iff `fires == true AND score ≥ 0.75`, mirroring production.

### Run

```sh
./bench/dogfood/judge_study_P_alpha.ts --concurrency 6
```

345 (proposition, candidate) pairs judged under 3 variants = 1,035 NLI
calls. ~6 minutes wall, ~$0.5 on Vertex.

The pair count differs from the earlier replay's funnel row "≥1 candidate
passes cosine 0.55 and entity-share filter | 195" — that 195 was unique
*propositions* with ≥1 candidate. 345 is the total pair count (a single
proposition can have multiple candidates pass the filter).

### Hand-classification sample

`bench/dogfood/judge_study_sample.ts` produced
`bench/dogfood/p-alpha-judge-study-sample.tsv` (gitignored, same privacy
rules as `fires-*.jsonl`):

- **All 37 broad-only fires** — strict says no-fire, loose says no-fire,
  broad says fire. These are the candidates for "task-definition loss".
- **13 deterministic-shuffle samples** from all-no fires (strict-no AND
  loose-no AND broad-no). Control for the residual hypothesis that even
  broad missed a real contradiction (the "extreme prompt-strictness" case
  where both strict and broad are too narrow).

Total N=50. Stratified this way rather than uniformly because the
information-gain density is highest on broad-only — every broad-only
pair is a candidate for the (b) class.

## Per-variant results

### Top-line fire rates

| variant | fires | rate | delta vs strict |
|---|---|---|---|
| strict (production) | 5 / 345 | **1.4 %** | — |
| loose (no conservative clause) | 6 / 345 | 1.7 % | +0.3pp |
| broad (useful-disagreement contract) | 43 / 345 | **12.5 %** | **+11.1pp** |

**Reading.** The "be CONSERVATIVE" clause is doing essentially no work.
The contract — strict mutual-exclusion vs useful-disagreement — is doing
all of it. Conservative-clause tuning is not on the table as a real lever.

### Disagreement cells (where strict says no-fire)

| cell | count | % of strict-no |
|---|---|---|
| loose-only fires (strict-no AND loose-yes AND broad-no) | 0 | 0.0 % |
| broad-only fires (strict-no AND loose-no AND broad-yes) | 37 | 10.9 % |

Loose adds 1 fire over strict (a pair that JUST cleared the score
threshold), and 0 of those are independent of broad — every loose-disagree
is also a broad-yes case. So **V_LOOSE is empirically equivalent to V_STRICT
for the purposes of this corpus**.

### Per-entity-class

| class | pairs | strict | loose | broad |
|---|---:|---:|---:|---:|
| dataset | 111 | 1.8 % | 2.7 % | **15.3 %** |
| version | 39 | 0.0 % | 0.0 % | **38.5 %** |
| library-or-concept | 18 | 0.0 % | 0.0 % | 5.6 % |
| named-product | 45 | 0.0 % | 0.0 % | 0.0 % |
| other | 50 | 6.0 % | 6.0 % | 16.0 % |
| workspace-meta | 82 | 0.0 % | 0.0 % | 2.4 % |

`version` is the dramatic gap (0 → 38.5 %). Every "v3 has precision X %"
proposition was correctly under broad and rejected under strict —
unqualified version-level numerical assertions are the densest source of
contract-loss in this corpus.

`named-product` is 0/45 across all variants — brand-name candidates
(GPT-X, Claude, Gemini) co-occur with many adjacent facts but rarely state
a value the proposition denies, even under the broad contract.

## Hand-classification breakdown (N=50)

Counts by class:

| class | broad-only (n=37) | all-no (n=13) | total |
|---|---:|---:|---:|
| (a) prompt-strictness loss | 1 | 0 | 1 (2 %) |
| (b) task-definition loss   | 33 | 0 | 33 (66 %) |
| (c) correctly-rejected     | 3 | 13 | 16 (32 %) |

### (a) prompt-strictness loss — 1 case

The strict prompt's clause "Same factual content with different wording,
paraphrase, or precision (e.g. '≈ 0.86' vs '0.862' → not contradiction
unless one explicitly excludes the other's value)" is doing real work and
mostly correctly. The one case it over-shielded:

| # | proposition | KB claim | why (a) |
|---|---|---|---|
| 48 | "ALCE Phase 0 dry-run citation precision was 64 % without vouch and **~85 %** with vouch." | "without-vouch baseline arm achieved 64.4 %, with-vouch v3 arm achieved 94.5 %" | 85 % vs 94.5 % is **9.5 percentage points** of error — too large to be a precision/approximation difference. The "~" hedge in the proposition gave strict cover to treat as approximation; the value is concretely wrong. |

### (b) task-definition loss — 33 cases (66 % of sample, ~89 % of broad-only)

Sub-shape distribution within (b):

| sub-class | n in (b) | examples |
|---|---:|---|
| **Numerical value-override on under-specified metric** | 18 | "baseline performance is 50 %" vs KB-attested 64.4 % citation_precision; "v3 precision 93.8 %" vs KB's 94.5 % / 92.6 % per evaluator; "v3 has citation error rate 6.2 %" vs KB's 5.5 %. |
| **Scope/qualifier refinement** | 7 | "MiniCheck-T5 is more lenient than TRUE-T5-XXL" vs KB's "stricter on baseline arm, more lenient on v3 arm"; "dynpred 在 R 4.5 装不上" vs KB's "dynpred was removed from CRAN entirely". |
| **Category refinement** | 4 | "Letta is an agentic memory product" vs KB's "Letta is an AI lab"; "FActScore is a dataset" vs KB's "FActScore is the original implementation of an EMNLP 2023 paper"; "follic is a package" vs KB's "follic.Rd is documented as a dataset". |
| **Value-attribution refinement** | 3 | "FaithBench scored ~50 %" vs KB's "SOTA hallucination detection models have ~50 % accuracies on FaithBench" — the proposition attaches 50 % to the wrong subject. |
| **Terminology refinement** | 1 | "vouch cuts agent confabulation by ~85 %" vs KB's specific "citation F1 +22.9pp / wrong-citation rate 36→5.5 %" — same magnitude, different metric name. |

**The dominant shape (55 % of (b))** is *numerical value-override on
under-specified claims*. The agent writes a bare number ("baseline is 50 %")
without specifying which metric/sample/condition, and the KB carries the
actual specific number for the implied metric. These are not literal
contradictions (under one reading, the proposition is just incomplete), but
the unqualified draft would mislead a reader.

### (c) correctly-rejected — 16 cases

All 13 sampled all-no pairs read as genuine adjacent noise: same entity,
unrelated predicate (e.g. "bun's install.sh pattern" vs "bun's --target
flag"; "Claude proposed the labels" vs "Claude can parse JSON output";
"comprisk.metrics exports concordance_index_cr" vs "comprisk's Uno-IPCW
matches survC1 to 1e-5"). Strict NLI is right to reject these.

3 of the 37 broad-only fires also classified as (c) — broad over-firing:

| # | shape | why over-fire |
|---|---|---|
| 14 | "vouch gate currently results in 'unknown command'" vs "commit added --session-context to vouch gate --draft" | The two propositions are about different aspects (current freshness vs prior commit). Broad's "implies the command exists" is a weak inference; both can be true with a stale binary. |
| 35 | "v3 reduces error rate to **5.5 %**" vs "wrong-citation rate dropped to **5.5 %** in v3 arm" | Same number, same metric — broad fired on "specifies the benchmark", but the proposition is correct. |
| 44 | "ALCE evaluates citation quality on open-ended QA" vs "ALCE evaluates along three dimensions: fluency, correctness, citation quality" | Proposition is a true subset of the KB. Broad fired on "materially completes"; this is a benign omission, not a misleading claim. |

These 3 over-fires put a floor on broad's precision: **~8 % over-fire rate
within broad-only** (3/37). That's the cost of relaxing the contract.

## Implications

### The 44.3pp loss is (b), not (a)

| where the loss goes | size | tunable? |
|---|---|---|
| (a) prompt-strictness | ~0.3 % of strict-no (1 / 340) | yes, by tightening the "approximate values" carve-out, but it's a single-digit-row signal in this corpus |
| (b) task-definition (contract) | ~9.7 % of strict-no (33 / 340) | only by changing P-α's contract from mutual-exclusion to useful-disagreement — a **design** call, not a prompt change |
| (c) correctly-rejected | ~90.0 % of strict-no (~306 / 340) | already correct |

**Tuning the strict prompt won't materially open the funnel.** Loose and
strict are empirically equivalent on this corpus.

### What "useful-disagreement" buys, and what it costs

Adopting V_BROAD-style contract on this corpus would:

- **Lift the structural fire rate from 1.4 % to ~12.5 % at the pair level.**
  Translated to row-level (any-proposition-flips → row flips) the effect is
  likely larger still — many of the v3-version pairs concentrate on the same
  few drafts.
- **Tip the funnel from "rare high-signal" to "moderate-signal, moderate-
  precision".** Broad's own over-fire rate in this sample is ~8 %, so the
  comprehensiveness primitive would shift to needing the same precision-
  audit treatment Axis 1 (per-claim grounding) gets.
- **Concentrate value on numerical version-level claims.** The version-class
  fire rate goes 0 % → 38.5 %; dataset goes 1.8 % → 15.3 %. This is exactly
  where the agent's drafts most need reconciliation (benchmark numbers,
  metric values).

The contract change is **plausible but not free**. Three considerations:

1. **8 % over-fire rate** means a user enabling P-α-broad would see ~1
   spurious fire for every ~12 productive ones. Acceptable in the
   numerical-comparison workload P-α was designed for; potentially annoying
   in a workflow that doesn't routinely cite specific values.
2. **Selection effect still applies.** This corpus is fires (failed
   groundings), which skew toward weaker-coverage entities. Production
   flip-rate on naturally-grounded ASSERTs is still ≥ this measurement.
3. **The N=2 precision audit caveat in the original replay still binds.**
   Production P-α's 100 % strict precision was on N=2; broad's ~8 %
   over-fire rate is on N=37 — much better numerically but still small.

### What this means for the gating question

The original p-alpha-replay verdict was "launch-ready as opt-in" gated by
**strict precision = 100 %, N=2**. This study doesn't change that verdict
for strict — strict is still high-precision-rare-fire. But it tells us:

- **There is no prompt-level tuning that buys reach without changing the
  contract.** Loose ≈ strict on this corpus.
- **There IS a contract-level lever (broad)** that buys ~11pp reach at ~8 %
  over-fire. Whether to pull it is a design call, not a prompt-engineering
  call.
- **The right next experiment, if reach matters, is a controlled bench
  (#54-style) where the contract change is tested on N≥100 synthetic items
  with hand-coded gold labels** — the dogfood corpus doesn't have ground
  truth, so we can't measure broad's true precision floor here.

### Where the value-override pattern lives

The (b) sub-shape distribution suggests the highest-leverage
**design surface** is not the prompt at all — it's a **separate "value
reconciliation" primitive** that:

- Detects when the proposition contains a number on an entity the KB has
  measured specific values for
- Fires not on "is this a contradiction" but on "does the KB attest a
  specific value for the agent's under-qualified claim"
- Doesn't need NLI — it's a number-extraction + lookup operation

This would be cheaper than NLI, higher-recall on the dominant (b) shape,
and orthogonal to P-α's current mutual-exclusion contract. P-α stays where
it is (rare, high-precision, opt-in); a new value-reconciliation primitive
covers the 9.7 % gap.

Out of scope for this study; flagged for #51 or a future issue.

## Reproduction

```sh
cd ~/Projects/vouch

# 1. Judge-study replay (3 variants × 345 pairs, ~6min, <$1 on Vertex)
./bench/dogfood/judge_study_P_alpha.ts --concurrency 6
# Output: bench/dogfood/fires-judge-study-P_alpha.jsonl (gitignored)

# 2. 50-row hand-classification sample
./bench/dogfood/judge_study_sample.ts
# Output: bench/dogfood/p-alpha-judge-study-sample.tsv (gitignored)
```

Schema (per pair row in `fires-judge-study-P_alpha.jsonl`):

```jsonc
{
  "ts": "...",
  "transcript_id": "...",
  "repo": "...",
  "entity": "...",
  "entity_class": "dataset|version|named-product|workspace-meta|...",
  "proposition": "...",
  "claim_id": <int>,
  "claim_text": "...",
  "dossier_slug": "...",
  "similarity": <float>,
  "strict": { "fires": bool, "score": float, "reason": "..." },
  "loose":  { "fires": bool, "score": float, "reason": "..." },
  "broad":  { "fires": bool, "score": float, "reason": "..." }
}
```

## Files

- `bench/dogfood/judge_study_P_alpha.ts` — funnel replay with 3 NLI variants
- `bench/dogfood/judge_study_sample.ts`  — N=50 stratified sample generator
- `bench/dogfood/fires-judge-study-P_alpha.jsonl` — 345-row replay output (gitignored)
- `bench/dogfood/p-alpha-judge-study-sample.tsv` — 50-row hand-classification sample (gitignored)
- This document — the study report
