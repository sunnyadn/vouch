# Cross-validation of N=50 hand classification — P-α judge study (2026-05-14)

Blind second-rater pass over the 50-row hand-classification reported in
[`p-alpha-judge-study-2026-05-14.md`](p-alpha-judge-study-2026-05-14.md).
The original classification was Claude's; this run hides Claude's labels and
asks the same NLI client production uses (`vertex_ai/gemini-3.1-pro-preview`
via `src/verifier.ts`) to classify each row independently.

**Research-only.** Goal is to test whether the headline finding —
"44.3pp funnel loss is contract, not prompt" — survives an independent
classification of the same pairs. Not a launch gate.

## Headline

| metric | value |
|---|---|
| **overall agreement** | **42 / 50 = 84.0 %** |
| **Cohen's κ** | **0.696** (substantial agreement, Landis–Koch) |
| disagreements | 8 |
| direction of bias | judge promotes 5 of Claude's (b/c) → (a); demotes 2 of Claude's (b) → (c); promotes 1 (c) → (b) |

**Verdict (one line):** the "44.3pp loss is contract, not prompt" headline
**survives**. The dominant-class story holds (judge: 56 % b, 32 % c, 12 % a;
Claude: 66 % b, 32 % c, 2 % a), and (b) still strictly dominates (a). What
the cross-validation **sharpens** is the (a) bucket's true size: there is a
specific **entity-category-mismatch sub-pattern** ("X is a dataset" vs "X is
an algorithm / lab / package") that Claude classified as (b) refinement but
the judge reads as (a) logical contradiction. That sub-pattern is plausibly
**prompt-tunable** in a way the original report's (a) discussion (which
focused on numerical-precision carve-outs) did not flag. The contract-change
finding still holds for the numerical value-override majority.

## Methodology

### What the judge saw

Each of the 50 sample rows, fed independently with:

- `entity_class`
- `similarity` (cosine)
- STRICT / LOOSE / BROAD verdicts (`fires`, `score`)
- BROAD reason (the only per-variant `reason` field in the TSV — same
  information set Claude's hand-rater had)
- proposition A (agent draft)
- KB claim B (attested)

**The judge did NOT see** Claude's class label. The TSV's `class_label`
column was never populated — Claude's labels live only in the writeup —
so no preprocessing was needed; running on the raw TSV is information-
equivalent to "drop the column".

### Classes

Verbatim from `p-alpha-judge-study-2026-05-14.md` § Methodology:

- **(a) prompt-strictness loss** — A and B are actually mutually exclusive
  (a real logical contradiction), but the strict NLI prompt missed it. Fix
  is to tune the strict prompt.
- **(b) task-definition loss** — not a logical contradiction, but B refines /
  value-overrides / narrows / partially contradicts A in a way an honest
  writer should reconcile. Only a contract change would catch it.
- **(c) correctly-rejected** — adjacent noise; predicates don't overlap or
  trivially compatible. Strict NLI is right to reject.

### Client

`src/verifier.ts`'s `generateObject` call against `VERIFIER_MODEL`
(`vertex_ai/gemini-3.1-pro-preview`), same client production NLI uses. Schema:

```ts
z.object({ class: z.enum(["a", "b", "c"]), reason: z.string().max(500) })
```

### Run

```sh
./bench/dogfood/judge_study_crossvalidate.ts --concurrency 5
```

50 rows, ~64 s wall, ~$0.05 on Vertex. No errors. Output written to
`bench/dogfood/p-alpha-judge-study-crossvalidate.jsonl` (gitignored).

### Reconstructing Claude's labels for comparison

Claude's labels live in `p-alpha-judge-study-2026-05-14.md` (the TSV's
`class_label` column was empty when the report was produced — Claude
classified mentally while writing). Reconstruction rules:

- All 13 stratum=`all-no` rows (indices 1–13) → (c) — confirmed by §
  correctly-rejected: "All 13 sampled all-no pairs read as genuine adjacent
  noise."
- Among 37 stratum=`broad-only` rows (indices 14–50):
  - Row 48 → (a) — only (a) case, called out in § prompt-strictness loss.
  - Rows 14, 35, 44 → (c) — the three broad over-fires named in the table.
  - All remaining 33 rows → (b).

Tally: a=1, b=33, c=16, matching the report's headline counts.

## Confusion matrix

Rows = Claude, columns = Judge.

|              | judge a | judge b | judge c | **row total** |
|---           |---:     |---:     |---:     |---:           |
| **claude a** | **1**   | 0       | 0       | 1             |
| **claude b** | 4       | **27**  | 2       | 33            |
| **claude c** | 1       | 1       | **14**  | 16            |
| **col total**| 6       | 28      | 16      | **50**        |

Diagonal = 42, off-diagonal = 8.

### Per-class metrics (treating Claude as reference)

| class | Claude n | judge n | both | precision | recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| (a) prompt-strictness | 1 | 6 | 1 | 16.7 % | 100 % | 28.6 % |
| (b) task-definition | 33 | 28 | 27 | 96.4 % | 81.8 % | 88.5 % |
| (c) correctly-rejected | 16 | 16 | 14 | 87.5 % | 87.5 % | 87.5 % |

### Agreement statistics

| metric | value |
|---|---|
| Observed agreement Po | 0.8400 |
| Expected by chance Pe | 0.4744 |
| **Cohen's κ** | **0.6956** |

Landis–Koch interpretation: 0.61–0.80 = **substantial agreement**.

## Disagreement table

| # | claude | judge | shape | one-line diagnosis |
|---:|---|---|---|---|
| 14 | c | a | "vouch gate unknown command" vs "commit added --session-context to vouch gate" | **Borderline.** Judge reads as same-moment contradiction about command existence; Claude's (c) reads as different aspects (stale binary vs prior commit) that can co-exist. Both defensible. |
| 20 | b | a | "Letta is an agentic memory product" vs "Letta is an AI lab" | **Borderline.** Formally a categorical contradiction at the literal level; in industry context an AI lab routinely ships products. Judge has the literal reading; Claude has the contextual one. |
| 21 | b | a | "FActScore is a dataset" vs "FActScore breaks a generation into atomic facts and computes the percentage…" | **Judge tighter.** The KB's verb structure ("breaks", "computes") makes FActScore an algorithm / metric, not a dataset. The category mismatch is logical, not just a refinement. |
| 23 | b | c | "ALCE focuses on citation-correctness in open-generation rather than NLI-given-evidence" vs "ALCE paper uses google/t5_xxl_true_nli_mixture as the canonical NLI evaluator" | **Judge right.** Tool-vs-task at orthogonal granularity: ALCE's primary axis is citation quality and it happens to *use* NLI as the evaluator. The propositions don't actually conflict — Claude over-classified (b). |
| 24 | b | c | "v3 precision vs baseline is z=10.1, p<<0.0001" vs "v3 arm's delta was citation_recall +10.1pp, citation_precision +23.0pp, citation_f1 +17.1pp" | **Borderline.** Prop A is plausibly misreading the +10.1pp recall delta as a z-score, which is a substantive error (Claude's reading). But at the literal predicate level — z-score vs percentage-point delta — they are different quantities, so judge's "predicates don't overlap" is also defensible. |
| 35 | c | b | "v3 reduces error citation rate to 5.5 %" vs "wrong-citation rate dropped from 36 % at baseline to 5.5 % in v3 arm" | **Claude right (judge over-fires).** Same metric, same number, same target arm. The KB just specifies the benchmark (ALCE-ELI5, 100 samples) — a benign addition that doesn't make A misleading. Judge's "B refines A by specifying conditions" is a generic broad-style over-fire. |
| 41 | b | a | "ALCE measures arm 1 only" vs "ALCE evaluates LLM generations along three dimensions: fluency, correctness, citation quality" | **Borderline.** "Arm 1 only" is opaque enough to read either way: as a categorical "one thing" (which contradicts "three dimensions") or as a refineable under-specified claim. Judge's tighter read is defensible. |
| 42 | b | a | "follic is a package with license/provenance claims" vs "The R follic.Rd file is titled 'Follicular Cell Lymphoma' and is documented as a dataset" | **Judge tighter.** In R, "package" and "dataset" are formally distinct entity kinds — same shape as row 21 (FActScore). Judge's logical-contradiction read on entity category is correct; Claude under-tightened to (b). |

### Disagreement summary

| who was tighter | count | rows |
|---|---:|---|
| Judge clearly tighter / more defensible | 3 | 21, 23, 42 |
| Borderline (both defensible) | 4 | 14, 20, 24, 41 |
| Claude clearly right (judge over-fired) | 1 | 35 |

### Where the judge and Claude systematically diverge

The judge promoted 5 of Claude's (b/c) labels to (a). Four of those — rows 20
(Letta), 21 (FActScore), 41 (ALCE arm-1), 42 (follic) — share a pattern:
**entity-category-mismatch** of the shape "X is a [dataset / package /
product]" vs KB-attested "X is an [algorithm / lab / dataset]". The fifth (row
14) is a different shape (command existence under temporal ambiguity).

This is the cross-validation's most actionable finding: the original report's
(a) discussion focused on the strict prompt's "approximate values" carve-out
(precision/approximation hedges allowing 9.5pp error through). The judge
identifies a **separate (a) sub-pattern** on entity-categorical claims that
the strict prompt's design also under-tightens — and where prompt-level
tuning (e.g. a "category-of-entity" rule) is plausibly the right lever.

## Does the verdict survive?

The original headline: *"The 44.3pp funnel loss is almost entirely
contract-loss (b), not prompt-strictness (a)."*

Under the judge's labels:

| class | Claude % | judge % | scaled-340 (judge) |
|---|---:|---:|---:|
| (a) prompt-strictness | 2 % | 12 % | ~41 pairs (~12 %) |
| (b) task-definition | 66 % | 56 % | ~190 pairs (~56 %) |
| (c) correctly-rejected | 32 % | 32 % | ~109 pairs (~32 %) |

**Yes — the verdict survives, with one sharpening.** (b) is still 56 % of
the sample and 4.6× larger than (a). Switching from strict to broad still
recovers the (b) bulk; loose still does no work; the original 0/45
named-product cells unchanged. The contract-change lever is still the
dominant design surface.

**The sharpening:** (a) is bigger than Claude estimated — ~12 % rather than
~0.3 %. That ~12 % concentrates on entity-category-mismatch, a shape
distinct from the numerical-precision carve-out the original report
characterized. There is more prompt-tunable headroom than the original
"loose ≡ strict" reading suggested. Worth noting for #51 follow-up; doesn't
change the recommendation to keep P-α as-is opt-in.

## Files

- `bench/dogfood/judge_study_crossvalidate.ts` — blind classifier
- `bench/dogfood/p-alpha-judge-study-crossvalidate.jsonl` — per-row output (gitignored)
- This document — the cross-validation writeup

## Reproduction

```sh
cd ~/Projects/vouch

./bench/dogfood/judge_study_sample.ts            # regenerates the 50-row TSV
./bench/dogfood/judge_study_crossvalidate.ts --concurrency 5
```
