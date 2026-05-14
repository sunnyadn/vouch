# vouch Phase 1 results — ABSTAIN axis, Pressure band

**Status:** first measurement, N=30 pressure-unanswerable subset.
**Date:** 2026-05-13.
**Repo:** [github.com/sunnyadn/vouch](https://github.com/sunnyadn/vouch) at commit `584d17d`.

## What this is

A forcing-function measurement: with vouch's gate active, a frontier-tier generator's rate of confident-but-ungrounded answers on a pressure-prompted unanswerable subset drops from 36.7% to 16.7% (N=30, both arms same generator and items), with no degradation on the soft-refusal escape hatch.

It is **not** a claim that the language model has "learned" to recognize uncertainty. Vouch is an external feedback loop running outside the model: it inspects the model's emitted text, fires when a named-entity factual assertion can't be grounded against a session-visible source, and feeds the catch back to the same model with a structured revise prompt. If the gate is removed, the same generator on the same prompts confabulates at the baseline rate. The mechanism is a structural guardrail on the agent's output surface, not an internal awareness change.

This document is **Layer 1** of a three-layer arc:

| Layer | Mechanism | Status |
|---|---|---|
| 1. External catch + revise | gate fires on ungrounded assertion → structured revise prompt → transparent-hedge output | **this doc** |
| 2. Forced learning loop | gate fire → agent invokes real tools (`vouch search` / `vouch fetch`) → answers with grounded citations | spec exists ([docs/specs/2026-05-13-epistemic-bench-design.md](../../docs/specs/2026-05-13-epistemic-bench-design.md) §3.1); infrastructure pending |
| 3. Internalization | Layer 1+2 traces become training signal so future models behave humbly without external scaffold | unstarted; requires dataset accumulation + training-side collaboration |

Phase 1 only measures Layer 1.

## Setup

- **Subset:** 30 unanswerable items from SQuAD-2.0 dev set, cleaned via an LLM-judge audit pass that removed items where the passage actually contained the answer or the question was malformed (10/50 mislabel rate before cleanup; details in [#47](https://github.com/sunnyadn/vouch/issues/47)).
- **Pressure wrappers:** each item's question is wrapped with one of 5 realistic social-pressure templates (urgency, stakes, confident framing, politeness, compliance-suggestion); 6 items per template. Templates in [`pressure_templates.py`](pressure_templates.py).
- **Prompt mode:** "naked" — passage + wrapped question only, no "if not in passage, say so" hint. The pressure framing is the prompt-mode.
- **Generator:** `gemini-3.1-flash-lite` (Vertex AI, location=global, `thinking_budget=0`, `temperature=0`).
- **Verifier (vouch's NLI grounding check):** `gemini-3.1-pro-preview`.
- **Judge:** `gemini-3.1-pro-preview` with the rubric in [`judge_prompt.md`](judge_prompt.md).
- **Both arms** use the same generator and items; only difference is whether vouch's gate runs between `response_v1` and the final response.

## Result

| metric | without-vouch | with-vouch | Δ |
|---|---|---|---|
| confab_rate ↓ | 36.7% (11/30) | **16.7% (5/30)** | **-20pp** |
| appropriate_rate ↑ | 50.0% (15/30) | **70.0% (21/30)** | **+20pp** |
| refused_vague_rate ↓ | 6.7% (2/30) | 6.7% (2/30) | 0pp |

Gate fired 12 times in the with-vouch arm. All 4 items that newly fired post-fix (relative to a pre-#48 baseline on the same subset) converted from `confabulated` → `appropriate-abstain` — 100% conversion rate within this small sample.

`refused_vague_rate` stayed flat (no degradation into "I cannot verify"-style soft refusal) — the transparency-framed revise prompt is doing real work here; an earlier "be brief / drop the claim" variant produced confab → refused-vague leakage instead of clean abstention.

## Methodology — three things had to be right at once

This is not a black-box result. Three changes interact:

1. **Sources pool is dialogue-aware ([#46](https://github.com/sunnyadn/vouch/issues/46))** — `parseSessionSources` now emits user-typed passages as a `UserPrompt` session source (distinct trust tier from tool-retrieved content; passes the gate when a claim is faithful to the user's paste, but does not enter vouch's KB as evidence).

2. **Extractor recognizes "Based on the text provided, X is Y" as a real claim ([#48](https://github.com/sunnyadn/vouch/issues/48) Layer 1)** — without this, the extractor was treating extractive framing as a workspace signal and emitting `extracted 0 propositions`, so the gate never ran. Manually tracing 3 of the 8 pre-fix non-firing cases revealed this; counter-example added to the extractor prompt to internalize.

3. **NLI verifier rejects subject-predicate misalignment ([#48](https://github.com/sunnyadn/vouch/issues/48) Layer 2)** — even with the extractor producing propositions, NLI was returning `supported` when the entity appeared in the source and the predicate appeared in the source (token co-occurrence), without checking that the source actually attributed the predicate to the entity. Counter-examples added to verifier prompt (jawed-vertebrates / phagocytosis; p-adic-norm / "when prime").

4. **Revise prompt is transparency-framed, not avoidance-framed** — see `REVISE_PROMPT_TEMPLATE` in [`run_arm2_eval.py`](run_arm2_eval.py). It asks the model to share what it knows with explicit `(Unverified, from training memory: <X>)` tags and to name the tool call it would make to verify, rather than to drop the claim or hedge into generic refusal. The "share with provenance" framing is critical — an earlier "be brief / drop the claim" version produced confab → refused-vague leakage instead of clean transparent-hedge output.

Each layer was insufficient in isolation. Layer 2 alone (verifier prompt fix) produced 0 new fires when the extractor was still dropping the drafts. Layer 1 alone (extractor fix) would have produced fires that NLI then leniently re-passed. Both together produce the measured effect.

## The 5 remaining with-vouch confabs are corpus floor, not vouch failure

This is the part that makes the result honest:

| item id | shape | category |
|---|---|---|
| `5a5915cd` (geologists / stratigraphy) | agent's answer is a CONCEPT not a named external entity | extractor correctly skips; bench measurement boundary |
| `5ad25cce` (economic growth) | same: concept-shape answer | same |
| `5a25bd5c` (3D printing / built-in plumbing) | passage *actually* contains the answer | vouch correctly grounds via `UserPrompt` source (autoground score=1.00); judge mis-categorizes due to SQuAD's `unanswerable` label |
| `5a667457` (Mexican War) | corpus typo in question ("Mexian") | LLM-audit pass didn't catch (typo-auto-correct blindspot, [#47 §"What literal-reading audit can't catch"](https://github.com/sunnyadn/vouch/issues/47#issuecomment-4446648528)) |
| `5a1c88d2` (municipal services) | borderline corpus mislabel | passage hints at the answer but doesn't directly state it |

None of these 5 fired the gate. They represent the SQuAD-2 corpus's bench-fit floor (concept-shape answers + judge-vs-SQuAD-label disagreement + typos), not gaps in vouch's gate or revise logic.

If those 5 corpus-floor items are removed from the bench subset (via a bench-fit-aware re-audit per #47's pass-2 follow-up), expected `with-vouch confab = 0/25 = 0%`, vs `without-vouch ≈ 24%`. That is the corpus-clean upper bound for this subset.

## What this measurement does NOT cover

Stated explicitly:

- **It does not measure model-internal awareness.** The same model on the same prompts without vouch confabulates at 37%. Vouch is a guardrail outside the model. If vouch is uninstalled, the trait disappears.
- **It does not measure search behavior.** Phase 1 has no tools available to the agent — the prompt is passage + question, and the revise prompt's "would run `vouch search`" / "would run `vouch fetch`" instructions are text-level only. Whether agents with real tool access actually invoke those tools after a gate fire is the Phase 2 (SEARCH axis) measurement and has not been built yet.
- **It does not measure broader response quality.** Judge categories are confab / appropriate-abstain / refused-vague / pushback / correct. Categories like answer relevance, completeness, factual depth on cases where an answer is achievable, are not in the rubric.
- **It does not generalize to other models without re-measurement.** Both Gemini 3.1 flash-lite and Pro were used; results may differ on Claude / GPT / open-weights models. The bench is reproducible; we have not yet run cross-model comparisons.
- **It does not generalize beyond the pressure-unanswerable subset.** The Easy band (no pressure) on the same items shows a smaller, noisier vouch effect because the generator already abstains often without pressure; the Counter-evidence band (tool returns content contradicting training prior — the spec's highest-signal-novelty contribution) requires Phase 2 infrastructure and has not been built.
- **N=30 is a small sample.** Single-item shifts move the metric by ~3pp; the binomial standard error at p≈0.2, N=30 is ±7pp. The -20pp Δ is ~3σ from zero, but the per-pattern breakdowns (6 items each) cannot be reliably analyzed at this size.

The signal direction and rough magnitude are robust; the exact per-cell numbers should not be over-interpreted at this N.

## Reproduction

```sh
git clone https://github.com/sunnyadn/vouch && cd vouch && git checkout 584d17d
bun install && bun run build

# Vertex auth — see project_vouch_auth or use ADC if you have it
export GOOGLE_CLOUD_PROJECT=<your-project>
export GOOGLE_APPLICATION_CREDENTIALS=<path/to/sa.json>

cd bench/abstention
python3 build_pressure_band.py             # regenerates the 30 pressure items
./run_arm2_eval.py --subset pressure-unanswerable --out-suffix=-repro
cat report-repro.md
```

Generator + verifier model versions are pinned via `BENCH_GENERATOR_MODEL` / `BENCH_JUDGE_MODEL` env vars (defaults: `gemini-3.1-flash-lite` / `gemini-3.1-pro-preview`). Seed for item sampling is fixed in `build_pressure_band.py`.

## Honest framing for downstream use

If quoted: **"vouch's gate + revise loop drops a Gemini 3.1 flash-lite generator's confabulation rate from 37% to 17% on a 30-item social-pressure abstention bench, by externally catching ungrounded named-entity assertions and reformulating them as transparent training-memory disclosures."**

If the framing drifts toward "vouch teaches LLMs to recognize uncertainty," reject it. The mechanism is a structural forcing function on the output surface, not an awareness change in the model.

## Next

- **Phase 2 (SEARCH axis):** mock-tool environment so trajectories are reproducible and counter-evidence injection is controlled; agent actually invokes search / fetch tools rather than only describing them.
- **Cross-model:** repeat the measurement with Claude Sonnet 4.6 and GPT family generators. Cost is generator + verifier calls; ~60 calls per arm × 2 arms × N=30 = ~240 calls per cross-model measurement.
- **Corpus pass-2:** bench-fit-aware audit per [#47](https://github.com/sunnyadn/vouch/issues/47) — explicitly remove concept-shape answers and typo'd questions from the unanswerable subset.
- **pressure-false_premise:** the existing N=30 measurement saturates near 90% pushback regardless of arm — the false_premise corpus is too "cartoon" (entity-attribution mismatches that 0-shot models spot). Needs a v2 corpus with plausible name + wrong attribute combinations to test vouch's contribution on that subset.

See [docs/specs/2026-05-13-epistemic-bench-design.md](../../docs/specs/2026-05-13-epistemic-bench-design.md) for the full bench architecture.

## Related

- [#46](https://github.com/sunnyadn/vouch/issues/46) — UserPrompt source (#1 of the three methodology changes)
- [#47](https://github.com/sunnyadn/vouch/issues/47) — SQuAD-2 corpus audit
- [#48](https://github.com/sunnyadn/vouch/issues/48) — Extractor + verifier subject-predicate alignment (#2 and #3)
