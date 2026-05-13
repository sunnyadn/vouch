# `bench/abstention/` — vouch arm-2 measurement (issue #36)

**What this measures.** vouch has two arms: (1) precision — when the agent asserts, it's grounded; (2) epistemic humility — push the agent to read/search/push-back when the answer isn't in reach, instead of confabulating. Arm 1 has #35 (gate recall + precision) and #6 (ALCE citation quality). This directory is **arm 2**.

## Design (Phase A)

Three subsets + a control:

| subset | n | source | correct behavior |
|---|---|---|---|
| (a) unanswerable | 50 | SQuAD-2.0 unanswerable subset | abstain / say "not in passage" |
| (b) false-premise | 50 | hand-curated | push back on premise |
| (c) searchable-but-not-provided | — | deferred to Phase B (agentic harness) | search then answer |
| (control) answerable | 50 | SQuAD-2.0 answerable subset | correct answer |

## Harness

`run_arm2_eval.py` runs each item twice:

- **without-vouch**: generator (default `claude-haiku-4-5`) answers, vanilla, no Stop hook.
- **with-vouch**: generator answers with Stop hook = `vouch gate --transcript-stdin --strict`. On fire the agent gets ONE revise round, then we record the final response.

Responses scored by an LLM judge (default `gemini-3.1-pro-preview`, mirroring vouch's verifier) into `{correct, appropriate-abstain, appropriate-pushback, confabulated, refused-vague}`.

## Metrics

- **Confabulation rate** ↓ on (a) and (b) — confident wrong / accepts-false-premise. Headline.
- **Appropriate-response rate** ↑ on (a) and (b) — explicit abstain / pushback.
- **Retained accuracy** on control — guardrail; with-vouch must not collapse vs without-vouch.

Report: `report.md` (or `report-N50-N50-N50.md` if size-tagged) with the with/without delta on each subset.

## Files

- `sample_squadv2.py` — pull + filter HuggingFace `rajpurkar/squad_v2`; emit `unanswerable.jsonl` and `control.jsonl`
- `false_premise.jsonl` — hand-curated false-premise items (committed verbatim; see header for premise-error taxonomy)
- `judge_prompt.md` — the judge classification prompt
- `run_arm2_eval.py` — runs both arms, scores, emits report
- `report.md` — output

## Notes

- This bench IS committed (unlike `bench/gate-recall/` which carries private CC transcripts). All three subsets here are either public-set adapted (SQuAD-2.0 is CC-BY-SA 4.0) or hand-built generic prose with no user-private content.
- Generator and judge models pinned in `run_arm2_eval.py`; sweep variants saved separately.
- Per the #36 plan-comment on the issue: Phase A is non-agentic. Phase B will add (c) with a multi-turn agentic harness that gives the generator retrieval tools.
