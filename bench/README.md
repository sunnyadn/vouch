# The evaluation behind the numbers

This directory is the harness behind every number in the top-level README. The short version
of the protocol: measure block-precision on trajectories where the reviewer should stay silent,
measure block-recall on trajectories with a known planted or labeled hallucination, adjudicate
every disputed fire with a judge that is never from the model family being judged, and count
ambiguous calls against the gate.

## Which numbers you can recompute here, and which you can't

| claim (top-level README) | source | independently recomputable? |
|---|---|---|
| block-precision ~80% (95% CI 70–87) | `agenthallu-eval/` on AgentHallu clean trajectories | **Yes** — data, adjudication log, and scripts are all committed. Raw floor 77%, blind-judge estimate ~80%, author-affiliated optimistic bound ~87%; the headline is the blind number. See `agenthallu-eval/precision-adjudication.md`. |
| recall ~60% (provisional) | private gate-recall benchmark | **No** — see "What is withheld" below. Treat it as the author's provisional measurement, not an established figure. |
| cry-wolf on its own repo (disabled after repeated false blocks) | `decision-audit/` replay corpus | Partially — the replay scripts and case definitions are committed; some underlying session content is withheld. |
| reviewer cost / fail-open under large traces | `quant-repro/` + hook time-budget behavior | Yes — fixtures include full frozen traces. |

## Directory map

- **`agenthallu-eval/`** — the precision headline. AgentHallu clean trajectories, 30 cases × 3
  reps; every BLOCK fire adjudicated case-by-case in `precision-adjudication.md` (the committed
  gold), raw evidence in `precision-fps.json`. Cross-family and variance harnesses
  (`xfamily-precision.ts`, `analyze-variance.py`) probe run-to-run precision swing.
- **`quant-repro/`** — author-unbiased live gold. Real headless workers (opus / haiku / kimi) did
  genuine quant tasks with planted traps (look-ahead, multiple testing, transaction costs); their
  own conclusions became the FIRE/NOFIRE cases. No human wrote the conclusions being judged.
  Frozen traces + labels in `fixtures/` (see its README).
- **`decision-audit/`** — the cry-wolf corpus: replays of real false blocks (including the one
  that got vouch disabled on its own repository) plus one `verify-*.ts` harness per fix, so every
  prompt/matcher change is tested against the regression it was meant to kill and the recall it
  must not lose.
- **`dogfood/`** — mining of live gate fires from my own sessions. Mostly withheld (below); the
  committed `*-probe-fires.jsonl` files are small detector probes whose content is technical only.
- **`gate-recall/`** — withheld in full.
- **`phase0/`, `deepseek-eval/`, `verify-replay/`, `lib/`** — earlier NLI/faithfulness spikes,
  reviewer-backend bake-off, replay plumbing, shared model matrix.

## What is withheld, and why

The recall benchmark (`gate-recall/`) and the dogfood fire corpus (`dogfood/fires-*.jsonl`,
`p-alpha-judge-study-sample.tsv`) are built from extracts of my own Claude Code conversation
transcripts. They contain private session content, so they are gitignored and stay out of the
repository. The consequence is stated plainly rather than hidden: **the ~60% recall figure is not
independently reproducible from this repo**, which is one of the reasons the README labels it
provisional (the other: it was measured under reviewer-quota strain). A reconstructable
synthetic-only recall benchmark is the obvious next step and does not exist yet.

## Adjudication protocol

Three rules, applied everywhere a number is reported:

1. **Blind where possible.** The second rater never sees the first rater's labels
   (`agenthallu-eval` blind judge; the P-α cross-validation used the same discipline).
2. **Cross-family.** A fire by the deployed reviewer (kimi) is adjudicated by a different model
   family (opus / gemini), so no model scores its own side.
3. **Ambiguity counts against the gate.** Every ambiguous case in the committed adjudication logs
   is resolved toward cry-wolf, to counteract author bias. Where an author-affiliated pass gives a
   better number, it is reported as the optimistic bound and explicitly not the headline.
