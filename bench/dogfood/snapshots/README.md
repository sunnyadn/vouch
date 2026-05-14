# bench/dogfood/snapshots — session-metric snapshots

Each `YYYY-MM-DDTHHMM.jsonl` file is the output of `session_metrics.ts
--format jsonl` at that timestamp. Snapshots are committed (unlike
`fires-*.jsonl` which carry verbatim agent drafts and are gitignored)
because session_metrics outputs aggregate counts only — no privacy
content.

## How to compare two snapshots

```sh
# Compute delta: today's aggregate vs prior snapshot
bench/dogfood/session_metrics.ts --aggregate
diff bench/dogfood/snapshots/<prior>.jsonl bench/dogfood/snapshots/<today>.jsonl
```

Or use jq for a clean diff:

```sh
jq -s '.[0] as $a | .[1] as $b | {
  humility_pct_delta: ($b.humility_pct - $a.humility_pct),
  addressed_fetch_delta: ($b.addressed_fetch - $a.addressed_fetch),
  addressed_remove_delta: ($b.addressed_remove - $a.addressed_remove),
}' <(bench/dogfood/session_metrics.ts --aggregate) <prior-snapshot.jsonl>
```

## What direction is good

- `humility_pct` ↑: agent surfacing more uncertainty (good — explicit-
  uncertainty stances or counter-evidence reconciliation increasing).
  Decline of >10pp on a single week suggests something off (regression
  in extractor's HEDGE classification, or agent over-correcting toward
  confidence after Stage 3 escalation discipline trains it that way).

- `addressed_fetch` ↑ relative to `addressed_remove` ↓: agents are
  verifying via tool calls rather than silently dropping entities.
  This is the #50 (A) Stage 2 working signal — fires are converting
  to grounded outcomes rather than dodges.

- `awaiting_revise` ↓ over a session: pre-existing fires are getting
  resolved across turns. A growing backlog signals dodges accumulating.

## Snapshots so far

- `2026-05-14T0615.jsonl` — first snapshot. Captures state after one
  marathon session of primitive ship (25 commits today, three new
  axes: per-claim grounding / humility / comprehensiveness). 3
  transcripts in last 6 days. Aggregate humility 29.5%; addressed/
  fetch=2, addressed/remove=9 (heavily skewed remove — partly
  legitimate topic-drops, partly silent-rephrase dodges that
  #50 A Stage 3.5 would distinguish but hasn't shipped).
