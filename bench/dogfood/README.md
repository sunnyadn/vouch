# bench/dogfood — mining vouch gate-fires from live Claude Code use

Real-distribution counterpart to `bench/abstention` (synthetic SQuAD-2 prompts).
Walks the author's recent `~/.claude/projects/<repo>/<uuid>.jsonl` transcripts,
extracts every Stop-hook gate-fire event, and emits per-fire rows with the
draft that fired + the gate's structured fire output + prior-user context.

The harvested `fires-*.jsonl` is **gitignored** — it embeds user-private session
text. The `extract_fires.ts` tool is committed; the data isn't.

## Why this exists

`bench/abstention` (SQuAD-2 unanswerable + control + hand-curated false_premise)
is synthetic and has corpus quality issues (see #47, #48). The gate fires the
author actually experiences while working are the cleanest possible signal of
vouch's real precision/recall:

- Every fire is from an actual draft the author wrote, in an actual workflow,
  on an actual repo. No prompt-shape artifacts.
- The proposition + entity + reason are exactly what vouch emitted, with no
  reconstruction loss.
- Manual classification of (true-positive / false-positive / pattern) becomes
  both (a) the personal-use-proving evidence in the launch criteria and (b) a
  ground-truth fixture library for tightening #45 / #46 / #48 / future rules.

## Run

```sh
./extract_fires.ts                       # last 14 days, all ~/.claude/projects
./extract_fires.ts --days 7              # only last 7 days
./extract_fires.ts --project redacted-meta  # filter to one project dir basename
./extract_fires.ts --out fires.jsonl     # custom output path
```

Output goes to `bench/dogfood/fires-last<N>d.jsonl` by default. Per-row schema:

```jsonc
{
  "ts": "2026-05-13T19:32:00.000Z",
  "transcript_id": "1213c543-…",
  "repo": "-Users-sunny-Projects-redacted-meta",
  "git_branch": "main",
  "cwd": "/Users/…",
  "fire_text": "Stop hook feedback: …",       // full gate stderr block
  "propositions": [                            // parsed from the bullet rows
    { "entity": "vouch", "proposition": "…", "candidates_count": 3, "reason": "…" }
  ],
  "draft": "…",                                // assistant text that fired
  "prior_user": "…",                           // user turn that triggered the draft
  "post_fire_draft": "…",                      // the revise — used to classify shape
  "manual_label": null                         // filled by classify_fires.ts
}
```

The summary printed to stderr includes per-repo counts and top entities, which
already exposes some patterns at the bird's-eye level (e.g. high counts on
workspace-internal entity names = workspace-postfilter coverage gaps).

## Manual classification — `classify_fires.ts`

Keypress TUI that walks unlabeled fires and writes labels to
`fires-labeled.jsonl` (gitignored, append-only). Re-running picks up where
you left off (skips rows whose `transcript_id|ts` is already labeled).

```sh
./classify_fires.ts                            # default input fires-last14d.jsonl
./classify_fires.ts --filter-repo redacted-meta   # only fires from this project
./classify_fires.ts --filter-from 2026-05-13   # only fires after this ISO date
./classify_fires.ts --stats                    # no TUI, just print metrics
```

### Label vocabulary (single keystroke)

| key | class              | meaning                                                                                |
|-----|--------------------|----------------------------------------------------------------------------------------|
| A   | `verified`         | TP fire. Agent ran `vouch fetch` / `vouch claim` and the revise cites the new dossier. |
| H   | `hedged`           | TP fire. Agent kept the claim with explicit `(Unverified, from training memory)` tag. |
| C   | `continued-confab` | TP fire. Agent ignored the fire and repeated the same ungrounded claim.               |
| D   | `dodge`            | TP fire. Agent silently rephrased to remove the entity, OR argued the fire was FP without verifying. The #50 binding pattern. |
| F   | `false-positive`   | Fire was wrong (extractor over-fire / workspace-meta misjudge / NLI too strict).      |
| S   | `skip`             | Defer (ambiguous, multi-claim mixed verdicts, needs more transcript context).         |
| U   | undo               | Walk the last label back, both in-memory and on disk.                                  |
| Q   | quit               | Save and exit (progress always saves on every label).                                  |

### Derived metrics (printed on quit / via `--stats`)

```
gate_lift_rate = (A + H) / (A + H + C + D)
  Of true-positive fires, fraction that produced grounded or explicit-
  uncertainty output. Phase 1 bench result claims ~70% under controlled
  prompts; this measures it in real-world dogfood.

dodge_rate     = D / (A + H + C + D)
  #50 binding metric — the gap a tighter forcing function must close.

confab_persist = C / TP
  How often a fire failed to alter agent output.

fp_rate        = F / total
  Inverse of vouch's gate precision in dogfood corpus.
```

### Week-over-week measurement loop

The intended cadence:

1. Each weekend, run `extract_fires.ts --days 7` to get the new week's fires.
2. Run `classify_fires.ts` and label them (target: ≤30min if the week has
   ~30–50 fires).
3. Run `--stats` to print the metric row; record it in a tracking file
   (e.g. `fires-weekly-metrics.csv`, gitignored).
4. Compare week-over-week. After a gate fix lands, the metric move tells you
   whether the change actually shifted behavior in the wild.

This is the measurement infrastructure for the #50 forcing-function work and
the launch claim's "reduces confabulation in agent workflows" assertion.

A sanitized subset (entities + propositions only, no draft content) may be
publishable as `bench/gate-precision-public/` for a public benchmark — that
work hasn't started.
