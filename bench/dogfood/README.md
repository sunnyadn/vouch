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
  "manual_label": null                         // human-pass classification slot
}
```

The summary printed to stderr includes per-repo counts and top entities, which
already exposes some patterns at the bird's-eye level (e.g. high counts on
workspace-internal entity names = workspace-postfilter coverage gaps).

## Manual classification workflow (TODO)

The current harness emits 200+ fires in ~14 days of live use. Next steps:

1. Build a small CLI (or notebook) that walks rows interactively, prompts
   true-positive / false-positive / pattern-tag, and writes back to the JSONL.
2. Aggregate into a frequency table: per-pattern fire count, per-entity-class
   fire count, fire-by-repo distribution.
3. Use the high-frequency false-positive patterns as the priority queue for
   gate refinements; use the true-positive patterns as the "vouch saved me
   from N hallucinations this week" launch evidence.

A sanitized subset (entities + propositions only, no draft content) may be
publishable as `bench/gate-precision-public/` for a public benchmark — that
work hasn't started.
