# vouch primitive stack — state as of 2026-05-14 06:00 UTC

A reference index of the gate-side primitives that exist and how to invoke them.
Originated from one sustained working session 2026-05-13 → 2026-05-14 focused on
the question: **does today's vouch work help LLMs be more humble + more
comprehensive?**

Answer arrived at: vouch's existing surface is on the *per-claim-grounding* axis
(anti-fabrication). Today added two new axes: **humility** and
**comprehensiveness**. Per-claim grounding is necessary-not-sufficient for
either of those; the two new axes are independent primitives.

## Axis 1 — per-claim grounding (was the only axis before today)

| Primitive | Default | Surface |
|---|---|---|
| Gate fire on ungrounded ASSERT | ON | exit 2 in strict mode |
| Session-source autoground (#39) | ON | auto-grounds via Read/WebFetch/Bash tool results |
| KB nearest candidates inline (#50 E, 4afb5ae) | ON | renders top-2 entity-sharing KB claims with NLI gap reason |
| Per-session fire counter (P1, 7918118) | ON | `[vouch-gate] session so far: …` |
| #50 (A) Stage 1 awaiting_revise ledger (e4ee6c4) | ON | tracks fires for cross-turn check |
| #50 (A) Stage 2 revise classification (469b72c) | ON | classifies next-turn revise as fetch / hedge / remove |
| #50 (A) Stage 3 escalation (f9228ec) | OFF | `VOUCH_GATE_ESCALATE_UNADDRESSED=1` → block on unaddressed prior fires |

## Axis 2 — humility (new today)

Two orthogonal signals — per-claim hedge rate AND turn-level gap enumeration.

| Primitive | Default | Surface |
|---|---|---|
| Humility ratio (P-γ Stage 1, 7b3418b) | ON | `[vouch-gate] humility: H+S / total = X%` when total ≥ 10 |
| Blind-spot detection (P-γ.5, 5d72fc4) | ON | `[vouch-gate] blind-spots this turn: N enumerated` |
| Over-confidence flag (P-γ Stage 2, c91a33b) | ON | `⚠ over-confidence flag: N asserts, 0 hedges, 0 gaps` |
| `[gap: <text>]` explicit marker | always parsed | counted toward blind-spot total |
| Natural-phrase gap detection | always parsed | 7 patterns ("haven't verified", "worth checking", etc) |

### Healthy ranges (provisional, based on today's session)

- Humility ratio: **10–25% explicit-uncertainty** on grounded analytical work.
  ≤5% suggests over-confident claiming; ≥40% suggests over-hedging or topic
  with poor KB coverage.
- Blind-spots per turn: at least **1 per heavy-claim turn** (3+ ASSERTs).
  0 blind-spots on a multi-claim turn fires the P-γ Stage 2 over-confidence
  flag.

### How to use as an agent

Add `[gap: <specific thing you didn't check>]` near your claims when there's a
real facet you didn't verify. Cheap to write, makes the limits of retrieval
visible to the user. Don't pad with vacuous gaps to game the metric — 0
honest gaps > 3 fake ones.

## Axis 3 — comprehensiveness (new today)

Confirming evidence is necessary-not-sufficient. The gate also searches for
contradicting claims.

| Primitive | Default | Surface |
|---|---|---|
| KB counter-evidence pull (P-α, 07e43f0) | **OFF** | `VOUCH_GATE_COUNTER_EVIDENCE=1` → flips grounded ASSERTs to ungrounded if KB has contradicting supported claim |
| Web-counter-evidence suggestion (P-α.5, 71dfaf4) | follows P-α env | `[vouch-gate] P-α: N grounded ASSERTs cleared KB counter; consider vouch search "<entity> limitations" --provider ddg` |

### How to enable

```sh
# Enable KB counter-evidence pulls (cost: ~$0.05-0.2/turn additional LLM)
export VOUCH_GATE_COUNTER_EVIDENCE=1

# Run vouch normally — gate now searches for contradictions on every
# grounded ASSERT. When found, fire with three productive paths:
#   → reconcile (synthesis-of: support, contra)
#   → supersede (file new, mark old wrong)
#   → hedge (unverified, sources disagree)
```

### How to use as an agent

When counter-evidence fires on a grounded ASSERT, prefer **reconcile** (most
counter-evidence indicates a real perspective gap rather than pure error).
Use `[synthesis-of: <support-id>, <contra-id>]` tagging and write the
synthesis claim acknowledging both sides.

## Axis 4 — revise enforcement (was new today, ships with #50 A)

After a fire, classify how the agent's next-turn revise addressed the prior
fire's entity. If env enabled, escalate when no productive revise shape was
detected.

| Primitive | Default | Surface |
|---|---|---|
| Per-turn revise check (Stage 2) | ON | `[vouch-gate] revise check: N addressed (X fetch / Y hedge / Z remove)` |
| Backlog count (Stage 1) | ON | `[vouch-gate] revise backlog: N awaiting verification` |
| Strict escalation (Stage 3) | **OFF** | `VOUCH_GATE_ESCALATE_UNADDRESSED=1` → re-fire if prior turn's entity unaddressed |

## Recommended opt-in sequence

For a session where vouch should be maximally strict:

```sh
export VOUCH_GATE_ESCALATE_UNADDRESSED=1    # block on unaddressed prior fires
export VOUCH_GATE_COUNTER_EVIDENCE=1        # block on KB counter-evidence
```

For a session where vouch should be maximally visible-but-non-blocking
(default): leave both env vars unset. Stage 1/2 advisories still render.

## Diagnostic recipes

### Check humility on the current session

```sh
# Render-only — read the next Stop hook stderr for humility/blind-spot lines.
# Or query session_claims directly:
sqlite3 ~/.vouch/store.db "
  SELECT stance, COUNT(*) FROM session_claims
  WHERE transcript_id = '<your-transcript-uuid>'
  GROUP BY stance ORDER BY 2 DESC;
"
```

### Check awaiting_revise backlog

```sh
sqlite3 ~/.vouch/store.db "
  SELECT turn_idx, entity, substr(proposition, 1, 80)
  FROM session_claims
  WHERE transcript_id = '<uuid>' AND awaiting_revise = 1
  ORDER BY turn_idx;
"
```

### Count today's fire-classification distribution

```sh
bench/dogfood/extract_fires.ts --days 1
bench/dogfood/auto_classify_fires.ts --concurrency 6
bench/dogfood/classify_fires.ts --stats
```

## What's still on the roadmap (not shipped today)

1. **P-α automation** — currently P-α renders a SUGGESTION for web counter-check.
   Could auto-execute the suggested search + fetch + NLI. Cost: +2-5s/turn,
   web rate-limit risk. Worth piloting if dogfood data shows users follow
   the suggestion at high rate.

2. **P-β multi-source requirement** — stakes-weighted ≥2 independent dossiers
   for high-confidence ASSERTs. Needs a stakes signal (per-claim?, per-entity?).

3. **Confidence calibration** — `[confidence: 0.X]` tag for ATOMIC claims;
   calibration error tracked over time. Currently soft_score exists for
   derived-claim types but not for ATOMIC.

4. **Pre-write check** — intercept drafts BEFORE Stop hook (e.g. via
   UserPromptSubmit) and force retrieval before drafting. The Phase 1
   ABSTAIN-Pressure bench's revise prompt already does this in controlled
   conditions; production integration is the question.

5. **Silent-rephrase detection (#50 A Stage 3.5)** — embedding-similarity
   check on the 'remove' classification: if the new draft has a HIGH-cosine
   proposition to the fired one but the entity is gone, that's silent-
   rephrase dodge, not topic-drop. Currently 'remove' clears the awaiting
   ledger unconditionally.

## Today's commits (chronological, 18:51 → 06:00 UTC)

```
9cea08f  fix(#46): UserPrompt source — close #46
41943a5  chore(#47): SQuAD-2 audit — close #47
584d17d  fix(#48): subject-predicate alignment — close #48
7ffb3f2  docs(bench): PHASE1-RESULTS.md launch artifact
e853516  fix(#49): quote-in-dossier invariant — close #49
0cfd238  bench(dogfood): classify_fires.ts + post_fire_draft
978166e  bench(dogfood): auto_classify_fires.ts + audit-mode
f560a32  bench(dogfood): week-0 baseline (dodge_rate 64.5%)
2956458  feat(#50 B): vouch search suggester (later: 0% lift on dodge corpus)
4afb5ae  feat(#50 E): inline KB candidates (later: 0.9% upper-bound lift)
d9292ac  bench(dogfood): counterfactual_E.ts + diagnosis reverse
7918118  feat(#50 P1): per-session fire counter
e4ee6c4  feat(#50 A Stage 1): awaiting_revise ledger
469b72c  feat(#50 A Stage 2): classifyReviseAction + advisory
f9228ec  feat(#50 A Stage 3): opt-in escalation env=1
07e43f0  feat(P-α): counter-evidence pull
7b3418b  feat(P-γ Stage 1): humility metric
5d72fc4  feat(P-γ.5): blind-spot enumeration
908c62c  docs(skill): teach agents to USE today's primitives
c91a33b  feat(P-γ Stage 2): per-turn over-confidence advisory
71dfaf4  feat(P-α.5): web-counter-evidence suggestion
<this commit> docs(dogfood): primitives reference index
```

24 commits across this session. Three new axes added to vouch's surface.
