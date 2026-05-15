# vouch layer architecture — 2026-05-15

**Status:** specification + retroactive inventory + migration backlog.
**Author:** sunny + Claude Code.
**Trigger:** today's session produced 4 prod commits all at L5 (post-hoc
gate). Audit revealed gate.ts at 3000+ LOC because L2/L3 are empty —
every "make the agent verify" responsibility falls through to the only
layer that has tools to swing.

This doc fixes that by (a) explicitly partitioning vouch into 5 layers,
(b) inventorying current state, (c) flagging which existing gate.ts
logic belongs elsewhere, (d) setting LOC red lines per layer.

## The 5 layers

| # | Layer | Vehicle | Job | LOC cap |
|---|---|---|---|---|
| L1 | Training | export `session_claims` as training corpus | shape disposition; out of scope for runtime but vouch can produce the data | n/a |
| L2 | Agent system prompt | plugin's own `CLAUDE.md` fragment | encode `"before asserting about external named entity, verify; track your own humility ratio; name what you didn't check"` as baked rules every turn | ≤ 200 lines of rules |
| L3 | Pre-prompt augmentation | `UserPromptSubmit` hook | scan user prompt → extract mentioned entities → KB lookup → inject structured "entities + verification status + suggested action" block into agent context BEFORE agent drafts | hook script ≤ 300 LOC |
| L4 | In-flow tools | `vouch search` / `vouch fetch` / `vouch claim` / `vouch attest` / `vouch session show` | give agent verbs to call mid-flow when it wants to verify, attest, or check | each command ≤ 500 LOC |
| L5 | Post-draft gate | `vouch gate` (Stop hook) | minimal blocker on ungrounded ASSERTs — extractor + grounder + block decision. **Not** a place to drive behavior; behavior belongs in L2/L3 | **≤ 800 LOC** (currently 3000+) |

**Hard rule.** Each layer answers a different question:

- L2: "what should the agent *always* do?"
- L3: "what should the agent know *before this turn*?"
- L4: "what verbs can the agent invoke?"
- L5: "what should we refuse to ship?"

A feature that conflates these is mis-layered.

## Retroactive inventory: where is current gate.ts logic?

Audit of `src/gate.ts` (3000+ LOC at HEAD = `cc1dbf9`). Tagged with
target layer:

| Logic in gate.ts | LOC est. | Current layer | Should be |
|---|---|---|---|
| ASSERT extractor (proposition + stance) | ~400 | L5 | **L5 stays** — this is the gate's core job |
| KB grounder (hybrid search + NLI) | ~300 | L5 | **L5 stays** — pre-block check |
| Block decision (`--strict` exit code) | ~50 | L5 | **L5 stays** |
| Session-source auto-ground (issue #21) | ~200 | L5 | **L5 stays** — efficiency on L5 |
| Workspace-meta post-filter (#40) | ~300 | L5 | **L3 candidate** — entity classification pre-pass could happen at pre-prompt |
| Hedge escape closure (#42) | ~50 | L5 | **L2** — agent should be told what counts as a real hedge; not gate's job to police |
| Revise check / escalation (#50) | ~250 | L5 | **L5 stays** — backlog tracking is fire-time business |
| Over-confidence flag (P-γ Stage 2) | ~100 | L5 | **L2** — agent reads "your humility budget is low this turn" from prompt, not from fire stderr |
| Blind-spot detector (P-γ.5) | ~50 | L5 | **L2** — `"name what you didn't check"` is a writing rule, not a regex check |
| Humility ratio surface (P-γ Stage 1) | ~50 | L5 | **L3** — inject "humility 8.4% this session, target 15%+" into pre-prompt context |
| P-α counter-evidence pull | ~200 | L5 | **L4** — make `vouch counter <claim>` an explicit tool the agent calls when adversarially probing |
| P-α.5 web-counter suggestion | ~30 | L5 | **L2** rule (`"when claim is unfalsifiable in KB, run vouch search ... limitations"`) |
| **Exa inline (today, 1f2e955)** | ~80 | L5 | **L3** — pre-prompt scan should pre-fetch the canonical source, not the gate at fire time |
| **Dossier persist (today, cc1dbf9)** | ~40 | L5 | **L3** — same: pre-prompt fetch writes the dossier |
| **Verify-driven message rewrite (today, e6c12a7)** | ~30 | L5 | **L2** — `"silent delete hides what you didn't check"` is a writing rule baked into agent prompt, not a per-fire string |
| Harvest pipeline (`[inference-from:]` etc.) | ~400 | L5 | **L4** — these are user-authored tags the agent emits; harvest is agent-driven, should be a `vouch harvest` tool |
| ANSI color helper (today) | ~10 | L5 | **L5 stays** trivially |

**Roll-up**: of ~3000 gate.ts LOC, **~1530 LOC (51%) belongs in L2 or L3**.
Migrating that out drops gate.ts to ~1500 LOC — still over the 800 cap
but tractable.

## What needs to exist for migration

L2 — **plugin CLAUDE.md fragment** (currently empty). Distribute via
Claude Code plugin manifest; gets loaded on plugin install. Contains:

- "before asserting about a named external entity, verify via vouch
  search or hedge explicitly"
- "track humility ratio per turn; if 0 hedges/speculates in 3+ ASSERTs,
  add at least one `[gap: …]` or hedge"
- "silent delete hides what you didn't check; if you delete, name what
  you removed and why"
- "for claims about your own working repo or vouch's own internals,
  attest via `vouch attest --from-session-tool` not `vouch claim`"

L3 — **UserPromptSubmit hook** (currently empty). New hook script
`hooks/userpromptsubmit.ts`. Job per invocation:

1. Read user prompt from stdin payload
2. Light entity extractor (cheap LLM call or reuse gate's extractor)
3. For each entity, KB lookup + recent-dossier check
4. If KB miss on a load-bearing entity → optional Exa pre-fetch (write
   dossier + L1-derived claim summary)
5. Inject structured text block into the next system context

The L3 hook is **where Exa inline + dossier persist belong** — fired
once per user turn, not once per gate fire. This eliminates the "every
fire pays Exa" cost concern that motivated the dossier cache.

L4 — **tools already exist** (search/fetch/claim/attest). Two
additions:

- `vouch counter <proposition>`: the P-α path the gate currently runs
  automatically. Surface as an agent-callable verb so "go look for
  contradictions on X" is a thing the agent can do explicitly.
- `vouch harvest <file>`: extract `[inference-from:]` / `[gap: …]`
  tagged derived claims out of any markdown the agent writes. Currently
  inlined in the Stop hook; should be a tool the agent invokes.

L5 — **gate stays, slims**. Migration steps:

- Strip ~1500 LOC of L2/L3 logic
- Cap at 800 LOC; CI fails on exceeding

## Discipline (forcing functions)

**Rule 1 — PR-level layer declaration.**
Every PR's commit message must start with a layer tag: `L2:` / `L3:` /
`L4:` / `L5:`. Cross-layer commits use the highest. Reviewer rejects
ambiguous tags.

**Rule 2 — LOC red lines (CI enforced).**
- `src/gate.ts`: ≤ 800 LOC. CI fails on PR that pushes it over.
- Per L4 command file: ≤ 500 LOC.
- L3 hook scripts: ≤ 300 LOC each.

**Rule 3 — Pre-commit layer audit.**
Before adding to gate.ts, the contributor (human or LLM) must
explicitly answer: *"why can't this be in L2 or L3?"* in the commit
message. Saying "it was easier to put it in L5" rejects.

**Rule 4 — Periodic distribution audit (monthly).**
Generate per-layer LOC distribution; if L5 share > 60% of total, halt
new L5 work, prioritize migration.

## Migration backlog

Priority-ordered. Each item is a separate PR.

1. **[L2 + L3] Stand up plugin skeleton** — create `@vouch/plugin`
   directory layout (plugin.json + CLAUDE.md fragment + hooks/ +
   skills/). 1-2 hrs.
2. **[L3] UserPromptSubmit hook v0** — entity extraction + KB lookup +
   structured context injection. Migrates Exa inline (1f2e955) + dossier
   persist (cc1dbf9). 3-4 hrs.
3. **[L2] CLAUDE.md rules v0** — write the ~10 baked-in rules.
   Migrates verify-driven message (e6c12a7) + blind-spot rule (P-γ.5) +
   hedge escape rule (#42). 1 hr.
4. **[L5] Strip gate.ts** — delete migrated logic, target ≤ 1500 LOC
   intermediate, ≤ 800 final after 5-7 lands. 2-3 hrs.
5. **[L3] humility ratio injection** — pre-prompt shows current session
   ratio + target. Migrates P-γ Stage 1 surfacing. 1 hr.
6. **[L4] `vouch counter` tool** — extract P-α path to agent-callable
   verb. 2 hrs.
7. **[L4] `vouch harvest` tool** — extract harvest pipeline to verb. 2 hrs.

**Total est.**: 12-15 hrs of focused work. Spread across 2-3 days.

## Rollback policy

If migrating L5 → L3 (e.g., the UserPromptSubmit hook) regresses
behavior on the session_claims metrics (humility ratio drops,
addressed_via shifts further toward `remove`), keep the L5 piece in
parallel for 2 weeks, then decide. Don't blindly delete L5 logic
without forward measurement.

## What this doc commits us to

- gate.ts is **post-hoc minimal blocker**, not a behavior engine
- agent behavior changes go in L2 (rules) or L3 (per-turn context),
  never new L5 code
- new external evidence (web fetch / source caching) is L3 work,
  triggered by user prompt, not by L5 fire
- LOC red lines enforce these or we're back to today's mess in 6 weeks

## What this doc explicitly does NOT commit us to

- replacing existing L4 tool commands
- new training pipeline (L1 export is a downstream thought, not in
  current scope)
- removing the gate; gate stays, just slims
- a particular hook implementation language (the existing Stop hook is
  TypeScript via the compiled `vouch` binary; UserPromptSubmit can be
  the same shape)
