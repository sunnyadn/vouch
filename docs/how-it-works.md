# How vouch works

vouch is *passive*: the agent calls no commands and follows no protocol. A set of
Claude Code hooks record what the agent does and check what it says against that.
There is nothing to cooperate with, so nothing to forget or game — vouch watches, it
doesn't ask.

## Two grounding axes

Every factual claim must be backed by evidence in the session trace, on one of two axes:

- **Own-work claims** — *"tests pass"*, *"the bug is fixed"*, *"I refactored X"* — are
  checked against what the agent actually ran and observed (the recorded tool results).
- **External claims** — *"Drizzle is the leading TypeScript ORM"*, *"the paper says Y"*,
  *"library Z supports W"* — must be backed by a `WebSearch`/`WebFetch` in the trace whose
  result actually supports them. Training memory ≠ verified knowledge.

## The mechanism

Three steps, all automatic:

1. **Every tool result is recorded.** A `PostToolUse` hook appends every command, file
   read, and search — *with its output* — to a session trace. Never a filtered subset: a
   capture-time filter is itself the omission the capture exists to catch.
2. **At commit and turn-end, the reviewer reads that trace.** For each factual claim, it
   queries the trace — *did the agent actually run this test, read this file, fetch this
   source?* — instead of trusting the prose.
3. **Ungrounded claims are blocked**, with the exact offending span, why it's unsupported,
   and what to do about it.

## What the reviewer catches

- **Active fabrication** — a claim that contradicts the evidence (*"all tests pass"* while
  the recorded run shows failures).
- **Passive fabrication** — a claim with no supporting evidence, own-work or external,
  including absence claims asserted with no search.
- **Research insufficiency** — a conclusion whose scope exceeds the evidence (*"performance
  resolved"* after fixing one query).
- **Unfalsifiable conclusion** — a causal claim or a *"fixed it"* with no discriminating
  test behind it (see below). A claim that can be neither confirmed nor refuted isn't
  grounded.
- **Decision contradicts a project lesson** — a choice that repeats a documented mistake
  (the reviewer reads the project's auto-memory; see below).
- **Omission** — unresolved failures the summary doesn't acknowledge.

## Falsifiability — demand the discriminating test

The most expensive bad conclusions aren't false, they're *unfalsifiable as stated*: the
agent never said what observation would prove it wrong, so it can neither confirm nor refute
its own claim. This bites hardest in **causal attribution** — *"the test failed **because**
of the cache, so I disabled it"* — where the cause is asserted but never isolated.

For a causal or *"fixed it"* claim, vouch checks the trace for a **discriminating test** —
an observation that would have come out *differently* if the claim were false:

- a **fix** needs a **red→green**: a check that *failed before* the change and *passes
  after*. No before-state, no fix — just a change that happens to coexist with green.
- a **cause** needs **isolation**: *"if the cache is the cause, disabling it makes the test
  pass"* — and then actually running it. If it still fails, the cache wasn't the cause.

When the discriminating test is missing, vouch flags the claim (blocking if it's conclusive)
and, in the suggestion, names the falsification the agent should *state and run* — turning a
confident guess into a checked result. It is treated as research insufficiency: a conclusion
with no way to be wrong is a conclusion with no evidence.

## Two layers

Cheapest first:

- **Deterministic commit gate** (free, no API key). Parses test-count claims in a pending
  commit and **blocks** when they contradict a recorded run. No LLM, so no flakiness.
- **LLM reviewer** (nuanced, agentic). At commit and at the end of a turn it queries the
  full trace and compares the agent's claims against what it actually observed.

There is **no self-bypass**: the commit gate has no env-var off-switch. The LLM reviewer
**fails open** — an error never breaks the session — which means an outage is silent; see
[Configuration → `vouch doctor`](configuration.md) for how to confirm it's live.

## The hooks

| Hook | What it does |
| --- | --- |
| `PreToolUse` (Bash) | `pre-commit-gate` — block a commit whose claims contradict recorded runs |
| `PreToolUse` (Edit/Write) | `pre-edit-gate` — warn when editing a file not read this session |
| `PostToolUse` / `…Failure` | `trace-append` — capture every tool result as evidence |
| `Stop` | `stop-review` — review the agent's final response |

## Project knowledge

The reviewer reads the project's **auto-memory**
(`~/.claude/projects/<encoded>/memory/*.md` — the same lessons the agent's own memory
loads) so it can catch a decision that contradicts something the project already learned.
There is no separate knowledge store to maintain; the memory files are the single source.

## History: the retired Fetch-Before-Claim model

Earlier vouch enforced external grounding through an explicit **Fetch-Before-Claim**
protocol with a persistent claim KB — the agent had to run `vouch fetch` / `vouch claim`,
pick a verbatim quote, and let an NLI judge verify it. That is retired. The agent fetches
sources during normal work anyway, and the trace already captures them — so the reviewer
audits both axes passively, with no ceremony and no KB to maintain.
