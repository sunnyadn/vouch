# vouch

**vouch is a passive anti-hallucination gate for AI coding agents.** It watches
the agent work and catches the moments where its words drift from what it
actually did or verified — fabricated results, conclusions with no investigation
behind them, "I searched and found nothing" without a search, a claim about an
external library or paper with no source fetched.

It is *passive*: the agent calls no commands and follows no protocol. Hooks do
everything. This is the whole point — a system the agent has to cooperate with
is a system the agent can forget to use, or game. vouch works by watching, not
by asking.

## Two grounding axes, one rule

Every claim must be backed by evidence in the session trace:

- **Own-work claims** — "tests pass", "the bug is fixed", "I refactored X" — are
  checked against what the agent actually ran and observed (the recorded tool
  results).
- **External claims** — "Drizzle is the leading TypeScript ORM", "the paper says
  Y", "library Z supports W" — must be backed by a `WebSearch`/`WebFetch` in the
  trace whose result actually supports them. Training memory ≠ verified knowledge.

(Earlier vouch enforced external grounding through an explicit Fetch-Before-Claim
protocol with a persistent claim KB. That is retired: the agent fetches sources
during normal work anyway, and the trace already captures them — so the reviewer
audits both axes passively, with no ceremony and no KB to maintain.)

## What the reviewer catches

- **Active fabrication** — a claim that contradicts the evidence ("all tests
  pass" while the recorded run shows failures).
- **Passive fabrication** — a claim with no supporting evidence, own-work or
  external, including absence claims asserted with no search.
- **Research insufficiency** — a conclusion whose scope exceeds the evidence
  ("performance resolved" after fixing one query).
- **Decision contradicts a project lesson** — a choice that repeats a documented
  mistake (the reviewer reads the project's auto-memory).
- **Omission** — unresolved failures the summary doesn't acknowledge.

## How a catch works

Three steps, all automatic — the agent does nothing:

1. **Every tool result is recorded.** A `PostToolUse` hook appends every command,
   file read, and search — *with its output* — to a session trace. Never a filtered
   subset: a capture-time filter is itself the omission the capture exists to catch.
2. **At commit and turn-end, the reviewer reads that trace.** For each factual claim
   the agent makes, it queries the trace — *did the agent actually run this test,
   read this file, fetch this source?* — instead of trusting the prose.
3. **Ungrounded claims are blocked**, with the exact offending span, why it's
   unsupported, and what to do about it.

### Three failures you already know

**1. The fake "it's done."** The agent wraps up: *"Fixed the login bug — all 142
tests pass."* You're one click from trusting it. But the trace shows the test run
came back with failures (or never ran at all). vouch stops the commit cold:

```
⛔ vouch reviewer (BLOCK): [active-fabrication]
   "all 142 tests pass" — the recorded run shows 3 failures.
   → Fix the failures or report them; don't claim green over red.
```

It can't ship the lie. The only way past the gate is to make the claim *true*.

**2. The invented API.** *"Just pass `{ retry: true }` to `fetch()` — it retries
automatically."* Confident, plausible, and completely made up — and the trace shows
no docs were ever opened. vouch blocks it as an external claim with no source:
training memory is not verified knowledge. One `WebFetch` of the real docs either
grounds the claim or exposes it.

**3. The premature all-clear.** *"Audited the codebase — no security issues."* The
trace shows it read two files. vouch flags the gap: a sweeping "no issues" backed by
a sliver of evidence is the most dangerous false confidence there is. Narrow the
claim to what you actually checked — or go check the rest.

## How it runs

Two layers, cheapest first:

- **Deterministic commit gate** (free, no API key). Parses test-count claims in a
  pending commit and **blocks** when they contradict a recorded run. No LLM.
- **LLM reviewer** (nuanced, agentic). At commit and at the end of a turn it
  queries the full trace and compares the agent's claims against what it actually
  observed. Fail-open: a reviewer error never breaks the session — so run
  `vouch doctor` to confirm it's actually live (a dead key / drained quota fails
  silently).

There is no self-bypass: the commit gate has no env-var off-switch.

## Install

vouch is the **`vouch` CLI** (globally on your `PATH`) plus a **hooks manifest**.
The hooks call `vouch` by name. **Terminal-launched Claude Code only** — a hook
can't reliably find a PATH-installed binary under the desktop app's environment.
Requires [Bun](https://bun.sh) ≥ 1.3.

1. **Install the CLI globally** from a clone:

```bash
git clone https://github.com/sunnyadn/vouch
cd vouch
bun install
bun link          # global `vouch` → this clone (keep the clone around)
```

2. **Configure the reviewer** — copy `.env.example` to `.env` and set an
   `ANTHROPIC_API_KEY` (and `ANTHROPIC_BASE_URL` / `VOUCH_REVIEWER_MODEL` if you
   use a compatible gateway like DeepSeek or kimi). Then:

```bash
vouch doctor      # confirms the API key, endpoint, a live round-trip, the gate,
                  # project-memory, and trace capture
```

3. **Install the hooks** — the repo is its own marketplace
   (`.claude-plugin/marketplace.json` → `./plugin`):

```text
/plugin marketplace add /path/to/this/repo
/plugin install vouch@vouch
```

   Then restart the session and re-run `vouch doctor` to confirm.

| Hook | What it does |
| --- | --- |
| `PreToolUse` (Bash) | `pre-commit-gate` — block a commit whose claims contradict recorded runs |
| `PreToolUse` (Edit/Write) | `pre-edit-gate` — warn when editing a file not read this session |
| `PostToolUse` / `…Failure` | `trace-append` — capture every tool result as evidence |
| `Stop` | `stop-review` — review the agent's final response |

## Project knowledge

The reviewer reads the project's **auto-memory**
(`~/.claude/projects/<encoded>/memory/*.md`) so it can catch a decision that
contradicts something the project already learned. There is no separate store.

## Develop

```bash
bun test
bun run lint
```
