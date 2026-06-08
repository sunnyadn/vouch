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

### Typical catches

| The agent claims… | …but the trace shows | vouch's move |
| --- | --- | --- |
| "all tests pass" / "fixed it" | the recorded test run failed, or none ran | **block** — run them, report the real result |
| "updated **all** the call sites" | files edited, no search for the usages | **block** — grep first, or say "the ones I found" |
| "the default is 4 threads" | a file it read earlier this session says `16` | **block** — contradicts what it already saw |
| "as we established, X works like Y" | nothing this session backs it — it's memory | **block** — cite this session's evidence, or re-check |
| "…all set up cleanly" | an install/command errored mid-session | **flag** — acknowledge the failure you skipped |
| "there's no library for this" | no `WebSearch`/`WebFetch` in the trace | **block** — search before claiming absence |
| "`Promise.all` cancels the rest" | a precise API claim, no docs fetched | **block** — fetch the source; memory ≠ knowledge |
| "no security issues" | a sweeping conclusion, two files read | **flag** — scope it to what you actually checked |
| "decision: skip the cache layer" | repeats a mistake the project's memory records | **block** — the project already learned this |

Two are worth their weight alone: it catches a claim that contradicts something the
agent **read 40 tool-calls earlier** (perfect session recall, which you'd never track),
and it stops *"there's no library for X"* — the most expensive sentence an agent says —
by making it run the search first. Every block names the exact span, why it's
unsupported, and the fix:

```
⛔ vouch reviewer (BLOCK): [active-fabrication]
   "defaults to 4 threads" — worker.config.ts (read earlier) shows poolSize: 16.
   → re-read the file or correct the number.
```

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
