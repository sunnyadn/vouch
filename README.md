# vouch

**A passive anti-hallucination gate for AI coding agents.** It records everything the
agent *does* — every command, file read, and search — then blocks the moments its words
drift from that: fabricated results, conclusions with no investigation behind them,
*"there's no library for X"* asserted without a search. The agent follows no protocol and
can't forget to use it — the hooks do everything. vouch watches; it doesn't ask.

## What it catches

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

Two earn their keep alone: it catches a claim that contradicts something the agent **read
40 tool-calls earlier** (perfect session recall, which you'd never track), and it stops
*"there's no library for X"* — the most expensive sentence an agent says — by making it run
the search first. Every block names the exact span, why, and the fix:

```
⛔ vouch reviewer (BLOCK): [active-fabrication]
   "defaults to 4 threads" — worker.config.ts (read earlier) shows poolSize: 16.
   → re-read the file or correct the number.
```

**[→ How it works](docs/how-it-works.md)** — the mechanism, the two grounding axes, the
two layers (free deterministic gate + agentic LLM reviewer), and what each catch means.

## Install

**Terminal-launched Claude Code only.** Requires [Bun](https://bun.sh) ≥ 1.3.

```bash
# 1 — the CLI, global on your PATH
git clone https://github.com/sunnyadn/vouch && cd vouch
bun install && bun link

# 2 — the reviewer's key
cp .env.example .env     # set ANTHROPIC_API_KEY (+ BASE_URL / VOUCH_REVIEWER_MODEL)
vouch doctor             # confirm the key, endpoint, and a live round-trip
```

```text
# 3 — the hooks (the repo is its own marketplace)
/plugin marketplace add /path/to/this/repo
/plugin install vouch@vouch       # then restart the session
```

**[→ Configuration](docs/configuration.md)** — DeepSeek / kimi gateways, model choice, the
`VOUCH_REVIEWER_OFF` switch, and what each `vouch doctor` check means (and why a quiet vouch
might be a dead one).

## Develop

```bash
bun test
bun run lint
```
