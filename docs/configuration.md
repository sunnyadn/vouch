# Configuration

vouch reads its config from a `.env` file in the project directory — Bun auto-loads it on
every hook fire. Run `vouch doctor` after any change to confirm it took effect.

## The reviewer's LLM

The reviewer needs an Anthropic-compatible endpoint and key.

| Variable | What it does |
| --- | --- |
| `ANTHROPIC_API_KEY` | **Required.** Without it the LLM reviewer is OFF (only the deterministic gate runs). |
| `ANTHROPIC_BASE_URL` | Endpoint. Default `api.anthropic.com`. |
| `VOUCH_REVIEWER_MODEL` | Model id as named on your gateway. |
| `VOUCH_REVIEWER_OFF` | Set to `1` to skip the LLM reviewer deliberately (the deterministic gate still runs). |
| `VOUCH_CORPUS_PATH` | Where verdicts are logged. Default `~/.claude/vouch-corpus.jsonl`. |

> **A Claude Code subscription / OAuth login does _not_ export a key.** Without one the
> reviewer silently no-ops and only the deterministic gate runs. `vouch doctor` flags this.

### Gateways

Any Anthropic-compatible endpoint works — set the base URL and model:

- **DeepSeek** — pay-per-token, no hard quota:
  ```
  ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
  VOUCH_REVIEWER_MODEL=deepseek-v4-pro
  ```
- **Moonshot / kimi**:
  ```
  ANTHROPIC_BASE_URL=https://api.kimi.com/coding/
  VOUCH_REVIEWER_MODEL=kimi-k2.6
  ```

## `vouch doctor`

`vouch doctor` is the only reliable way to confirm the reviewer is actually alive — it
makes a **real round-trip**, bypassing the fail-open path. It reports: the API key, the
endpoint, the model, a timed round-trip, the deterministic gate, project-memory load, and
trace-capture writability.

> **The reviewer fails open — on purpose.** A reviewer error (dead key, drained quota,
> network blip) never breaks your session; it just produces an empty verdict. To keep that
> from being **silent**, a failed-open review now prints a one-line, non-blocking notice at
> the turn/commit it happened on:
>
> ```
> ⚠ vouch reviewer unavailable — it failed open and reviewed nothing this turn
>    (drained quota, bad key, or timeout). You are UNGATED until it recovers. Run `vouch doctor`.
> ```
>
> Every verdict is also tagged `status: reviewed | skipped | failed` in the corpus, so a
> clean pass (`reviewed`, empty) is distinguishable from a death (`failed`, empty). **If you
> see the notice — or just stop seeing vouch fire — run `vouch doctor`.**
