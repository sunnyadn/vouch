#!/usr/bin/env bash
# Subscription-backed repro worker — a FREE, kimi-independent replacement for kimi-task, for
# generating author-out-of-loop live quant cases. Runs a headless `claude -p` on the logged-in
# SUBSCRIPTION (env-strips ANTHROPIC_API_KEY so it falls back to the plan, proven 2026-06-19).
# The worker runs in CWD; vouch's PostToolUse hook captures its trace into CWD/.vouch-trace.jsonl;
# the worker writes its own CONCLUSION.md. Then review with:
#   bun bench/quant-repro/review-live-repro.ts --backend claude-p --dir <CWD>
#
# Usage: bench/quant-repro/sub-worker.sh <cwd> <prompt-file>
set -uo pipefail
CWD="$1"; PROMPT_FILE="$2"
CLAUDE_BIN="${CLAUDE_BIN:-$HOME/.local/bin/claude}"
PROMPT="$(cat "$PROMPT_FILE")"
# Optional MODEL env picks the worker model (e.g. haiku) — a WEAKER worker overclaims more, which is
# what you want when generating FIRE cases to test the gate's RECALL (opus workers are too careful).
MODEL_ARG=()
[ -n "${MODEL:-}" ] && MODEL_ARG=(--model "$MODEL")
cd "$CWD"
env -u ANTHROPIC_API_KEY -u ANTHROPIC_BASE_URL -u ANTHROPIC_AUTH_TOKEN \
  "$CLAUDE_BIN" --print --output-format json --permission-mode bypassPermissions "${MODEL_ARG[@]}" "$PROMPT" \
  > worker-run.json 2>worker-run.err
RC=$?
echo "worker exit=$RC  cwd=$CWD"
echo "trace events: $(wc -l < .vouch-trace.jsonl 2>/dev/null || echo 0)  | CONCLUSION.md: $([ -f CONCLUSION.md ] && echo yes || echo MISSING)"
exit $RC
