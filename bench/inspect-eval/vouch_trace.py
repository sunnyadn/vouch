"""vouch's trace layer, ported to Python for the inspect eval.

Mirrors the TypeScript source:
  - event extraction   src/core/evidence-capture.ts
  - queryHistory       src/core/reviewer-agentic.ts:34
  - windowAroundMatch  same file :69
  - formatHits         same file :81
  - buildHistoryIndex  same file :113

The TS side is the authoritative implementation. This exists so the same semantics
can run inside an inspect solver. Change one, sync the other, or this eval stops
measuring the reviewer that actually ships.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CapturedEvent:
    tool: str
    command: str | None = None
    file_path: str | None = None
    exit_code: int = 0
    stdout: str = ""


def _event_from_hook(raw: dict) -> CapturedEvent | None:
    """One PostToolUse / PostToolUseFailure hook record -> CapturedEvent."""
    tool = raw.get("tool_name")
    if not tool:
        return None
    tool_input = raw.get("tool_input") or {}
    response = raw.get("tool_response")
    failed = raw.get("hook_event_name") == "PostToolUseFailure"

    if tool == "Bash":
        command = tool_input.get("command")
        if failed:
            # A failed Bash call carries no tool_response; its output is the
            # top-level `error` field.
            return CapturedEvent(tool, command=command, exit_code=1,
                                 stdout=raw.get("error") or "")
        out = ""
        if isinstance(response, dict):
            out = response.get("stdout") or ""
            if response.get("stderr"):
                out = f"{out}\n{response['stderr']}"
        return CapturedEvent(tool, command=command, stdout=out)

    if tool == "Read":
        content = ""
        if isinstance(response, dict):
            file_block = response.get("file")
            if isinstance(file_block, dict):
                content = file_block.get("content") or ""
        return CapturedEvent(tool, file_path=tool_input.get("file_path"),
                             stdout=content)

    if tool in ("Write", "Edit"):
        # What was written IS this step's output: the reviewer has to be able to
        # query for what the agent put on disk.
        content = tool_input.get("content") or tool_input.get("new_string") or ""
        if not content and isinstance(response, dict):
            content = response.get("content") or ""
        return CapturedEvent(tool, file_path=tool_input.get("file_path"),
                             stdout=content)

    if tool in ("WebSearch", "WebFetch"):
        return CapturedEvent(tool,
                             command=tool_input.get("query") or tool_input.get("url"),
                             stdout=json.dumps(response)[:4000]
                             if response is not None else "")

    return CapturedEvent(tool, exit_code=1 if failed else 0)


def load_trace(path: str | Path) -> list[CapturedEvent]:
    events: list[CapturedEvent] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        event = _event_from_hook(json.loads(line))
        if event is not None:
            events.append(event)
    return events


def _window_around_match(out: str, pattern: re.Pattern[str], budget: int) -> str:
    """Window the output on the MATCH, not the head.

    A head-only slice made the gate miss facts reported from the tail of long
    outputs (it saw 3 of 6 probe reps) and cry-wolf. The truncation marker has to
    stay visible, or the reviewer reads "I can't see it" as "it isn't there".
    """
    if len(out) <= budget:
        return out
    match = pattern.search(out)
    center = match.start() if match else 0  # matched the command/path, not the body
    start = max(0, center - budget // 3)
    end = min(len(out), start + budget)
    head = f"…[{start} chars before] " if start > 0 else ""
    tail = f" …[+{len(out) - end} more chars]" if end < len(out) else ""
    return f"{head}{out[start:end]}{tail}"


def query_history(events: list[CapturedEvent], pattern: str, *,
                  max_hits: int = 12, per_output: int = 2000) -> list[dict]:
    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error:
        regex = re.compile(re.escape(pattern), re.IGNORECASE)

    hits: list[dict] = []
    for event in events:
        haystack = "\n".join([event.command or "", event.file_path or "",
                              event.stdout or ""])
        if not regex.search(haystack):
            continue
        hits.append({
            "tool": event.tool,
            "command": event.command,
            "file_path": event.file_path,
            "exit_code": event.exit_code,
            "output": _window_around_match(event.stdout or "", regex, per_output),
        })
    return hits[-max_hits:]


def format_hits(hits: list[dict]) -> str:
    """Order must be chronological AND labelled as such.

    An unlabelled most-recent-first list reads as "the history shows the OPPOSITE
    order" and induced an active-fabrication false fire on a grounded red-green fix
    (bench/verify-replay, R4).
    """
    if not hits:
        return "(no matching events in the session history)"
    parts = []
    for hit in hits:
        if hit["file_path"]:
            label = f"{hit['tool']} {hit['file_path']}"
        else:
            label = f"{hit['tool']} `{hit['command'] or ''}` (exit {hit['exit_code']})"
        parts.append(f"{label}\n{hit['output']}" if hit["output"] else label)
    body = "\n---\n".join(parts)
    return ("Matching events in CHRONOLOGICAL order "
            f"(earliest first → latest last):\n{body}")


def build_history_index(events: list[CapturedEvent]) -> str:
    """Names and paths only, NO outputs. Those are pulled via query_history."""
    reads: list[str] = []
    edits: list[str] = []
    cmds: list[str] = []
    web = 0
    for event in events:
        if event.tool == "Read" and event.file_path:
            if event.file_path not in reads:
                reads.append(event.file_path)
        elif event.tool in ("Edit", "Write") and event.file_path:
            if event.file_path not in edits:
                edits.append(event.file_path)
        elif event.tool in ("WebSearch", "WebFetch"):
            web += 1
        elif event.tool == "Bash" and event.command:
            mark = "✓" if event.exit_code == 0 else "✗"
            cmds.append(f"{mark} {event.command[:140]}")

    lines = [f"Total events: {len(events)}"]
    lines.append(f"Files read ({len(reads)}): {', '.join(reads[-40:]) or 'none'}")
    lines.append(f"Files edited ({len(edits)}): {', '.join(edits[-40:]) or 'none'}")
    lines.append(f"Web searches: {web}")
    lines.append(f"Commands run ({len(cmds)}, query_history for output):")
    for cmd in cmds[-40:]:
        lines.append(f"  {cmd}")
    if len(cmds) > 40:
        lines.append(f"  … {len(cmds) - 40} older (query_history to reach them)")
    return "\n".join(lines)
