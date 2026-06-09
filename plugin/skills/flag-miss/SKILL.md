---
description: >-
  Capture a vouch reviewer MISS when the user indicates that a claim in Claude's
  recent response was hallucinated, fabricated, or asserted without actually
  verifying it against the trace — and vouch's gate did NOT catch it. Trigger on
  pushback like "you made that up", "that's a hallucination", "you didn't
  actually run/check that", "where's the evidence", "that's not grounded", or any
  correction that a stated result/conclusion had no support. Records the claim and
  the exact trace vouch saw as human-labeled gold corpus.
allowed-tools: Bash(vouch flag:*)
---

A claim in your recent response was **ungrounded** — a hallucination, a fabricated
result, or something you asserted without verifying against the trace — and vouch's
reviewer missed it. This is vouch's highest-signal event: a human-caught miss is the
only **unfakeable recall gold**, because the reviewer can't grade its own work.
Capture it now, without making the user do anything:

1. **Run this** — you fill in both parts from the conversation, never ask the user:

   ```bash
   vouch flag "<the exact ungrounded claim, quoted> — ungrounded because <what the trace did or didn't show>"
   ```

   `vouch flag` auto-attaches the most recent review's response and the full trace
   vouch saw; you only supply the claim and the reason.

2. **Confirm in one line** that the miss was captured, then fix the actual problem the
   user raised.

Only trigger for a genuinely ungrounded **claim** (a result/conclusion/fact with no
backing) — not ordinary disagreement, style feedback, or a judgment call. When in
doubt, skip the capture and just address the correction.
