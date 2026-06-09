// CLI dispatcher — hook handlers only. Project knowledge lives in auto-memory
// (read by the reviewer's loadProjectFindings); there is no finding store.

import { Command } from "commander";
import { appendTrace, readTrace } from "../core/active-task.ts";
import type { CapturedEvent } from "../core/evidence-capture.ts";
import { setSilent } from "../core/log.ts";
import type { ReviewVerdict } from "../core/reviewer.ts";

// Shared reviewer for both gates: the agentic reviewer — a tool-use loop that queries the
// FULL un-windowed trace, fixing the windowing/truncation/post-commit-summary cry-wolves.
// (reviewer-factored.ts is a researched burial-robust candidate, kept in bench until live
// evidence earns a swap — see probe27-29.)
async function runReviewer(args: {
  action: string;
  actionType: "commit" | "stop-response";
  allEvents: CapturedEvent[];
}): Promise<ReviewVerdict> {
  const { loadProjectFindings } = await import("../core/reviewer.ts");
  const projectFindings = await loadProjectFindings();
  const { anthropicReviewerAgentic } = await import("../core/reviewer-agentic.ts");
  return anthropicReviewerAgentic({
    action: args.action,
    actionType: args.actionType,
    events: args.allEvents,
    projectFindings,
  });
}

export async function dispatch(argv: string[]): Promise<number> {
  const program = new Command();
  program
    .name("vouch")
    .description("vouch — anti-hallucination system for AI agents")
    .version("0.2.0");

  // ----- hook (Claude Code hook entrypoints; event JSON arrives on stdin)
  const hook = program
    .command("hook")
    .description("Claude Code hook entrypoints (read the event payload on stdin)");
  hook
    .command("trace-append")
    .description("PostToolUse: append event to trace + inject grounding text for test/build")
    .action(async () => {
      try {
        const text = await Bun.stdin.text();
        if (!text.trim()) return;
        let event: Record<string, unknown>;
        try {
          event = JSON.parse(text);
        } catch {
          return;
        }
        await appendTrace(event);
        const { parseCapturedEvents, groundingSummary } = await import(
          "../core/evidence-capture.ts"
        );
        const [captured] = parseCapturedEvents([event]);
        if (captured) {
          const summary = groundingSummary(captured);
          if (summary) process.stdout.write(`${summary}\n`);
        }
      } catch {
        // never break the session
      }
    });
  hook
    .command("pre-commit-gate")
    .description("PreToolUse BLOCK: blocks a git commit whose claims contradict observed evidence")
    .action(async () => {
      // No env-var bypass — the agent can write .env, so any check here is
      // self-bypassable. If the reviewer blocks, present the issue to the user.
      try {
        let payload: { tool_name?: unknown; tool_input?: { command?: unknown } };
        try {
          payload = JSON.parse(await Bun.stdin.text());
        } catch {
          return;
        }
        if (payload.tool_name !== "Bash") return;
        const command =
          typeof payload.tool_input?.command === "string" ? payload.tool_input.command : "";
        if (!command) return;
        const { proseCommitGate, extractClaimArgsFromCommand } = await import(
          "../core/prose-stop.ts"
        );
        const { parseCapturedEvents, findLatestRun, observationsOnly } = await import(
          "../core/evidence-capture.ts"
        );
        // Adjudicate claims against OBSERVATIONS only — never the agent's own
        // commit/add actions (their -m text carries prior claims; a failed attempt
        // reads as "fabricated"). See observationsOnly in evidence-capture.
        const allEvents = parseCapturedEvents(await readTrace());
        const traceEvents = observationsOnly(allEvents);

        // Layer 1: deterministic count-compare (free, instant, airtight)
        const r = await proseCommitGate({
          command,
          runsProvider: async (kind) => {
            const run = findLatestRun(traceEvents, kind);
            return run ? [run] : [];
          },
        });
        if (r.blocks) {
          process.stderr.write(r.message);
          process.exit(2);
        }

        // Layer 2: LLM reviewer auditing the commit MESSAGE's claims against the trace.
        // NOT a `git diff` snapshot: at PreToolUse a compound `git add && git commit` hasn't
        // staged yet, so `git diff --cached` would be EMPTY and the review would silently
        // skip (the common-case blind spot). The agentic reviewer already receives the full
        // trace (allEvents) — what changed is in there — and the auditable CLAIMS live in the
        // -m message, so that is the action we review. extractClaimArgsFromCommand is the same
        // message extractor Layer 1 uses, so both layers see the same claims.
        const { shouldCallReviewer, formatReviewMessage, formatReviewerHealthNote } = await import(
          "../core/reviewer.ts"
        );
        const { GIT_COMMIT_RE } = await import("../core/evidence-capture.ts");
        const message = GIT_COMMIT_RE.test(command)
          ? extractClaimArgsFromCommand(command).join("\n").trim()
          : "";
        if (shouldCallReviewer() && message) {
          const verdict = await runReviewer({
            action: message,
            actionType: "commit",
            allEvents,
          });
          const { captureVerdict } = await import("../core/corpus.ts");
          captureVerdict({ actionType: "commit", action: message, events: allEvents, verdict });
          const healthNote = formatReviewerHealthNote(verdict);
          if (healthNote) process.stderr.write(`${healthNote}\n`);
          const reviewMsg = formatReviewMessage(verdict);
          if (reviewMsg) {
            const hasBlock = verdict.issues.some((i) => i.severity === "block");
            process.stderr.write(reviewMsg);
            if (hasBlock) process.exit(2);
          }
        }
      } catch {
        // a hook must never break the session
      }
    });
  hook
    .command("pre-edit-gate")
    .description("PreToolUse ADVISE: warns when editing a file not Read this session")
    .action(async () => {
      try {
        let payload: { tool_name?: unknown; tool_input?: { file_path?: unknown } };
        try {
          payload = JSON.parse(await Bun.stdin.text());
        } catch {
          return;
        }
        const filePath =
          typeof payload.tool_input?.file_path === "string" ? payload.tool_input.file_path : null;
        if (!filePath) return;
        const { parseCapturedEvents, filesReadInSession } = await import(
          "../core/evidence-capture.ts"
        );
        const events = parseCapturedEvents(await readTrace());
        const readFiles = filesReadInSession(events);
        if (!readFiles.has(filePath)) {
          process.stderr.write(
            `⚠ vouch research-sufficiency: editing ${filePath.split("/").pop()} without reading it this session. ` +
              `Read the file to ground edits in its current state.\n`,
          );
        }
      } catch {
        // a hook must never break the session
      }
    });
  hook
    .command("stop-review")
    .description(
      "Stop hook: checks for unresolved negatives, ungrounded claims, absence-without-search",
    )
    .action(async () => {
      try {
        let payload: { transcript_path?: unknown; stop_hook_active?: unknown };
        try {
          payload = JSON.parse(await Bun.stdin.text());
        } catch {
          return;
        }

        const messages: string[] = [];
        let reviewerBlocked = false;
        // Claude Code sets stop_hook_active=true when a prior Stop block already forced
        // continuation. Block at most ONCE per stop-cycle: force one reconsideration,
        // then let the agent stop even if still flagged — no infinite thrash loop.
        const alreadyForced = payload.stop_hook_active === true;
        const {
          parseCapturedEvents,
          unresolvedNegatives,
          eventsSinceLastCommit,
          hasAbsenceClaimWithoutSearch,
          observationsOnly,
        } = await import("../core/evidence-capture.ts");
        // eventsSinceLastCommit first (it needs to SEE commits to find the cutoff),
        // then strip the agent's own commit/add actions from the evidence window.
        const allEvents = parseCapturedEvents(await readTrace());
        const traceEvents = observationsOnly(eventsSinceLastCommit(allEvents));

        const { shouldCallReviewer: reviewerOn } = await import("../core/reviewer.ts");
        const reviewerEnabled = reviewerOn();

        // 1. Unresolved negatives (deterministic fallback when reviewer is off)
        if (!reviewerEnabled) {
          const negatives = unresolvedNegatives(traceEvents);
          if (negatives.length > 0) {
            const lines = ["⚠ vouch omission check: unresolved negative signals this session:"];
            for (const n of negatives.slice(0, 5)) {
              lines.push(`  • \`${n.command}\` → exit ${n.exitCode}`);
            }
            if (negatives.length > 5) lines.push(`  … and ${negatives.length - 5} more`);
            lines.push("  (re-run successfully or acknowledge these failures before concluding.)");
            messages.push(lines.join("\n"));
          }
        }

        // 2. LLM reviewer on the agent's final response
        const tp = payload.transcript_path;
        const { existsSync } = await import("node:fs");
        if (typeof tp === "string" && existsSync(tp)) {
          const transcriptText = await Bun.file(tp).text();
          const { extractLastAssistantText } = await import("../core/prose-stop.ts");
          const draft = extractLastAssistantText(transcriptText);

          // 2a. Deterministic: absence claim without search
          if (draft && hasAbsenceClaimWithoutSearch(draft, traceEvents)) {
            messages.push(
              "⚠ vouch grounding: you claimed something doesn't exist or you're unaware " +
                "of something, but ran no WebSearch/WebFetch this session. " +
                "Search before asserting absence — training memory ≠ verified knowledge.",
            );
          }

          // 2b. LLM reviewer
          const { formatReviewMessage, formatReviewerHealthNote } = await import(
            "../core/reviewer.ts"
          );
          if (reviewerEnabled && draft && draft.length > 50) {
            const verdict = await runReviewer({
              action: draft.slice(0, 4000),
              actionType: "stop-response",
              allEvents,
            });
            const { captureVerdict } = await import("../core/corpus.ts");
            captureVerdict({
              actionType: "stop-response",
              action: draft.slice(0, 4000),
              events: allEvents,
              verdict,
            });
            const healthNote = formatReviewerHealthNote(verdict);
            if (healthNote) messages.push(healthNote);
            const reviewMsg = formatReviewMessage(verdict);
            if (reviewMsg) messages.push(reviewMsg);
            if (verdict.issues.some((i) => i.severity === "block")) reviewerBlocked = true;
          }
        }

        if (messages.length > 0) process.stderr.write(`${messages.join("\n")}\n`);
        // Stop BLOCK: a block-severity reviewer issue exits 2, which feeds the message
        // back and forces the agent to keep going — it must confront the ungrounded
        // conclusion before it can stop. Guarded by alreadyForced so it fires at most
        // once per stop-cycle. Advise-level (warn) issues stay exit 0.
        if (reviewerBlocked && !alreadyForced) process.exit(2);
      } catch {
        // a hook must never break the session
      }
    });

  program
    .command("doctor")
    .description(
      "diagnose whether vouch is set up to catch things (API key, endpoint, latency, capture)",
    )
    .action(async () => {
      const { runDoctor, formatDoctor } = await import("../core/doctor.ts");
      const checks = await runDoctor();
      process.stdout.write(`${formatDoctor(checks)}\n`);
      // Exit non-zero on a blocking (✗) issue so CI / `vouch doctor && …` can DETECT a dead
      // setup instead of trusting a 0 exit. Warns (⚠) are not blocking → stay 0.
      if (checks.some((c) => c.ok === false)) process.exit(1);
    });

  try {
    setSilent(false);
    await program.parseAsync(["node", "vouch", ...argv]);
    return 0;
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    process.stderr.write(`vouch: ${msg}\n`);
    return 1;
  }
}
