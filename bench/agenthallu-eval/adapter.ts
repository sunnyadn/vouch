// AgentHallu (arXiv 2601.06818) → the DEPLOYED agentic reviewer on deepseek.
//
// INDEPENDENT, human-labeled gold (labeler ⟂ subject — the exact thing the self-labeled corpus
// can't give us). 693 multi-step agent trajectories, 443 hallucinated (5 categories) + 250 clean.
//   - RECALL: on a hallucinated trajectory, does the reviewer FIRE on the labeled bad step?
//   - PRECISION (--clean): on a clean trajectory, does it correctly STAY SILENT?
//
// Data: github.com/liuxuannan/AgentHallu → set $AGENTHALLU_DIR to the `AgentHallu/` dir
// (default /tmp/AgentHallu-fresh/AgentHallu). Run:
//   bun bench/agenthallu-eval/adapter.ts [--category "Tool-Use Hallucination"] [--limit N] [--reps K] [--clean]
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import type { CapturedEvent } from "../../src/core/evidence-capture.ts";

const ROOT = join(import.meta.dir, "..", "..");
for (const line of readFileSync(join(ROOT, ".env"), "utf8").split("\n")) {
  const m = line.match(/^([A-Z_]+)=(.*)$/);
  if (m?.[1] && !process.env[m[1]]) process.env[m[1]] = m[2] ?? "";
}
const { anthropicReviewerAgentic } = await import("../../src/core/reviewer-agentic.ts");

const DATA = process.env.AGENTHALLU_DIR ?? "/tmp/AgentHallu-fresh/AgentHallu";
const args = process.argv.slice(2);
const flag = (n: string, d: string) => {
  const i = args.indexOf(n);
  return i >= 0 ? (args[i + 1] ?? d) : d;
};
const CLEAN = args.includes("--clean"); // precision mode: expect NOFIRE on clean trajectories
const CATEGORY = flag("--category", "Tool-Use Hallucination");
const LIMIT = Number(flag("--limit", "15"));
const REPS = Number(flag("--reps", "2"));

interface Step {
  step: number;
  content: string | null;
  tool_calls?: Array<{ name: string; arguments: unknown }>;
  tool_responses?: string[];
}
interface Traj {
  is_hallucination: string | boolean;
  hallucination_step: number | string | null;
  hallucination_category: string | null;
  hallucination_subcategory: string | null;
  agent_type: string;
  agent_answer: string;
  explanation?: string;
  history: Step[];
}

// The tool calls/results the agent had ALREADY gathered before `upto` — the grounding the
// reviewer gets to query (AgentHallu steps → CapturedEvents).
function eventsBefore(t: Traj, upto: number): CapturedEvent[] {
  const events: CapturedEvent[] = [];
  for (const s of t.history) {
    if (s.step >= upto) break;
    const tcs = s.tool_calls ?? [];
    const trs = s.tool_responses ?? [];
    for (let i = 0; i < tcs.length; i++) {
      const tr = trs[i] ?? "";
      const err = /error|fail|not found/i.test(tr);
      events.push({
        tool: "Bash",
        command: `${tcs[i]!.name} ${JSON.stringify(tcs[i]!.arguments)}`,
        stdout: tr,
        stderr: "",
        exitCode: err ? 1 : 0,
        isNegative: err,
      });
    }
  }
  return events;
}

// AgentHallu fields aren't always strings (agent_answer / content can be structured) — coerce.
const asText = (x: unknown): string => (typeof x === "string" ? x : x == null ? "" : JSON.stringify(x));

function stepText(s: Step | undefined): string {
  if (!s) return "";
  const parts: string[] = [];
  if (s.content) parts.push(asText(s.content));
  for (const tc of s.tool_calls ?? []) parts.push(`[tool: ${tc.name}(${JSON.stringify(tc.arguments)})]`);
  return parts.join("\n");
}

function loadTrajectories(): Traj[] {
  const out: Traj[] = [];
  for (const fw of readdirSync(DATA)) {
    let files: string[];
    try {
      files = readdirSync(join(DATA, fw));
    } catch {
      continue;
    }
    for (const f of files) {
      if (!f.endsWith(".json")) continue;
      let t: Traj;
      try {
        t = JSON.parse(readFileSync(join(DATA, fw, f), "utf8"));
      } catch {
        continue;
      }
      const hallu = t.is_hallucination === true || t.is_hallucination === "true";
      if (CLEAN) {
        if (hallu) continue; // precision: clean trajectories only
      } else {
        if (!hallu || t.hallucination_category !== CATEGORY) continue; // recall: this category
      }
      out.push(t);
    }
  }
  return out;
}

const trajectories = loadTrajectories().slice(0, LIMIT);
console.log(
  `AgentHallu → deepseek reviewer | mode=${CLEAN ? "PRECISION (clean, expect NOFIRE)" : `RECALL ${CATEGORY} (expect FIRE)`} | ${trajectories.length} trajectories × ${REPS} reps\n`,
);

let hits = 0; // recall: fired-on-hallucination; precision: fired-on-clean (a FALSE positive)
const total = trajectories.length;
for (const t of trajectories) {
  const halluStep = Number(t.hallucination_step);
  // RECALL: review the labeled bad step, with the grounding gathered before it.
  // PRECISION: review the final answer, with ALL the grounding gathered.
  const action = CLEAN
    ? asText(t.agent_answer) || stepText(t.history.at(-1))
    : stepText(t.history.find((s) => s.step === halluStep)) || asText(t.agent_answer);
  const events = CLEAN ? eventsBefore(t, Number.MAX_SAFE_INTEGER) : eventsBefore(t, halluStep);

  let fired = 0;
  for (let r = 0; r < REPS; r++) {
    const v = await anthropicReviewerAgentic({ action, actionType: "stop-response", events, projectFindings: [] });
    if (v.issues.some((i) => i.severity === "block" || i.severity === "warn")) fired++;
  }
  const firedMajority = fired > REPS / 2;
  if (firedMajority) hits++;
  // RECALL wants fire (✓ = caught); PRECISION wants silence (✓ = no false alarm)
  const good = CLEAN ? !firedMajority : firedMajority;
  const sub = t.hallucination_subcategory || t.hallucination_category || "clean";
  console.log(`${good ? "✓" : "✗"} [${t.agent_type}] ${sub}  (fired ${fired}/${REPS})`);
}

console.log("\n── Scorecard ──");
if (CLEAN) {
  console.log(`PRECISION: ${total - hits}/${total} clean trajectories correctly silent (${(((total - hits) / total) * 100).toFixed(0)}%) — ${hits} false alarms`);
} else {
  console.log(`RECALL: ${hits}/${total} caught (${((hits / total) * 100).toFixed(0)}%) — ${total - hits} missed`);
}
