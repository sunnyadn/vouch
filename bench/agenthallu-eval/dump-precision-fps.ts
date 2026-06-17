// One-shot: re-surface the AgentHallu CLEAN-mode BLOCK false-positives and dump FULL detail
// (answer + the exact trace the reviewer saw + the firing verdict) so a cross-family judge (NOT
// the kimi reviewer, NOT the author's self-grade) can adjudicate grounded-vs-not per case.
//
// Mirrors adapter.ts CLEAN logic exactly (action = agent_answer; events = all grounding) but, for
// every case that BLOCK-fires by majority, writes a JSON record instead of a 160-char snippet.
//   bun bench/agenthallu-eval/dump-precision-fps.ts [--limit 30] [--reps 3] [--out /tmp/ah-fps.json]
import { readdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import type { CapturedEvent } from "../../src/core/evidence-capture.ts";

const ROOT = join(import.meta.dir, "..", "..");
for (const line of readFileSync(join(ROOT, ".env"), "utf8").split("\n")) {
  const m = line.match(/^([A-Z_]+)=(.*)$/);
  if (m?.[1] && !process.env[m[1]]) process.env[m[1]] = m[2] ?? "";
}
const { reviewWithRetry } = await import("../lib/reviewer-retry.ts");

const DATA = process.env.AGENTHALLU_DIR ?? "/tmp/AgentHallu-fresh/AgentHallu";
const args = process.argv.slice(2);
const flag = (n: string, d: string) => {
  const i = args.indexOf(n);
  return i >= 0 ? (args[i + 1] ?? d) : d;
};
const LIMIT = Number(flag("--limit", "30"));
const REPS = Number(flag("--reps", "3"));
const OUT = flag("--out", "/tmp/ah-fps.json");

interface Step {
  step: number;
  content: string | null;
  tool_calls?: Array<{ name: string; arguments: unknown }>;
  tool_responses?: string[];
}
interface Traj {
  is_hallucination: string | boolean;
  hallucination_category: string | null;
  hallucination_subcategory: string | null;
  agent_type: string;
  agent_answer: string;
  history: Step[];
}
const asText = (x: unknown): string => (typeof x === "string" ? x : x == null ? "" : JSON.stringify(x));
function stepText(s: Step | undefined): string {
  if (!s) return "";
  const parts: string[] = [];
  if (s.content) parts.push(asText(s.content));
  for (const tc of s.tool_calls ?? []) parts.push(`[tool: ${tc.name}(${JSON.stringify(tc.arguments)})]`);
  return parts.join("\n");
}
function eventsBefore(t: Traj): CapturedEvent[] {
  const events: CapturedEvent[] = [];
  for (const s of t.history) {
    const tcs = s.tool_calls ?? [];
    const trs = s.tool_responses ?? [];
    for (let i = 0; i < tcs.length; i++) {
      const tr = trs[i] ?? "";
      const err = /error|fail|not found/i.test(tr);
      events.push({ tool: "Bash", command: `${tcs[i]!.name} ${JSON.stringify(tcs[i]!.arguments)}`, stdout: tr, stderr: "", exitCode: err ? 1 : 0, isNegative: err });
    }
  }
  return events;
}
function loadClean(): Traj[] {
  const out: Traj[] = [];
  for (const fw of readdirSync(DATA)) {
    let files: string[];
    try { files = readdirSync(join(DATA, fw)); } catch { continue; }
    for (const f of files) {
      if (!f.endsWith(".json")) continue;
      let t: Traj;
      try { t = JSON.parse(readFileSync(join(DATA, fw, f), "utf8")); } catch { continue; }
      const hallu = t.is_hallucination === true || t.is_hallucination === "true";
      if (!hallu) out.push(t);
    }
  }
  return out;
}

const trajectories = loadClean().slice(0, LIMIT);
const model = process.env.VOUCH_REVIEWER_MODEL ?? "(default)";
console.log(`AgentHallu PRECISION dump → ${model} | ${trajectories.length} clean × ${REPS} reps → ${OUT}\n`);

let failOpens = 0;
const fps: unknown[] = [];
let idx = 0;
for (const t of trajectories) {
  const action = asText(t.agent_answer) || stepText(t.history.at(-1));
  const events = eventsBefore(t);
  let blockFired = 0, valid = 0;
  let lastBlockIssues: { type: string; severity: string; detail: string; quote?: string }[] = [];
  for (let r = 0; r < REPS; r++) {
    const v = await reviewWithRetry({ action, actionType: "stop-response", events, projectFindings: [] }, () => failOpens++);
    if (!v) continue;
    valid++;
    if (v.issues.some((i) => i.severity === "block")) {
      blockFired++;
      lastBlockIssues = v.issues as typeof lastBlockIssues;
    }
  }
  const blockMajority = valid > 0 && blockFired * 2 > valid;
  const tag = `[${t.agent_type}] ${t.hallucination_subcategory || "clean"}`;
  console.log(`${blockMajority ? "✗ BLOCK-FP" : "✓ silent  "} ${tag}  (block ${blockFired}/${valid})`);
  if (blockMajority) {
    fps.push({
      idx: idx,
      framework: t.agent_type,
      subcategory: t.hallucination_subcategory,
      agent_answer: action,
      trace: events.map((e) => ({ call: e.command, response: e.stdout })),
      verdict_issues: lastBlockIssues,
      block_reps: `${blockFired}/${valid}`,
    });
  }
  idx++;
}
if (failOpens) console.log(`\n⚠ ${failOpens} fail-open(s) — quota/429 pressure.`);
console.log(`\n${fps.length} BLOCK-FP(s) dumped → ${OUT}`);
writeFileSync(OUT, JSON.stringify(fps, null, 2));
