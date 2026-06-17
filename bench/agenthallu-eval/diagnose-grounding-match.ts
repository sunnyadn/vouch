// Diagnose the grounding-MATCH cry-wolves (idx 3, 5 from precision-fps.json): does the reviewer
// QUERY for the answer's value, FIND it in the trace, and STILL fire fabrication? That distinguishes
// a genuine grounding-MATCH failure (queries, finds, fires anyway — the real live bug) from an
// AgentHallu data-shape artifact (bare final_answer value with no prose). Dumps the query trail +
// full verdict per rep, NO scoring — pure mechanism.
//   bun bench/agenthallu-eval/diagnose-grounding-match.ts [--idx 3,5] [--reps 3]
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
const flag = (n: string, d: string) => { const i = args.indexOf(n); return i >= 0 ? (args[i + 1] ?? d) : d; };
const WANT = new Set(flag("--idx", "3,5").split(",").map(Number));
const REPS = Number(flag("--reps", "3"));

interface Step { step: number; content: string | null; tool_calls?: Array<{ name: string; arguments: unknown }>; tool_responses?: string[]; }
interface Traj { is_hallucination: string | boolean; hallucination_subcategory: string | null; agent_type: string; agent_answer: string; history: Step[]; }
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
    const tcs = s.tool_calls ?? [], trs = s.tool_responses ?? [];
    for (let i = 0; i < tcs.length; i++) {
      const tr = trs[i] ?? "", err = /error|fail|not found/i.test(tr);
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
      if (!(t.is_hallucination === true || t.is_hallucination === "true")) out.push(t);
    }
  }
  return out;
}

const all = loadClean().slice(0, 30);
console.log(`Diagnosing idx ${[...WANT].join(",")} | ${process.env.VOUCH_REVIEWER_MODEL} | ${REPS} reps\n`);
for (let idx = 0; idx < all.length; idx++) {
  if (!WANT.has(idx)) continue;
  const t = all[idx]!;
  const action = asText(t.agent_answer) || stepText(t.history.at(-1));
  const events = eventsBefore(t);
  console.log(`\n══════ idx ${idx}  [${t.agent_type}]  answer=${JSON.stringify(action.slice(0, 80))} ══════`);
  for (let r = 0; r < REPS; r++) {
    const v = await anthropicReviewerAgentic({ action, actionType: "stop-response", events, projectFindings: [] });
    const trail = (v.queries ?? []).map((q) => `${JSON.stringify(q.pattern)}→${q.hits}`).join("  ");
    console.log(`\n  rep ${r}  status=${v.status ?? "ok"}  queries[${(v.queries ?? []).length}]: ${trail}`);
    for (const i of v.issues) console.log(`    [${i.severity}] ${i.type}  quote=${JSON.stringify(i.quote)}\n      detail: ${i.detail}`);
    if (!v.issues.length) console.log(`    (no issues — silent)`);
  }
}
