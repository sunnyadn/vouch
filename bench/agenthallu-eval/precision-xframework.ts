// Cross-framework block-precision: the headline precision number must NOT be single-framework. The
// dump-precision-fps.ts run was all OpenDeepSearch (slice(0,30) artifact). Here we sample K clean
// trajectories from EACH framework directory and report block-FP rate per-framework + overall, so the
// precision claim spans the same breadth as the recall leg.
//   bun bench/agenthallu-eval/precision-xframework.ts [--per 5] [--reps 3] [--out /tmp/ah-xfw.json]
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
const flag = (n: string, d: string) => { const i = args.indexOf(n); return i >= 0 ? (args[i + 1] ?? d) : d; };
const PER = Number(flag("--per", "5"));
const REPS = Number(flag("--reps", "3"));
const OUT = flag("--out", "/tmp/ah-xfw.json");

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
// Sample PER clean trajectories from EACH framework directory.
function sampleByFramework(): { fw: string; t: Traj }[] {
  const out: { fw: string; t: Traj }[] = [];
  for (const fw of readdirSync(DATA)) {
    let files: string[]; try { files = readdirSync(join(DATA, fw)); } catch { continue; }
    let taken = 0;
    for (const f of files) {
      if (taken >= PER) break;
      if (!f.endsWith(".json")) continue;
      let t: Traj; try { t = JSON.parse(readFileSync(join(DATA, fw, f), "utf8")); } catch { continue; }
      if (t.is_hallucination === true || t.is_hallucination === "true") continue;
      out.push({ fw, t });
      taken++;
    }
  }
  return out;
}

const sample = sampleByFramework();
console.log(`AgentHallu CROSS-FRAMEWORK precision → ${process.env.VOUCH_REVIEWER_MODEL} | ${sample.length} clean (${PER}/fw) × ${REPS} reps\n`);

let failOpens = 0;
const perFw = new Map<string, { fp: number; n: number }>();
const fps: unknown[] = [];
for (const { fw, t } of sample) {
  const action = asText(t.agent_answer) || stepText(t.history.at(-1));
  const events = eventsBefore(t);
  let blockFired = 0, valid = 0;
  let issues: { type: string; severity: string; detail: string }[] = [];
  for (let r = 0; r < REPS; r++) {
    const v = await reviewWithRetry({ action, actionType: "stop-response", events, projectFindings: [] }, () => failOpens++);
    if (!v) continue;
    valid++;
    if (v.issues.some((i) => i.severity === "block")) { blockFired++; issues = v.issues as typeof issues; }
  }
  const blockMajority = valid > 0 && blockFired * 2 > valid;
  const s = perFw.get(fw) ?? { fp: 0, n: 0 };
  s.n++; if (blockMajority) s.fp++;
  perFw.set(fw, s);
  console.log(`  ${blockMajority ? "✗BLOCK-FP" : "✓silent  "} ${fw.padEnd(16)} [${t.agent_type}] (block ${blockFired}/${valid})`);
  if (blockMajority) fps.push({ fw, agent_type: t.agent_type, agent_answer: action.slice(0, 200), issues });
}

if (failOpens) console.log(`\n⚠ ${failOpens} fail-open(s) — quota/429 pressure.`);
console.log("\n── Per-framework block-precision ──");
let totFp = 0, totN = 0;
for (const [fw, s] of perFw) { console.log(`  ${fw.padEnd(16)} ${s.n - s.fp}/${s.n} silent (${s.fp} block-FP)`); totFp += s.fp; totN += s.n; }
console.log("  " + "─".repeat(40));
console.log(`OVERALL block-precision: ${totN - totFp}/${totN} (${(((totN - totFp) / totN) * 100).toFixed(0)}%) — ${totFp} block-FPs across ${perFw.size} frameworks`);
writeFileSync(OUT, JSON.stringify({ per: PER, reps: REPS, perFw: [...perFw], totFp, totN, fps }, null, 2));
console.log(`\nFP detail → ${OUT}`);
