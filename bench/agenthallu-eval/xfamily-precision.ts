// Cross-family block-PRECISION adjudication on AgentHallu clean trajectories.
//
// WHY: the deployed gate's precision number is self-adjudicated (author bias) or judged by a
// SAME-FAMILY model (kimi-judges-kimi → biased toward agreeing). This closes that gap with a
// CROSS-FAMILY second reviewer: run the deployed kimi gate AND claude-p (subscription, Anthropic
// family) as an INDEPENDENT reviewer on the SAME clean trajectories, then adjudicate each kimi
// false-positive by whether the cross-family reviewer corroborates it.
//
// METHOD (not a meta-judge of kimi's verdict — avoids circularity / reasoning-contamination):
//   - kimi block-fires on a CLEAN trajectory  → a candidate FALSE POSITIVE.
//   - claude-p reviews the SAME action+trace independently:
//       · claude-p ALSO block-fires → DEFENSIBLE (a strong cross-family reviewer agrees the answer
//         overreaches the trace — likely a gold-semantics mismatch, NOT a cry-wolf).
//       · claude-p stays silent      → CRY-WOLF (kimi flagged a groundable answer).
//   raw precision      = clean trajectories kimi stayed silent on / total.
//   adjudicated prec.  = (silent + defensible) / total   [cry-wolves still count against precision].
//
// CAVEAT (keep in any writeup): claude-p is a different MODEL and a different HARNESS (full trace
// inline vs the prod query_history tool) → it is a GOLD/ceiling reviewer, used here only to remove
// author/same-family bias from the FP adjudication, NOT as "what a claude gate would do in prod".
//
// Run:  AGENTHALLU_DIR=/tmp/AgentHallu-fresh/AgentHallu/AgentHallu \
//         bun bench/agenthallu-eval/xfamily-precision.ts [--limit 30] [--reps 3] [--out /tmp/ah-xfam.json]
import { readdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import type { CapturedEvent } from "../../src/core/evidence-capture.ts";

const ROOT = join(import.meta.dir, "..", "..");
for (const line of readFileSync(join(ROOT, ".env"), "utf8").split("\n")) {
  const m = line.match(/^([A-Z_]+)=(.*)$/);
  if (m?.[1] && !process.env[m[1]]) process.env[m[1]] = m[2] ?? "";
}
const { reviewWithRetry } = await import("../lib/reviewer-retry.ts");
const { claudePReview } = await import("../lib/reviewer-claude-p.ts");

const DATA = process.env.AGENTHALLU_DIR ?? "/tmp/AgentHallu-fresh/AgentHallu/AgentHallu";
const args = process.argv.slice(2);
const flag = (n: string, d: string) => {
  const i = args.indexOf(n);
  return i >= 0 ? (args[i + 1] ?? d) : d;
};
const LIMIT = Number(flag("--limit", "30"));
const REPS = Number(flag("--reps", "3"));
const OUT = flag("--out", "/tmp/ah-xfam.json");

interface Step {
  step: number;
  content: string | null;
  tool_calls?: Array<{ name: string; arguments: unknown }>;
  tool_responses?: string[];
}
interface Traj {
  is_hallucination: string | boolean;
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
// All tool calls/results the agent gathered (clean trajectories: the full grounding it had).
function allEvents(t: Traj): CapturedEvent[] {
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
      if (t.is_hallucination === true || t.is_hallucination === "true") continue; // clean only
      out.push(t);
    }
  }
  return out;
}

// Returns the fraction of REPS that fired at BLOCK severity (fail-open reps excluded from denom).
async function blockFireRate(
  review: (ctx: { action: string; actionType: "stop-response"; events: CapturedEvent[]; projectFindings: string[] }) => Promise<{ issues: { severity: string }[]; status?: string } | null>,
  action: string,
  events: CapturedEvent[],
  reps: number,
): Promise<{ block: number; valid: number }> {
  let block = 0;
  let valid = 0;
  for (let r = 0; r < reps; r++) {
    const v = await review({ action, actionType: "stop-response", events, projectFindings: [] });
    if (!v || v.status === "failed") continue;
    valid++;
    if (v.issues.some((i) => i.severity === "block")) block++;
  }
  return { block, valid };
}

const clean = loadClean().slice(0, LIMIT);
console.log(`X-FAMILY PRECISION | gate=kimi (${process.env.VOUCH_REVIEWER_MODEL}) | judge=claude-p (${process.env.VOUCH_CLAUDEP_MODEL ?? "subscription default"}) | ${clean.length} clean × ${REPS} reps (claude-p ×1 on FPs only)\n`);

interface Row { idx: number; agent_type: string; kimiBlock: boolean; claudepBlock: boolean | null; verdict: "correct-silent" | "cry-wolf" | "defensible" | "kimi-dead"; }
const rows: Row[] = [];
let kimiSilent = 0;
let cryWolf = 0;
let defensible = 0;
let kimiDead = 0;

for (let idx = 0; idx < clean.length; idx++) {
  const t = clean[idx]!;
  const action = asText(t.agent_answer) || stepText(t.history.at(-1));
  const events = allEvents(t);

  const k = await blockFireRate(reviewWithRetry as never, action, events, REPS);
  if (k.valid === 0) {
    rows.push({ idx, agent_type: t.agent_type, kimiBlock: false, claudepBlock: null, verdict: "kimi-dead" });
    kimiDead++;
    console.log(`· [${t.agent_type}] kimi DEAD (all reps fail-open) — excluded`);
    continue;
  }
  const kimiBlock = k.block * 2 > k.valid; // block-majority
  if (!kimiBlock) {
    rows.push({ idx, agent_type: t.agent_type, kimiBlock: false, claudepBlock: null, verdict: "correct-silent" });
    kimiSilent++;
    console.log(`✓ [${t.agent_type}] kimi silent (correct)  (block ${k.block}/${k.valid})`);
    continue;
  }
  // kimi FALSE-POSITIVE on a clean trajectory → adjudicate with the cross-family reviewer.
  const c = await blockFireRate(claudePReview as never, action, events, 1);
  const claudepBlock = c.valid > 0 && c.block > 0;
  const claudepDead = c.valid === 0;
  const verdict = claudepDead ? "cry-wolf" : claudepBlock ? "defensible" : "cry-wolf"; // claude-p dead → conservatively a cry-wolf (no corroboration)
  if (verdict === "defensible") defensible++;
  else cryWolf++;
  rows.push({ idx, agent_type: t.agent_type, kimiBlock: true, claudepBlock: claudepDead ? null : claudepBlock, verdict });
  console.log(`✗ [${t.agent_type}] kimi FP (block ${k.block}/${k.valid}) → claude-p ${claudepDead ? "DEAD" : claudepBlock ? "ALSO fires → DEFENSIBLE" : "silent → CRY-WOLF"}`);
}

const total = clean.length - kimiDead;
const rawPrec = kimiSilent / total;
const adjPrec = (kimiSilent + defensible) / total;
console.log("\n── X-family precision (clean = expect silence) ──");
console.log(`  total scored:        ${total}  (${kimiDead} kimi-dead excluded)`);
console.log(`  kimi silent:         ${kimiSilent}`);
console.log(`  kimi FP → defensible:${defensible}  (cross-family claude-p corroborates the fire)`);
console.log(`  kimi FP → CRY-WOLF:  ${cryWolf}  (claude-p stays silent on the groundable answer)`);
console.log(`  RAW block-precision:        ${kimiSilent}/${total} = ${(rawPrec * 100).toFixed(0)}%`);
console.log(`  X-FAMILY adjudicated prec.: ${kimiSilent + defensible}/${total} = ${(adjPrec * 100).toFixed(0)}%`);
writeFileSync(OUT, JSON.stringify({ gate: process.env.VOUCH_REVIEWER_MODEL, judge: process.env.VOUCH_CLAUDEP_MODEL ?? "subscription", reps: REPS, total, kimiSilent, defensible, cryWolf, kimiDead, rawPrec, adjPrec, rows }, null, 2));
console.log(`\nwrote ${OUT}`);
