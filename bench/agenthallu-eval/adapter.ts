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
const { verifyMajority } = await import("../verify-replay/verifier.ts");

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
// --verify: two-stage — each detector fire is re-judged by the SAME model as a stage-2
// verifier (majority of --vreps); the rep counts as fired only if upheld. Validated on the
// coding gold (bench/verify-replay): kimi-detect + kimi-self-verify 10/10 at REPS=4.
const VERIFY = args.includes("--verify");
const VREPS = Number(flag("--vreps", "2"));

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
        if (!hallu) continue;
        if (CATEGORY !== "all" && t.hallucination_category !== CATEGORY) continue; // recall: this category
      }
      out.push(t);
    }
  }
  return out;
}

// --category all → sample up to LIMIT *per category* (breadth across categories AND frameworks,
// since category correlates with framework); otherwise take the first LIMIT of one category.
const ALL = !CLEAN && CATEGORY === "all";
const loaded = loadTrajectories();
let trajectories: Traj[];
if (ALL) {
  const byCat = new Map<string, Traj[]>();
  for (const t of loaded) {
    const c = t.hallucination_category ?? "?";
    if (!byCat.has(c)) byCat.set(c, []);
    byCat.get(c)!.push(t);
  }
  trajectories = [...byCat.values()].flatMap((arr) => arr.slice(0, LIMIT));
} else {
  trajectories = loaded.slice(0, LIMIT);
}

// Print the ACTUAL model/endpoint — the A/B swaps these via env, so the output must say which.
const endpoint = process.env.ANTHROPIC_BASE_URL ?? "api.anthropic.com";
const model = process.env.VOUCH_REVIEWER_MODEL ?? "(default)";
console.log(
  `AgentHallu → ${model} @ ${endpoint} | mode=${CLEAN ? "PRECISION (clean, expect NOFIRE)" : `RECALL ${ALL ? "ALL categories" : CATEGORY} (expect FIRE)`} | ${trajectories.length} trajectories × ${REPS} reps${VERIFY ? ` | TWO-STAGE self-verify ×${VREPS}` : ""}\n`,
);
// Stage-2 verifier = the SAME model/endpoint the detector runs on (self-verify).
const verifierModel = {
  name: "self",
  apiKey: process.env.ANTHROPIC_API_KEY ?? "",
  baseURL: process.env.ANTHROPIC_BASE_URL ?? "",
  model: process.env.VOUCH_REVIEWER_MODEL ?? "",
};

let hits = 0; // recall: fired-on-hallucination; precision: fired-on-clean (a FALSE positive)
let failOpens = 0;
const total = trajectories.length;
const byCat = new Map<string, { good: number; n: number }>(); // per-category tally
for (const t of trajectories) {
  const halluStep = Number(t.hallucination_step);
  // RECALL: review the labeled bad step, with the grounding gathered before it.
  // PRECISION: review the final answer, with ALL the grounding gathered.
  const action = CLEAN
    ? asText(t.agent_answer) || stepText(t.history.at(-1))
    : stepText(t.history.find((s) => s.step === halluStep)) || asText(t.agent_answer);
  const events = CLEAN ? eventsBefore(t, Number.MAX_SAFE_INTEGER) : eventsBefore(t, halluStep);

  let fired = 0;
  let valid = 0; // fail-open-aware (mirrors deepseek-eval c4d002e): status:"failed" is a DEAD
  let killed = 0; // fires the stage-2 verifier rejected (two-stage only)
  let firedIssues: { type: string; detail: string }[] = [];
  for (let r = 0; r < REPS; r++) {
    // rep (429/error), not a silent no-fire — retry with backoff, else exclude from the denominator.
    let v = await anthropicReviewerAgentic({ action, actionType: "stop-response", events, projectFindings: [] });
    for (let retry = 0; v.status === "failed" && retry < 3; retry++) {
      failOpens++;
      await new Promise((res) => setTimeout(res, 2500 * (retry + 1)));
      v = await anthropicReviewerAgentic({ action, actionType: "stop-response", events, projectFindings: [] });
    }
    if (v.status === "failed") continue;
    valid++;
    if (v.issues.some((i) => i.severity === "block" || i.severity === "warn")) {
      if (VERIFY) {
        const upheld = await verifyMajority(verifierModel, { action, events }, v.issues, VREPS);
        if (upheld === false) {
          killed++;
          continue; // stage-2 rejected every flag — the rep does NOT fire
        }
        // upheld === null (verifier dead) counts as fired: fail toward the detector's verdict
      }
      fired++;
      firedIssues = v.issues; // keep a firing verdict so we can show WHY (esp. cry-wolf reasons)
    }
  }
  const firedMajority = valid > 0 && fired * 2 > valid;
  if (firedMajority) hits++;
  // RECALL wants fire (✓ = caught); PRECISION wants silence (✓ = no false alarm)
  const good = CLEAN ? !firedMajority : firedMajority;
  const cat = CLEAN ? "clean" : t.hallucination_category ?? "?";
  const s = byCat.get(cat) ?? { good: 0, n: 0 };
  s.n++;
  if (good) s.good++;
  byCat.set(cat, s);
  const sub = t.hallucination_subcategory || t.hallucination_category || "clean";
  console.log(
    `${good ? "✓" : "✗"} [${t.agent_type}] ${sub}  (fired ${fired}/${valid}${killed ? `, ${killed} killed by verify` : ""}${valid < REPS ? `, ${REPS - valid} dead` : ""})`,
  );
  // On a CRY-WOLF (clean trajectory it flagged), print why — the diagnostic for precision work.
  if (CLEAN && firedMajority) for (const i of firedIssues) console.log(`     ↳ ${i.type}: ${i.detail.slice(0, 160)}`);
}

if (failOpens) console.log(`\n⚠ ${failOpens} fail-open(s) hit during the run (retried) — quota/429 pressure.`);
console.log("\n── Scorecard ──");
if (ALL || CLEAN) {
  for (const [cat, s] of byCat)
    console.log(`  ${cat.padEnd(30)} ${s.good}/${s.n} ${CLEAN ? "silent" : "caught"} (${((s.good / s.n) * 100).toFixed(0)}%)`);
  console.log("  " + "─".repeat(40));
}
if (CLEAN) {
  console.log(`PRECISION: ${total - hits}/${total} clean correctly silent (${(((total - hits) / total) * 100).toFixed(0)}%) — ${hits} false alarms`);
} else {
  console.log(`RECALL: ${hits}/${total} caught (${((hits / total) * 100).toFixed(0)}%) — ${total - hits} missed`);
}
