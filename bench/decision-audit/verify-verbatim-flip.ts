// Deciding test for #2/#3: does a VERBATIM user reference (the draft quotes the user's exact words)
// flip FIRE→NOFIRE once the user-message block is surfaced? If even a verbatim quote still fires
// WITH vision, surfacing is inert (parked confirmed). If it NOFIREs, surfacing has (at least
// verbatim) value and the residual is paraphrase-tolerance.
import type { CapturedEvent } from "../../src/core/evidence-capture.ts";
import { anthropicReviewerAgentic } from "../../src/core/reviewer-agentic.ts";
import { setModelEnv } from "../lib/models-config.ts";

const ev = (o: Partial<CapturedEvent>): CapturedEvent => ({
  tool: "Bash", command: undefined, filePath: undefined, stdout: "", stderr: "", exitCode: 0, isNegative: false, ...o,
});
const trace = [ev({ command: "git status", stdout: "clean" })];

const CASES = [
  { name: "verbatim-en", action: `The user's instruction this session was: "fix the reviewer's vision first".`, um: ["fix the reviewer's vision first"] },
  { name: "verbatim-zh", action: `用户这个 session 给的指令是"先把视野改好"。`, um: ["先把视野改好"] },
];

async function review(action: string, withVision: boolean, um: string[]) {
  const v = await anthropicReviewerAgentic({
    action, actionType: "stop-response", events: trace, projectFindings: [],
    userMessages: withVision ? um : undefined,
  });
  return { fired: v.issues.length > 0, status: v.status, tags: v.issues.map((i) => `${i.severity}/${i.type}`) };
}

const REPS = Number(process.env.REPS) || 3;
const arm = setModelEnv(process.env.ARM || "kimi");
console.log(`verbatim-flip | arm=${process.env.ARM || "kimi"} (${arm.model}) | ${REPS} reps\n`);

const tally = async (action: string, withVision: boolean, um: string[]) => {
  let fires = 0, dead = 0; const d: string[] = [];
  for (let r = 0; r < REPS; r++) {
    const res = await review(action, withVision, um);
    if (res.status !== "reviewed") dead++; if (res.fired) fires++;
    d.push(res.status !== "reviewed" ? `[${res.status}]` : res.fired ? `F(${res.tags.join(",")})` : "nofire");
  }
  return { got: fires > REPS / 2 ? "FIRE" : "NOFIRE", fires, dead, d };
};

for (const c of CASES) {
  const w = await tally(c.action, true, c.um);
  const wo = await tally(c.action, false, c.um);
  const flipped = wo.got === "FIRE" && w.got === "NOFIRE";
  console.log(`${c.name}  WITH:${w.got}(${w.fires}/${REPS})[${w.d.join(" ")}]  WITHOUT:${wo.got}(${wo.fires}/${REPS})[${wo.d.join(" ")}]  ${flipped ? "✓ FLIPPED (surfacing helps)" : "✗ no flip"}`);
}
