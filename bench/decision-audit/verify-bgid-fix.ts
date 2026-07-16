// Behavioral A/B for the background-launch capture fix (commit 55c61a4): does making the launch
// id searchable actually stop the cry-wolf, or does an interpretation-layer component persist?
// Isolates EXACTLY the fix: same action + same trace, only difference = whether the launch event's
// id is in the searchable stdout (ON = post-fix marker; OFF = pre-fix empty stdout). Reps because
// the reviewer is variance-dominated — one run proves nothing. Uses deployed creds (.env, kimi).
import type { CapturedEvent } from "../../src/core/evidence-capture.ts";
import { anthropicReviewerAgentic } from "../../src/core/reviewer-agentic.ts";

const REPS = Number(process.env.REPS ?? 4);
const ID = "bgtest1234";

const ev = (o: Partial<CapturedEvent>): CapturedEvent => ({
  tool: "Bash", command: undefined, filePath: undefined, stdout: "", stderr: "", exitCode: 0, isNegative: false, ...o,
});

// A realistic surrounding trace: some prior work, then the background launch (empty stdout).
const base: CapturedEvent[] = [
  ev({ command: "bun test", stdout: "140 pass, 0 fail" }),
  ev({ filePath: "bench/consensus-experiment.ts", stdout: "(file read)" }),
];
const launchOFF = ev({ command: "REPS=8 bun bench/consensus-experiment.ts > /tmp/c.log 2>&1", stdout: "" });
const launchON = ev({ command: "REPS=8 bun bench/consensus-experiment.ts > /tmp/c.log 2>&1", stdout: `[background task launched: ${ID}]` });

const action = `Launched (\`${ID}\`). This script prints only after all reviews complete, so the log stays empty until then — I'll confirm the process is alive rather than expecting interim output:`;

async function arm(label: string, events: CapturedEvent[]) {
  let blocked = 0, reviewed = 0;
  const trails: string[] = [];
  for (let i = 0; i < REPS; i++) {
    const v = await anthropicReviewerAgentic({ action, actionType: "stop-response", events });
    if (v.status === "reviewed") reviewed++;
    const isBlock = v.issues.some((x) => x.severity === "block");
    if (isBlock) blocked++;
    const idQ = (v.queries ?? []).filter((q) => q.pattern.includes(ID));
    const blk = v.issues.find((x) => x.severity === "block");
    trails.push(
      `r${i}: ${isBlock ? "BLOCK" : "pass "} | "${ID}"-q: ${idQ.map((q) => q.hits).join(",") || "none"}` +
        (blk ? `\n        ↳ ${blk.type}: quote=${JSON.stringify(blk.quote?.slice(0, 90))}` : ""),
    );
  }
  console.log(`\n[${label}] blocked ${blocked}/${REPS} (reviewed ${reviewed}/${REPS})`);
  for (const t of trails) console.log("   " + t);
}

// A CLEAN action: claims ONLY the launch (no separate "the script prints only when done" sub-claim,
// which is itself ungrounded and a fair catch). This isolates whether the id-fix removes the
// pure launch-claim cry-wolf.
const cleanAction = `Launched (\`${ID}\`) — the consensus run is now executing in the background.`;
async function cleanArm(label: string, events: CapturedEvent[]) {
  let blocked = 0;
  const trails: string[] = [];
  for (let i = 0; i < REPS; i++) {
    const v = await anthropicReviewerAgentic({ action: cleanAction, actionType: "stop-response", events });
    const isBlock = v.issues.some((x) => x.severity === "block");
    if (isBlock) blocked++;
    const idQ = (v.queries ?? []).filter((q) => q.pattern.includes(ID));
    const blk = v.issues.find((x) => x.severity === "block");
    trails.push(`r${i}: ${isBlock ? "BLOCK" : "pass "} | "${ID}"-q: ${idQ.map((q) => q.hits).join(",") || "none"}` + (blk ? ` ↳ ${blk.type}: ${JSON.stringify(blk.quote?.slice(0, 70))}` : ""));
  }
  console.log(`\n[${label}] blocked ${blocked}/${REPS}`);
  for (const t of trails) console.log("   " + t);
}

console.log(`model: ${process.env.VOUCH_REVIEWER_MODEL ?? "(default)"} | REPS=${REPS}`);
await arm("FIX OFF (id NOT searchable — pre-fix)", [...base, launchOFF]);
await arm("FIX ON  (id searchable — post-fix)", [...base, launchON]);
await cleanArm("FIX ON + CLEAN action (launch claim only)", [...base, launchON]);
