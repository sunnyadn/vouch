// Full-pipeline behavioral red→green for eye-fix #5 (harness-error draft skip).
// The bug: a rate-limit/API error surfaced as the LAST assistant text block was reviewed as the
// agent's own draft → the reviewer fired `active-fabrication` on a SYSTEM string the agent never
// wrote (the documented #5 cry-wolf: "API Error: Server is temporarily limiting requests …" flagged
// as a fabricated termination reason).
//   RED   = feed the RAW last assistant text (the harness error) to the reviewer → expect FIRE.
//   GREEN = extractLastAssistantText SKIPS the error → reviewer sees the GENUINE prior draft → NOFIRE.
// Same transcript, same creds/model; the ONLY variable is whether the #5 skip is applied. The genuine
// draft is a bare, groundable statement so GREEN's nofire isolates the skip (not draft quality).
import type { CapturedEvent } from "../../src/core/evidence-capture.ts";
import { extractLastAssistantText, isHarnessError } from "../../src/core/prose-stop.ts";
import { anthropicReviewerAgentic } from "../../src/core/reviewer-agentic.ts";

const GENUINE = "I read src/core/prose-stop.ts this session.";
const HARNESS = "API Error: Server is temporarily limiting requests · Please try again later.";

// A real transcript shape: a genuine assistant draft, then a trailing assistant block that is just the
// harness error (the masquerade). extractLastAssistantText should return GENUINE, not HARNESS.
const asst = (text: string) => JSON.stringify({ type: "assistant", message: { role: "assistant", content: [{ type: "text", text }] } });
const transcript = [
  JSON.stringify({ type: "user", message: { role: "user", content: "review the skip" } }),
  asst(GENUINE),
  asst(HARNESS), // the masquerading error block — what the pre-#5 code reviewed
].join("\n");

// Sanity: confirm the extractor classifies + skips as designed before the behavioral arms.
const extracted = extractLastAssistantText(transcript);
console.log(`isHarnessError(HARNESS)=${isHarnessError(HARNESS)}  isHarnessError(GENUINE)=${isHarnessError(GENUINE)}`);
console.log(`extractLastAssistantText → ${extracted === GENUINE ? "GENUINE ✓ (error skipped)" : `"${extracted.slice(0, 60)}" ✗`}`);

// The trace: the genuine draft is groundable (the Read happened); nothing supports the harness string.
const events: CapturedEvent[] = [
  { tool: "Read", command: undefined, filePath: "src/core/prose-stop.ts", stdout: "", stderr: "", exitCode: 0, isNegative: false },
];

const REPS = Number(process.env.REPS) || 4;

const runArm = async (label: string, draft: string): Promise<number> => {
  let fires = 0;
  console.log(`\n--- ${label} draft="${draft.slice(0, 50)}…" ---`);
  for (let i = 0; i < REPS; i++) {
    const v = await anthropicReviewerAgentic({ action: draft, actionType: "stop-response", events });
    const fired = v.issues.length > 0;
    if (fired) fires++;
    console.log(`  rep ${i + 1}: ${fired ? "FIRE" : "nofire"} [${v.issues.map((x) => `${x.type}`).join(",") || "-"}]`);
    if (fired) for (const x of v.issues) console.log(`         • ${x.type}: ${(x.detail ?? "").slice(0, 150)}`);
  }
  console.log(`  ${label}: ${fires}/${REPS} fired`);
  return fires;
};

console.log(`\nmodel=${process.env.VOUCH_REVIEWER_MODEL ?? "(default)"} key=${process.env.ANTHROPIC_API_KEY ? "set" : "MISSING"} REPS=${REPS}`);
const red = await runArm("RED   (raw error reviewed, pre-#5)", HARNESS);
const green = await runArm("GREEN (extracted genuine draft, #5)", extracted);
console.log(`\n#5 red→green: RED ${red}/${REPS} fired, GREEN ${green}/${REPS} fired`);
console.log(red > green
  ? `✓ the #5 skip prevents the harness-error cry-wolf (${red}→${green})`
  : `✗ no clean separation (red=${red} green=${green}) — the error string did not reliably cry-wolf`);
