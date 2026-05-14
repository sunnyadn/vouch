#!/usr/bin/env bun
/**
 * classify_fires.ts — keypress TUI for hand-labeling vouch gate fires
 * produced by extract_fires.ts.
 *
 * Label vocabulary (single keystroke, case-insensitive):
 *
 *   A  verified         — true-positive fire. Agent ran `vouch fetch` /
 *                         `vouch claim` (or otherwise grew the KB) and the
 *                         revise cites the new dossier.
 *   H  hedged           — true-positive fire. Agent kept the claim with
 *                         an explicit "(Unverified, from training memory)"
 *                         tag or equivalent provenance disclosure.
 *   C  continued-confab — true-positive fire. Agent ignored the fire and
 *                         repeated the same ungrounded claim.
 *   D  dodge            — true-positive fire. Agent rephrased to remove
 *                         the named entity OR argued the fire was a false
 *                         positive without verifying. The #50-binding
 *                         pattern.
 *   F  false-positive   — fire was wrong. Agent's claim was actually fine
 *                         (extractor over-fire, workspace-meta misjudge,
 *                         NLI too strict against a quote that was already
 *                         present, etc).
 *   S  skip             — defer; revisit later. Reasons: ambiguous,
 *                         multi-claim fire with mixed verdicts, needs
 *                         transcript context this row doesn't carry.
 *   N  notes            — open the row in $EDITOR (or vi) for free-form
 *                         analysis; sets the row's notes field on next
 *                         label.
 *   ←  undo             — revert the last label (press 'u' or '['). Walks
 *                         the in-memory ring back one step.
 *   →  next             — accept current label and advance (any A/H/C/D/F/S
 *                         keystroke moves forward automatically).
 *   Q  quit             — save and exit. Progress always saves on label;
 *                         this is for early exit between rows.
 *
 * Derived metrics (printed on quit / via --stats):
 *   gate_lift_rate = (A + H) / (A + H + C + D)
 *     "of true-positive fires, what fraction produced grounded or
 *      explicit-uncertainty output". Phase 1 bench result claims ~70%
 *      under controlled conditions; this measures it in the wild.
 *   dodge_rate     = D / (A + H + C + D)
 *     #50 binding metric — the gap a tighter forcing function must close.
 *   fp_rate        = F / total
 *     inverse of vouch precision in the dogfood corpus.
 *
 * Storage: appends labeled rows to fires-labeled.jsonl (one row per
 * label event, includes original FireRow content + manual_label + label_ts).
 * Re-running picks up where you left off (skips rows whose transcript_id
 * + ts pair is already present in the labeled file).
 *
 * Usage:
 *   ./classify_fires.ts                              # default input fires-last14d.jsonl
 *   ./classify_fires.ts --in fires-last7d.jsonl
 *   ./classify_fires.ts --out fires-labeled.jsonl
 *   ./classify_fires.ts --stats                      # don't open TUI, just print metrics
 *   ./classify_fires.ts --filter-repo redacted-meta     # only fires from this project dir
 *   ./classify_fires.ts --filter-from 2026-05-13     # only fires after this date (ISO)
 */
import { readFileSync, writeFileSync, existsSync, appendFileSync } from "node:fs";
import { join } from "node:path";

interface Args {
  in: string;
  out: string;
  stats: boolean;
  audit: boolean;
  filterRepo?: string;
  filterFrom?: string;
  filterClass?: string;
}

function parseArgs(): Args {
  const args = process.argv.slice(2);
  const out: Args = {
    in: join(import.meta.dir, "fires-last14d.jsonl"),
    out: join(import.meta.dir, "fires-labeled.jsonl"),
    stats: false,
    audit: false,
  };
  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (a === "--in") out.in = args[++i]!;
    else if (a === "--out") out.out = args[++i]!;
    else if (a === "--stats") out.stats = true;
    else if (a === "--audit-mode") out.audit = true;
    else if (a === "--filter-repo") out.filterRepo = args[++i];
    else if (a === "--filter-from") out.filterFrom = args[++i];
    else if (a === "--filter-class") out.filterClass = args[++i];
    else if (a === "-h" || a === "--help") {
      console.error(
        "Usage: classify_fires.ts [--in PATH] [--out PATH] [--stats] [--audit-mode] [--filter-class CLASS] [--filter-repo NAME] [--filter-from YYYY-MM-DD]",
      );
      process.exit(0);
    }
  }
  return out;
}

interface FireRow {
  ts: string;
  transcript_id: string;
  repo: string;
  git_branch?: string;
  cwd?: string;
  fire_text: string;
  propositions: Array<{
    entity: string;
    proposition: string;
    candidates_count: number | null;
    reason: string;
  }>;
  draft: string;
  prior_user: string;
  post_fire_draft?: string;
  manual_label: { class: string; notes?: string } | null;
}

interface LabeledRow extends FireRow {
  manual_label: {
    class: string;
    notes?: string;
    auto?: boolean;
    reasoning?: string;
    auto_audited?: boolean;
    human_override?: boolean;
    original_auto_class?: string;
  };
  label_ts: string;
  judge_model?: string;
}

const VALID_CLASSES = new Set(["verified", "hedged", "continued-confab", "dodge", "false-positive", "skip"]);
const KEY_TO_CLASS: Record<string, string> = {
  a: "verified",
  h: "hedged",
  c: "continued-confab",
  d: "dodge",
  f: "false-positive",
  s: "skip",
};

function loadJsonl<T>(path: string): T[] {
  if (!existsSync(path)) return [];
  const text = readFileSync(path, "utf8");
  return text
    .split("\n")
    .filter((l) => l.trim())
    .map((l) => {
      try {
        return JSON.parse(l) as T;
      } catch {
        return null;
      }
    })
    .filter((x): x is T => x !== null);
}

function rowKey(r: { transcript_id: string; ts: string }): string {
  return `${r.transcript_id}|${r.ts}`;
}

function truncate(s: string, n: number): string {
  if (!s) return "";
  if (s.length <= n) return s;
  return s.slice(0, n) + `\n  … [${s.length - n} more chars]`;
}

function printRow(row: FireRow, idx: number, total: number): void {
  console.log("\x1b[2J\x1b[H"); // clear screen
  console.log(`\x1b[1;36m═══ fire ${idx + 1} / ${total} ═══\x1b[0m`);
  console.log(`  ts:        ${row.ts}`);
  console.log(`  repo:      ${row.repo}`);
  console.log(`  branch:    ${row.git_branch || "—"}`);
  console.log(`  transcript: ${row.transcript_id.slice(0, 8)}…`);
  console.log("");

  console.log(`\x1b[1;33m── prior_user ──\x1b[0m`);
  console.log(`  ${truncate(row.prior_user, 400).split("\n").join("\n  ")}`);
  console.log("");

  console.log(`\x1b[1;35m── draft (the text that fired) ──\x1b[0m`);
  console.log(`  ${truncate(row.draft, 800).split("\n").join("\n  ")}`);
  console.log("");

  console.log(`\x1b[1;31m── fire_text (vouch gate block) ──\x1b[0m`);
  console.log(`  ${truncate(row.fire_text, 400).split("\n").join("\n  ")}`);
  console.log("");

  if (row.propositions.length) {
    console.log(`\x1b[1;31m── propositions (${row.propositions.length}) ──\x1b[0m`);
    for (const p of row.propositions) {
      console.log(`  • \x1b[1m${p.entity}\x1b[0m: "${truncate(p.proposition, 200)}"`);
      console.log(`      candidates=${p.candidates_count ?? "?"}`);
    }
    console.log("");
  }

  console.log(`\x1b[1;32m── post_fire_draft (the revise) ──\x1b[0m`);
  if (row.post_fire_draft && row.post_fire_draft.trim()) {
    console.log(`  ${truncate(row.post_fire_draft, 1000).split("\n").join("\n  ")}`);
  } else {
    console.log(`  \x1b[2m(empty — session may have ended at the fire, or extract pass too early)\x1b[0m`);
  }
  console.log("");

  console.log(
    `\x1b[1m[A]\x1b[0m verified  \x1b[1m[H]\x1b[0m hedged  \x1b[1m[C]\x1b[0m continued-confab  \x1b[1m[D]\x1b[0m dodge  \x1b[1m[F]\x1b[0m false-positive  \x1b[1m[S]\x1b[0m skip  \x1b[1m[U]\x1b[0m undo  \x1b[1m[Q]\x1b[0m quit`,
  );
}

function printStats(labeled: LabeledRow[]): void {
  const counts: Record<string, number> = {};
  let autoCount = 0;
  let humanAudited = 0;
  let humanOverridden = 0;
  for (const r of labeled) {
    counts[r.manual_label.class] = (counts[r.manual_label.class] || 0) + 1;
    if (r.manual_label.auto) autoCount++;
    if (r.manual_label.auto_audited) humanAudited++;
    if (r.manual_label.human_override) humanOverridden++;
  }
  const A = counts["verified"] || 0;
  const H = counts["hedged"] || 0;
  const C = counts["continued-confab"] || 0;
  const D = counts["dodge"] || 0;
  const F = counts["false-positive"] || 0;
  const S = counts["skip"] || 0;
  const total = labeled.length;
  const tpDenom = A + H + C + D;

  const pct = (n: number, d: number) => (d > 0 ? `${((n / d) * 100).toFixed(1)}%` : "—");

  console.log("");
  console.log(`\x1b[1mlabel distribution\x1b[0m  (total = ${total})`);
  console.log(`  verified         ${A.toString().padStart(4)}   ${pct(A, total)}`);
  console.log(`  hedged           ${H.toString().padStart(4)}   ${pct(H, total)}`);
  console.log(`  continued-confab ${C.toString().padStart(4)}   ${pct(C, total)}`);
  console.log(`  dodge            ${D.toString().padStart(4)}   ${pct(D, total)}`);
  console.log(`  false-positive   ${F.toString().padStart(4)}   ${pct(F, total)}`);
  console.log(`  skip             ${S.toString().padStart(4)}   ${pct(S, total)}`);
  console.log("");
  console.log(`\x1b[1mderived metrics\x1b[0m  (denom = TP fires = A+H+C+D = ${tpDenom})`);
  console.log(`  gate_lift_rate     = (A + H) / TP        = ${pct(A + H, tpDenom)}`);
  console.log(`  dodge_rate         = D / TP              = ${pct(D, tpDenom)}      \x1b[2m(#50 binding)\x1b[0m`);
  console.log(`  confab_persist     = C / TP              = ${pct(C, tpDenom)}`);
  console.log(`  fp_rate            = F / total           = ${pct(F, total)}        \x1b[2m(inverse vouch precision)\x1b[0m`);
  console.log("");
  if (autoCount > 0) {
    console.log(`\x1b[1maudit provenance\x1b[0m`);
    console.log(`  auto-labeled (LLM)        ${autoCount.toString().padStart(4)}   ${pct(autoCount, total)}`);
    console.log(`  human-audited             ${humanAudited.toString().padStart(4)}   ${pct(humanAudited, autoCount)} of auto`);
    console.log(`  human override on audit   ${humanOverridden.toString().padStart(4)}   ${pct(humanOverridden, Math.max(1, humanAudited))} of audited (judge error rate proxy)`);
    console.log("");
  }
}

async function readKeypress(): Promise<string> {
  return new Promise((resolve) => {
    const stdin = process.stdin;
    stdin.setRawMode(true);
    stdin.resume();
    stdin.setEncoding("utf8");
    const onData = (key: string) => {
      stdin.setRawMode(false);
      stdin.pause();
      stdin.removeListener("data", onData);
      resolve(key);
    };
    stdin.on("data", onData);
  });
}

async function runAuditMode(args: Args): Promise<void> {
  const labeled = loadJsonl<LabeledRow>(args.out);
  if (!labeled.length) {
    console.error(`no labeled rows in ${args.out} — run auto_classify_fires.ts first`);
    process.exit(1);
  }

  // Audit only rows that were auto-labeled and haven't been human-audited yet
  let pending = labeled
    .map((r, i) => ({ r, i }))
    .filter(({ r }) => r.manual_label.auto && !r.manual_label.auto_audited);

  if (args.filterRepo) pending = pending.filter(({ r }) => r.repo.includes(args.filterRepo!));
  if (args.filterFrom) pending = pending.filter(({ r }) => r.ts >= args.filterFrom!);
  if (args.filterClass) pending = pending.filter(({ r }) => r.manual_label.class === args.filterClass);

  console.error(`audit-mode: ${labeled.length} labeled total, ${pending.length} pending audit`);
  if (!pending.length) {
    console.error("nothing to audit.");
    printStats(labeled);
    return;
  }

  await new Promise((r) => setTimeout(r, 600));

  let cursor = 0;
  const history: number[] = []; // indices in `labeled` that were modified this session, for undo

  while (cursor < pending.length) {
    const { r: row, i: globalIdx } = pending[cursor]!;
    const judgeClass = row.manual_label.class;
    const judgeReasoning = row.manual_label.reasoning || "(no reasoning recorded)";

    // Render row (same shape as printRow but with judge banner)
    console.log("\x1b[2J\x1b[H");
    console.log(`\x1b[1;36m═══ audit ${cursor + 1} / ${pending.length} ═══\x1b[0m`);
    console.log(`  ts:        ${row.ts}`);
    console.log(`  repo:      ${row.repo}`);
    console.log(`  branch:    ${row.git_branch || "—"}`);
    console.log("");
    console.log(`\x1b[1;33m── prior_user ──\x1b[0m`);
    console.log(`  ${truncate(row.prior_user, 350).split("\n").join("\n  ")}`);
    console.log("");
    console.log(`\x1b[1;35m── draft ──\x1b[0m`);
    console.log(`  ${truncate(row.draft, 700).split("\n").join("\n  ")}`);
    console.log("");
    if (row.propositions.length) {
      console.log(`\x1b[1;31m── propositions ──\x1b[0m`);
      for (const p of row.propositions.slice(0, 3)) {
        console.log(`  • \x1b[1m${p.entity}\x1b[0m: "${truncate(p.proposition, 180)}"`);
      }
      console.log("");
    }
    console.log(`\x1b[1;32m── post_fire_draft (revise) ──\x1b[0m`);
    if (row.post_fire_draft?.trim()) {
      console.log(`  ${truncate(row.post_fire_draft, 900).split("\n").join("\n  ")}`);
    } else {
      console.log(`  \x1b[2m(empty)\x1b[0m`);
    }
    console.log("");
    console.log(`\x1b[1;44;37m JUDGE \x1b[0m  class = \x1b[1m${judgeClass}\x1b[0m   model = ${row.judge_model || "?"}`);
    console.log(`  reasoning: ${truncate(judgeReasoning, 500).split("\n").join("\n             ")}`);
    console.log("");
    console.log(
      `\x1b[1m[Enter/Y]\x1b[0m accept  \x1b[1m[A]\x1b[0m verified  \x1b[1m[H]\x1b[0m hedged  \x1b[1m[C]\x1b[0m confab  \x1b[1m[D]\x1b[0m dodge  \x1b[1m[F]\x1b[0m fp  \x1b[1m[S]\x1b[0m skip  \x1b[1m[U]\x1b[0m undo  \x1b[1m[Q]\x1b[0m quit`,
    );

    const key = (await readKeypress()).toLowerCase();

    if (key === "q" || key === "") {
      writeFileSync(args.out, labeled.map((r) => JSON.stringify(r)).join("\n") + "\n");
      console.log("");
      console.log(`audit session: reviewed ${history.length}, advancing cursor at ${cursor}.`);
      printStats(labeled);
      return;
    }

    if (key === "u" || key === "[") {
      if (!history.length) {
        await new Promise((r) => setTimeout(r, 300));
        continue;
      }
      const prevIdx = history.pop()!;
      const prev = labeled[prevIdx]!;
      // Revert the audit flag (and any override) on the previous decision
      prev.manual_label.auto_audited = false;
      if (prev.manual_label.original_auto_class) {
        prev.manual_label.class = prev.manual_label.original_auto_class;
        delete prev.manual_label.original_auto_class;
        delete prev.manual_label.human_override;
      }
      cursor = Math.max(0, cursor - 1);
      // Re-include the previous row in pending if it was filtered-in originally — simplest: just decrement
      writeFileSync(args.out, labeled.map((r) => JSON.stringify(r)).join("\n") + "\n");
      continue;
    }

    if (key === "\r" || key === "\n" || key === "y") {
      // Accept judge label
      labeled[globalIdx]!.manual_label.auto_audited = true;
      history.push(globalIdx);
    } else {
      const newClass = KEY_TO_CLASS[key];
      if (!newClass) continue; // unknown key, redraw
      labeled[globalIdx]!.manual_label.original_auto_class = judgeClass;
      labeled[globalIdx]!.manual_label.class = newClass;
      labeled[globalIdx]!.manual_label.auto_audited = true;
      labeled[globalIdx]!.manual_label.human_override = true;
      history.push(globalIdx);
    }
    // Persist after each decision
    writeFileSync(args.out, labeled.map((r) => JSON.stringify(r)).join("\n") + "\n");
    cursor++;
  }

  console.log("");
  console.log(`\x1b[1;32maudit complete — reviewed ${pending.length} rows.\x1b[0m`);
  printStats(labeled);
}

async function main() {
  const args = parseArgs();

  if (args.audit) {
    await runAuditMode(args);
    return;
  }

  const all = loadJsonl<FireRow>(args.in);
  if (!all.length) {
    console.error(`no fires in ${args.in}`);
    process.exit(1);
  }

  let filtered = all;
  if (args.filterRepo) {
    filtered = filtered.filter((r) => r.repo.includes(args.filterRepo!));
  }
  if (args.filterFrom) {
    filtered = filtered.filter((r) => r.ts >= args.filterFrom!);
  }

  const labeled = loadJsonl<LabeledRow>(args.out);

  if (args.stats) {
    printStats(labeled);
    return;
  }

  const labeledKeys = new Set(labeled.map(rowKey));
  const remaining = filtered.filter((r) => !labeledKeys.has(rowKey(r)));

  console.error(`input:     ${args.in}  (${all.length} total, ${filtered.length} after filters)`);
  console.error(`labeled:   ${args.out}  (${labeled.length} already labeled)`);
  console.error(`remaining: ${remaining.length}`);
  if (!remaining.length) {
    console.error("nothing to label.");
    printStats(labeled);
    return;
  }

  // Brief pause so the user sees the input summary
  await new Promise((r) => setTimeout(r, 600));

  let idx = 0;
  const sessionLabels: LabeledRow[] = [];

  while (idx < remaining.length) {
    const row = remaining[idx]!;
    printRow(row, idx, remaining.length);
    const key = (await readKeypress()).toLowerCase();

    if (key === "q" || key === "") {
      // q or Ctrl-C
      console.log("");
      console.log(`session: labeled ${sessionLabels.length} this run.`);
      printStats([...labeled, ...sessionLabels]);
      return;
    }

    if (key === "u" || key === "[") {
      // undo last
      if (sessionLabels.length === 0) {
        console.log("\x1b[2m  (nothing to undo)\x1b[0m");
        await new Promise((r) => setTimeout(r, 400));
        continue;
      }
      sessionLabels.pop();
      idx = Math.max(0, idx - 1);
      // Need to remove from disk too — rewrite the labeled file
      const fresh = loadJsonl<LabeledRow>(args.out);
      const toKeep = fresh.slice(0, -1);
      writeFileSync(args.out, toKeep.map((r) => JSON.stringify(r)).join("\n") + (toKeep.length ? "\n" : ""));
      continue;
    }

    const cls = KEY_TO_CLASS[key];
    if (!cls) {
      // unknown key, redraw
      continue;
    }

    const labeled_row: LabeledRow = {
      ...row,
      manual_label: { class: cls },
      label_ts: new Date().toISOString(),
    };
    sessionLabels.push(labeled_row);
    appendFileSync(args.out, JSON.stringify(labeled_row) + "\n");
    idx++;
  }

  console.log("");
  console.log(`\x1b[1;32mdone — labeled ${sessionLabels.length} this run, ${labeled.length + sessionLabels.length} total.\x1b[0m`);
  printStats([...labeled, ...sessionLabels]);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
