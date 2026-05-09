#!/usr/bin/env bun
/**
 * Smoke test for the `vouch gate` extractor prompt against a real LLM.
 *
 * Mocked unit tests (tests/gate.test.ts) can't validate prompt behavior — they
 * stub the classifier. Use this script after editing buildExtractPrompt to
 * confirm the prompt still classifies known fixtures correctly.
 *
 * Usage:
 *   bun run scripts/gate-smoke.ts
 *   VOUCH_GATE_MODEL=vertex_ai/gemini-3.1-flash-lite bun run scripts/gate-smoke.ts
 *
 * Each fixture asserts an `expected` outcome:
 *   - "skip": extractor should return zero pairs (workspace / textbook /
 *     hedged / claim-id-annotated)
 *   - "extract": extractor should return at least one pair (genuine
 *     external named-entity claim)
 *
 * Exit codes: 0 all pass, 1 any failure.
 */
import { DEFAULT_GATE_MODEL, extractPairs } from "../src/gate.ts";

interface Fixture {
  name: string;
  draft: string;
  expected: "skip" | "extract";
  /** Optional: substrings expected to appear in extracted entity names. */
  expectedEntities?: string[];
  /** SUN-61 class label for grouping. */
  klass?: "A" | "B" | "C" | "TP" | "PRIOR";
  /** Notes for humans reading output. */
  note?: string;
}

const fixtures: Fixture[] = [
  // ---------------------------------------------------------------------
  // SUN-61 class A — domain-textbook background
  // ---------------------------------------------------------------------
  {
    name: "A1: Fine-Gray method-of-X",
    klass: "A",
    expected: "skip",
    draft: "Fine-Gray is a statistical model used for competing risks analysis.",
    note: "Textbook-level domain background. Should NOT require vouch.",
  },
  {
    name: "A2: CauseSpecificCox method-of-X",
    klass: "A",
    expected: "skip",
    draft: "CauseSpecificCox is a statistical model used for competing risks analysis.",
  },
  {
    name: "A3: Gray test method-of-X",
    klass: "A",
    expected: "skip",
    draft: "Gray test is a statistical method for comparing cumulative incidence functions.",
  },

  // ---------------------------------------------------------------------
  // SUN-61 class B — workspace feature-support
  // ---------------------------------------------------------------------
  {
    name: "B1: third-party method as supported feature",
    klass: "B",
    expected: "skip",
    draft:
      "Wolbers + Uno IPCW concordance is a supported feature in our toolkit.",
    note: "Methodology is external but assertion is about workspace support.",
  },
  {
    name: "B2: roadmap framing",
    klass: "B",
    expected: "skip",
    draft: "Fine-Gray fitting is on the roadmap for our next release.",
  },

  // ---------------------------------------------------------------------
  // SUN-61 class C — claim-id parenthetical recognized as grounding
  // ---------------------------------------------------------------------
  {
    name: "C1: (vouch claim N) annotation",
    klass: "C",
    expected: "skip",
    draft:
      "Time-dep AUC for CR — Blanche P, Dartigues J-F, Jacqmin-Gadda H (2013), Statistics in Medicine 32(30):5381-5397 (vouch claim 140).",
    note: "Citation already grounded by claim_id; extractor should skip.",
  },
  {
    name: "C2: (claim_id: N) annotation, multiple",
    klass: "C",
    expected: "skip",
    draft:
      "Three citations: Gerds (claim 137), riskRegression (claim 139), Blanche (claim 140) — all grounded.",
  },

  // ---------------------------------------------------------------------
  // True-positive baseline (must continue to extract)
  // ---------------------------------------------------------------------
  {
    name: "TP: Patronus Lynx perf claim (baseline)",
    klass: "TP",
    expected: "extract",
    draft:
      "Patronus Lynx is an 8B-parameter hallucination detection model with 87.7% accuracy on HaluBench.",
    expectedEntities: ["Lynx", "Patronus", "HaluBench"],
    note: "Genuine external entity + specific perf number. Must still block.",
  },

  // ---------------------------------------------------------------------
  // Phase-1 regression spot-checks (should remain "skip")
  // ---------------------------------------------------------------------
  {
    name: "PRIOR: own-CLI surface (vouch internals)",
    klass: "PRIOR",
    expected: "skip",
    draft: "vouch claim accepts --type ATOMIC and --dossier <slug> flags.",
  },
  {
    name: "PRIOR: hedged from training memory",
    klass: "PRIOR",
    expected: "skip",
    draft:
      "FEVER has roughly 185k claims (unverified, from training memory — let me verify).",
  },
];

function passes(fx: Fixture, pairs: { entity: string; assertion: string }[]): boolean {
  if (fx.expected === "skip") return pairs.length === 0;
  if (!pairs.length) return false;
  if (fx.expectedEntities?.length) {
    const blob = pairs.map((p) => p.entity.toLowerCase()).join(" ");
    return fx.expectedEntities.some((e) => blob.includes(e.toLowerCase()));
  }
  return true;
}

async function main() {
  const model = process.env.VOUCH_GATE_MODEL || DEFAULT_GATE_MODEL;
  console.log(`# vouch gate smoke — model=${model}`);
  console.log(`# ${fixtures.length} fixtures\n`);

  let passed = 0;
  let failed = 0;
  const failures: string[] = [];

  for (const fx of fixtures) {
    const t0 = Date.now();
    const pairs = await extractPairs(fx.draft, model);
    const dt = Date.now() - t0;

    if (pairs === null) {
      console.log(`[ERROR ] ${fx.name} (${dt}ms) — extractor failed`);
      failed++;
      failures.push(fx.name + " (extractor null)");
      continue;
    }

    const ok = passes(fx, pairs);
    const tag = ok ? "PASS  " : "FAIL  ";
    const klass = fx.klass ? `[${fx.klass}]` : "    ";
    console.log(
      `[${tag}] ${klass} ${fx.name} (${dt}ms) — expected=${fx.expected}, got=${pairs.length} pair(s)`,
    );
    if (pairs.length) {
      for (const p of pairs) {
        console.log(`         · entity=${JSON.stringify(p.entity)} assertion=${JSON.stringify(p.assertion)}`);
      }
    }
    if (fx.note && !ok) console.log(`         note: ${fx.note}`);
    if (ok) passed++;
    else {
      failed++;
      failures.push(fx.name);
    }
  }

  console.log(`\nSummary: ${passed}/${fixtures.length} passed`);
  if (failed) {
    console.log(`Failures:`);
    for (const f of failures) console.log(`  • ${f}`);
    process.exit(1);
  }
}

main().catch((e) => {
  console.error("smoke crashed:", e?.message || e);
  process.exit(2);
});
