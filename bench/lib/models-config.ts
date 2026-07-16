// Shared model/credential config for the bench runners — the env loader + MODELS_DEF that was
// copy-pasted verbatim into every decision-audit runner (run.ts, prompt-experiment.ts,
// precision-experiment.ts, diagnose-*.ts). Centralizing it means a change to an env-key name or a
// new arm is edited ONCE, not in N places (the same drift concern reviewer-retry.ts centralizes).
//
// `.env` convention: ANTHROPIC_*/VOUCH_REVIEWER_MODEL = the DEPLOYED model; DEEPSEEK_*/KIMI_* =
// the bench A/B arms. setModelEnv() points the reviewer at one arm by mutating process.env (the
// reviewer reads ANTHROPIC_API_KEY/ANTHROPIC_BASE_URL/VOUCH_REVIEWER_MODEL at call time).

import { readFileSync } from "node:fs";
import { join } from "node:path";

const ROOT = join(import.meta.dir, "..", "..");
const envFile = readFileSync(join(ROOT, ".env"), "utf8");

export const envOf = (k: string): string => envFile.match(new RegExp(`^${k}=(.*)$`, "m"))?.[1] ?? "";

export interface ModelArm {
  apiKey: string;
  baseURL: string;
  model: string;
}

export const MODELS_DEF: Record<string, ModelArm> = {
  deepseek: { apiKey: envOf("DEEPSEEK_API_KEY"), baseURL: envOf("DEEPSEEK_BASE_URL"), model: envOf("DEEPSEEK_MODEL") },
  kimi: { apiKey: envOf("KIMI_API_KEY"), baseURL: envOf("KIMI_BASE_URL"), model: envOf("KIMI_MODEL") },
};

// Point the reviewer at one arm. Optionally set/clear the experiment-only prompt extra in the
// same call (UNSET in production, so the deployed prompt is byte-identical).
export function setModelEnv(name: string, promptExtra?: string): ModelArm {
  const m = MODELS_DEF[name];
  if (!m) throw new Error(`unknown model arm: ${name}`);
  process.env.ANTHROPIC_API_KEY = m.apiKey;
  process.env.ANTHROPIC_BASE_URL = m.baseURL;
  process.env.VOUCH_REVIEWER_MODEL = m.model;
  if (promptExtra) process.env.VOUCH_REVIEWER_PROMPT_EXTRA = promptExtra;
  else delete process.env.VOUCH_REVIEWER_PROMPT_EXTRA;
  return m;
}
