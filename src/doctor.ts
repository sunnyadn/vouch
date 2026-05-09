/** vouch doctor — diagnose config / env / DB / Claude Code hook installation.
 *
 * Pure local checks (no API calls). Reports per-check status with actionable
 * fix hints so first-touch users can self-resolve config issues without
 * digging through stack traces.
 */
import { existsSync, readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

import { DB_PATH, EMBEDDER_MODEL, VERIFIER_MODEL } from "./config.ts";
import { parseModelString } from "./providers.ts";
import * as store from "./store.ts";

type CheckStatus = "ok" | "warn" | "fail";

interface Check {
  name: string;
  status: CheckStatus;
  detail: string;
  fix?: string;
}

interface DoctorReport {
  ok: boolean;
  checks: Check[];
}

function checkProviderCredential(
  provider: string,
  role: "verifier" | "embedder",
): Check {
  const name = `${role}_credential`;
  if (provider === "anthropic") {
    return process.env.ANTHROPIC_API_KEY
      ? { name, status: "ok", detail: "ANTHROPIC_API_KEY is set" }
      : {
          name,
          status: "fail",
          detail: "ANTHROPIC_API_KEY is not set",
          fix: "export ANTHROPIC_API_KEY=sk-ant-... (or add to ~/.vouch/.env)",
        };
  }
  if (provider === "openai") {
    return process.env.OPENAI_API_KEY
      ? { name, status: "ok", detail: "OPENAI_API_KEY is set" }
      : {
          name,
          status: "fail",
          detail: "OPENAI_API_KEY is not set",
          fix: "export OPENAI_API_KEY=sk-... (or add to ~/.vouch/.env)",
        };
  }
  if (provider === "vertex_ai") {
    const credPath = process.env.GOOGLE_APPLICATION_CREDENTIALS;
    if (!credPath) {
      return {
        name,
        status: "fail",
        detail: "GOOGLE_APPLICATION_CREDENTIALS is not set",
        fix: "Point to your service-account JSON key: export GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa-key.json",
      };
    }
    if (!existsSync(credPath)) {
      return {
        name,
        status: "fail",
        detail: `GOOGLE_APPLICATION_CREDENTIALS points to a missing file: ${credPath}`,
        fix: "Verify the path is correct and the file is readable",
      };
    }
    const project = process.env.GOOGLE_CLOUD_PROJECT || process.env.GOOGLE_VERTEX_PROJECT;
    if (!project) {
      return {
        name,
        status: "fail",
        detail: "GOOGLE_CLOUD_PROJECT (or GOOGLE_VERTEX_PROJECT) is not set",
        fix: "export GOOGLE_CLOUD_PROJECT=<your-gcp-project-id>",
      };
    }
    return {
      name,
      status: "ok",
      detail: `SA key at ${credPath}, project=${project}`,
    };
  }
  return {
    name,
    status: "fail",
    detail: `Unknown provider: ${provider}`,
  };
}

export function runDoctor(): DoctorReport {
  const checks: Check[] = [];

  // 1. DB connectivity
  try {
    store.getDb().prepare("SELECT 1").get();
    checks.push({
      name: "db",
      status: "ok",
      detail: `${DB_PATH} (open + writable)`,
    });
  } catch (e: any) {
    checks.push({
      name: "db",
      status: "fail",
      detail: `Could not open ${DB_PATH}: ${e?.message || String(e)}`,
      fix: "Ensure the parent directory exists and is writable, or set VOUCH_DB_PATH",
    });
  }

  // 2. Verifier model + credential
  let verifierProvider: string | null = null;
  try {
    const parsed = parseModelString(VERIFIER_MODEL);
    verifierProvider = parsed.provider;
    checks.push({
      name: "verifier_model",
      status: "ok",
      detail: `${parsed.provider}/${parsed.model}`,
    });
  } catch (e: any) {
    checks.push({
      name: "verifier_model",
      status: "fail",
      detail: e?.message || String(e),
      fix: "Set VOUCH_VERIFIER_MODEL to a supported provider/model (e.g. anthropic/claude-sonnet-4-6)",
    });
  }
  if (verifierProvider) checks.push(checkProviderCredential(verifierProvider, "verifier"));

  // 3. Embedder model + credential (anthropic has no embedding endpoint via vouch)
  let embedderProvider: string | null = null;
  try {
    const parsed = parseModelString(EMBEDDER_MODEL);
    embedderProvider = parsed.provider;
    if (parsed.provider === "anthropic") {
      checks.push({
        name: "embedder_model",
        status: "fail",
        detail: "anthropic has no embedding endpoint exposed via vouch",
        fix: "Use vertex_ai/* or openai/* for VOUCH_EMBEDDER_MODEL — or unset to disable embeddings (vouch claim still works; vouch search will be unavailable)",
      });
    } else {
      checks.push({
        name: "embedder_model",
        status: "ok",
        detail: `${parsed.provider}/${parsed.model}`,
      });
    }
  } catch (e: any) {
    checks.push({
      name: "embedder_model",
      status: "warn",
      detail: e?.message || String(e),
      fix: "Embedder unset/invalid → vouch claim still works, but vouch search will be unavailable",
    });
  }
  if (embedderProvider && embedderProvider !== "anthropic") {
    checks.push(checkProviderCredential(embedderProvider, "embedder"));
  }

  // 4. Claude Code Stop hook (optional integration)
  const settingsPath = join(homedir(), ".claude", "settings.json");
  if (!existsSync(settingsPath)) {
    checks.push({
      name: "claude_code_hook",
      status: "warn",
      detail: `${settingsPath} not found — no Claude Code integration`,
      fix: "Optional. Install only if you use vouch from within Claude Code (see README §'Claude Code integration')",
    });
  } else {
    try {
      const settings = JSON.parse(readFileSync(settingsPath, "utf-8"));
      const stopHooks = settings?.hooks?.Stop ?? [];
      const hasVouchGate = JSON.stringify(stopHooks).includes("vouch gate");
      if (hasVouchGate) {
        checks.push({
          name: "claude_code_hook",
          status: "ok",
          detail: `vouch gate Stop hook installed in ${settingsPath}`,
        });
      } else {
        checks.push({
          name: "claude_code_hook",
          status: "warn",
          detail: `Settings file present but no vouch gate Stop hook`,
          fix: "See README §'Claude Code integration' for the JSON snippet to merge in",
        });
      }
    } catch (e: any) {
      checks.push({
        name: "claude_code_hook",
        status: "warn",
        detail: `Could not parse ${settingsPath}: ${e?.message || String(e)}`,
      });
    }
  }

  const ok = !checks.some((c) => c.status === "fail");
  return { ok, checks };
}
