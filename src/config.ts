import { existsSync, readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

// Load env from ~/.vouch/.env BEFORE reading any var. Bun auto-loads .env
// from cwd, but the skill calls `vouch` from arbitrary cwd — we need a
// stable per-user config location.
function loadDotEnv(path: string) {
  if (!existsSync(path)) return;
  for (const raw of readFileSync(path, "utf-8").split("\n")) {
    const line = raw.trim();
    if (!line || line.startsWith("#")) continue;
    const eq = line.indexOf("=");
    if (eq < 0) continue;
    const key = line.slice(0, eq).trim();
    let val = line.slice(eq + 1).trim();
    if ((val.startsWith('"') && val.endsWith('"')) || (val.startsWith("'") && val.endsWith("'"))) {
      val = val.slice(1, -1);
    }
    if (!(key in process.env)) process.env[key] = val;
  }
}
loadDotEnv(join(homedir(), ".vouch", ".env"));
loadDotEnv(join(process.cwd(), ".env"));

// Vercel AI SDK's @ai-sdk/google-vertex looks for GOOGLE_VERTEX_PROJECT/LOCATION,
// not the gcloud-canonical names. Map across so users only set one.
if (!process.env.GOOGLE_VERTEX_PROJECT && process.env.GOOGLE_CLOUD_PROJECT) {
  process.env.GOOGLE_VERTEX_PROJECT = process.env.GOOGLE_CLOUD_PROJECT;
}
if (!process.env.GOOGLE_VERTEX_LOCATION && process.env.GOOGLE_CLOUD_LOCATION) {
  process.env.GOOGLE_VERTEX_LOCATION = process.env.GOOGLE_CLOUD_LOCATION;
}

export const DB_PATH = process.env.VOUCH_DB_PATH || join(homedir(), ".vouch", "store.db");

export const VERIFIER_MODEL =
  process.env.VOUCH_VERIFIER_MODEL || "vertex_ai/gemini-3.1-pro-preview";

export const EMBEDDER_MODEL =
  process.env.VOUCH_EMBEDDER_MODEL || "vertex_ai/text-embedding-005";

export const SUPPORT_THRESHOLD = parseFloat(
  process.env.VOUCH_SUPPORT_THRESHOLD || "0.6",
);

export const MAX_SOURCE_CHARS = 60_000;

export const VERTEX_PROJECT =
  process.env.GOOGLE_VERTEX_PROJECT || process.env.GOOGLE_CLOUD_PROJECT || "";
export const VERTEX_LOCATION =
  process.env.GOOGLE_VERTEX_LOCATION || process.env.GOOGLE_CLOUD_LOCATION || "global";
