import { homedir } from "node:os";
import { join } from "node:path";

export const DB_PATH = process.env.VOUCH_DB_PATH || join(homedir(), ".vouch", "store.db");

export const VERIFIER_MODEL =
  process.env.VOUCH_VERIFIER_MODEL || "vertex_ai/gemini-3.1-pro-preview";

export const EMBEDDER_MODEL =
  process.env.VOUCH_EMBEDDER_MODEL || "vertex_ai/text-embedding-005";

export const SUPPORT_THRESHOLD = parseFloat(
  process.env.VOUCH_SUPPORT_THRESHOLD || "0.6",
);

export const MAX_SOURCE_CHARS = 60_000;

export const VERTEX_PROJECT = process.env.GOOGLE_CLOUD_PROJECT || "";
export const VERTEX_LOCATION = process.env.GOOGLE_CLOUD_LOCATION || "global";
