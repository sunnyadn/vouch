/** LiteLLM-style provider/model string router → Vercel AI SDK model factories.
 *
 * Accepts strings like "vertex_ai/gemini-3.1-pro-preview", "openai/gpt-4o",
 * "anthropic/claude-sonnet-4-6". The bare model name (no slash) is treated as
 * Vertex by default — same convention as our Python LiteLLM config.
 */
import { createVertex } from "@ai-sdk/google-vertex";
import { createOpenAI } from "@ai-sdk/openai";
import { createAnthropic } from "@ai-sdk/anthropic";
import type { LanguageModel, EmbeddingModel } from "ai";

import { VERTEX_LOCATION, VERTEX_PROJECT } from "./config.ts";

let _vertex: ReturnType<typeof createVertex> | null = null;
let _openai: ReturnType<typeof createOpenAI> | null = null;
let _anthropic: ReturnType<typeof createAnthropic> | null = null;

function vertex() {
  if (_vertex) return _vertex;
  _vertex = createVertex({
    project: VERTEX_PROJECT || undefined,
    location: VERTEX_LOCATION,
  });
  return _vertex;
}

function openai() {
  if (_openai) return _openai;
  _openai = createOpenAI({});
  return _openai;
}

function anthropic() {
  if (_anthropic) return _anthropic;
  _anthropic = createAnthropic({});
  return _anthropic;
}

interface ParsedModel {
  provider: "vertex_ai" | "openai" | "anthropic";
  model: string;
}

export function parseModelString(s: string): ParsedModel {
  const idx = s.indexOf("/");
  if (idx === -1) {
    // No prefix: default to vertex
    return { provider: "vertex_ai", model: s };
  }
  const prov = s.slice(0, idx);
  const model = s.slice(idx + 1);
  if (prov === "vertex_ai" || prov === "vertex" || prov === "google-vertex") {
    return { provider: "vertex_ai", model };
  }
  if (prov === "openai") return { provider: "openai", model };
  if (prov === "anthropic") return { provider: "anthropic", model };
  throw new Error(
    `Unsupported provider "${prov}" in model string "${s}". ` +
      `Supported: vertex_ai/, openai/, anthropic/`,
  );
}

export function getLanguageModel(modelString: string): LanguageModel {
  const { provider, model } = parseModelString(modelString);
  if (provider === "vertex_ai") return vertex()(model);
  if (provider === "openai") return openai()(model);
  if (provider === "anthropic") return anthropic()(model);
  throw new Error(`unreachable: ${provider}`);
}

export function getEmbeddingModel(modelString: string): EmbeddingModel {
  const { provider, model } = parseModelString(modelString);
  if (provider === "vertex_ai") return vertex().textEmbeddingModel(model as any);
  if (provider === "openai") return openai().textEmbeddingModel(model as any);
  // Anthropic has no embedding endpoint
  throw new Error(
    `Embedding not available for provider "${provider}". ` +
      `Use vertex_ai/* or openai/*.`,
  );
}
