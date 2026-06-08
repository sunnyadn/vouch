// NLI verifier — absorbed (adapted, re-providered) from vouch's verifier.ts.
//
// vouch ran this on Vertex/Gemini via the `ai` SDK; vouch is Anthropic-only
// (the critic, src/agent/critic-runner.ts), so this routes through the SAME
// @anthropic-ai/sdk + ANTHROPIC_API_KEY surface — one provider story. The
// conservative-entailment prompt and the transient-vs-content error split carry
// over near-verbatim because they are the precision knob: a loose judge passes
// confabulations.
//
// The verdict shape is deliberately small and the verifier is INJECTABLE
// (NliVerify) so the prose-gate's decision logic can be unit-tested with a
// deterministic fake (tests/prose-gate.test.ts) while the eval uses the real
// model (bench/prose-gate/eval.ts).

import Anthropic from "@anthropic-ai/sdk";

export interface NliVerdict {
  supported: boolean;
  /** 0..1 confidence in the verdict */
  score: number;
  reason: string;
}

/** A claim→source entailment judge. Throws on verifier failure (caller decides
 *  fail-open via classifyError). */
export type NliVerify = (claim: string, source: string) => Promise<NliVerdict>;

/** Network / auth / rate-limit / timeout → transient (the caller should fail
 *  OPEN, never block the agent on infra failure). Everything else → content. */
export function classifyError(err: unknown): "transient" | "content" {
  const msg = err instanceof Error ? err.message.toLowerCase() : String(err).toLowerCase();
  if (
    msg.includes("rate") ||
    msg.includes("429") ||
    msg.includes("timeout") ||
    msg.includes("etimedout") ||
    msg.includes("econnreset") ||
    msg.includes("network") ||
    msg.includes("fetch failed") ||
    msg.includes("401") ||
    msg.includes("403") ||
    msg.includes("overloaded") ||
    msg.includes("529")
  ) {
    return "transient";
  }
  return "content";
}

const DEFAULT_NLI_MODEL = "claude-sonnet-4-5-20250929";

const NLI_PROMPT = `You are a strict natural-language-inference judge. Decide whether the SOURCE passage supports the CLAIM.

Rules:
- SUPPORTED only if the source DIRECTLY and FULLY entails the claim. Every
  material part of the claim must be present in, or strictly implied by, the source.
- If the source is merely topical, partially overlapping, or needs an extra
  unstated assumption, answer UNSUPPORTED.
- A claim with a specific number / quantity / id / name is SUPPORTED only if that
  exact number / quantity / id / name appears (or is exactly computable) in the source.
- Be conservative: when uncertain, answer UNSUPPORTED.

Output JSON only, no prose, no code fences:
{"supported": true|false, "score": 0..1, "reason": "<=200 chars"}`;

export function hasAnthropicCreds(): boolean {
  return !!process.env.ANTHROPIC_API_KEY;
}

/** Extract the first balanced {...} JSON object from model text (tolerates
 *  stray prose / code fences the model may add despite "JSON only"). */
function extractJsonObject(text: string): string | null {
  const start = text.indexOf("{");
  if (start === -1) return null;
  let depth = 0;
  for (let i = start; i < text.length; i++) {
    if (text[i] === "{") depth++;
    else if (text[i] === "}") {
      depth--;
      if (depth === 0) return text.slice(start, i + 1);
    }
  }
  return null;
}

/** Real NLI via Anthropic. Throws on missing creds, API failure, or unparseable
 *  output; callers classify the error and fail open on transient. */
export const anthropicNli: NliVerify = async (claim, source) => {
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) throw new Error("ANTHROPIC_API_KEY not set — NLI verifier requires it");

  const model = process.env.VOUCH_REVIEWER_MODEL ?? DEFAULT_NLI_MODEL;
  const client = new Anthropic({ apiKey });

  const message = await client.messages.create({
    model,
    max_tokens: 512,
    messages: [
      {
        role: "user",
        content: `${NLI_PROMPT}\n\nCLAIM:\n${claim}\n\nSOURCE:\n${source}`,
      },
    ],
  });

  const text = message.content
    .filter((b): b is Anthropic.TextBlock => b.type === "text")
    .map((b) => b.text)
    .join("");

  const json = extractJsonObject(text);
  if (!json) throw new Error(`NLI returned no JSON object: ${text.slice(0, 120)}`);
  const parsed = JSON.parse(json) as Partial<NliVerdict>;
  if (typeof parsed.supported !== "boolean") {
    throw new Error(`NLI JSON missing boolean 'supported': ${json.slice(0, 120)}`);
  }
  return {
    supported: parsed.supported,
    score: typeof parsed.score === "number" ? parsed.score : 0.5,
    reason: typeof parsed.reason === "string" ? parsed.reason : "",
  };
};
