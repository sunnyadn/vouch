import { embed, embedMany } from "ai";

import { EMBEDDER_MODEL } from "./config.ts";
import { getEmbeddingModel } from "./providers.ts";

function l2Normalize(v: number[]): Float32Array {
  let sum = 0;
  for (const x of v) sum += x * x;
  const norm = Math.sqrt(sum) || 1;
  const out = new Float32Array(v.length);
  for (let i = 0; i < v.length; i++) out[i] = v[i]! / norm;
  return out;
}

export async function embedOne(text: string): Promise<Float32Array> {
  const model = getEmbeddingModel(EMBEDDER_MODEL);
  const { embedding } = await embed({ model, value: text });
  return l2Normalize(embedding);
}

export async function embedBatch(texts: string[]): Promise<Float32Array[]> {
  const model = getEmbeddingModel(EMBEDDER_MODEL);
  const { embeddings } = await embedMany({ model, values: texts });
  return embeddings.map(l2Normalize);
}
