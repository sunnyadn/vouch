export type ClaimType =
  | "ATOMIC"
  | "SYNTHESIS"
  | "INFERENCE"
  | "INTERPRETATION"
  | "HYPOTHESIS"
  | "QUOTATION";

export type ClaimStatus = "supported" | "unsupported" | "insufficient" | "recorded";

export type DependencyType = "inference" | "support";

export interface Dossier {
  slug: string;
  source_url: string;
  source_type: string;
  capture_date: string;
  last_refetched: string | null;
  source_hash: string | null;
  title: string | null;
  content: string;
  embedding: Float32Array | null;
  publication_date: string | null;
  author_attribution: string | null;
}

export interface Claim {
  id: number;
  dossier_slug: string;
  claim_text: string;
  source_passage: string | null;
  nli_score: number | null;
  status: ClaimStatus | string;
  verified_at: string;
  claim_type: ClaimType | string | null;
  topic: string | null;
  author: string | null;
  soft_score: number | null;
  attribution: string | null;
  superseded_by: number | null;
  supersede_reason: string | null;
  source_offset_start: number | null;
  source_offset_end: number | null;
  embedding: Float32Array | null;
  verification: string | null;
}

export interface ClaimDependency {
  claim_id: number;
  depends_on_id: number;
  dependency_type: DependencyType;
}

export interface VerifyResult {
  status: ClaimStatus;
  score: number;
  source_passage: string;
  verifier: string;
}

export interface SubmitClaimRequest {
  text: string;
  claim_type: ClaimType;
  topic?: string;
  attribution?: string;
  author?: string;
  /** ATOMIC/QUOTATION: dossier_slug from a prior `vouch fetch`. */
  dossier_slug?: string;
  /** ATOMIC/QUOTATION: verbatim 1–3 sentence quote from the dossier. */
  source_quote?: string;
  /** ATOMIC only: let vouch pick the best supporting passage from the dossier. */
  auto_quote?: boolean;
  /** SYNTHESIS: ≥2 (dossier_slug, quote) pairs. */
  sources?: { dossier_slug: string; quote: string }[];
  /** INFERENCE/INTERPRETATION/HYPOTHESIS: upstream claim IDs (DAG edges). */
  depends_on_ids?: number[];
  /** HYPOTHESIS / soft cases: caller's own confidence. */
  soft_score?: number;
}
