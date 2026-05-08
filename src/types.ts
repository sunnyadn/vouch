export type ClaimType =
  | "ATOMIC"
  | "SYNTHESIS"
  | "INFERENCE"
  | "INTERPRETATION"
  | "HYPOTHESIS"
  | "QUOTATION";

export type ClaimStatus = "supported" | "unsupported" | "insufficient";

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
  source_url?: string;
  source_quote?: string;
  source_title?: string;
  publication_date?: string;
  author_attribution?: string;
  sources?: { url: string; quote: string; title?: string }[];
  depends_on_ids?: number[];
  soft_score?: number;
}
