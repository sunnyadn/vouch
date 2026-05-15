/** L4 harvest verb — file derived claims tagged inline in a draft.
 *
 * Two consumers:
 *
 *   1. Stop-hook gate (src/gate.ts): on a PASSING draft, harvests every
 *      `[inference-from:]` / `[synthesis-of:]` / `[interpretation:]` /
 *      `[hypothesis]` segment as a derived claim of the matching type.
 *      `[verified: <id>]` tags don't file anything; they're validation
 *      checks on existing ids (flagged when dangling / superseded /
 *      non-supported).
 *
 *   2. Agent-callable CLI: `vouch harvest <file>` — runs the same pipeline
 *      on any markdown file. Useful when the agent wrote a tagged draft
 *      outside a gate-instrumented session (e.g., saved to disk, run from
 *      a script).
 *
 * Single source of truth: gate.ts now imports harvestDerivedClaims /
 * safeHarvest from here instead of inlining them.
 *
 * Tag table:
 *   [verified: <id>]                  → validate existing claim ids
 *   [inference-from: <id1>, <id2>]    → INFERENCE claim, deps cited
 *   [synthesis-of: <id1>, <id2>]      → SYNTHESIS claim, deps cited
 *   [interpretation: <id>]            → INTERPRETATION claim, single dep
 *   [hypothesis]                      → HYPOTHESIS claim, no deps required
 *   [hypothesis; score: 0.4]          → soft_score override
 *
 * Tags inside code fences, inline backticks, blockquotes, or indented
 * code blocks are ignored — those are prose ABOUT the syntax, not
 * assertions using it.
 *
 * Dedup: same (claim_text, claim_type, depends_on-set) already in the KB →
 * not re-filed (re-emitting a draft after a block, or the agent running
 * `vouch claim` by hand on the same segment).
 *
 * Fail-soft: bad tags become advisory flags rather than throwing. Embed
 * failures don't block the claim record — just means the claim isn't
 * hybrid-searchable. safeHarvest wraps the whole pipeline in try/catch
 * so the gate never sinks an otherwise-passing turn on a harvest bug.
 */

import { embedOne } from "./embedder.ts";
import * as store from "./store.ts";
import type { ClaimType } from "./types.ts";

export type DerivedTagKind =
  | "verified"
  | "inference-from"
  | "synthesis-of"
  | "interpretation"
  | "hypothesis";

// Longest alternatives first — `inference-from` before `inference`, etc. — so
// the regex engine commits to the specific form when the agent wrote one.
// Bare `inference` / `synthesis` are accepted as sloppy-agent aliases.
const DERIVED_TAG_RE =
  /\[(verified|inference-from|inference|synthesis-of|synthesis|interpretation|hypothesis)\b([^\]]*)\]/gi;

const TAG_ALIAS: Record<string, DerivedTagKind> = {
  inference: "inference-from",
  synthesis: "synthesis-of",
};

const TAG_TO_CLAIM_TYPE: Record<Exclude<DerivedTagKind, "verified">, ClaimType> = {
  "inference-from": "INFERENCE",
  "synthesis-of": "SYNTHESIS",
  interpretation: "INTERPRETATION",
  hypothesis: "HYPOTHESIS",
};

const SOFT_SCORE_DEFAULT: Record<ClaimType, number> = {
  INFERENCE: 0.7,
  SYNTHESIS: 0.7,
  INTERPRETATION: 0.7,
  HYPOTHESIS: 0.4,
  ATOMIC: 0.7,
  QUOTATION: 0.7,
};

/** Max derived claims harvested from one draft (mirrors the extractor's cap). */
const HARVEST_MAX = 20;
/** A cleaned segment longer than this is almost certainly a tagging artifact
 *  (a whole paragraph collapsed into one "claim"); file it but flag it. */
const SEGMENT_LONG_CHARS = 800;

export interface DerivedTag {
  kind: DerivedTagKind;
  /** Cited upstream claim ids (empty for `[hypothesis]`). */
  ids: number[];
  /** `; score: <0..1>` override, if present. */
  softScore: number | null;
  /** Verbatim text the tag annotates — back to the previous tag / sentence /
   *  line boundary. Empty when the tag has no preceding text. */
  segment: string;
}

export interface HarvestedClaim {
  claim_id: number;
  claim_type: ClaimType;
  claim_text: string;
  depends_on: number[];
  soft_score: number | null;
}

export interface HarvestResult {
  /** Newly filed derived claims. */
  filed: HarvestedClaim[];
  /** Tags that pointed at a (claim_text, type, deps) already in the KB — not
   *  re-filed. The id is the existing claim. */
  skipped: { claim_id: number; claim_type: ClaimType; claim_text: string }[];
  /** Soft advisories: dangling `[verified:]` ids, missing / superseded
   *  upstreams, malformed or oversized tags. Never affect the exit code. */
  flags: string[];
}

// Abbreviations whose trailing `.` is not a sentence terminator. Kept tiny —
// just the ones that actually show up in research prose.
const SEG_ABBREVS = new Set([
  "e.g", "i.e", "etc", "vs", "cf", "al", "fig", "figs", "eq", "eqn", "eqns",
  "sec", "tab", "approx", "vol", "pp", "et", "ed", "eds",
]);

/** True when the `.` ending `before` is part of an abbreviation ("e.g.",
 *  "Fig.", "et al.") or a multi-initial ("U.S.", "J.K.") — i.e. NOT a real
 *  sentence boundary. */
function isAbbrevBoundary(before: string): boolean {
  if (/[A-Za-z]\.[A-Za-z]$/.test(before)) return true; // "...U.S" / "...e.g"
  const w = before.match(/\b[\w.]+$/)?.[0];
  if (!w) return false;
  return SEG_ABBREVS.has(w.toLowerCase().replace(/^\.+|\.+$/g, ""));
}

/** Trim a tag's preceding text down to the proposition it annotates: drop
 *  everything before the last *real* sentence / newline boundary, then strip a
 *  leading list marker and any clause-joiner punctuation left over from the
 *  slice. Conservative — we never rewrite the agent's words, only trim boundary
 *  cruft. */
function cleanSegment(s: string): string {
  let t = s.trim();
  if (!t) return "";
  // A boundary is `.!?` + whitespace + (uppercase | quote/paren | line break),
  // or a blank line. The uppercase lookahead + abbrev guard dodge "e.g. Table"
  // / "U.S. Census" / "scored 0.9. it" mid-sentence false positives.
  const re = /([.!?])\s+(?=[A-Z"'(\[])|\n+/g;
  let cut = 0;
  let m: RegExpExecArray | null;
  while ((m = re.exec(t))) {
    if (m[1] && isAbbrevBoundary(t.slice(0, m.index))) continue;
    cut = m.index + m[0].length;
  }
  t = t.slice(cut).trim();
  t = t.replace(/^(?:[-*•–—]\s+|\d+[.)]\s+)/, "");   // leading list marker / ordinal
  t = t.replace(/^[\s.,;:—–-]+/, "");                 // leading punctuation from the cut
  return t.trim();
}

interface CharRange { start: number; end: number; }

/** Compute character ranges that should be skipped during harvest: inline code,
 *  code fences, blockquotes, and indented code blocks. */
export function computeProtectedRanges(draft: string): CharRange[] {
  const ranges: CharRange[] = [];
  const lines = draft.split("\n");
  let offset = 0;
  let i = 0;
  while (i < lines.length) {
    const line = lines[i]!;
    const lineEnd = offset + line.length;
    const fenceMatch = line.match(/^(```+|~~~+)/);
    if (fenceMatch) {
      const fenceLen = fenceMatch[1]!.length;
      const fenceStart = offset;
      i++;
      offset = lineEnd + 1;
      while (i < lines.length) {
        const closeLine = lines[i]!;
        const closeMatch = closeLine.match(/^(```+|~~~+)/);
        if (closeMatch && closeMatch[1]!.length >= fenceLen) {
          ranges.push({ start: fenceStart, end: offset + closeLine.length });
          i++;
          offset += closeLine.length + 1;
          break;
        }
        offset += closeLine.length + 1;
        i++;
      }
      continue;
    }
    if (line.startsWith("> ")) {
      ranges.push({ start: offset, end: lineEnd });
    }
    if (/^\s{4,}/.test(line) && !/^\s{4,}[-*•\d]/.test(line)) {
      ranges.push({ start: offset, end: lineEnd });
    }
    // Inline code spans on this line
    let j = 0;
    while (j < line.length) {
      if (line[j] !== "`") { j++; continue; }
      let runLen = 0;
      const openStart = j;
      while (j < line.length && line[j] === "`") { runLen++; j++; }
      let k = j;
      let found = false;
      while (k < line.length) {
        if (line[k] !== "`") { k++; continue; }
        let closeLen = 0;
        while (k < line.length && line[k] === "`") { closeLen++; k++; }
        if (closeLen === runLen) {
          ranges.push({ start: offset + openStart, end: offset + k });
          found = true;
          break;
        }
      }
      if (!found) break;
      j = k;
    }
    i++;
    offset = lineEnd + 1;
  }
  return ranges;
}

/** Parse `[verified|inference-from|synthesis-of|interpretation|hypothesis ...]`
 *  tags out of a draft, pairing each with the verbatim segment it annotates.
 *  `skipRanges` excludes matches whose start falls inside a code/quote region. */
export function parseDerivedTags(draft: string, skipRanges?: CharRange[]): DerivedTag[] {
  const out: DerivedTag[] = [];
  if (!draft) return out;
  DERIVED_TAG_RE.lastIndex = 0;
  let prevEnd = 0;
  let m: RegExpExecArray | null;
  while ((m = DERIVED_TAG_RE.exec(draft))) {
    const matchStart = m.index;
    const matchEnd = matchStart + m[0].length;
    if (skipRanges?.some((r) => matchStart >= r.start && matchStart < r.end)) {
      prevEnd = matchEnd;
      continue;
    }
    const raw = m[1]!.toLowerCase();
    const kind = (TAG_ALIAS[raw] ?? raw) as DerivedTagKind;
    const args = m[2] ?? "";
    // Pull out a `; score: 0.85` directive first, then read ids from the rest —
    // doing it the other way round would scrape "0" and "85" off the score.
    const scoreM = args.match(/score\s*:?\s*([01](?:\.\d+)?|\.\d+)\b/i);
    const softScore = scoreM ? Math.max(0, Math.min(1, parseFloat(scoreM[1]!))) : null;
    let idText = args;
    if (scoreM) {
      const si = scoreM.index ?? 0;
      idText = args.slice(0, si) + args.slice(si + scoreM[0].length);
    }
    const ids = (idText.match(/\d+/g) ?? []).map((n) => parseInt(n, 10));
    const segment = cleanSegment(draft.slice(prevEnd, matchStart));
    out.push({ kind, ids, softScore, segment });
    prevEnd = matchEnd;
  }
  return out;
}

/** Harvest tagged derived claims from a passing draft. Pure local SQLite writes
 *  plus one `embedOne` per filed claim (best-effort — a failed embed just means
 *  the claim isn't hybrid-searchable). Never throws on bad tags; bad tags become
 *  `flags`. */
export async function harvestDerivedClaims(
  draft: string,
  opts: { attribution?: string } = {},
): Promise<HarvestResult> {
  const result: HarvestResult = { filed: [], skipped: [], flags: [] };
  const attribution = opts.attribution ?? "claude-skill";
  const skipRanges = computeProtectedRanges(draft);

  for (const tag of parseDerivedTags(draft, skipRanges)) {
    if (result.filed.length >= HARVEST_MAX) {
      result.flags.push(`harvest cap (${HARVEST_MAX}) reached — remaining tagged segments not filed`);
      break;
    }

    if (tag.kind === "verified") {
      if (!tag.ids.length) {
        result.flags.push("[verified] tag has no claim id");
        continue;
      }
      for (const id of tag.ids) {
        const c = store.getClaim(id);
        if (!c) result.flags.push(`[verified: ${id}] — claim ${id} is not in the KB`);
        else if (c.superseded_by != null)
          result.flags.push(`[verified: ${id}] — claim ${id} is superseded by ${c.superseded_by}`);
        else if (c.status !== "supported")
          result.flags.push(`[verified: ${id}] — claim ${id} has status=${c.status}, not supported`);
      }
      continue;
    }

    const claimType = TAG_TO_CLAIM_TYPE[tag.kind];

    // Resolve upstream ids (skip for HYPOTHESIS — no dependency required).
    let deps: number[] = [];
    if (claimType !== "HYPOTHESIS") {
      const present: number[] = [];
      for (const id of tag.ids) {
        const c = store.getClaim(id);
        if (!c) {
          result.flags.push(`[${tag.kind}] cites claim ${id} which is not in the KB — dropped from depends_on`);
          continue;
        }
        if (c.superseded_by != null)
          result.flags.push(`[${tag.kind}] cites claim ${id} which is superseded by ${c.superseded_by}`);
        present.push(id);
      }
      deps = present;
      if (claimType === "INTERPRETATION" && deps.length > 1) {
        result.flags.push(`[interpretation:] cites ${deps.length} ids — using the first (${deps[0]})`);
        deps = [deps[0]!];
      }
      if (deps.length === 0) {
        result.flags.push(`[${tag.kind}] has no resolvable upstream claim id — not filed`);
        continue;
      }
    }

    const text = tag.segment;
    if (!text) {
      result.flags.push(`[${tag.kind}] tag has no annotated text — not filed`);
      continue;
    }
    if (!/[A-Za-z\p{L}]/u.test(text)) {
      result.flags.push(`[${tag.kind}] segment has no alphabetic content — not filed`);
      continue;
    }
    if (text.length > SEGMENT_LONG_CHARS) {
      result.flags.push(`[${tag.kind}] segment is ${text.length} chars — may be a tagging artifact; filed as-is`);
    }

    // Dedup: same (claim_text, claim_type, depends_on-set) already in the KB →
    // don't re-file (re-emitting a draft after a block, or the agent also
    // running `vouch claim` by hand).
    const sorted = (xs: number[]) => [...xs].sort((a, b) => a - b).join(",");
    const want = sorted(deps);
    const dup = store
      .findClaimsByTextType(text, claimType)
      .find((e) => sorted(e.depends_on.map((d) => d.depends_on_id)) === want);
    if (dup) {
      result.skipped.push({ claim_id: dup.id, claim_type: claimType, claim_text: text });
      continue;
    }

    const softScore = tag.softScore ?? SOFT_SCORE_DEFAULT[claimType];
    // Inherit topic when every upstream claim agrees on one.
    let topic: string | null = null;
    if (deps.length) {
      const topics = new Set(
        deps.map((id) => store.getClaim(id)?.topic ?? null).filter((x): x is string => !!x),
      );
      if (topics.size === 1) topic = [...topics][0]!;
    }
    let emb: Float32Array | null = null;
    try {
      emb = await embedOne(text);
    } catch {
      // non-fatal — claim still records, just won't be hybrid-searchable
    }
    const cid = store.recordClaim({
      dossier_slug: "",
      claim_text: text,
      score: null,
      status: "recorded",
      source_passage:
        claimType === "HYPOTHESIS"
          ? null
          : `tag-harvested ${claimType.toLowerCase()} over claim(s) ${deps.join(", ")}`,
      claim_type: claimType,
      topic,
      author: "gate-harvest",
      attribution,
      soft_score: softScore,
      depends_on_ids: deps.length ? deps : undefined,
      dependency_type: claimType === "SYNTHESIS" ? "support" : "inference",
      embedding: emb,
      verification: "tag-harvest",
    });
    result.filed.push({ claim_id: cid, claim_type: claimType, claim_text: text, depends_on: deps, soft_score: softScore });
  }
  return result;
}

/** Wrap harvestDerivedClaims so a harvest bug never sinks an otherwise-
 *  passing gate turn. Returns `undefined` when there's nothing to report
 *  (clean draft with no tags). */
export async function safeHarvest(draft: string): Promise<HarvestResult | undefined> {
  try {
    const h = await harvestDerivedClaims(draft);
    return h.filed.length || h.skipped.length || h.flags.length ? h : undefined;
  } catch {
    return undefined;
  }
}
