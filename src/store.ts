import { Database } from "bun:sqlite";
import { mkdirSync } from "node:fs";
import { dirname } from "node:path";
import { createHash } from "node:crypto";

import { DB_PATH } from "./config.ts";
import type { Claim, ClaimDependency, ClaimType, Dossier, DependencyType } from "./types.ts";

let _db: Database | null = null;

export function getDb(): Database {
  if (_db) return _db;
  mkdirSync(dirname(DB_PATH), { recursive: true });
  _db = new Database(DB_PATH);
  _db.exec("PRAGMA journal_mode=WAL;");
  initSchema(_db);
  return _db;
}

function initSchema(db: Database) {
  db.exec(`
    CREATE TABLE IF NOT EXISTS dossiers (
      slug TEXT PRIMARY KEY,
      source_url TEXT NOT NULL,
      source_type TEXT NOT NULL,
      capture_date TEXT NOT NULL,
      last_refetched TEXT,
      source_hash TEXT,
      title TEXT,
      content TEXT,
      embedding BLOB,
      publication_date TEXT,
      author_attribution TEXT
    );
    CREATE TABLE IF NOT EXISTS claims (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      dossier_slug TEXT,
      claim_text TEXT NOT NULL,
      source_passage TEXT,
      nli_score REAL,
      status TEXT NOT NULL,
      verified_at TEXT NOT NULL,
      claim_type TEXT,
      topic TEXT,
      author TEXT,
      soft_score REAL,
      attribution TEXT,
      superseded_by INTEGER,
      supersede_reason TEXT,
      source_offset_start INTEGER,
      source_offset_end INTEGER,
      embedding BLOB
    );
  `);
  // Additive migrations. SQLite lacks ADD COLUMN IF NOT EXISTS.
  for (const stmt of [
    `ALTER TABLE claims ADD COLUMN verification TEXT;`,
    // dossier provenance class: 'third-party' (web), 'workspace' (local file
    // the agent read), 'attested' (user self-declared). NULL on legacy rows /
    // the plain `vouch fetch` path — treated as 'third-party' by readers.
    `ALTER TABLE dossiers ADD COLUMN scope TEXT;`,
  ]) {
    try {
      db.exec(stmt);
    } catch (e: any) {
      if (e?.message?.includes("duplicate column")) {
        // Column already exists — no-op for idempotency
      } else {
        throw e;
      }
    }
  }
  db.exec(`
    CREATE TABLE IF NOT EXISTS claim_dependencies (
      claim_id INTEGER NOT NULL,
      depends_on_id INTEGER NOT NULL,
      dependency_type TEXT,
      PRIMARY KEY (claim_id, depends_on_id)
    );
    CREATE INDEX IF NOT EXISTS idx_dossiers_url ON dossiers(source_url);
    CREATE INDEX IF NOT EXISTS idx_claims_status ON claims(status);
    CREATE INDEX IF NOT EXISTS idx_claims_topic ON claims(topic);
    CREATE INDEX IF NOT EXISTS idx_claims_type ON claims(claim_type);
    CREATE INDEX IF NOT EXISTS idx_dep_claim ON claim_dependencies(claim_id);
    CREATE INDEX IF NOT EXISTS idx_dep_target ON claim_dependencies(depends_on_id);
  `);
  // Session claims ledger (issue #43) — per-transcript record of what the
  // agent has asserted in THIS session, with verdicts. Powers cross-turn
  // self-contradiction detection (a stateless gate cannot see "you said X
  // earlier; now saying ¬X").
  db.exec(`
    CREATE TABLE IF NOT EXISTS session_claims (
      transcript_id TEXT NOT NULL,
      turn_idx INTEGER NOT NULL,
      claim_idx INTEGER NOT NULL,
      proposition TEXT NOT NULL,
      entity TEXT NOT NULL,
      stance TEXT NOT NULL,
      verdict TEXT NOT NULL,
      reason TEXT,
      embedding BLOB,
      ts TEXT NOT NULL,
      superseded_by_turn INTEGER,
      superseded_by_claim INTEGER,
      retracted INTEGER NOT NULL DEFAULT 0,
      PRIMARY KEY (transcript_id, turn_idx, claim_idx)
    );
    CREATE INDEX IF NOT EXISTS idx_session_claims_tid ON session_claims(transcript_id);
    CREATE INDEX IF NOT EXISTS idx_session_claims_active
      ON session_claims(transcript_id, retracted, superseded_by_turn);
  `);
  // #50 (A) Stage 1: per-entity revise tracking. When a claim fires
  // (verdict='ungrounded' on an ASSERT), we set awaiting_revise=1. The
  // intent is to check, on subsequent turns, whether the agent (a) ran a
  // verification tool referencing the entity, (b) added an explicit hedge
  // tag adjacent to it, or (c) removed the entity from any new claim. The
  // outcome is recorded via addressed_via ∈ {'fetch','hedge','remove',null}
  // and addressed_in_turn. Stage 1 only writes awaiting_revise; the
  // detection/clearing logic ships in Stage 2.
  const sessionCols = db
    .prepare("PRAGMA table_info(session_claims)")
    .all() as Array<{ name: string }>;
  const colNames = new Set(sessionCols.map((c) => c.name));
  if (!colNames.has("awaiting_revise")) {
    db.exec("ALTER TABLE session_claims ADD COLUMN awaiting_revise INTEGER NOT NULL DEFAULT 0");
  }
  if (!colNames.has("addressed_via")) {
    db.exec("ALTER TABLE session_claims ADD COLUMN addressed_via TEXT");
  }
  if (!colNames.has("addressed_in_turn")) {
    db.exec("ALTER TABLE session_claims ADD COLUMN addressed_in_turn INTEGER");
  }
  db.exec(
    `CREATE INDEX IF NOT EXISTS idx_session_claims_awaiting
       ON session_claims(transcript_id, awaiting_revise)
       WHERE awaiting_revise = 1`,
  );
}

// ---------------------------------------------------------------------------
// Embedding marshaling — store as Float32 little-endian BLOB
// ---------------------------------------------------------------------------

export function embToBlob(emb: number[] | Float32Array): Buffer {
  const arr = emb instanceof Float32Array ? emb : new Float32Array(emb);
  return Buffer.from(arr.buffer, arr.byteOffset, arr.byteLength);
}

export function blobToEmb(blob: Buffer | Uint8Array | null): Float32Array | null {
  if (!blob) return null;
  const buf = blob instanceof Buffer ? blob : Buffer.from(blob);
  return new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
}

// ---------------------------------------------------------------------------
// Slug derivation
// ---------------------------------------------------------------------------

export function slugFromUrl(sourceType: string, url: string): string {
  const safe = url
    .replace(/^https?:\/\//, "")
    .replace(/[/?&=:]/g, "-")
    .slice(0, 80);
  const today = new Date().toISOString().slice(0, 10);
  return `evidence/${sourceType}/${safe}-${today}`;
}

/** Look up the most recent dossier ingested via vouch fetch (source_type
 *  derived from a fetcher, not "agent-quote"). Used for cache hits within the
 *  freshness window. */
export function getRecentFetchedDossier(url: string, withinHours = 24): Dossier | null {
  const cutoff = new Date(Date.now() - withinHours * 3600 * 1000).toISOString();
  const row = getDb()
    .prepare(
      "SELECT * FROM dossiers WHERE source_url = ? AND source_type != 'agent-quote' AND capture_date >= ? ORDER BY capture_date DESC LIMIT 1",
    )
    .get(url, cutoff) as any;
  if (!row) return null;
  row.embedding = blobToEmb(row.embedding);
  return row as Dossier;
}

export function sha256Hex(text: string): string {
  return createHash("sha256").update(text).digest("hex");
}

// ---------------------------------------------------------------------------
// Dossier ops
// ---------------------------------------------------------------------------

export interface WriteDossierInput {
  source_url: string;
  source_type: string;
  title?: string | null;
  verbatim_content: string;
  source_hash?: string | null;
  captured_at?: string;
  embedding?: number[] | Float32Array | null;
  publication_date?: string | null;
  author_attribution?: string | null;
  /** 'third-party' | 'workspace' | 'attested'. Defaults to NULL (≡ third-party). */
  scope?: string | null;
  slug?: string;
}

export function writeDossier(d: WriteDossierInput): string {
  const slug = d.slug || slugFromUrl(d.source_type, d.source_url);
  const captured = d.captured_at || new Date().toISOString();
  const hash = d.source_hash || sha256Hex(d.verbatim_content);
  const embBlob = d.embedding ? embToBlob(d.embedding) : null;

  getDb()
    .prepare(
      `INSERT OR REPLACE INTO dossiers
       (slug, source_url, source_type, capture_date, last_refetched, source_hash, title, content,
        embedding, publication_date, author_attribution, scope)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    )
    .run(
      slug,
      d.source_url,
      d.source_type,
      captured,
      new Date().toISOString(),
      hash,
      d.title || null,
      d.verbatim_content,
      embBlob,
      d.publication_date || null,
      d.author_attribution || null,
      d.scope || null,
    );
  return slug;
}

export function getDossier(slug: string): Dossier | null {
  const row = getDb().prepare("SELECT * FROM dossiers WHERE slug = ?").get(slug) as any;
  if (!row) return null;
  row.embedding = blobToEmb(row.embedding);
  return row as Dossier;
}

export function getDossierByUrl(url: string): Dossier | null {
  const db = getDb();
  let row = db
    .prepare("SELECT * FROM dossiers WHERE source_url = ? ORDER BY capture_date DESC LIMIT 1")
    .get(url) as any;
  if (!row) {
    // arxiv id-based variation match
    const arxivMatch = url.match(/arxiv\.org\/(?:abs|pdf|html)\/(\d{4}\.\d{4,5})/);
    if (arxivMatch) {
      row = db
        .prepare(
          "SELECT * FROM dossiers WHERE source_url LIKE ? ORDER BY capture_date DESC LIMIT 1",
        )
        .get(`%${arxivMatch[1]}%`) as any;
    } else {
      const stripped = url.replace(/^https?:\/\//, "").replace(/\/$/, "");
      if (stripped) {
        row = db
          .prepare(
            "SELECT * FROM dossiers WHERE source_url LIKE ? OR source_url LIKE ? ORDER BY capture_date DESC LIMIT 1",
          )
          .get(`%${stripped}`, `%${stripped}/`) as any;
      }
    }
  }
  if (!row) return null;
  row.embedding = blobToEmb(row.embedding);
  return row as Dossier;
}

export function listDossiers(opts: { source_type?: string; limit?: number } = {}): any[] {
  const limit = opts.limit ?? 50;
  const db = getDb();
  if (opts.source_type) {
    return db
      .prepare(
        `SELECT slug, title, source_url, source_type, capture_date, length(content) AS content_len,
                publication_date, author_attribution, scope
         FROM dossiers WHERE source_type = ? ORDER BY capture_date DESC LIMIT ?`,
      )
      .all(opts.source_type, limit);
  }
  return db
    .prepare(
      `SELECT slug, title, source_url, source_type, capture_date, length(content) AS content_len,
              publication_date, author_attribution, scope
       FROM dossiers ORDER BY capture_date DESC LIMIT ?`,
    )
    .all(limit);
}

export function getDossierEmbedding(slug: string): Float32Array | null {
  const row = getDb()
    .prepare("SELECT embedding FROM dossiers WHERE slug = ?")
    .get(slug) as { embedding: Buffer | null } | undefined;
  if (!row?.embedding) return null;
  return blobToEmb(row.embedding);
}

// ---------------------------------------------------------------------------
// Claim ops
// ---------------------------------------------------------------------------

export interface RecordClaimInput {
  dossier_slug: string;
  claim_text: string;
  score: number | null;
  status: string;
  source_passage?: string | null;
  claim_type?: ClaimType;
  topic?: string | null;
  author?: string | null;
  attribution?: string | null;
  soft_score?: number | null;
  depends_on_ids?: number[];
  dependency_type?: DependencyType;
  source_offset_start?: number | null;
  source_offset_end?: number | null;
  embedding?: number[] | Float32Array | null;
  verification?: string | null;
}

export function recordClaim(input: RecordClaimInput): number {
  const db = getDb();
  const now = new Date().toISOString();
  const embBlob = input.embedding ? embToBlob(input.embedding) : null;

  const result = db
    .prepare(
      `INSERT INTO claims
       (dossier_slug, claim_text, source_passage, nli_score, status, verified_at,
        claim_type, topic, author, soft_score, attribution,
        source_offset_start, source_offset_end, embedding, verification)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    )
    .run(
      input.dossier_slug,
      input.claim_text,
      input.source_passage ?? null,
      input.score,
      input.status,
      now,
      input.claim_type ?? "ATOMIC",
      input.topic ?? null,
      input.author ?? null,
      input.soft_score ?? null,
      input.attribution ?? null,
      input.source_offset_start ?? null,
      input.source_offset_end ?? null,
      embBlob,
      input.verification ?? null,
    );
  const cid = Number(result.lastInsertRowid);

  if (input.depends_on_ids?.length) {
    const stmt = db.prepare(
      "INSERT OR IGNORE INTO claim_dependencies (claim_id, depends_on_id, dependency_type) VALUES (?, ?, ?)",
    );
    const depType = input.dependency_type || "inference";
    for (const dep of input.depends_on_ids) {
      stmt.run(cid, dep, depType);
    }
  }
  return cid;
}

const CLAIM_USER_COLS =
  "id, dossier_slug, claim_text, source_passage, nli_score, status, verified_at, " +
  "claim_type, topic, author, soft_score, attribution, superseded_by, supersede_reason, " +
  "source_offset_start, source_offset_end, verification";

export function getClaim(id: number): (Claim & { depends_on: ClaimDependency[] }) | null {
  const db = getDb();
  const row = db.prepare(`SELECT ${CLAIM_USER_COLS} FROM claims WHERE id = ?`).get(id) as any;
  if (!row) return null;
  const deps = db
    .prepare("SELECT depends_on_id, dependency_type FROM claim_dependencies WHERE claim_id = ?")
    .all(id) as ClaimDependency[];
  row.depends_on = deps;
  return row;
}

/** Claims (any status) with this exact text + type, each with its dependency
 *  edges. Used by the gate's tagged-derived-claim harvest to dedup — re-emitting
 *  a draft after a block must not double-file the same derived claim. */
export function findClaimsByTextType(
  claimText: string,
  claimType: ClaimType,
): { id: number; status: string; superseded_by: number | null; depends_on: ClaimDependency[] }[] {
  const db = getDb();
  const rows = db
    .prepare("SELECT id, status, superseded_by FROM claims WHERE claim_text = ? AND claim_type = ?")
    .all(claimText, claimType) as { id: number; status: string; superseded_by: number | null }[];
  const depStmt = db.prepare(
    "SELECT depends_on_id, dependency_type FROM claim_dependencies WHERE claim_id = ?",
  );
  return rows.map((r) => ({
    id: r.id,
    status: r.status,
    superseded_by: r.superseded_by,
    depends_on: depStmt.all(r.id) as ClaimDependency[],
  }));
}

export interface KbDelta {
  atomicSupported: number;
  derivedHarvested: number;
  dossiersSnapshotted: number;
}

export function getKbDelta(turnAnchor: string): KbDelta {
  const db = getDb();
  const atomicSupported = db
    .prepare(
      `SELECT COUNT(*) AS n FROM claims
       WHERE verified_at >= ? AND claim_type = 'ATOMIC' AND status = 'supported' AND verification = 'nli-session'`,
    )
    .get(turnAnchor) as { n: number };
  const derivedHarvested = db
    .prepare(`SELECT COUNT(*) AS n FROM claims WHERE verified_at >= ? AND author = 'gate-harvest'`)
    .get(turnAnchor) as { n: number };
  const dossiersSnapshotted = db
    .prepare(
      `SELECT COUNT(*) AS n FROM dossiers WHERE capture_date >= ? AND source_type LIKE 'session-%'`,
    )
    .get(turnAnchor) as { n: number };
  return {
    atomicSupported: atomicSupported.n,
    derivedHarvested: derivedHarvested.n,
    dossiersSnapshotted: dossiersSnapshotted.n,
  };
}

export interface ListClaimsOpts {
  topic?: string;
  status?: string;
  dossier_slug?: string;
  claim_type?: string;
  contains?: string;
  limit?: number;
  author?: string;
  verification?: string;
  depends_on_id?: number;
  since?: string;
  newestFirst?: boolean;
}

export function listClaims(opts: ListClaimsOpts = {}): Claim[] {
  const where: string[] = [];
  const params: any[] = [];
  if (opts.topic) {
    where.push("topic = ?");
    params.push(opts.topic);
  }
  if (opts.status) {
    where.push("status = ?");
    params.push(opts.status);
  }
  if (opts.dossier_slug) {
    where.push("dossier_slug = ?");
    params.push(opts.dossier_slug);
  }
  if (opts.claim_type) {
    where.push("claim_type = ?");
    params.push(opts.claim_type);
  }
  if (opts.contains) {
    where.push("claim_text LIKE ?");
    params.push(`%${opts.contains}%`);
  }
  if (opts.author) {
    where.push("author = ?");
    params.push(opts.author);
  }
  if (opts.verification) {
    where.push("verification = ?");
    params.push(opts.verification);
  }
  if (opts.since) {
    where.push("verified_at >= ?");
    params.push(opts.since);
  }
  if (opts.depends_on_id != null) {
    where.push(
      "EXISTS (SELECT 1 FROM claim_dependencies cd WHERE cd.claim_id = claims.id AND cd.depends_on_id = ?)",
    );
    params.push(opts.depends_on_id);
  }
  let sql =
    "SELECT id, dossier_slug, claim_text, source_passage, nli_score, status, verified_at, claim_type, topic, author, soft_score, attribution, superseded_by, supersede_reason, source_offset_start, source_offset_end, verification FROM claims";
  if (where.length) sql += " WHERE " + where.join(" AND ");
  sql += opts.newestFirst ? " ORDER BY verified_at DESC, id DESC LIMIT ?" : " ORDER BY id DESC LIMIT ?";
  params.push(opts.limit ?? 50);
  return getDb().prepare(sql).all(...params) as Claim[];
}

export interface RecentSummary {
  total: number;
  supported: number;
  unsupported: number;
  recorded: number;
  dossiers: number;
  authorBreakdown: Record<string, number>;
}

// ---------------------------------------------------------------------------
// Digest — content-bearing KB mutation summary
// ---------------------------------------------------------------------------

const CLAIM_TYPE_TO_KIND: Record<string, string> = {
  INFERENCE: "inference-from",
  SYNTHESIS: "synthesis-of",
  INTERPRETATION: "interpretation",
  HYPOTHESIS: "hypothesis",
};

export interface DigestEntry {
  since: string;
  claims: {
    id: number;
    claim_type: string | null;
    claim_text: string;
    nli_score: number | null;
    status: string;
    dossier_slug: string | null;
    source_passage: string | null;
    dossier_source_url: string | null;
    verified_at: string;
    author: string | null;
    verification: string | null;
  }[];
  derived_claims: {
    id: number;
    kind: string;
    claim_text: string;
    upstream_ids: number[];
    soft_score: number | null;
    verified_at: string;
  }[];
  supersedes: {
    old_id: number;
    new_id: number;
    reason: string | null;
  }[];
  dossiers: {
    slug: string;
    source_url: string;
    source_type: string;
    capture_date: string;
    scope: string | null;
  }[];
  dependencies: {
    claim_id: number;
    depends_on_ids: number[];
  }[];
}

export function getDigest(since: string): DigestEntry {
  const db = getDb();

  const claimRows = db
    .prepare(
      `SELECT c.id, c.claim_type, c.claim_text, c.nli_score, c.status,
              c.dossier_slug, c.source_passage, c.verified_at, c.author, c.verification, c.soft_score,
              d.source_url as dossier_source_url
       FROM claims c
       LEFT JOIN dossiers d ON c.dossier_slug = d.slug
       WHERE c.verified_at >= ?
       ORDER BY c.verified_at DESC`,
    )
    .all(since) as any[];

  const claims: DigestEntry["claims"] = [];
  const derived_claims: DigestEntry["derived_claims"] = [];

  for (const c of claimRows) {
    if (c.verification === "tag-harvest") {
      const deps = db
        .prepare("SELECT depends_on_id FROM claim_dependencies WHERE claim_id = ? ORDER BY depends_on_id")
        .all(c.id) as { depends_on_id: number }[];
      derived_claims.push({
        id: c.id,
        kind: CLAIM_TYPE_TO_KIND[c.claim_type] || c.claim_type || "UNKNOWN",
        claim_text: c.claim_text,
        upstream_ids: deps.map((d) => d.depends_on_id),
        soft_score: c.soft_score,
        verified_at: c.verified_at,
      });
    } else {
      claims.push({
        id: c.id,
        claim_type: c.claim_type,
        claim_text: c.claim_text,
        nli_score: c.nli_score,
        status: c.status,
        dossier_slug: c.dossier_slug,
        source_passage: c.source_passage,
        dossier_source_url: c.dossier_source_url,
        verified_at: c.verified_at,
        author: c.author,
        verification: c.verification,
      });
    }
  }

  const supersedes = db
    .prepare(
      `SELECT old.id as old_id, old.superseded_by as new_id, old.supersede_reason as reason
       FROM claims old
       JOIN claims new ON old.superseded_by = new.id
       WHERE new.verified_at >= ? AND old.superseded_by IS NOT NULL`,
    )
    .all(since) as any[];

  const dossiers = db
    .prepare(
      `SELECT slug, source_url, source_type, capture_date, scope
       FROM dossiers
       WHERE capture_date >= ?
       ORDER BY capture_date DESC`,
    )
    .all(since) as any[];

  const depRows = db
    .prepare(
      `SELECT cd.claim_id, cd.depends_on_id
       FROM claim_dependencies cd
       JOIN claims c ON cd.claim_id = c.id
       WHERE c.verified_at >= ?
       ORDER BY cd.claim_id, cd.depends_on_id`,
    )
    .all(since) as { claim_id: number; depends_on_id: number }[];

  const depMap = new Map<number, number[]>();
  for (const d of depRows) {
    if (!depMap.has(d.claim_id)) depMap.set(d.claim_id, []);
    depMap.get(d.claim_id)!.push(d.depends_on_id);
  }
  const dependencies = [...depMap.entries()].map(([claim_id, depends_on_ids]) => ({
    claim_id,
    depends_on_ids,
  }));

  return { since, claims, derived_claims, supersedes, dossiers, dependencies };
}

export function getSessionStart(gapHours = 2): string {
  const db = getDb();
  const gapMs = gapHours * 3600 * 1000;

  const claimRows = db
    .prepare(
      "SELECT verified_at FROM claims WHERE verified_at IS NOT NULL ORDER BY verified_at DESC LIMIT 200",
    )
    .all() as { verified_at: string }[];

  const dossierRows = db
    .prepare(
      "SELECT capture_date AS verified_at FROM dossiers WHERE capture_date IS NOT NULL ORDER BY capture_date DESC LIMIT 200",
    )
    .all() as { verified_at: string }[];

  const allTimestamps = [...claimRows, ...dossierRows]
    .map((r) => new Date(r.verified_at).getTime())
    .filter((t) => !isNaN(t))
    .sort((a, b) => b - a);

  if (allTimestamps.length === 0) {
    return new Date(Date.now() - 24 * 3600 * 1000).toISOString();
  }

  for (let i = 0; i < allTimestamps.length - 1; i++) {
    const gap = allTimestamps[i]! - allTimestamps[i + 1]!;
    if (gap > gapMs) {
      return new Date(allTimestamps[i]!).toISOString();
    }
  }

  if (allTimestamps.length < 3) {
    return new Date(Date.now() - 24 * 3600 * 1000).toISOString();
  }

  return new Date(allTimestamps[allTimestamps.length - 1]!).toISOString();
}

export function formatDigestMarkdown(digest: DigestEntry): string {
  const lines: string[] = [];
  const date = new Date().toISOString().slice(0, 10);
  lines.push(`# KB Digest — ${date}`);
  lines.push("");
  lines.push(`*Window: ${digest.since} → now*`);
  lines.push("");

  const isEmpty =
    digest.claims.length === 0 &&
    digest.derived_claims.length === 0 &&
    digest.supersedes.length === 0 &&
    digest.dossiers.length === 0 &&
    digest.dependencies.length === 0;

  if (isEmpty) {
    lines.push("(nothing entered the KB in this window)");
    lines.push("");
    return lines.join("\n");
  }

  if (digest.claims.length) {
    lines.push(`## New verified claims (${digest.claims.length})`);
    lines.push("");
    for (const c of digest.claims) {
      const quote = c.source_passage
        ? ` · "${c.source_passage.slice(0, 160)}${c.source_passage.length > 160 ? "…" : ""}"`
        : "";
      const dossier = c.dossier_slug ? ` · Dossier: \`${c.dossier_slug}\`` : "";
      lines.push(
        `- **#${c.id}** · ${c.claim_type || "UNKNOWN"} · score ${c.nli_score ?? "—"} · "${c.claim_text.slice(0, 200)}${c.claim_text.length > 200 ? "…" : ""}"${dossier}${quote}`,
      );
    }
    lines.push("");
  }

  if (digest.derived_claims.length) {
    lines.push(`## Harvested derived claims (${digest.derived_claims.length})`);
    lines.push("");
    for (const d of digest.derived_claims) {
      const upstream = d.upstream_ids.length ? ` · Upstream: ${d.upstream_ids.map((id) => `#${id}`).join(", ")}` : "";
      lines.push(
        `- **#${d.id}** · ${d.kind} · "${d.claim_text.slice(0, 200)}${d.claim_text.length > 200 ? "…" : ""}"${upstream}`,
      );
    }
    lines.push("");
  }

  if (digest.supersedes.length) {
    lines.push(`## Supersedes (${digest.supersedes.length})`);
    lines.push("");
    for (const s of digest.supersedes) {
      lines.push(`- #${s.old_id} → #${s.new_id} · ${s.reason || "no reason given"}`);
    }
    lines.push("");
  }

  if (digest.dossiers.length) {
    lines.push(`## New dossiers (${digest.dossiers.length})`);
    lines.push("");
    for (const d of digest.dossiers) {
      lines.push(`- \`${d.slug}\` · ${d.source_url} · ${d.capture_date.slice(0, 10)} · ${d.source_type}`);
    }
    lines.push("");
  }

  if (digest.dependencies.length) {
    lines.push(`## New dependency edges (${digest.dependencies.length})`);
    lines.push("");
    for (const dep of digest.dependencies) {
      lines.push(`- #${dep.claim_id} → depends on ${dep.depends_on_ids.map((id) => `#${id}`).join(", ")}`);
    }
    lines.push("");
  }

  return lines.join("\n");
}

export function listRecentClaimsSummary(since: string): RecentSummary {
  const db = getDb();
  const statusRow = db
    .prepare(
      `SELECT
        COUNT(*) AS total,
        SUM(CASE WHEN status = 'supported' THEN 1 ELSE 0 END) AS supported,
        SUM(CASE WHEN status = 'unsupported' THEN 1 ELSE 0 END) AS unsupported,
        SUM(CASE WHEN status = 'recorded' THEN 1 ELSE 0 END) AS recorded
       FROM claims WHERE verified_at >= ?`,
    )
    .get(since) as { total: number; supported: number; unsupported: number; recorded: number };
  const dossierRow = db
    .prepare(
      `SELECT COUNT(DISTINCT dossier_slug) AS n FROM claims WHERE verified_at >= ? AND dossier_slug IS NOT NULL AND dossier_slug != ''`,
    )
    .get(since) as { n: number };
  const authorRows = db
    .prepare(
      `SELECT author, COUNT(*) AS n FROM claims WHERE verified_at >= ? GROUP BY author`,
    )
    .all(since) as { author: string | null; n: number }[];
  const authorBreakdown: Record<string, number> = {};
  for (const row of authorRows) {
    authorBreakdown[row.author ?? "(none)"] = row.n;
  }
  return {
    total: statusRow.total,
    supported: statusRow.supported,
    unsupported: statusRow.unsupported,
    recorded: statusRow.recorded,
    dossiers: dossierRow.n,
    authorBreakdown,
  };
}

export function listTopics(): { topic: string; n_claims: number; n_supported: number }[] {
  return getDb()
    .prepare(
      `SELECT topic, COUNT(*) AS n_claims,
              SUM(CASE WHEN status='supported' THEN 1 ELSE 0 END) AS n_supported
       FROM claims WHERE topic IS NOT NULL
       GROUP BY topic ORDER BY MAX(id) DESC`,
    )
    .all() as any;
}

export function getClaimChain(claimId: number, maxDepth = 6): {
  root: number;
  nodes: Record<number, Claim>;
  edges: { from: number; to: number; type: string }[];
  node_count: number;
} {
  const db = getDb();
  const nodes: Record<number, Claim> = {};
  const edges: { from: number; to: number; type: string }[] = [];
  const visited = new Set<number>();
  const stack: { id: number; depth: number }[] = [{ id: claimId, depth: 0 }];

  while (stack.length) {
    const { id, depth } = stack.pop()!;
    if (visited.has(id) || depth > maxDepth) continue;
    visited.add(id);
    const row = db.prepare(`SELECT ${CLAIM_USER_COLS} FROM claims WHERE id = ?`).get(id) as any;
    if (!row) continue;
    nodes[id] = row;
    const deps = db
      .prepare(
        "SELECT depends_on_id, dependency_type FROM claim_dependencies WHERE claim_id = ?",
      )
      .all(id) as { depends_on_id: number; dependency_type: string }[];
    for (const d of deps) {
      edges.push({ from: id, to: d.depends_on_id, type: d.dependency_type });
      stack.push({ id: d.depends_on_id, depth: depth + 1 });
    }
  }
  return { root: claimId, nodes, edges, node_count: Object.keys(nodes).length };
}

export function supersedeClaim(oldId: number, newId: number, reason: string): boolean {
  const db = getDb();
  const oldRow = db.prepare("SELECT 1 FROM claims WHERE id = ?").get(oldId);
  const newRow = db.prepare("SELECT 1 FROM claims WHERE id = ?").get(newId);
  if (!oldRow || !newRow) return false;
  db.prepare("UPDATE claims SET superseded_by = ?, supersede_reason = ? WHERE id = ?").run(
    newId,
    reason,
    oldId,
  );
  return true;
}

// ---------------------------------------------------------------------------
// Hybrid search — cosine over claims.embedding ∪ dossiers.embedding
// ---------------------------------------------------------------------------

function cosine(a: Float32Array, b: Float32Array): number {
  const n = Math.min(a.length, b.length);
  let dot = 0;
  for (let i = 0; i < n; i++) dot += a[i]! * b[i]!;
  return dot;
}

export interface SearchHit {
  kind: "claim" | "dossier";
  similarity: number;
  id?: number;
  slug?: string;
  text: string;
  title?: string | null;
  status?: string;
  topic?: string | null;
  source_url?: string;
}

export function searchHybrid(
  queryEmbedding: Float32Array,
  topK = 10,
): SearchHit[] {
  const db = getDb();
  const hits: SearchHit[] = [];

  for (const row of db
    .prepare(
      "SELECT id, claim_text, status, topic, embedding FROM claims WHERE embedding IS NOT NULL",
    )
    .iterate() as Iterable<any>) {
    const emb = blobToEmb(row.embedding);
    if (!emb) continue;
    hits.push({
      kind: "claim",
      similarity: cosine(emb, queryEmbedding),
      id: row.id,
      text: row.claim_text,
      status: row.status,
      topic: row.topic,
    });
  }
  for (const row of db
    .prepare(
      "SELECT slug, title, source_url, content, embedding FROM dossiers WHERE embedding IS NOT NULL",
    )
    .iterate() as Iterable<any>) {
    const emb = blobToEmb(row.embedding);
    if (!emb) continue;
    hits.push({
      kind: "dossier",
      similarity: cosine(emb, queryEmbedding),
      slug: row.slug,
      text: (row.content || "").slice(0, 400),
      title: row.title,
      source_url: row.source_url,
    });
  }

  hits.sort((a, b) => b.similarity - a.similarity);
  return hits.slice(0, topK);
}

// ---------------------------------------------------------------------------
// Session claims ledger (issue #43)
// ---------------------------------------------------------------------------

export interface SessionClaimRow {
  transcript_id: string;
  turn_idx: number;
  claim_idx: number;
  proposition: string;
  entity: string;
  stance: string;
  /** grounded | ungrounded | reclassified | escalated | contradicted */
  verdict: string;
  reason: string | null;
  ts: string;
  superseded_by_turn: number | null;
  superseded_by_claim: number | null;
  /** 0 | 1 */
  retracted: number;
  embedding?: Float32Array | null;
}

export interface RecordSessionClaimInput {
  transcript_id: string;
  turn_idx: number;
  claim_idx: number;
  proposition: string;
  entity: string;
  stance: string;
  verdict: string;
  reason?: string | null;
  embedding?: Float32Array | null;
  /** #50 (A) Stage 1: set to 1 when this is a fire (verdict='ungrounded'
   *  on stance='ASSERT'). Cleared in subsequent turns by detection logic
   *  (Stage 2 / 3) when the revise satisfies one of: tool-call referencing
   *  entity / hedge-tag near entity / demonstrable removal. */
  awaiting_revise?: 0 | 1;
}

/** Return the next unused turn_idx for this transcript (0 if no rows yet). */
export function getNextSessionTurnIdx(transcript_id: string): number {
  const row = getDb()
    .prepare("SELECT MAX(turn_idx) AS m FROM session_claims WHERE transcript_id = ?")
    .get(transcript_id) as { m: number | null } | undefined;
  return (row?.m ?? -1) + 1;
}

/** Per-session verdict counts on session_claims — for surfacing in the gate's
 *  per-turn message so the agent (and the user) can see whether fires are
 *  accumulating without being resolved. Counts are over all turns in this
 *  transcript's session_claims history; the gate appends this on every Stop
 *  hook so trend is visible turn-by-turn.
 *
 *  - `total`        = every claim the gate has processed in this session
 *  - `ungrounded`   = fires (verdict='ungrounded'); the dodge-attractor surface
 *  - `grounded`     = passed grounding directly OR via session-source autoground
 *  - `reclassified` = postfilter caught it as workspace-meta etc; no fire
 *  - `retracted`    = explicitly retracted in a later turn
 */
export function getSessionFireCounts(transcript_id: string): {
  total: number;
  ungrounded: number;
  grounded: number;
  reclassified: number;
  retracted: number;
  awaiting_revise: number;
  /** P-γ stance breakdown: counts of confident-vs-uncertain stances over
   *  the full session. ASSERT is the confident truth-bearing stance; HEDGE
   *  and SPECULATE are the explicit-uncertainty truth-bearing stances. The
   *  humility ratio (HEDGE+SPECULATE) / (ASSERT+HEDGE+SPECULATE) is the
   *  starting humility signal made visible per Stop hook. */
  asserts: number;
  hedges: number;
  speculates: number;
} {
  const row = getDb()
    .prepare(
      `SELECT
         COUNT(*) AS total,
         SUM(CASE WHEN verdict = 'ungrounded' THEN 1 ELSE 0 END) AS ungrounded,
         SUM(CASE WHEN verdict = 'grounded' THEN 1 ELSE 0 END) AS grounded,
         SUM(CASE WHEN verdict = 'reclassified' THEN 1 ELSE 0 END) AS reclassified,
         SUM(CASE WHEN retracted = 1 THEN 1 ELSE 0 END) AS retracted,
         SUM(CASE WHEN awaiting_revise = 1 THEN 1 ELSE 0 END) AS awaiting_revise,
         SUM(CASE WHEN stance = 'ASSERT' THEN 1 ELSE 0 END) AS asserts,
         SUM(CASE WHEN stance = 'HEDGE' THEN 1 ELSE 0 END) AS hedges,
         SUM(CASE WHEN stance = 'SPECULATE' THEN 1 ELSE 0 END) AS speculates
       FROM session_claims
       WHERE transcript_id = ?`,
    )
    .get(transcript_id) as
    | {
        total: number;
        ungrounded: number;
        grounded: number;
        reclassified: number;
        retracted: number;
        awaiting_revise: number;
        asserts: number;
        hedges: number;
        speculates: number;
      }
    | undefined;
  return {
    total: row?.total ?? 0,
    ungrounded: row?.ungrounded ?? 0,
    grounded: row?.grounded ?? 0,
    reclassified: row?.reclassified ?? 0,
    retracted: row?.retracted ?? 0,
    awaiting_revise: row?.awaiting_revise ?? 0,
    asserts: row?.asserts ?? 0,
    hedges: row?.hedges ?? 0,
    speculates: row?.speculates ?? 0,
  };
}

export function recordSessionClaim(input: RecordSessionClaimInput): void {
  const ts = new Date().toISOString();
  const embBlob = input.embedding ? embToBlob(input.embedding) : null;
  // #50 (A) Stage 1: when caller doesn't pass awaiting_revise, derive it —
  // a fire (verdict='ungrounded' + stance='ASSERT') needs revise tracking.
  // Other shapes (RETRACT, REFER, WORKSPACE, etc.) don't.
  const awaiting =
    input.awaiting_revise ??
    (input.verdict === "ungrounded" && input.stance === "ASSERT" ? 1 : 0);
  getDb()
    .prepare(
      `INSERT OR REPLACE INTO session_claims
       (transcript_id, turn_idx, claim_idx, proposition, entity, stance, verdict, reason,
        embedding, ts, superseded_by_turn, superseded_by_claim, retracted, awaiting_revise)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, 0, ?)`,
    )
    .run(
      input.transcript_id,
      input.turn_idx,
      input.claim_idx,
      input.proposition,
      input.entity,
      input.stance,
      input.verdict,
      input.reason ?? null,
      embBlob,
      ts,
      awaiting,
    );
}

/** List session_claims rows that are flagged awaiting_revise=1 (= fires
 *  from prior turns that haven't been classified as fetched / hedged /
 *  removed yet). Sorted oldest-first so the agent can see what's been
 *  outstanding longest. #50 (A) Stage 2 consumes this. */
export function listAwaitingReviseClaims(transcript_id: string): SessionClaimRow[] {
  return (
    getDb()
      .prepare(
        `SELECT * FROM session_claims
         WHERE transcript_id = ? AND awaiting_revise = 1 AND retracted = 0
         ORDER BY turn_idx ASC, claim_idx ASC`,
      )
      .all(transcript_id) as any[]
  ).map((r) => {
    r.embedding = blobToEmb(r.embedding);
    return r as SessionClaimRow;
  });
}

/** Mark a session_claims row as addressed by a subsequent turn's revise.
 *  via ∈ {'fetch','hedge','remove'} — the classification of how the agent
 *  resolved the prior fire. Clears awaiting_revise to 0 so it stops
 *  appearing in the backlog. #50 (A) Stage 2. */
export function markAddressedAwaiting(
  transcript_id: string,
  turn_idx: number,
  claim_idx: number,
  via: "fetch" | "hedge" | "remove",
  addressed_in_turn: number,
): void {
  getDb()
    .prepare(
      `UPDATE session_claims
         SET awaiting_revise = 0,
             addressed_via = ?,
             addressed_in_turn = ?
       WHERE transcript_id = ? AND turn_idx = ? AND claim_idx = ?`,
    )
    .run(via, addressed_in_turn, transcript_id, turn_idx, claim_idx);
}

export function getSessionClaim(
  transcript_id: string,
  turn_idx: number,
  claim_idx: number,
): SessionClaimRow | null {
  const row = getDb()
    .prepare(
      `SELECT * FROM session_claims
       WHERE transcript_id = ? AND turn_idx = ? AND claim_idx = ?`,
    )
    .get(transcript_id, turn_idx, claim_idx) as any;
  if (!row) return null;
  row.embedding = blobToEmb(row.embedding);
  return row as SessionClaimRow;
}

export function listSessionClaims(
  transcript_id: string,
  opts: { include_retracted?: boolean; only_active?: boolean } = {},
): SessionClaimRow[] {
  const db = getDb();
  let sql = `SELECT * FROM session_claims WHERE transcript_id = ?`;
  if (opts.only_active) {
    sql += ` AND retracted = 0 AND superseded_by_turn IS NULL`;
  } else if (!opts.include_retracted) {
    sql += ` AND retracted = 0`;
  }
  sql += ` ORDER BY turn_idx ASC, claim_idx ASC`;
  return (db.prepare(sql).all(transcript_id) as any[]).map((r) => {
    r.embedding = blobToEmb(r.embedding);
    return r as SessionClaimRow;
  });
}

/** Find prior session claims (same transcript, not retracted, not superseded,
 *  verdict in {grounded, escalated}) whose embedding is cosine-close to
 *  queryEmb. Returns top-K above threshold, sorted desc by similarity.
 *  Excludes WORKSPACE / REFER / reclassified rows — those aren't assertions
 *  the agent is making, so they can't be contradicted. */
export function findSessionContradictionCandidates(
  transcript_id: string,
  queryEmb: Float32Array,
  opts: { topK?: number; minCos?: number; excludeTurn?: number } = {},
): Array<{ row: SessionClaimRow; similarity: number }> {
  const topK = opts.topK ?? 10;
  const minCos = opts.minCos ?? 0.6;
  const rows = listSessionClaims(transcript_id, { only_active: true });
  const hits: Array<{ row: SessionClaimRow; similarity: number }> = [];
  for (const r of rows) {
    if (!r.embedding) continue;
    if (opts.excludeTurn !== undefined && r.turn_idx === opts.excludeTurn) continue;
    if (r.stance === "WORKSPACE" || r.stance === "REFER" || r.verdict === "reclassified") continue;
    const s = cosine(r.embedding, queryEmb);
    if (s >= minCos) hits.push({ row: r, similarity: s });
  }
  hits.sort((a, b) => b.similarity - a.similarity);
  return hits.slice(0, topK);
}

/** Soft-delete a prior session claim. Reason is appended to the row's reason
 *  field so the audit log retains the original verdict reason. */
export function markSessionClaimRetracted(
  transcript_id: string,
  turn_idx: number,
  claim_idx: number,
  reason: string,
): boolean {
  const db = getDb();
  const res = db
    .prepare(
      `UPDATE session_claims
       SET retracted = 1,
           reason = COALESCE(reason, '') || (CASE WHEN COALESCE(reason, '') = '' THEN '' ELSE ' | ' END) || 'retracted: ' || ?
       WHERE transcript_id = ? AND turn_idx = ? AND claim_idx = ?`,
    )
    .run(reason, transcript_id, turn_idx, claim_idx);
  return res.changes > 0;
}

/** Mark an old session claim as superseded by a new one in the same session. */
export function markSessionClaimSuperseded(
  transcript_id: string,
  old_turn: number,
  old_claim: number,
  new_turn: number,
  new_claim: number,
): boolean {
  const db = getDb();
  const res = db
    .prepare(
      `UPDATE session_claims
       SET superseded_by_turn = ?, superseded_by_claim = ?
       WHERE transcript_id = ? AND turn_idx = ? AND claim_idx = ?`,
    )
    .run(new_turn, new_claim, transcript_id, old_turn, old_claim);
  return res.changes > 0;
}
