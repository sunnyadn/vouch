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
