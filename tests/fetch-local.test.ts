/** Local-file fetch tests — mocks LLM calls, uses real SQLite store. */
import { afterEach, beforeAll, describe, expect, it, mock } from "bun:test";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const tmp = mkdtempSync(join(tmpdir(), "vouch-test-"));
process.env.VOUCH_DB_PATH = join(tmp, "test.db");

mock.module("ai", () => ({
  embed: () => Promise.resolve({ embedding: [1, 0, 0, 0] }),
}));

mock.module("../src/providers.ts", () => ({
  getLanguageModel: () => ({ id: "test-model" }) as any,
  getEmbeddingModel: () => ({ id: "test-embedder" }) as any,
}));

mock.module("../src/embedder.ts", () => ({
  embedOne: () => Promise.resolve(new Float32Array([1, 0, 0, 0])),
  embedBatch: () => Promise.resolve([new Float32Array([1, 0, 0, 0])]),
}));

const store = await import("../src/store.ts");
const { fetchAndStore } = await import("../src/fetch.ts");

afterEach(() => {
  const db = store.getDb();
  db.exec("DELETE FROM claim_dependencies; DELETE FROM claims; DELETE FROM dossiers;");
});

beforeAll(() => {
  process.on("exit", () => {
    try {
      rmSync(tmp, { recursive: true, force: true });
    } catch {}
  });
});

describe("fetch local file", () => {
  it("fetches file:// absolute path and persists dossier", async () => {
    const path = join(tmp, "test1.txt");
    writeFileSync(path, "Hello from local file.");

    const result = await fetchAndStore(`file://${path}`);
    expect(result.cached).toBe(false);
    expect(result.source_type).toBe("local-file");
    expect(result.source_url).toBe(`file://${path}`);
    expect(result.content).toContain("Hello from local file");
    expect(result.content_chars).toBe(22);

    const d = store.getDossier(result.dossier_slug);
    expect(d).not.toBeNull();
    expect(d!.content).toBe("Hello from local file.");
    expect(d!.source_hash).toBe(store.sha256Hex("Hello from local file."));
    expect(d!.title).toBe("test1.txt");
  });

  it("fetches relative path and persists dossier", async () => {
    const path = join(tmp, "test2.txt");
    writeFileSync(path, "Relative path content.");

    // Use relative path from cwd
    const rel = "./tests/fetch-local.test.ts"; // a file we know exists
    const result = await fetchAndStore(rel);
    expect(result.cached).toBe(false);
    expect(result.source_type).toBe("local-file");
    expect(result.source_url).toBe(`file://${join(process.cwd(), rel)}`);
  });

  it("caches re-fetch within 24h", async () => {
    const path = join(tmp, "test3.txt");
    writeFileSync(path, "Cache me.");

    const r1 = await fetchAndStore(`file://${path}`);
    expect(r1.cached).toBe(false);

    const r2 = await fetchAndStore(`file://${path}`);
    expect(r2.cached).toBe(true);
    expect(r2.dossier_slug).toBe(r1.dossier_slug);
  });

  it("force-refetch updates hash when content changes", async () => {
    const path = join(tmp, "test4.txt");
    writeFileSync(path, "Original content.");

    const r1 = await fetchAndStore(`file://${path}`);
    const d1 = store.getDossier(r1.dossier_slug)!;
    const hash1 = d1.source_hash;

    writeFileSync(path, "Changed content.");
    const r2 = await fetchAndStore(`file://${path}`, { forceRefetch: true });
    expect(r2.cached).toBe(false);

    const d2 = store.getDossier(r2.dossier_slug)!;
    expect(d2.source_hash).not.toBe(hash1);
    expect(d2.content).toBe("Changed content.");
    expect(d2.last_refetched).not.toBe(d1.last_refetched);
  });

  it("rejects binary files with clear error", async () => {
    const path = join(tmp, "test.bin");
    writeFileSync(path, Buffer.from([0x00, 0x01, 0x02, 0xff]));

    await expect(fetchAndStore(`file://${path}`)).rejects.toThrow(
      /local-file fetch supports text files/,
    );
  });

  it("rejects non-existent file:// path", async () => {
    await expect(fetchAndStore("file:///nonexistent/path/foo.txt")).rejects.toThrow(
      /local file not found/,
    );
  });

  it("falls through to HTTP for non-existent bare paths", async () => {
    // A bare path that does not exist should be treated as a URL and fail
    // at the HTTP layer (or fetcher routing), not as a local file.
    await expect(fetchAndStore("./definitely-does-not-exist-12345.txt")).rejects.toThrow();
  });
});
