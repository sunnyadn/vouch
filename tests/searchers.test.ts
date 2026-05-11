/** Citation-search adapter tests — pure parsing + normalization, no subprocess.
 *
 * `searchCitations` shells out to opencli; the spawn is a thin wrapper around
 * `parseSearchPayload`, which is what these tests exercise (stubbed stdout in
 * each provider's real JSON shape — no network).
 */
import { describe, expect, it } from "bun:test";
import {
  parseSearchPayload,
  searchCitations,
  isSearchProvider,
  titleSimilarity,
  SEARCH_PROVIDERS,
} from "../src/searchers.ts";

describe("isSearchProvider", () => {
  it("accepts the four supported providers", () => {
    for (const p of ["pubmed", "arxiv", "openalex", "google-scholar"]) {
      expect(isSearchProvider(p)).toBe(true);
    }
  });
  it("rejects unknown providers", () => {
    expect(isSearchProvider("semanticscholar")).toBe(false);
    expect(isSearchProvider("crossref")).toBe(false);
    expect(isSearchProvider("")).toBe(false);
  });
  it("SEARCH_PROVIDERS is the public list", () => {
    expect(SEARCH_PROVIDERS).toEqual(["pubmed", "arxiv", "openalex", "google-scholar"]);
  });
});

describe("parseSearchPayload — provider shapes", () => {
  it("openalex array → normalized candidates", () => {
    const stdout = JSON.stringify([
      {
        rank: 1,
        id: "W4389520670",
        title: "Enabling Large Language Models to Generate Text with Citations",
        year: 2023,
        citations: 312,
        firstAuthor: "Tianyu Gao",
        venue: "",
        doi: "10.18653/v1/2023.emnlp-main.398",
        url: "https://openalex.org/W4389520670",
      },
    ]);
    const out = parseSearchPayload("openalex", stdout, "", 0);
    expect(out).toHaveLength(1);
    expect(out[0]).toEqual({
      provider: "openalex",
      rank: 1,
      title: "Enabling Large Language Models to Generate Text with Citations",
      url: "https://openalex.org/W4389520670",
      year: 2023,
      authors: "Tianyu Gao",
      doi: "10.18653/v1/2023.emnlp-main.398",
      venue: "",
    });
  });

  it("pubmed array → maps pmid/journal field names", () => {
    const stdout = JSON.stringify([
      {
        rank: 1,
        pmid: "10204198",
        title: "A proportional hazards model for the subdistribution of a competing risk",
        authors: "Fine JP, Gray RJ.",
        journal: "Journal of the American Statistical Association",
        year: "1999",
        doi: "10.1080/01621459.1999.10474144",
        url: "https://pubmed.ncbi.nlm.nih.gov/10204198/",
      },
    ]);
    const out = parseSearchPayload("pubmed", stdout, "", 0);
    expect(out).toHaveLength(1);
    expect(out[0]!.authors).toBe("Fine JP, Gray RJ.");
    expect(out[0]!.venue).toBe("Journal of the American Statistical Association");
    expect(out[0]!.url).toBe("https://pubmed.ncbi.nlm.nih.gov/10204198/");
  });

  it("arxiv array → maps published→year", () => {
    const stdout = JSON.stringify([
      {
        id: "2305.14627",
        title: "Enabling Large Language Models to Generate Text with Citations",
        authors: "Tianyu Gao, Howard Yen, Jiatong Yu, Danqi Chen",
        published: "2023-05-24",
        primary_category: "cs.CL",
        url: "https://arxiv.org/abs/2305.14627",
      },
    ]);
    const out = parseSearchPayload("arxiv", stdout, "", 0);
    expect(out[0]!.year).toBe("2023-05-24");
    expect(out[0]!.rank).toBeUndefined();
  });

  it("drops entries missing title or url", () => {
    const stdout = JSON.stringify([
      { title: "Has no url", year: 2020 },
      { url: "https://example.com/x", year: 2021 },
      { title: "Keeper", url: "https://example.com/keep" },
    ]);
    const out = parseSearchPayload("openalex", stdout, "", 0);
    expect(out).toHaveLength(1);
    expect(out[0]!.title).toBe("Keeper");
  });

  it("tolerates an update-notice banner before the JSON", () => {
    const stdout =
      "  Update available: v1.7.14 → v1.7.16\n  Run: npm install -g @jackwener/opencli\n\n" +
      JSON.stringify([{ title: "T", url: "https://x", rank: 1 }]);
    const out = parseSearchPayload("arxiv", stdout, "", 0);
    expect(out).toHaveLength(1);
    expect(out[0]!.title).toBe("T");
  });
});

describe("parseSearchPayload — error envelopes", () => {
  it("opencli { ok:false } error → throws with the message", () => {
    const stdout = JSON.stringify({
      ok: false,
      error: { code: "COMMAND_EXEC", message: "arXiv API HTTP 429", exitCode: 1 },
    });
    expect(() => parseSearchPayload("arxiv", stdout, "", 1)).toThrow(/429/);
  });

  it("non-array JSON → throws", () => {
    expect(() => parseSearchPayload("openalex", JSON.stringify({ foo: 1 }), "", 0)).toThrow(
      /expected a JSON array/,
    );
  });

  it("non-JSON output + nonzero exit → throws with stderr", () => {
    expect(() => parseSearchPayload("pubmed", "garbage not json", "boom from stderr", 1)).toThrow(
      /boom from stderr/,
    );
  });

  it("empty output + nonzero exit → throws", () => {
    expect(() => parseSearchPayload("pubmed", "", "", 2)).toThrow(/exit 2/);
  });
});

describe("searchCitations — provider validation (no spawn)", () => {
  it("rejects an unknown provider before spawning", async () => {
    await expect(searchCitations("semanticscholar", "x")).rejects.toThrow(
      /unknown citation provider/,
    );
  });
});

describe("titleSimilarity", () => {
  it("identical titles → 1", () => {
    expect(titleSimilarity("A proportional hazards model", "A proportional hazards model")).toBe(1);
  });
  it("query terms mostly in title → high", () => {
    const sim = titleSimilarity(
      "Fine Gray proportional hazards subdistribution competing risk",
      "A proportional hazards model for the subdistribution of a competing risk",
    );
    expect(sim).toBeGreaterThan(0.4);
  });
  it("unrelated → low", () => {
    const sim = titleSimilarity(
      "ALCE benchmark citation evaluation",
      "Monocyte-to-HDL ratio associated with renal mortality in older individuals",
    );
    expect(sim).toBeLessThan(0.15);
  });
  it("empty → 0", () => {
    expect(titleSimilarity("", "anything")).toBe(0);
    expect(titleSimilarity("anything", "")).toBe(0);
  });
});
