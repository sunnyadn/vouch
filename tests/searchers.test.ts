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
  __test,
} from "../src/searchers.ts";

const { parseDdgHtml, resolveDdgRedirect, decodeHtmlEntities } = __test;

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

describe("DDG HTML parsing (no network)", () => {
  // Minimal but realistic slice of html.duckduckgo.com/html/ markup.
  const sampleHtml = `
    <div class="result results_links results_links_deep web-result">
      <div class="links_main">
        <h2 class="result__title">
          <a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fnginx.org%2Fen%2Fdocs%2Fhttp%2Fngx_http_limit_req_module.html&amp;rut=abc">Module ngx_http_limit_req_module</a>
        </h2>
        <a class="result__snippet" href="https://nginx.org/...">Limits the request processing rate per a defined key, in particular the rate of requests from a single IP.</a>
      </div>
    </div>
    <div class="result results_links results_links_deep web-result">
      <div class="links_main">
        <h2 class="result__title">
          <a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.getpagespeed.com%2Fnginx-rate-limiting&amp;rut=def">NGINX Rate Limiting: Complete Guide</a>
        </h2>
        <a class="result__snippet">Configure NGINX rate limiting with limit_req_zone &amp; limit_req. Covers burst, nodelay &amp; multiple zones.</a>
      </div>
    </div>`;

  it("extracts title/url/snippet, unwrapping the DDG redirect", () => {
    const out = parseDdgHtml(sampleHtml, 10);
    expect(out).toHaveLength(2);
    expect(out[0]).toEqual({
      title: "Module ngx_http_limit_req_module",
      url: "https://nginx.org/en/docs/http/ngx_http_limit_req_module.html",
      snippet:
        "Limits the request processing rate per a defined key, in particular the rate of requests from a single IP.",
    });
    expect(out[1]!.url).toBe("https://www.getpagespeed.com/nginx-rate-limiting");
    expect(out[1]!.snippet).toContain("burst, nodelay & multiple zones");
  });

  it("respects the limit", () => {
    expect(parseDdgHtml(sampleHtml, 1)).toHaveLength(1);
  });

  it("returns [] for a page with no result anchors", () => {
    expect(parseDdgHtml("<html><body>no results here</body></html>", 10)).toEqual([]);
  });

  it("passes through an already-absolute href", () => {
    expect(resolveDdgRedirect("https://example.com/x")).toBe("https://example.com/x");
    expect(resolveDdgRedirect("//example.com/x")).toBe("https://example.com/x");
  });

  it("unwraps the uddg redirect param", () => {
    expect(
      resolveDdgRedirect("//duckduckgo.com/l/?uddg=https%3A%2F%2Ffoo.dev%2Fbar%3Fa%3D1&amp;rut=z"),
    ).toBe("https://foo.dev/bar?a=1");
  });

  it("decodeHtmlEntities handles the common entities", () => {
    expect(decodeHtmlEntities("a &amp; b &lt;c&gt; &quot;d&quot; &#39;e&#39; &#x2014;")).toBe(
      'a & b <c> "d" \'e\' —',
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
