/** Suggester tests — #50 (B).
 *
 *  Verifies the deterministic provider-hint cascade and the always-search
 *  default. No store / no LLM / no network — this module is pure functions
 *  over the raw draft text.
 */
import { describe, expect, it } from "bun:test";
import { suggestVerification, detectProviderHint, renderSuggestionLine } from "../src/suggest.ts";

describe("detectProviderHint", () => {
  it("returns null for an empty draft", () => {
    expect(detectProviderHint("")).toBeNull();
  });

  it("returns null for prose with no ID or domain signal", () => {
    expect(detectProviderHint("ALCE is a benchmark for citation eval.")).toBeNull();
  });

  it("arxiv: 'arXiv:2302.09664' identifier", () => {
    expect(
      detectProviderHint("Kuhn et al's paper (arXiv:2302.09664) introduces semantic entropy."),
    ).toBe("arxiv");
  });

  it("arxiv: 'arXiv: 2302.09664' (space after colon)", () => {
    expect(detectProviderHint("see arXiv: 2302.09664 for details")).toBe("arxiv");
  });

  it("arxiv: version suffix v3 doesn't prevent detection", () => {
    expect(detectProviderHint("Latest revision is arXiv:2302.09664v3.")).toBe("arxiv");
  });

  it("arxiv: explicit arxiv.org domain mention", () => {
    expect(detectProviderHint("source: https://arxiv.org/abs/2302.09664")).toBe("arxiv");
  });

  it("pubmed: PMID identifier", () => {
    expect(detectProviderHint("see PMID: 12345678 for the original report")).toBe("pubmed");
  });

  it("pubmed: pubmed.ncbi.nlm.nih.gov domain", () => {
    expect(
      detectProviderHint("https://pubmed.ncbi.nlm.nih.gov/12345678/ has the abstract"),
    ).toBe("pubmed");
  });

  it("openalex: DOI prefix '10.1145/...'", () => {
    expect(detectProviderHint("the work (doi:10.1145/3580305.3599434) shows …")).toBe("openalex");
  });

  it("openalex: doi.org URL", () => {
    expect(detectProviderHint("https://doi.org/10.1145/3580305.3599434")).toBe("openalex");
  });

  it("priority: arxiv beats DOI when both appear", () => {
    expect(
      detectProviderHint("see arXiv:2302.09664 (doi:10.1145/foo) for the result"),
    ).toBe("arxiv");
  });

  it("does NOT confuse a bare 5-digit number with an arxiv ID", () => {
    // arxiv IDs have a literal dot — `12345` alone shouldn't match.
    expect(detectProviderHint("the dataset has 12345 examples")).toBeNull();
  });

  it("does NOT match a version string '1.0.0' or '2024.05.14' as arxiv", () => {
    // arxiv pattern requires the `arXiv:` prefix OR the domain. A bare
    // numeric.numeric without context is too noisy to safely hint.
    expect(detectProviderHint("vouch 0.2.1 released; SemVer 2024.05.14 today")).toBeNull();
  });
});

describe("suggestVerification", () => {
  it("default: no signal in draft → bare vouch search", () => {
    expect(suggestVerification("ALCE", "ALCE is a benchmark for citation eval.")).toBe(
      'vouch search "ALCE"',
    );
  });

  it("arxiv signal → --provider arxiv", () => {
    expect(
      suggestVerification(
        "Kuhn et al",
        "Kuhn et al's paper (arXiv:2302.09664) introduces semantic entropy.",
      ),
    ).toBe('vouch search "Kuhn et al" --provider arxiv');
  });

  it("pubmed signal → --provider pubmed", () => {
    expect(
      suggestVerification("Smith et al", "Smith et al, PMID:12345678, reported …"),
    ).toBe('vouch search "Smith et al" --provider pubmed');
  });

  it("DOI signal → --provider openalex (vouch has no semantic-scholar)", () => {
    expect(
      suggestVerification("Jones et al", "Jones et al (doi:10.1038/nature12373) showed …"),
    ).toBe('vouch search "Jones et al" --provider openalex');
  });

  it("escapes double-quotes in entity name for shell safety", () => {
    expect(suggestVerification('FActScore "v2"', "FActScore v2 is …")).toBe(
      'vouch search "FActScore \\"v2\\""',
    );
  });

  it("uses the RAW DRAFT, not the stripped proposition (regression test)", () => {
    // The extractor often strips arxiv IDs from the proposition. The suggester
    // must look at the raw draft to recover the ID hint. This test pins that
    // behavior: pass a proposition-shaped string for entity, a richer string
    // for draft, and confirm the draft is what drives provider selection.
    const proposition = "Kuhn et al's paper introduces semantic entropy.";
    const draft = `${proposition} See arXiv:2302.09664 for the camera-ready.`;
    // Suggester receives entity + draft; entity is taken as-is, draft is
    // pattern-scanned.
    expect(suggestVerification("Kuhn et al", draft)).toContain("--provider arxiv");
    // Sanity check: the proposition alone (no ID) gives no hint.
    expect(suggestVerification("Kuhn et al", proposition)).toBe(
      'vouch search "Kuhn et al"',
    );
  });
});

describe("renderSuggestionLine", () => {
  it("emits the bullet-aligned 6-space indent prefix", () => {
    const line = renderSuggestionLine('vouch search "Kuhn et al" --provider arxiv');
    expect(line).toMatch(/^ {6}→ suggested:/);
    expect(line).toContain('vouch search "Kuhn et al" --provider arxiv');
  });
});
