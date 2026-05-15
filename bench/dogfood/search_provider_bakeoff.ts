#!/usr/bin/env bun
// search_provider_bakeoff.ts — head-to-head search quality on vouch's
// gate-fire entities. Tests three AI-leaning search APIs (Exa / Tavily /
// Brave) on the 8 real fetchable-but-deleted cases from the
// session_claims audit (2026-05-14/15). Goal: which provider returns
// canonical doc URLs as top-1 most reliably?
//
// API keys via env: EXA_API_KEY / TAVILY_API_KEY / BRAVE_SEARCH_API_KEY
// (any subset). Providers missing a key are gracefully skipped.
//
// Sign-up links (1k-ish free queries each):
//   • Tavily: https://www.tavily.com/ (Free Research plan, 1k/mo)
//   • Exa:    https://exa.ai/ ($10 trial ≈ 1.4k searches one-shot)
//   • Brave:  https://api-dashboard.search.brave.com/ ($5/mo new-user credit)

import { writeFileSync } from "fs";
import { dirname, join } from "path";
import { fileURLToPath } from "url";

const HERE = dirname(fileURLToPath(import.meta.url));

const CASES: { entity: string; context: string; expected_canonical_domain: string[] }[] = [
  { entity: "compromise", context: "JavaScript NLP library", expected_canonical_domain: ["github.com/spencermountain/compromise", "npmjs.com/package/compromise"] },
  { entity: "wink-nlp", context: "JavaScript NLP library", expected_canonical_domain: ["github.com/winkjs/wink-nlp", "npmjs.com/package/wink-nlp", "winkjs.org"] },
  { entity: "Vertex Gemini 3.1 Pro", context: "Google Cloud language model service", expected_canonical_domain: ["cloud.google.com", "ai.google.dev"] },
  { entity: "marimo", context: "Python reactive notebook", expected_canonical_domain: ["marimo.io", "github.com/marimo-team"] },
  { entity: "KernelSHAP", context: "model explainability algorithm", expected_canonical_domain: ["shap.readthedocs.io", "github.com/shap/shap", "arxiv.org"] },
  { entity: "fastcmprsk", context: "R competing-risks survival package", expected_canonical_domain: ["cran.r-project.org/package=fastcmprsk", "github.com"] },
  { entity: "FActScore", context: "factuality evaluation dataset and method", expected_canonical_domain: ["github.com/shmsw25/FActScore", "arxiv.org/abs/2305.14251"] },
  { entity: "Letta", context: "AI agent memory framework / lab", expected_canonical_domain: ["letta.com", "github.com/letta-ai"] },
];

type ProviderResult = {
  provider: string;
  ok: boolean;
  results: { rank: number; title: string; url: string }[];
  error?: string;
  latency_ms: number;
};

async function searchExa(query: string): Promise<ProviderResult> {
  const key = process.env.EXA_API_KEY;
  if (!key) return { provider: "exa", ok: false, results: [], error: "EXA_API_KEY not set", latency_ms: 0 };
  const t0 = Date.now();
  try {
    const r = await fetch("https://api.exa.ai/search", {
      method: "POST",
      headers: { "Content-Type": "application/json", "x-api-key": key },
      body: JSON.stringify({ query, numResults: 3, type: "auto" }),
    });
    const latency_ms = Date.now() - t0;
    if (!r.ok) return { provider: "exa", ok: false, results: [], error: `HTTP ${r.status}: ${await r.text()}`, latency_ms };
    const data: any = await r.json();
    return {
      provider: "exa",
      ok: true,
      results: (data.results ?? []).slice(0, 3).map((x: any, i: number) => ({ rank: i + 1, title: x.title ?? "", url: x.url ?? "" })),
      latency_ms,
    };
  } catch (e: any) {
    return { provider: "exa", ok: false, results: [], error: e.message, latency_ms: Date.now() - t0 };
  }
}

async function searchTavily(query: string): Promise<ProviderResult> {
  const key = process.env.TAVILY_API_KEY;
  if (!key) return { provider: "tavily", ok: false, results: [], error: "TAVILY_API_KEY not set", latency_ms: 0 };
  const t0 = Date.now();
  try {
    const r = await fetch("https://api.tavily.com/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ api_key: key, query, max_results: 3, search_depth: "basic" }),
    });
    const latency_ms = Date.now() - t0;
    if (!r.ok) return { provider: "tavily", ok: false, results: [], error: `HTTP ${r.status}: ${await r.text()}`, latency_ms };
    const data: any = await r.json();
    return {
      provider: "tavily",
      ok: true,
      results: (data.results ?? []).slice(0, 3).map((x: any, i: number) => ({ rank: i + 1, title: x.title ?? "", url: x.url ?? "" })),
      latency_ms,
    };
  } catch (e: any) {
    return { provider: "tavily", ok: false, results: [], error: e.message, latency_ms: Date.now() - t0 };
  }
}

async function searchBrave(query: string): Promise<ProviderResult> {
  const key = process.env.BRAVE_SEARCH_API_KEY;
  if (!key) return { provider: "brave", ok: false, results: [], error: "BRAVE_SEARCH_API_KEY not set", latency_ms: 0 };
  const t0 = Date.now();
  try {
    const url = `https://api.search.brave.com/res/v1/web/search?q=${encodeURIComponent(query)}&count=3`;
    const r = await fetch(url, {
      headers: { "X-Subscription-Token": key, "Accept": "application/json" },
    });
    const latency_ms = Date.now() - t0;
    if (!r.ok) return { provider: "brave", ok: false, results: [], error: `HTTP ${r.status}: ${await r.text()}`, latency_ms };
    const data: any = await r.json();
    return {
      provider: "brave",
      ok: true,
      results: (data.web?.results ?? []).slice(0, 3).map((x: any, i: number) => ({ rank: i + 1, title: x.title ?? "", url: x.url ?? "" })),
      latency_ms,
    };
  } catch (e: any) {
    return { provider: "brave", ok: false, results: [], error: e.message, latency_ms: Date.now() - t0 };
  }
}

function scoreCanonical(result: { rank: number; url: string }, expectedDomains: string[]): boolean {
  return expectedDomains.some(d => result.url.toLowerCase().includes(d.toLowerCase()));
}

async function main() {
  const allRows: any[] = [];
  for (const c of CASES) {
    const query = `${c.entity} ${c.context}`;
    const [exa, tavily, brave] = await Promise.all([searchExa(query), searchTavily(query), searchBrave(query)]);
    const row = {
      entity: c.entity,
      context: c.context,
      expected_canonical_domain: c.expected_canonical_domain,
      providers: { exa, tavily, brave },
      top1_canonical: {
        exa: exa.results[0] ? scoreCanonical(exa.results[0], c.expected_canonical_domain) : false,
        tavily: tavily.results[0] ? scoreCanonical(tavily.results[0], c.expected_canonical_domain) : false,
        brave: brave.results[0] ? scoreCanonical(brave.results[0], c.expected_canonical_domain) : false,
      },
      any_top3_canonical: {
        exa: exa.results.some(r => scoreCanonical(r, c.expected_canonical_domain)),
        tavily: tavily.results.some(r => scoreCanonical(r, c.expected_canonical_domain)),
        brave: brave.results.some(r => scoreCanonical(r, c.expected_canonical_domain)),
      },
    };
    allRows.push(row);
    console.log(`\n[${c.entity}] ${c.context}`);
    for (const p of ["exa", "tavily", "brave"] as const) {
      const r = row.providers[p];
      if (!r.ok) {
        console.log(`  ${p.padEnd(8)} SKIP: ${r.error}`);
        continue;
      }
      const top1Mark = row.top1_canonical[p] ? "★" : " ";
      console.log(`  ${p.padEnd(8)} (${r.latency_ms}ms):`);
      for (const x of r.results) {
        const mark = scoreCanonical(x, c.expected_canonical_domain) ? "✓" : " ";
        console.log(`    ${mark} #${x.rank} ${x.url}`);
      }
    }
  }

  // Tally
  const tally = {
    top1_canonical: {
      exa: allRows.filter(r => r.top1_canonical.exa).length,
      tavily: allRows.filter(r => r.top1_canonical.tavily).length,
      brave: allRows.filter(r => r.top1_canonical.brave).length,
    },
    any_top3_canonical: {
      exa: allRows.filter(r => r.any_top3_canonical.exa).length,
      tavily: allRows.filter(r => r.any_top3_canonical.tavily).length,
      brave: allRows.filter(r => r.any_top3_canonical.brave).length,
    },
    cases_n: allRows.length,
    avg_latency_ms: {
      exa: Math.round(allRows.reduce((a, r) => a + (r.providers.exa.latency_ms || 0), 0) / allRows.length),
      tavily: Math.round(allRows.reduce((a, r) => a + (r.providers.tavily.latency_ms || 0), 0) / allRows.length),
      brave: Math.round(allRows.reduce((a, r) => a + (r.providers.brave.latency_ms || 0), 0) / allRows.length),
    },
  };

  console.log("\n=== TALLY ===");
  console.log(JSON.stringify(tally, null, 2));

  writeFileSync(join(HERE, "search-provider-bakeoff-rows.jsonl"), allRows.map(r => JSON.stringify(r)).join("\n"));
  writeFileSync(join(HERE, "search-provider-bakeoff-summary.json"), JSON.stringify({ tally, cases: CASES.map(c => c.entity) }, null, 2));
  console.log(`\nWrote rows + summary to bench/dogfood/`);
}

main().catch(e => { console.error(e); process.exit(1); });
