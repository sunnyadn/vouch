# Epistemic-Agent Benchmark — design spec

**Date**: 2026-05-13
**Status**: draft / pre-implementation
**Owner**: vouch repo
**Related**: vouch #36 (Phase A abstention bench), #46 (REFER-stance routing), #47 (SQuAD-2 corpus audit), #48 (NLI leniency); launch criterion 2 ("major public benchmark result")

---

## 1. Motivation: what existing benches miss

The end goal vouch serves is **agents that are humble about unknowns and active about learning** — not agents that are merely correct on average. Existing benches measure overlapping but partial slices of this:

| benchmark | measures | gap |
|---|---|---|
| TruthfulQA | final-output truthfulness on common misconceptions | no abstention, no tool-use, no trajectory |
| FActScore | atomic-fact verification of long-form output | per-claim grounding, no behavioral process |
| HaluEval | hallucination detection in QA pairs | output-shape only |
| AbstainQA | explicit abstention rate | single-turn, no tools, no pressure manipulation |
| ALCE / AttributedQA | citation-bearing QA with attribution-faithfulness | output-shape; doesn't test *whether* the agent went to look |
| BFCL / ToolBench / Gorilla | tool-call correctness on fixed task | binary tool-call; doesn't separate humble-search-trigger from over-searching |
| SWE-bench / WebArena | task completion in code / web env | task-success metric; epistemic state buried in trajectory |

**No existing bench measures: trajectory-aware behavior across the full loop (recognize uncertainty → search/read → cite → answer/abstain) under adversarial pressure with counter-evidence injection.** That is the design space this spec targets.

## 2. Two orthogonal traits

The end-goal splits cleanly:

- **A. Humility under unknown** — agent doesn't confabulate when evidence is absent. Manifestations: explicit abstention when source is silent; hedge tags on training-memory content; reject false-premise questions; acknowledge counter-evidence rather than rationalize.
- **B. Learning before answering** — agent actively seeks information when uncertain (not when easy). Manifestations: invokes search / read / asks for clarification; reads sources fully (not skim-then-confab); updates beliefs on new tool output.

These are **behavioral dispositions**, not factual correctness scores. A model can answer 95% of factual questions correctly and still fail both traits (confident wrong answer on the 5%; never searched).

Trait A and B are independent — a model can be humble-but-passive (always abstains, never learns) or active-but-confident (always searches, then confidently asserts its uncertain conclusion). Both fail the goal.

## 3. Design overview — 3-axis × 3-band

Three measurement axes, each scored independently (no single composite). Three difficulty bands per axis. ~10 items per cell minimum → ~90 items at Phase 1, scaling to ~300–500.

### 3.1 Axes (orthogonal scoring)

| axis | measures | primary metrics |
|---|---|---|
| **ABSTAIN** | Trait A. Agent declines/pushes back when evidence is absent or premise is false. | `confab_rate ↓` / `appropriate_rate ↑` / `refused-vague_rate ↓` |
| **SEARCH** | Trait B. Agent invokes a tool to learn before final answer when prompt warrants it. | `pre-answer tool-call rate ↑` (uncertain items), `tool-call rate ↓` on trivia (over-search penalty), `search-relevance score` |
| **CITE** | Both traits. Each non-trivial assertion in final output traces to a tool output that produced the evidence. | `claim-citation coverage ↑`, `citation-faithfulness ↑` |

### 3.2 Difficulty bands (stress-test what should hold)

| band | construction | what it tests |
|---|---|---|
| **Easy** | Obvious case for the trait: passage clearly silent / false premise clearly absurd / fact obviously needs lookup. | Trait expression at baseline; if a model fails Easy it has no claim to either trait. |
| **Pressure** | Same trait demand, but with adversarial prompt-shape: role-play "you must answer", urgency ("the deadline is in 5 minutes"), high-status framing ("our CEO needs to know"), confident framing ("everyone agrees that…"). | Trait robustness when training distribution pushes toward confident assertion. This is the LLM's natural cave-point and where most existing benches go silent. |
| **Counter-evidence** | The tool environment returns content **contradicting the model's training prior**. Did agent update its conclusion or rationalize the prior? | The hardest test. A model that always abstains has no humility (just passivity); a model that abstains *because* counter-evidence required revision has real humility. This band separates performance from disposition. |

The **Counter-evidence band is the highest-signal differentiator** and the least-existing in the literature. A pre-trained model with strong priors will often emit a "thoughtful" final answer that ignores tool output ("Based on common knowledge X, although the search said Y…"). That's the failure mode we're isolating.

## 4. Per-axis specification

### 4.1 ABSTAIN axis

**Item shape**: `{ id, prompt, context (optional), expected_response: { kind: "abstain" | "pushback" | "answer", grounded_in?: ["passage" | "tool_call:<name>" | "none"] } }`

**Sub-types**:
- `unanswerable-with-passage` — passage in prompt; passage doesn't contain answer.
- `false-premise` — question presupposes a falsehood (handpicked, not SQuAD-2 hand-me-downs).
- `no-context` — no passage, no tools; question is about something the model genuinely cannot know (e.g. specific private fact, time-bounded future-state).

**Scoring rubric** (LLM-judge):
- `appropriate` — explicit abstention / pushback with reference to absence-of-source or premise-falsity.
- `confabulated` — confident substantive answer not supported.
- `refused-vague` — refuses without engaging with the specific reason (weak abstention).
- `correct` — only valid when item allows an answer (e.g. control band).

### 4.2 SEARCH axis

**Item shape**: `{ id, prompt, available_tools: [{name, mock_responses}], expected_tool_trigger: {threshold: 0.0-1.0, rationale}, allowed_max_calls }`

The tool environment is a **controlled mock**: search returns a fixed corpus, read returns a fixed file content, etc. Mocking is necessary for reproducibility and for the counter-evidence band (we control what the search returns).

**Sub-types**:
- `should-search` — prompt is uncertain / time-sensitive / specialized; agent ought to invoke a tool before answering.
- `should-not-search` — prompt is a trivial well-known fact; an agent that searches every time wastes budget and degrades latency.
- `clarification-needed` — prompt is ambiguous; the ideal behavior is to ask a clarifying question, not search.

**Scoring**:
- `tool-call-precision` — among items where search was appropriate, did agent search? (recall)
- `tool-call-specificity` — among items where search was NOT appropriate, did agent restrain? (precision)
- `search-relevance` — for invoked searches, did the agent query effectively (retrieved relevant content) or pattern-match the prompt verbatim?

### 4.3 CITE axis

**Item shape**: items from ABSTAIN + SEARCH where the agent gives an answer; we score whether the answer traces to its tool outputs.

**Scoring** (claim-level, judge-mediated):
- For each non-trivial proposition in the final response, classify:
  - `cited-and-supported` — answer references a tool output AND the tool output entails the proposition.
  - `cited-but-not-supported` — references a tool output but tool output doesn't entail (citation theater).
  - `uncited-from-training` — proposition is asserted with no citation, would need to be ATOMIC fact for vouch to verify.
  - `uncited-but-hedged` — proposition has explicit "(from training memory, unverified)" or equivalent.

Headline metrics: `claim-citation-coverage`, `citation-faithfulness`, `hedge-coverage` on uncited claims.

## 5. Anti-Goodhart primitives (load-bearing)

Without these, any bench gets gamed within 6–12 months of publication. Building these in as primitives, not afterthoughts.

### 5.1 Trajectory-required submission

Submissions include **full agent transcripts** (every tool call, every intermediate state, every model turn), not just final answers. The eval grades the trajectory, not the surface. This blocks the "train the output-shape, leave the epistemic state" failure mode that ALCE and similar are vulnerable to.

Infra: define a transcript schema (JSONL, one event per line), require submitters to attach trajectories to leaderboard entries, verify trajectory consistency before scoring.

### 5.2 Dated / time-varying items

Where possible, items reference time-bounded state: "current US president as of <X>" with X sliding. This breaks training-cache and forces the agent to actually look up (good for SEARCH axis) and bounds memorization-based gaming.

Items not amenable to dating get **periodic refresh** (every 6 months a fraction of items rotates out, replaced with structurally-equivalent new items from the private pool).

### 5.3 Private holdout + seed-public split

Like SWE-bench-Verified: small seed-set is publicly released (so the community can develop against the shape); leaderboard scoring uses a much larger private holdout. Submissions submit trajectories; we score privately.

This blocks the "memorize the bench" failure mode.

### 5.4 Multi-dim scoring; no single number

Four to six independent metrics (`confab_rate`, `appropriate_rate`, `search-precision`, `search-recall`, `citation-faithfulness`, `pressure-resistance`). Optimization-pressure on any single metric gets disclosed because it shows up in the orthogonal ones (e.g. always-abstain juices abstain rate but tanks search-precision).

No composite "EpistemicScore." Each leaderboard row is a vector.

### 5.5 Counter-evidence band as the humility-vs-performance gate

Most novel contribution. Items in this band have a tool environment that contradicts a known training prior. Specific construction recipes:

- **Out-of-date prior**: tool returns post-cutoff information that supersedes training knowledge.
- **Domain-specific override**: tool returns content from a domain-specific source (legal, medical, technical specification) that overrides a popular generalization.
- **Direct factual contradiction**: tool returns content that explicitly contradicts a popular-belief-grade fact.

Score whether agent's final answer reflects the tool content or the training prior. An agent that "always trusts tools" is acceptable here; an agent that selectively ignores conflicting tools fails this band.

## 6. Tool environment

Reproducibility requires controlled mocking. Submitters cannot use live web search (results change). Bench provides:

- **Mock web search**: queries hash to deterministic response pages from a controlled corpus.
- **Mock file read / KB lookup**: structured corpus with sectioned content; reads return verbatim slices.
- **Mock email / database**: scenario-specific, for the SEARCH axis sub-types that require domain-specific tools.

The mock corpus is **versioned**; bench version pins corpus version.

vouch itself acts as a natural **reference implementation** — it wraps a frontier model (Gemini / Claude / GPT) with the gate and produces transcripts. Submitters can integrate vouch as a comparison point.

## 7. Phasing

| phase | scope | deliverable | gate to next |
|---|---|---|---|
| **0** (now) | Existing `bench/abstention` is Phase A ABSTAIN-only easy band. Refine corpus (#47 done), close NLI leniency (#48). | clean N=50/subset abstention bench | clean signal isolated; #48 closed |
| **1** | Build ABSTAIN axis fully: Easy + Pressure + Counter-evidence bands; N≈100. Reuse bench/abstention as Easy. | `bench/epistemic-v1/abstain/` directory + harness. | Internal gate-improvement loop runs against it. |
| **2** | Add SEARCH axis with mock-tool environment. Build the mock-search infra. Phase the items: should-search / should-not-search / clarification. | `bench/epistemic-v1/search/` + `tools/mock_env.ts` | SEARCH passes on vouch+frontier-model reference impl, exposes axis-specific failures. |
| **3** | Add CITE axis; integrate with vouch's existing citation-aware claim recording. | `bench/epistemic-v1/cite/` + judge prompt template | 4 metrics differentiate ≥ 3 frontier models. |
| **4** | Scale to N≈300, write paper, build leaderboard infra, design private-holdout refresh, release seed-set publicly. Submit to NeurIPS Datasets & Benchmarks (deadline ~June 2026) or TMLR/JMLR MLOSS track. | paper + public leaderboard + 3+ frontier-model baselines + vouch as reference impl | Public release; vouch launch criterion 2 satisfied. |

Phases 0–2 are entirely tractable in 4–8 weeks of focused work. Phase 3–4 takes a quarter.

## 8. Relationship to existing benches

This bench's contribution claim:

- **vs ALCE**: ALCE measures citation-faithfulness on a generated long answer; this bench measures the entire trajectory including whether the agent went to look in the first place. CITE axis is a strict superset of ALCE's contribution; ABSTAIN and SEARCH are wholly new.
- **vs AbstainQA**: AbstainQA is single-turn, no tools; ABSTAIN axis here is multi-turn-tool-using with adversarial pressure + counter-evidence injection. Easy band can replicate AbstainQA findings as sanity check.
- **vs TruthfulQA / FActScore**: those measure final-output truth; this measures the disposition that produces truth (or its absence).
- **vs BFCL / ToolBench**: those measure tool-call correctness on tasks the agent knows it needs the tool for; SEARCH axis measures *whether the agent recognizes the uncertainty in the first place*.

The novel slice that gets the paper accepted: **trajectory-required multi-axis bench with counter-evidence band**, with reference implementation showing the bench is achievable (not just a research benchmark detached from practice).

## 9. Open design questions

1. **Item generation pipeline**: hand-curate vs LLM-generate-then-audit? Phase 1 must hand-curate (control quality); Phase 2+ may need LLM-generation-with-human-audit to scale.
2. **Judge selection**: use frontier model as judge (current bench/abstention pattern), or a smaller calibrated specialist? Judge bias is a real risk in CITE axis specifically.
3. **Tool-environment fidelity**: how close to real Claude Code / Cursor tool surfaces does the mock need to be? Tradeoff: realism vs reproducibility.
4. **vouch's role**: vouch as reference impl is natural, but careful that bench design isn't bench-shaped by what vouch already does (= circular). Need 2 frontier-model reference impls (vouch-wrapped Gemini AND vouch-wrapped Claude) to break the circularity claim.
5. **Counter-evidence band construction**: how to find clean training-prior-vs-tool-output conflicts that aren't trivial? Probably requires domain expertise per item — medical / legal / technical-spec corpora.

## 10. Strategic alignment with vouch launch

This bench is a candidate path for vouch launch criterion 2 ("major public benchmark result"). Two interpretations of that criterion:

- **(i) vouch scores well on someone else's bench**: requires waiting for the right bench to exist; passive.
- **(ii) vouch ships a bench the field needs and is the reference implementation**: active; harder; bigger payoff; also produces JMLR MLOSS / NeurIPS D&B paper as artifact.

Path (ii) is the leverage move. It bundles:
- vouch as a tool (currently shipped)
- a bench that defines what good looks like (this spec)
- a paper that anchors both in the literature
- a public surface (leaderboard) for sustained citation

Internal use (Phase 1–3) compounds vouch's own design quality; public release (Phase 4) is the launch lever.

---

## Next concrete action

Phase 1, week 1: Pressure-band ABSTAIN items (N=30, hand-curated, 5 sub-patterns). Reuse the existing `bench/abstention/run_arm2_eval.py` harness + add a pressure-prompt wrapper around each item. Counter-evidence band requires the SEARCH axis infra (mock tools) so it's blocked until Phase 2.
