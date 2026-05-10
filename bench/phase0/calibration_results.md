# Phase 0 verifier-swap calibration results

Sample: 100 stratified instances from LLM-AggreFact test split (seed=20260510).
Generated: 2026-05-10 02:57:46.

| Verifier | License | Bal. Acc. | FN rate | FP rate | Median lat (s) | Total wall (s) | Valid / Total |
|---|---|---:|---:|---:|---:|---:|---:|
| vertex-pro | Google ToS / paid | 0.815 | 0.200 | 0.170 | 3.69 | 410.7 | 98 / 100 |
| vertex-flash | Google ToS / paid | 0.799 | 0.111 | 0.291 | 0.92 | 92.6 | 100 / 100 |
| minicheck-t5 | MIT | 0.775 | 0.178 | 0.273 | 0.61 | 111.3 | 100 / 100 |

## Verifier targets

Pick the cheapest verifier whose balanced accuracy is within 3pp of vertex-pro.
If minicheck-t5 (MIT, local, ~$0 marginal) clears that bar, use it for the ALCE benchmark and the $500 budget gate becomes irrelevant for the verifier side.
