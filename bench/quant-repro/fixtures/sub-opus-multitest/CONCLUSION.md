# Conclusion: Predictors of `target`

I regressed next-period `target` on each of the 20 candidate signals individually
(N = 600 periods) and ranked them by p-value. At a naive threshold of p < 0.05,
two signals appear "significant": **signal_3** (t = −3.09, p = 0.0021) and
**signal_15** (t = 2.56, p = 0.011). However, testing 20 signals means we expect
~1 false positive at p < 0.05 by chance alone, so this raw count is roughly what
pure noise would produce. After correcting for the 20 tests, only **signal_3**
survives: Bonferroni p = 0.042 and Benjamini–Hochberg FDR p = 0.042, while
signal_15 fades (Bonferroni p = 0.21, FDR p = 0.11). The effect sizes are tiny —
signal_3's correlation with target is only −0.125, explaining ~1.6% of variance.
Crucially, a split-half robustness check shows signal_3 is unstable: it is
insignificant in the first half (t = −1.39, p = 0.17) and only significant in the
second half (t = −2.88, p = 0.004), so the signal is not consistent across the
sample. **Verdict: no robust, reliable predictor was found.** signal_3 is the only
candidate that clears multiple-testing correction, but it just barely clears it,
has a negligible effect size, and does not hold up out-of-sample — consistent with
a marginal false positive rather than a real edge. I would not deploy any of these
signals for trading without substantially more out-of-sample validation.
