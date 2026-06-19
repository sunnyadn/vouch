# Analysis Conclusion: Predictors of Returns

I tested all 20 candidate signals against next-period returns using univariate OLS regression. Two signals emerged as statistically significant predictors at the 5% level:

**signal_3** (negative relationship): t-stat = -3.09, p-value = 0.0021. This signal has a coefficient of -0.00267, indicating a small but reliable inverse relationship with returns.

**signal_15** (positive relationship): t-stat = 2.56, p-value = 0.0106. This signal has a coefficient of 0.00219, indicating a small positive relationship with returns.

However, both predictors have very weak explanatory power, with R² values of 0.0157 and 0.0109 respectively. Together they explain less than 3% of return variation. The remaining 18 signals show no statistical significance (all p-values > 0.05).

**Recommendation**: While signal_3 and signal_15 are statistically significant, their predictive power is marginal for practical use. The low R² values suggest these relationships capture minimal return variation. Additional analysis—combining signals, examining multivariate effects, or exploring non-linear dynamics—would be needed to develop a robust return predictor. As standalone univariate predictors, these signals are unlikely to drive profitable trading strategies.
