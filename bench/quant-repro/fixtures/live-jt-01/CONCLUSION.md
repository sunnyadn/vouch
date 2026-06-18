# Momentum Strategy Replication: Jegadeesh & Titman (1993)

Using Ken French's 10 Portfolios Formed on Prior 12-2 Returns (value-weighted, monthly), I constructed a winners-minus-losers (WML) momentum portfolio as the return difference between the `Hi PRIOR` decile (past winners) and the `Lo PRIOR` decile (past losers). From January 1927 to April 2026 (1192 monthly observations), the WML portfolio earned an average monthly return of **1.145%** (annualized: 13.74%).

The strategy's monthly volatility was 7.929% (annualized: 27.47%), yielding a t-statistic of **4.987** and a two-sided p-value of **0.0000**. This difference is statistically significant at the 5% level, consistent with the Jegadeesh-Titman momentum effect.

The annualized Sharpe ratio was 0.500. Overall, the analysis successfully reproduces the classic momentum premium: past winners significantly outperformed past losers over the long sample, though this evidence reflects value-weighted U.S. equities and does not account for transaction costs or risk adjustment.
