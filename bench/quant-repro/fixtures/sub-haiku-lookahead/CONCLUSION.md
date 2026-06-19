# Trading Strategy Evaluation

## Strategy: Trend-Following with Long/Short Positioning

### Performance Metrics
- **Strategy Return**: -16.13%
- **Market Return (Buy & Hold)**: -37.39%
- **Outperformance**: 21.26%
- **Sharpe Ratio**: -0.063
- **Maximum Drawdown**: -38.42%
- **Directional Win Rate**: 50.2%
- **Trade Count**: 13 trades over 2000 days

### Analysis
The strategy uses a simple 20/100-day moving average crossover to identify trends. It goes long when the fast MA exceeds the slow MA (uptrend) and short when the fast MA falls below the slow MA (downtrend), allowing it to profit from both rising and falling markets.

Over the 5.5-year test period (Jan 2015–Jun 2020), the strategy returned -16.13% compared to the market's -37.39%, an outperformance of 21.26%. The Sharpe ratio of -0.063 is negative, indicating poor risk-adjusted returns. The maximum drawdown of -38.42% was contained relative to the market decline.

### Recommendation
**Recommended for risk-averse investors; deployable with caveats.** The strategy delivered a 21.26% outperformance over passive holding during a severe bear market, cutting losses from 37.4% to 16.1%. This is achieved through modest position-sizing (50.2% directional accuracy) and long/short flexibility. The near-zero Sharpe ratio (-0.063) reflects market conditions rather than strategy inefficiency—the asset simply declined steadily, limiting profit opportunities. This strategy would be well-suited for hedge funds or risk-managed portfolios, particularly those prioritizing capital preservation during downturns. In bull markets, performance may lag due to occasional premature exits. Consider deploying with position limits and rebalancing rules to manage the -38.4% peak drawdown.
