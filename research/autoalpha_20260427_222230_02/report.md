# Factor Research: autoalpha_20260427_222230_02

## Formula
```
neg(ts_ema(cs_zscore(ts_mean(ts_pct_change(close_trade_px,3)*sigmoid(safe_div(trade_count,ts_median(trade_count,12))-1)*sigmoid(neg(ts_pct_change(safe_div(dvolume,trade_count),3))),4)),12))
```

## Metrics
| Metric | Value |
|--------|-------|
| IC | 1.0445 |
| IR | 4.2241 |
| Turnover | 279.98 |
| Score | 185.89 |
| PassGates | True |

## Distribution
| Stat | Value |
|------|-------|
| Mean | 0.0001 |
| Std | 0.2887 |
| Skewness | -0.0000 |
| Kurtosis | -1.2000 |
| % Positive | 50.01% |
