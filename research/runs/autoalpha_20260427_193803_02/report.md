# Factor Research: autoalpha_20260427_193803_02

## Formula
```
neg(ts_decay_linear(cs_zscore(ts_mean((ts_ema(close_trade_px,5)-ts_ema(close_trade_px,20))*sigmoid(ts_pct_change(safe_div(dvolume,trade_count),5)),4)),15))
```

## Metrics
| Metric | Value |
|--------|-------|
| IC | 0.9370 |
| IR | 3.3080 |
| Turnover | 115.36 |
| Score | 159.92 |
| PassGates | True |

## Distribution
| Stat | Value |
|------|-------|
| Mean | 0.0001 |
| Std | 0.2887 |
| Skewness | -0.0000 |
| Kurtosis | -1.2000 |
| % Positive | 50.01% |
