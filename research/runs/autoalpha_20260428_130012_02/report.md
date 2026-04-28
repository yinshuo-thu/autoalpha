# Factor Research: autoalpha_20260428_130012_02

## Formula
```
neg(ts_decay_linear(cs_zscore(ts_mean(ts_pct_change(close_trade_px,4)*sigmoid(ts_zscore(trade_count,12)-ts_zscore(dvolume,12)),4)),15))
```

## Metrics
| Metric | Value |
|--------|-------|
| IC | 1.0048 |
| IR | 4.0453 |
| Turnover | 249.37 |
| Score | 177.02 |
| PassGates | True |

## Distribution
| Stat | Value |
|------|-------|
| Mean | 0.0001 |
| Std | 0.2887 |
| Skewness | 0.0000 |
| Kurtosis | -1.2000 |
| % Positive | 50.01% |
