# Factor Research: autoalpha_20260427_232013_02

## Formula
```
ts_decay_linear(cs_zscore(ts_mean((safe_div(vwap-close_trade_px,vwap)+neg(safe_div(abs(delta(close_trade_px,2)),vwap)))*sigmoid(ts_zscore(trade_count,12)-ts_zscore(dvolume,12)),4)),15)
```

## Metrics
| Metric | Value |
|--------|-------|
| IC | 0.7122 |
| IR | 4.0032 |
| Turnover | 210.10 |
| Score | 121.47 |
| PassGates | True |

## Distribution
| Stat | Value |
|------|-------|
| Mean | 0.0001 |
| Std | 0.2887 |
| Skewness | -0.0000 |
| Kurtosis | -1.2000 |
| % Positive | 50.01% |
