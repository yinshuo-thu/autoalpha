# Factor Research: autoalpha_20260427_201529_02

## Formula
```
neg(ts_decay_linear(cs_zscore(neg((safe_div(close_trade_px-vwap,vwap)+safe_div(close_trade_px-low_trade_px,high_trade_px-low_trade_px))*(1-safe_div(ts_mean(safe_div(dvolume,trade_count),6),ts_quantile(safe_div(dvolume,trade_count),24,0.7))))),15))
```

## Metrics
| Metric | Value |
|--------|-------|
| IC | 0.5528 |
| IR | 2.1747 |
| Turnover | 278.29 |
| Score | 0.00 |
| PassGates | False |

## Distribution
| Stat | Value |
|------|-------|
| Mean | 0.0001 |
| Std | 0.2887 |
| Skewness | 0.0000 |
| Kurtosis | -1.2000 |
| % Positive | 50.01% |
