# Factor Research: autoalpha_20260427_224636_01

## Formula
```
neg(neg(ts_decay_linear(cs_zscore(ts_mean(safe_div(high_trade_px-close_trade_px,high_trade_px-low_trade_px)*sigmoid(neg(ts_zscore(dvolume,12))),4)),15)))
```

## Metrics
| Metric | Value |
|--------|-------|
| IC | 0.6192 |
| IR | 4.8172 |
| Turnover | 308.42 |
| Score | 102.06 |
| PassGates | True |

## Distribution
| Stat | Value |
|------|-------|
| Mean | 0.0001 |
| Std | 0.2887 |
| Skewness | 0.0000 |
| Kurtosis | -1.2000 |
| % Positive | 50.01% |
