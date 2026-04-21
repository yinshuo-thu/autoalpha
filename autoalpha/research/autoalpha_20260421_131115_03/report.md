# Factor Research: autoalpha_20260421_131115_03

## Formula
```
neg(cs_zscore(ts_decay_linear((ts_corr(delta(close_trade_px,1), delta(trade_count,1), 10) * ts_rank(delta(trade_count,1), 24)) / (1 + ts_std(delta(close_trade_px,1), 12)), 5)))
```

## Platform Metrics
| Metric | Value |
|--------|-------|
| IC | 0.6112 |
| IR | 3.0776 |
| Turnover | 3.55 |
| Score | 106.90 |
| PassGates | True |

## Alpha Distribution
| Stat | Value |
|------|-------|
| Mean | 0.0001 |
| Std | 0.2887 |
| Skewness | -0.0000 |
| Kurtosis | -1.2000 |
| % Positive | 50.01% |

## IC Decay (by lag)
| Lag | IC×100 |
|-----|--------|
| 0 | 0.8597 |
| 1 | 0.7888 |
| 2 | 0.6726 |
| 3 | 0.5994 |
| 4 | 0.5466 |
| 5 | 0.4949 |
| 6 | 0.4446 |
| 7 | 0.3931 |
| 8 | 0.3525 |
| 9 | 0.3234 |
| 10 | 0.2941 |
