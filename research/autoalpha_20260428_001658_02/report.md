# Factor Research: autoalpha_20260428_001658_02

## Formula
```
neg(ts_mean(cs_scale(tanh(safe_div(close_trade_px-lag(vwap,4),ts_std(vwap,12)))*sigmoid(ts_pct_change(trade_count,8)+ts_pct_change(dvolume,8))),12))
```

## Metrics
| Metric | Value |
|--------|-------|
| IC | 1.0412 |
| IR | 4.9083 |
| Turnover | 294.79 |
| Score | 198.01 |
| PassGates | True |

## Distribution
| Stat | Value |
|------|-------|
| Mean | 0.0001 |
| Std | 0.2887 |
| Skewness | -0.0000 |
| Kurtosis | -1.2000 |
| % Positive | 50.01% |
