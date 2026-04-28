# Factor Card: autoalpha_20260428_130012_02

## Snapshot
- Status: PASS
- Theme: participation + smoothed intraday signal + reversion/contrast
- IC / IR / TVR / Score: 1.005 / 4.045 / 249.37 / 177.02

## Agent Thesis
This factor targets post-impulse exhaustion: a 1-hour price push is less durable when order count strength is not matched by traded amount strength. The trade_count-minus-dvolume z-score spread acts as a small-order crowding proxy, and sigmoid keeps the gate smooth and robust. Multiplying this gate by recent price impulse isolates potentially overextended moves, then applying neg() converts it into a mean-reversion signal. Cross-sectional z-scoring and a 15-bar decay smoother are used to stabilize dispersion and control turnover.

## Gate Notes
- IC predictive power: pass
- IR consistency: pass
- TVR turnover: pass
- position concentration: pass
- LowCorrelation: pass

## Diagnostics
- IC mean: 1.005
- Daily IC count: 483
- Alpha mean/std: 0.00010 / 0.28868
- % Positive: 0.500
- Full-sample IC: 1.005
- Positive month ratio: 0.750

## Formula
```text
neg(ts_decay_linear(cs_zscore(ts_mean(ts_pct_change(close_trade_px,4)*sigmoid(ts_zscore(trade_count,12)-ts_zscore(dvolume,12)),4)),15))
```
