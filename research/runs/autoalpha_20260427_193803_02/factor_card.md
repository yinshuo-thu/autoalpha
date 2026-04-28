# Factor Card: autoalpha_20260427_193803_02

## Snapshot
- Status: PASS
- Theme: participation + smoothed intraday signal + reversion/contrast
- IC / IR / TVR / Score: 0.937 / 3.308 / 115.36 / 159.92

## Agent Thesis
This factor captures intraday continuation when short-vs-medium price trend is positive and trade amount per print is improving, which can indicate larger informed orders rather than just more small trades. The sigmoid gate on per-trade dvolume change suppresses weak-flow moves and emphasizes moves backed by order-size quality. Cross-sectional z-scoring plus a 15-bar decay smoother targets stability and lower turnover, addressing the main failure mode in recent runs.

## Gate Notes
- IC predictive power: pass
- IR consistency: pass
- TVR turnover: pass
- position concentration: pass
- LowCorrelation: pass

## Diagnostics
- IC mean: 0.937
- Daily IC count: 483
- Alpha mean/std: 0.00010 / 0.28868
- % Positive: 0.500
- Full-sample IC: 0.937
- Positive month ratio: 0.875

## Formula
```text
neg(ts_decay_linear(cs_zscore(ts_mean((ts_ema(close_trade_px,5)-ts_ema(close_trade_px,20))*sigmoid(ts_pct_change(safe_div(dvolume,trade_count),5)),4)),15))
```
