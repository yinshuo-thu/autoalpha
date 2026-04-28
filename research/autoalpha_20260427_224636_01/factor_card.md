# Factor Card: autoalpha_20260427_224636_01

## Snapshot
- Status: PASS
- Theme: participation + smoothed intraday signal + reversion/contrast
- IC / IR / TVR / Score: 0.619 / 4.817 / 308.42 / 102.06

## Agent Thesis
This factor targets supply-withdrawal continuation: when bars finish close to the high, sellers are not pressing into the close. If that pattern occurs while traded value is below its own recent z-scored baseline, the move is more likely driven by inventory lock-up than broad aggressive buying, which can persist for several bars. I use a short local average for pattern persistence, cross-sectional z-scoring for comparability, and a 15-bar decay smoother to control turnover.

## Gate Notes
- IC predictive power: pass
- IR consistency: pass
- TVR turnover: pass
- position concentration: pass
- LowCorrelation: pass

## Diagnostics
- IC mean: 0.619
- Daily IC count: 484
- Alpha mean/std: 0.00010 / 0.28868
- % Positive: 0.500
- Full-sample IC: 0.619
- Positive month ratio: 0.917

## Formula
```text
neg(neg(ts_decay_linear(cs_zscore(ts_mean(safe_div(high_trade_px-close_trade_px,high_trade_px-low_trade_px)*sigmoid(neg(ts_zscore(dvolume,12))),4)),15)))
```
