# Factor Card: autoalpha_20260421_131115_03

## Snapshot
- Status: PASS
- Theme: participation + smoothed intraday signal + reversion/contrast
- IC / IR / TVR / Score: 0.611 / 3.078 / 321.70 / 79.00
- Days / Coverage: 726 / 1.0

## Agent Thesis
This factor targets intraday continuation when price changes are positively synchronized with trade-count changes, capturing moves supported by real participation rather than isolated prints. It then gates strength by requiring current trade-count acceleration to be high versus its own recent history, so weak-activity drifts are down-weighted. A short linear decay plus volatility normalization helps reduce noise and turnover while keeping broad cross-sectional coverage.

## Gate Notes
- IC predictive power: pass
- IR consistency: pass
- TVR turnover: pass
- position concentration: pass

## Diagnostics
- IC decay lag 0: 0.860
- IC half-life: 7 bars
- Daily IC count: 726
- Alpha mean/std: 0.00010 / 0.28868
- Positive / in-bounds: 0.500 / 1.000

## Risk Notes
- Turnover gate uses corrected restricted raw alpha diff sum x100; current tvr=321.70.
- Concentration max/min bps: maxx=7.92, minn=-7.87.

## Formula
```text
neg(cs_zscore(ts_decay_linear((ts_corr(delta(close_trade_px,1), delta(trade_count,1), 10) * ts_rank(delta(trade_count,1), 24)) / (1 + ts_std(delta(close_trade_px,1), 12)), 5)))
```
