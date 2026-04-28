# Factor Card: autoalpha_20260427_222230_02

## Snapshot
- Status: PASS
- Theme: participation + smoothed intraday signal + reversion/contrast
- IC / IR / TVR / Score: 1.044 / 4.224 / 279.98 / 185.89

## Agent Thesis
This factor targets late-stage intraday exhaustion: a 30–45 minute move becomes fragile when trade_count stays high versus its robust median while average trade size (dvolume/trade_count) is shrinking. That combination often reflects order-splitting and retail chasing into passive absorption, so continuation weakens and short-horizon mean reversion rises. The signal scales directional move by two soft participation gates, then applies cross-sectional standardization and a 12-bar EMA wrapper to suppress turnover noise.

## Gate Notes
- IC predictive power: pass
- IR consistency: pass
- TVR turnover: pass
- position concentration: pass
- LowCorrelation: pass

## Diagnostics
- IC mean: 1.044
- Daily IC count: 484
- Alpha mean/std: 0.00010 / 0.28868
- % Positive: 0.500
- Full-sample IC: 1.044
- Positive month ratio: 0.750

## Formula
```text
neg(ts_ema(cs_zscore(ts_mean(ts_pct_change(close_trade_px,3)*sigmoid(safe_div(trade_count,ts_median(trade_count,12))-1)*sigmoid(neg(ts_pct_change(safe_div(dvolume,trade_count),3))),4)),12))
```
