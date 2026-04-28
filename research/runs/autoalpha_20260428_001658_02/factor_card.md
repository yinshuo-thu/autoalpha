# Factor Card: autoalpha_20260428_001658_02

## Snapshot
- Status: PASS
- Theme: VWAP dislocation + participation + smoothed intraday signal
- IC / IR / TVR / Score: 1.041 / 4.908 / 294.79 / 198.01

## Agent Thesis
This factor targets short intraday continuation when price is persistently above/below a recent VWAP anchor and participation is expanding. The core uses normalized VWAP-drift strength, while a smooth gate boosts signals only when both trade_count and dvolume are accelerating, filtering weak moves. A 12-bar outer mean is used to suppress churn and keep turnover in a safer range.

## Gate Notes
- IC predictive power: pass
- IR consistency: pass
- TVR turnover: pass
- position concentration: pass
- LowCorrelation: pass

## Diagnostics
- IC mean: 1.041
- Daily IC count: 484
- Alpha mean/std: 0.00010 / 0.28868
- % Positive: 0.500
- Full-sample IC: 1.041
- Positive month ratio: 0.833

## Formula
```text
neg(ts_mean(cs_scale(tanh(safe_div(close_trade_px-lag(vwap,4),ts_std(vwap,12)))*sigmoid(ts_pct_change(trade_count,8)+ts_pct_change(dvolume,8))),12))
```
