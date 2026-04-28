# Factor Card: autoalpha_20260427_232013_02

## Snapshot
- Status: PASS
- Theme: VWAP dislocation + participation + smoothed intraday signal
- IC / IR / TVR / Score: 0.712 / 4.003 / 210.10 / 121.47

## Agent Thesis
This factor targets intraday sell-pressure exhaustion: when close sits below VWAP, but bar-to-bar price travel is compressed, downside often reflects absorption rather than fresh informed selling. I gate that setup by elevated trade_count relative to dvolume, capturing many small prints with weak capital commitment. The signal is then cross-sectionally standardized and slow-smoothed to reduce turnover and improve stability.

## Gate Notes
- IC predictive power: pass
- IR consistency: pass
- TVR turnover: pass
- position concentration: pass
- LowCorrelation: pass

## Diagnostics
- IC mean: 0.712
- Daily IC count: 484
- Alpha mean/std: 0.00010 / 0.28868
- % Positive: 0.500
- Full-sample IC: 0.712
- Positive month ratio: 0.833

## Formula
```text
ts_decay_linear(cs_zscore(ts_mean((safe_div(vwap-close_trade_px,vwap)+neg(safe_div(abs(delta(close_trade_px,2)),vwap)))*sigmoid(ts_zscore(trade_count,12)-ts_zscore(dvolume,12)),4)),15)
```
