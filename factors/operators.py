import numpy as np
import pandas as pd

EPS = 1e-8

# Ensure proper time series handling
def wrap_ts(df, func, window):
    # Unstack security_id, roll, then stack
    if df.empty: return df
    unstacked = df.unstack('security_id')
    rolled = func(unstacked, window)
    return rolled.stack('security_id').reorder_levels(df.index.names).sort_index()

def lag(df, periods=1):
    return wrap_ts(df, lambda x, p: x.shift(p), periods)

def delta(df, periods=1):
    return df - lag(df, periods)

def ts_mean(df, window):
    return wrap_ts(df, lambda x, w: x.rolling(w, min_periods=1).mean(), window)

def ts_std(df, window):
    return wrap_ts(df, lambda x, w: x.rolling(w, min_periods=2).std(), window)

def ts_sum(df, window):
    return wrap_ts(df, lambda x, w: x.rolling(w, min_periods=1).sum(), window)

def ts_max(df, window):
    return wrap_ts(df, lambda x, w: x.rolling(w, min_periods=1).max(), window)

def ts_min(df, window):
    return wrap_ts(df, lambda x, w: x.rolling(w, min_periods=1).min(), window)

def ts_median(df, window):
    return wrap_ts(df, lambda x, w: x.rolling(w, min_periods=1).median(), window)

def ts_quantile(df, window, q):
    q = float(q)
    return wrap_ts(df, lambda x, w: x.rolling(w, min_periods=1).quantile(q), window)

def ts_skew(df, window):
    return wrap_ts(df, lambda x, w: x.rolling(w, min_periods=3).skew(), window)

def ts_kurt(df, window):
    return wrap_ts(df, lambda x, w: x.rolling(w, min_periods=4).kurt(), window)

def ts_ema(df, span):
    return wrap_ts(df, lambda x, w: x.ewm(span=int(w), adjust=False, min_periods=1).mean(), span)

def ts_argmax(df, window):
    def _argmax(x, w):
        return x.rolling(w, min_periods=1).apply(lambda arr: float(np.nanargmax(arr)) + 1.0 if np.isfinite(arr).any() else np.nan, raw=True)
    return wrap_ts(df, _argmax, window)

def ts_argmin(df, window):
    def _argmin(x, w):
        return x.rolling(w, min_periods=1).apply(lambda arr: float(np.nanargmin(arr)) + 1.0 if np.isfinite(arr).any() else np.nan, raw=True)
    return wrap_ts(df, _argmin, window)

def ts_pct_change(df, periods=1):
    return safe_div(df, lag(df, periods)) - 1.0

def ts_minmax_norm(df, window):
    lo = ts_min(df, window)
    hi = ts_max(df, window)
    return safe_div(df - lo, hi - lo)

def ts_zscore(df, window):
    mean = ts_mean(df, window)
    std = ts_std(df, window)
    return safe_div(df - mean, std)

def ts_cov(df_x, df_y, window):
    if df_x.empty or df_y.empty: return df_x
    ux = df_x.unstack('security_id')
    uy = df_y.unstack('security_id')
    rolled = ux.rolling(window, min_periods=1).cov(uy)
    return rolled.stack('security_id').reorder_levels(df_x.index.names).sort_index()

def ts_corr(df_x, df_y, window):
    if df_x.empty or df_y.empty: return df_x
    ux = df_x.unstack('security_id')
    uy = df_y.unstack('security_id')
    rolled = ux.rolling(window, min_periods=1).corr(uy)
    return rolled.stack('security_id').reorder_levels(df_x.index.names).sort_index()

def ts_rank(df, window):
    # Natively vectorized Cython rolling rank (substitutes 55 million Python lambda calls)
    # Replaced with bespoke Numba LLVM C-Engine
    from core.c_backend import c_ts_rank_2d
    def _rank_c_wrapper(x, w):
        res = c_ts_rank_2d(x.values, w)
        return pd.DataFrame(res, index=x.index, columns=x.columns)
    return wrap_ts(df, _rank_c_wrapper, window)

def ts_decay_linear(df, window):
    from core.c_backend import c_ts_decay_linear_2d
    def _decay_c_wrapper(x, w):
        res = c_ts_decay_linear_2d(x.values, w)
        return pd.DataFrame(res, index=x.index, columns=x.columns)
    return wrap_ts(df, _decay_c_wrapper, window)

# Cross-sectional
def wrap_cs(df, func):
    if df.empty: return df
    unstacked = df.unstack('security_id')
    processed = func(unstacked)
    return processed.stack('security_id').reorder_levels(df.index.names).sort_index()

def cs_rank(df):
    def _rank(x):
        n = x.count(axis=1)
        return x.rank(axis=1).sub(1).div((n - 1).replace(0, np.nan), axis=0)
    return wrap_cs(df, _rank)

def cs_demean(df):
    def _demean(x):
        arr = x.values.copy()
        means = np.nanmean(arr, axis=1, keepdims=True)
        # In-place subtract
        np.subtract(arr, means, out=arr, where=~np.isnan(arr))
        return pd.DataFrame(arr, index=x.index, columns=x.columns)
    return wrap_cs(df, _demean)

def cs_zscore(df):
    def _zscore(x):
        arr = x.values.copy()
        means = np.nanmean(arr, axis=1, keepdims=True)
        stds = np.nanstd(arr, axis=1, keepdims=True)
        
        # In-place modifications
        np.subtract(arr, means, out=arr, where=~np.isnan(arr))
        np.divide(arr, stds + EPS, out=arr, where=~np.isnan(arr))
        return pd.DataFrame(arr, index=x.index, columns=x.columns)
    return wrap_cs(df, _zscore)

def cs_scale(df):
    def _scale(x):
        demeaned = x.sub(x.mean(axis=1), axis=0)
        denom = demeaned.abs().sum(axis=1).replace(0, np.nan)
        return demeaned.div(denom, axis=0)
    return wrap_cs(df, _scale)

def cs_winsorize(df, p=0.01):
    p = float(p)
    def _winsorize(x):
        lo = x.quantile(p, axis=1)
        hi = x.quantile(1.0 - p, axis=1)
        return x.clip(lower=lo, upper=hi, axis=0)
    return wrap_cs(df, _winsorize)

def cs_quantile(df, q=0.5):
    q = float(q)
    def _quantile(x):
        threshold = x.quantile(q, axis=1)
        return x.sub(threshold, axis=0)
    return wrap_cs(df, _quantile)

def cs_neutralize(x, y):
    """Cross-sectional residual of x after linear projection on y at each bar."""
    if x.empty:
        return x
    ux = x.unstack('security_id')
    uy = y.unstack('security_id').reindex_like(ux)
    x_dm = ux.sub(ux.mean(axis=1), axis=0)
    y_dm = uy.sub(uy.mean(axis=1), axis=0)
    beta = (x_dm * y_dm).sum(axis=1) / ((y_dm * y_dm).sum(axis=1) + EPS)
    residual = x_dm - y_dm.mul(beta, axis=0)
    return residual.stack('security_id').reorder_levels(x.index.names).sort_index()

# Safe math
def _series_like(value):
    return isinstance(value, (pd.Series, pd.DataFrame))

def safe_div(a, b):
    if _series_like(b):
        denom = b.replace(0, EPS)
    else:
        denom = EPS if b == 0 else b
    return a / denom

def signed_power(a, p):
    return np.sign(a) * (np.abs(a) ** p)

def signed_log(a):
    return np.sign(a) * np.log1p(np.abs(a))

def safe_log(a):
    return np.log1p(np.abs(a))

def safe_sqrt(a):
    return np.sqrt(np.abs(a))

def sigmoid(a):
    return 1.0 / (1.0 + np.exp(-np.clip(a, -50, 50)))

def clamp(a, lo, hi):
    return a.clip(float(lo), float(hi)) if hasattr(a, "clip") else np.clip(a, float(lo), float(hi))

def min_of(a, b):
    return np.minimum(a, b)

def max_of(a, b):
    return np.maximum(a, b)

def ifelse(cond, a, b):
    mask = cond > 0
    if hasattr(a, "where"):
        return a.where(mask, b)
    if hasattr(cond, "index"):
        return pd.Series(np.where(mask, a, b), index=cond.index)
    return np.where(mask, a, b)

def gt(a, b):
    return (a > b).astype(float)

def ge(a, b):
    return (a >= b).astype(float)

def lt(a, b):
    return (a < b).astype(float)

def le(a, b):
    return (a <= b).astype(float)

def eq(a, b):
    return (np.abs(a - b) <= EPS).astype(float)

def and_op(a, b):
    return ((a > 0) & (b > 0)).astype(float)

def or_op(a, b):
    return ((a > 0) | (b > 0)).astype(float)

def not_op(a):
    return (a <= 0).astype(float)

def mean_of(*signals):
    if not signals:
        raise ValueError("mean_of requires at least one signal")
    total = signals[0]
    for sig in signals[1:]:
        total = total + sig
    return total / float(len(signals))

def weighted_sum(*args):
    if len(args) < 2 or len(args) % 2 != 0:
        raise ValueError("weighted_sum expects pairs: weight, signal")
    out = None
    weight_total = 0.0
    for i in range(0, len(args), 2):
        w = float(args[i])
        sig = args[i + 1]
        out = sig * w if out is None else out + sig * w
        weight_total += abs(w)
    return out / (weight_total + EPS)

def combine_rank(*signals):
    if not signals:
        raise ValueError("combine_rank requires at least one signal")
    ranked = [cs_rank(sig) for sig in signals]
    return mean_of(*ranked)
