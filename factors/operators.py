import numpy as np
import pandas as pd

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

def ts_zscore(df, window):
    eps = 1e-8
    mean = ts_mean(df, window)
    std = ts_std(df, window)
    return (df - mean) / (std + eps)

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
    eps = 1e-8
    def _zscore(x):
        arr = x.values.copy()
        means = np.nanmean(arr, axis=1, keepdims=True)
        stds = np.nanstd(arr, axis=1, keepdims=True)
        
        # In-place modifications
        np.subtract(arr, means, out=arr, where=~np.isnan(arr))
        np.divide(arr, stds + eps, out=arr, where=~np.isnan(arr))
        return pd.DataFrame(arr, index=x.index, columns=x.columns)
    return wrap_cs(df, _zscore)

# Safe math
def safe_div(a, b):
    eps = 1e-8
    return a / b.replace(0, eps)

def signed_power(a, p):
    return np.sign(a) * (np.abs(a) ** p)
