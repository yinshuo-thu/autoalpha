import numpy as np
import pandas as pd

def winsorize(series, method='mad', k=5):
    if series.empty: return series
    if method == 'mad':
        unstacked = series.unstack('security_id')
        md = unstacked.median(axis=1)
        # Use abs() on the subtracted result
        mad = unstacked.sub(md, axis=0).abs().median(axis=1)
        eps = 1e-8
        lower = md - k * (mad + eps)
        upper = md + k * (mad + eps)
        clipped = unstacked.clip(lower=lower, upper=upper, axis=0)
        return clipped.stack('security_id').reorder_levels(series.index.names).sort_index()
    elif method == 'percentile':
        unstacked = series.unstack('security_id')
        q_low = unstacked.quantile(0.01, axis=1)
        q_high = unstacked.quantile(0.99, axis=1)
        clipped = unstacked.clip(lower=q_low, upper=q_high, axis=0)
        return clipped.stack('security_id').reorder_levels(series.index.names).sort_index()
    return series

def normalize(series, method='rank'):
    from factors.operators import cs_rank, cs_zscore, cs_demean
    if method == 'rank':
        # Convert scale to [-0.5, 0.5]
        r = cs_rank(series)
        return cs_demean(r)
    elif method == 'zscore':
        return cs_zscore(series)
    return series

def smooth(series, method='ewm', span=2):
    if method == 'ewm':
        if series.empty: return series
        unstacked = series.unstack('security_id')
        smoothed = unstacked.ewm(span=span).mean()
        return smoothed.stack('security_id').reorder_levels(series.index.names).sort_index()
    elif method == 'sma':
        if series.empty: return series
        unstacked = series.unstack('security_id')
        smoothed = unstacked.rolling(span, min_periods=1).mean()
        return smoothed.stack('security_id').reorder_levels(series.index.names).sort_index()
    return series

def clip(series, q=0.01):
    if series.empty: return series
    unstacked = series.unstack('security_id')
    arr = unstacked.values.copy()
    
    abs_sum = np.nansum(np.abs(arr), axis=1, keepdims=True)
    abs_sum[abs_sum == 0] = np.nan
    
    np.divide(arr, abs_sum, out=arr, where=~np.isnan(arr))
    np.clip(arr, -q, q, out=arr)
    
    capped = pd.DataFrame(arr, index=unstacked.index, columns=unstacked.columns)
    return capped.stack('security_id').reorder_levels(series.index.names).sort_index()

def apply_postprocess(series, config):
    if not config:
        return series
    
    res = series.copy()
    
    if 'winsorize' in config and config['winsorize']:
        cfg = config['winsorize']
        res = winsorize(res, method=cfg.get('method', 'mad'), k=cfg.get('k', 5))
        
    if 'normalize' in config and config['normalize']:
        cfg = config['normalize']
        res = normalize(res, method=cfg.get('method', 'rank'))
        
    if 'smooth' in config and config['smooth']:
        cfg = config['smooth']
        res = smooth(res, method=cfg.get('method', 'ewm'), span=cfg.get('span', 2))
        
    if 'clip' in config and config['clip']:
        cfg = config['clip']
        res = clip(res, q=cfg.get('q', 0.01))
        
    return res

# Default Templates mapped exactly from plan
TEMPLATES = {
    'stable_low_turnover': {
        'winsorize': {'method': 'mad', 'k': 5},
        'normalize': {'method': 'rank'},
        'smooth': {'method': 'ewm', 'span': 4},
        'clip': {'q': 0.01}
    },
    'aggressive_high_ic': {
        'winsorize': {'method': 'mad', 'k': 5},
        'normalize': {'method': 'zscore'},
        'smooth': {'method': 'ewm', 'span': 2},
        'clip': {'q': 0.05}
    }
}
