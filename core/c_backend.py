import numpy as np
import numba
from numba import njit, prange

@njit(parallel=True, fastmath=True, cache=True)
def c_ts_decay_linear_2d(arr_in, window):
    """
    Computes expanding weighted moving average over Axis 0 (Time) for each parallel Axis 1 (Security).
    arr_in shape: (T, N)
    Utilizes raw multi-threaded block computing.
    """
    T, N = arr_in.shape
    out = np.full((T, N), np.nan)
    
    # Pre-calculate linear weights
    weights = np.empty(window, dtype=np.float64)
    w_sum = 0.0
    for i in range(window):
        weights[i] = float(i + 1)
        w_sum += weights[i]
    for i in range(window):
        weights[i] /= w_sum
        
    for j in prange(N):
        for i in range(window - 1, T):
            valid = True
            v_sum = 0.0
            for k in range(window):
                val = arr_in[i - window + 1 + k, j]
                if np.isnan(val):
                    valid = False
                    break
                v_sum += val * weights[k]
            if valid:
                out[i, j] = v_sum
    return out

@njit(parallel=True, fastmath=True, cache=True)
def c_ts_rank_2d(arr_in, window):
    """
    Computes rolling cross-time rank (pct) for each security using pure C machine code.
    """
    T, N = arr_in.shape
    out = np.full((T, N), np.nan)
    min_periods = window // 2
    
    for j in prange(N):
        for i in range(T):
            start = max(0, i - window + 1)
            valid_cnt = 0
            for k in range(start, i + 1):
                if not np.isnan(arr_in[k, j]):
                    valid_cnt += 1
            
            if valid_cnt >= min_periods:
                curr_val = arr_in[i, j]
                if not np.isnan(curr_val):
                    less_cnt = 0
                    equal_cnt = 0
                    for k in range(start, i + 1):
                        v = arr_in[k, j]
                        if not np.isnan(v):
                            if v < curr_val:
                                less_cnt += 1
                            elif v == curr_val:
                                equal_cnt += 1
                    rank = less_cnt + (equal_cnt + 1.0) / 2.0
                    out[i, j] = rank / valid_cnt
    return out
