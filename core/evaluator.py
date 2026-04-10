import numpy as np
import pandas as pd

def apply_trading_restriction(alpha, restriction):
    # Restriction: 0 = none, 1 = short res, 2 = long res, 3 = both
    if restriction is None or restriction.empty:
        return alpha
    
    df = pd.DataFrame({'alpha': alpha, 'rest': restriction})
    df['rest'] = df['rest'].fillna(0)
    
    mask_short = (df['rest'] == 1) & (df['alpha'] < 0)
    mask_long = (df['rest'] == 2) & (df['alpha'] > 0)
    mask_both = (df['rest'] == 3)
    
    df.loc[mask_short | mask_long | mask_both, 'alpha'] = np.nan
    return df['alpha']

def calc_bar_ic(alpha, resp):
    df = pd.DataFrame({'alpha': alpha, 'resp': resp}).dropna()
    if df.empty: return pd.Series(dtype=float)
    
    alpha_un = df['alpha'].unstack('security_id')
    resp_un = df['resp'].unstack('security_id')
    
    return alpha_un.corrwith(resp_un, axis=1, method='pearson')

def calc_daily_ic(bar_ic):
    return bar_ic.groupby('date').mean()

def calc_ir(daily_ic):
    m = daily_ic.mean()
    s = daily_ic.std()
    if pd.isna(s) or s == 0: return 0.0
    return (m / s) * np.sqrt(252)

def calc_turnover(alpha):
    unstacked = alpha.unstack('security_id')
    diff_abs = unstacked.diff().abs().sum(axis=1)
    tot_abs = unstacked.abs().sum(axis=1)
    tvr_bar = diff_abs / tot_abs.replace(0, np.nan)
    return tvr_bar.groupby('date').sum()

def calc_book_stats(alpha):
    df = alpha.to_frame('alpha').dropna()
    if df.empty: return 0, 0, 0, 0
    
    unb = df['alpha'].unstack('security_id')
    abs_sum = unb.abs().sum(axis=1)
    weights = unb.div(abs_sum.replace(0, np.nan), axis=0) * 10_000
    
    max_w_bar = weights.max(axis=1)
    min_w_bar = weights.min(axis=1)
    
    max_w_bar_clean = max_w_bar.dropna()
    min_w_bar_clean = min_w_bar.dropna()
    
    maxx = max_w_bar_clean.max() if not max_w_bar_clean.empty else 0
    minn = min_w_bar_clean.min() if not min_w_bar_clean.empty else 0
    
    max_d = max_w_bar_clean.groupby('date').max().mean() if not max_w_bar_clean.empty else 0
    min_d = min_w_bar_clean.groupby('date').min().mean() if not min_w_bar_clean.empty else 0
    
    return maxx, minn, max_d, min_d

def calc_rank_ic(alpha, resp):
    """Spearman rank correlation (rank IC)."""
    df = pd.DataFrame({'alpha': alpha, 'resp': resp}).dropna()
    if df.empty:
        return pd.Series(dtype=float)
    alpha_un = df['alpha'].unstack('security_id')
    resp_un = df['resp'].unstack('security_id')
    # Rank both sides then correlate
    alpha_ranked = alpha_un.rank(axis=1)
    resp_ranked = resp_un.rank(axis=1)
    return alpha_ranked.corrwith(resp_ranked, axis=1, method='pearson')

def calc_monthly_ic(daily_ic):
    """Monthly mean IC."""
    idx = pd.to_datetime(daily_ic.index)
    monthly = daily_ic.groupby([idx.year, idx.month]).mean()
    if not isinstance(monthly.index, pd.MultiIndex):
        monthly.index = pd.MultiIndex.from_tuples(monthly.index, names=['year', 'month'])
    return monthly

def calc_rolling_ic(daily_ic, window=20):
    """Rolling IC with specified window."""
    return daily_ic.rolling(window, min_periods=max(1, window // 2)).mean()

def calc_positive_ic_ratio(daily_ic):
    """Fraction of days with positive IC."""
    if daily_ic.empty:
        return 0.0
    return float((daily_ic > 0).sum() / len(daily_ic))

def calc_monthly_ir(daily_ic):
    """Monthly IR values."""
    idx = pd.to_datetime(daily_ic.index)
    result = {}
    for (y, m), grp in daily_ic.groupby([idx.year, idx.month]):
        m_ic = grp.mean()
        m_std = grp.std()
        ir = (m_ic / m_std * np.sqrt(252)) if (pd.notnull(m_std) and m_std > 0) else 0.0
        result[f"{y}-{m:02d}"] = {'IC': float(m_ic * 100) if pd.notnull(m_ic) else 0,
                                   'IR': float(ir) if pd.notnull(ir) else 0}
    return result

def calc_stability_score(daily_ic):
    """Monthly stability score: fraction of months with positive mean IC."""
    idx = pd.to_datetime(daily_ic.index)
    monthly_means = daily_ic.groupby([idx.year, idx.month]).mean()
    if monthly_means.empty:
        return 0.0
    return float((monthly_means > 0).sum() / len(monthly_means))


def calc_bar_ic_wide(alpha_un, resp_un):
    return alpha_un.corrwith(resp_un, axis=1, method='pearson')

def calc_turnover_wide(alpha_un):
    diff_abs = alpha_un.diff().abs().sum(axis=1)
    tot_abs = alpha_un.abs().sum(axis=1)
    tvr_bar = diff_abs / tot_abs.replace(0, np.nan)
    return tvr_bar.groupby('date').sum()

def calc_book_stats_wide(alpha_un):
    df = alpha_un.dropna(how='all')
    if df.empty: return 0, 0, 0, 0
    abs_sum = df.abs().sum(axis=1)
    weights = df.div(abs_sum.replace(0, np.nan), axis=0) * 10_000
    max_w_bar_clean = weights.max(axis=1).dropna()
    min_w_bar_clean = weights.min(axis=1).dropna()
    maxx = max_w_bar_clean.max() if not max_w_bar_clean.empty else 0
    minn = min_w_bar_clean.min() if not min_w_bar_clean.empty else 0
    max_d = max_w_bar_clean.groupby('date').max().mean() if not max_w_bar_clean.empty else 0
    min_d = min_w_bar_clean.groupby('date').min().mean() if not min_w_bar_clean.empty else 0
    return maxx, minn, max_d, min_d

def calc_rank_ic_wide(alpha_un, resp_un):
    alpha_ranked = alpha_un.rank(axis=1)
    resp_ranked = resp_un.rank(axis=1)
    return alpha_ranked.corrwith(resp_ranked, axis=1, method='pearson')

def evaluate_official(alpha, resp, restriction):
    alpha_tradeable = apply_trading_restriction(alpha, restriction)
    
    alpha_tr_un = alpha_tradeable.unstack('security_id')
    alpha_un = alpha.unstack('security_id')
    resp_un = resp.unstack('security_id')
    
    bar_ic = calc_bar_ic_wide(alpha_tr_un, resp_un)
    daily_ic = calc_daily_ic(bar_ic)
    
    ic_raw = daily_ic.mean()
    ic_bps = ic_raw * 100
    ir = calc_ir(daily_ic)
    
    daily_tvr = calc_turnover_wide(alpha_un)
    tvr = daily_tvr.mean()
    
    maxx, minn, max_d, min_d = calc_book_stats_wide(alpha_un)
    
    gate_ic = ic_raw > 0.006
    gate_ir = ir > 2.5
    gate_tvr = tvr < 400
    gate_conc = (maxx < 50) and (abs(minn) < 50) and (max_d < 20) and (abs(min_d) < 20)
    
    pass_gates = gate_ic and gate_ir and gate_tvr and gate_conc
    
    score = 0
    if pass_gates:
        score = max(0, (ic_raw - 0.0005 * tvr)) * np.sqrt(ir) * 100
        
    return {
        'IC': float(ic_bps) if pd.notnull(ic_bps) else 0,
        'IR': float(ir) if pd.notnull(ir) else 0,
        'Turnover': float(tvr) if pd.notnull(tvr) else 0,
        'maxx': float(maxx) if pd.notnull(maxx) else 0,
        'minn': float(minn) if pd.notnull(minn) else 0,
        'max_mean': float(max_d) if pd.notnull(max_d) else 0,
        'min_mean': float(min_d) if pd.notnull(min_d) else 0,
        'Score': float(score),
        'PassGates': bool(pass_gates),
        'GatesDetail': {
            'IC': bool(gate_ic), 'IR': bool(gate_ir), 'Turnover': bool(gate_tvr), 'Concentration': bool(gate_conc)
        }
    }

def evaluate_research(alpha, resp, restriction, metrics_official):
    alpha_tradeable = apply_trading_restriction(alpha, restriction)
    alpha_tr_un = alpha_tradeable.unstack('security_id')
    alpha_un = alpha.unstack('security_id')
    resp_un = resp.unstack('security_id')

    bar_ic = calc_bar_ic_wide(alpha_tr_un, resp_un)
    daily_ic = calc_daily_ic(bar_ic)
    daily_tvr = calc_turnover_wide(alpha_un)
    
    bar_rank_ic = calc_rank_ic_wide(alpha_tr_un, resp_un)
    daily_rank_ic = calc_daily_ic(bar_rank_ic)
    rank_ic = daily_rank_ic.mean() * 100 if not daily_rank_ic.empty else 0.0
    
    monthly_ic = calc_monthly_ic(daily_ic)
    monthly_ir_data = calc_monthly_ir(daily_ic)
    rolling_ic = calc_rolling_ic(daily_ic, window=20)
    stability_score = calc_stability_score(daily_ic)
    positive_ic_ratio = calc_positive_ic_ratio(daily_ic)
    
    df_ic = daily_ic.to_frame('IC')
    df_ic['year'] = pd.to_datetime(df_ic.index).year
    df_tvr = daily_tvr.to_frame('tvr')
    df_tvr['year'] = pd.to_datetime(df_tvr.index).year
    
    yearly = {}
    for y in sorted(df_ic['year'].unique()):
        y_ic = df_ic[df_ic['year'] == y]['IC'].mean() * 100
        y_ir = calc_ir(df_ic[df_ic['year'] == y]['IC'])
        y_tvr = df_tvr[df_tvr['year'] == y]['tvr'].mean()
        yearly[str(y)] = {
            'IC': float(y_ic) if pd.notnull(y_ic) else 0,
            'IR': float(y_ir) if pd.notnull(y_ir) else 0,
            'Turnover': float(y_tvr) if pd.notnull(y_tvr) else 0
        }
    
    ic_bps = metrics_official.get('IC', 0)
    tvr_val = metrics_official.get('Turnover', 0)
    ic_minus_tvr = (ic_bps / 100.0) - 0.0005 * tvr_val
        
    return {
        'overall': metrics_official,
        'rank_ic': float(rank_ic) if pd.notnull(rank_ic) else 0,
        'ic_minus_tvr': float(ic_minus_tvr),
        'stability_score': float(stability_score),
        'positive_ic_ratio': float(positive_ic_ratio),
        'yearly': yearly,
        'monthly_ir': monthly_ir_data,
        'daily_ic': daily_ic,
        'daily_rank_ic': daily_rank_ic,
        'bar_ic': bar_ic,
        'daily_tvr': daily_tvr,
        'rolling_ic': rolling_ic
    }

class Evaluator:
    @staticmethod
    def run(alpha, resp, restriction):
        off = evaluate_official(alpha, resp, restriction)
        return evaluate_research(alpha, resp, restriction, off)

