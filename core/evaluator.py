import numpy as np
import pandas as pd

from core.submission import ALLOWED_UTC_TIMES

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

def filter_submission_times_wide(frame):
    if frame is None or frame.empty:
        return frame
    datetimes = pd.to_datetime(frame.index.get_level_values('datetime'))
    mask = datetimes.strftime('%H:%M:%S').isin(ALLOWED_UTC_TIMES)
    return frame[mask]

def apply_trading_restriction_wide(alpha_un, restriction_un):
    if restriction_un is None or restriction_un.empty:
        return alpha_un

    restriction_aligned = restriction_un.reindex_like(alpha_un).fillna(0)
    blocked = (
        ((restriction_aligned == 1) & (alpha_un < 0))
        | ((restriction_aligned == 2) & (alpha_un > 0))
        | (restriction_aligned == 3)
    )
    return alpha_un.mask(blocked)

def calc_turnover_submission_wide(alpha_un):
    df = alpha_un.dropna(how='all')
    if df.empty:
        return pd.Series(dtype=float), pd.DataFrame()

    abs_sum = df.abs().sum(axis=1)
    weights = df.div(abs_sum.replace(0, np.nan), axis=0) * 10_000

    turnover_bar_bps = weights.diff().abs().sum(axis=1)
    date_vals = pd.to_datetime(weights.index.get_level_values('date'))
    same_day_prev = np.r_[False, date_vals[1:] == date_vals[:-1]]
    turnover_bar_bps = turnover_bar_bps.where(same_day_prev)

    # Official-like口径更接近平台结果：
    # 先转成书面权重(bps)，再按日取“bar 平均换手”，最后除以 10 转成百分数。
    daily_tvr = turnover_bar_bps.groupby('date').mean() / 10.0
    return daily_tvr, weights

def calc_submission_position_stats_wide(weights):
    if weights is None or weights.empty:
        return {
            'bl': 0.0,
            'bs': 0.0,
            'nl': 0.0,
            'ns': 0.0,
            'nt': 0.0,
            'nd': 0.0,
            'max': 0.0,
            'min': 0.0,
        }

    long_weights = weights.where(weights > 0)

    long_book = weights.clip(lower=0).sum(axis=1) / 5000.0
    short_book = (-weights.clip(upper=0)).sum(axis=1) / 5000.0
    long_max_bar = long_weights.max(axis=1).dropna()
    long_min_bar = long_weights.min(axis=1).dropna()

    return {
        'bl': float(long_book.mean()) if not long_book.empty else 0.0,
        'bs': float(short_book.mean()) if not short_book.empty else 0.0,
        'nl': float((weights > 0).sum(axis=1).mean()) if not weights.empty else 0.0,
        'ns': float((weights < 0).sum(axis=1).mean()) if not weights.empty else 0.0,
        'nt': float((weights != 0).sum(axis=1).mean()) if not weights.empty else 0.0,
        'nd': float(pd.to_datetime(weights.index.get_level_values('date')).nunique()),
        'max': float(long_max_bar.groupby('date').max().mean()) if not long_max_bar.empty else 0.0,
        'min': float(long_min_bar.groupby('date').min().mean()) if not long_min_bar.empty else 0.0,
    }

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
        'unstacked': {
            'alpha_tr_un': alpha_tr_un,
            'alpha_un': alpha_un,
            'resp_un': resp_un
        },
        'GatesDetail': {
            'IC': bool(gate_ic), 'IR': bool(gate_ir), 'Turnover': bool(gate_tvr), 'Concentration': bool(gate_conc)
        }
    }

def evaluate_submission_like_wide(alpha_un, resp_un, restriction_un):
    alpha_sub = filter_submission_times_wide(alpha_un)
    resp_sub = filter_submission_times_wide(resp_un).reindex_like(alpha_sub)
    restriction_sub = filter_submission_times_wide(restriction_un).reindex_like(alpha_sub)

    alpha_tr_un = apply_trading_restriction_wide(alpha_sub, restriction_sub)

    bar_ic = calc_bar_ic_wide(alpha_tr_un, resp_sub)
    daily_ic = calc_daily_ic(bar_ic)

    ic_display = daily_ic.mean() * 100 if not daily_ic.empty else 0.0
    ir = calc_ir(daily_ic)

    daily_tvr, weights = calc_turnover_submission_wide(alpha_sub)
    tvr = daily_tvr.mean() if not daily_tvr.empty else 0.0

    maxx, minn, max_d, min_d = calc_book_stats_wide(alpha_sub)
    pos_stats = calc_submission_position_stats_wide(weights)

    # ic_display 与导出的 IC 同量级（~1e-2）；与 score 中 (IC - 0.0005*Tvr) 一致，阈值用 0.006。
    gate_ic = ic_display > 0.006
    gate_ir = ir > 2.5
    gate_tvr = tvr < 400
    gate_conc = (maxx < 50) and (abs(minn) < 50) and (max_d < 20) and (abs(min_d) < 20)
    pass_gates = gate_ic and gate_ir and gate_tvr and gate_conc

    score = 0.0
    if pass_gates:
        # 平台结果中的 IC 使用展示单位（mean daily corr * 100），不是小数原值。
        score = max(0.0, ic_display - 0.0005 * tvr) * np.sqrt(ir) * 100

    result_preview = {
        'IC': float(ic_display) if pd.notnull(ic_display) else 0.0,
        'IR': float(ir) if pd.notnull(ir) else 0.0,
        'tvr': float(tvr) if pd.notnull(tvr) else 0.0,
        'bl': float(pos_stats['bl']),
        'bs': float(pos_stats['bs']),
        'nl': float(pos_stats['nl']),
        'ns': float(pos_stats['ns']),
        'nt': float(pos_stats['nt']),
        'nd': float(pos_stats['nd']),
        'maxx': float(maxx) if pd.notnull(maxx) else 0.0,
        'max': float(pos_stats['max']),
        'minn': float(abs(minn)) if pd.notnull(minn) else 0.0,
        'min': float(pos_stats['min']),
        'score': float(score),
    }

    return {
        'IC': float(ic_display) if pd.notnull(ic_display) else 0.0,
        'IR': float(ir) if pd.notnull(ir) else 0.0,
        'Turnover': float(tvr) if pd.notnull(tvr) else 0.0,
        'Score': float(score),
        'PassGates': bool(pass_gates),
        'maxx': float(maxx) if pd.notnull(maxx) else 0.0,
        'minn': float(abs(minn)) if pd.notnull(minn) else 0.0,
        'max_mean': float(max_d) if pd.notnull(max_d) else 0.0,
        'min_mean': float(abs(min_d)) if pd.notnull(min_d) else 0.0,
        'score_formula': 'score = (IC - 0.0005 * Turnover) * sqrt(IR) * 100',
        'metric_mode': 'submission_like',
        'GatesDetail': {
            'IC': bool(gate_ic),
            'IR': bool(gate_ir),
            'Turnover': bool(gate_tvr),
            'Concentration': bool(gate_conc),
        },
        'result_preview': result_preview,
    }

def evaluate_research(metrics_official):
    un = metrics_official.pop('unstacked')
    alpha_tr_un = un['alpha_tr_un']
    alpha_un = un['alpha_un']
    resp_un = un['resp_un']

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
        return evaluate_research(off)

    @staticmethod
    def run_submission_like(alpha, resp, restriction):
        alpha_un = alpha.unstack('security_id')
        resp_un = resp.unstack('security_id')
        restriction_un = restriction.unstack('security_id') if restriction is not None and not restriction.empty else pd.DataFrame(index=alpha_un.index, columns=alpha_un.columns)
        return evaluate_submission_like_wide(alpha_un, resp_un, restriction_un)
