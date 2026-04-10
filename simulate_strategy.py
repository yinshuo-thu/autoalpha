"""
simulate_strategy.py — Strategy Backtest (Real Data)

Converts factor values or model predictions into a backtest.
Returns PnL, Sharpe, MaxDD, Turnover, and a time series.
"""
import pandas as pd
import numpy as np

def run_strategy_simulation(factor_series, data_hub, factor_name):
    """Run a simple long/short strategy based on factor exposures."""
    if factor_series is None or factor_series.empty:
        return {'status': 'error', 'reason': 'Empty factor series'}
        
    resp_series = data_hub.resp['resp']
    df = pd.DataFrame({'alpha': factor_series, 'resp': resp_series}).dropna()
    
    if len(df) < 100:
        return {'status': 'error', 'reason': 'Not enough overlapping data'}
        
    dates = df.index.get_level_values('date')
    
    # Simple cross-sectional ranking -> weights summing to 0, abs summing to 1
    # df = df.reset_index()
    def to_weights(group):
        if len(group) == 0: return group
        ranks = group.rank() - 1
        demeaned = ranks - ranks.mean()
        sum_abs = demeaned.abs().sum()
        if sum_abs == 0: return demeaned * 0
        return demeaned / sum_abs
        
    weights = df['alpha'].groupby('date').transform(to_weights)
    df['weight'] = weights
    
    # Returns
    df['pnl'] = df['weight'] * df['resp']
    
    # 1. Total Cumulative PnL Curve
    daily_pnl = df.groupby('date')['pnl'].sum()
    cum_pnl = daily_pnl.cumsum()
    
    # 2. Sharpe Ratio (assuming daily returns)
    mean_ret = daily_pnl.mean()
    std_ret = daily_pnl.std()
    sharpe = (mean_ret / std_ret) * np.sqrt(252) if std_ret > 0 else 0
    
    # 3. Max Drawdown
    running_max = cum_pnl.cummax()
    drawdown = cum_pnl - running_max
    max_dd = drawdown.min()
    
    # 4. Turnover
    daily_weights = df['weight'].unstack('security_id').fillna(0)
    turnover = daily_weights.diff().abs().sum(axis=1) / 2
    avg_turnover = turnover.mean()
    
    # 5. Hit Ratio
    hit_ratio = (daily_pnl > 0).mean()
    
    # Split metrics
    train_mask = daily_pnl.index <= '2022-12-31'
    val_mask = (daily_pnl.index >= '2023-01-01') & (daily_pnl.index <= '2023-12-31')
    test_mask = daily_pnl.index >= '2024-01-01'
    
    def calc_split(mask):
        dp = daily_pnl[mask]
        if len(dp) < 2: return {'sharpe': 0, 'pnl': 0, 'max_dd': 0}
        m = dp.mean()
        s = dp.std()
        sh = (m / s) * np.sqrt(252) if s > 0 else 0
        pnl = dp.sum()
        dd = (dp.cumsum() - dp.cumsum().cummax()).min()
        return {'sharpe': float(sh), 'pnl': float(pnl), 'max_dd': float(dd)}
        
    result = {
        'factor_name': factor_name,
        'status': 'success',
        'metrics': {
            'sharpe': float(sharpe),
            'max_drawdown': float(max_dd),
            'avg_turnover': float(avg_turnover),
            'hit_ratio': float(hit_ratio),
            'total_pnl': float(cum_pnl.iloc[-1]) if len(cum_pnl) > 0 else 0
        },
        'splits': {
            'train': calc_split(train_mask),
            'val': calc_split(val_mask),
            'test': calc_split(test_mask)
        },
        'time_series': {
            'cum_pnl': [{'date': str(d), 'pnl': float(v)} for d, v in cum_pnl.items() if np.isfinite(v)],
            'drawdown': [{'date': str(d), 'dd': float(v)} for d, v in drawdown.items() if np.isfinite(v)]
        }
    }
    
    return result
