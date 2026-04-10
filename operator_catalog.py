"""
operator_catalog.py — Complete whitelist of allowed DSL operators

Organized by category: cross-sectional, time-series, math, conditional, combination.
Each operator has name, params, description, example, and notes.
"""

OPERATORS = {
    # ── Cross-Sectional ──
    'rank':       {'category': 'cross_section', 'params': 'x', 'desc': 'Cross-sectional percentile rank [0,1]',
                   'example': 'rank(volume)', 'notes': 'Ranks across all securities at each timestamp'},
    'zscore':     {'category': 'cross_section', 'params': 'x', 'desc': 'Cross-sectional z-score',
                   'example': 'zscore(close_trade_px)', 'notes': '(x - mean) / std across securities'},
    'scale':      {'category': 'cross_section', 'params': 'x', 'desc': 'Scale to sum of abs = 1',
                   'example': 'scale(volume)', 'notes': 'Normalize to zero-mean, unit abs sum'},
    'demean':     {'category': 'cross_section', 'params': 'x', 'desc': 'Cross-sectional de-mean',
                   'example': 'demean(close_trade_px)', 'notes': 'Subtract cross-sectional mean'},
    'winsorize':  {'category': 'cross_section', 'params': 'x, p', 'desc': 'Winsorize at p-th percentile',
                   'example': 'winsorize(volume, 0.01)', 'notes': 'Clip tails at p and 1-p percentiles'},

    # ── Time-Series ──
    'delay':      {'category': 'time_series', 'params': 'x, n', 'desc': 'Lag value by n bars',
                   'example': 'delay(close_trade_px, 1)', 'notes': 'Same as shift(n)'},
    'delta':      {'category': 'time_series', 'params': 'x, n', 'desc': 'Change over n bars: x - delay(x, n)',
                   'example': 'delta(close_trade_px, 1)', 'notes': 'First difference'},
    'ts_mean':    {'category': 'time_series', 'params': 'x, n', 'desc': 'Rolling mean over n bars',
                   'example': 'ts_mean(volume, 20)', 'notes': 'Simple moving average'},
    'ts_std':     {'category': 'time_series', 'params': 'x, n', 'desc': 'Rolling std over n bars',
                   'example': 'ts_std(close_trade_px, 20)', 'notes': 'Rolling volatility'},
    'ts_sum':     {'category': 'time_series', 'params': 'x, n', 'desc': 'Rolling sum over n bars',
                   'example': 'ts_sum(volume, 10)', 'notes': 'Cumulative volume'},
    'ts_rank':    {'category': 'time_series', 'params': 'x, n', 'desc': 'Time-series rank over n bars',
                   'example': 'ts_rank(close_trade_px, 20)', 'notes': 'Percentile rank in lookback window'},
    'ts_min':     {'category': 'time_series', 'params': 'x, n', 'desc': 'Rolling minimum over n bars',
                   'example': 'ts_min(low_trade_px, 20)', 'notes': 'Lowest value in window'},
    'ts_max':     {'category': 'time_series', 'params': 'x, n', 'desc': 'Rolling maximum over n bars',
                   'example': 'ts_max(high_trade_px, 20)', 'notes': 'Highest value in window'},
    'ts_corr':    {'category': 'time_series', 'params': 'x, y, n', 'desc': 'Rolling correlation',
                   'example': 'ts_corr(close_trade_px, volume, 20)', 'notes': 'Pearson correlation in window'},
    'ts_cov':     {'category': 'time_series', 'params': 'x, y, n', 'desc': 'Rolling covariance',
                   'example': 'ts_cov(close_trade_px, volume, 20)', 'notes': 'Covariance in window'},
    'ts_zscore':  {'category': 'time_series', 'params': 'x, n', 'desc': 'Time-series z-score: (x - ts_mean) / ts_std',
                   'example': 'ts_zscore(volume, 20)', 'notes': 'Standardized within lookback'},
    'decay_linear': {'category': 'time_series', 'params': 'x, n', 'desc': 'Linearly-weighted moving average',
                     'example': 'decay_linear(close_trade_px, 10)', 'notes': 'Recent bars weighted more'},

    # ── Math ──
    'add':        {'category': 'math', 'params': 'x, y', 'desc': 'Addition: x + y',
                   'example': 'add(volume, dvolume)', 'notes': 'Element-wise add'},
    'sub':        {'category': 'math', 'params': 'x, y', 'desc': 'Subtraction: x - y',
                   'example': 'sub(high_trade_px, low_trade_px)', 'notes': 'Element-wise subtract'},
    'mul':        {'category': 'math', 'params': 'x, y', 'desc': 'Multiplication: x * y',
                   'example': 'mul(close_trade_px, volume)', 'notes': 'Element-wise multiply'},
    'div':        {'category': 'math', 'params': 'x, y', 'desc': 'Safe division: x / y (eps-protected)',
                   'example': 'div(dvolume, volume)', 'notes': 'Avoids division by zero'},
    'abs':        {'category': 'math', 'params': 'x', 'desc': 'Absolute value',
                   'example': 'abs(delta(close_trade_px, 1))', 'notes': ''},
    'log':        {'category': 'math', 'params': 'x', 'desc': 'Natural logarithm (log of abs)',
                   'example': 'log(volume)', 'notes': 'Uses log(abs(x) + 1) for safety'},
    'signed_log': {'category': 'math', 'params': 'x', 'desc': 'Sign-preserving log',
                   'example': 'signed_log(delta(close_trade_px, 1))', 'notes': 'sign(x) * log(1 + abs(x))'},
    'sqrt':       {'category': 'math', 'params': 'x', 'desc': 'Square root (of abs)',
                   'example': 'sqrt(volume)', 'notes': 'sqrt(abs(x))'},
    'pow':        {'category': 'math', 'params': 'x, p', 'desc': 'Power: sign(x) * abs(x)^p',
                   'example': 'pow(delta(close_trade_px, 1), 0.5)', 'notes': 'Sign-preserving power'},
    'clip':       {'category': 'math', 'params': 'x, a, b', 'desc': 'Clip values to [a, b]',
                   'example': 'clip(zscore(volume), -3, 3)', 'notes': 'Bounds outliers'},
    'neg':        {'category': 'math', 'params': 'x', 'desc': 'Negate: -x',
                   'example': 'neg(delta(close_trade_px, 1))', 'notes': 'Flip sign'},

    # ── Conditional ──
    'ifelse':     {'category': 'conditional', 'params': 'cond, a, b', 'desc': 'If cond > 0 then a else b',
                   'example': 'ifelse(gt(volume, ts_mean(volume, 20)), close_trade_px, 0)', 'notes': ''},
    'gt':         {'category': 'conditional', 'params': 'x, y', 'desc': 'Greater than: x > y → 1 else 0',
                   'example': 'gt(volume, ts_mean(volume, 20))', 'notes': 'Boolean mask'},
    'lt':         {'category': 'conditional', 'params': 'x, y', 'desc': 'Less than: x < y → 1 else 0',
                   'example': 'lt(close_trade_px, vwap)', 'notes': 'Boolean mask'},
    'and_op':     {'category': 'conditional', 'params': 'a, b', 'desc': 'Logical AND',
                   'example': 'and_op(gt(volume, 1000), lt(close_trade_px, vwap))', 'notes': ''},
    'or_op':      {'category': 'conditional', 'params': 'a, b', 'desc': 'Logical OR',
                   'example': 'or_op(gt(volume, 10000), gt(trade_count, 100))', 'notes': ''},
    'not_op':     {'category': 'conditional', 'params': 'a', 'desc': 'Logical NOT',
                   'example': 'not_op(gt(volume, 1000))', 'notes': ''},

    # ── Combination ──
    'mean_of':    {'category': 'combination', 'params': 'x1, x2, ...', 'desc': 'Mean of multiple signals',
                   'example': 'mean_of(rank(volume), rank(dvolume))', 'notes': 'Equal-weighted average'},
    'weighted_sum': {'category': 'combination', 'params': 'w1, x1, w2, x2, ...', 'desc': 'Weighted sum',
                     'example': 'weighted_sum(0.7, rank(volume), 0.3, rank(dvolume))', 'notes': ''},
    'combine_rank': {'category': 'combination', 'params': 'x1, x2, ...', 'desc': 'Rank each then average',
                     'example': 'combine_rank(volume, dvolume)', 'notes': 'Each signal ranked then averaged'},
}

# Allowed operator names set
ALLOWED_OPERATORS = set(OPERATORS.keys())

# Also allow infix-style expressions using these binary operators
INFIX_OPERATORS = {'+': 'add', '-': 'sub', '*': 'mul', '/': 'div'}


def get_operators_by_category():
    """Group operators by category for frontend display."""
    groups = {}
    for name, meta in OPERATORS.items():
        cat = meta['category']
        if cat not in groups:
            groups[cat] = []
        groups[cat].append({'name': name, **meta})
    return groups


def get_operator_list():
    """Flat list of all operators for frontend."""
    return [{'name': name, **meta} for name, meta in OPERATORS.items()]


def search_operators(query):
    """Search operators by name or description."""
    query = query.lower()
    return [{'name': n, **m} for n, m in OPERATORS.items()
            if query in n.lower() or query in m['desc'].lower()]
