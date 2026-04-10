"""
factor_specs.py — Candidate Factor Template Library

Defines factor templates with parameter spaces for the autoresearch agent.
Each spec is a formula template that can be instantiated with specific parameters.
"""
import random
import itertools

# ─── Factor Template Categories ─────────────────────────────────────────────

FACTOR_TEMPLATES = [
    # ── Category: Mean Reversion ──
    {
        "name": "mr_zscore",
        "category": "mean_reversion",
        "formula_template": "cs_rank(-ts_zscore(close_trade_px, {window}))",
        "param_grid": {"window": [5, 8, 10, 15, 20, 30]},
        "postprocess_default": "stable_low_turnover",
        "description": "Mean reversion via Z-score of close price"
    },
    {
        "name": "mr_vwap_dev",
        "category": "mean_reversion",
        "formula_template": "cs_rank(-(close_trade_px / vwap - 1))",
        "param_grid": {},
        "postprocess_default": "stable_low_turnover",
        "description": "VWAP deviation reversal"
    },
    {
        "name": "mr_vwap_zscore",
        "category": "mean_reversion",
        "formula_template": "cs_rank(-ts_zscore(close_trade_px / vwap - 1, {window}))",
        "param_grid": {"window": [5, 10, 15, 20]},
        "postprocess_default": "stable_low_turnover",
        "description": "Z-score of VWAP deviation"
    },
    {
        "name": "mr_multi_horizon",
        "category": "mean_reversion",
        "formula_template": "cs_rank(-ts_zscore(close_trade_px, {w1})) + cs_rank(-ts_zscore(close_trade_px, {w2}))",
        "param_grid": {"w1": [5, 8, 10], "w2": [15, 20, 30]},
        "postprocess_default": "stable_low_turnover",
        "description": "Multi-horizon mean reversion blend"
    },
    {
        "name": "mr_decay_reversal",
        "category": "mean_reversion",
        "formula_template": "cs_rank(ts_decay_linear(-delta(close_trade_px, {delta_w}), {decay_w}))",
        "param_grid": {"delta_w": [1, 3, 5], "decay_w": [5, 10, 15]},
        "postprocess_default": "stable_low_turnover",
        "description": "Decay-weighted price reversal"
    },

    # ── Category: Momentum ──
    {
        "name": "mom_delta_rank",
        "category": "momentum",
        "formula_template": "cs_rank(delta(close_trade_px, {window}))",
        "param_grid": {"window": [1, 3, 5, 10, 15]},
        "postprocess_default": "stable_low_turnover",
        "description": "Simple ranked momentum"
    },
    {
        "name": "mom_decay",
        "category": "momentum",
        "formula_template": "cs_rank(ts_decay_linear(delta(close_trade_px, {delta_w}), {decay_w}))",
        "param_grid": {"delta_w": [1, 3, 5], "decay_w": [8, 10, 15, 20]},
        "postprocess_default": "stable_low_turnover",
        "description": "Decay-weighted momentum"
    },
    {
        "name": "mom_vol_confirm",
        "category": "momentum",
        "formula_template": "cs_rank(delta(close_trade_px, {window}) * safe_div(volume, ts_mean(volume, {vol_w})))",
        "param_grid": {"window": [1, 3, 5], "vol_w": [10, 16, 20]},
        "postprocess_default": "stable_low_turnover",
        "description": "Volume-confirmed momentum"
    },

    # ── Category: VWAP ──
    {
        "name": "vwap_deviation",
        "category": "vwap",
        "formula_template": "cs_rank(ts_decay_linear(-(close_trade_px / vwap - 1), {window}))",
        "param_grid": {"window": [5, 10, 15, 20]},
        "postprocess_default": "stable_low_turnover",
        "description": "Decayed VWAP deviation"
    },
    {
        "name": "vwap_rank_diff",
        "category": "vwap",
        "formula_template": "cs_rank(close_trade_px) - cs_rank(vwap)",
        "param_grid": {},
        "postprocess_default": "stable_low_turnover",
        "description": "Cross-sectional rank difference: close vs VWAP"
    },
    {
        "name": "vwap_ts_rank",
        "category": "vwap",
        "formula_template": "cs_rank(ts_rank(close_trade_px / vwap - 1, {window}))",
        "param_grid": {"window": [5, 10, 15, 20]},
        "postprocess_default": "stable_low_turnover",
        "description": "Time-series rank of VWAP deviation"
    },

    # ── Category: Volume ──
    {
        "name": "vol_surprise",
        "category": "volume",
        "formula_template": "cs_rank(ts_zscore(volume, {window}))",
        "param_grid": {"window": [5, 10, 15, 20]},
        "postprocess_default": "stable_low_turnover",
        "description": "Volume surprise Z-score"
    },
    {
        "name": "vol_price_div",
        "category": "volume",
        "formula_template": "cs_rank(-delta(close_trade_px, {pw}) * ts_rank(-volume, {vw}))",
        "param_grid": {"pw": [1, 3, 5], "vw": [5, 10, 15]},
        "postprocess_default": "stable_low_turnover",
        "description": "Price-volume divergence"
    },
    {
        "name": "vol_accel",
        "category": "volume",
        "formula_template": "cs_rank(delta(safe_div(volume, ts_mean(volume, {w1})), {w2}))",
        "param_grid": {"w1": [10, 16, 20], "w2": [1, 3, 5]},
        "postprocess_default": "stable_low_turnover",
        "description": "Volume acceleration"
    },
    {
        "name": "trade_count_anomaly",
        "category": "volume",
        "formula_template": "cs_rank(ts_zscore(trade_count, {window}))",
        "param_grid": {"window": [5, 10, 15, 20]},
        "postprocess_default": "stable_low_turnover",
        "description": "Trade count anomaly"
    },
    {
        "name": "dvolume_ratio",
        "category": "volume",
        "formula_template": "cs_rank(safe_div(dvolume, ts_mean(dvolume, {window})))",
        "param_grid": {"window": [5, 10, 16, 20]},
        "postprocess_default": "stable_low_turnover",
        "description": "Dollar volume ratio signal"
    },

    # ── Category: Volatility ──
    {
        "name": "vol_range",
        "category": "volatility",
        "formula_template": "cs_rank(-safe_div(high_trade_px - low_trade_px, ts_mean(high_trade_px - low_trade_px, {window})))",
        "param_grid": {"window": [5, 10, 15, 20]},
        "postprocess_default": "stable_low_turnover",
        "description": "Intraday range compression (reversal on low vol)"
    },
    {
        "name": "vol_change_rate",
        "category": "volatility",
        "formula_template": "cs_rank(delta(ts_std(close_trade_px, {w1}), {w2}))",
        "param_grid": {"w1": [5, 10, 15], "w2": [1, 3, 5]},
        "postprocess_default": "stable_low_turnover",
        "description": "Volatility change rate"
    },
    {
        "name": "vol_regime",
        "category": "volatility",
        "formula_template": "cs_rank(safe_div(ts_std(close_trade_px, {w_short}), ts_std(close_trade_px, {w_long})))",
        "param_grid": {"w_short": [5, 8], "w_long": [15, 20, 30]},
        "postprocess_default": "stable_low_turnover",
        "description": "Short/long volatility ratio"
    },

    # ── Category: Intraday Pattern ──
    {
        "name": "intra_close_open",
        "category": "intraday",
        "formula_template": "cs_rank(ts_decay_linear(close_trade_px - open_trade_px, {window}))",
        "param_grid": {"window": [5, 10, 15]},
        "postprocess_default": "stable_low_turnover",
        "description": "Close-to-open gap persistence"
    },
    {
        "name": "intra_position",
        "category": "intraday",
        "formula_template": "cs_rank(ts_decay_linear(safe_div(close_trade_px - low_trade_px, high_trade_px - low_trade_px + 1e-8), {window}))",
        "param_grid": {"window": [5, 10, 15]},
        "postprocess_default": "stable_low_turnover",
        "description": "Intraday price position within range"
    },
    {
        "name": "intra_range_vol",
        "category": "intraday",
        "formula_template": "cs_rank(safe_div(high_trade_px - low_trade_px, ts_mean(high_trade_px - low_trade_px, {window})) * sign(delta(dvolume, {dw})))",
        "param_grid": {"window": [5, 10, 15], "dw": [1, 3, 5]},
        "postprocess_default": "aggressive_high_ic",
        "description": "Range expansion with volume direction"
    },

    # ── Category: Cross-Sectional Rank ──
    {
        "name": "cs_close_rank_change",
        "category": "rank",
        "formula_template": "cs_rank(close_trade_px) - lag(cs_rank(close_trade_px), {window})",
        "param_grid": {"window": [1, 3, 5, 10]},
        "postprocess_default": "stable_low_turnover",
        "description": "Change in cross-sectional rank"
    },
    {
        "name": "cs_vol_price_rank_diff",
        "category": "rank",
        "formula_template": "cs_rank(volume) - cs_rank(abs(delta(close_trade_px, {window})))",
        "param_grid": {"window": [1, 3, 5]},
        "postprocess_default": "stable_low_turnover",
        "description": "Volume rank vs price-change rank divergence"
    },

    # ── Category: Multi-Window / Acceleration ──
    {
        "name": "accel_price",
        "category": "acceleration",
        "formula_template": "cs_rank(delta(delta(close_trade_px, {w1}), {w2}))",
        "param_grid": {"w1": [1, 3, 5], "w2": [1, 3, 5]},
        "postprocess_default": "stable_low_turnover",
        "description": "Price acceleration (second derivative)"
    },
    {
        "name": "ratio_short_long",
        "category": "acceleration",
        "formula_template": "cs_rank(safe_div(ts_mean(close_trade_px, {w_short}), ts_mean(close_trade_px, {w_long})) - 1)",
        "param_grid": {"w_short": [3, 5, 8], "w_long": [15, 20, 30]},
        "postprocess_default": "stable_low_turnover",
        "description": "Short/long moving average ratio"
    },
    {
        "name": "ts_rank_multi",
        "category": "acceleration",
        "formula_template": "cs_rank(ts_rank(delta(close_trade_px, {w1}), {w2}))",
        "param_grid": {"w1": [1, 3, 5], "w2": [10, 15, 20]},
        "postprocess_default": "stable_low_turnover",
        "description": "Time-series rank of momentum"
    },

    # ── Category: Composite ──
    {
        "name": "composite_mr_vol",
        "category": "composite",
        "formula_template": "cs_rank(-ts_zscore(close_trade_px, {mr_w})) + cs_rank(ts_zscore(volume, {vol_w}))",
        "param_grid": {"mr_w": [5, 10, 15], "vol_w": [5, 10, 15]},
        "postprocess_default": "stable_low_turnover",
        "description": "Mean reversion + volume surprise composite"
    },
    {
        "name": "composite_vwap_range",
        "category": "composite",
        "formula_template": "cs_rank(-(close_trade_px / vwap - 1)) * cs_rank(-safe_div(high_trade_px - low_trade_px, ts_mean(high_trade_px - low_trade_px, {window})))",
        "param_grid": {"window": [5, 10, 15]},
        "postprocess_default": "stable_low_turnover",
        "description": "VWAP reversion weighted by low-volatility regime"
    },
    {
        "name": "composite_decay_blend",
        "category": "composite",
        "formula_template": "cs_rank(ts_decay_linear(-delta(close_trade_px, {w1}), {w2}) + ts_decay_linear(-(close_trade_px / vwap - 1), {w3}))",
        "param_grid": {"w1": [1, 3, 5], "w2": [5, 10, 15], "w3": [5, 10, 15]},
        "postprocess_default": "stable_low_turnover",
        "description": "Blended decay reversal + VWAP signal"
    },
]


def get_all_specs():
    """Return all factor template specs."""
    return FACTOR_TEMPLATES


def get_spec_by_name(name):
    """Get a specific factor spec by name."""
    for spec in FACTOR_TEMPLATES:
        if spec['name'] == name:
            return spec
    return None


def instantiate_spec(spec, params=None):
    """
    Instantiate a factor spec with specific parameters.
    If params is None, randomly sample from param_grid.
    Returns: (formula_str, params_dict, run_id)
    """
    if not spec['param_grid']:
        return spec['formula_template'], {}, spec['name']

    if params is None:
        params = {}
        for key, values in spec['param_grid'].items():
            params[key] = random.choice(values)

    formula = spec['formula_template'].format(**params)
    param_str = "_".join(f"{k}{v}" for k, v in sorted(params.items()))
    run_id = f"{spec['name']}_{param_str}"

    return formula, params, run_id


def get_random_spec():
    """Pick a random spec and instantiate it."""
    spec = random.choice(FACTOR_TEMPLATES)
    return spec, *instantiate_spec(spec)


def get_all_instantiations(spec):
    """Generate all parameter combinations for a spec."""
    if not spec['param_grid']:
        formula, params, run_id = instantiate_spec(spec)
        return [(formula, params, run_id)]

    keys = sorted(spec['param_grid'].keys())
    values = [spec['param_grid'][k] for k in keys]
    results = []
    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        formula, _, run_id = instantiate_spec(spec, params)
        results.append((formula, params, run_id))
    return results


def mutate_spec(spec, params, metrics):
    """
    Mutate parameters based on evaluation feedback.
    Returns new params dict.
    """
    if not spec['param_grid']:
        return params

    new_params = dict(params)
    ic = metrics.get('IC', 0)
    tvr = metrics.get('Turnover', 0)
    ir = metrics.get('IR', 0)

    # Strategy: if turnover too high, increase windows; if IC too low, try different windows
    for key, values in spec['param_grid'].items():
        current = new_params.get(key, values[0])
        idx = values.index(current) if current in values else 0

        if tvr > 400 and idx < len(values) - 1:
            # Increase window to reduce turnover
            new_params[key] = values[min(idx + 1, len(values) - 1)]
        elif ic < 0.3 and ir < 0:
            # Try a different random value
            new_params[key] = random.choice(values)
        elif ic < 0.6:
            # Nudge slightly
            candidates = [i for i in range(len(values)) if i != idx]
            if candidates:
                new_params[key] = values[random.choice(candidates)]

    return new_params


def count_total_combinations():
    """Count total number of unique factor instances across all specs."""
    total = 0
    for spec in FACTOR_TEMPLATES:
        if not spec['param_grid']:
            total += 1
        else:
            keys = sorted(spec['param_grid'].keys())
            values = [spec['param_grid'][k] for k in keys]
            combos = 1
            for v in values:
                combos *= len(v)
            total += combos
    return total
