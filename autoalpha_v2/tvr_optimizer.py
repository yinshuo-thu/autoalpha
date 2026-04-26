"""
autoalpha_v2/tvr_optimizer.py

TVR reduction optimizer: when a factor fails screening due to high turnover,
try four smoothing combinations in order and return the first that passes.

Combinations (tried in order, cheapest first):
  1. EMA smoothing (span=10) — most reliable, minimal IC loss
  2. Persistence blend (EWM alpha=0.2) — direct position-mixing
  3. Extreme quantile + zero middle — preserve signal at tails only
  4. Rolling mean (window=15) — strongest TVR suppression
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# Smoothing combinations
# ─────────────────────────────────────────────────────────────────────────────

def _rerank(un: pd.DataFrame, original: pd.Series) -> pd.Series:
    """Cross-sectional percentile rank → [-0.5, 0.5], stacked back."""
    ranked = un.rank(axis=1, pct=True) - 0.5
    return (
        ranked.stack("security_id")
        .reorder_levels(original.index.names)
        .sort_index()
        .clip(-1.0, 1.0)
        .astype("float32")
    )


def combo_ema(alpha: pd.Series, span: int = 10) -> pd.Series:
    """Combination 1: EMA smoothing across time bars, then re-rank."""
    un = alpha.unstack("security_id")
    smoothed = un.ewm(span=span, adjust=True).mean()
    return _rerank(smoothed, alpha)


def combo_persistence(alpha: pd.Series, blend_alpha: float = 0.2) -> pd.Series:
    """Combination 2: Blend new signal with previous position (EWM alpha=blend_alpha)."""
    un = alpha.unstack("security_id")
    blended = un.ewm(alpha=blend_alpha, adjust=False).mean()
    return _rerank(blended, alpha)


def combo_extremes(alpha: pd.Series, q: float = 0.2) -> pd.Series:
    """Combination 3: Keep top/bottom q quantile, zero out the middle."""
    un = alpha.unstack("security_id")
    lo = un.quantile(q, axis=1)
    hi = un.quantile(1.0 - q, axis=1)
    mask = un.le(lo, axis=0) | un.ge(hi, axis=0)
    sparse = un.where(mask, 0.0)
    return _rerank(sparse, alpha)


def combo_rolling(alpha: pd.Series, window: int = 15) -> pd.Series:
    """Combination 4: Rolling mean smoothing + re-rank."""
    un = alpha.unstack("security_id")
    smoothed = un.rolling(window=window, min_periods=1).mean()
    return _rerank(smoothed, alpha)


# Ordered list: (name, function, kwargs)
_COMBOS = [
    ("ema_10",         combo_ema,          {"span": 10}),
    ("persistence_02", combo_persistence,  {"blend_alpha": 0.2}),
    ("extremes_q20",   combo_extremes,     {"q": 0.2}),
    ("rolling_15",     combo_rolling,      {"window": 15}),
]


# ─────────────────────────────────────────────────────────────────────────────
# Approximate TVR (no API call needed — used to quickly rank candidates)
# ─────────────────────────────────────────────────────────────────────────────

def approx_tvr(alpha: pd.Series) -> float:
    """
    Lightweight TVR proxy: mean absolute position change per bar,
    normalised by mean absolute position size. Correlates well with
    competition TVR without requiring an API call.
    """
    try:
        un = alpha.unstack("security_id").astype(float)
        diff = un.diff().abs().sum(axis=1)
        scale = un.abs().sum(axis=1).replace(0, np.nan)
        ratio = (diff / scale).dropna()
        if ratio.empty:
            return 9999.0
        return float(ratio.mean()) * 10000.0
    except Exception:
        return 9999.0


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def try_reduce_tvr(
    alpha_screen: pd.Series,
    hub: Any,
    screen_days: list[str],
    evaluate_fn: Any,          # pipeline.evaluate_alpha
    max_tvr: float = 380.0,
    min_ic: float = 0.10,
) -> tuple[pd.Series | None, dict | None, str]:
    """
    Try TVR-reduction combinations on the screened alpha.

    Returns:
        (smoothed_alpha, metrics, combo_name)  — first combo that achieves TVR < max_tvr
        (None, None, "")                        — if all combos fail
    """
    # Pre-rank candidates by approximate TVR (cheap, no API)
    candidates = []
    for name, fn, kwargs in _COMBOS:
        try:
            smoothed = fn(alpha_screen, **kwargs)
            est_tvr = approx_tvr(smoothed)
            candidates.append((est_tvr, name, fn, kwargs, smoothed))
        except Exception:
            pass
    candidates.sort(key=lambda x: x[0])  # try lowest estimated TVR first

    for est_tvr, name, fn, kwargs, smoothed in candidates:
        try:
            metrics = evaluate_fn(smoothed, hub, screen_days)
            tvr = float(metrics.get("Turnover", 0) or 0)
            ic = float(metrics.get("IC", 0) or 0)
            print(f"  [tvr-opt] {name}: TVR={tvr:.0f}  IC={ic:.3f}  (est_tvr={est_tvr:.0f})")
            if tvr <= max_tvr and ic >= min_ic:
                return smoothed, metrics, name
        except Exception as e:
            print(f"  [tvr-opt] {name}: error — {e}")

    return None, None, ""
