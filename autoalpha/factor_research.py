"""
autoalpha/factor_research.py

Automated factor research analysis script.
Called after each factor is evaluated in pipeline.py.

Outputs per factor (in autoalpha/research/<run_id>/):
  - report.json  : IC, decay, alpha distribution stats
  - report.md    : Human-readable markdown summary
  - analysis.png : 4-panel chart (IC decay, distribution, metrics, formula)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

RESEARCH_DIR = Path(__file__).resolve().parent / "research"

ALLOWED_UTC_TIMES = {
    "01:45:00", "02:00:00", "02:15:00", "02:30:00",
    "02:45:00", "03:00:00", "03:15:00", "03:30:00",
    "05:15:00", "05:30:00", "05:45:00", "06:00:00",
    "06:15:00", "06:30:00", "06:45:00", "07:00:00",
}


def _try_plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        return None


def _filter_allowed(series: pd.Series) -> pd.Series:
    """Keep only submission bar timestamps."""
    dt_vals = series.index.get_level_values("datetime")
    times = pd.to_datetime(dt_vals).strftime("%H:%M:%S")
    mask = pd.Series(times, index=series.index).isin(ALLOWED_UTC_TIMES)
    return series.loc[mask]


def _compute_ic_decay(
    alpha_wide: pd.DataFrame,
    resp_wide: pd.DataFrame,
    rest_wide: pd.DataFrame,
    max_lag: int = 10,
) -> Dict[int, float]:
    """IC at lag 0..max_lag (bars). Measures signal persistence."""
    decay: Dict[int, float] = {}
    for lag in range(max_lag + 1):
        a_lagged = alpha_wide.shift(lag)
        ic_vals: List[float] = []
        for date in a_lagged.index:
            if date not in resp_wide.index:
                continue
            a = a_lagged.loc[date]
            r = resp_wide.loc[date]
            mask = (rest_wide.loc[date] == 0) if date in rest_wide.index else pd.Series(True, index=a.index)
            valid = a.notna() & r.notna() & mask.astype(bool)
            if valid.sum() < 20:
                continue
            ic = a[valid].corr(r[valid])
            if not np.isnan(ic):
                ic_vals.append(ic * 100)
        decay[lag] = float(np.mean(ic_vals)) if ic_vals else 0.0
    return decay


def _compute_alpha_stats(alpha: pd.Series) -> Dict[str, float]:
    """Cross-sectional and temporal statistics of alpha values."""
    vals = alpha.dropna()
    if len(vals) == 0:
        return {}
    return {
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "skew": float(vals.skew()),
        "kurt": float(vals.kurtosis()),
        "min": float(vals.min()),
        "max": float(vals.max()),
        "pct_positive": float((vals > 0).mean()),
        "pct_in_bounds": float(((vals >= -0.5) & (vals <= 0.5)).mean()),
    }


def _compute_daily_ic(
    alpha_wide: pd.DataFrame,
    resp_wide: pd.DataFrame,
    rest_wide: pd.DataFrame,
) -> pd.Series:
    """Daily IC series for rolling / cumulative analysis."""
    daily_ics: Dict[Any, float] = {}
    for date in alpha_wide.index:
        if date not in resp_wide.index:
            continue
        a = alpha_wide.loc[date]
        r = resp_wide.loc[date]
        mask = (rest_wide.loc[date] == 0) if date in rest_wide.index else pd.Series(True, index=a.index)
        valid = a.notna() & r.notna() & mask.astype(bool)
        if valid.sum() < 20:
            continue
        ic = a[valid].corr(r[valid])
        if not np.isnan(ic):
            daily_ics[date] = ic * 100
    return pd.Series(daily_ics, name="daily_IC")


def _generate_plots(
    run_id: str,
    formula: str,
    metrics: Dict[str, Any],
    alpha_stats: Dict[str, float],
    ic_decay: Dict[int, float],
    daily_ic: pd.Series,
    out_dir: Path,
    plt: Any,
) -> None:
    """Generate 4-panel analysis chart."""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Factor Analysis — {run_id}", fontsize=13, fontweight="bold")

        # Panel 1: IC Decay
        ax = axes[0, 0]
        lags = list(ic_decay.keys())
        vals = [ic_decay[l] for l in lags]
        colors = ["#2196F3" if v >= 0 else "#F44336" for v in vals]
        ax.bar(lags, vals, color=colors, width=0.6)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Lag (bars)")
        ax.set_ylabel("Mean IC×100")
        ax.set_title("IC Decay by Lag")
        ax.set_xticks(lags)
        if vals:
            ic0 = ic_decay.get(0, 0)
            halflife = next((l for l, v in ic_decay.items() if l > 0 and abs(v) < abs(ic0) * 0.5), "—")
            ax.text(0.97, 0.95, f"Half-life ≈ {halflife} bars",
                    transform=ax.transAxes, ha="right", va="top", fontsize=9,
                    bbox=dict(facecolor="white", alpha=0.7))

        # Panel 2: Alpha Distribution
        ax = axes[0, 1]
        ax.set_title(f"Alpha Distribution\n"
                     f"skew={alpha_stats.get('skew', 0):.2f}  "
                     f"kurt={alpha_stats.get('kurt', 0):.2f}  "
                     f"{alpha_stats.get('pct_positive', 0):.1%} positive")
        ax.text(0.5, 0.5,
                f"mean={alpha_stats.get('mean', 0):.4f}\n"
                f"std={alpha_stats.get('std', 0):.4f}\n"
                f"min={alpha_stats.get('min', 0):.4f}\n"
                f"max={alpha_stats.get('max', 0):.4f}\n"
                f"in [-0.5,0.5]: {alpha_stats.get('pct_in_bounds', 0):.1%}",
                transform=ax.transAxes, ha="center", va="center", fontsize=12,
                bbox=dict(boxstyle="round", facecolor="#E3F2FD", alpha=0.8))
        ax.axis("off")

        # Panel 3: Cumulative Daily IC
        ax = axes[1, 0]
        if len(daily_ic) > 0:
            cum_ic = daily_ic.cumsum()
            ax.plot(range(len(cum_ic)), cum_ic.values, color="#2196F3", linewidth=1.5)
            ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
            ax.fill_between(range(len(cum_ic)), cum_ic.values, 0,
                            where=cum_ic.values >= 0, alpha=0.2, color="#2196F3")
            ax.fill_between(range(len(cum_ic)), cum_ic.values, 0,
                            where=cum_ic.values < 0, alpha=0.2, color="#F44336")
            ax.set_xlabel("Trading Days")
            ax.set_ylabel("Cumulative IC×100")
            ax.set_title("Cumulative Daily IC")
        else:
            ax.text(0.5, 0.5, "No daily IC data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=12)
            ax.axis("off")

        # Panel 4: Metrics Summary
        ax = axes[1, 1]
        pass_color = "#E8F5E9" if metrics.get("PassGates") else "#FFEBEE"
        ax.text(0.5, 0.5,
                f"IC     = {metrics.get('IC', 0):.4f}\n"
                f"IR     = {metrics.get('IR', 0):.4f}\n"
                f"tvr    = {metrics.get('Turnover', 0):.1f}\n"
                f"Score  = {metrics.get('Score', 0):.2f}\n"
                f"Pass   = {'✓' if metrics.get('PassGates') else '✗'}\n\n"
                f"{formula[:70]}{'...' if len(formula) > 70 else ''}",
                transform=ax.transAxes, ha="center", va="center", fontsize=11,
                fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor=pass_color, alpha=0.8))
        ax.set_title("Metrics & Formula")
        ax.axis("off")

        plt.tight_layout()
        plot_path = out_dir / "analysis.png"
        plt.savefig(plot_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"  [research] Plot saved → {plot_path}")
    except Exception as e:
        print(f"  [research] Plot generation failed: {e}")


def analyze_factor(
    run_id: str,
    formula: str,
    alpha: pd.Series,
    metrics: Dict[str, Any],
    hub: Any,
    eval_days: List[str],
) -> str:
    """
    Run full factor research analysis for one evaluated factor.

    Args:
        run_id:     Unique factor identifier
        formula:    DSL formula string
        alpha:      Alpha series (MultiIndex: date/datetime/security_id)
        metrics:    Dict with IC, IR, Turnover, Score, PassGates
        hub:        DataHub instance (for resp and trading_restriction)
        eval_days:  List of trading-day strings in the evaluation window

    Returns:
        Path string to the research output directory.
    """
    out_dir = RESEARCH_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    alpha_stats = _compute_alpha_stats(alpha)
    ic_decay: Dict[int, float] = {}
    daily_ic = pd.Series(dtype=float)

    try:
        day_ts = [pd.to_datetime(d) for d in eval_days]
        resp_col = "resp" if "resp" in hub.resp.columns else hub.resp.columns[0]
        rest_col = (
            "trading_restriction"
            if "trading_restriction" in hub.trading_restriction.columns
            else hub.trading_restriction.columns[0]
        )

        resp_slice = hub.resp.loc[hub.resp.index.get_level_values("date").isin(day_ts)]
        rest_slice = hub.trading_restriction.loc[
            hub.trading_restriction.index.get_level_values("date").isin(day_ts)
        ]

        # Filter to submission bar times only
        alpha_f = _filter_allowed(alpha)
        resp_f = _filter_allowed(resp_slice[resp_col])
        rest_f = _filter_allowed(rest_slice[rest_col])

        # Build wide DataFrames
        alpha_wide = alpha_f.unstack("security_id")
        resp_wide = resp_f.unstack("security_id").reindex_like(alpha_wide)
        rest_wide = rest_f.unstack("security_id").reindex_like(alpha_wide).fillna(0)

        ic_decay = _compute_ic_decay(alpha_wide, resp_wide, rest_wide, max_lag=10)
        daily_ic = _compute_daily_ic(alpha_wide, resp_wide, rest_wide)

    except Exception as e:
        print(f"  [research] IC analysis failed: {e}")
        ic_decay = {i: 0.0 for i in range(11)}

    # Generate plots
    plt = _try_plt()
    if plt is not None:
        _generate_plots(
            run_id=run_id,
            formula=formula,
            metrics=metrics,
            alpha_stats=alpha_stats,
            ic_decay=ic_decay,
            daily_ic=daily_ic,
            out_dir=out_dir,
            plt=plt,
        )

    # Save JSON report
    report = {
        "run_id": run_id,
        "formula": formula,
        "metrics": metrics,
        "alpha_stats": alpha_stats,
        "ic_decay": {str(k): v for k, v in ic_decay.items()},
        "daily_ic": {str(k): float(v) for k, v in daily_ic.items()},
        "n_daily_ic": len(daily_ic),
        "created_at": pd.Timestamp.now().isoformat(),
    }
    report_path = out_dir / "report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    # Save Markdown summary
    decay_rows = "\n".join(
        f"| {lag} | {val:.4f} |" for lag, val in ic_decay.items()
    )
    md = f"""# Factor Research: {run_id}

## Formula
```
{formula}
```

## Platform Metrics
| Metric | Value |
|--------|-------|
| IC | {metrics.get('IC', 0):.4f} |
| IR | {metrics.get('IR', 0):.4f} |
| Turnover | {metrics.get('Turnover', 0):.2f} |
| Score | {metrics.get('Score', 0):.2f} |
| PassGates | {metrics.get('PassGates', False)} |

## Alpha Distribution
| Stat | Value |
|------|-------|
| Mean | {alpha_stats.get('mean', 0):.4f} |
| Std | {alpha_stats.get('std', 0):.4f} |
| Skewness | {alpha_stats.get('skew', 0):.4f} |
| Kurtosis | {alpha_stats.get('kurt', 0):.4f} |
| % Positive | {alpha_stats.get('pct_positive', 0):.2%} |

## IC Decay (by lag)
| Lag | IC×100 |
|-----|--------|
{decay_rows}
"""
    (out_dir / "report.md").write_text(md, encoding="utf-8")
    print(f"  [research] Report saved → {out_dir}")
    return str(out_dir)
