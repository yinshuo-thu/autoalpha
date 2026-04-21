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
import re
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


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _fmt(value: Any, digits: int = 3) -> str:
    return f"{_safe_float(value):.{digits}f}"


def _round4(value: Any) -> float:
    return round(_safe_float(value), 4)


def _round_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            out[key] = _round4(value)
        else:
            out[key] = value
    return out


def _compress_series(series: pd.Series, max_points: int = 80) -> list[dict[str, Any]]:
    """Return a compact chart-friendly time series."""
    if series is None or len(series) == 0:
        return []
    clean = series.dropna()
    if len(clean) == 0:
        return []
    if len(clean) > max_points:
        idx = np.linspace(0, len(clean) - 1, max_points).round().astype(int)
        clean = clean.iloc[np.unique(idx)]
    return [
        {"x": str(index)[:10], "value": _round4(value)}
        for index, value in clean.items()
    ]


def _formula_inputs(formula: str) -> list[str]:
    fields = [
        "open", "high", "low", "close", "volume", "vwap", "amount",
        "trade_count", "dvolume",
    ]
    lowered = (formula or "").lower()
    return [field for field in fields if re.search(rf"\b{re.escape(field)}\b", lowered)]


def _formula_theme(formula: str) -> str:
    lowered = (formula or "").lower()
    themes: list[str] = []
    if "vwap" in lowered:
        themes.append("VWAP dislocation")
    if "trade_count" in lowered or "volume" in lowered or "dvolume" in lowered:
        themes.append("participation")
    if "ts_decay_linear" in lowered or "ts_mean" in lowered:
        themes.append("smoothed intraday signal")
    if "neg(" in lowered or " - " in lowered:
        themes.append("reversion/contrast")
    if "ts_rank" in lowered or "cs_rank" in lowered:
        themes.append("rank-normalized")
    return " + ".join(themes[:3]) if themes else "generic intraday alpha"


def _gate_notes(metrics: Dict[str, Any]) -> list[str]:
    detail = metrics.get("GatesDetail") or {}
    if not detail:
        return ["Gate detail unavailable"]
    notes: list[str] = []
    labels = {
        "IC": "IC predictive power",
        "IR": "IR consistency",
        "Turnover": "TVR turnover",
        "Concentration": "position concentration",
        "Coverage": "coverage",
    }
    for key, label in labels.items():
        if key in detail:
            notes.append(f"{label}: {'pass' if detail.get(key) else 'fail'}")
    for key, value in detail.items():
        if key not in labels:
            notes.append(f"{key}: {'pass' if value else 'fail'}")
    return notes


def build_factor_card(
    run_id: str,
    formula: str,
    metrics: Dict[str, Any],
    alpha_stats: Dict[str, float],
    ic_decay: Dict[int | str, float],
    daily_ic: pd.Series | Dict[str, Any] | None = None,
    daily_rank_ic: pd.Series | Dict[str, Any] | None = None,
    temporal: Optional[Dict[str, Any]] = None,
    group_returns: Optional[Dict[str, Any]] = None,
    regimes: Optional[list[dict[str, Any]]] = None,
    stability: Optional[Dict[str, Any]] = None,
    redundancy: Optional[Dict[str, Any]] = None,
    thought_process: str = "",
) -> Dict[str, Any]:
    """Build the factor card consumed by the LOG UI for submit-ready factors."""
    daily_count = len(daily_ic) if daily_ic is not None else 0
    ic0 = _safe_float(ic_decay.get(0, ic_decay.get("0", 0.0)) if ic_decay else 0.0)
    half_life = "not observed"
    for raw_lag, raw_val in (ic_decay or {}).items():
        try:
            lag = int(raw_lag)
        except Exception:
            continue
        if lag > 0 and abs(_safe_float(raw_val)) < abs(ic0) * 0.5:
            half_life = f"{lag} bars"
            break

    preview = metrics.get("result_preview") or {}
    pass_gates = bool(metrics.get("PassGates", False))
    distribution = {
        **_round_dict(alpha_stats),
        "p1": _round4(alpha_stats.get("p1", 0.0)),
        "p5": _round4(alpha_stats.get("p5", 0.0)),
        "p50": _round4(alpha_stats.get("p50", 0.0)),
        "p95": _round4(alpha_stats.get("p95", 0.0)),
        "p99": _round4(alpha_stats.get("p99", 0.0)),
    }
    card = {
        "run_id": run_id,
        "title": run_id,
        "status": "PASS" if pass_gates else "REJECTED",
        "theme": _formula_theme(formula),
        "thesis": thought_process or "No agent thesis was recorded for this factor.",
        "definition": {
            "formula": formula,
            "inputs": _formula_inputs(formula),
            "update_frequency": "15-minute platform bar",
            "prediction_horizon": "next submitted response bar; decay panel probes lagged bars",
            "universe": "Scientech competition equity universe after trading_restriction filtering",
            "postprocess": "cross-sectional rank/clip or zscore/clip before submission",
        },
        "metrics": {
            "IC": _safe_float(metrics.get("IC", preview.get("IC", 0.0))),
            "IR": _safe_float(metrics.get("IR", preview.get("IR", 0.0))),
            "tvr": _safe_float(metrics.get("Turnover", metrics.get("tvr", preview.get("tvr", 0.0)))),
            "Score": _safe_float(metrics.get("Score", preview.get("score", 0.0))),
            "nd": _safe_float(preview.get("nd", daily_count)),
            "cover_all": int(preview.get("cover_all", 1 if daily_count else 0)),
        },
        "distribution": distribution,
        "histogram": alpha_stats.get("histogram", []),
        "temporal": temporal or {},
        "prediction": {
            "ic_mean": _round4(metrics.get("IC", preview.get("IC", 0.0))),
            "icir": _round4(metrics.get("IR", preview.get("IR", 0.0))),
            "rank_ic": _round4(daily_rank_ic.mean() if hasattr(daily_rank_ic, "mean") and len(daily_rank_ic) else 0.0),
            "rolling_ic": _compress_series(pd.Series(daily_ic).rolling(20, min_periods=5).mean() if daily_ic is not None else pd.Series(dtype=float)),
            "horizon_ic": [
                {"horizon": f"{int(lag) * 15}m" if int(lag) > 0 else "0m", "ic": _round4(value)}
                for lag, value in (ic_decay or {}).items()
                if str(lag).isdigit() and int(lag) in (0, 1, 5, 10)
            ],
        },
        "layering": group_returns or {},
        "regime": regimes or [],
        "stability": stability or {},
        "redundancy": redundancy or {},
        "gate_notes": _gate_notes(metrics),
        "diagnostics": {
            "ic_decay_0": ic0,
            "ic_half_life": half_life,
            "daily_ic_count": daily_count,
            "alpha_mean": _safe_float(alpha_stats.get("mean", 0.0)),
            "alpha_std": _safe_float(alpha_stats.get("std", 0.0)),
            "pct_positive": _safe_float(alpha_stats.get("pct_positive", 0.0)),
            "pct_in_bounds": _safe_float(alpha_stats.get("pct_in_bounds", 0.0)),
        },
        "risk_notes": [
            f"Turnover gate uses corrected restricted raw alpha diff sum x100; current tvr={_fmt(metrics.get('Turnover', 0), 2)}.",
            f"Concentration max/min bps: maxx={_fmt(metrics.get('maxx', 0), 2)}, minn={_fmt(metrics.get('minn', 0), 2)}.",
        ],
        "formula": formula,
        "created_at": pd.Timestamp.now().isoformat(),
    }
    card["metrics"] = _round_dict(card["metrics"])
    return card


def write_factor_card(out_dir: Path, card: Dict[str, Any]) -> str:
    """Persist JSON and Markdown versions of one factor card."""
    out_dir.mkdir(parents=True, exist_ok=True)
    card_path = out_dir / "factor_card.json"
    card_path.write_text(json.dumps(card, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    metrics = card.get("metrics", {})
    diagnostics = card.get("diagnostics", {})
    gate_rows = "\n".join(f"- {item}" for item in card.get("gate_notes", []))
    risk_rows = "\n".join(f"- {item}" for item in card.get("risk_notes", []))
    md = f"""# Factor Card: {card.get('run_id', '')}

## Snapshot
- Status: {card.get('status', '')}
- Theme: {card.get('theme', '')}
- IC / IR / TVR / Score: {_fmt(metrics.get('IC'))} / {_fmt(metrics.get('IR'))} / {_fmt(metrics.get('tvr'), 2)} / {_fmt(metrics.get('Score'), 2)}
- Days / Coverage: {_fmt(metrics.get('nd'), 0)} / {metrics.get('cover_all', 0)}

## Agent Thesis
{card.get('thesis', '')}

## Gate Notes
{gate_rows}

## Diagnostics
- IC decay lag 0: {_fmt(diagnostics.get('ic_decay_0'))}
- IC half-life: {diagnostics.get('ic_half_life', '')}
- Daily IC count: {diagnostics.get('daily_ic_count', 0)}
- Alpha mean/std: {_fmt(diagnostics.get('alpha_mean'), 5)} / {_fmt(diagnostics.get('alpha_std'), 5)}
- Positive / in-bounds: {_fmt(diagnostics.get('pct_positive'), 3)} / {_fmt(diagnostics.get('pct_in_bounds'), 3)}

## Risk Notes
{risk_rows}

## Formula
```text
{card.get('formula', '')}
```
"""
    (out_dir / "factor_card.md").write_text(md, encoding="utf-8")
    return str(card_path)


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
    total = len(alpha)
    vals = alpha.dropna()
    if len(vals) == 0:
        return {}
    hist_counts, hist_edges = np.histogram(vals.astype(float), bins=20)
    return {
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "skew": float(vals.skew()),
        "kurt": float(vals.kurtosis()),
        "min": float(vals.min()),
        "max": float(vals.max()),
        "p1": float(vals.quantile(0.01)),
        "p5": float(vals.quantile(0.05)),
        "p50": float(vals.quantile(0.50)),
        "p95": float(vals.quantile(0.95)),
        "p99": float(vals.quantile(0.99)),
        "missing_rate": float(1.0 - (len(vals) / max(total, 1))),
        "extreme_share": float(((vals <= vals.quantile(0.01)) | (vals >= vals.quantile(0.99))).mean()),
        "pct_positive": float((vals > 0).mean()),
        "pct_in_bounds": float(((vals >= -0.5) & (vals <= 0.5)).mean()),
        "histogram": [
            {
                "bin": f"{hist_edges[idx]:.3f}..{hist_edges[idx + 1]:.3f}",
                "value": int(hist_counts[idx]),
            }
            for idx in range(len(hist_counts))
        ],
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


def _compute_daily_rank_ic(
    alpha_wide: pd.DataFrame,
    resp_wide: pd.DataFrame,
    rest_wide: pd.DataFrame,
) -> pd.Series:
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
        ic = a[valid].rank(pct=True).corr(r[valid].rank(pct=True))
        if not np.isnan(ic):
            daily_ics[date] = ic * 100
    return pd.Series(daily_ics, name="daily_rank_IC")


def _compute_group_returns(alpha_wide: pd.DataFrame, resp_wide: pd.DataFrame, rest_wide: pd.DataFrame) -> Dict[str, Any]:
    grouped: Dict[int, list[float]] = {idx: [] for idx in range(1, 11)}
    long_short: list[float] = []
    cumulative: list[float] = []
    running = 0.0

    for date in alpha_wide.index:
        if date not in resp_wide.index:
            continue
        a = alpha_wide.loc[date]
        r = resp_wide.loc[date]
        mask = (rest_wide.loc[date] == 0) if date in rest_wide.index else pd.Series(True, index=a.index)
        valid = a.notna() & r.notna() & mask.astype(bool)
        if valid.sum() < 50:
            continue
        frame = pd.DataFrame({"alpha": a[valid], "resp": r[valid]}).sort_values("alpha")
        try:
            frame["bucket"] = pd.qcut(frame["alpha"].rank(method="first"), 10, labels=False) + 1
        except Exception:
            continue
        bucket_ret = frame.groupby("bucket")["resp"].mean() * 10000
        for bucket, value in bucket_ret.items():
            grouped[int(bucket)].append(float(value))
        if 1 in bucket_ret and 10 in bucket_ret:
            spread = float(bucket_ret.loc[10] - bucket_ret.loc[1])
            long_short.append(spread)
            running += spread
            cumulative.append(running)

    return {
        "decile_returns_bps": [
            {"bucket": bucket, "value": _round4(np.mean(values) if values else 0.0)}
            for bucket, values in grouped.items()
        ],
        "top_minus_bottom_bps": _round4(np.mean(long_short) if long_short else 0.0),
        "cumulative_top_minus_bottom": [
            {"x": str(idx + 1), "value": _round4(value)}
            for idx, value in enumerate(cumulative[-80:])
        ],
    }


def _compute_regimes(daily_ic: pd.Series, resp_wide: pd.DataFrame) -> list[dict[str, Any]]:
    if daily_ic is None or len(daily_ic) == 0 or resp_wide.empty:
        return []
    daily_resp_std = resp_wide.std(axis=1).reindex(daily_ic.index)
    daily_resp_mean = resp_wide.mean(axis=1).reindex(daily_ic.index)
    regimes = []
    high_vol = daily_resp_std >= daily_resp_std.median()
    high_trend = daily_resp_mean.abs() >= daily_resp_mean.abs().median()
    positive = daily_resp_mean >= 0
    splits = {
        "高波动": high_vol,
        "低波动": ~high_vol,
        "趋势日": high_trend,
        "震荡日": ~high_trend,
        "上行响应": positive,
        "下行响应": ~positive,
    }
    for name, mask in splits.items():
        vals = daily_ic.loc[mask.fillna(False)]
        regimes.append({
            "regime": name,
            "ic": _round4(vals.mean() if len(vals) else 0.0),
            "days": int(len(vals)),
        })
    return regimes


def _compute_stability(daily_ic: pd.Series, alpha_wide: pd.DataFrame, resp_wide: pd.DataFrame, rest_wide: pd.DataFrame) -> Dict[str, Any]:
    if daily_ic is None or len(daily_ic) == 0:
        return {"monthly_ic": [], "yearly_ic": [], "splits": [], "clipped_ic": 0.0}

    dated = daily_ic.copy()
    dated.index = pd.to_datetime(dated.index)
    monthly = dated.groupby(dated.index.to_period("M")).mean()
    yearly = dated.groupby(dated.index.year).mean()
    n = len(dated)
    splits = []
    for label, subset in {
        "Train": dated.iloc[: max(1, n // 3)],
        "Val": dated.iloc[max(1, n // 3): max(2, 2 * n // 3)],
        "Test": dated.iloc[max(2, 2 * n // 3):],
    }.items():
        splits.append({"split": label, "ic": _round4(subset.mean() if len(subset) else 0.0), "days": int(len(subset))})

    clipped_ic_vals: list[float] = []
    clipped_alpha = alpha_wide.clip(
        lower=alpha_wide.quantile(0.01, axis=1),
        upper=alpha_wide.quantile(0.99, axis=1),
        axis=0,
    )
    for date in clipped_alpha.index:
        if date not in resp_wide.index:
            continue
        a = clipped_alpha.loc[date]
        r = resp_wide.loc[date]
        mask = (rest_wide.loc[date] == 0) if date in rest_wide.index else pd.Series(True, index=a.index)
        valid = a.notna() & r.notna() & mask.astype(bool)
        if valid.sum() < 20:
            continue
        ic = a[valid].corr(r[valid])
        if not np.isnan(ic):
            clipped_ic_vals.append(float(ic * 100))

    return {
        "monthly_ic": _compress_series(monthly, max_points=36),
        "yearly_ic": [{"x": str(index), "value": _round4(value)} for index, value in yearly.items()],
        "splits": splits,
        "clipped_ic": _round4(np.mean(clipped_ic_vals) if clipped_ic_vals else 0.0),
    }


def _compute_redundancy(formula: str) -> Dict[str, Any]:
    """Lightweight redundancy proxy when full historical alpha pool is unavailable."""
    tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", formula or ""))
    best = {"run_id": "", "similarity": 0.0}
    try:
        from autoalpha import knowledge_base as kb

        for item in kb.get_all_factors():
            other_formula = item.get("formula", "")
            if other_formula == formula:
                continue
            other_tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", other_formula or ""))
            union = tokens | other_tokens
            sim = len(tokens & other_tokens) / len(union) if union else 0.0
            if sim > best["similarity"]:
                best = {"run_id": item.get("run_id", ""), "similarity": sim}
    except Exception:
        pass
    return {
        "family": _formula_theme(formula),
        "max_formula_token_overlap": _round4(best["similarity"]),
        "nearest_factor": best["run_id"],
        "target_corr_proxy": "IC/100",
    }


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
    thought_process: str = "",
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
    daily_rank_ic = pd.Series(dtype=float)
    temporal: Dict[str, Any] = {}
    group_returns: Dict[str, Any] = {}
    regimes: list[dict[str, Any]] = []
    stability: Dict[str, Any] = {}
    redundancy: Dict[str, Any] = {}

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
        daily_rank_ic = _compute_daily_rank_ic(alpha_wide, resp_wide, rest_wide)
        temporal = {
            "daily_mean": _compress_series(alpha_wide.mean(axis=1), max_points=80),
            "daily_std": _compress_series(alpha_wide.std(axis=1), max_points=80),
            "coverage": _compress_series(alpha_wide.notna().mean(axis=1), max_points=80),
            "rolling_drift": _compress_series(alpha_wide.mean(axis=1).rolling(20, min_periods=5).mean(), max_points=80),
        }
        group_returns = _compute_group_returns(alpha_wide, resp_wide, rest_wide)
        regimes = _compute_regimes(daily_ic, resp_wide)
        stability = _compute_stability(daily_ic, alpha_wide, resp_wide, rest_wide)
        redundancy = _compute_redundancy(formula)

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

    factor_card = None
    card_path = ""
    if bool(metrics.get("PassGates", False)):
        factor_card = build_factor_card(
            run_id=run_id,
            formula=formula,
            metrics=metrics,
            alpha_stats=alpha_stats,
            ic_decay=ic_decay,
            daily_ic=daily_ic,
            daily_rank_ic=daily_rank_ic,
            temporal=temporal,
            group_returns=group_returns,
            regimes=regimes,
            stability=stability,
            redundancy=redundancy,
            thought_process=thought_process,
        )
        card_path = write_factor_card(out_dir, factor_card)

    # Save JSON report
    report = {
        "run_id": run_id,
        "formula": formula,
        "metrics": metrics,
        "alpha_stats": alpha_stats,
        "ic_decay": {str(k): v for k, v in ic_decay.items()},
        "daily_ic": {str(k): float(v) for k, v in daily_ic.items()},
        "n_daily_ic": len(daily_ic),
        "factor_card": factor_card,
        "factor_card_path": card_path,
        "created_at": pd.Timestamp.now().isoformat(),
    }
    report_path = out_dir / "report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    # Save Markdown summary
    decay_rows = "\n".join(
        f"| {lag} | {val:.4f} |" for lag, val in ic_decay.items()
    )
    if factor_card:
        card_summary = (
            f"- Theme: {factor_card.get('theme')}\n"
            f"- Status: {factor_card.get('status')}\n"
            f"- Thesis: {factor_card.get('thesis')}"
        )
    else:
        card_summary = "No factor card was generated because this factor did not pass all submit gates."
    md = f"""# Factor Research: {run_id}

## Formula
```
{formula}
```

## Factor Card
{card_summary}

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
