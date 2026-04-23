"""
autoalpha_v2/factor_research.py

Lightweight factor research analysis.
Computes: key characteristics, time-series appearance, basic analysis,
and inter-factor similarity.  No heavy per-bar Python loops.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

RESEARCH_DIR = Path(__file__).resolve().parent / "research"
CORRELATION_CACHE_PATH = Path(__file__).resolve().parent / "factor_correlations.json"

ALLOWED_UTC_TIMES = {
    "01:45:00", "02:00:00", "02:15:00", "02:30:00",
    "02:45:00", "03:00:00", "03:15:00", "03:30:00",
    "05:15:00", "05:30:00", "05:45:00", "06:00:00",
    "06:15:00", "06:30:00", "06:45:00", "07:00:00",
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value) if value is not None else default
    except Exception:
        return default


def _fmt(value: Any, digits: int = 3) -> str:
    return f"{_safe_float(value):.{digits}f}"


def _round4(value: Any) -> float:
    return round(_safe_float(value), 4)


def _round_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        k: _round4(v) if isinstance(v, (int, float, np.integer, np.floating)) else v
        for k, v in data.items()
    }


def _compress_series(series: Any, max_points: int = 80) -> list[dict[str, Any]]:
    if series is None or len(series) == 0:
        return []
    clean = pd.Series(series).replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return []
    if len(clean) > max_points:
        idx = np.linspace(0, len(clean) - 1, max_points).round().astype(int)
        clean = clean.iloc[np.unique(idx)]
    return [{"x": _format_series_index(i), "value": _round4(v)} for i, v in clean.items()]


def _format_series_index(index: Any) -> str:
    if isinstance(index, tuple):
        for part in index:
            try:
                ts = pd.Timestamp(part)
                if not pd.isna(ts):
                    return ts.strftime("%Y-%m-%d")
            except Exception:
                continue
        return " / ".join(str(p) for p in index)
    try:
        ts = pd.Timestamp(index)
        if not pd.isna(ts):
            return ts.strftime("%Y-%m-%d")
    except Exception:
        pass
    return str(index)[:24]


def _index_date_key(index: Any) -> Any:
    if isinstance(index, tuple) and index:
        index = index[0]
    try:
        return pd.Timestamp(index).normalize()
    except Exception:
        return index


def _filter_allowed(series: pd.Series) -> pd.Series:
    dt_vals = series.index.get_level_values("datetime")
    times = pd.to_datetime(dt_vals).strftime("%H:%M:%S")
    mask = pd.Series(times, index=series.index).isin(ALLOWED_UTC_TIMES)
    return series.loc[mask]


def _apply_restriction_wide(alpha_wide: pd.DataFrame, rest_wide: pd.DataFrame) -> pd.DataFrame:
    if alpha_wide.empty or rest_wide is None or rest_wide.empty:
        return alpha_wide
    restriction = rest_wide.reindex_like(alpha_wide).fillna(0)
    blocked = (
        ((restriction == 1) & (alpha_wide < 0))
        | ((restriction == 2) & (alpha_wide > 0))
        | (restriction == 3)
    )
    return alpha_wide.mask(blocked)


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def _compute_alpha_stats(alpha: pd.Series) -> Dict[str, Any]:
    """Distribution statistics on the raw alpha series — no wide matrix needed."""
    total = len(alpha)
    vals = alpha.dropna()
    if len(vals) == 0:
        return {}
    hist_counts, hist_edges = np.histogram(vals.astype(float), bins=20)
    p1 = float(vals.quantile(0.01))
    p99 = float(vals.quantile(0.99))
    return {
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "skew": float(vals.skew()),
        "kurt": float(vals.kurtosis()),
        "min": float(vals.min()),
        "max": float(vals.max()),
        "p1": p1,
        "p5": float(vals.quantile(0.05)),
        "p50": float(vals.quantile(0.50)),
        "p95": float(vals.quantile(0.95)),
        "p99": p99,
        "missing_rate": float(1.0 - len(vals) / max(total, 1)),
        "extreme_share": float(((vals <= p1) | (vals >= p99)).mean()),
        "pct_positive": float((vals > 0).mean()),
        "pct_in_bounds": float(((vals >= -0.5) & (vals <= 0.5)).mean()),
        "histogram": [
            {"bin": f"{hist_edges[i]:.3f}..{hist_edges[i + 1]:.3f}", "value": int(hist_counts[i])}
            for i in range(len(hist_counts))
        ],
    }


def _compute_daily_ic(
    alpha_wide: pd.DataFrame,
    resp_wide: pd.DataFrame,
    rest_wide: pd.DataFrame,
) -> pd.Series:
    """Vectorized bar-level Pearson IC aggregated to daily mean — no Python row loop."""
    alpha_r = _apply_restriction_wide(alpha_wide, rest_wide)
    resp_al = resp_wide.reindex_like(alpha_wide)

    a = alpha_r.values.astype(np.float64)
    r = resp_al.values.astype(np.float64)
    valid = np.isfinite(a) & np.isfinite(r)
    n = valid.sum(axis=1)  # (n_bars,)

    a_sum = np.where(valid, a, 0.0).sum(axis=1, keepdims=True)
    r_sum = np.where(valid, r, 0.0).sum(axis=1, keepdims=True)
    n_k = np.where(n[:, None] > 0, n[:, None], 1)
    a_dm = np.where(valid, a - a_sum / n_k, 0.0)
    r_dm = np.where(valid, r - r_sum / n_k, 0.0)

    num = (a_dm * r_dm).sum(axis=1)
    denom = np.sqrt((a_dm ** 2).sum(axis=1) * (r_dm ** 2).sum(axis=1))

    ic_bar = np.where((n >= 20) & (denom > 1e-12), num / denom * 100.0, np.nan)
    series = pd.Series(ic_bar, index=alpha_wide.index).dropna()
    if series.empty:
        return pd.Series(dtype=float, name="daily_IC")
    return series.groupby(series.index.map(_index_date_key)).mean().rename("daily_IC")


def _compute_temporal(alpha_wide: pd.DataFrame) -> Dict[str, Any]:
    """Daily mean, std, and coverage — pure groupby, no loops."""
    g = alpha_wide.index.map(_index_date_key)
    return {
        "daily_mean": _compress_series(alpha_wide.mean(axis=1).groupby(g).mean(), max_points=80),
        "daily_std": _compress_series(alpha_wide.std(axis=1).groupby(g).mean(), max_points=80),
        "coverage": _compress_series(alpha_wide.notna().mean(axis=1).groupby(g).mean(), max_points=80),
    }


def _compute_regimes(daily_ic: pd.Series, resp_wide: pd.DataFrame) -> list[dict[str, Any]]:
    """IC breakdown by high/low-volatility and up/down market."""
    if daily_ic is None or len(daily_ic) == 0 or resp_wide.empty:
        return []
    g = resp_wide.index.map(_index_date_key)
    resp_std = resp_wide.std(axis=1).groupby(g).mean().reindex(daily_ic.index)
    resp_mean = resp_wide.mean(axis=1).groupby(g).mean().reindex(daily_ic.index)
    high_vol = resp_std >= resp_std.median()
    high_trend = resp_mean.abs() >= resp_mean.abs().median()
    positive = resp_mean >= 0
    regimes = []
    for name, mask in [
        ("高波动", high_vol), ("低波动", ~high_vol),
        ("趋势日", high_trend), ("震荡日", ~high_trend),
        ("上行", positive), ("下行", ~positive),
    ]:
        vals = daily_ic.loc[mask.fillna(False)]
        regimes.append({
            "regime": name,
            "ic": _round4(float(vals.mean()) if len(vals) else 0.0),
            "days": int(len(vals)),
        })
    return regimes


def _compute_monthly_ic(
    daily_ic: pd.Series,
    resp_wide: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Monthly IC breakdown; optionally merged with monthly market return and volatility."""
    if daily_ic is None or len(daily_ic) == 0:
        return {"monthly": [], "yearly": [], "positive_month_ratio": 0.0, "full_sample_ic": 0.0}
    dated = daily_ic.copy()
    dated.index = pd.to_datetime(dated.index)
    monthly = dated.groupby(dated.index.to_period("M")).mean()
    yearly = dated.groupby(dated.index.year).mean()

    # Build monthly list; enrich with market state if resp_wide is provided
    monthly_list: list[dict] = []
    if resp_wide is not None and not resp_wide.empty:
        g = resp_wide.index.map(_index_date_key)
        # daily cross-sectional mean return and std
        daily_mean = resp_wide.mean(axis=1).groupby(g).mean()
        daily_std = resp_wide.std(axis=1).groupby(g).mean()
        daily_mean.index = pd.to_datetime(daily_mean.index)
        daily_std.index = pd.to_datetime(daily_std.index)
        monthly_ret = daily_mean.groupby(daily_mean.index.to_period("M")).mean()
        monthly_vol = daily_std.groupby(daily_std.index.to_period("M")).mean()
        monthly_abs_move = daily_mean.abs().groupby(daily_mean.index.to_period("M")).mean()
        # normalise vol to same scale as IC for visual overlay
        vol_norm = monthly_vol / (monthly_vol.quantile(0.9) + 1e-9)
        for period, ic_val in monthly.items():
            row: dict = {"x": str(period), "value": _round4(float(ic_val))}
            if period in monthly_ret:
                row["market_ret"] = _round4(float(monthly_ret[period]) * 100)
            if period in vol_norm:
                row["market_vol"] = _round4(float(vol_norm[period]))
            if period in monthly_abs_move:
                row["market_abs_move"] = _round4(float(monthly_abs_move[period]) * 100)
            monthly_list.append(row)
    else:
        monthly_list = _compress_series(monthly, max_points=36)

    return {
        "monthly": monthly_list[:36],
        "yearly": [{"x": str(k), "value": _round4(v)} for k, v in yearly.items()],
        "positive_month_ratio": _round4(float((monthly > 0).mean()) if len(monthly) else 0.0),
        "full_sample_ic": _round4(float(dated.mean()) if len(dated) else 0.0),
    }


def _formula_inputs(formula: str) -> list[str]:
    fields = ["open", "high", "low", "close", "volume", "vwap", "amount", "trade_count", "dvolume"]
    lowered = (formula or "").lower()
    return [f for f in fields if re.search(rf"\b{re.escape(f)}\b", lowered)]


def _formula_theme(formula: str) -> str:
    lowered = (formula or "").lower()
    themes: list[str] = []
    if "vwap" in lowered:
        themes.append("VWAP dislocation")
    if "trade_count" in lowered or "volume" in lowered or "dvolume" in lowered:
        themes.append("participation")
    if "ts_decay_linear" in lowered or "ts_mean" in lowered:
        themes.append("smoothed intraday signal")
    if "neg(" in lowered or re.search(r"-", lowered):
        themes.append("reversion/contrast")
    if "ts_rank" in lowered or "cs_rank" in lowered:
        themes.append("rank-normalized")
    return " + ".join(themes[:3]) if themes else "generic intraday alpha"


def _gate_notes(metrics: Dict[str, Any]) -> list[str]:
    detail = metrics.get("GatesDetail") or {}
    if not detail:
        return ["Gate detail unavailable"]
    labels = {
        "IC": "IC predictive power", "IR": "IR consistency",
        "Turnover": "TVR turnover", "Concentration": "position concentration",
    }
    notes = []
    for key, label in labels.items():
        if key in detail:
            notes.append(f"{label}: {'pass' if detail.get(key) else 'fail'}")
    for key, value in detail.items():
        if key not in labels:
            notes.append(f"{key}: {'pass' if value else 'fail'}")
    return notes


# ---------------------------------------------------------------------------
# Redundancy (inter-factor similarity)
# ---------------------------------------------------------------------------

def _read_parquet_alpha_wide(path: str, sample_dates: Optional[list[str]]) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    try:
        kwargs: dict = {"columns": ["date", "datetime", "security_id", "alpha"]}
        if sample_dates:
            kwargs["filters"] = [("date", "in", sample_dates)]
        frame = pd.read_parquet(path, **kwargs)
    except Exception:
        return pd.DataFrame()
    if frame.empty or "alpha" not in frame.columns:
        return pd.DataFrame()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    if sample_dates:
        frame = frame[frame["date"].dt.strftime("%Y-%m-%d").isin(sample_dates)]
    if frame.empty:
        return pd.DataFrame()
    return frame.set_index(["date", "datetime", "security_id"])["alpha"].unstack("security_id")


def _existing_alpha_path(item: Dict[str, Any]) -> str:
    for key in ("submit_path", "parquet_path"):
        path = str(item.get(key) or "")
        if path and Path(path).is_file():
            return path
    return ""


def _mean_bar_correlation(left: pd.DataFrame, right: pd.DataFrame) -> tuple[float, int]:
    common_index = left.index.intersection(right.index)
    common_cols = left.columns.intersection(right.columns)
    if len(common_index) == 0 or len(common_cols) < 20:
        return 0.0, 0
    a = left.loc[common_index, common_cols]
    b = right.loc[common_index, common_cols]
    vals = a.corrwith(b, axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if vals.empty:
        return 0.0, 0
    return float(vals.mean()), int(vals.count())


def _compute_redundancy(
    formula: str,
    run_id: str,
    alpha_wide: pd.DataFrame,
    max_corr_factors: int = 30,
) -> Dict[str, Any]:
    """Formula token overlap and realized-alpha correlation vs. existing passing factors."""
    tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", formula or ""))
    best: Dict[str, Any] = {"run_id": "", "similarity": 0.0}
    correlations: list[dict[str, Any]] = []
    try:
        from autoalpha_v2 import knowledge_base as kb
        factors = [f for f in kb.get_all_factors() if f.get("run_id") != run_id and f.get("PassGates")]
        sample_dates = [
            _format_series_index(d)
            for d in list(dict.fromkeys(alpha_wide.index.map(_index_date_key)))[-60:]
        ]

        # Compute token similarity and find nearest in one pass
        token_sims: dict[str, float] = {}
        for item in factors:
            other_formula = item.get("formula", "")
            if other_formula == formula:
                continue
            other_tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", other_formula or ""))
            union = tokens | other_tokens
            sim = len(tokens & other_tokens) / len(union) if union else 0.0
            rid = item.get("run_id", "")
            token_sims[rid] = sim
            if sim > best["similarity"]:
                best = {"run_id": rid, "similarity": sim}

        # Parallel parquet reads for passing factors.  Keep the cap generous so
        # factor cards can show broad redundancy, while avoiding unbounded I/O.
        top_factors = sorted(factors, key=lambda r: r.get("Score", 0), reverse=True)[:max_corr_factors]

        def _read_and_corr(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            path = _existing_alpha_path(item)
            other_wide = _read_parquet_alpha_wide(str(path), sample_dates)
            corr, n_bars = _mean_bar_correlation(alpha_wide, other_wide)
            if not n_bars:
                return None
            rid = item.get("run_id", "")
            return {
                "run_id": rid,
                "corr": _round4(corr),
                "abs_corr": _round4(abs(corr)),
                "n_bars": n_bars,
                "score": _round4(item.get("Score", 0)),
                "formula_token_overlap": _round4(token_sims.get(rid, 0.0)),
            }

        with ThreadPoolExecutor(max_workers=min(4, len(top_factors))) as pool:
            for result in as_completed([pool.submit(_read_and_corr, f) for f in top_factors]):
                row = result.result()
                if row is not None:
                    correlations.append(row)

    except Exception:
        pass
    correlations.sort(key=lambda r: r.get("abs_corr", 0), reverse=True)
    return {
        "family": _formula_theme(formula),
        "max_formula_token_overlap": _round4(best["similarity"]),
        "nearest_factor": best["run_id"],
        "similarity_basis": "formula token Jaccard overlap",
        "alpha_corr_basis": "mean bar-wise Pearson corr on latest 60 shared days",
        "top_alpha_correlations": correlations[:12],
        "max_alpha_corr": correlations[0]["corr"] if correlations else 0.0,
        "max_abs_alpha_corr": correlations[0]["abs_corr"] if correlations else 0.0,
        "correlation_count": len(correlations),
    }


# ---------------------------------------------------------------------------
# Factor card
# ---------------------------------------------------------------------------

def build_factor_card(
    run_id: str,
    formula: str,
    metrics: Dict[str, Any],
    alpha_stats: Dict[str, Any],
    daily_ic: Optional[pd.Series] = None,
    temporal: Optional[Dict[str, Any]] = None,
    monthly_ic: Optional[Dict[str, Any]] = None,
    regimes: Optional[list[dict[str, Any]]] = None,
    redundancy: Optional[Dict[str, Any]] = None,
    thought_process: str = "",
) -> Dict[str, Any]:
    preview = metrics.get("result_preview") or {}
    pass_gates = bool(metrics.get("PassGates", False))
    ic_mean = _safe_float(metrics.get("IC", preview.get("IC", 0.0)))

    _ic = daily_ic if daily_ic is not None and len(daily_ic) else pd.Series(dtype=float)
    rolling_ic = _compress_series(_ic.rolling(20, min_periods=5).mean(), max_points=80)
    daily_ic_series = _compress_series(_ic, max_points=120)

    card: Dict[str, Any] = {
        "run_id": run_id,
        "title": run_id,
        "status": "PASS" if pass_gates else "REJECTED",
        "theme": _formula_theme(formula),
        "thesis": thought_process or "No agent thesis was recorded.",
        "definition": {
            "formula": formula,
            "inputs": _formula_inputs(formula),
            "update_frequency": "15-minute submission grid",
            "postprocess": metrics.get("postprocess", "rank/clip"),
        },
        "metrics": _round_dict({
            "IC": _safe_float(metrics.get("IC", preview.get("IC", 0.0))),
            "IR": _safe_float(metrics.get("IR", preview.get("IR", 0.0))),
            "tvr": _safe_float(metrics.get("Turnover", metrics.get("tvr", preview.get("tvr", 0.0)))),
            "Score": _safe_float(metrics.get("Score", preview.get("score", 0.0))),
            "nd": _safe_float(preview.get("nd", len(daily_ic) if daily_ic is not None else 0)),
        }),
        "distribution": _round_dict({k: v for k, v in alpha_stats.items() if not isinstance(v, list)}),
        "histogram": alpha_stats.get("histogram", []),
        "temporal": temporal or {},
        "prediction": {
            "ic_mean": _round4(ic_mean),
            "icir": _round4(metrics.get("IR", preview.get("IR", 0.0))),
            "rolling_ic": rolling_ic,
            "daily_ic": daily_ic_series,
        },
        "monthly_ic": monthly_ic or {},
        "regime": regimes or [],
        "redundancy": redundancy or {},
        "gate_notes": _gate_notes(metrics),
        "diagnostics": {
            "ic_mean": _round4(ic_mean),
            "daily_ic_count": len(daily_ic) if daily_ic is not None else 0,
            "alpha_mean": _safe_float(alpha_stats.get("mean", 0.0)),
            "alpha_std": _safe_float(alpha_stats.get("std", 0.0)),
            "pct_positive": _safe_float(alpha_stats.get("pct_positive", 0.0)),
            "pct_in_bounds": _safe_float(alpha_stats.get("pct_in_bounds", 0.0)),
        },
        "risk_notes": [
            f"tvr={_fmt(metrics.get('Turnover', 0), 2)}, "
            f"maxx={_fmt(metrics.get('maxx', 0), 2)}, minn={_fmt(metrics.get('minn', 0), 2)}.",
        ],
        "formula": formula,
        "created_at": pd.Timestamp.now().isoformat(),
    }
    return card


def write_factor_card(out_dir: Path, card: Dict[str, Any]) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    card_path = out_dir / "factor_card.json"
    card_path.write_text(json.dumps(card, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    metrics = card.get("metrics", {})
    diagnostics = card.get("diagnostics", {})
    gate_rows = "\n".join(f"- {item}" for item in card.get("gate_notes", []))
    monthly = card.get("monthly_ic", {})
    md = f"""# Factor Card: {card.get('run_id', '')}

## Snapshot
- Status: {card.get('status', '')}
- Theme: {card.get('theme', '')}
- IC / IR / TVR / Score: {_fmt(metrics.get('IC'))} / {_fmt(metrics.get('IR'))} / {_fmt(metrics.get('tvr'), 2)} / {_fmt(metrics.get('Score'), 2)}

## Agent Thesis
{card.get('thesis', '')}

## Gate Notes
{gate_rows}

## Diagnostics
- IC mean: {_fmt(diagnostics.get('ic_mean'))}
- Daily IC count: {diagnostics.get('daily_ic_count', 0)}
- Alpha mean/std: {_fmt(diagnostics.get('alpha_mean'), 5)} / {_fmt(diagnostics.get('alpha_std'), 5)}
- % Positive: {_fmt(diagnostics.get('pct_positive'), 3)}
- Full-sample IC: {_fmt(monthly.get('full_sample_ic'))}
- Positive month ratio: {_fmt(monthly.get('positive_month_ratio'))}

## Formula
```text
{card.get('formula', '')}
```
"""
    (out_dir / "factor_card.md").write_text(md, encoding="utf-8")
    return str(card_path)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

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
    Lightweight factor research analysis.
    Keeps: key characteristics, time-series appearance, basic analysis,
    inter-factor similarity.
    """
    out_dir = RESEARCH_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    alpha_stats = _compute_alpha_stats(alpha)
    daily_ic: pd.Series = pd.Series(dtype=float)
    temporal: Dict[str, Any] = {}
    regimes: list[dict[str, Any]] = []
    monthly_ic: Dict[str, Any] = {}
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

        alpha_f = _filter_allowed(alpha)
        resp_f = _filter_allowed(resp_slice[resp_col])
        rest_f = _filter_allowed(rest_slice[rest_col])

        # Build wide matrices once
        alpha_wide = alpha_f.unstack("security_id")
        resp_wide = resp_f.unstack("security_id").reindex_like(alpha_wide)
        rest_wide = rest_f.unstack("security_id").reindex_like(alpha_wide).fillna(0)

        daily_ic = _compute_daily_ic(alpha_wide, resp_wide, rest_wide)
        temporal = _compute_temporal(alpha_wide)
        regimes = _compute_regimes(daily_ic, resp_wide)
        monthly_ic = _compute_monthly_ic(daily_ic, resp_wide)
        redundancy = _compute_redundancy(formula, run_id, alpha_wide)

    except Exception as e:
        print(f"  [research] Analysis failed: {e}")

    factor_card = None
    card_path = ""
    if bool(metrics.get("PassGates", False)):
        factor_card = build_factor_card(
            run_id=run_id,
            formula=formula,
            metrics=metrics,
            alpha_stats=alpha_stats,
            daily_ic=daily_ic,
            temporal=temporal,
            monthly_ic=monthly_ic,
            regimes=regimes,
            redundancy=redundancy,
            thought_process=thought_process,
        )
        card_path = write_factor_card(out_dir, factor_card)

    report = {
        "run_id": run_id,
        "formula": formula,
        "metrics": metrics,
        "alpha_stats": alpha_stats,
        "daily_ic": {str(k): float(v) for k, v in daily_ic.items()},
        "n_daily_ic": len(daily_ic),
        "factor_card": factor_card,
        "factor_card_path": card_path,
        "created_at": pd.Timestamp.now().isoformat(),
    }
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    md = f"""# Factor Research: {run_id}

## Formula
```
{formula}
```

## Metrics
| Metric | Value |
|--------|-------|
| IC | {metrics.get('IC', 0):.4f} |
| IR | {metrics.get('IR', 0):.4f} |
| Turnover | {metrics.get('Turnover', 0):.2f} |
| Score | {metrics.get('Score', 0):.2f} |
| PassGates | {metrics.get('PassGates', False)} |

## Distribution
| Stat | Value |
|------|-------|
| Mean | {alpha_stats.get('mean', 0):.4f} |
| Std | {alpha_stats.get('std', 0):.4f} |
| Skewness | {alpha_stats.get('skew', 0):.4f} |
| Kurtosis | {alpha_stats.get('kurt', 0):.4f} |
| % Positive | {alpha_stats.get('pct_positive', 0):.2%} |
"""
    (out_dir / "report.md").write_text(md, encoding="utf-8")
    print(f"  [research] Report saved → {out_dir}")
    return str(out_dir)


def update_all_factor_card_correlations() -> int:
    """
    Recompute pairwise alpha correlations for ALL passing factor cards.
    Called after a new passing factor is found so every card's redundancy
    section stays current.  Returns the number of cards updated.
    """
    from autoalpha_v2 import knowledge_base as _kb

    all_passing = [f for f in _kb.get_all_factors() if f.get("PassGates") and f.get("research_path")]
    if len(all_passing) < 2:
        return 0

    expected_ids = [
        str(f.get("run_id", ""))
        for f in sorted(all_passing, key=lambda item: float(item.get("Score", 0) or 0), reverse=True)[:30]
    ]
    cache: Dict[str, Any] = {}
    if CORRELATION_CACHE_PATH.is_file():
        try:
            cache = json.loads(CORRELATION_CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            cache = {}
    if not cache or cache.get("run_ids") != expected_ids:
        cache = build_factor_correlation_matrix(max_factors=30, update_cards=False)
    cache_rows = {
        run_id: {
            other_id: corr
            for other_id, corr in zip(cache.get("run_ids", []), row)
            if other_id != run_id and corr is not None
        }
        for run_id, row in zip(cache.get("run_ids", []), cache.get("matrix", []))
    }

    updated = 0
    for factor in all_passing:
        run_id = factor.get("run_id", "")
        research_path = factor.get("research_path", "")
        if not research_path:
            continue
        out_dir = Path(str(research_path))
        card_path = out_dir / "factor_card.json"
        if not card_path.is_file():
            continue

        try:
            with open(card_path, encoding="utf-8") as fh:
                card = json.load(fh)

            formula = factor.get("formula", "")
            # Load the factor's own parquet to build alpha_wide sample
            tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", formula or ""))
            nearest = {"run_id": "", "similarity": 0.0}
            rows = []
            score_map = {f.get("run_id", ""): f for f in all_passing}
            for other_id, corr in sorted(
                cache_rows.get(run_id, {}).items(),
                key=lambda item: abs(float(item[1] or 0)),
                reverse=True,
            ):
                other = score_map.get(other_id, {})
                other_tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", other.get("formula", "") or ""))
                union = tokens | other_tokens
                token_overlap = len(tokens & other_tokens) / len(union) if union else 0.0
                if token_overlap > nearest["similarity"]:
                    nearest = {"run_id": other_id, "similarity": token_overlap}
                rows.append({
                    "run_id": other_id,
                    "corr": _round4(corr),
                    "abs_corr": _round4(abs(float(corr))),
                    "n_bars": cache.get("n_bars", {}).get(f"{run_id}|{other_id}", 0)
                    or cache.get("n_bars", {}).get(f"{other_id}|{run_id}", 0),
                    "score": _round4(other.get("Score", 0)),
                    "formula_token_overlap": _round4(token_overlap),
                })
            rows.sort(key=lambda row: row.get("abs_corr", 0), reverse=True)
            redundancy = {
                "family": _formula_theme(formula),
                "max_formula_token_overlap": _round4(nearest["similarity"]),
                "nearest_factor": nearest["run_id"],
                "similarity_basis": "formula token Jaccard overlap",
                "alpha_corr_basis": cache.get("basis", "mean bar-wise Pearson corr on latest 60 shared days"),
                "top_alpha_correlations": rows[:12],
                "max_alpha_corr": rows[0]["corr"] if rows else 0.0,
                "max_abs_alpha_corr": rows[0]["abs_corr"] if rows else 0.0,
                "correlation_count": len(rows),
            }
            if rows:
                redundancy["alpha_corr_basis"] = cache.get("basis", redundancy.get("alpha_corr_basis", ""))
            card["redundancy"] = redundancy
            with open(card_path, "w", encoding="utf-8") as fh:
                json.dump(card, fh, indent=2, ensure_ascii=False, default=str)
            updated += 1
        except Exception as e:
            print(f"  [research] Correlation update failed for {run_id}: {e}")

    print(f"  [research] Updated correlations for {updated}/{len(all_passing)} factor cards")
    return updated


def _select_low_corr_factors(
    factors: List[Dict[str, Any]],
    run_ids: List[str],
    matrix: List[List[Optional[float]]],
    max_abs_corr: float = 0.7,
) -> Dict[str, Any]:
    """
    Low-correlation submit set:
    maximize selected factor count first; among equal-size sets, prefer higher
    sorted Score sequence from the first factor onward.
    """
    factor_by_id = {str(f.get("run_id", "")): f for f in factors}
    candidate_ids = sorted(
        [rid for rid in run_ids if rid in factor_by_id],
        key=lambda rid: (
            float(factor_by_id[rid].get("Score", 0) or 0),
            float(factor_by_id[rid].get("IC", 0) or 0),
        ),
        reverse=True,
    )
    scores = {rid: float(factor_by_id[rid].get("Score", 0) or 0) for rid in candidate_ids}
    index_by_id = {rid: idx for idx, rid in enumerate(run_ids)}

    def _pair_abs_corr(left: str, right: str) -> Optional[float]:
        li = index_by_id.get(left)
        ri = index_by_id.get(right)
        if li is None or ri is None:
            return None
        try:
            corr = matrix[li][ri]
        except Exception:
            return None
        return abs(float(corr)) if corr is not None else None

    compatible: Dict[str, set[str]] = {rid: set() for rid in candidate_ids}
    for left in candidate_ids:
        for right in candidate_ids:
            if left == right:
                continue
            corr = _pair_abs_corr(left, right)
            if corr is not None and corr <= max_abs_corr:
                compatible[left].add(right)

    best: List[str] = []

    def _selection_key(ids: List[str]) -> tuple[int, tuple[float, ...], float, tuple[str, ...]]:
        ordered = sorted(ids, key=lambda rid: (scores.get(rid, 0.0), rid), reverse=True)
        return (
            len(ordered),
            tuple(scores.get(rid, 0.0) for rid in ordered),
            sum(scores.get(rid, 0.0) for rid in ordered),
            tuple(ordered),
        )

    def _consider(ids: List[str]) -> None:
        nonlocal best
        if _selection_key(ids) > _selection_key(best):
            best = list(ids)

    def _search(selected_ids: List[str], remaining: List[str]) -> None:
        _consider(selected_ids)
        if len(selected_ids) + len(remaining) < len(best):
            return
        if not remaining:
            return

        head = remaining[0]
        tail = remaining[1:]
        _search(
            selected_ids + [head],
            [rid for rid in tail if rid in compatible[head]],
        )
        _search(selected_ids, tail)

    _search([], candidate_ids)
    selected = sorted(best, key=lambda rid: (scores.get(rid, 0.0), rid), reverse=True)

    selected_rows = []
    for run_id in selected:
        factor = factor_by_id[run_id]
        max_selected_corr = max(
            (_pair_abs_corr(run_id, other) or 0.0 for other in selected if other != run_id),
            default=0.0,
        )
        selected_rows.append({
            "run_id": run_id,
            "label": run_id.replace("autoalpha_", "").replace("_0", "_"),
            "score": _round4(factor.get("Score", 0)),
            "ic": _round4(factor.get("IC", 0)),
            "max_abs_corr_to_selected": _round4(max_selected_corr),
        })

    return {
        "threshold": max_abs_corr,
        "method": "maximize count first; tie-break lexicographically by descending Score under abs(pairwise corr) <= threshold",
        "count": len(selected_rows),
        "total_score": _round4(sum(float(row["score"] or 0) for row in selected_rows)),
        "factors": selected_rows,
    }


def build_factor_correlation_matrix(max_factors: int = 15, update_cards: bool = False) -> Dict[str, Any]:
    """
    Compute a compact pairwise realized-alpha correlation matrix for passing factors.
    Uses the latest 60 shared days, then writes a cache consumed by the records page.
    """
    from autoalpha_v2 import knowledge_base as _kb

    passing = [
        f for f in _kb.get_all_factors()
        if f.get("PassGates") and _existing_alpha_path(f)
    ]
    passing.sort(key=lambda item: float(item.get("Score", 0) or 0), reverse=True)
    passing = passing[:max_factors]
    run_ids = [str(f.get("run_id", "")) for f in passing]
    labels = [rid.replace("autoalpha_", "").replace("_0", "_") for rid in run_ids]
    if len(passing) < 2:
        payload = {
            "labels": labels,
            "run_ids": run_ids,
            "matrix": [[1.0] for _ in run_ids] if run_ids else [],
            "low_corr_selection": _select_low_corr_factors(
                passing, run_ids, [[1.0] for _ in run_ids] if run_ids else []
            ),
            "sample_dates": [],
            "n_bars": {},
            "basis": "mean bar-wise Pearson corr on latest 60 shared days",
            "updated_at": pd.Timestamp.now().isoformat(),
        }
        CORRELATION_CACHE_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return payload

    sample_dates: list[str] = []
    first_path = _existing_alpha_path(passing[0])
    try:
        date_frame = pd.read_parquet(str(first_path), columns=["date"])
        dates = pd.to_datetime(date_frame["date"]).dt.strftime("%Y-%m-%d").drop_duplicates().sort_values()
        sample_dates = dates.iloc[-60:].tolist()
    except Exception:
        sample_dates = []

    wides: dict[str, pd.DataFrame] = {}
    for factor in passing:
        path = _existing_alpha_path(factor)
        wide = _read_parquet_alpha_wide(str(path), sample_dates or None)
        if not wide.empty:
            wides[str(factor.get("run_id", ""))] = wide

    matrix: list[list[Optional[float]]] = []
    n_bars: dict[str, int] = {}
    for row_id in run_ids:
        row: list[Optional[float]] = []
        for col_id in run_ids:
            if row_id == col_id:
                row.append(1.0 if row_id in wides else None)
                continue
            a = wides.get(row_id)
            b = wides.get(col_id)
            if a is None or b is None:
                row.append(None)
                continue
            corr, bars = _mean_bar_correlation(a, b)
            row.append(_round4(corr) if bars else None)
            n_bars[f"{row_id}|{col_id}"] = bars
        matrix.append(row)

    payload = {
        "labels": labels,
        "run_ids": run_ids,
        "matrix": matrix,
        "low_corr_selection": _select_low_corr_factors(passing, run_ids, matrix),
        "sample_dates": sample_dates,
        "n_bars": n_bars,
        "basis": "mean bar-wise Pearson corr on latest 60 shared days",
        "updated_at": pd.Timestamp.now().isoformat(),
    }
    CORRELATION_CACHE_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if update_cards:
        update_all_factor_card_correlations()
    return payload
