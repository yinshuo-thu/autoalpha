"""
autoalpha_v2/factor_research.py

Lightweight factor research analysis.
Computes: key characteristics, time-series appearance, basic analysis,
and inter-factor similarity.  No heavy per-bar Python loops.
"""

from __future__ import annotations

import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

RESEARCH_DIR = Path(__file__).resolve().parent / "research"
CORRELATION_CACHE_PATH = Path(__file__).resolve().parent / "factor_correlations.json"
DEFAULT_CORRELATION_MATRIX_FACTORS = 0

ALLOWED_UTC_TIMES = {
    "01:45:00", "02:00:00", "02:15:00", "02:30:00",
    "02:45:00", "03:00:00", "03:15:00", "03:30:00",
    "05:15:00", "05:30:00", "05:45:00", "06:00:00",
    "06:15:00", "06:30:00", "06:45:00", "07:00:00",
}

_correlation_refresh_lock = threading.Lock()
_correlation_refresh_thread: threading.Thread | None = None
_correlation_refresh_update_cards = False


def _write_correlation_cache(payload: Dict[str, Any]) -> None:
    tmp_path = CORRELATION_CACHE_PATH.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(CORRELATION_CACHE_PATH)


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


def _read_daily_alpha_wide(path: str, sample_dates: Optional[list[str]]) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    try:
        kwargs: dict[str, Any] = {"columns": ["date", "security_id", "alpha"]}
        if sample_dates:
            kwargs["filters"] = [("date", "in", sample_dates)]
        frame = pd.read_parquet(path, **kwargs)
    except Exception:
        return pd.DataFrame()
    if frame.empty or "alpha" not in frame.columns:
        return pd.DataFrame()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    if sample_dates:
        frame = frame[frame["date"].dt.strftime("%Y-%m-%d").isin(sample_dates)]
    if frame.empty:
        return pd.DataFrame()
    daily = (
        frame.groupby(["date", "security_id"], sort=True)["alpha"]
        .mean()
        .astype("float32")
        .unstack("security_id")
        .sort_index()
    )
    return daily


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


def _load_correlation_cache() -> Dict[str, Any]:
    if CORRELATION_CACHE_PATH.is_file():
        try:
            payload = json.loads(CORRELATION_CACHE_PATH.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload
        except Exception:
            pass
    return {}


def _load_passing_factors() -> List[Dict[str, Any]]:
    from autoalpha_v2 import knowledge_base as _kb

    return [
        factor
        for factor in _kb.get_all_factors()
        if factor.get("PassGates") and _existing_alpha_path(factor)
    ]


def _sort_factors_for_trend(factors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        [factor for factor in factors if factor.get("run_id")],
        key=lambda item: (str(item.get("created_at", "")), str(item.get("run_id", ""))),
    )


def _sort_factors_for_heatmap(factors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return _sort_factors_for_trend(factors)


def _limit_correlation_factors(
    factors: List[Dict[str, Any]],
    max_factors: int,
) -> List[Dict[str, Any]]:
    if max_factors and max_factors > 0:
        return factors[:max_factors]
    return factors


def _expected_trend_run_ids(factors: List[Dict[str, Any]]) -> List[str]:
    return [str(factor.get("run_id", "")) for factor in _sort_factors_for_trend(factors)]


def _expected_heatmap_run_ids(factors: List[Dict[str, Any]], max_factors: int) -> List[str]:
    return [
        str(factor.get("run_id", ""))
        for factor in _limit_correlation_factors(_sort_factors_for_heatmap(factors), max_factors)
    ]


def _correlation_cache_status(
    payload: Dict[str, Any],
    max_factors: int = DEFAULT_CORRELATION_MATRIX_FACTORS,
    passing: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {"heatmap_stale": True, "trend_stale": True}

    passing = passing if passing is not None else _load_passing_factors()
    expected_trend_ids = _expected_trend_run_ids(passing)
    expected_heatmap_ids = _expected_heatmap_run_ids(passing, max_factors)

    heatmap_stale = not isinstance(payload.get("matrix"), list)
    if not heatmap_stale:
        actual_heatmap_ids = [str(run_id) for run_id in payload.get("run_ids", []) if run_id]
        heatmap_stale = (
            actual_heatmap_ids != expected_heatmap_ids
            or len(payload.get("matrix", [])) != len(actual_heatmap_ids)
        )

    trend_rows = payload.get("trend_rows")
    actual_trend_ids = []
    if isinstance(trend_rows, list):
        actual_trend_ids = [
            str(row.get("run_id", ""))
            for row in trend_rows
            if isinstance(row, dict) and row.get("run_id")
        ]
    trend_stale = actual_trend_ids != expected_trend_ids

    return {"heatmap_stale": heatmap_stale, "trend_stale": trend_stale}


def _resolve_correlation_sample_dates(
    factors: List[Dict[str, Any]],
    existing_cache: Optional[Dict[str, Any]] = None,
) -> list[str]:
    cached_dates = []
    if isinstance(existing_cache, dict):
        cached_dates = [
            str(item)
            for item in existing_cache.get("sample_dates", [])
            if str(item)
        ]
    if cached_dates:
        return cached_dates[-60:]

    if not factors:
        return []

    latest_factor = _sort_factors_for_trend(factors)[-1]
    latest_path = _existing_alpha_path(latest_factor)
    if not latest_path:
        return []

    try:
        parquet = pq.ParquetFile(str(latest_path))
        if "date" in parquet.schema.names:
            unique_dates: set[str] = set()
            collected: list[str] = []
            for row_group_idx in range(parquet.metadata.num_row_groups - 1, -1, -1):
                table = parquet.read_row_group(row_group_idx, columns=["date"])
                if table.num_rows <= 0:
                    continue
                chunk_dates = (
                    pd.to_datetime(table.column("date").to_pandas())
                    .dt.strftime("%Y-%m-%d")
                    .drop_duplicates()
                    .tolist()
                )
                for date_str in reversed(chunk_dates):
                    if date_str and date_str not in unique_dates:
                        unique_dates.add(date_str)
                        collected.append(date_str)
                if len(unique_dates) >= 60:
                    break
            if collected:
                return sorted(collected)[-60:]
    except Exception:
        pass

    try:
        date_frame = pd.read_parquet(str(latest_path), columns=["date"])
        dates = pd.to_datetime(date_frame["date"]).dt.strftime("%Y-%m-%d").drop_duplicates().sort_values()
        return dates.iloc[-60:].tolist()
    except Exception:
        return []


def refresh_factor_correlation_trend_rows(
    payload: Optional[Dict[str, Any]] = None,
    *,
    factors: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    payload = dict(payload or _load_correlation_cache())
    passing = factors if factors is not None else _load_passing_factors()
    sample_dates = _resolve_correlation_sample_dates(passing, existing_cache=payload)
    payload["trend_rows"] = _build_factor_correlation_trend_rows(
        passing,
        sample_dates,
        existing_rows=payload.get("trend_rows"),
    )
    payload["sample_dates"] = sample_dates
    payload["trend_basis"] = (
        "corr to prior passing factors using pooled daily cross-sectional Pearson corr on latest 60 shared days"
    )
    payload["updated_at"] = pd.Timestamp.now().isoformat()
    _validate_correlation_payload(
        payload,
        max_factors=DEFAULT_CORRELATION_MATRIX_FACTORS,
        passing=passing,
        require_heatmap=False,
    )
    _write_correlation_cache(payload)
    return payload


def load_factor_correlation_cache(
    max_factors: int = DEFAULT_CORRELATION_MATRIX_FACTORS,
    refresh: bool = False,
) -> Dict[str, Any]:
    if not refresh:
        payload = _load_correlation_cache()
        if payload:
            passing = _load_passing_factors()
            status = _correlation_cache_status(payload, max_factors=max_factors, passing=passing)
            if not status["trend_stale"] and not status["heatmap_stale"]:
                return payload
            if status["heatmap_stale"]:
                return build_factor_correlation_matrix(max_factors=max_factors)
            if status["trend_stale"]:
                return refresh_factor_correlation_trend_rows(payload, factors=passing)
    return build_factor_correlation_matrix(max_factors=max_factors)


def schedule_factor_correlation_refresh(
    *,
    update_cards: bool = False,
    max_factors: int = DEFAULT_CORRELATION_MATRIX_FACTORS,
) -> bool:
    """
    Refresh correlations in a background thread.
    Returns True if a new worker was started, False if an existing worker will
    pick up the refresh request.
    """
    global _correlation_refresh_thread, _correlation_refresh_update_cards

    def _runner() -> None:
        global _correlation_refresh_thread, _correlation_refresh_update_cards
        while True:
            with _correlation_refresh_lock:
                requested_update_cards = _correlation_refresh_update_cards
                _correlation_refresh_update_cards = False
            try:
                build_factor_correlation_matrix(
                    max_factors=max_factors,
                    update_cards=requested_update_cards,
                )
            except Exception as exc:
                print(f"[factor_research] Background correlation refresh failed: {exc}")
            with _correlation_refresh_lock:
                if _correlation_refresh_update_cards:
                    continue
                _correlation_refresh_thread = None
                break

    with _correlation_refresh_lock:
        _correlation_refresh_update_cards = _correlation_refresh_update_cards or update_cards
        if _correlation_refresh_thread is not None and _correlation_refresh_thread.is_alive():
            return False
        _correlation_refresh_thread = threading.Thread(
            target=_runner,
            name="autoalpha-correlation-refresh",
            daemon=True,
        )
        _correlation_refresh_thread.start()
        return True


def read_factor_correlation_cache_snapshot(
    *,
    max_factors: int = DEFAULT_CORRELATION_MATRIX_FACTORS,
    schedule_refresh: bool = False,
) -> Dict[str, Any]:
    payload = _load_correlation_cache() or {}
    passing = _load_passing_factors()
    status = _correlation_cache_status(payload, max_factors=max_factors, passing=passing)
    heatmap_ids = [str(run_id) for run_id in payload.get("run_ids", []) if run_id]
    trend_ids = [
        str(row.get("run_id", ""))
        for row in payload.get("trend_rows", [])
        if isinstance(row, dict) and row.get("run_id")
    ] if isinstance(payload.get("trend_rows"), list) else []

    response = dict(payload) if isinstance(payload, dict) else {}
    response.setdefault("labels", [])
    response.setdefault("run_ids", [])
    response.setdefault("matrix", [])
    response.setdefault("trend_rows", [])
    response.setdefault("updated_at", "")
    response["meta"] = {
        "heatmap_stale": status["heatmap_stale"],
        "trend_stale": status["trend_stale"],
        "expected_heatmap_count": len(_expected_heatmap_run_ids(passing, max_factors)),
        "expected_trend_count": len(_expected_trend_run_ids(passing)),
        "actual_heatmap_count": len(heatmap_ids),
        "actual_trend_count": len(trend_ids),
        "refreshing": False,
    }

    if schedule_refresh and (status["heatmap_stale"] or status["trend_stale"]):
        started = schedule_factor_correlation_refresh(
            update_cards=True,
            max_factors=max_factors,
        )
        response["meta"]["refreshing"] = started or bool(
            _correlation_refresh_thread is not None and _correlation_refresh_thread.is_alive()
        )

    return response


def _validate_correlation_payload(
    payload: Dict[str, Any],
    *,
    max_factors: int,
    passing: List[Dict[str, Any]],
    require_heatmap: bool = True,
) -> None:
    expected_heatmap_ids = _expected_heatmap_run_ids(passing, max_factors)
    expected_trend_ids = _expected_trend_run_ids(passing)

    actual_heatmap_ids = [str(run_id) for run_id in payload.get("run_ids", []) if run_id]
    actual_trend_ids = [
        str(row.get("run_id", ""))
        for row in payload.get("trend_rows", [])
        if isinstance(row, dict) and row.get("run_id")
    ] if isinstance(payload.get("trend_rows"), list) else []

    if actual_trend_ids != expected_trend_ids:
        raise ValueError(
            f"trend_rows mismatch: expected {len(expected_trend_ids)} ids, got {len(actual_trend_ids)}"
        )

    if not require_heatmap:
        return

    if actual_heatmap_ids != expected_heatmap_ids:
        raise ValueError(
            f"heatmap run_ids mismatch: expected {len(expected_heatmap_ids)} ids, got {len(actual_heatmap_ids)}"
        )

    matrix = payload.get("matrix", [])
    if not isinstance(matrix, list) or len(matrix) != len(actual_heatmap_ids):
        raise ValueError(
            f"heatmap matrix row mismatch: expected {len(actual_heatmap_ids)}, got {len(matrix) if isinstance(matrix, list) else 'invalid'}"
        )
    for row in matrix:
        if not isinstance(row, list) or len(row) != len(actual_heatmap_ids):
            raise ValueError(
                f"heatmap matrix column mismatch: expected {len(actual_heatmap_ids)}"
            )


def _build_factor_correlation_trend_rows(
    factors: List[Dict[str, Any]],
    sample_dates: Optional[list[str]],
    existing_rows: List[Dict[str, Any]] | None = None,
) -> List[Dict[str, Any]]:
    ordered_factors = _sort_factors_for_trend(factors)
    if not ordered_factors:
        return []

    existing_by_id = {
        str(row.get("run_id", "")): row
        for row in (existing_rows or [])
        if isinstance(row, dict) and row.get("run_id")
    }

    normalized_series: list[pd.Series] = []
    for factor in ordered_factors:
        run_id = str(factor.get("run_id", ""))
        path = _existing_alpha_path(factor)
        if not path:
            continue
        daily = _read_daily_alpha_wide(path, sample_dates or None)
        if not daily.empty:
            demeaned = daily.sub(daily.mean(axis=1), axis=0)
            scaled = demeaned.div(daily.std(axis=1).replace(0.0, np.nan), axis=0)
            stacked = scaled.stack(future_stack=True).replace([np.inf, -np.inf], np.nan)
            if not stacked.empty:
                normalized_series.append(stacked.rename(run_id).astype("float32"))

    corr_matrix = pd.DataFrame()
    if normalized_series:
        aligned = pd.concat(normalized_series, axis=1, join="outer")
        corr_matrix = aligned.corr(method="pearson", min_periods=200)

    trend_rows: list[Dict[str, Any]] = []
    prior_run_ids: list[str] = []
    for index, factor in enumerate(ordered_factors, start=1):
        run_id = str(factor.get("run_id", ""))
        row: Dict[str, Any]
        if run_id in corr_matrix.columns and prior_run_ids:
            values = corr_matrix[run_id].reindex(prior_run_ids).dropna()
            if not values.empty:
                row = {
                    "run_id": run_id,
                    "avg_corr": _round4(float(values.mean())),
                    "max_corr": _round4(float(values.max())),
                    "min_corr": _round4(float(values.min())),
                    "pair_count": int(values.count()),
                    "basis": "corr to prior passing factors using pooled daily cross-sectional Pearson corr on latest 60 shared days",
                }
            else:
                row = existing_by_id.get(run_id) or {
                    "run_id": run_id,
                    "avg_corr": 0.0,
                    "max_corr": 0.0,
                    "min_corr": 0.0,
                    "pair_count": 0,
                    "basis": "corr to prior passing factors using pooled daily cross-sectional Pearson corr on latest 60 shared days",
                }
        else:
            row = {
                "run_id": run_id,
                "avg_corr": 0.0,
                "max_corr": 0.0,
                "min_corr": 0.0,
                "pair_count": 0,
                "basis": "first passing factor has no prior factor correlations",
            }
        trend_rows.append({
            "index": index,
            "label": f"#{index}",
            "run_id": run_id,
            "created_at": factor.get("created_at", ""),
            "generation": int(factor.get("generation", 0) or 0),
            "tested_index": index,
            "score": _round4(factor.get("Score", 0)),
            "ic": _round4(factor.get("IC", 0)),
            "avg_corr": _round4(row.get("avg_corr", 0)),
            "max_corr": _round4(row.get("max_corr", 0)),
            "min_corr": _round4(row.get("min_corr", 0)),
            "pair_count": int(row.get("pair_count", 0) or 0),
            "basis": row.get("basis", "corr to prior passing factors using pooled daily cross-sectional Pearson corr on latest 60 shared days"),
        })
        prior_run_ids.append(run_id)
    return trend_rows


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
        top_factors = _sort_factors_for_heatmap(factors)[:max_corr_factors]

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

    expected_ids = _expected_heatmap_run_ids(all_passing, DEFAULT_CORRELATION_MATRIX_FACTORS)
    cache: Dict[str, Any] = {}
    if CORRELATION_CACHE_PATH.is_file():
        try:
            cache = json.loads(CORRELATION_CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            cache = {}
    if not cache or cache.get("run_ids") != expected_ids:
        cache = build_factor_correlation_matrix(max_factors=DEFAULT_CORRELATION_MATRIX_FACTORS, update_cards=False)
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


def _compute_pairwise_correlation_payload(
    factors: List[Dict[str, Any]],
    *,
    sample_dates: Optional[list[str]],
) -> Dict[str, Any]:
    ordered = list(factors)
    run_ids = [str(f.get("run_id", "")) for f in ordered]
    labels = [rid.replace("autoalpha_", "").replace("_0", "_") for rid in run_ids]

    if not ordered:
        return {
            "run_ids": [],
            "labels": [],
            "matrix": [],
            "n_bars": {},
            "wides": {},
        }

    wides: dict[str, pd.DataFrame] = {}

    def _load_wide(factor: Dict[str, Any]) -> tuple[str, pd.DataFrame]:
        run_id = str(factor.get("run_id", ""))
        path = _existing_alpha_path(factor)
        if not path:
            return run_id, pd.DataFrame()
        return run_id, _read_parquet_alpha_wide(str(path), sample_dates or None)

    max_workers = min(12, max(1, len(ordered)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_run = {
            executor.submit(_load_wide, factor): str(factor.get("run_id", ""))
            for factor in ordered
        }
        for future in as_completed(future_to_run):
            run_id, wide = future.result()
            if run_id and not wide.empty:
                wides[run_id] = wide

    matrix: list[list[Optional[float]]] = [[None] * len(run_ids) for _ in run_ids]
    n_bars: dict[str, int] = {}
    for row_index, row_id in enumerate(run_ids):
        for col_index in range(row_index, len(run_ids)):
            col_id = run_ids[col_index]
            if row_id == col_id:
                matrix[row_index][col_index] = 1.0 if row_id in wides else None
                continue
            a = wides.get(row_id)
            b = wides.get(col_id)
            if a is None or b is None:
                continue
            corr, bars = _mean_bar_correlation(a, b)
            rounded = _round4(corr) if bars else None
            matrix[row_index][col_index] = rounded
            matrix[col_index][row_index] = rounded
            n_bars[f"{row_id}|{col_id}"] = bars
            n_bars[f"{col_id}|{row_id}"] = bars

    return {
        "run_ids": run_ids,
        "labels": labels,
        "matrix": matrix,
        "n_bars": n_bars,
        "wides": wides,
    }


def build_factor_correlation_matrix(
    max_factors: int = DEFAULT_CORRELATION_MATRIX_FACTORS,
    update_cards: bool = False,
) -> Dict[str, Any]:
    """
    Compute a compact pairwise realized-alpha correlation matrix for passing factors.
    Uses the latest 60 shared days, then writes a cache consumed by the records page.
    """
    existing_cache = _load_correlation_cache()
    passing_all = _load_passing_factors()
    ordered_passing = _sort_factors_for_heatmap(passing_all)
    passing = _limit_correlation_factors(ordered_passing, max_factors)

    sample_dates = _resolve_correlation_sample_dates(passing_all, existing_cache=existing_cache)

    trend_rows = _build_factor_correlation_trend_rows(
        passing_all,
        sample_dates,
        existing_rows=existing_cache.get("trend_rows"),
    )
    heatmap_payload = _compute_pairwise_correlation_payload(
        passing,
        sample_dates=sample_dates,
    )
    if passing == ordered_passing:
        selection_payload = heatmap_payload
    else:
        selection_payload = _compute_pairwise_correlation_payload(
            ordered_passing,
            sample_dates=sample_dates,
        )

    if len(passing) < 2:
        payload = {
            "labels": heatmap_payload["labels"],
            "run_ids": heatmap_payload["run_ids"],
            "matrix": heatmap_payload["matrix"] or ([[1.0] for _ in heatmap_payload["run_ids"]] if heatmap_payload["run_ids"] else []),
            "low_corr_selection": _select_low_corr_factors(
                ordered_passing,
                selection_payload["run_ids"],
                selection_payload["matrix"] or ([[1.0] for _ in selection_payload["run_ids"]] if selection_payload["run_ids"] else []),
            ),
            "sample_dates": [],
            "n_bars": heatmap_payload["n_bars"],
            "basis": "mean bar-wise Pearson corr on latest 60 shared days",
            "trend_rows": trend_rows,
            "trend_basis": "corr to prior passing factors using pooled daily cross-sectional Pearson corr on latest 60 shared days",
            "updated_at": pd.Timestamp.now().isoformat(),
        }
        _validate_correlation_payload(payload, max_factors=max_factors, passing=passing_all)
        _write_correlation_cache(payload)
        return payload

    payload = {
        "labels": heatmap_payload["labels"],
        "run_ids": heatmap_payload["run_ids"],
        "matrix": heatmap_payload["matrix"],
        "low_corr_selection": _select_low_corr_factors(
            ordered_passing,
            selection_payload["run_ids"],
            selection_payload["matrix"],
        ),
        "sample_dates": sample_dates,
        "n_bars": heatmap_payload["n_bars"],
        "basis": "mean bar-wise Pearson corr on latest 60 shared days",
        "trend_rows": trend_rows,
        "trend_basis": "corr to prior passing factors using pooled daily cross-sectional Pearson corr on latest 60 shared days",
        "updated_at": pd.Timestamp.now().isoformat(),
    }
    _validate_correlation_payload(payload, max_factors=max_factors, passing=passing_all)
    _write_correlation_cache(payload)
    if update_cards:
        update_all_factor_card_correlations()
    return payload
