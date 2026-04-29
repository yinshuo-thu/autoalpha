"""
autoalpha_v3/pipeline.py

End-to-end pipeline:
  1. LLM generates formula idea (Claude via third-party relay)
  2. formula_validator checks syntax + future-leakage
  3. DataHub provides 15m OHLCV data
  4. FormulaEngine evaluates the DSL expression
  5. evaluator computes IC / IR / turnover metrics
  6. SubmissionBuilder exports alpha to parquet

No mocks — all steps use real data and a real API call.
"""

from __future__ import annotations

import gc
import json
import os
import re
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# ── project root on sys.path ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
PROJECT_PARENT = PROJECT_ROOT.parent
for _path in (PROJECT_PARENT, PROJECT_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from autoalpha_v3 import factor_research
from autoalpha_v3 import knowledge_base as kb
from autoalpha_v3.error_utils import AutoAlphaRuntimeError, humanize_error
from autoalpha_v3.inspiration_db import record_pass as record_inspiration_pass, record_usage as record_inspiration_usage
from autoalpha_v3.llm_client import generate_idea, summarize_factor_tldr
from core.evaluator import evaluate_submission_like_wide
from core.feishu_bot import FeishuNotifier
from formula_validator import validate_formula
from prepare_data import DataHub, get_trading_days
from factors import operators as _ops
from paths import CACHE_ROOT
from runtime_config import load_runtime_config

# Feishu notifications are enabled only when FEISHU_WEBHOOK_URL is configured.
_feishu = FeishuNotifier()

AUTOALPHA_OUT = PROJECT_ROOT / "output"
AUTOALPHA_TRACE_DIR = PROJECT_ROOT / "process_logs"
DISCOVERY_START = "2022-01-01"
DISCOVERY_END = "2023-12-31"
OOS_START = "2024-01-01"
OOS_END = "2024-12-31"
_DATAHUB_CACHE: dict[tuple[str, str], DataHub] = {}
_EVAL_WIDE_CACHE: dict[tuple[int, str, str, int], tuple[pd.DataFrame, pd.DataFrame]] = {}
_EVAL_WIDE_CACHE_ORDER: list[tuple[int, str, str, int]] = []
_EVAL_WIDE_CACHE_MAX = 3
_PV_DAYS_CACHE: dict[int, list[str]] = {}
_PV_SLICE_CACHE: dict[tuple[int, str, str], pd.DataFrame] = {}
_PV_SLICE_CACHE_ORDER: list[tuple[int, str, str]] = []
_PV_SLICE_CACHE_MAX = 3


def _futures_mode() -> bool:
    return os.environ.get("AUTOALPHA_ASSET_CLASS", "futures").strip().lower() in {"future", "futures"}


def _split_research_days(all_days: list[str], eval_days_count: int) -> tuple[list[str], list[str], list[str], list[str], str]:
    """Return eval/requested/oos/full-export days, adapting v3 date windows to futures data."""
    discovery_days = _filter_days(all_days, DISCOVERY_START, DISCOVERY_END)
    if discovery_days:
        requested = discovery_days[-eval_days_count:] if eval_days_count > 0 else discovery_days
        oos_days = _filter_days(all_days, OOS_START, OOS_END)
        full_export = _filter_days(all_days, DISCOVERY_START, OOS_END)
        return discovery_days, requested, oos_days, full_export, "v3_train_2022_2023_oos_2024_report_only"

    if not _futures_mode():
        return [], [], [], [], "v3_train_2022_2023_oos_2024_report_only"

    ordered = sorted(all_days)
    split = max(1, int(len(ordered) * 0.7))
    split = min(split, len(ordered))
    discovery_days = ordered[:split]
    if eval_days_count > 0:
        requested = discovery_days[-min(eval_days_count, len(discovery_days)):]
    else:
        requested = discovery_days
    oos_days = ordered[split:]
    return discovery_days, requested, oos_days, ordered, "futures_adaptive_discovery_oos"


def _get_data_hub(start: str | None = None, end: str | None = None) -> DataHub:
    key = (start or "__full__", end or "__full__")
    cached = _DATAHUB_CACHE.get(key)
    if cached is not None:
        print(f"[pipeline] Reusing cached DataHub ({key[0]} → {key[1]})")
        return cached
    hub = DataHub(start=start, end=end) if start or end else DataHub()
    _DATAHUB_CACHE.clear()
    _DATAHUB_CACHE[key] = hub
    return hub


def _get_eval_wide_frames(hub: DataHub, days: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    key = (id(hub), days[0], days[-1], len(days))
    cached = _EVAL_WIDE_CACHE.get(key)
    if cached is not None:
        return cached

    day_keys = {pd.to_datetime(d).strftime("%Y-%m-%d") for d in days}
    resp_df = hub.resp
    rest_df = hub.trading_restriction

    resp_dates = pd.to_datetime(resp_df.index.get_level_values("date")).strftime("%Y-%m-%d")
    resp_slice = resp_df.loc[pd.Index(resp_dates).isin(day_keys)]
    rest_dates = pd.to_datetime(rest_df.index.get_level_values("date")).strftime("%Y-%m-%d")
    rest_slice = rest_df.loc[pd.Index(rest_dates).isin(day_keys)]

    resp_col = "resp" if "resp" in resp_df.columns else resp_df.columns[0]
    rest_col = "trading_restriction" if "trading_restriction" in rest_df.columns else rest_df.columns[0]

    resp_un = resp_slice[resp_col].unstack("security_id")
    rest_un = rest_slice[rest_col].unstack("security_id")

    _EVAL_WIDE_CACHE[key] = (resp_un, rest_un)
    _EVAL_WIDE_CACHE_ORDER.append(key)
    while len(_EVAL_WIDE_CACHE_ORDER) > _EVAL_WIDE_CACHE_MAX:
        old_key = _EVAL_WIDE_CACHE_ORDER.pop(0)
        _EVAL_WIDE_CACHE.pop(old_key, None)
    return resp_un, rest_un


def _pv_trading_days(pv: pd.DataFrame) -> list[str]:
    key = id(pv)
    cached = _PV_DAYS_CACHE.get(key)
    if cached is not None:
        return cached
    days = sorted(
        pd.to_datetime(pv.index.get_level_values("date").unique()).strftime("%Y-%m-%d").tolist()
    )
    _PV_DAYS_CACHE.clear()
    _PV_DAYS_CACHE[key] = days
    return days


def _slice_pv_window(pv: pd.DataFrame, warmup_ts: pd.Timestamp, eval_end_ts: pd.Timestamp) -> pd.DataFrame:
    key = (id(pv), warmup_ts.strftime("%Y-%m-%d"), eval_end_ts.strftime("%Y-%m-%d"))
    cached = _PV_SLICE_CACHE.get(key)
    if cached is not None:
        return cached

    date_keys = pd.to_datetime(pv.index.get_level_values("date")).strftime("%Y-%m-%d")
    warmup_key = warmup_ts.strftime("%Y-%m-%d")
    eval_end_key = eval_end_ts.strftime("%Y-%m-%d")
    sub_pv = pv.loc[(date_keys >= warmup_key) & (date_keys <= eval_end_key)]
    max_rows = int(os.environ.get("AUTOALPHA_PV_SLICE_CACHE_MAX_ROWS", "18000000") or 18_000_000)
    if len(sub_pv) <= max_rows:
        _PV_SLICE_CACHE[key] = sub_pv
        _PV_SLICE_CACHE_ORDER.append(key)
        while len(_PV_SLICE_CACHE_ORDER) > _PV_SLICE_CACHE_MAX:
            old_key = _PV_SLICE_CACHE_ORDER.pop(0)
            _PV_SLICE_CACHE.pop(old_key, None)
    return sub_pv

# Extended operator registry: superset of core/formula_engine.py OPS_REGISTRY
# Adds neg, clip, ts_corr, ts_cov, and infix aliases that the LLM may emit.
_EXT_OPS = {
    "lag":           _ops.lag,
    "delay":         _ops.lag,
    "delta":         _ops.delta,
    "ts_mean":       _ops.ts_mean,
    "ts_std":        _ops.ts_std,
    "ts_sum":        _ops.ts_sum,
    "ts_max":        _ops.ts_max,
    "ts_min":        _ops.ts_min,
    "ts_median":     _ops.ts_median,
    "ts_quantile":   _ops.ts_quantile,
    "ts_skew":       _ops.ts_skew,
    "ts_kurt":       _ops.ts_kurt,
    "ts_ema":        _ops.ts_ema,
    "ts_argmax":     _ops.ts_argmax,
    "ts_argmin":     _ops.ts_argmin,
    "ts_pct_change": _ops.ts_pct_change,
    "ts_minmax_norm": _ops.ts_minmax_norm,
    "ts_zscore":     _ops.ts_zscore,
    "ts_rank":       _ops.ts_rank,
    "ts_decay_linear": _ops.ts_decay_linear,
    "decay_linear":  _ops.ts_decay_linear,
    "ts_corr":       _ops.ts_corr,
    "ts_cov":        _ops.ts_cov,
    "cs_rank":       _ops.cs_rank,
    "rank":          _ops.cs_rank,
    "cs_demean":     _ops.cs_demean,
    "demean":        _ops.cs_demean,
    "cs_zscore":     _ops.cs_zscore,
    "zscore":        _ops.cs_zscore,
    "cs_scale":      _ops.cs_scale,
    "scale":         _ops.cs_scale,
    "cs_winsorize":  _ops.cs_winsorize,
    "winsorize":     _ops.cs_winsorize,
    "cs_quantile":   _ops.cs_quantile,
    "cs_neutralize": _ops.cs_neutralize,
    "safe_div":      _ops.safe_div,
    "div":           _ops.safe_div,
    "signed_power":  _ops.signed_power,
    "pow":           _ops.signed_power,
    "abs":           np.abs,
    "sign":          np.sign,
    "log":           _ops.safe_log,
    "signed_log":    _ops.signed_log,
    "sqrt":          _ops.safe_sqrt,
    "sigmoid":       _ops.sigmoid,
    "tanh":          np.tanh,
    "neg":           lambda x: -x,
    "clip":          _ops.clamp,
    "clamp":         _ops.clamp,
    "min_of":        _ops.min_of,
    "max_of":        _ops.max_of,
    "ifelse":        _ops.ifelse,
    "gt":            _ops.gt,
    "ge":            _ops.ge,
    "lt":            _ops.lt,
    "le":            _ops.le,
    "eq":            _ops.eq,
    "and_op":        _ops.and_op,
    "or_op":         _ops.or_op,
    "not_op":        _ops.not_op,
    "mean_of":       _ops.mean_of,
    "weighted_sum":  _ops.weighted_sum,
    "combine_rank":  _ops.combine_rank,
    "np":            np,
}


def _notify_pipeline_error(
    title: str,
    error: Any,
    *,
    stage: str,
    run_id: str = "",
    formula: str = "",
) -> None:
    """Send a concise Feishu alert for actionable pipeline failures."""
    friendly, suggestion, error_code, raw = humanize_error(error)
    try:
        _feishu.send_error_notification(
            title=title,
            summary=friendly,
            stage=stage,
            error_code=error_code,
            suggestion=suggestion,
            raw_detail=raw,
            run_id=run_id,
            formula=formula,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
    except Exception as notify_error:
        print(f"  [feishu] ⚠ Failed to send error notification: {notify_error}")


def _eval_formula(formula_str: str, df: pd.DataFrame) -> pd.Series:
    """Evaluate DSL formula with the extended operator registry."""
    local_env = {col: df[col] for col in df.columns}
    try:
        code = compile(formula_str, "<formula>", "eval")
        result = eval(code, _EXT_OPS, local_env)  # noqa: S307
        return result
    except SyntaxError as e:
        raise ValueError(f"Syntax error in formula: {e}")
    except Exception as e:
        raise RuntimeError(f"Runtime error evaluating formula: {e}")


def _append_trace(trace_path: Path, stage: str, payload: dict[str, Any]) -> None:
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "ts": datetime.now().isoformat(),
        "stage": stage,
        **payload,
    }
    with open(trace_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Post-processing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _postprocess(series: pd.Series, mode: str) -> pd.Series:
    """
    Light post-processing: cross-sectional rank then clip to [-1, 1].
    Keeps alpha bounded (required by SubmissionBuilder) and reduces extreme
    position weights, helping pass the concentration gates.
    """
    if series.empty:
        return series
    try:
        un = series.unstack("security_id")
    except Exception:
        return series.clip(-1.0, 1.0)

    if mode in ("rank_clip", "stable_low_turnover"):
        # Percentile rank → [0,1] → shift to [-0.5, 0.5]
        ranked = un.rank(axis=1, pct=True) - 0.5
    else:  # zscore_clip / aggressive_high_ic
        mu  = un.mean(axis=1)
        std = un.std(axis=1).replace(0, np.nan)
        ranked = un.sub(mu, axis=0).div(std, axis=0).clip(-3, 3) / 3

    stacked = ranked.stack("security_id").reorder_levels(series.index.names).sort_index()
    return stacked.clip(-1.0, 1.0).astype("float32")


# ─────────────────────────────────────────────────────────────────────────────
# Core alpha computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_alpha(
    formula: str,
    pv: pd.DataFrame,
    days: list[str],
    lookback_days: int = 20,
    postprocess_mode: str = "rank_clip",
) -> pd.Series:
    """
    Compute alpha values for the given date range.

    Args:
        formula:          DSL expression string
        pv:               full 15m OHLCV DataFrame (MultiIndex: date/datetime/security_id)
        days:             list of trading-day strings to EVALUATE (without warmup)
        lookback_days:    how many extra days before `days[0]` to load as warmup
        postprocess_mode: 'rank_clip' or 'zscore_clip'

    Returns:
        pd.Series with MultiIndex (date, datetime, security_id), alpha in [-1, 1].
    """
    all_days = _pv_trading_days(pv)
    eval_start = days[0]
    eval_end   = days[-1]

    # Find warmup start
    try:
        idx0 = all_days.index(eval_start)
    except ValueError:
        idx0 = 0
    warmup_idx = max(0, idx0 - lookback_days)
    warmup_start = all_days[warmup_idx]

    # Slice: warmup + eval window
    warmup_ts    = pd.to_datetime(warmup_start)
    eval_end_ts  = pd.to_datetime(eval_end)

    sub_pv = _slice_pv_window(pv, warmup_ts, eval_end_ts)

    # Compute alpha
    raw_alpha = _eval_formula(formula, sub_pv)
    if not isinstance(raw_alpha, pd.Series):
        raw_alpha = pd.Series(raw_alpha)

    # Post-process (cross-section rank / zscore)
    alpha_pp = _postprocess(raw_alpha, postprocess_mode)

    # Trim to eval window only (drop warmup)
    eval_key = pd.to_datetime(eval_start).strftime("%Y-%m-%d")
    alpha_dates = pd.to_datetime(alpha_pp.index.get_level_values("date")).strftime("%Y-%m-%d")
    alpha_eval = alpha_pp.loc[alpha_dates >= eval_key]
    return alpha_eval


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_alpha(alpha: pd.Series, hub: DataHub, days: list[str]) -> dict[str, Any]:
    """Compute IC / IR / turnover / gates using evaluate_submission_like_wide."""
    if _futures_mode():
        from core.futures_alpha import evaluate_tick_h60_alpha

        day_set = {str(d) for d in days}
        try:
            mask = pd.to_datetime(alpha.index.get_level_values("date")).strftime("%Y-%m-%d").isin(day_set)
            alpha = alpha.loc[mask]
        except Exception:
            pass
        tick_metrics = evaluate_tick_h60_alpha(alpha)
        return {k: v for k, v in tick_metrics.items() if k not in {"daily_ic", "tick_frames"}}

    alpha_un = alpha.unstack("security_id")
    resp_base, rest_base = _get_eval_wide_frames(hub, days)
    resp_un = resp_base.reindex_like(alpha_un)
    rest_un = rest_base.reindex_like(alpha_un).fillna(0)

    return evaluate_submission_like_wide(alpha_un, resp_un, rest_un)


def _cfg_int(cfg: dict[str, str], key: str, default: int) -> int:
    try:
        return int(str(cfg.get(key, default) or default))
    except (TypeError, ValueError):
        return default


def _cfg_float(cfg: dict[str, str], key: str, default: float) -> float:
    try:
        return float(str(cfg.get(key, default) or default))
    except (TypeError, ValueError):
        return default


def _cfg_bool(cfg: dict[str, str], key: str, default: bool = False) -> bool:
    value = str(cfg.get(key, str(int(default))) or "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _filter_days(days: list[str], start: str, end: str) -> list[str]:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    return [
        day for day in days
        if start_ts <= pd.Timestamp(day) <= end_ts
    ]


def _read_alpha_series_from_parquet(path: str) -> pd.Series | None:
    if not path or not os.path.exists(path):
        return None
    try:
        frame = pd.read_parquet(path)
    except Exception:
        return None
    if (
        isinstance(frame, pd.DataFrame)
        and not isinstance(frame.index, pd.MultiIndex)
        and {"date", "datetime", "security_id"}.issubset(frame.columns)
    ):
        frame = frame.copy()
        frame["date"] = pd.to_datetime(frame["date"])
        frame["datetime"] = pd.to_datetime(frame["datetime"])
        frame = frame.set_index(["date", "datetime", "security_id"]).sort_index()
    if isinstance(frame, pd.Series):
        series = frame
    else:
        numeric_cols = [
            col for col in frame.columns
            if pd.api.types.is_numeric_dtype(frame[col])
        ]
        if not numeric_cols:
            return None
        preferred = "alpha" if "alpha" in numeric_cols else numeric_cols[0]
        series = frame[preferred]
    if not isinstance(series.index, pd.MultiIndex):
        return None
    if "date" not in series.index.names or "security_id" not in series.index.names:
        return None
    return series.dropna().astype("float32")


def _flatten_for_corr(series: pd.Series, days: list[str]) -> pd.Series:
    if series is None or series.empty:
        return pd.Series(dtype="float32")
    day_ts = set(pd.to_datetime(days))
    try:
        mask = series.index.get_level_values("date").isin(day_ts)
        series = series.loc[mask]
    except Exception:
        return pd.Series(dtype="float32")
    if series.empty:
        return pd.Series(dtype="float32")
    return series.dropna().astype("float32")


def _compute_library_correlation(
    alpha: pd.Series,
    days: list[str],
    *,
    max_factors: int = 40,
) -> dict[str, Any]:
    """Compute max train-window correlation versus already accepted v3 factors."""
    current = _flatten_for_corr(alpha, days)
    if current.empty:
        return {"max_abs_corr": 0.0, "max_corr": 0.0, "closest_run_id": "", "compared": 0}

    accepted = kb.list_valid_factors()
    accepted = sorted(
        accepted,
        key=lambda item: (float(item.get("Score", 0) or 0), float(item.get("IC", 0) or 0)),
        reverse=True,
    )[:max(1, max_factors)]

    best = {"max_abs_corr": 0.0, "max_corr": 0.0, "closest_run_id": "", "compared": 0}
    for item in accepted:
        prior = _read_alpha_series_from_parquet(str(item.get("parquet_path", "") or ""))
        if prior is None or prior.empty:
            continue
        prior = _flatten_for_corr(prior, days)
        if prior.empty:
            continue
        joined = pd.concat([current.rename("current"), prior.rename("prior")], axis=1, join="inner").dropna()
        if len(joined) < 10_000:
            continue
        corr = joined["current"].corr(joined["prior"])
        if not np.isfinite(corr):
            continue
        best["compared"] = int(best["compared"]) + 1
        abs_corr = abs(float(corr))
        if abs_corr > float(best["max_abs_corr"]):
            best = {
                "max_abs_corr": abs_corr,
                "max_corr": float(corr),
                "closest_run_id": str(item.get("run_id", "")),
                "compared": int(best["compared"]),
            }
    return best


def _apply_low_correlation_gate(
    metrics: dict[str, Any],
    corr_info: dict[str, Any],
    cfg: dict[str, str],
) -> dict[str, Any]:
    threshold = _cfg_float(cfg, "AUTOALPHA_MAX_LIBRARY_CORR", 0.72)
    max_abs_corr = float(corr_info.get("max_abs_corr", 0.0) or 0.0)
    pass_corr = max_abs_corr <= threshold
    updated = dict(metrics)
    detail = dict(updated.get("GatesDetail") or {})
    detail["LowCorrelation"] = bool(pass_corr)
    updated["GatesDetail"] = detail
    updated["correlation"] = {
        **corr_info,
        "threshold": threshold,
        "PassLowCorrelation": bool(pass_corr),
    }
    if not pass_corr:
        updated["PassGates"] = False
        updated["Score"] = 0.0
    return updated


def _metric_value(metrics: dict[str, Any], key: str) -> float:
    if key == "tvr":
        return float(metrics.get("tvr", metrics.get("Turnover", 0.0)) or 0.0)
    return float(metrics.get(key, 0.0) or 0.0)


def _period_row(
    *,
    period: str,
    label: str,
    start: str,
    end: str,
    days: int,
    metrics: dict[str, Any],
    used_for_discovery: bool,
) -> dict[str, Any]:
    return {
        "period": period,
        "label": label,
        "start": start,
        "end": end,
        "days": int(days),
        "IC": _metric_value(metrics, "IC"),
        "IR": _metric_value(metrics, "IR"),
        "tvr": _metric_value(metrics, "tvr"),
        "Turnover": _metric_value(metrics, "tvr"),
        "Score": _metric_value(metrics, "Score"),
        "PassGates": bool(metrics.get("PassGates", False)),
        "used_for_discovery": bool(used_for_discovery),
    }


def _oos_comparison(train_metrics: dict[str, Any], oos_metrics: dict[str, Any]) -> dict[str, Any]:
    train_ic = _metric_value(train_metrics, "IC")
    oos_ic = _metric_value(oos_metrics, "IC")
    train_score = _metric_value(train_metrics, "Score")
    oos_score = _metric_value(oos_metrics, "Score")
    return {
        "IC_delta": oos_ic - train_ic,
        "Score_delta": oos_score - train_score,
        "IR_delta": _metric_value(oos_metrics, "IR") - _metric_value(train_metrics, "IR"),
        "tvr_delta": _metric_value(oos_metrics, "tvr") - _metric_value(train_metrics, "tvr"),
        "IC_retention": (oos_ic / train_ic) if abs(train_ic) > 1e-12 else 0.0,
        "Score_retention": (oos_score / train_score) if abs(train_score) > 1e-12 else 0.0,
        "trend": "improved" if oos_score > train_score else ("stable" if abs(oos_score - train_score) <= max(1e-6, abs(train_score) * 0.05) else "weaker"),
        "used_for_feedback": False,
    }


def _formula_compute_cost(formula: str) -> dict[str, Any]:
    lowered = (formula or "").lower()
    rolling_ops = re.findall(
        r"\b(ts_(?:mean|ema|std|sum|max|min|median|quantile|zscore|rank|minmax_norm|decay_linear|corr|cov|skew|kurt|argmax|argmin|pct_change))\s*\(",
        lowered,
    )
    soft_ops = re.findall(r"\b(?:tanh|sigmoid|signed_log|safe_div|mean_of|weighted_sum|combine_rank)\s*\(", lowered)
    return {
        "chars": len(formula or ""),
        "rolling_ops": len(rolling_ops),
        "unique_rolling_ops": sorted(set(rolling_ops)),
        "soft_ops": len(soft_ops),
    }


def _complexity_rejection_reason(formula: str, cfg: dict[str, str]) -> str:
    cost = _formula_compute_cost(formula)
    max_chars = _cfg_int(cfg, "AUTOALPHA_MAX_FORMULA_CHARS", 340)
    max_rolling_ops = _cfg_int(cfg, "AUTOALPHA_MAX_ROLLING_OPS", 5)
    if int(cost["chars"]) > max_chars:
        return f"formula too long: chars={cost['chars']}>{max_chars}"
    if int(cost["rolling_ops"]) > max_rolling_ops:
        return f"too many rolling operators: rolling_ops={cost['rolling_ops']}>{max_rolling_ops}"
    return ""


def _evaluate_with_optional_flip(
    alpha: pd.Series,
    formula: str,
    hub: DataHub,
    days: list[str],
) -> tuple[pd.Series, str, dict[str, Any]]:
    metrics = evaluate_alpha(alpha, hub, days)
    ic = float(metrics.get("IC", 0) or 0)
    flip_threshold = -0.3
    if ic >= flip_threshold:
        return alpha, formula, metrics

    print("  [flip] IC is negative — testing flipped alpha direction")
    flipped_alpha = -alpha
    flipped_formula = f"neg({formula})"
    flipped_metrics = evaluate_alpha(flipped_alpha, hub, days)

    def _choice_key(item: dict[str, Any]) -> tuple[int, float, float, float]:
        return (
            int(bool(item.get("PassGates", False))),
            float(item.get("Score", 0) or 0),
            float(item.get("IC", 0) or 0),
            float(item.get("IR", 0) or 0),
        )

    if _choice_key(flipped_metrics) > _choice_key(metrics):
        print(
            "  [flip] Accepted flipped direction "
            f"IC={float(flipped_metrics.get('IC', 0) or 0):.4f} "
            f"IR={float(flipped_metrics.get('IR', 0) or 0):.4f} "
            f"score={float(flipped_metrics.get('Score', 0) or 0):.4f}"
        )
        return flipped_alpha, flipped_formula, flipped_metrics

    print(
        "  [flip] Kept original direction "
        f"IC={float(metrics.get('IC', 0) or 0):.4f}; "
        f"flipped_IC={float(flipped_metrics.get('IC', 0) or 0):.4f}"
    )
    return alpha, formula, metrics


def _apply_tvr_combo(alpha: pd.Series, combo_name: str) -> pd.Series:
    if not combo_name:
        return alpha
    try:
        from autoalpha_v3.tvr_optimizer import combo_ema, combo_persistence, combo_extremes, combo_rolling
        combo_map = {
            "ema_10":         lambda a: combo_ema(a, span=10),
            "persistence_02": lambda a: combo_persistence(a, blend_alpha=0.2),
            "extremes_q20":   lambda a: combo_extremes(a, q=0.2),
            "rolling_15":     lambda a: combo_rolling(a, window=15),
        }
        if combo_name in combo_map:
            return combo_map[combo_name](alpha)
    except Exception as e:
        print(f"  [tvr-opt] Could not apply combo {combo_name}: {e}")
    return alpha


def _screen_failure_details(
    metrics: dict[str, Any],
    cfg: dict[str, str],
    expected_days: int | None = None,
) -> list[dict[str, Any]]:
    """Return structured details explaining exactly which screen gates failed."""
    ic = float(metrics.get("IC", 0) or 0)
    ir = float(metrics.get("ICIR", metrics.get("IR", 0)) or 0)
    preview = metrics.get("result_preview") or {}
    nd = float(preview.get("nd", metrics.get("nd", 0)) or 0)
    cover_all = preview.get("cover_all", metrics.get("cover_all"))
    min_ic = _cfg_float(cfg, "AUTOALPHA_SCREEN_MIN_IC", 0.02)
    min_ir = _cfg_float(cfg, "AUTOALPHA_SCREEN_MIN_IR", 1.0)
    fails: list[dict[str, Any]] = []
    if abs(ic) < min_ic:
        fails.append({"key": "IC", "value": ic, "threshold": min_ic, "direction": "abs >=", "message": f"|IC|={abs(ic):.4f}<{min_ic}"})
    if abs(ir) < min_ir:
        fails.append({"key": "ICIR", "value": ir, "threshold": min_ir, "direction": "abs >=", "message": f"|ICIR|={abs(ir):.3f}<{min_ir}"})
    if expected_days and nd and nd < expected_days:
        fails.append({"key": "Days", "value": nd, "threshold": expected_days, "direction": ">=", "message": f"Days={nd:.0f}/{expected_days}"})
    if cover_all is not None and int(bool(cover_all)) == 0:
        fails.append({"key": "Coverage", "value": cover_all, "threshold": 1, "direction": "=", "message": "Coverage=0"})
    if not fails and float(metrics.get("Score", 0) or 0) <= 0:
        fails.append({"key": "Score", "value": float(metrics.get("Score", 0) or 0), "threshold": 0, "direction": ">", "message": "score=0"})
    return fails


def _screen_failure_reason(
    metrics: dict[str, Any],
    cfg: dict[str, str],
    expected_days: int | None = None,
) -> str:
    """Return a compact string explaining exactly which screen gates failed."""
    details = _screen_failure_details(metrics, cfg, expected_days=expected_days)
    return ", ".join(str(item.get("message", "")) for item in details if item.get("message")) or "screen gate failed"


def _screen_promotion_failure_details(
    metrics: dict[str, Any],
    cfg: dict[str, str],
    expected_days: int | None = None,
) -> list[dict[str, Any]]:
    """Return details for the stricter full-eval promotion gate."""
    ic = float(metrics.get("IC", 0) or 0)
    ir = float(metrics.get("ICIR", metrics.get("IR", 0)) or 0)
    preview = metrics.get("result_preview") or {}
    nd = float(preview.get("nd", metrics.get("nd", 0)) or 0)
    min_ic = _cfg_float(
        cfg,
        "AUTOALPHA_SCREEN_PROMOTE_MIN_IC",
        max(_cfg_float(cfg, "AUTOALPHA_SCREEN_MIN_IC", 0.02), 0.02),
    )
    min_ir = _cfg_float(
        cfg,
        "AUTOALPHA_SCREEN_PROMOTE_MIN_IR",
        max(_cfg_float(cfg, "AUTOALPHA_SCREEN_MIN_IR", 1.0), 1.0),
    )
    fails: list[dict[str, Any]] = []
    if abs(ic) < min_ic:
        fails.append({"key": "IC", "value": ic, "threshold": min_ic, "direction": "abs >=", "message": f"promote |IC|={abs(ic):.4f}<{min_ic}"})
    if abs(ir) < min_ir:
        fails.append({"key": "ICIR", "value": ir, "threshold": min_ir, "direction": "abs >=", "message": f"promote |ICIR|={abs(ir):.3f}<{min_ir}"})
    if expected_days and nd and nd < expected_days:
        fails.append({"key": "Days", "value": nd, "threshold": expected_days, "direction": ">=", "message": f"Days={nd:.0f}/{expected_days}"})
    if not fails and not (bool(metrics.get("PassGates")) or float(metrics.get("Score", 0) or 0) > 0):
        fails.append({"key": "Score", "value": float(metrics.get("Score", 0) or 0), "threshold": 0, "direction": ">", "message": "screen not pass and score=0"})
    return fails


def _screen_promotion_failure_reason(
    metrics: dict[str, Any],
    cfg: dict[str, str],
    expected_days: int | None = None,
) -> str:
    details = _screen_promotion_failure_details(metrics, cfg, expected_days=expected_days)
    return ", ".join(str(item.get("message", "")) for item in details if item.get("message")) or "screen promotion gate failed"


def _should_promote_from_screen(metrics: dict[str, Any], cfg: dict[str, str]) -> bool:
    ic = float(metrics.get("IC", 0) or 0)
    ir = float(metrics.get("ICIR", metrics.get("IR", 0)) or 0)
    score = float(metrics.get("Score", 0) or 0)
    min_ic = _cfg_float(
        cfg,
        "AUTOALPHA_SCREEN_PROMOTE_MIN_IC",
        max(_cfg_float(cfg, "AUTOALPHA_SCREEN_MIN_IC", 0.02), 0.02),
    )
    min_ir = _cfg_float(
        cfg,
        "AUTOALPHA_SCREEN_PROMOTE_MIN_IR",
        max(_cfg_float(cfg, "AUTOALPHA_SCREEN_MIN_IR", 1.0), 1.0),
    )
    return bool(metrics.get("PassGates")) or score > 0 or (abs(ic) >= min_ic and abs(ir) >= min_ir)


def _confirmation_failure_details(metrics: dict[str, Any], cfg: dict[str, str]) -> list[dict[str, Any]]:
    ic = float(metrics.get("IC", 0) or 0)
    ir = float(metrics.get("ICIR", metrics.get("IR", 0)) or 0)
    min_ic = _cfg_float(cfg, "AUTOALPHA_CONFIRM_MIN_IC", 0.02)
    min_ir = _cfg_float(cfg, "AUTOALPHA_CONFIRM_MIN_IR", 1.0)
    fails: list[dict[str, Any]] = []
    if abs(ic) < min_ic:
        fails.append({"key": "IC", "value": ic, "threshold": min_ic, "direction": "abs >=", "message": f"confirm |IC|={abs(ic):.4f}<{min_ic}"})
    if abs(ir) < min_ir:
        fails.append({"key": "ICIR", "value": ir, "threshold": min_ir, "direction": "abs >=", "message": f"confirm |ICIR|={abs(ir):.3f}<{min_ir}"})
    return fails


def _should_promote_from_confirmation(metrics: dict[str, Any], cfg: dict[str, str]) -> bool:
    if bool(metrics.get("PassGates")) or float(metrics.get("Score", 0) or 0) > 0:
        return True
    return not _confirmation_failure_details(metrics, cfg)


def _confirmation_failure_reason(metrics: dict[str, Any], cfg: dict[str, str]) -> str:
    details = _confirmation_failure_details(metrics, cfg)
    return ", ".join(str(item.get("message", "")) for item in details if item.get("message")) or "confirmation gate failed"


def _gate_failure_reason(metrics: dict[str, Any]) -> str:
    detail = metrics.get("GatesDetail") or {}
    failed = [str(k) for k, ok in detail.items() if not ok]
    if failed:
        return "full gate failed: " + ", ".join(failed)
    if not metrics.get("PassGates"):
        return "full gate failed"
    return ""


def _should_materialize_artifacts(metrics: dict[str, Any], cfg: dict[str, str]) -> bool:
    if bool(metrics.get("PassGates")):
        return True
    if not _cfg_bool(cfg, "AUTOALPHA_EXPORT_PROMISING_NONPASS", False):
        return False
    ic = float(metrics.get("IC", 0) or 0)
    ir = float(metrics.get("IR", 0) or 0)
    score = float(metrics.get("Score", 0) or 0)
    min_ic = _cfg_float(cfg, "AUTOALPHA_RESEARCH_MIN_IC", 0.02)
    min_ir = _cfg_float(cfg, "AUTOALPHA_RESEARCH_MIN_IR", 1.0)
    return bool(metrics.get("PassGates")) or score > 0 or (abs(ic) >= min_ic and abs(ir) >= min_ir)


# ─────────────────────────────────────────────────────────────────────────────
# Parquet export
# ─────────────────────────────────────────────────────────────────────────────

ALLOWED_UTC_TIMES = {
    "01:45:00", "02:00:00", "02:15:00", "02:30:00",
    "02:45:00", "03:00:00", "03:15:00", "03:30:00",
    "05:15:00", "05:30:00", "05:45:00", "06:00:00",
    "06:15:00", "06:30:00", "06:45:00", "07:00:00",
}


def export_parquet(
    alpha: pd.Series,
    run_id: str,
    out_dir: Path,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> Path:
    """
    Save alpha to parquet in submission format with a complete (datetime × security_id)
    cartesian grid per date, as required by the simulator's cartesian_unstack.

    Formulas with lookback warmup produce sparse early dates (fewer bars, uneven
    security sets). expand_to_full_grid fills the missing cells with NaN so every
    date has exactly 16 bars × universe_size rows.
    """
    from core.submission import SubmissionBuilder

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.pq"

    # Build MultiIndex frame expected by SubmissionBuilder
    alpha_frame = SubmissionBuilder._ensure_frame(alpha)

    frame_start = alpha_frame.index.get_level_values("date").min()
    frame_end = alpha_frame.index.get_level_values("date").max()
    start_date = start_date or frame_start
    end_date = end_date or frame_end
    if hasattr(start_date, "strftime"):
        start_date = start_date.strftime("%Y-%m-%d")
    if hasattr(end_date, "strftime"):
        end_date = end_date.strftime("%Y-%m-%d")

    expanded = SubmissionBuilder.expand_to_full_grid(
        alpha_frame, str(start_date), str(end_date), chunk_days=30
    )
    alpha_col = SubmissionBuilder._alpha_col(expanded)
    non_null = int(expanded[alpha_col].notna().sum()) if not expanded.empty else 0
    if non_null == 0:
        raise ValueError(
            "Expanded submission grid contains zero non-null alpha values; "
            "check date/datetime/security_id index alignment before export."
        )
    SubmissionBuilder.build(expanded, str(out_path))

    row_count = len(expanded)
    print(f"  [export] Saved {row_count:,} rows (full grid, non-null alpha={non_null:,}) → {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(
    n_ideas: int = 3,
    eval_days_count: int = 0,
    parents: Optional[List] = None,
    idea_index_offset: int = 0,
) -> List[dict]:
    """
    Full pipeline: generate → validate → compute → evaluate → export.

    Args:
        n_ideas:        number of ideas to request from the LLM
        eval_days_count: how many of the most recent trading days to evaluate;
                         0 or below means use the full history
        parents:        optional list of prior factor results to feed the LLM
        idea_index_offset: absolute idea index offset for source/exploration scheduling

    Returns:
        list of result dicts (one per valid, successfully computed idea)
    """
    AUTOALPHA_OUT.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trace_path = AUTOALPHA_TRACE_DIR / f"run_{timestamp}.jsonl"
    cfg = load_runtime_config()
    idea_pause_sec = max(0.0, _cfg_float(cfg, "AUTOALPHA_IDEA_PAUSE_MS", 250.0) / 1000.0)

    # ── 1. Load data ONCE ────────────────────────────────────────────────────
    print("[pipeline] Loading DataHub …")
    try:
        full_days = get_trading_days()
        if not full_days:
            raise AutoAlphaRuntimeError(
                "交易日列表为空，当前无法执行因子评估。",
                raw_message="get_trading_days() returned no trading days.",
                suggestion="检查数据目录和 DataHub 读取配置。",
                error_code="empty_trading_days",
            )
        discovery_all_full, requested_eval_days, _, _, split_mode = _split_research_days(full_days, eval_days_count)
        if eval_days_count > 0 and discovery_all_full:
            warmup_days = max(30, _cfg_int(cfg, "AUTOALPHA_DATA_WARMUP_DAYS", 45))
            eval_start_idx = full_days.index(requested_eval_days[0])
            load_start_idx = max(0, eval_start_idx - warmup_days)
            load_start = full_days[load_start_idx]
            load_end = OOS_END if split_mode != "futures_adaptive_discovery_oos" else full_days[-1]
            if load_end not in full_days:
                load_end = full_days[-1]
            exact_cache = Path(CACHE_ROOT) / f"pv_15m_{load_start}_{load_end}.parquet"
            cache_status = "hit" if exact_cache.is_file() else "superset-or-build"
            print(
                f"[pipeline] Data window: {load_start} → {load_end} "
                f"(discovery_eval_days={len(requested_eval_days)}, warmup_days={eval_start_idx - load_start_idx}, "
                f"cache={cache_status})"
            )
            hub = _get_data_hub(start=load_start, end=load_end)
        else:
            requested_eval_days = discovery_all_full
            print(f"[pipeline] Data window: {full_days[0]} → {full_days[-1]} (full configured history)")
            hub = _get_data_hub()
        pv = hub.pv_15m
        all_days = hub.get_trading_days_list()
    except Exception as e:
        _notify_pipeline_error("AutoAlpha 数据加载失败", e, stage="DataHub 初始化")
        raise

    if not all_days:
        error = AutoAlphaRuntimeError(
            "交易日列表为空，当前无法执行因子评估。",
            raw_message="DataHub.get_trading_days_list() returned no trading days.",
            suggestion="检查数据目录和 DataHub 读取配置。",
            error_code="empty_trading_days",
        )
        _notify_pipeline_error("AutoAlpha 数据加载失败", error, stage="交易日读取")
        raise error

    discovery_days_all, requested_eval_days, oos_days, full_export_days, split_mode = _split_research_days(all_days, eval_days_count)
    if not discovery_days_all:
        error = AutoAlphaRuntimeError(
            "Discovery 交易日为空，当前无法执行因子发现。",
            raw_message="No trading days in configured discovery window.",
            suggestion="检查数据目录、期货产品过滤和日期范围配置。",
            error_code="empty_discovery_days",
        )
        _notify_pipeline_error("AutoAlpha v3 数据切分失败", error, stage="Discovery window")
        raise error

    eval_days = discovery_days_all if eval_days_count <= 0 else [d for d in requested_eval_days if d in set(discovery_days_all)]
    screen_days_count = min(
        len(eval_days),
        max(60, _cfg_int(cfg, "AUTOALPHA_SCREEN_DAYS", 160)),
    )
    screen_days = eval_days[-screen_days_count:]
    eval_mode = "ALL" if eval_days_count <= 0 else str(eval_days_count)
    oos_label = f"{oos_days[0]} → {oos_days[-1]} ({len(oos_days)} days)" if oos_days else "unavailable"
    full_export_label = f"{full_export_days[0]} → {full_export_days[-1]} ({len(full_export_days)} days)" if full_export_days else "unavailable"
    print(f"[pipeline] Discovery window: {eval_days[0]} → {eval_days[-1]} ({len(eval_days)} days, requested={eval_mode})")
    print(f"[pipeline] Screen window: {screen_days[0]} → {screen_days[-1]} ({len(screen_days)} days)")
    print(f"[pipeline] OOS report-only window: {oos_label}")
    print(f"[pipeline] Export parquet window: {full_export_label}")
    _append_trace(
        trace_path,
        "run_start",
        {
            "mode": split_mode,
            "eval_days": len(eval_days),
            "screen_days": len(screen_days),
            "oos_days": len(oos_days),
            "full_export_days": len(full_export_days),
            "discovery_start": eval_days[0],
            "discovery_end": eval_days[-1],
            "oos_start": oos_days[0] if oos_days else "",
            "oos_end": oos_days[-1] if oos_days else "",
            "parent_run_ids": [item.get("run_id", "") for item in parents or []],
        },
    )

    # ── 2. Generate ideas (cache-first, then bounded LLM generation) ───────────
    from concurrent.futures import ThreadPoolExecutor, as_completed as _as_completed
    from autoalpha_v3.idea_cache import get_default_cache, parent_context_signature
    from autoalpha_v3.llm_client import plan_generation_slot

    idea_cache = get_default_cache()
    idea_concurrency = max(1, _cfg_int(cfg, "AUTOALPHA_IDEA_CONCURRENCY", 1))
    prompt_version = str(cfg.get("AUTOALPHA_PROMPT_VERSION", "v3-oos-lowcorr-20260427") or "v3-oos-lowcorr-20260427")
    parent_signature = parent_context_signature(parents)

    ideas: list[dict] = []
    generation_errors: list[dict[str, str]] = []

    def _fetch_one_idea(local_i: int) -> tuple[int, Optional[dict], Optional[dict]]:
        absolute_i = idea_index_offset + local_i
        print(f"\n[pipeline] Generating idea {local_i+1}/{n_ideas} via LLM (slot={absolute_i}) …")
        try:
            idea = generate_idea(parents=parents, idea_index=absolute_i, total_ideas=n_ideas)
            if idea:
                idea.setdefault("parent_signature", parent_signature)
            if idea and idea.get("formula") and not idea.get("idea_cache_id"):
                idea["idea_cache_id"] = idea_cache.register_generated_idea(idea, consumed=True)
            return local_i, idea, None
        except Exception as e:
            friendly, suggestion, _, raw = humanize_error(e)
            return local_i, None, {"friendly": friendly, "suggestion": suggestion, "raw": raw}

    def _source_fields(idea: dict[str, Any]) -> dict[str, Any]:
        source_type = idea.get("inspiration_source_type") or "none"
        source_types = idea.get("inspiration_source_types") or []
        inspiration_ids = idea.get("inspiration_ids") or []
        return {
            "inspiration_source_type": source_type,
            "inspiration_source_types": source_types,
            "inspiration_ids": inspiration_ids,
            "idea_cache_id": idea.get("idea_cache_id"),
            "archetype": idea.get("archetype", ""),
            "archetype_label": idea.get("archetype_label", ""),
            "generation_mode": idea.get("generation_mode", ""),
            "target_source": idea.get("target_source", source_type),
            "parent_signature": idea.get("parent_signature", ""),
            "prompt_version": idea.get("prompt_version", ""),
            "mutation_type": idea.get("mutation_type", ""),
            "rag_trace": idea.get("rag_trace", {}),
        }

    def _record_idea_outcome(idea: dict[str, Any], outcome: str) -> None:
        try:
            idea_cache.record_outcome(idea.get("idea_cache_id"), outcome)
        except Exception as exc:
            print(f"  [WARN] Idea outcome writeback failed: {exc}")

    def _append_idea(i: int, idea: dict[str, Any], source_label: str) -> None:
        print(f"\n[pipeline] Idea {i+1}/{n_ideas} from {source_label}: {idea.get('formula','')[:60]}")
        print(f"  formula    : {idea.get('formula')}")
        print(f"  thought    : {str(idea.get('thought_process',''))[:120]}")
        ideas.append(idea)
        try:
            record_inspiration_usage(idea.get("inspiration_ids") or [])
        except Exception as exc:
            print(f"  [WARN] Inspiration usage writeback failed: {exc}")
        _append_trace(
            trace_path,
            "idea_generated",
            {
                "idea_index": i + 1,
                "formula": idea.get("formula", ""),
                "thought_process": idea.get("thought_process", ""),
                "source": source_label,
                **_source_fields(idea),
            },
        )

    def _pop_cache_for_slot(local_i: int) -> Optional[dict]:
        absolute_i = idea_index_offset + local_i
        target_source = plan_generation_slot(absolute_i).get("target_source", "")
        cached = idea_cache.pop(
            parent_signature=parent_signature,
            target_source=target_source,
            prompt_version=prompt_version,
        )
        if not cached:
            return None
        cached.setdefault("parent_signature", parent_signature)
        return cached

    # Drain any ready cache entries first so we do not hit the gateway unnecessarily.
    while len(ideas) < n_ideas:
        local_i = len(ideas)
        cached = _pop_cache_for_slot(local_i)
        if not cached:
            break
        _append_idea(local_i, cached, "cache")

    missing = n_ideas - len(ideas)
    if missing > 0:
        join_timeout = max(0.0, _cfg_float(cfg, "AUTOALPHA_IDEA_CACHE_JOIN_TIMEOUT_SEC", 20.0))
        fill_done = idea_cache.join_fill(timeout=join_timeout)
        if not fill_done:
            print(f"[idea_cache] Background fill still running after {join_timeout:.0f}s; continuing with inline generation")
        # Fill may have completed while we were waiting — drain cache before going inline.
        while len(ideas) < n_ideas:
            local_i = len(ideas)
            cached = _pop_cache_for_slot(local_i)
            if not cached:
                break
            _append_idea(local_i, cached, "cache-after-fill")
        missing = n_ideas - len(ideas)
    if missing > 0:
        with ThreadPoolExecutor(max_workers=min(idea_concurrency, missing)) as pool:
            start_i = len(ideas)
            futures = {pool.submit(_fetch_one_idea, start_i + i): start_i + i for i in range(missing)}
            for fut in _as_completed(futures):
                i, idea, err = fut.result()
                if idea:
                    _append_idea(i, idea, "LLM")
                elif err:
                    generation_errors.append(err)
                    extra = f" 建议: {err.get('suggestion')}" if err.get("suggestion") else ""
                    print(f"  [WARN] LLM call failed: {err.get('friendly')}{extra}")
                    _append_trace(
                        trace_path,
                        "idea_generation_error",
                        {
                            "idea_index": i + 1,
                            "friendly": err.get("friendly", ""),
                            "suggestion": err.get("suggestion", ""),
                            "raw": err.get("raw", ""),
                        },
                    )

    # Background prefill may have succeeded while foreground generation was failing.
    while len(ideas) < n_ideas:
        local_i = len(ideas)
        cached = _pop_cache_for_slot(local_i)
        if not cached:
            break
        _append_idea(local_i, cached, "cache-retry")

    if idea_pause_sec > 0 and ideas:
        time.sleep(idea_pause_sec)

    if not ideas:
        primary = generation_errors[0] if generation_errors else {
            "friendly": "LLM 没有返回任何可用因子。",
            "suggestion": "检查 API Key、模型和网关状态。",
            "raw": "No ideas generated.",
        }
        error = AutoAlphaRuntimeError(
            primary["friendly"],
            raw_message=primary["raw"],
            suggestion=primary["suggestion"],
            error_code="idea_generation_failed",
        )
        _notify_pipeline_error("AutoAlpha 因子生成失败", error, stage="LLM 生成")
        raise error

    # ── 3. Validate + compute ─────────────────────────────────────────────────
    results = []
    existing_formula_keys = kb.get_existing_formula_keys()
    batch_formula_keys: set[str] = set()
    for idx, idea in enumerate(ideas):
        formula  = idea.get("formula", "").strip()
        postmode = idea.get("postprocess", "rank_clip")
        lookback = int(idea.get("lookback_days", 20))
        run_id   = f"autoalpha_{timestamp}_{idx+1:02d}"

        print(f"\n[pipeline] Processing idea {idx+1}: {formula}")

        formula_key = kb.canonical_formula(formula)
        if formula_key in existing_formula_keys or formula_key in batch_formula_keys:
            print("  [SKIP] Duplicate formula against KB or current batch")
            _append_trace(trace_path, "duplicate_formula", {"run_id": run_id, "formula": formula})
            _record_idea_outcome(idea, "duplicate")
            results.append({
                "run_id": run_id,
                "formula": formula,
                "thought_process": idea.get("thought_process", ""),
                **_source_fields(idea),
                "status": "duplicate",
                "errors": "Duplicate formula signature already exists in knowledge base or current batch.",
            })
            continue
        batch_formula_keys.add(formula_key)

        # 3a. Validate (syntax + whitelist + future-leakage)
        vr = validate_formula(formula)
        if not vr.valid:
            print(f"  [SKIP] Validation failed: {vr.errors}")
            _append_trace(
                trace_path,
                "validation_failed",
                {"run_id": run_id, "formula": formula, "errors": list(vr.errors or [])},
            )
            _record_idea_outcome(idea, "syntax_error")
            results.append({
                "run_id": run_id,
                "formula": formula,
                "thought_process": idea.get("thought_process", ""),
                **_source_fields(idea),
                "status": "invalid",
                "errors": vr.errors,
            })
            continue
        if vr.warnings:
            print(f"  [WARN] {vr.warnings}")
            _append_trace(
                trace_path,
                "validation_warning",
                {"run_id": run_id, "formula": formula, "warnings": list(vr.warnings or [])},
            )

        complexity_reason = _complexity_rejection_reason(formula, cfg)
        if complexity_reason:
            cost = _formula_compute_cost(formula)
            print(f"  [SKIP] Complexity gate rejected: {complexity_reason}")
            _append_trace(
                trace_path,
                "complexity_rejected",
                {"run_id": run_id, "formula": formula, "reason": complexity_reason, "cost": cost},
            )
            _record_idea_outcome(idea, "complexity_rejected")
            results.append({
                "run_id": run_id,
                "formula": formula,
                "thought_process": idea.get("thought_process", ""),
                **_source_fields(idea),
                "postprocess": postmode,
                "lookback_days": lookback,
                "status": "complexity_rejected",
                "errors": complexity_reason,
                "complexity": cost,
                "PassGates": False,
                "Score": 0,
                "eval_days": len(eval_days),
                "artifact_policy": "skipped_complexity",
            })
            continue

        # 3b. Compute alpha on a shorter screening window first
        try:
            alpha_screen = compute_alpha(
                formula      = formula,
                pv           = pv,
                days         = screen_days,
                lookback_days= lookback,
                postprocess_mode = postmode,
            )
            print(f"  [screen-compute] alpha shape={alpha_screen.shape}, "
                  f"non-null={alpha_screen.notna().sum()}, "
                  f"range=[{alpha_screen.min():.3f}, {alpha_screen.max():.3f}]")
        except Exception as e:
            print(f"  [SKIP] Computation error: {e}")
            _append_trace(
                trace_path,
                "screen_compute_error",
                {"run_id": run_id, "formula": formula, "error": str(e)},
            )
            _record_idea_outcome(idea, "compute_error")
            results.append({
                "run_id": run_id,
                "formula": formula,
                "thought_process": idea.get("thought_process", ""),
                **_source_fields(idea),
                "status": "compute_error",
                "error": str(e),
            })
            continue

        # 3c. Fast screen metrics on recent subset
        try:
            alpha_screen, formula, screen_metrics = _evaluate_with_optional_flip(
                alpha_screen, formula, hub, screen_days
            )
            sic = float(screen_metrics.get("IC", 0) or 0)
            sir = float(screen_metrics.get("IR", 0) or 0)
            stvr = float(screen_metrics.get("Turnover", 0) or 0)
            spass = bool(screen_metrics.get("PassGates", False))
            sscore = float(screen_metrics.get("Score", 0) or 0)
            print(f"  [screen-metrics] IC={sic:.4f}  IR={sir:.4f}  tvr={stvr:.2f}  "
                  f"PassGates={spass}  score={sscore:.4f}")
            _append_trace(
                trace_path,
                "screen_metrics",
                {
                    "run_id": run_id,
                    "formula": formula,
                    "IC": sic,
                    "IR": sir,
                    "Turnover": stvr,
                    "PassGates": spass,
                    "Score": sscore,
                },
            )
        except Exception as e:
            print(f"  [WARN] Screen evaluation error: {e}")
            screen_metrics = {}
            _append_trace(
                trace_path,
                "screen_eval_error",
                {"run_id": run_id, "formula": formula, "error": str(e)},
            )

        tvr_opt_combo: str = ""   # name of winning TVR optimization combo, used in step 3d
        if not _should_promote_from_screen(screen_metrics, cfg):
            # 3c-opt. TVR rescue: if IC is promising but TVR is too high, try smoothing combos
            min_ic = _cfg_float(
                cfg,
                "AUTOALPHA_SCREEN_PROMOTE_MIN_IC",
                max(_cfg_float(cfg, "AUTOALPHA_SCREEN_MIN_IC", 0.02), 0.02),
            )
            stvr = float(screen_metrics.get("Turnover", 0) or 0)
            sic  = float(screen_metrics.get("IC", 0) or 0)
            tvr_rescued = False
            if (not _futures_mode()) and stvr > _cfg_float(cfg, "AUTOALPHA_SCREEN_PROMOTE_MAX_TVR", 360.0) and sic >= min_ic:
                print(f"  [tvr-opt] TVR={stvr:.0f} too high but IC={sic:.3f} promising — trying smoothing combos")
                try:
                    from autoalpha_v3.tvr_optimizer import try_reduce_tvr
                    opt_alpha, opt_metrics, opt_name = try_reduce_tvr(
                        alpha_screen=alpha_screen,
                        hub=hub,
                        screen_days=screen_days,
                        evaluate_fn=evaluate_alpha,
                        max_tvr=_cfg_float(cfg, "AUTOALPHA_SCREEN_PROMOTE_MAX_TVR", 360.0),
                        min_ic=min_ic,
                    )
                    if opt_alpha is not None and opt_metrics is not None:
                        print(f"  [tvr-opt] Rescued via {opt_name} — TVR={opt_metrics.get('Turnover',0):.0f}  IC={opt_metrics.get('IC',0):.3f}")
                        screen_metrics = opt_metrics
                        formula = f"tvr_opt:{opt_name}({formula})"
                        tvr_opt_combo = opt_name
                        tvr_rescued = True
                        _append_trace(trace_path, "tvr_opt_rescued",
                                      {"run_id": run_id, "combo": opt_name, "metrics": opt_metrics})
                    else:
                        print("  [tvr-opt] All combos failed — screening out")
                except Exception as e:
                    print(f"  [tvr-opt] Optimizer error: {e}")

            if not tvr_rescued:
                fail_details = _screen_promotion_failure_details(screen_metrics, cfg, expected_days=len(screen_days))
                reason = _screen_promotion_failure_reason(screen_metrics, cfg, expected_days=len(screen_days))
                print(f"  [screen] Rejected ({reason}) — skipping full-history evaluation")
                _append_trace(
                    trace_path,
                    "screen_rejected",
                    {"run_id": run_id, "formula": formula, "screen_metrics": screen_metrics},
                )
                _record_idea_outcome(idea, "screened_out")
                results.append({
                    "run_id": run_id,
                    "formula": formula,
                    "thought_process": idea.get("thought_process", ""),
                    **_source_fields(idea),
                    "postprocess": postmode,
                    "lookback_days": lookback,
                    "status": "screened_out",
                    "screen_fail_reason": reason,
                    "screen_fail_details": fail_details,
                    "IC": screen_metrics.get("IC", 0),
                    "IR": screen_metrics.get("IR", 0),
                    "tvr": screen_metrics.get("Turnover", 0),
                    "PassGates": False,
                    "Score": screen_metrics.get("Score", 0),
                    "gates_detail": screen_metrics.get("GatesDetail", {}),
                    "parquet_path": None,
                    "eval_days": len(eval_days),
                    "research_path": None,
                    "screening": {
                        "stage": "recent_subset",
                        "days": len(screen_days),
                        "covered_days": (screen_metrics.get("result_preview") or {}).get("nd", 0),
                        "promoted": False,
                        "fail_details": fail_details,
                        "result_preview": screen_metrics.get("result_preview", {}),
                        "gates_detail": screen_metrics.get("GatesDetail", {}),
                    },
                    "artifact_policy": "skipped_low_signal",
                })
                gc.collect()
                continue

        # 3c-confirm. Non-overlapping early-window confirmation before expensive
        # full discovery evaluation. This catches recent-window false positives.
        confirm_days_limit = max(0, _cfg_int(cfg, "AUTOALPHA_CONFIRM_DAYS", 120))
        confirm_days_count = min(confirm_days_limit, max(0, len(eval_days) - len(screen_days)))
        confirm_days = eval_days[:confirm_days_count] if confirm_days_count >= 60 else []
        if confirm_days:
            try:
                confirm_formula = formula
                if tvr_opt_combo and confirm_formula.startswith("tvr_opt:"):
                    confirm_formula = confirm_formula.split("(", 1)[1].rstrip(")")
                alpha_confirm = compute_alpha(
                    formula=confirm_formula,
                    pv=pv,
                    days=confirm_days,
                    lookback_days=lookback,
                    postprocess_mode=postmode,
                )
                alpha_confirm = _apply_tvr_combo(alpha_confirm, tvr_opt_combo)
                confirm_metrics = evaluate_alpha(alpha_confirm, hub, confirm_days)
                cic = float(confirm_metrics.get("IC", 0) or 0)
                cir = float(confirm_metrics.get("IR", 0) or 0)
                ctvr = float(confirm_metrics.get("Turnover", 0) or 0)
                cpass = bool(confirm_metrics.get("PassGates", False))
                cscore = float(confirm_metrics.get("Score", 0) or 0)
                print(
                    f"  [confirm-metrics] {confirm_days[0]}→{confirm_days[-1]} "
                    f"IC={cic:.4f}  IR={cir:.4f}  tvr={ctvr:.2f}  "
                    f"PassGates={cpass}  score={cscore:.4f}"
                )
                _append_trace(
                    trace_path,
                    "confirmation_metrics",
                    {
                        "run_id": run_id,
                        "formula": formula,
                        "window_start": confirm_days[0],
                        "window_end": confirm_days[-1],
                        "days": len(confirm_days),
                        "IC": cic,
                        "IR": cir,
                        "Turnover": ctvr,
                        "PassGates": cpass,
                        "Score": cscore,
                    },
                )
            except Exception as e:
                print(f"  [confirm] Evaluation error: {e} — rejecting before full-history evaluation")
                confirm_metrics = {"error": str(e), "PassGates": False}

            if not _should_promote_from_confirmation(confirm_metrics, cfg):
                fail_details = _confirmation_failure_details(confirm_metrics, cfg)
                reason = _confirmation_failure_reason(confirm_metrics, cfg)
                print(f"  [confirm] Rejected ({reason}) — skipping full-history evaluation")
                _append_trace(
                    trace_path,
                    "confirmation_rejected",
                    {
                        "run_id": run_id,
                        "formula": formula,
                        "screen_metrics": screen_metrics,
                        "confirmation_metrics": confirm_metrics,
                        "reason": reason,
                    },
                )
                _record_idea_outcome(idea, "screened_out")
                results.append({
                    "run_id": run_id,
                    "formula": formula,
                    "thought_process": idea.get("thought_process", ""),
                    **_source_fields(idea),
                    "postprocess": postmode,
                    "lookback_days": lookback,
                    "status": "screened_out",
                    "screen_fail_reason": reason,
                    "confirmation_fail_reason": reason,
                    "confirmation_fail_details": fail_details,
                    "IC": confirm_metrics.get("IC", 0),
                    "IR": confirm_metrics.get("IR", 0),
                    "tvr": confirm_metrics.get("Turnover", 0),
                    "PassGates": False,
                    "Score": confirm_metrics.get("Score", 0),
                    "gates_detail": confirm_metrics.get("GatesDetail", {}),
                    "parquet_path": None,
                    "eval_days": len(eval_days),
                    "research_path": None,
                    "screening": {
                        "stage": "confirmation_window",
                        "days": len(screen_days),
                        "window_start": screen_days[0],
                        "window_end": screen_days[-1],
                        "covered_days": (screen_metrics.get("result_preview") or {}).get("nd", 0),
                        "promoted": True,
                        "IC": screen_metrics.get("IC", 0),
                        "IR": screen_metrics.get("IR", 0),
                        "Turnover": screen_metrics.get("Turnover", 0),
                        "Score": screen_metrics.get("Score", 0),
                        "result_preview": screen_metrics.get("result_preview", {}),
                        "gates_detail": screen_metrics.get("GatesDetail", {}),
                    },
                    "confirmation": {
                        "stage": "early_discovery_window",
                        "days": len(confirm_days),
                        "window_start": confirm_days[0],
                        "window_end": confirm_days[-1],
                        "metrics": confirm_metrics,
                        "promoted": False,
                        "fail_details": fail_details,
                    },
                    "artifact_policy": "skipped_confirmation_failed",
                })
                gc.collect()
                continue

        # 3d. Discovery-window compute + evaluation for promoted ideas.
        # v3 intentionally learns only on 2022-2023. 2024 is report-only below.
        raw_formula = formula
        direction_flipped = False
        try:
            # Use the original formula (strip tvr_opt: prefix) for DSL evaluation
            if tvr_opt_combo and formula.startswith("tvr_opt:"):
                raw_formula = formula.split("(", 1)[1].rstrip(")")
            alpha = compute_alpha(
                formula=raw_formula,
                pv=pv,
                days=eval_days,
                lookback_days=lookback,
                postprocess_mode=postmode,
            )
            # Re-apply the same TVR-reduction combo that passed screening
            if tvr_opt_combo:
                try:
                    from autoalpha_v3.tvr_optimizer import combo_ema, combo_persistence, combo_extremes, combo_rolling
                    _combo_map = {
                        "ema_10":         lambda a: combo_ema(a, span=10),
                        "persistence_02": lambda a: combo_persistence(a, blend_alpha=0.2),
                        "extremes_q20":   lambda a: combo_extremes(a, q=0.2),
                        "rolling_15":     lambda a: combo_rolling(a, window=15),
                    }
                    if tvr_opt_combo in _combo_map:
                        alpha = _combo_map[tvr_opt_combo](alpha)
                        print(f"  [tvr-opt] Applied {tvr_opt_combo} to full-history alpha")
                except Exception as e:
                    print(f"  [tvr-opt] Could not apply combo to full alpha: {e}")
            print(f"  [full-compute] alpha shape={alpha.shape}, "
                  f"non-null={alpha.notna().sum()}, "
                  f"range=[{alpha.min():.3f}, {alpha.max():.3f}]")
            pre_eval_formula = formula
            alpha, formula, metrics = _evaluate_with_optional_flip(alpha, formula, hub, eval_days)
            direction_flipped = formula != pre_eval_formula and formula == f"neg({pre_eval_formula})"
            corr_info = _compute_library_correlation(
                alpha,
                eval_days,
                max_factors=_cfg_int(cfg, "AUTOALPHA_CORR_REFERENCE_FACTORS", 40),
            )
            metrics = _apply_low_correlation_gate(metrics, corr_info, cfg)
            ic = float(metrics.get("IC", 0) or 0)
            ir = float(metrics.get("IR", 0) or 0)
            tvr = float(metrics.get("Turnover", 0) or 0)
            passes = bool(metrics.get("PassGates", False))
            score = float(metrics.get("Score", 0) or 0)
            print(f"  [full-metrics] IC={ic:.4f}  IR={ir:.4f}  tvr={tvr:.2f}  "
                  f"PassGates={passes}  score={score:.4f}")
            if corr_info.get("compared", 0):
                print(
                    "  [corr-gate] "
                    f"max_abs_corr={corr_info.get('max_abs_corr', 0):.3f} "
                    f"closest={corr_info.get('closest_run_id') or '--'} "
                    f"pass={metrics.get('GatesDetail', {}).get('LowCorrelation')}"
                )
            _append_trace(
                trace_path,
                "full_metrics",
                {
                    "run_id": run_id,
                    "formula": formula,
                    "IC": ic,
                    "IR": ir,
                    "Turnover": tvr,
                    "PassGates": passes,
                    "Score": score,
                    "correlation": metrics.get("correlation", {}),
                },
            )
        except Exception as e:
            print(f"  [WARN] Full evaluation error: {e}")
            _append_trace(
                trace_path,
                "full_eval_error",
                {"run_id": run_id, "formula": formula, "error": str(e)},
            )
            _record_idea_outcome(idea, "compute_error")
            results.append({
                "run_id": run_id,
                "formula": formula,
                "thought_process": idea.get("thought_process", ""),
                **_source_fields(idea),
                "status": "compute_error",
                "error": str(e),
            })
            gc.collect()
            continue

        if not passes:
            reason = _gate_failure_reason(metrics)
            print(f"  [full] Rejected ({reason}) — skipping 2024 OOS, full-window export, and research")
            _append_trace(
                trace_path,
                "full_rejected",
                {
                    "run_id": run_id,
                    "formula": formula,
                    "reason": reason,
                    "metrics": metrics,
                    "screen_metrics": screen_metrics,
                    "artifact_policy": "skipped_full_gate_failed",
                },
            )
            _record_idea_outcome(idea, "screened_out")
            results.append({
                "run_id": run_id,
                "formula": formula,
                "thought_process": idea.get("thought_process", ""),
                **_source_fields(idea),
                "postprocess": postmode,
                "lookback_days": lookback,
                "status": "screened_out",
                "screen_fail_reason": reason,
                "full_fail_reason": reason,
                "IC": metrics.get("IC", 0),
                "IR": metrics.get("IR", 0),
                "tvr": metrics.get("Turnover", 0),
                "PassGates": False,
                "Score": metrics.get("Score", 0),
                "gates_detail": metrics.get("GatesDetail", {}),
                "correlation": metrics.get("correlation", {}),
                "parquet_path": None,
                "eval_days": len(eval_days),
                "eval_window": {
                    "mode": "discovery_train_only",
                    "start": eval_days[0],
                    "end": eval_days[-1],
                    "leakage_guard": "2024 OOS metrics are report-only and not used for discovery feedback.",
                    "export_parquet_window": "skipped because discovery PassGates=false",
                },
                "oss_2024": {
                    "start": oos_days[0] if oos_days else "",
                    "end": oos_days[-1] if oos_days else "",
                    "days": len(oos_days),
                    "metrics": {"skipped": True, "reason": "discovery PassGates=false"},
                    "used_for_feedback": False,
                },
                "full_2022_2024": {
                    "start": full_export_days[0] if full_export_days else "",
                    "end": full_export_days[-1] if full_export_days else "",
                    "days": len(full_export_days),
                    "metrics": {"skipped": True, "reason": "discovery PassGates=false"},
                    "used_for_discovery": False,
                },
                "period_metrics": [
                    _period_row(
                        period="discovery",
                        label="Discovery 2022-2023",
                        start=eval_days[0],
                        end=eval_days[-1],
                        days=len(eval_days),
                        metrics=metrics,
                        used_for_discovery=True,
                    )
                ],
                "oos_comparison": {},
                "research_path": None,
                "screening": {
                    "stage": "full_eval",
                    "days": len(screen_days),
                    "window_start": screen_days[0],
                    "window_end": screen_days[-1],
                    "covered_days": (screen_metrics.get("result_preview") or {}).get("nd", 0),
                    "promoted": True,
                    "full_evaluated": True,
                    "full_fail_reason": reason,
                    "IC": screen_metrics.get("IC", 0),
                    "IR": screen_metrics.get("IR", 0),
                    "Turnover": screen_metrics.get("Turnover", 0),
                    "Score": screen_metrics.get("Score", 0),
                    "result_preview": screen_metrics.get("result_preview", {}),
                    "gates_detail": screen_metrics.get("GatesDetail", {}),
                },
                "artifact_policy": "skipped_full_gate_failed",
            })
            del alpha
            gc.collect()
            continue

        oos_metrics: dict[str, Any] = {}
        full_export_alpha: pd.Series | None = None
        full_metrics: dict[str, Any] = {}
        if oos_days:
            try:
                alpha_oos = compute_alpha(
                    formula=raw_formula,
                    pv=pv,
                    days=oos_days,
                    lookback_days=lookback,
                    postprocess_mode=postmode,
                )
                if tvr_opt_combo:
                    try:
                        from autoalpha_v3.tvr_optimizer import combo_ema, combo_persistence, combo_extremes, combo_rolling
                        _combo_map = {
                            "ema_10":         lambda a: combo_ema(a, span=10),
                            "persistence_02": lambda a: combo_persistence(a, blend_alpha=0.2),
                            "extremes_q20":   lambda a: combo_extremes(a, q=0.2),
                            "rolling_15":     lambda a: combo_rolling(a, window=15),
                        }
                        if tvr_opt_combo in _combo_map:
                            alpha_oos = _combo_map[tvr_opt_combo](alpha_oos)
                    except Exception as e:
                        print(f"  [oos-2024] Could not apply TVR combo: {e}")
                if direction_flipped:
                    alpha_oos = -alpha_oos
                oos_metrics = evaluate_alpha(alpha_oos, hub, oos_days)
                print(
                    "  [oos-2024] report-only "
                    f"IC={float(oos_metrics.get('IC', 0) or 0):.4f}  "
                    f"IR={float(oos_metrics.get('IR', 0) or 0):.4f}  "
                    f"tvr={float(oos_metrics.get('Turnover', 0) or 0):.2f}  "
                    f"PassGates={bool(oos_metrics.get('PassGates', False))}"
                )
                _append_trace(
                    trace_path,
                    "oos_2024_metrics_report_only",
                    {
                        "run_id": run_id,
                        "formula": formula,
                        "IC": oos_metrics.get("IC", 0),
                        "IR": oos_metrics.get("IR", 0),
                        "Turnover": oos_metrics.get("Turnover", 0),
                        "PassGates": oos_metrics.get("PassGates", False),
                        "Score": oos_metrics.get("Score", 0),
                        "leakage_guard": "not used for screening, parent selection, PassGates, Score, or target-valid stop",
                    },
                )
            except Exception as e:
                print(f"  [oos-2024] report-only evaluation failed: {e}")
                oos_metrics = {"error": str(e)}

        if full_export_days:
            try:
                full_export_alpha = compute_alpha(
                    formula=raw_formula,
                    pv=pv,
                    days=full_export_days,
                    lookback_days=lookback,
                    postprocess_mode=postmode,
                )
                if tvr_opt_combo:
                    try:
                        from autoalpha_v3.tvr_optimizer import combo_ema, combo_persistence, combo_extremes, combo_rolling
                        _combo_map = {
                            "ema_10":         lambda a: combo_ema(a, span=10),
                            "persistence_02": lambda a: combo_persistence(a, blend_alpha=0.2),
                            "extremes_q20":   lambda a: combo_extremes(a, q=0.2),
                            "rolling_15":     lambda a: combo_rolling(a, window=15),
                        }
                        if tvr_opt_combo in _combo_map:
                            full_export_alpha = _combo_map[tvr_opt_combo](full_export_alpha)
                    except Exception as e:
                        print(f"  [full-2022-2024] Could not apply TVR combo: {e}")
                if direction_flipped:
                    full_export_alpha = -full_export_alpha
                full_metrics = evaluate_alpha(full_export_alpha, hub, full_export_days)
                print(
                    "  [full-2022-2024] export/report "
                    f"IC={float(full_metrics.get('IC', 0) or 0):.4f}  "
                    f"IR={float(full_metrics.get('IR', 0) or 0):.4f}  "
                    f"tvr={float(full_metrics.get('Turnover', 0) or 0):.2f}  "
                    f"Score={float(full_metrics.get('Score', 0) or 0):.4f}"
                )
            except Exception as e:
                print(f"  [full-2022-2024] full export/report evaluation failed: {e}")
                full_export_alpha = None
                full_metrics = {"error": str(e)}

        period_metrics = [
            _period_row(
                period="discovery",
                label="Discovery 2022-2023",
                start=eval_days[0],
                end=eval_days[-1],
                days=len(eval_days),
                metrics=metrics,
                used_for_discovery=True,
            ),
            _period_row(
                period="oos_2024",
                label="OOS 2024",
                start=oos_days[0] if oos_days else "",
                end=oos_days[-1] if oos_days else "",
                days=len(oos_days),
                metrics=oos_metrics,
                used_for_discovery=False,
            ),
            _period_row(
                period="full_2022_2024",
                label="Full 2022-2024",
                start=full_export_days[0] if full_export_days else "",
                end=full_export_days[-1] if full_export_days else "",
                days=len(full_export_days),
                metrics=full_metrics,
                used_for_discovery=False,
            ),
        ]

        # 3e. Export parquet + research only for promising full-eval outputs
        out_path = None
        research_path = None
        artifact_policy = "skipped_low_signal"
        if _should_materialize_artifacts(metrics, cfg):
            artifact_policy = "materialized"
            try:
                out_path = export_parquet(
                    full_export_alpha if full_export_alpha is not None else alpha,
                    run_id,
                    AUTOALPHA_OUT,
                    start_date=full_export_days[0] if full_export_days else None,
                    end_date=full_export_days[-1] if full_export_days else None,
                )
            except Exception as e:
                print(f"  [WARN] Export error: {e}")
                out_path = None

            try:
                research_path = factor_research.analyze_factor(
                    run_id=run_id,
                    formula=formula,
                    alpha=alpha,
                    metrics=metrics,
                    hub=hub,
                    eval_days=eval_days,
                    thought_process=idea.get("thought_process", ""),
                )
            except Exception as e:
                print(f"  [WARN] Research analysis error: {e}")

            # If this factor passes gates, update pairwise correlations for ALL passing cards
            if metrics.get("PassGates") and research_path:
                try:
                    started = factor_research.schedule_factor_correlation_refresh(update_cards=True)
                    print(f"  [research] Global correlation refresh scheduled: started={started}")
                except Exception as e:
                    print(f"  [WARN] Global correlation update failed: {e}")
        else:
            print("  [artifacts] Skipping parquet/research for low-signal full-history result")
            _append_trace(
                trace_path,
                "artifact_skipped",
                {"run_id": run_id, "formula": formula, "policy": "skipped_low_signal"},
            )

        result = {
            "run_id":         run_id,
            "formula":        formula,
            "thought_process": idea.get("thought_process", ""),
            **_source_fields(idea),
            "postprocess":    postmode,
            "lookback_days":  lookback,
            "status":         "ok",
            "IC":             metrics.get("IC", 0),
            "IR":             metrics.get("IR", 0),
            "tvr":            metrics.get("Turnover", 0),
            "PassGates":      metrics.get("PassGates", False),
            "Score":          metrics.get("Score", 0),
            "gates_detail":   metrics.get("GatesDetail", {}),
            "correlation":    metrics.get("correlation", {}),
            "parquet_path":   str(out_path) if out_path else None,
            "eval_days":      len(eval_days),
            "eval_window": {
                "mode": "discovery_train_only",
                "start": eval_days[0],
                "end": eval_days[-1],
                "leakage_guard": "2024 OOS metrics are report-only and not used for discovery feedback.",
                "export_parquet_window": "2022-2024 full three-year panel",
            },
            "oss_2024": {
                "start": oos_days[0] if oos_days else "",
                "end": oos_days[-1] if oos_days else "",
                "days": len(oos_days),
                "metrics": oos_metrics,
                "used_for_feedback": False,
            },
            "full_2022_2024": {
                "start": full_export_days[0] if full_export_days else "",
                "end": full_export_days[-1] if full_export_days else "",
                "days": len(full_export_days),
                "metrics": full_metrics,
                "used_for_discovery": False,
            },
            "period_metrics": period_metrics,
            "oos_comparison": _oos_comparison(metrics, oos_metrics),
            "research_path":  research_path,
            "screening": {
                "stage": "recent_subset",
                "days": len(screen_days),
                "window_start": screen_days[0],
                "window_end": screen_days[-1],
                "covered_days": (screen_metrics.get("result_preview") or {}).get("nd", 0),
                "promoted": True,
                "IC": screen_metrics.get("IC", 0),
                "IR": screen_metrics.get("IR", 0),
                "Turnover": screen_metrics.get("Turnover", 0),
                "Score": screen_metrics.get("Score", 0),
                "result_preview": screen_metrics.get("result_preview", {}),
                "gates_detail": screen_metrics.get("GatesDetail", {}),
            },
            "artifact_policy": artifact_policy,
        }
        results.append(result)
        _record_idea_outcome(idea, "passing" if metrics.get("PassGates") else "screened_out")
        if metrics.get("PassGates"):
            try:
                record_inspiration_pass(idea.get("inspiration_ids") or [])
            except Exception as exc:
                print(f"  [WARN] Inspiration pass writeback failed: {exc}")
        _append_trace(
            trace_path,
            "result_saved",
            {
                "run_id": run_id,
                "formula": formula,
                "status": result.get("status"),
                "PassGates": result.get("PassGates"),
                "Score": result.get("Score"),
                "correlation": result.get("correlation", {}),
                "oss_2024": result.get("oss_2024", {}),
                "artifact_policy": artifact_policy,
            },
        )

        # 3f. Feishu notification for passing factors
        if metrics.get("PassGates"):
            try:
                base_description = str(idea.get("thought_process", "")).strip()
                tldr = summarize_factor_tldr(
                    base_description,
                    formula=formula,
                    source_hint=f"发现窗口 2022-2023 共 {len(eval_days)} 天；2024 OOS 仅报告；后处理 {postmode}；lookback {lookback} 天",
                )
                description = (
                    f"{base_description}\n\n"
                    f"📅 发现窗口: 2022-2023 / {len(eval_days)} 交易日\n"
                    f"🧪 2024 OOS: 仅报告，不参与反馈\n"
                    f"🔧 后处理: {postmode} | 回看: {lookback}天"
                )
                _feishu.send_factor_notification(
                    factor_name=run_id,
                    description=description,
                    metrics={
                        "Score": score,
                        "PassGates": True,
                        "IC": ic,
                        "IR": ir,
                        "Turnover": tvr,
                        "tldr": tldr,
                    },
                    formula=formula,
                    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                )
                print(f"  [feishu] ✅ Notification sent for {run_id}")
            except Exception as e:
                print(f"  [feishu] ⚠ Failed to send notification: {e}")

        gc.collect()

    # ── 4. Save run manifest ─────────────────────────────────────────────────
    manifest_path = AUTOALPHA_OUT / f"run_{timestamp}.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[pipeline] Manifest saved → {manifest_path}")
    _append_trace(trace_path, "run_complete", {"manifest_path": str(manifest_path), "result_count": len(results)})

    return results
