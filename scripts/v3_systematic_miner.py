from __future__ import annotations

import argparse
import gc
import json
import math
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_PARENT = PROJECT_ROOT.parent
for _path in (PROJECT_PARENT, PROJECT_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from autoalpha_v3 import knowledge_base as kb
from autoalpha_v3.pipeline import (
    AUTOALPHA_OUT,
    DISCOVERY_END,
    DISCOVERY_START,
    OOS_END,
    OOS_START,
    _apply_low_correlation_gate,
    _read_alpha_series_from_parquet,
    evaluate_alpha,
    export_parquet,
)
from manual.manual_factor_runner import (
    CandidateSpec,
    ManualFactorDataset,
    audit_manual_spec,
    compute_raw,
    cs_rank,
    generate_candidates,
)
from runtime_config import load_runtime_config


SYSTEMATIC_ROOT = PROJECT_ROOT / "systematic"
REPORT_ROOT = SYSTEMATIC_ROOT / "reports"


def _date_mask(index: pd.MultiIndex, start: str, end: str) -> np.ndarray:
    dates = pd.to_datetime(index.get_level_values("date"))
    return (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))


def _slice_wide(frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    return frame.loc[_date_mask(frame.index, start, end)]


def _wide_to_long(alpha: pd.DataFrame) -> pd.Series:
    series = alpha.stack(dropna=False).rename("alpha")
    return series.replace([np.inf, -np.inf], np.nan).dropna().astype("float32")


def _wide_to_corr_vector(alpha: pd.DataFrame) -> np.ndarray:
    return alpha.to_numpy(dtype="float32", copy=True).ravel()


def _corr_info_from_vectors(
    current: np.ndarray,
    accepted_vectors: list[tuple[str, np.ndarray]],
) -> dict[str, Any]:
    if current.size == 0 or not accepted_vectors:
        return {"max_abs_corr": 0.0, "max_corr": 0.0, "closest_run_id": "", "compared": 0}

    current64 = current.astype("float64", copy=False)
    current_mask = np.isfinite(current64)
    best = {"max_abs_corr": 0.0, "max_corr": 0.0, "closest_run_id": "", "compared": 0}
    for run_id, prior in accepted_vectors:
        if prior.size != current.size:
            continue
        prior64 = prior.astype("float64", copy=False)
        mask = current_mask & np.isfinite(prior64)
        if int(mask.sum()) < 10_000:
            continue
        x = current64[mask]
        y = prior64[mask]
        x = x - x.mean()
        y = y - y.mean()
        denom = float(np.sqrt(np.dot(x, x) * np.dot(y, y)))
        if denom <= 0:
            continue
        corr = float(np.dot(x, y) / denom)
        if not np.isfinite(corr):
            continue
        best["compared"] = int(best["compared"]) + 1
        abs_corr = abs(corr)
        if abs_corr > float(best["max_abs_corr"]):
            best = {
                "max_abs_corr": abs_corr,
                "max_corr": corr,
                "closest_run_id": run_id,
                "compared": int(best["compared"]),
            }
    return best


def _seed_accepted_vectors(
    train_index: pd.MultiIndex,
    columns: pd.Index,
) -> list[tuple[str, np.ndarray]]:
    vectors: list[tuple[str, np.ndarray]] = []
    for item in kb.list_valid_factors():
        parquet_path = str(item.get("parquet_path", "") or "")
        series = _read_alpha_series_from_parquet(parquet_path)
        if series is None or series.empty:
            continue
        try:
            wide = series.unstack("security_id").reindex(index=train_index, columns=columns)
        except Exception:
            continue
        vectors.append((str(item.get("run_id", "")), _wide_to_corr_vector(wide)))
        print(f"[v3sys] seeded corr reference {item.get('run_id')} from {parquet_path}", flush=True)
    return vectors


def _canonical_params(params: dict[str, Any]) -> str:
    return json.dumps(params or {}, sort_keys=True, ensure_ascii=False)


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


def _dsl_expression(expression: str) -> str:
    text = expression
    replacements = {
        "rank(": "cs_rank(",
        "zscore(": "ts_zscore(",
        "mean(": "ts_mean(",
        "std(": "ts_std(",
        "ema(": "ts_ema(",
        "delay(": "lag(",
        "eps": "1e-8",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    field_replacements = {
        "close": "close_trade_px",
        "open": "open_trade_px",
        "high": "high_trade_px",
        "low": "low_trade_px",
    }
    for src, dst in field_replacements.items():
        text = re.sub(rf"\b{src}\b", dst, text)
    return text


def _display_formula(spec: CandidateSpec) -> str:
    base = _dsl_expression(spec.expression)
    smooth = int(spec.params.get("smooth", 0) or 0)
    return f"ts_ema({base}, {smooth})" if smooth > 1 else base


def _make_spec(family: str, params: dict[str, Any], direction: int) -> CandidateSpec:
    labels = {
        "close_zscore": "Close Z-Score Stretch",
        "volume_conditioned_return": "Return x Volume Surprise",
        "dvolume_conditioned_return": "Return x Dollar-Volume Surprise",
        "trade_conditioned_return": "Return x Trade-Count Surprise",
        "avg_trade_conditioned_return": "Return x Average Trade Value Surprise",
        "volatility_conditioned_return": "Return / Recent Volatility",
        "range_conditioned_body": "Body x Range Surprise",
        "range_conditioned_location": "Range Location x Range Surprise",
        "vwap_gap_with_dvol": "VWAP Gap x Dollar-Volume Surprise",
        "zscore_vwap_gap": "VWAP Gap Z-Score",
        "zscore_body_fraction": "Body Fraction Z-Score",
        "ema_spread": "Short-Long EMA Spread",
    }
    if family == "ema_spread":
        expr = f"rank({direction:+d} * (ema(close,{params['short']})/ema(close,{params['long']}) - 1))"
    elif family == "close_zscore":
        expr = f"rank({direction:+d} * zscore(close_trade_px, {params['window']}))"
    elif family == "volume_conditioned_return":
        expr = f"rank({direction:+d} * close_ret * (volume/mean(volume,{params['window']})))"
    elif family == "dvolume_conditioned_return":
        expr = f"rank({direction:+d} * close_ret * (dvolume/mean(dvolume,{params['window']})))"
    elif family == "trade_conditioned_return":
        expr = f"rank({direction:+d} * close_ret * (trade_count/mean(trade_count,{params['window']})))"
    elif family == "avg_trade_conditioned_return":
        expr = (
            f"rank({direction:+d} * close_ret * "
            f"((dvolume/trade_count)/mean(dvolume/trade_count,{params['window']})))"
        )
    elif family == "volatility_conditioned_return":
        expr = f"rank({direction:+d} * close_ret / std(close_ret,{params['window']}))"
    elif family == "range_conditioned_body":
        expr = f"rank({direction:+d} * body_frac * (range_pct/mean(range_pct,{params['window']})))"
    elif family == "range_conditioned_location":
        expr = f"rank({direction:+d} * range_loc * (range_pct/mean(range_pct,{params['window']})))"
    elif family == "vwap_gap_with_dvol":
        expr = f"rank({direction:+d} * (close_trade_px/vwap-1) * (dvolume/mean(dvolume,{params['window']})))"
    elif family == "zscore_vwap_gap":
        expr = f"rank({direction:+d} * zscore(close_trade_px/vwap - 1,{params['window']}))"
    elif family == "zscore_body_fraction":
        expr = f"rank({direction:+d} * zscore((close-open)/(high-low+eps),{params['window']}))"
    else:
        raise ValueError(f"Unsupported v3 systematic family: {family}")

    return CandidateSpec(
        family=family,
        family_label=labels[family],
        params=params,
        direction=direction,
        expression=expr,
        description=(
            f"v3 systematic candidate for {labels[family]} with params={params}; "
            "selected only by 2022-2023 discovery metrics plus low-correlation gating."
        ),
    )


def generate_v3_candidates() -> list[CandidateSpec]:
    """Expand the v2 manual families into a v3 diversity-oriented candidate pool."""
    def _with_smooth(spec: CandidateSpec, smooth: int) -> CandidateSpec:
        if smooth <= 1:
            return spec
        params = dict(spec.params)
        params["smooth"] = smooth
        return CandidateSpec(
            family=spec.family,
            family_label=spec.family_label,
            params=params,
            direction=spec.direction,
            expression=spec.expression,
            description=f"{spec.description} Additional {smooth}-bar EMA smoothing is applied to reduce turnover.",
        )

    by_key = {}
    for spec in generate_candidates():
        by_key.setdefault(spec.key, spec)
        for smooth in (8, 16, 32):
            smoothed = _with_smooth(spec, smooth)
            by_key.setdefault(smoothed.key, smoothed)

    window_families = [
        "close_zscore",
        "volume_conditioned_return",
        "dvolume_conditioned_return",
        "trade_conditioned_return",
        "avg_trade_conditioned_return",
        "volatility_conditioned_return",
        "range_conditioned_body",
        "range_conditioned_location",
        "vwap_gap_with_dvol",
        "zscore_vwap_gap",
        "zscore_body_fraction",
    ]
    windows = [4, 6, 8, 10, 12, 16, 20, 24, 32, 48]
    for family in window_families:
        for window in windows:
            for direction in (-1, 1):
                spec = _make_spec(family, {"window": window}, direction)
                by_key.setdefault(spec.key, spec)
                for smooth in (8, 16, 32):
                    smoothed = _with_smooth(spec, smooth)
                    by_key.setdefault(smoothed.key, smoothed)

    for short, long in [(3, 12), (4, 16), (5, 20), (6, 24), (8, 32), (10, 40), (12, 48)]:
        for direction in (-1, 1):
            spec = _make_spec("ema_spread", {"short": short, "long": long}, direction)
            by_key.setdefault(spec.key, spec)
            for smooth in (8, 16):
                smoothed = _with_smooth(spec, smooth)
                by_key.setdefault(smoothed.key, smoothed)

    family_priority = {
        "ema_spread": 0,
        "close_zscore": 1,
        "volatility_conditioned_return": 2,
        "zscore_body_fraction": 3,
        "zscore_vwap_gap": 4,
        "avg_trade_conditioned_return": 5,
        "trade_conditioned_return": 6,
        "dvolume_conditioned_return": 7,
        "volume_conditioned_return": 8,
        "vwap_gap_with_dvol": 9,
        "range_conditioned_body": 10,
        "range_conditioned_location": 11,
        "range_location": 12,
        "wick_imbalance": 13,
        "body_fraction": 14,
    }

    def _expected_smoothness(spec: CandidateSpec) -> int:
        if "long" in spec.params:
            return int(spec.params["long"])
        if "window" in spec.params:
            return int(spec.params["window"])
        return 0

    return sorted(
        by_key.values(),
        key=lambda spec: (
            family_priority.get(spec.family, 99),
            -int(spec.params.get("smooth", 0) or 0),
            -_expected_smoothness(spec),
            spec.key,
        ),
    )


def _score_sort_key(record: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(record.get("Score", 0.0) or 0.0),
        float(record.get("IC", 0.0) or 0.0),
        float(record.get("IR", 0.0) or 0.0),
    )


def _cfg_bool(cfg: dict[str, Any], key: str, default: bool) -> bool:
    raw = cfg.get(key)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _maybe_run_combo_retrain(
    *,
    cfg: dict[str, Any],
    started_at: float,
    target_valid: int,
) -> bool:
    if not _cfg_bool(cfg, "AUTOALPHA_V3_AUTO_COMBO_RETRAIN", True):
        return False
    min_valid = int(cfg.get("AUTOALPHA_MODEL_LAB_MIN_FACTORS", "10") or 10)
    min_runtime_sec = int(cfg.get("AUTOALPHA_MODEL_LAB_MIN_RUNTIME_SEC", "3600") or 3600)
    total_valid = len(kb.list_valid_factors())
    elapsed = time.time() - started_at
    if total_valid < min_valid or elapsed < min_runtime_sec:
        return False

    from autoalpha_v3.rolling_model_lab import run_model_lab

    target_for_lab = max(min_valid, min(total_valid, max(target_valid, total_valid)))
    print(
        "[v3sys] combo retrain trigger satisfied | "
        f"valid={total_valid} min_valid={min_valid} elapsed={elapsed:.0f}s min_elapsed={min_runtime_sec}s "
        f"| target_for_lab={target_for_lab}",
        flush=True,
    )
    summary = run_model_lab(
        target_valid_count=target_for_lab,
        ideas_per_round=0,
        eval_days_count=0,
        max_rounds=0,
        sleep_seconds=0.0,
        train_days=int(cfg.get("AUTOALPHA_ROLLING_TRAIN_DAYS", "126") or 126),
        test_days=int(cfg.get("AUTOALPHA_ROLLING_TEST_DAYS", "126") or 126),
        step_days=int(cfg.get("AUTOALPHA_ROLLING_STEP_DAYS", "126") or 126),
        allow_partial=True,
    )
    print(
        f"[v3sys] combo retrain finished | best_model={summary.get('best_model')} "
        f"selected={summary.get('selected_factor_count')} best_score={summary.get('best_score')}",
        flush=True,
    )
    return True


def _record_result(
    *,
    spec: CandidateSpec,
    alpha_train: pd.Series,
    alpha_full: pd.Series,
    metrics: dict[str, Any],
    corr_info: dict[str, Any],
    oos_metrics: dict[str, Any],
    full_metrics: dict[str, Any],
    train_days: list[str],
    oos_days: list[str],
    full_days: list[str],
    run_stamp: str,
    accepted_index: int,
) -> dict[str, Any]:
    run_id = f"v3sys_{run_stamp}_{accepted_index:02d}"
    parquet_path = export_parquet(
        alpha_full,
        run_id,
        AUTOALPHA_OUT,
        start_date=full_days[0] if full_days else DISCOVERY_START,
        end_date=full_days[-1] if full_days else OOS_END,
    )
    period_metrics = [
        _period_row(
            period="discovery",
            label="Discovery 2022-2023",
            start=DISCOVERY_START,
            end=DISCOVERY_END,
            days=len(train_days),
            metrics=metrics,
            used_for_discovery=True,
        ),
        _period_row(
            period="oos_2024",
            label="OOS 2024",
            start=OOS_START,
            end=OOS_END,
            days=len(oos_days),
            metrics=oos_metrics,
            used_for_discovery=False,
        ),
        _period_row(
            period="full_2022_2024",
            label="Full 2022-2024",
            start=full_days[0] if full_days else DISCOVERY_START,
            end=full_days[-1] if full_days else OOS_END,
            days=len(full_days),
            metrics=full_metrics,
            used_for_discovery=False,
        ),
    ]
    result = {
        "run_id": run_id,
        "formula": _display_formula(spec),
        "thought_process": spec.description,
        "IC": float(metrics.get("IC", 0.0) or 0.0),
        "IR": float(metrics.get("IR", 0.0) or 0.0),
        "tvr": float(metrics.get("Turnover", 0.0) or 0.0),
        "Turnover": float(metrics.get("Turnover", 0.0) or 0.0),
        "Score": float(metrics.get("Score", 0.0) or 0.0),
        "PassGates": bool(metrics.get("PassGates", False)),
        "correlation": metrics.get("correlation", corr_info),
        "oss_2024": {
            "start": OOS_START,
            "end": OOS_END,
            "days": len(oos_days),
            "metrics": oos_metrics,
            "used_for_feedback": False,
        },
        "full_2022_2024": {
            "start": full_days[0] if full_days else DISCOVERY_START,
            "end": full_days[-1] if full_days else OOS_END,
            "days": len(full_days),
            "metrics": full_metrics,
            "used_for_discovery": False,
        },
        "period_metrics": period_metrics,
        "oos_comparison": _oos_comparison(metrics, oos_metrics),
        "eval_window": {
            "discovery_start": DISCOVERY_START,
            "discovery_end": DISCOVERY_END,
            "discovery_days": len(train_days),
            "oos_start": OOS_START,
            "oos_end": OOS_END,
            "oos_days": len(oos_days),
            "oos_used_for_feedback": False,
            "export_parquet_window": "2022-2024 full three-year panel",
        },
        "postprocess": "manual_cs_rank",
        "lookback_days": 0,
        "status": "ok",
        "inspiration_source_type": "systematic",
        "inspiration_source_types": ["systematic"],
        "inspiration_ids": [],
        "generation_mode": "v3_systematic_lowcorr",
        "target_source": "systematic_train_only",
        "prompt_version": "v3-systematic-oos-lowcorr-20260427",
        "parquet_path": str(parquet_path),
        "eval_days": len(train_days),
        "screening": {
            "method": "full discovery window",
            "days": len(train_days),
            "used_2024": False,
            "exported_days": len(full_days),
        },
        "systematic_spec": {
            "family": spec.family,
            "family_label": spec.family_label,
            "params": spec.params,
            "direction": spec.direction,
            "expression": spec.expression,
            "display_formula": _display_formula(spec),
            "future_info_check": audit_manual_spec(spec),
        },
    }
    kb.add_factor(result, parent_run_ids=[])
    return result


def mine(target_valid: int, max_candidates: int = 0) -> dict[str, Any]:
    cfg = load_runtime_config()
    started_at = time.time()
    combo_retrained = False
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)

    dataset = ManualFactorDataset()
    all_days = dataset.trading_days
    train_days = [day for day in all_days if pd.Timestamp(DISCOVERY_START) <= pd.Timestamp(day) <= pd.Timestamp(DISCOVERY_END)]
    oos_days = [day for day in all_days if pd.Timestamp(OOS_START) <= pd.Timestamp(day) <= pd.Timestamp(OOS_END)]
    full_days = [day for day in all_days if pd.Timestamp(DISCOVERY_START) <= pd.Timestamp(day) <= pd.Timestamp(OOS_END)]
    if not train_days:
        raise RuntimeError("No 2022-2023 discovery days available.")

    candidates = generate_v3_candidates()
    discoveries: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    existing_factors = kb.list_valid_factors()
    existing_valid = len(existing_factors)
    accepted_needed = max(0, target_valid - existing_valid)
    train_index = _slice_wide(dataset.close, DISCOVERY_START, DISCOVERY_END).index
    accepted_vectors: list[tuple[str, np.ndarray]] = _seed_accepted_vectors(train_index, dataset.columns)
    accepted_spec_keys = {
        (
            str((item.get("systematic_spec") or {}).get("family", "")),
            _canonical_params((item.get("systematic_spec") or {}).get("params") or {}),
            int((item.get("systematic_spec") or {}).get("direction", 0) or 0),
        )
        for item in existing_factors
        if (item.get("systematic_spec") or {}).get("family")
    }
    family_counts = Counter(
        str((item.get("systematic_spec") or {}).get("family", ""))
        for item in existing_factors
        if (item.get("systematic_spec") or {}).get("family")
    )
    candidates = sorted(
        enumerate(candidates),
        key=lambda item: (
            family_counts.get(item[1].family, 0),
            item[0],
        ),
    )
    candidates = [item[1] for item in candidates]
    if max_candidates > 0:
        candidates = candidates[:max_candidates]
    max_corr_threshold = float(cfg.get("AUTOALPHA_MAX_LIBRARY_CORR", 0.72) or 0.72)
    max_per_family = int(cfg.get("AUTOALPHA_V3_MAX_PER_FAMILY", 3) or 3)
    print(
        f"[v3sys] dataset loaded | train={train_days[0]}->{train_days[-1]} days={len(train_days)} "
        f"| oos={oos_days[0] if oos_days else 'NA'}->{oos_days[-1] if oos_days else 'NA'} days={len(oos_days)} "
        f"| export_full={full_days[0] if full_days else 'NA'}->{full_days[-1] if full_days else 'NA'} days={len(full_days)} "
        f"| candidates={len(candidates)} | target_valid={target_valid} "
        f"| existing_family_counts={dict(family_counts)}",
        flush=True,
    )
    if accepted_needed == 0:
        return {"run_stamp": run_stamp, "accepted": [], "message": "target already satisfied"}

    for idx, spec in enumerate(candidates, start=1):
        spec_key = (spec.family, _canonical_params(spec.params), int(spec.direction))
        if spec_key in accepted_spec_keys:
            continue
        if family_counts.get(spec.family, 0) >= max_per_family:
            continue
        started = time.time()
        audit = audit_manual_spec(spec)
        if not audit["passed"]:
            print(f"[v3sys] {idx:03d}/{len(candidates)} skip audit {spec.key}: {audit['errors']}", flush=True)
            continue

        raw = compute_raw(spec, dataset)
        alpha_wide = cs_rank(raw).astype("float32").replace([np.inf, -np.inf], np.nan)
        smooth = int(spec.params.get("smooth", 0) or 0)
        if smooth > 1:
            alpha_wide = alpha_wide.ewm(span=smooth, adjust=False, min_periods=max(2, min(smooth, 8))).mean().astype("float32")
        alpha_train_wide = _slice_wide(alpha_wide, DISCOVERY_START, DISCOVERY_END)
        alpha_oos_wide = _slice_wide(alpha_wide, OOS_START, OOS_END) if oos_days else alpha_wide.iloc[0:0]
        alpha_full_wide = _slice_wide(alpha_wide, DISCOVERY_START, OOS_END)
        alpha_train = _wide_to_long(alpha_train_wide)
        alpha_oos = _wide_to_long(alpha_oos_wide)
        alpha_full = _wide_to_long(alpha_full_wide)
        corr_vector = _wide_to_corr_vector(alpha_train_wide)

        corr_info = _corr_info_from_vectors(corr_vector, accepted_vectors)
        if float(corr_info.get("max_abs_corr", 0.0) or 0.0) > max_corr_threshold:
            metrics = {
                "IC": 0.0,
                "IR": 0.0,
                "Turnover": 0.0,
                "Score": 0.0,
                "PassGates": False,
                "correlation": {
                    **corr_info,
                    "threshold": max_corr_threshold,
                    "PassLowCorrelation": False,
                },
            }
            oos_metrics = {}
            full_metrics = {}
        else:
            metrics = evaluate_alpha(alpha_train, dataset.hub, train_days)
            metrics = _apply_low_correlation_gate(metrics, corr_info, cfg)
            if metrics.get("PassGates"):
                oos_metrics = evaluate_alpha(alpha_oos, dataset.hub, oos_days) if oos_days and not alpha_oos.empty else {}
                full_metrics = evaluate_alpha(alpha_full, dataset.hub, full_days) if full_days and not alpha_full.empty else {}
            else:
                oos_metrics = {}
                full_metrics = {}

        record = {
            "key": spec.key,
            "family": spec.family,
            "params": _canonical_params(spec.params),
            "direction": spec.direction,
            "expression": spec.expression,
            "IC": float(metrics.get("IC", 0.0) or 0.0),
            "IR": float(metrics.get("IR", 0.0) or 0.0),
            "Turnover": float(metrics.get("Turnover", 0.0) or 0.0),
            "Score": float(metrics.get("Score", 0.0) or 0.0),
            "PassGates": bool(metrics.get("PassGates", False)),
            "max_abs_corr": float(corr_info.get("max_abs_corr", 0.0) or 0.0),
            "closest_run_id": corr_info.get("closest_run_id", ""),
            "oos_IC": float(oos_metrics.get("IC", 0.0) or 0.0) if oos_metrics else 0.0,
            "oos_IR": float(oos_metrics.get("IR", 0.0) or 0.0) if oos_metrics else 0.0,
            "oos_Score": float(oos_metrics.get("Score", 0.0) or 0.0) if oos_metrics else 0.0,
            "full_Score": float(full_metrics.get("Score", 0.0) or 0.0) if full_metrics else 0.0,
            "elapsed_seconds": round(time.time() - started, 3),
        }
        discoveries.append(record)

        print(
            f"[v3sys] {idx:03d}/{len(candidates)} {spec.key} | "
            f"IC={record['IC']:.4f} IR={record['IR']:.2f} TVR={record['Turnover']:.2f} "
            f"Score={record['Score']:.3f} Corr={record['max_abs_corr']:.3f} "
            f"Pass={record['PassGates']} | OOS_IC={record['oos_IC']:.4f}",
            flush=True,
        )

        if record["PassGates"]:
            result = _record_result(
                spec=spec,
                alpha_train=alpha_train,
                alpha_full=alpha_full,
                metrics=metrics,
                corr_info=corr_info,
                oos_metrics=oos_metrics,
                full_metrics=full_metrics,
                train_days=train_days,
                oos_days=oos_days,
                full_days=full_days,
                run_stamp=run_stamp,
                accepted_index=existing_valid + len(accepted) + 1,
            )
            accepted.append(result)
            accepted_vectors.append((result["run_id"], corr_vector))
            accepted_spec_keys.add(spec_key)
            family_counts[spec.family] += 1
            corr_vector = np.array([], dtype="float32")
            print(
                f"[v3sys] accepted {result['run_id']} | total_valid={existing_valid + len(accepted)}/{target_valid}",
                flush=True,
            )
            if not combo_retrained:
                combo_retrained = _maybe_run_combo_retrain(
                    cfg=cfg,
                    started_at=started_at,
                    target_valid=target_valid,
                )
            auto_combo = _cfg_bool(cfg, "AUTOALPHA_V3_AUTO_COMBO_RETRAIN", True)
            min_valid_for_combo = int(cfg.get("AUTOALPHA_MODEL_LAB_MIN_FACTORS", "10") or 10)
            min_runtime_for_combo = int(cfg.get("AUTOALPHA_MODEL_LAB_MIN_RUNTIME_SEC", "3600") or 3600)
            should_keep_mining_for_combo = (
                auto_combo
                and not combo_retrained
                and len(kb.list_valid_factors()) >= min_valid_for_combo
                and time.time() - started_at < min_runtime_for_combo
            )
            if len(accepted) >= accepted_needed and not should_keep_mining_for_combo:
                del raw, alpha_wide, alpha_train_wide, alpha_oos_wide, alpha_full_wide, alpha_train, alpha_oos, alpha_full, corr_vector
                gc.collect()
                break

        del raw, alpha_wide, alpha_train_wide, alpha_oos_wide, alpha_full_wide, alpha_train, alpha_oos, alpha_full, corr_vector
        gc.collect()

    if not combo_retrained:
        combo_retrained = _maybe_run_combo_retrain(
            cfg=cfg,
            started_at=started_at,
            target_valid=target_valid,
        )

    report_csv = REPORT_ROOT / f"v3_systematic_search_{run_stamp}.csv"
    pd.DataFrame(discoveries).sort_values(["PassGates", "Score", "IC"], ascending=False).to_csv(report_csv, index=False)
    report_json = REPORT_ROOT / f"v3_systematic_accepted_{run_stamp}.json"
    report_json.write_text(json.dumps(accepted, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    return {
        "run_stamp": run_stamp,
        "target_valid": target_valid,
        "existing_valid": existing_valid,
        "accepted_count": len(accepted),
        "total_valid": len(kb.list_valid_factors()),
        "combo_retrained": combo_retrained,
        "report_csv": str(report_csv),
        "report_json": str(report_json),
        "accepted": [
            {
                "run_id": item["run_id"],
                "Score": item["Score"],
                "IC": item["IC"],
                "IR": item["IR"],
                "tvr": item["tvr"],
                "max_abs_corr": item.get("correlation", {}).get("max_abs_corr", 0.0),
            }
            for item in accepted
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v3 systematic train-only factor miner.")
    parser.add_argument("--target-valid", type=int, default=10)
    parser.add_argument("--max-candidates", type=int, default=0)
    args = parser.parse_args()
    summary = mine(target_valid=args.target_valid, max_candidates=args.max_candidates)
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    if int(summary.get("total_valid", 0) or 0) < int(args.target_valid):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
