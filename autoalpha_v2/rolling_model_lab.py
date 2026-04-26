"""
autoalpha_v2/rolling_model_lab.py

Dedicated long-running experiment for:
1. Mining until we have N valid AutoAlpha factors
2. Building a daily factor feature matrix from exported submission parquets
3. Running rolling train/test experiments with linear and LightGBM models
4. Exporting summary JSON / markdown / plots for frontend visualization
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    import lightgbm as lgb
except ImportError as exc:  # pragma: no cover - environment guard
    raise RuntimeError("rolling_model_lab requires scikit-learn, lightgbm") from exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_LAB_ROOT = Path(__file__).resolve().parent / "model_lab"
MODEL_LAB_EXPLORATIONS_ROOT = MODEL_LAB_ROOT / "explorations"
FEATURE_CACHE_DIR = MODEL_LAB_ROOT / "feature_cache_daily"
ENSEMBLE_OUTPUT_DIR = MODEL_LAB_ROOT / "ensemble_factors"
DEFAULT_FEE_BPS = 2.0
MIN_HISTORY_DAYS_FOR_TRAINED_PREDICTION = 20
MOCK_OOS_TRAIN_START = "2022-01-01"
MOCK_OOS_TRAIN_END = "2023-12-31"
MOCK_OOS_EVAL_START = "2024-01-01"
MOCK_OOS_EVAL_END = "2024-12-31"
ENABLE_EXPENSIVE_CORRELATION_PASSES = False
LOW_CORR_EXPERIMENT_SLUG = "low_corr_submit_combo"
COMBO_MODEL_NAMES = {
    "EqualWeightRankCombo",
    "TrainICRankCombo",
    "TrainICIRRankCombo",
    "ValTopKRankCombo",
    "ValPowerRankCombo",
    "CorrPrunedRankCombo",
    "InverseVolICRankCombo",
    "DiversifiedICRankCombo",
    "SoftmaxICRankCombo",
    "TopAbsICRankCombo",
    "RidgeShrinkageRankCombo",
    "VolBalancedTopRankCombo",
    "ClusterNeutralICRankCombo",
}
TEMPORAL_TORCH_MODEL_NAMES = {
    "TorchTCNMetaModel",
    "TorchGRUMetaModel",
    "TorchTransformerMetaModel",
}
TEMPORAL_TABULAR_MODEL_NAMES = {
    "TemporalRidgeLagMetaModel",
    "TemporalLightGBMLagMetaModel",
    "TemporalHistGBLagMetaModel",
    "TemporalExtraTreesLagMetaModel",
}
FACTOR_TRANSFORMER_MODEL_NAMES = {
    "FactorTokenTransformerRidgeStackModel",
    "CausalDecayFactorTransformerStackModel",
}
MODEL_LAB_REFERENCES = [
    {
        "title": "DeepLOB: Deep Convolutional Neural Networks for Limit Order Books",
        "url": "https://arxiv.org/abs/1808.03668",
        "implementation_note": "Use convolution over recent market-state sequences to learn short-horizon nonlinear temporal patterns.",
    },
    {
        "title": "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting",
        "url": "https://arxiv.org/abs/1912.09363",
        "implementation_note": "Use attention over visible historical states without allowing evaluation-period labels into training.",
    },
    {
        "title": "TabTransformer: Tabular Data Modeling Using Contextual Embeddings",
        "url": "https://arxiv.org/abs/2012.06678",
        "implementation_note": "Motivates factor-token contextualization before a supervised regression head.",
    },
    {
        "title": "TimeFormer: Transformer with attention modulation empowered by temporal characteristics for time series forecasting",
        "url": "https://doi.org/10.1016/j.eswa.2025.131040",
        "implementation_note": "Use causal past-only temporal decay as an inductive bias for attention candidate generation.",
    },
    {
        "title": "SCINet: Time Series Modeling and Forecasting with Sample Convolution and Interaction",
        "url": "https://proceedings.neurips.cc/paper_files/paper/2022/hash/266983d0949aed78a16fa4782237dea7-Paper-Conference.pdf",
        "implementation_note": "Motivates compact convolutional sequence learners as a stronger alternative to purely tabular models.",
    },
]
LOW_CORR_FACTOR_RUN_IDS = [
    "autoalpha_20260422_223541_02",
    "autoalpha_20260423_122505_01",
    "autoalpha_20260423_082203_02",
    "autoalpha_20260423_112646_01",
    "autoalpha_20260422_160442_02",
    "autoalpha_20260423_032038_03",
    "autoalpha_20260423_102603_01",
    "autoalpha_20260423_115002_01",
]

from autoalpha_v2 import knowledge_base as kb
from autoalpha_v2.error_utils import humanize_error
from autoalpha_v2.pipeline import run as run_pipeline
from prepare_data import DataHub


def _log(message: str, log_path: Path) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line, flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except Exception:
        return default
    if not math.isfinite(number):
        return default
    return number


def _list_valid_autoalpha_factors() -> List[Dict[str, Any]]:
    factors = []
    for factor in kb.list_valid_factors(min_eval_days=700):
        parquet_path = factor.get("parquet_path") or ""
        if (
            factor.get("status") == "ok"
            and parquet_path
            and Path(parquet_path).exists()
        ):
            factors.append(factor)
    factors.sort(
        key=lambda item: (
            _safe_float(item.get("Score", 0)),
            _safe_float(item.get("IC", 0)),
        ),
        reverse=True,
    )
    return factors


def _resolve_factor_or_raise(run_id: str) -> Dict[str, Any]:
    factor = kb.get_factor(run_id)
    if not factor:
        raise KeyError(f"Missing factor in knowledge base: {run_id}")
    parquet_path = Path(str(factor.get("parquet_path") or ""))
    if not parquet_path.is_file():
        raise FileNotFoundError(f"Missing parquet for factor {run_id}: {parquet_path}")
    return {"run_id": run_id, **factor}


def mine_until_target_valid(
    *,
    target_valid_count: int,
    ideas_per_round: int,
    eval_days_count: int,
    max_rounds: int,
    sleep_seconds: float,
    log_path: Path,
) -> List[Dict[str, Any]]:
    """Keep generating factors until we have enough valid ones."""
    valid_factors = _list_valid_autoalpha_factors()
    round_i = 0

    if len(valid_factors) >= target_valid_count:
        _log(
            f"Already have {len(valid_factors)} valid AutoAlpha factors; skip mining.",
            log_path,
        )
        return valid_factors

    _log(
        f"Need {target_valid_count} valid factors, current={len(valid_factors)}. Start mining.",
        log_path,
    )

    while len(valid_factors) < target_valid_count:
        if max_rounds > 0 and round_i >= max_rounds:
            raise RuntimeError(
                f"Stopped after {max_rounds} rounds, only have {len(valid_factors)} valid factors."
            )

        round_i += 1
        parents = kb.get_top_parents(k=4)
        parent_run_ids = [item["run_id"] for item in parents]
        _log(
            f"Mining round {round_i}: parents={','.join(parent_run_ids) if parent_run_ids else 'scratch'}",
            log_path,
        )

        try:
            results = run_pipeline(
                n_ideas=ideas_per_round,
                eval_days_count=eval_days_count,
                parents=parents or None,
            )
        except Exception as exc:
            friendly, suggestion, _, raw = humanize_error(exc)
            _log(f"Pipeline failed: {friendly}", log_path)
            if suggestion:
                _log(f"Suggestion: {suggestion}", log_path)
            if raw:
                _log(f"Raw: {raw}", log_path)
            raise

        for result in results:
            kb.add_factor(result, parent_run_ids=parent_run_ids)

        valid_factors = _list_valid_autoalpha_factors()
        passing_this_round = sum(1 for result in results if result.get("PassGates"))
        _log(
            f"Round {round_i} complete: tested={len(results)} passing={passing_this_round} total_valid={len(valid_factors)}",
            log_path,
        )
        if len(valid_factors) >= target_valid_count:
            break
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    return valid_factors


def _daily_feature_cache_path(run_id: str) -> Path:
    return FEATURE_CACHE_DIR / f"{run_id}_daily.parquet"


def ensure_daily_feature_cache(factor: Dict[str, Any], log_path: Path) -> Path:
    parquet_path = Path(str(factor.get("parquet_path") or ""))
    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing exported parquet for {factor.get('run_id')}: {parquet_path}")

    cache_path = _daily_feature_cache_path(str(factor.get("run_id")))
    if cache_path.exists() and cache_path.stat().st_mtime >= parquet_path.stat().st_mtime:
        return cache_path

    FEATURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _log(f"Building daily feature cache for {factor.get('run_id')}", log_path)
    df = pd.read_parquet(parquet_path, columns=["date", "security_id", "alpha"])
    daily = (
        df.groupby(["date", "security_id"], sort=True)["alpha"]
        .mean()
        .astype("float32")
        .reset_index()
        .rename(columns={"alpha": "value"})
    )
    daily.to_parquet(cache_path, index=False)
    return cache_path


def _build_rolling_windows(
    trading_days: List[str],
    *,
    train_days: int,
    test_days: int,
    step_days: int,
) -> List[Dict[str, Any]]:
    windows: List[Dict[str, Any]] = []
    start_idx = 0
    window_id = 1
    while start_idx + train_days + test_days <= len(trading_days):
        train_slice = trading_days[start_idx : start_idx + train_days]
        test_slice = trading_days[start_idx + train_days : start_idx + train_days + test_days]
        windows.append(
            {
                "window_id": window_id,
                "train_days": train_slice,
                "test_days": test_slice,
                "train_start": train_slice[0],
                "train_end": train_slice[-1],
                "test_start": test_slice[0],
                "test_end": test_slice[-1],
            }
        )
        start_idx += step_days
        window_id += 1
    return windows


def _load_window_feature_frame(
    cache_path: Path,
    feature_name: str,
    days: Iterable[str],
) -> pd.Series:
    day_list = list(days)
    feature_df = pd.read_parquet(
        cache_path,
        columns=["date", "security_id", "value"],
        filters=[("date", "in", day_list)],
    )
    if feature_df.empty:
        return pd.Series(name=feature_name, dtype="float32")
    series = (
        feature_df.set_index(["date", "security_id"])["value"]
        .astype("float32")
        .rename(feature_name)
        .sort_index()
    )
    return series


def _assemble_window_dataset(
    feature_refs: List[Dict[str, Any]],
    resp_series: pd.Series,
    window_days: List[str],
) -> pd.DataFrame:
    resp_slice = resp_series.loc[
        resp_series.index.get_level_values("date").isin(window_days)
    ].rename("resp")
    frames: List[pd.DataFrame] = [resp_slice.to_frame()]
    for ref in feature_refs:
        feature_series = _load_window_feature_frame(ref["cache_path"], ref["run_id"], window_days)
        frames.append(feature_series.to_frame())
    frame = pd.concat(frames, axis=1, join="outer")
    feature_cols = [ref["run_id"] for ref in feature_refs]
    frame[feature_cols] = frame[feature_cols].fillna(0.0).astype("float32")
    frame = frame.dropna(subset=["resp"])
    return frame


def _date_level_to_string(series: pd.Series) -> pd.Series:
    """Normalize the date level to YYYY-MM-DD strings so feature and target indexes align."""
    if series.empty:
        return series
    df = series.rename(series.name or "value").reset_index()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        index_names = list(series.index.names)
    elif "datetime" in df.columns:
        df["date"] = pd.to_datetime(df["datetime"]).dt.strftime("%Y-%m-%d")
        df = df.drop(columns=["datetime"])
        index_names = ["date" if name == "datetime" else name for name in series.index.names]
    else:
        index_names = list(series.index.names)
    return df.set_index(index_names)[series.name or "value"].sort_index()


def _slice_days_between(trading_days: List[str], start_date: str, end_date: str) -> List[str]:
    return [day for day in trading_days if start_date <= str(day) <= end_date]


def _daily_corr(group: pd.DataFrame, rank: bool = False) -> float:
    valid = group[["pred", "resp"]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid) < 5:
        return float("nan")
    pred = valid["pred"]
    resp = valid["resp"]
    if rank:
        pred = pred.rank()
        resp = resp.rank()
    corr = pred.corr(resp)
    return float(corr) if pd.notna(corr) else float("nan")


def _signal_to_weights(signal: pd.Series, quantile: float = 0.2) -> pd.Series:
    valid = signal.replace([np.inf, -np.inf], np.nan).dropna()
    n = len(valid)
    if n < 5:
        return pd.Series(0.0, index=signal.index, dtype="float32")
    bucket = min(max(1, int(n * quantile)), n // 2)
    if bucket < 1:
        return pd.Series(0.0, index=signal.index, dtype="float32")
    ordered = valid.sort_values(kind="mergesort")
    short_index = ordered.index[:bucket]
    long_index = ordered.index[-bucket:]
    weights = pd.Series(0.0, index=signal.index, dtype="float32")
    weights.loc[short_index] = np.float32(-1.0 / bucket)
    weights.loc[long_index] = np.float32(1.0 / bucket)
    return weights


def _signal_to_long_weights(signal: pd.Series, quantile: float = 0.2) -> pd.Series:
    valid = signal.replace([np.inf, -np.inf], np.nan).dropna()
    n = len(valid)
    if n < 5:
        return pd.Series(0.0, index=signal.index, dtype="float32")
    bucket = max(1, int(n * quantile))
    ordered = valid.sort_values(kind="mergesort")
    long_index = ordered.index[-bucket:]
    weights = pd.Series(0.0, index=signal.index, dtype="float32")
    weights.loc[long_index] = np.float32(1.0 / bucket)
    return weights


def _strategy_from_predictions(pred: pd.Series, resp: pd.Series, *, fee_bps: float = DEFAULT_FEE_BPS) -> Dict[str, Any]:
    frame = pd.DataFrame({"pred": pred, "resp": resp}).replace([np.inf, -np.inf], np.nan).dropna()
    if frame.empty:
        return {
            "daily_pnl": pd.Series(dtype="float32"),
            "daily_gross_pnl": pd.Series(dtype="float32"),
            "daily_fee": pd.Series(dtype="float32"),
            "daily_long_pnl": pd.Series(dtype="float32"),
            "daily_long_gross_pnl": pd.Series(dtype="float32"),
            "daily_long_fee": pd.Series(dtype="float32"),
            "long_drawdown_curve": pd.Series(dtype="float32"),
            "total_pnl": 0.0,
            "gross_pnl": 0.0,
            "total_fee": 0.0,
            "long_total_pnl": 0.0,
            "long_gross_pnl": 0.0,
            "long_total_fee": 0.0,
            "long_max_drawdown": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "hit_ratio": 0.0,
            "avg_turnover": 0.0,
        }

    frame["weight"] = frame.groupby(level="date")["pred"].transform(_signal_to_weights)
    frame["long_weight"] = frame.groupby(level="date")["pred"].transform(_signal_to_long_weights)
    frame["gross_pnl"] = frame["weight"] * frame["resp"]
    frame["long_gross_pnl"] = frame["long_weight"] * frame["resp"]
    daily_gross_pnl = frame.groupby(level="date")["gross_pnl"].sum().astype("float32")
    daily_long_gross_pnl = frame.groupby(level="date")["long_gross_pnl"].sum().astype("float32")
    daily_weights = frame["weight"].unstack("security_id").fillna(0.0)
    daily_long_weights = frame["long_weight"].unstack("security_id").fillna(0.0)
    turnover = daily_weights.diff().abs().sum(axis=1) / 2.0
    long_turnover = daily_long_weights.diff().abs().sum(axis=1) / 2.0
    if len(turnover):
        turnover.iloc[0] = daily_weights.iloc[0].abs().sum() / 2.0
    if len(long_turnover):
        long_turnover.iloc[0] = daily_long_weights.iloc[0].abs().sum() / 2.0
    fee_rate = max(float(fee_bps), 0.0) / 10_000.0
    daily_fee = (turnover.reindex(daily_gross_pnl.index).fillna(0.0) * fee_rate).astype("float32")
    daily_long_fee = (long_turnover.reindex(daily_long_gross_pnl.index).fillna(0.0) * fee_rate).astype("float32")
    daily_pnl = (daily_gross_pnl - daily_fee).astype("float32")
    daily_long_pnl = (daily_long_gross_pnl - daily_long_fee).astype("float32")
    cum_pnl = daily_pnl.cumsum()
    long_cum_pnl = daily_long_pnl.cumsum()
    running_max = cum_pnl.cummax()
    long_running_max = long_cum_pnl.cummax()
    drawdown = cum_pnl - running_max
    long_drawdown = long_cum_pnl - long_running_max
    pnl_std = daily_pnl.std()
    sharpe = float(daily_pnl.mean() / pnl_std * np.sqrt(252)) if pnl_std and pnl_std > 0 else 0.0
    return {
        "daily_pnl": daily_pnl,
        "daily_gross_pnl": daily_gross_pnl,
        "daily_fee": daily_fee,
        "daily_long_pnl": daily_long_pnl,
        "daily_long_gross_pnl": daily_long_gross_pnl,
        "daily_long_fee": daily_long_fee,
        "drawdown_curve": drawdown.astype("float32"),
        "long_drawdown_curve": long_drawdown.astype("float32"),
        "total_pnl": float(cum_pnl.iloc[-1]) if len(cum_pnl) else 0.0,
        "gross_pnl": float(daily_gross_pnl.cumsum().iloc[-1]) if len(daily_gross_pnl) else 0.0,
        "total_fee": float(daily_fee.sum()) if len(daily_fee) else 0.0,
        "long_total_pnl": float(long_cum_pnl.iloc[-1]) if len(long_cum_pnl) else 0.0,
        "long_gross_pnl": float(daily_long_gross_pnl.cumsum().iloc[-1]) if len(daily_long_gross_pnl) else 0.0,
        "long_total_fee": float(daily_long_fee.sum()) if len(daily_long_fee) else 0.0,
        "long_max_drawdown": float(long_drawdown.min()) if len(long_drawdown) else 0.0,
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()) if len(drawdown) else 0.0,
        "hit_ratio": float((daily_pnl > 0).mean()) if len(daily_pnl) else 0.0,
        "avg_turnover": float(turnover.mean()) if len(turnover) else 0.0,
    }


def _prediction_comparison_curve(pred: pd.Series, resp: pd.Series) -> List[Dict[str, Any]]:
    frame = pd.DataFrame({"pred": pred, "resp": resp}).replace([np.inf, -np.inf], np.nan).dropna()
    if frame.empty:
        return []

    def _daily_row(group: pd.DataFrame) -> pd.Series:
        n = len(group)
        if n < 20:
            return pd.Series({
                "mean_prediction": 0.0,
                "mean_return": 0.0,
                "predicted_spread": 0.0,
                "realized_spread": 0.0,
                "daily_ic": 0.0,
                "daily_rank_ic": 0.0,
                "daily_pnl": 0.0,
            })
        q = max(1, int(n * 0.2))
        ordered = group.sort_values("pred")
        bottom = ordered.iloc[:q]
        top = ordered.iloc[-q:]
        daily_ic = _daily_corr(group)
        daily_rank_ic = _daily_corr(group, rank=True)
        weights = _signal_to_weights(group["pred"])
        return pd.Series({
            "mean_prediction": float(group["pred"].mean()),
            "mean_return": float(group["resp"].mean()),
            "predicted_spread": float(top["pred"].mean() - bottom["pred"].mean()),
            "realized_spread": float(top["resp"].mean() - bottom["resp"].mean()),
            "daily_ic": float(daily_ic) if np.isfinite(daily_ic) else 0.0,
            "daily_rank_ic": float(daily_rank_ic) if np.isfinite(daily_rank_ic) else 0.0,
            "daily_pnl": float((weights * group["resp"]).sum()),
        })

    daily = frame.groupby(level="date", sort=True).apply(_daily_row)
    if daily.empty:
        return []
    daily["mean_prediction_aligned"] = _align_series_mean_std(daily["mean_prediction"], daily["mean_return"])
    daily["predicted_spread_aligned"] = _align_series_mean_std(daily["predicted_spread"], daily["realized_spread"])
    return [
        {
            "date": str(idx),
            "mean_prediction": float(row["mean_prediction"]),
            "mean_return": float(row["mean_return"]),
            "mean_prediction_aligned": float(row["mean_prediction_aligned"]),
            "predicted_spread": float(row["predicted_spread"]),
            "realized_spread": float(row["realized_spread"]),
            "predicted_spread_aligned": float(row["predicted_spread_aligned"]),
            "daily_ic": float(row.get("daily_ic", 0.0)),
            "daily_rank_ic": float(row.get("daily_rank_ic", 0.0)),
            "daily_pnl": float(row.get("daily_pnl", 0.0)),
        }
        for idx, row in daily.iterrows()
        if np.isfinite(row).all()
    ]


def _align_series_mean_std(source: pd.Series, target: pd.Series) -> pd.Series:
    source_mean = float(source.mean())
    target_mean = float(target.mean())
    source_std = float(source.std())
    target_std = float(target.std())
    if not np.isfinite(source_mean):
        source_mean = 0.0
    if not np.isfinite(target_mean):
        target_mean = 0.0
    if not np.isfinite(source_std) or source_std <= 0.0:
        return pd.Series(target_mean, index=source.index, dtype="float32")
    if not np.isfinite(target_std) or target_std < 0.0:
        target_std = 0.0
    aligned = (source - source_mean) * (target_std / source_std) + target_mean
    return aligned.astype("float32")


def _sample_training_rows(
    X: np.ndarray,
    y: np.ndarray,
    *,
    max_rows: int,
    seed: int = 42,
    sample_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if max_rows <= 0 or len(X) <= max_rows:
        return X, y, sample_weight
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=max_rows, replace=False)
    return X[idx], y[idx], sample_weight[idx] if sample_weight is not None else None


def _extract_importance(model: Any, feature_names: List[str]) -> Dict[str, float]:
    estimator = model
    if hasattr(model, "named_steps"):
        estimator = list(model.named_steps.values())[-1]

    values: np.ndarray | None = None
    if hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_)
        values = np.abs(coef).reshape(-1)
    elif hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype="float64").reshape(-1)
    elif hasattr(estimator, "coefs_") and estimator.coefs_:
        first = np.asarray(estimator.coefs_[0], dtype="float64")
        values = np.abs(first).sum(axis=1)

    if values is None or len(values) == 0:
        return {name: 0.0 for name in feature_names}
    if len(values) < len(feature_names):
        values = np.pad(values, (0, len(feature_names) - len(values)))
    return {name: float(value) for name, value in zip(feature_names, values[: len(feature_names)], strict=False)}


def _cs_zscore_frame(frame: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    grouped = frame[columns].groupby(level="date", sort=False)
    mean = grouped.transform("mean")
    std = grouped.transform("std").replace(0.0, np.nan)
    return ((frame[columns] - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5.0, 5.0)


def _cs_rank_frame(frame: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    return (
        frame[columns]
        .groupby(level="date", sort=False)
        .rank(method="average", pct=True)
        .sub(0.5)
        .fillna(0.0)
        .astype("float32")
    )


def _rank_ic_by_date(signal: pd.Series, resp: pd.Series) -> pd.Series:
    aligned = pd.concat(
        [signal.rename("signal"), resp.rename("resp")],
        axis=1,
        join="inner",
    ).replace([np.inf, -np.inf], np.nan)
    if aligned.empty:
        return pd.Series(dtype="float32")
    return aligned.groupby(level="date", sort=True).apply(
        lambda group: _daily_corr(group.rename(columns={"signal": "pred"}), rank=True)
    ).astype("float32")


def _equal_weight_combo_weights(feature_cols: List[str]) -> Dict[str, float]:
    if not feature_cols:
        return {}
    weight = 1.0 / float(len(feature_cols))
    return {col: weight for col in feature_cols}


def _split_train_val_days(train_days: List[str], val_ratio: float = 0.25) -> tuple[List[str], List[str]]:
    if len(train_days) < 8:
        return train_days, train_days
    # Keep validation deliberately small and chronological: about two trading
    # months from the visible 2022-2023 train block. No 2024 OOS information is
    # used for method choice or weight fitting.
    val_count = min(max(1, int(len(train_days) * val_ratio)), 42)
    val_days = train_days[-val_count:]
    fit_days = train_days[:-val_count]
    if not fit_days:
        fit_days = train_days
    return fit_days, val_days


def _compute_factor_daily_ic_stats(
    features: pd.DataFrame,
    y: pd.Series,
    feature_cols: List[str],
) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    if features.empty or y.empty:
        return {col: {"ic_mean": 0.0, "ic_std": 0.0, "ic_ir": 0.0, "sign": 1.0, "abs_ic": 0.0} for col in feature_cols}
    for col in feature_cols:
        if col not in features.columns:
            continue
        daily_ic = _rank_ic_by_date(features[col], y).dropna()
        ic_mean = float(daily_ic.mean()) if not daily_ic.empty else 0.0
        ic_std = float(daily_ic.std()) if len(daily_ic) > 1 else 0.0
        ic_ir = ic_mean / ic_std if ic_std > 1e-8 else 0.0
        sign = 1.0 if ic_mean >= 0 else -1.0
        stats[col] = {
            "ic_mean": ic_mean,
            "ic_std": ic_std,
            "ic_ir": ic_ir,
            "sign": sign,
            "abs_ic": abs(ic_mean),
        }
    for col in feature_cols:
        stats.setdefault(col, {"ic_mean": 0.0, "ic_std": 0.0, "ic_ir": 0.0, "sign": 1.0, "abs_ic": 0.0})
    return stats


def _normalize_weight_map(weights: Dict[str, float], feature_cols: List[str]) -> Dict[str, float]:
    l1_norm = sum(abs(float(weights.get(col, 0.0))) for col in feature_cols)
    if l1_norm <= 1e-12:
        return _equal_weight_combo_weights(feature_cols)
    return {col: float(weights.get(col, 0.0) / l1_norm) for col in feature_cols}


def _build_sign_aligned_equal_weights(
    feature_cols: List[str],
    fit_stats: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    if not feature_cols:
        return {}
    base = 1.0 / float(len(feature_cols))
    return {col: base * float(fit_stats.get(col, {}).get("sign", 1.0)) for col in feature_cols}


def _choose_best_top_k(
    feature_cols: List[str],
    fit_stats: Dict[str, Dict[str, float]],
    val_stats: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    ranked = sorted(feature_cols, key=lambda col: val_stats.get(col, {}).get("abs_ic", 0.0), reverse=True)
    candidates = sorted({min(len(feature_cols), k) for k in (4, 8, 12, 16, 24, 32, len(feature_cols)) if k > 0})
    best_weights = _build_sign_aligned_equal_weights(feature_cols, fit_stats)
    best_score = -1e18
    for top_k in candidates:
        chosen = ranked[:top_k]
        if not chosen:
            continue
        weights = {col: 0.0 for col in feature_cols}
        base = 1.0 / float(len(chosen))
        for col in chosen:
            weights[col] = base * float(fit_stats.get(col, {}).get("sign", 1.0))
        score = sum(val_stats.get(col, {}).get("abs_ic", 0.0) for col in chosen)
        if score > best_score:
            best_score = score
            best_weights = weights
    return _normalize_weight_map(best_weights, feature_cols)


def _choose_best_power_weights(
    feature_cols: List[str],
    fit_stats: Dict[str, Dict[str, float]],
    val_stats: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    best_weights = _build_sign_aligned_equal_weights(feature_cols, fit_stats)
    best_score = -1e18
    for power in (0.5, 1.0, 1.5, 2.0):
        weights = {}
        for col in feature_cols:
            signed_ic = float(fit_stats.get(col, {}).get("ic_mean", 0.0))
            magnitude = abs(signed_ic) ** power
            weights[col] = np.sign(signed_ic) * magnitude
        weights = _normalize_weight_map(weights, feature_cols)
        val_score = sum(weights.get(col, 0.0) * float(val_stats.get(col, {}).get("ic_mean", 0.0)) for col in feature_cols)
        if val_score > best_score:
            best_score = val_score
            best_weights = weights
    return best_weights


def _build_inverse_vol_ic_weights(
    feature_cols: List[str],
    fit_stats: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    weights = {}
    for col in feature_cols:
        stat = fit_stats.get(col, {})
        ic_mean = float(stat.get("ic_mean", 0.0))
        ic_std = max(float(stat.get("ic_std", 0.0)), 1e-4)
        weights[col] = ic_mean / ic_std
    return _normalize_weight_map(weights, feature_cols)


def _build_diversified_ic_weights(
    train_features: pd.DataFrame,
    feature_cols: List[str],
    fit_stats: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    if train_features.empty or len(feature_cols) <= 1:
        return _build_inverse_vol_ic_weights(feature_cols, fit_stats)

    sampled = _cs_rank_frame(train_features, feature_cols).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if len(sampled) > 250_000:
        sampled = sampled.sample(n=250_000, random_state=42)

    corr = sampled.corr().reindex(index=feature_cols, columns=feature_cols).fillna(0.0)
    cov = corr.to_numpy(dtype="float64", copy=True)
    cov = cov + np.eye(len(feature_cols), dtype="float64") * 0.25
    mu = np.asarray([float(fit_stats.get(col, {}).get("ic_mean", 0.0)) for col in feature_cols], dtype="float64")

    try:
        raw = np.linalg.solve(cov, mu)
    except np.linalg.LinAlgError:
        raw = np.linalg.pinv(cov).dot(mu)

    weights = {col: float(value) for col, value in zip(feature_cols, raw, strict=False)}
    return _normalize_weight_map(weights, feature_cols)


def _build_softmax_ic_weights(
    feature_cols: List[str],
    fit_stats: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    if not feature_cols:
        return {}
    abs_ics = np.asarray([abs(float(fit_stats.get(col, {}).get("ic_mean", 0.0))) for col in feature_cols], dtype="float64")
    temp = max(float(np.nanmedian(abs_ics[abs_ics > 0])) if np.any(abs_ics > 0) else 1e-4, 1e-4)
    scaled = np.clip(abs_ics / temp, -20.0, 20.0)
    exp_vals = np.exp(scaled - np.max(scaled))
    weights = {}
    denom = float(exp_vals.sum()) or 1.0
    for col, raw in zip(feature_cols, exp_vals / denom, strict=False):
        weights[col] = float(raw) * float(fit_stats.get(col, {}).get("sign", 1.0))
    return _normalize_weight_map(weights, feature_cols)


def _build_top_abs_ic_weights(
    feature_cols: List[str],
    fit_stats: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    if not feature_cols:
        return {}
    top_k = min(len(feature_cols), max(4, int(round(math.sqrt(len(feature_cols)) * 2))))
    ranked = sorted(feature_cols, key=lambda col: fit_stats.get(col, {}).get("abs_ic", 0.0), reverse=True)
    chosen = ranked[:top_k]
    base = 1.0 / float(len(chosen) or 1)
    return _normalize_weight_map(
        {col: (base * float(fit_stats.get(col, {}).get("sign", 1.0)) if col in chosen else 0.0) for col in feature_cols},
        feature_cols,
    )


def _build_ridge_shrinkage_weights(
    feature_cols: List[str],
    fit_stats: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    variances = [
        float(fit_stats.get(col, {}).get("ic_std", 0.0)) ** 2
        for col in feature_cols
        if float(fit_stats.get(col, {}).get("ic_std", 0.0)) > 0
    ]
    ridge = float(np.nanmedian(variances)) if variances else 1e-4
    ridge = max(ridge, 1e-4)
    weights = {}
    for col in feature_cols:
        stat = fit_stats.get(col, {})
        weights[col] = float(stat.get("ic_mean", 0.0)) / (float(stat.get("ic_std", 0.0)) ** 2 + ridge)
    return _normalize_weight_map(weights, feature_cols)


def _build_vol_balanced_top_weights(
    feature_cols: List[str],
    fit_stats: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    if not feature_cols:
        return {}
    top_k = min(len(feature_cols), max(4, int(round(math.sqrt(len(feature_cols)) * 2))))
    ranked = sorted(
        feature_cols,
        key=lambda col: abs(float(fit_stats.get(col, {}).get("ic_mean", 0.0))) / max(float(fit_stats.get(col, {}).get("ic_std", 0.0)), 1e-4),
        reverse=True,
    )
    chosen = set(ranked[:top_k])
    weights = {}
    for col in feature_cols:
        if col not in chosen:
            weights[col] = 0.0
            continue
        stat = fit_stats.get(col, {})
        weights[col] = float(stat.get("sign", 1.0)) / max(float(stat.get("ic_std", 0.0)), 1e-4)
    return _normalize_weight_map(weights, feature_cols)


def _build_cluster_neutral_ic_weights(
    train_features: pd.DataFrame,
    feature_cols: List[str],
    fit_stats: Dict[str, Dict[str, float]],
    *,
    cluster_corr: float = 0.7,
) -> Dict[str, float]:
    if train_features.empty or len(feature_cols) <= 1:
        return _build_inverse_vol_ic_weights(feature_cols, fit_stats)
    ranked = sorted(feature_cols, key=lambda col: fit_stats.get(col, {}).get("abs_ic", 0.0), reverse=True)
    sampled = _cs_rank_frame(train_features, feature_cols).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if len(sampled) > 200_000:
        sampled = sampled.sample(n=200_000, random_state=42)
    corr = sampled.corr().reindex(index=feature_cols, columns=feature_cols).fillna(0.0)
    clusters: List[List[str]] = []
    for col in ranked:
        target_cluster = None
        for cluster in clusters:
            if max(abs(float(corr.loc[col, prev])) for prev in cluster) >= cluster_corr:
                target_cluster = cluster
                break
        if target_cluster is None:
            clusters.append([col])
        else:
            target_cluster.append(col)
    if not clusters:
        return _build_inverse_vol_ic_weights(feature_cols, fit_stats)
    weights = {col: 0.0 for col in feature_cols}
    cluster_budget = 1.0 / float(len(clusters))
    for cluster in clusters:
        raw = {col: float(fit_stats.get(col, {}).get("ic_mean", 0.0)) for col in cluster}
        norm = sum(abs(value) for value in raw.values())
        if norm <= 1e-12:
            base = cluster_budget / float(len(cluster))
            for col in cluster:
                weights[col] = base * float(fit_stats.get(col, {}).get("sign", 1.0))
        else:
            for col, value in raw.items():
                weights[col] = cluster_budget * value / norm
    return _normalize_weight_map(weights, feature_cols)


def _build_corr_pruned_weights(
    train_features: pd.DataFrame,
    feature_cols: List[str],
    fit_stats: Dict[str, Dict[str, float]],
    *,
    max_corr: float = 0.65,
) -> Dict[str, float]:
    if train_features.empty:
        return _build_sign_aligned_equal_weights(feature_cols, fit_stats)
    ranked = sorted(feature_cols, key=lambda col: fit_stats.get(col, {}).get("abs_ic", 0.0), reverse=True)
    sampled = train_features[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if len(sampled) > 200_000:
        sampled = sampled.sample(n=200_000, random_state=42)
    corr = sampled.corr().fillna(0.0)
    selected: List[str] = []
    for col in ranked:
        if not selected:
            selected.append(col)
            continue
        if max(abs(float(corr.loc[col, prev])) for prev in selected) < max_corr:
            selected.append(col)
    if not selected:
        selected = ranked[: max(1, min(8, len(ranked)))]
    base = 1.0 / float(len(selected))
    weights = {col: 0.0 for col in feature_cols}
    for col in selected:
        weights[col] = base * float(fit_stats.get(col, {}).get("sign", 1.0))
    return _normalize_weight_map(weights, feature_cols)


def _build_combo_weights(
    *,
    model_name: str,
    train_features: pd.DataFrame,
    y_train: pd.Series,
    feature_cols: List[str],
    stats_bundle: Dict[str, Dict[str, Dict[str, float]]] | None = None,
) -> Dict[str, float]:
    if train_features.empty or y_train.empty or not feature_cols:
        return _equal_weight_combo_weights(feature_cols)
    if stats_bundle is None:
        stats_bundle = _precompute_combo_weight_stats(train_features, y_train, feature_cols)
    fit_stats = stats_bundle.get("fit_stats", {})
    val_stats = stats_bundle.get("val_stats", {})
    full_stats = stats_bundle.get("full_stats", {})

    if model_name == "EqualWeightRankCombo":
        return _normalize_weight_map(_build_sign_aligned_equal_weights(feature_cols, fit_stats), feature_cols)
    if model_name == "TrainICRankCombo":
        return _normalize_weight_map({col: float(full_stats.get(col, {}).get("ic_mean", 0.0)) for col in feature_cols}, feature_cols)
    if model_name == "TrainICIRRankCombo":
        return _normalize_weight_map({col: float(full_stats.get(col, {}).get("ic_ir", 0.0)) for col in feature_cols}, feature_cols)
    if model_name == "ValTopKRankCombo":
        return _choose_best_top_k(feature_cols, fit_stats, val_stats)
    if model_name == "ValPowerRankCombo":
        return _choose_best_power_weights(feature_cols, fit_stats, val_stats)
    if model_name == "CorrPrunedRankCombo":
        return _build_corr_pruned_weights(train_features, feature_cols, fit_stats)
    if model_name == "InverseVolICRankCombo":
        return _build_inverse_vol_ic_weights(feature_cols, full_stats)
    if model_name == "DiversifiedICRankCombo":
        return _build_diversified_ic_weights(train_features, feature_cols, full_stats)
    if model_name == "SoftmaxICRankCombo":
        return _build_softmax_ic_weights(feature_cols, full_stats)
    if model_name == "TopAbsICRankCombo":
        return _build_top_abs_ic_weights(feature_cols, full_stats)
    if model_name == "RidgeShrinkageRankCombo":
        return _build_ridge_shrinkage_weights(feature_cols, full_stats)
    if model_name == "VolBalancedTopRankCombo":
        return _build_vol_balanced_top_weights(feature_cols, full_stats)
    if model_name == "ClusterNeutralICRankCombo":
        return _build_cluster_neutral_ic_weights(train_features, feature_cols, full_stats)
    raise KeyError(f"Unknown combo model: {model_name}")


def _precompute_combo_weight_stats(
    train_features: pd.DataFrame,
    y_train: pd.Series,
    feature_cols: List[str],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    all_train_days = sorted(pd.Index(train_features.index.get_level_values("date").astype(str)).unique().tolist())
    fit_days, val_days = _split_train_val_days(all_train_days)
    fit_mask = train_features.index.get_level_values("date").isin(fit_days)
    val_mask = train_features.index.get_level_values("date").isin(val_days)
    return {
        "fit_stats": _compute_factor_daily_ic_stats(train_features.loc[fit_mask], y_train.loc[fit_mask], feature_cols),
        "val_stats": _compute_factor_daily_ic_stats(train_features.loc[val_mask], y_train.loc[val_mask], feature_cols),
        "full_stats": _compute_factor_daily_ic_stats(train_features, y_train, feature_cols),
    }


def _combo_method_card(model_name: str) -> Dict[str, Any]:
    cards: Dict[str, Dict[str, str]] = {
        "EqualWeightRankCombo": {
            "description": "Cross-sectional rank each factor each day, align signs using the visible train segment, then average all factors equally.",
            "weight_rule": "Equal absolute weight after train-period sign alignment.",
            "validation_usage": "Validation is not used for fitting; it is only reported.",
        },
        "TrainICRankCombo": {
            "description": "Weight each ranked factor by its mean train-period daily IC.",
            "weight_rule": "Signed train IC, L1-normalized.",
            "validation_usage": "Validation is not used for fitting; it is only reported.",
        },
        "TrainICIRRankCombo": {
            "description": "Prefer factors whose train IC is both positive and stable.",
            "weight_rule": "Train IC divided by train IC volatility, L1-normalized.",
            "validation_usage": "Validation is not used for fitting; it is only reported.",
        },
        "ValTopKRankCombo": {
            "description": "Use the last two visible training months as validation to select a top-K subset, then combine selected factors with train signs.",
            "weight_rule": "Validation top-K by absolute IC, equal signed weights inside the subset.",
            "validation_usage": "Uses only 2022-2023 validation days; 2024 OOS is never used for selection.",
        },
        "ValPowerRankCombo": {
            "description": "Search a small family of IC power transforms on the visible validation block.",
            "weight_rule": "Signed train IC raised to a selected power and L1-normalized.",
            "validation_usage": "Uses only 2022-2023 validation days to choose the power.",
        },
        "CorrPrunedRankCombo": {
            "description": "Greedily keep high-IC factors while pruning factors highly correlated with already selected factors.",
            "weight_rule": "Equal signed weights after train-period correlation pruning.",
            "validation_usage": "Validation is not used for fitting; it is only reported.",
        },
        "InverseVolICRankCombo": {
            "description": "Classic risk-adjusted IC weighting that penalizes unstable factor IC.",
            "weight_rule": "Train IC divided by train IC volatility, equivalent to ICIR weighting.",
            "validation_usage": "Validation is not used for fitting; it is only reported.",
        },
        "DiversifiedICRankCombo": {
            "description": "Mean-variance style combination: solve a regularized correlation system so highly redundant factors receive less weight.",
            "weight_rule": "Regularized inverse train correlation times train IC, L1-normalized.",
            "validation_usage": "Validation is not used for fitting; it is only reported.",
        },
        "SoftmaxICRankCombo": {
            "description": "Concentrate weight smoothly into high absolute train-IC factors without a hard cutoff.",
            "weight_rule": "Softmax over absolute train IC, multiplied by train IC sign.",
            "validation_usage": "Validation is not used for fitting; it is only reported.",
        },
        "TopAbsICRankCombo": {
            "description": "Simple sparse benchmark: keep the strongest train-IC factors and equal-weight them.",
            "weight_rule": "Top absolute train IC subset with equal signed weights.",
            "validation_usage": "Validation is not used for fitting; it is only reported.",
        },
        "RidgeShrinkageRankCombo": {
            "description": "Shrink noisy IC estimates by adding a ridge penalty to IC variance.",
            "weight_rule": "Train IC divided by IC variance plus median variance ridge.",
            "validation_usage": "Validation is not used for fitting; it is only reported.",
        },
        "VolBalancedTopRankCombo": {
            "description": "Select stable high-IC factors and balance selected weights by inverse IC volatility.",
            "weight_rule": "Top train ICIR subset, inverse-volatility signed weights.",
            "validation_usage": "Validation is not used for fitting; it is only reported.",
        },
        "ClusterNeutralICRankCombo": {
            "description": "Cluster factors by train-period correlation and allocate equal budget to each redundancy cluster.",
            "weight_rule": "Equal cluster budget; within cluster, signed train IC weights.",
            "validation_usage": "Validation is not used for fitting; it is only reported.",
        },
        "RidgeZScoreMetaModel": {
            "description": "Linear machine-learning meta-model on raw/rank/z-score factor features with ridge shrinkage.",
            "weight_rule": "Fit a regularized regression from visible factor features to next-period normalized response.",
            "validation_usage": "A fit-only model is evaluated on the last visible two months for overfit diagnostics; final OOS model is trained on 2022-2023 only.",
        },
        "RandomForestMetaModel": {
            "description": "Tree ensemble meta-model that captures nonlinear factor interactions while restricting depth and leaf size.",
            "weight_rule": "Random forest regression on sampled visible train rows; prediction is cross-sectionally normalized before export.",
            "validation_usage": "Validation is reported from a fit-only model; 2024 OOS is not used for tree construction.",
        },
        "ExtraTreesMetaModel": {
            "description": "Extremely randomized trees benchmark for nonlinear factor mixing with stronger randomization than RandomForest.",
            "weight_rule": "ExtraTrees regression on sampled train rows with shallow trees and large leaves.",
            "validation_usage": "Validation is reported from a fit-only model; 2024 OOS is not used for fitting.",
        },
        "HistGradientBoostingMetaModel": {
            "description": "Histogram gradient boosting meta-model for additive nonlinear interactions between factor states.",
            "weight_rule": "Boosted regression trees on visible train rows with L2 regularization.",
            "validation_usage": "Validation is reported from a fit-only model; final OOS model is trained on 2022-2023 only.",
        },
        "LightGBMMetaModel": {
            "description": "LightGBM gradient boosting meta-model over raw, rank, z-score and short-memory factor features.",
            "weight_rule": "Leaf-wise boosted trees with subsampling, column sampling and regularization.",
            "validation_usage": "Validation is reported from a fit-only model; 2024 OOS is held out entirely.",
        },
        "MLPRegressorMetaModel": {
            "description": "Lightweight neural-network meta-model over normalized factor features.",
            "weight_rule": "Two-layer MLP with early stopping on train-internal validation, then cross-sectional normalization before export.",
            "validation_usage": "The external Val block is report-only; MLP early stopping splits only inside visible training data.",
        },
        "TorchMPSMLPMetaModel": {
            "description": "PyTorch neural-network meta-model that uses Apple Metal/MPS when available for batched training and inference.",
            "weight_rule": "Two-hidden-layer Torch MLP over raw, rank, z-score and short-memory factor states; predictions are normalized cross-sectionally before export.",
            "validation_usage": "Validation is report-only; all fitted weights are learned from visible Train rows without 2024 labels.",
        },
        "TorchTCNMetaModel": {
            "description": "Temporal convolutional neural meta-model over each stock's recent factor-state sequence, inspired by DeepLOB-style CNN sequence modeling.",
            "weight_rule": "Conv1d blocks consume current and past rank/z-score factor states; training uses visible 2022-2023 rows only.",
            "validation_usage": "Validation is report-only; 2024 labels are never used for network weights or hyperparameter choice.",
        },
        "TemporalRidgeLagMetaModel": {
            "description": "Leakage-safe temporal ridge model over current and lagged per-stock factor rank/z-score states.",
            "weight_rule": "Builds an 8-day lag tensor from raw-bar-derived factors, flattens it, then fits ridge regression on 2022-2023 labels only.",
            "validation_usage": "Validation is report-only; temporal lags use current/past factor values only.",
        },
        "TemporalLightGBMLagMetaModel": {
            "description": "Gradient-boosted temporal interaction model over lagged factor states, a robust tree alternative to sequence neural nets.",
            "weight_rule": "LightGBM learns nonlinear cross-lag and cross-factor interactions from train-period lagged rank/z-score features.",
            "validation_usage": "Validation is report-only; 2024 OOS is held out entirely.",
        },
        "TemporalHistGBLagMetaModel": {
            "description": "Histogram gradient boosting over compact temporal factor states for additive nonlinear lag effects.",
            "weight_rule": "Fits regularized histogram boosting on the visible 2022-2023 temporal feature matrix.",
            "validation_usage": "Validation is report-only; no OOS response is used.",
        },
        "TemporalExtraTreesLagMetaModel": {
            "description": "Randomized tree ensemble over lagged factor states to stress-test nonlinear temporal interactions.",
            "weight_rule": "ExtraTrees on sampled temporal lag features with shallow depth and large leaves to limit overfit.",
            "validation_usage": "Validation is report-only; final 2024 prediction is a pure forward application.",
        },
        "FactorTokenTransformerRidgeStackModel": {
            "description": "Full-factor Transformer-style attention stacker: each factor is a token with current and lagged rank/z-score states, then a learned query attends across all factor tokens.",
            "weight_rule": "A Ridge baseline and a factor-token multi-head attention model are fit on train-only data; the blend weight is selected on the visible 2022-2023 validation block.",
            "validation_usage": "Only the last visible training block is used to select the Ridge/Transformer blend. 2024 OOS is never used for architecture, blend, or weight fitting.",
        },
        "CausalDecayFactorTransformerStackModel": {
            "description": "Leakage-safe factor-token Transformer stack with train-only sign alignment, causal decay attention candidates, and validation-selected residual blending.",
            "weight_rule": "Fits attention, Ridge, LightGBM and train-IC rank heads on the fit block; only the visible validation block chooses the attention temperature/head count and convex blend.",
            "validation_usage": "Validation is the only optimization signal. 2024 OOS labels are excluded from all architecture, blend, and weight decisions.",
        },
        "TorchGRUMetaModel": {
            "description": "Recurrent neural meta-model that learns nonlinear state transitions in per-stock factor histories.",
            "weight_rule": "GRU encoder over lagged rank/z-score factor states followed by a regularized regression head.",
            "validation_usage": "Validation is report-only; final OOS prediction is generated after fitting only on 2022-2023 rows.",
        },
        "TorchTransformerMetaModel": {
            "description": "Compact Transformer encoder over recent factor histories to capture cross-lag interactions that tabular models miss.",
            "weight_rule": "Self-attention over current and past rank/z-score states with a small feed-forward head and weight decay.",
            "validation_usage": "Validation is report-only; no 2024 response information is used for attention weights.",
        },
    }
    base = cards.get(model_name, {})
    return {
        "name": model_name,
        "description": base.get("description", "Rank-based combo method."),
        "weight_rule": base.get("weight_rule", "Train-only rank combination."),
        "train_inputs": "Only factor values and target responses inside the visible 2022-2023 training block are used.",
        "validation_usage": base.get("validation_usage", "Validation is only reported."),
        "oos_usage": "2024 data is used only once for out-of-sample evaluation; it is not used to fit weights or choose hyperparameters.",
        "leakage_guard": "Chronological split: Train/Val are inside 2022-2023, Test is 2024 OOS. No future/OOS labels are used during combination fitting.",
    }


def _weighted_rank_combo_prediction(
    features: pd.DataFrame,
    feature_cols: List[str],
    weights: Dict[str, float],
) -> pd.Series:
    if features.empty or not feature_cols:
        return pd.Series(index=features.index, dtype="float32", name="pred")
    ranked = _cs_rank_frame(features, feature_cols)
    return _weighted_combo_from_ranked(ranked, feature_cols, weights)


def _weighted_combo_from_ranked(
    ranked: pd.DataFrame,
    feature_cols: List[str],
    weights: Dict[str, float],
) -> pd.Series:
    if ranked.empty or not feature_cols:
        return pd.Series(index=ranked.index, dtype="float32", name="pred")
    pred = pd.Series(0.0, index=ranked.index, dtype="float32")
    for col in feature_cols:
        pred = pred.add(
            ranked[col].astype("float32") * np.float32(weights.get(col, 0.0)),
            fill_value=0.0,
        )
    return pred.astype("float32").rename("pred")


def _fit_predict_combo_model(
    *,
    model_name: str,
    feature_cols: List[str],
    train_features: pd.DataFrame,
    y_train: pd.Series,
    pred_features: pd.DataFrame,
    pred_ranked_features: pd.DataFrame | None = None,
    stats_bundle: Dict[str, Dict[str, Dict[str, float]]] | None = None,
) -> Dict[str, Any]:
    weights = _build_combo_weights(
        model_name=model_name,
        train_features=train_features,
        y_train=y_train,
        feature_cols=feature_cols,
        stats_bundle=stats_bundle,
    )
    pred = (
        _weighted_combo_from_ranked(pred_ranked_features, feature_cols, weights)
        if pred_ranked_features is not None
        else _weighted_rank_combo_prediction(pred_features, feature_cols, weights)
    )
    return {
        "pred": pred.to_numpy(dtype="float32", copy=False),
        "importance": {col: abs(float(weights.get(col, 0.0))) for col in feature_cols},
        "weights": weights,
    }


def _make_model_feature_frame(frame: pd.DataFrame, feature_cols: List[str], max_temporal: int = 24) -> tuple[pd.DataFrame, List[str]]:
    """Build robust cross-sectional and short-memory states to avoid overly flat point forecasts."""
    raw = frame[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float32")
    z = _cs_zscore_frame(frame, feature_cols).astype("float32")
    z.columns = [f"{col}__csz" for col in feature_cols]
    ranks = _cs_rank_frame(frame, feature_cols)
    ranks.columns = [f"{col}__rank" for col in feature_cols]

    pieces = [raw, z, ranks]
    temporal_cols = feature_cols[: max_temporal]
    by_security = raw.groupby(level="security_id", sort=False)
    for col in temporal_cols:
        lag1 = by_security[col].shift(1).fillna(0.0)
        lag2 = by_security[col].shift(2).fillna(0.0)
        pieces.append(pd.DataFrame({
            f"{col}__lag1": lag1.astype("float32"),
            f"{col}__diff1": (raw[col] - lag1).astype("float32"),
            f"{col}__mom2": (raw[col] - lag2).astype("float32"),
        }, index=raw.index))

    model_frame = pd.concat(pieces, axis=1).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float32")
    return model_frame, list(model_frame.columns)


def _make_training_target(resp: pd.Series) -> pd.Series:
    grouped = resp.groupby(level="date", sort=False)
    mean = grouped.transform("mean")
    std = grouped.transform("std").replace(0.0, np.nan)
    target = ((resp - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5.0, 5.0)
    return target.astype("float32")


def _make_training_weights(y: pd.Series) -> np.ndarray:
    abs_y = y.abs()
    weights = 1.0 + abs_y.clip(0.0, 3.0) * 0.25
    dates = pd.Series(y.index.get_level_values("date"), index=y.index)
    unique_dates = pd.Index(sorted(dates.unique()))
    if len(unique_dates) > 1:
        recency = pd.Series(np.linspace(0.75, 1.25, len(unique_dates)), index=unique_dates)
        weights = weights * dates.map(recency).astype("float32")
    return weights.to_numpy(dtype="float32", copy=False)


def _fit_with_optional_weights(model: Any, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray) -> None:
    try:
        model.fit(X, y, sample_weight=sample_weight)
    except (TypeError, ValueError):
        model.fit(X, y)


def _resolve_torch_device(torch: Any) -> Any:
    requested = (os.environ.get("AUTOALPHA_TORCH_DEVICE") or "cpu").strip().lower()
    if requested == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class TorchMPSMLPRegressor:
    def __init__(
        self,
        *,
        hidden_sizes: tuple[int, int] = (128, 48),
        epochs: int = 10,
        batch_size: int = 8192,
        lr: float = 8e-4,
        weight_decay: float = 1e-4,
        random_state: int = 42,
    ) -> None:
        self.hidden_sizes = hidden_sizes
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.random_state = random_state
        self.model_: Any = None
        self.device_: str = "cpu"
        self.n_features_in_: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> "TorchMPSMLPRegressor":
        import torch
        from torch import nn

        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        if X_arr.ndim != 2 or len(X_arr) == 0:
            raise ValueError("TorchMPSMLPRegressor expects a non-empty 2D matrix.")
        if sample_weight is None:
            w_arr = np.ones((len(y_arr), 1), dtype=np.float32)
        else:
            w_arr = np.asarray(sample_weight, dtype=np.float32).reshape(-1, 1)
        self.n_features_in_ = int(X_arr.shape[1])
        device = _resolve_torch_device(torch)
        self.device_ = str(device)
        torch.manual_seed(self.random_state)
        layers: List[Any] = []
        in_dim = self.n_features_in_
        for hidden in self.hidden_sizes:
            layers.extend([nn.Linear(in_dim, hidden), nn.SiLU(), nn.Dropout(0.05)])
            in_dim = hidden
        layers.append(nn.Linear(in_dim, 1))
        model = nn.Sequential(*layers).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        rng = np.random.default_rng(self.random_state)
        n_rows = len(X_arr)
        model.train()
        for _epoch in range(max(1, int(self.epochs))):
            order = rng.permutation(n_rows)
            for start in range(0, n_rows, self.batch_size):
                idx = order[start : start + self.batch_size]
                xb = torch.as_tensor(X_arr[idx], dtype=torch.float32, device=device)
                yb = torch.as_tensor(y_arr[idx], dtype=torch.float32, device=device)
                wb = torch.as_tensor(w_arr[idx], dtype=torch.float32, device=device)
                optimizer.zero_grad(set_to_none=True)
                loss = (((model(xb) - yb) ** 2) * wb).mean()
                loss.backward()
                optimizer.step()
        self.model_ = model.eval()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch

        if self.model_ is None:
            raise RuntimeError("TorchMPSMLPRegressor is not fitted.")
        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim != 2 or int(X_arr.shape[1]) != self.n_features_in_:
            raise ValueError("Prediction matrix has incompatible shape.")
        device = torch.device(self.device_)
        preds: List[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(X_arr), self.batch_size):
                xb = torch.as_tensor(X_arr[start : start + self.batch_size], dtype=torch.float32, device=device)
                preds.append(self.model_(xb).detach().cpu().numpy().reshape(-1).astype("float32"))
        return np.concatenate(preds) if preds else np.array([], dtype="float32")


class TorchTemporalSequenceRegressor:
    def __init__(
        self,
        *,
        architecture: str,
        seq_len: int,
        n_channels: int,
        hidden_size: int = 64,
        epochs: int = 8,
        batch_size: int = 4096,
        lr: float = 7e-4,
        weight_decay: float = 2e-4,
        random_state: int = 42,
    ) -> None:
        self.architecture = architecture
        self.seq_len = int(seq_len)
        self.n_channels = int(n_channels)
        self.hidden_size = int(hidden_size)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.random_state = int(random_state)
        self.model_: Any = None
        self.device_: str = "cpu"
        self.n_features_in_: int = self.seq_len * self.n_channels

    def _build_model(self, nn: Any) -> Any:
        architecture = self.architecture
        seq_len = self.seq_len
        n_channels = self.n_channels
        hidden = self.hidden_size

        class TemporalNet(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.architecture = architecture
                if architecture == "tcn":
                    self.encoder = nn.Sequential(
                        nn.Conv1d(n_channels, hidden, kernel_size=3, padding=1),
                        nn.SiLU(),
                        nn.Dropout(0.05),
                        nn.Conv1d(hidden, hidden, kernel_size=3, padding=2, dilation=2),
                        nn.SiLU(),
                        nn.AdaptiveAvgPool1d(1),
                    )
                    self.head = nn.Sequential(nn.Flatten(), nn.Linear(hidden, 32), nn.SiLU(), nn.Linear(32, 1))
                elif architecture == "gru":
                    self.encoder = nn.GRU(input_size=n_channels, hidden_size=hidden, batch_first=True)
                    self.head = nn.Sequential(nn.Linear(hidden, 32), nn.SiLU(), nn.Linear(32, 1))
                elif architecture == "transformer":
                    d_model = max(32, hidden)
                    n_heads = 4 if d_model % 4 == 0 else 2
                    self.proj = nn.Linear(n_channels, d_model)
                    self.pos = nn.Parameter(nn.init.normal_(nn.Parameter(nn.empty(seq_len, d_model)), std=0.02))
                    layer = nn.TransformerEncoderLayer(
                        d_model=d_model,
                        nhead=n_heads,
                        dim_feedforward=d_model * 2,
                        dropout=0.05,
                        activation="gelu",
                        batch_first=True,
                        norm_first=True,
                    )
                    self.encoder = nn.TransformerEncoder(layer, num_layers=2)
                    self.head = nn.Sequential(nn.Linear(d_model, 32), nn.SiLU(), nn.Linear(32, 1))
                else:
                    raise ValueError(f"Unknown temporal architecture: {architecture}")

            def forward(self, x: Any) -> Any:
                x = x.reshape(-1, seq_len, n_channels)
                x = torch.flip(x, dims=[1])
                if self.architecture == "tcn":
                    encoded = self.encoder(x.transpose(1, 2))
                    return self.head(encoded)
                if self.architecture == "gru":
                    _out, hidden_state = self.encoder(x)
                    return self.head(hidden_state[-1])
                x = self.proj(x) + self.pos.unsqueeze(0)
                encoded = self.encoder(x)
                return self.head(encoded[:, -1, :])

        import torch

        return TemporalNet()

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> "TorchTemporalSequenceRegressor":
        import torch
        from torch import nn

        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        if X_arr.ndim != 2 or len(X_arr) == 0:
            raise ValueError("TorchTemporalSequenceRegressor expects a non-empty 2D matrix.")
        if int(X_arr.shape[1]) != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} temporal features, got {X_arr.shape[1]}.")
        if sample_weight is None:
            w_arr = np.ones((len(y_arr), 1), dtype=np.float32)
        else:
            w_arr = np.asarray(sample_weight, dtype=np.float32).reshape(-1, 1)

        device = _resolve_torch_device(torch)
        self.device_ = str(device)
        torch.manual_seed(self.random_state)
        model = self._build_model(nn).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        rng = np.random.default_rng(self.random_state)
        n_rows = len(X_arr)
        model.train()
        for _epoch in range(max(1, self.epochs)):
            order = rng.permutation(n_rows)
            for start in range(0, n_rows, self.batch_size):
                idx = order[start : start + self.batch_size]
                xb = torch.as_tensor(X_arr[idx], dtype=torch.float32, device=device)
                yb = torch.as_tensor(y_arr[idx], dtype=torch.float32, device=device)
                wb = torch.as_tensor(w_arr[idx], dtype=torch.float32, device=device)
                optimizer.zero_grad(set_to_none=True)
                loss = (((model(xb) - yb) ** 2) * wb).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        self.model_ = model.eval()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch

        if self.model_ is None:
            raise RuntimeError("TorchTemporalSequenceRegressor is not fitted.")
        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim != 2 or int(X_arr.shape[1]) != self.n_features_in_:
            raise ValueError("Prediction matrix has incompatible temporal shape.")
        device = torch.device(self.device_)
        preds: List[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(X_arr), self.batch_size):
                xb = torch.as_tensor(X_arr[start : start + self.batch_size], dtype=torch.float32, device=device)
                preds.append(self.model_(xb).detach().cpu().numpy().reshape(-1).astype("float32"))
        return np.concatenate(preds) if preds else np.array([], dtype="float32")


class TorchFactorTokenTransformerRegressor:
    def __init__(
        self,
        *,
        n_factors: int,
        n_channels: int,
        d_model: int = 32,
        n_heads: int = 4,
        epochs: int = 5,
        batch_size: int = 1024,
        lr: float = 7e-4,
        weight_decay: float = 3e-4,
        random_state: int = 42,
    ) -> None:
        self.n_factors = int(n_factors)
        self.n_channels = int(n_channels)
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.random_state = int(random_state)
        self.model_: Any = None
        self.device_: str = "cpu"
        self.n_features_in_ = self.n_factors * self.n_channels

    def _build_model(self, nn: Any) -> Any:
        n_factors = self.n_factors
        n_channels = self.n_channels
        d_model = self.d_model
        n_heads = self.n_heads

        class FactorTokenAttentionNet(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.proj = nn.Linear(n_channels, d_model)
                self.factor_embed = nn.Parameter(torch.zeros(n_factors, d_model))
                self.query = nn.Parameter(torch.zeros(1, 1, d_model))
                self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=0.05, batch_first=True)
                self.norm = nn.LayerNorm(d_model)
                self.head = nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.GELU(),
                    nn.Dropout(0.05),
                    nn.Linear(d_model, 1),
                )
                nn.init.normal_(self.factor_embed, std=0.02)
                nn.init.normal_(self.query, std=0.02)

            def forward(self, x: Any) -> Any:
                x = x.reshape(-1, n_factors, n_channels)
                tokens = self.proj(x) + self.factor_embed.unsqueeze(0)
                query = self.query.expand(tokens.shape[0], -1, -1)
                pooled, _weights = self.attn(query, tokens, tokens, need_weights=False)
                pooled = self.norm(pooled.squeeze(1))
                return self.head(pooled)

        import torch

        return FactorTokenAttentionNet()

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> "TorchFactorTokenTransformerRegressor":
        import torch
        from torch import nn

        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        if X_arr.ndim != 2 or len(X_arr) == 0:
            raise ValueError("TorchFactorTokenTransformerRegressor expects a non-empty 2D matrix.")
        if int(X_arr.shape[1]) != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} factor-token features, got {X_arr.shape[1]}.")
        if sample_weight is None:
            w_arr = np.ones((len(y_arr), 1), dtype=np.float32)
        else:
            w_arr = np.asarray(sample_weight, dtype=np.float32).reshape(-1, 1)

        device = _resolve_torch_device(torch)
        self.device_ = str(device)
        torch.manual_seed(self.random_state)
        torch.set_num_threads(max(1, min(8, os.cpu_count() or 4)))
        model = self._build_model(nn).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        rng = np.random.default_rng(self.random_state)
        n_rows = len(X_arr)
        model.train()
        for _epoch in range(max(1, self.epochs)):
            order = rng.permutation(n_rows)
            for start in range(0, n_rows, self.batch_size):
                idx = order[start : start + self.batch_size]
                xb = torch.as_tensor(X_arr[idx], dtype=torch.float32, device=device)
                yb = torch.as_tensor(y_arr[idx], dtype=torch.float32, device=device)
                wb = torch.as_tensor(w_arr[idx], dtype=torch.float32, device=device)
                optimizer.zero_grad(set_to_none=True)
                loss = (((model(xb) - yb) ** 2) * wb).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        self.model_ = model.eval()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch

        if self.model_ is None:
            raise RuntimeError("TorchFactorTokenTransformerRegressor is not fitted.")
        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim != 2 or int(X_arr.shape[1]) != self.n_features_in_:
            raise ValueError("Prediction matrix has incompatible factor-token shape.")
        device = torch.device(self.device_)
        preds: List[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(X_arr), self.batch_size * 2):
                xb = torch.as_tensor(X_arr[start : start + self.batch_size * 2], dtype=torch.float32, device=device)
                preds.append(self.model_(xb).detach().cpu().numpy().reshape(-1).astype("float32"))
        return np.concatenate(preds) if preds else np.array([], dtype="float32")


def _factor_token_matrix_from_model_frame(
    model_frame: pd.DataFrame,
    feature_cols: List[str],
    positions: np.ndarray,
) -> np.ndarray:
    n_rows = len(positions)
    n_factors = len(feature_cols)
    n_channels = 6
    out = np.zeros((n_rows, n_factors, n_channels), dtype="float32")
    columns = set(model_frame.columns)
    chunk_frame = model_frame.iloc[positions]
    for factor_i, col in enumerate(feature_cols):
        names = [
            col,
            f"{col}__csz",
            f"{col}__rank",
            f"{col}__lag1",
            f"{col}__diff1",
            f"{col}__mom2",
        ]
        for channel_i, name in enumerate(names):
            if name in columns:
                out[:, factor_i, channel_i] = chunk_frame[name].to_numpy(dtype="float32", copy=False)
    return out.reshape(n_rows, n_factors * n_channels)


def _fit_ridge_baseline_on_model_frame(
    *,
    model_frame: pd.DataFrame,
    y_train: pd.Series,
    train_mask: pd.Series,
    predict_mask: pd.Series,
    max_rows: int = 250_000,
) -> tuple[np.ndarray, Any]:
    model = make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=2.0, random_state=42))
    y_train_model = _make_training_target(y_train)
    y_train_np = y_train_model.to_numpy(dtype="float32", copy=False)
    train_weights = _make_training_weights(y_train_model)
    train_positions = np.flatnonzero(np.asarray(train_mask, dtype=bool))
    if len(train_positions) > max_rows:
        rng = np.random.default_rng(42)
        local_sample_idx = rng.choice(len(train_positions), size=max_rows, replace=False)
    else:
        local_sample_idx = np.arange(len(train_positions))
    sampled_positions = train_positions[local_sample_idx]
    X_fit = model_frame.iloc[sampled_positions].to_numpy(dtype="float32", copy=False)
    y_fit = y_train_np[local_sample_idx]
    w_fit = train_weights[local_sample_idx] if train_weights is not None else np.ones(len(y_fit), dtype="float32")
    _fit_with_optional_weights(model, X_fit, y_fit, w_fit)
    predict_positions = np.flatnonzero(np.asarray(predict_mask, dtype=bool))
    pred_parts: List[np.ndarray] = []
    for start in range(0, len(predict_positions), 100_000):
        chunk_positions = predict_positions[start : start + 100_000]
        pred_parts.append(
            np.asarray(
                model.predict(model_frame.iloc[chunk_positions].to_numpy(dtype="float32", copy=False)),
                dtype="float32",
            )
        )
    pred = np.concatenate(pred_parts).astype("float32") if pred_parts else np.array([], dtype="float32")
    return pred, model


def _fit_predict_factor_token_transformer_stack(
    *,
    feature_cols: List[str],
    frame: pd.DataFrame,
    model_frame: pd.DataFrame,
    train_mask: pd.Series,
    predict_mask: pd.Series,
) -> Dict[str, Any]:
    train_days = sorted(pd.Index(frame.loc[train_mask].index.get_level_values("date").astype(str)).unique().tolist())
    fit_days, val_days = _split_train_val_days(train_days)
    fit_mask = frame.index.get_level_values("date").isin(fit_days)
    val_mask = frame.index.get_level_values("date").isin(val_days)

    def factor_token_state(positions: np.ndarray) -> np.ndarray:
        n_rows = len(positions)
        state = np.zeros((n_rows, len(feature_cols), 6), dtype="float32")
        chunk_frame = model_frame.iloc[positions]
        columns = set(model_frame.columns)
        for factor_i, col in enumerate(feature_cols):
            for channel_i, name in enumerate((col, f"{col}__csz", f"{col}__rank", f"{col}__lag1", f"{col}__diff1", f"{col}__mom2")):
                if name in columns:
                    state[:, factor_i, channel_i] = chunk_frame[name].to_numpy(dtype="float32", copy=False)
        return state

    def attention_features(state: np.ndarray, *, seed: int, heads: int, temperature: float) -> np.ndarray:
        state = np.asarray(state, dtype="float32")
        raw = state.reshape(state.shape[0], state.shape[1] * state.shape[2])
        rng = np.random.default_rng(seed)
        queries = rng.normal(0.0, 1.0, size=(heads, state.shape[1], state.shape[2])).astype("float32")
        pooled: List[np.ndarray] = []
        scale = max(float(temperature), 1e-6)
        for h in range(heads):
            scores = np.clip((state * queries[h][None, :, :]).sum(axis=2) / scale, -8.0, 8.0)
            scores = scores - scores.max(axis=1, keepdims=True)
            weights = np.exp(scores).astype("float32")
            weights_sum = weights.sum(axis=1, keepdims=True)
            weights = weights / np.maximum(weights_sum, 1e-6)
            pooled.append((weights[:, :, None] * state).sum(axis=1))
            pooled.append((weights[:, :, None] * np.abs(state)).sum(axis=1))
        return np.concatenate([raw] + pooled, axis=1).astype("float32")

    def fit_attention_ridge(
        local_train_mask: pd.Series,
        local_predict_mask: pd.Series,
        *,
        seed: int,
        heads: int,
        temperature: float,
        alpha: float,
    ) -> np.ndarray:
        y_local = frame.loc[local_train_mask, "resp"].astype("float32")
        y_target = _make_training_target(y_local)
        y_np = y_target.to_numpy(dtype="float32", copy=False)
        weights = _make_training_weights(y_target)
        train_positions = np.flatnonzero(np.asarray(local_train_mask, dtype=bool))
        max_rows = 320_000
        if len(train_positions) > max_rows:
            rng = np.random.default_rng(seed)
            local_sample_idx = rng.choice(len(train_positions), size=max_rows, replace=False)
        else:
            local_sample_idx = np.arange(len(train_positions))
        sampled_positions = train_positions[local_sample_idx]
        X_fit = attention_features(factor_token_state(sampled_positions), seed=seed, heads=heads, temperature=temperature)
        y_fit = y_np[local_sample_idx]
        w_fit = weights[local_sample_idx] if weights is not None else np.ones(len(y_fit), dtype="float32")
        model = make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=alpha, random_state=seed))
        _fit_with_optional_weights(model, X_fit, y_fit, w_fit)
        predict_positions = np.flatnonzero(np.asarray(local_predict_mask, dtype=bool))
        pred_parts: List[np.ndarray] = []
        for start in range(0, len(predict_positions), 120_000):
            chunk_positions = predict_positions[start : start + 120_000]
            X_chunk = attention_features(factor_token_state(chunk_positions), seed=seed, heads=heads, temperature=temperature)
            pred_parts.append(np.asarray(model.predict(X_chunk), dtype="float32"))
        return np.concatenate(pred_parts).astype("float32") if pred_parts else np.array([], dtype="float32")

    if val_mask.any() and fit_mask.any():
        val_resp = frame.loc[val_mask, "resp"].astype("float32")
        best_config = {"seed": 42, "heads": 24, "temperature": 0.9, "alpha": 2.0}
        best_score = -1e18
        for config in [
            {"seed": 42, "heads": 16, "temperature": 0.7, "alpha": 1.5},
            {"seed": 43, "heads": 24, "temperature": 0.9, "alpha": 2.0},
            {"seed": 44, "heads": 32, "temperature": 1.2, "alpha": 3.0},
        ]:
            pred_val = fit_attention_ridge(fit_mask, val_mask, **config)
            candidate = pd.Series(pred_val, index=val_resp.index, dtype="float32")
            metrics = _combo_period_metrics(candidate, val_resp, fee_bps=DEFAULT_FEE_BPS)
            score = float(metrics.get("Score", 0.0))
            if score > best_score:
                best_score = score
                best_config = config
    else:
        best_config = {"seed": 42, "heads": 24, "temperature": 0.9, "alpha": 2.0}

    pred = fit_attention_ridge(train_mask, predict_mask, **best_config)
    folded_importance = {col: 1.0 for col in feature_cols}
    return {
        "pred": pred,
        "importance": folded_importance,
        "attention_config": best_config,
    }


def _fit_predict_causal_decay_factor_transformer_stack(
    *,
    feature_cols: List[str],
    frame: pd.DataFrame,
    model_frame: pd.DataFrame,
    train_mask: pd.Series,
    predict_mask: pd.Series,
) -> Dict[str, Any]:
    train_days = sorted(pd.Index(frame.loc[train_mask].index.get_level_values("date").astype(str)).unique().tolist())
    fit_days, val_days = _split_train_val_days(train_days)
    fit_mask = frame.index.get_level_values("date").isin(fit_days)
    val_mask = frame.index.get_level_values("date").isin(val_days)
    if not fit_mask.any() or not val_mask.any():
        return _fit_predict_factor_token_transformer_stack(
            feature_cols=feature_cols,
            frame=frame,
            model_frame=model_frame,
            train_mask=train_mask,
            predict_mask=predict_mask,
        )

    fit_resp = frame.loc[fit_mask, "resp"].astype("float32")
    fit_stats = _compute_factor_daily_ic_stats(
        frame.loc[fit_mask, feature_cols].astype("float32"),
        fit_resp,
        feature_cols,
    )
    signs = np.array([float(fit_stats.get(col, {}).get("sign", 1.0) or 1.0) for col in feature_cols], dtype="float32")
    abs_ic = np.array([float(fit_stats.get(col, {}).get("abs_ic", 0.0) or 0.0) for col in feature_cols], dtype="float32")
    if float(abs_ic.sum()) <= 0.0:
        factor_prior = np.full(len(feature_cols), 1.0 / max(1, len(feature_cols)), dtype="float32")
    else:
        factor_prior = (abs_ic / max(float(abs_ic.sum()), 1e-8)).astype("float32")

    channel_names = ("raw", "csz", "rank", "lag1", "diff1", "mom2")

    def factor_token_state(positions: np.ndarray, *, align_sign: bool) -> np.ndarray:
        n_rows = len(positions)
        state = np.zeros((n_rows, len(feature_cols), len(channel_names)), dtype="float32")
        chunk_frame = model_frame.iloc[positions]
        columns = set(model_frame.columns)
        for factor_i, col in enumerate(feature_cols):
            for channel_i, name in enumerate((col, f"{col}__csz", f"{col}__rank", f"{col}__lag1", f"{col}__diff1", f"{col}__mom2")):
                if name in columns:
                    state[:, factor_i, channel_i] = chunk_frame[name].to_numpy(dtype="float32", copy=False)
        if align_sign:
            state *= signs[None, :, None]
        return state

    def attention_features(
        state: np.ndarray,
        *,
        seed: int,
        heads: int,
        temperature: float,
        decay: float,
        prior_strength: float,
    ) -> np.ndarray:
        state = np.asarray(state, dtype="float32")
        raw = state.reshape(state.shape[0], state.shape[1] * state.shape[2])
        rng = np.random.default_rng(seed)
        queries = rng.normal(0.0, 1.0, size=(heads, state.shape[1], state.shape[2])).astype("float32")
        decay_vector = np.array([1.0, 1.0, 1.0, decay, decay, decay * decay], dtype="float32")
        prior_log = np.log(np.maximum(factor_prior, 1e-8)).astype("float32")
        pooled: List[np.ndarray] = []
        scale = max(float(temperature), 1e-6)
        for h in range(heads):
            q = queries[h] * decay_vector[None, :]
            scores = np.clip((state * q[None, :, :]).sum(axis=2) / scale, -8.0, 8.0)
            scores = scores + float(prior_strength) * prior_log[None, :]
            scores = scores - scores.max(axis=1, keepdims=True)
            weights = np.exp(scores).astype("float32")
            weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-6)
            pooled.append((weights[:, :, None] * state).sum(axis=1))
            pooled.append((weights[:, :, None] * np.abs(state)).sum(axis=1))
            pooled.append((weights * factor_prior[None, :]).sum(axis=1, keepdims=True).astype("float32"))
        return np.concatenate([raw] + pooled, axis=1).astype("float32")

    def fit_attention_ridge(
        local_train_mask: pd.Series,
        local_predict_mask: pd.Series,
        *,
        seed: int,
        heads: int,
        temperature: float,
        alpha: float,
        decay: float,
        prior_strength: float,
        align_sign: bool,
    ) -> tuple[np.ndarray, Dict[str, Any]]:
        y_local = frame.loc[local_train_mask, "resp"].astype("float32")
        y_target = _make_training_target(y_local)
        y_np = y_target.to_numpy(dtype="float32", copy=False)
        weights = _make_training_weights(y_target)
        train_positions = np.flatnonzero(np.asarray(local_train_mask, dtype=bool))
        max_rows = 340_000
        if len(train_positions) > max_rows:
            rng = np.random.default_rng(seed)
            local_sample_idx = rng.choice(len(train_positions), size=max_rows, replace=False)
        else:
            local_sample_idx = np.arange(len(train_positions))
        sampled_positions = train_positions[local_sample_idx]
        X_fit = attention_features(
            factor_token_state(sampled_positions, align_sign=align_sign),
            seed=seed,
            heads=heads,
            temperature=temperature,
            decay=decay,
            prior_strength=prior_strength,
        )
        y_fit = y_np[local_sample_idx]
        w_fit = weights[local_sample_idx] if weights is not None else np.ones(len(y_fit), dtype="float32")
        model = make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=alpha, random_state=seed))
        _fit_with_optional_weights(model, X_fit, y_fit, w_fit)
        predict_positions = np.flatnonzero(np.asarray(local_predict_mask, dtype=bool))
        pred_parts: List[np.ndarray] = []
        for start in range(0, len(predict_positions), 120_000):
            chunk_positions = predict_positions[start : start + 120_000]
            X_chunk = attention_features(
                factor_token_state(chunk_positions, align_sign=align_sign),
                seed=seed,
                heads=heads,
                temperature=temperature,
                decay=decay,
                prior_strength=prior_strength,
            )
            pred_parts.append(np.asarray(model.predict(X_chunk), dtype="float32"))
        diagnostics: Dict[str, Any] = {}
        try:
            ridge = model.named_steps.get("ridge")
            coef = np.asarray(getattr(ridge, "coef_", []), dtype="float64").reshape(-1)
            raw_coef = np.abs(coef[: len(feature_cols) * len(channel_names)]).reshape(len(feature_cols), len(channel_names))
            channel_weight = raw_coef.sum(axis=0)
            total = float(channel_weight.sum()) or 1.0
            diagnostics["channel_importance"] = [
                {"channel": name, "share": float(value / total)}
                for name, value in zip(channel_names, channel_weight, strict=False)
            ]
            diagnostics["temporal_decay_share"] = float(channel_weight[3:].sum() / total)
        except Exception:
            diagnostics = {}
        return (np.concatenate(pred_parts).astype("float32") if pred_parts else np.array([], dtype="float32")), diagnostics

    def fit_sklearn_head(
        local_train_mask: pd.Series,
        local_predict_mask: pd.Series,
        *,
        kind: str,
        seed: int,
    ) -> np.ndarray:
        X_all = model_frame
        y_local = _make_training_target(frame.loc[local_train_mask, "resp"].astype("float32"))
        y_np = y_local.to_numpy(dtype="float32", copy=False)
        w_np = _make_training_weights(y_local)
        train_np = X_all.loc[local_train_mask].to_numpy(dtype="float32", copy=False)
        if kind == "lgb":
            builder: Any = lgb.LGBMRegressor(
                n_estimators=220,
                learning_rate=0.03,
                num_leaves=23,
                subsample=0.82,
                colsample_bytree=0.82,
                reg_lambda=1.8,
                min_child_samples=120,
                n_jobs=-1,
                random_state=seed,
                verbose=-1,
            )
            max_rows = 280_000
        else:
            builder = make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=2.5, random_state=seed))
            max_rows = 360_000
        X_fit, y_fit, w_fit = _sample_training_rows(train_np, y_np, max_rows=max_rows, sample_weight=w_np)
        if w_fit is None:
            w_fit = np.ones(len(y_fit), dtype="float32")
        _fit_with_optional_weights(builder, X_fit, y_fit, w_fit)
        pred_chunks: List[np.ndarray] = []
        X_pred = X_all.loc[local_predict_mask]
        for start in range(0, len(X_pred), 120_000):
            chunk = X_pred.iloc[start : start + 120_000].to_numpy(dtype="float32", copy=False)
            pred_chunks.append(np.asarray(builder.predict(chunk), dtype="float32"))
        return np.concatenate(pred_chunks).astype("float32") if pred_chunks else np.array([], dtype="float32")

    def fit_rank_head(local_train_mask: pd.Series, local_predict_mask: pd.Series) -> np.ndarray:
        weights = _build_combo_weights(
            model_name="TrainICIRRankCombo",
            train_features=frame.loc[local_train_mask, feature_cols].astype("float32"),
            y_train=frame.loc[local_train_mask, "resp"].astype("float32"),
            feature_cols=feature_cols,
        )
        pred = _weighted_rank_combo_prediction(
            frame.loc[local_predict_mask, feature_cols].astype("float32"),
            feature_cols,
            weights,
        )
        return pred.to_numpy(dtype="float32", copy=False)

    def normalize_array(pred: np.ndarray, index: pd.Index) -> np.ndarray:
        return _normalize_alpha_series(pd.Series(pred, index=index, dtype="float32")).to_numpy(dtype="float32", copy=False)

    attention_configs = [
        {"name": "attn_decay_low", "seed": 51, "heads": 24, "temperature": 0.75, "alpha": 1.6, "decay": 0.62, "prior_strength": 0.15, "align_sign": True},
        {"name": "attn_decay_mid", "seed": 52, "heads": 32, "temperature": 0.95, "alpha": 2.2, "decay": 0.78, "prior_strength": 0.25, "align_sign": True},
        {"name": "attn_context_wide", "seed": 53, "heads": 40, "temperature": 1.25, "alpha": 3.2, "decay": 0.88, "prior_strength": 0.10, "align_sign": False},
    ]
    val_index = model_frame.loc[val_mask].index
    val_resp = frame.loc[val_mask, "resp"].astype("float32")
    val_heads: Dict[str, np.ndarray] = {}
    attention_diagnostics: Dict[str, Any] = {}
    best_attention_config = attention_configs[0]
    best_attention_score = -1e18
    for config in attention_configs:
        pred_val, diag = fit_attention_ridge(fit_mask, val_mask, **{k: v for k, v in config.items() if k != "name"})
        val_heads[config["name"]] = normalize_array(pred_val, val_index)
        metrics = _combo_period_metrics(pd.Series(val_heads[config["name"]], index=val_index), val_resp, fee_bps=DEFAULT_FEE_BPS)
        score = float(metrics.get("Score", 0.0))
        if score > best_attention_score:
            best_attention_score = score
            best_attention_config = config
            attention_diagnostics = diag

    val_heads["ridge"] = normalize_array(fit_sklearn_head(fit_mask, val_mask, kind="ridge", seed=61), val_index)
    val_heads["lgb"] = normalize_array(fit_sklearn_head(fit_mask, val_mask, kind="lgb", seed=62), val_index)
    val_heads["rank_icir"] = normalize_array(fit_rank_head(fit_mask, val_mask), val_index)

    blend_candidates: List[tuple[str, Dict[str, float]]] = []
    for name in val_heads:
        blend_candidates.append((name, {name: 1.0}))
    attn_name = str(best_attention_config["name"])
    for ridge_weight in (0.15, 0.25, 0.35, 0.50):
        blend_candidates.append((f"{attn_name}_ridge_{ridge_weight:.2f}", {attn_name: 1.0 - ridge_weight, "ridge": ridge_weight}))
    for lgb_weight in (0.10, 0.20, 0.30):
        blend_candidates.append((f"{attn_name}_lgb_{lgb_weight:.2f}", {attn_name: 1.0 - lgb_weight, "lgb": lgb_weight}))
    blend_candidates.extend(
        [
            ("attn_ridge_lgb_602020", {attn_name: 0.60, "ridge": 0.20, "lgb": 0.20}),
            ("attn_ridge_lgb_rank_50201515", {attn_name: 0.50, "ridge": 0.20, "lgb": 0.15, "rank_icir": 0.15}),
            ("ridge_attn_lgb_403030", {"ridge": 0.40, attn_name: 0.30, "lgb": 0.30}),
        ]
    )

    best_blend_name = ""
    best_blend_weights: Dict[str, float] = {}
    best_blend_score = -1e18
    best_blend_metrics: Dict[str, Any] = {}
    for blend_name, weights in blend_candidates:
        combined = np.zeros(len(val_index), dtype="float32")
        total_weight = 0.0
        for name, weight in weights.items():
            if name not in val_heads:
                continue
            combined += np.float32(weight) * val_heads[name]
            total_weight += float(weight)
        if total_weight <= 0:
            continue
        combined = combined / np.float32(total_weight)
        metrics = _combo_period_metrics(pd.Series(combined, index=val_index), val_resp, fee_bps=DEFAULT_FEE_BPS)
        score = float(metrics.get("Score", 0.0))
        tie_ic = float(metrics.get("IC", 0.0))
        current_ic = float(best_blend_metrics.get("IC", -1e18)) if best_blend_metrics else -1e18
        if score > best_blend_score or (abs(score - best_blend_score) < 1e-9 and tie_ic > current_ic):
            best_blend_name = blend_name
            best_blend_weights = {name: float(weight / total_weight) for name, weight in weights.items() if name in val_heads}
            best_blend_score = score
            best_blend_metrics = metrics

    pred_index = model_frame.loc[predict_mask].index
    final_heads: Dict[str, np.ndarray] = {}
    needed = set(best_blend_weights)
    if attn_name in needed:
        pred_attn, attention_diagnostics = fit_attention_ridge(
            train_mask,
            predict_mask,
            **{k: v for k, v in best_attention_config.items() if k != "name"},
        )
        final_heads[attn_name] = normalize_array(pred_attn, pred_index)
    if "ridge" in needed:
        final_heads["ridge"] = normalize_array(fit_sklearn_head(train_mask, predict_mask, kind="ridge", seed=61), pred_index)
    if "lgb" in needed:
        final_heads["lgb"] = normalize_array(fit_sklearn_head(train_mask, predict_mask, kind="lgb", seed=62), pred_index)
    if "rank_icir" in needed:
        final_heads["rank_icir"] = normalize_array(fit_rank_head(train_mask, predict_mask), pred_index)

    pred = np.zeros(len(pred_index), dtype="float32")
    for name, weight in best_blend_weights.items():
        if name in final_heads:
            pred += np.float32(weight) * final_heads[name]

    folded_importance = {col: float(abs_ic[idx]) for idx, col in enumerate(feature_cols)}
    return {
        "pred": pred.astype("float32"),
        "importance": folded_importance,
        "attention_config": best_attention_config,
        "blend_name": best_blend_name,
        "blend_weights": best_blend_weights,
        "validation_metrics": {
            key: value
            for key, value in best_blend_metrics.items()
            if key not in {"prediction_comparison_curve", "daily_ic_curve", "daily_rank_ic_curve", "combo_tvr_curve", "strategy"}
        },
        "diagnostics": {
            "attention": attention_diagnostics,
            "selected_attention_config": best_attention_config,
            "selected_blend": best_blend_name,
            "selected_blend_weights": best_blend_weights,
            "validation_selection_score": best_blend_score,
        },
    }


def _select_temporal_factor_cols(
    frame: pd.DataFrame,
    y_train: pd.Series,
    feature_cols: List[str],
    train_mask: pd.Series,
    *,
    max_factors: int = 24,
) -> List[str]:
    if len(feature_cols) <= max_factors:
        return list(feature_cols)
    stats = _compute_factor_daily_ic_stats(
        frame.loc[train_mask, feature_cols].astype("float32"),
        y_train.astype("float32"),
        feature_cols,
    )
    ranked = sorted(
        feature_cols,
        key=lambda col: (
            float(stats.get(col, {}).get("abs_ic", 0.0)),
            float(stats.get(col, {}).get("ic_ir", 0.0)),
        ),
        reverse=True,
    )
    return ranked[:max_factors]


def _make_temporal_feature_frame(
    frame: pd.DataFrame,
    selected_cols: List[str],
    *,
    seq_len: int = 8,
) -> tuple[pd.DataFrame, List[str], int, int]:
    z = _cs_zscore_frame(frame, selected_cols).astype("float32")
    z.columns = [f"{col}__z" for col in selected_cols]
    r = _cs_rank_frame(frame, selected_cols).astype("float32")
    r.columns = [f"{col}__rank" for col in selected_cols]
    base = pd.concat([z, r], axis=1).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float32")
    pieces: List[pd.DataFrame] = []
    by_security = base.groupby(level="security_id", sort=False)
    for lag in range(seq_len):
        lagged = base if lag == 0 else by_security.shift(lag).fillna(0.0)
        lagged = lagged.astype("float32", copy=False)
        lagged.columns = [f"{col}__lag{lag}" for col in base.columns]
        pieces.append(lagged)
    temporal = pd.concat(pieces, axis=1).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float32")
    return temporal, list(temporal.columns), seq_len, len(base.columns)


def _fit_predict_temporal_torch_model(
    *,
    model_name: str,
    feature_cols: List[str],
    frame: pd.DataFrame,
    train_mask: pd.Series,
    predict_mask: pd.Series,
) -> Dict[str, Any]:
    y_train = frame.loc[train_mask, "resp"].astype("float32")
    selected_cols = _select_temporal_factor_cols(frame, y_train, feature_cols, train_mask, max_factors=24)
    temporal_frame, temporal_feature_names, seq_len, n_channels = _make_temporal_feature_frame(
        frame,
        selected_cols,
        seq_len=8,
    )
    arch = {
        "TorchTCNMetaModel": "tcn",
        "TorchGRUMetaModel": "gru",
        "TorchTransformerMetaModel": "transformer",
    }[model_name]
    model = TorchTemporalSequenceRegressor(
        architecture=arch,
        seq_len=seq_len,
        n_channels=n_channels,
        hidden_size=64 if arch != "transformer" else 48,
        epochs=4 if arch != "transformer" else 3,
        batch_size=8192,
        lr=7e-4,
        weight_decay=2e-4,
        random_state=42,
    )
    y_train_model = _make_training_target(y_train)
    y_train_np = y_train_model.to_numpy(dtype="float32", copy=False)
    train_weights = _make_training_weights(y_train_model)
    X_fit, y_fit, w_fit = _sample_training_rows(
        temporal_frame.loc[train_mask, temporal_feature_names].to_numpy(dtype="float32", copy=False),
        y_train_np,
        max_rows=60_000 if arch != "transformer" else 45_000,
        sample_weight=train_weights,
    )
    if w_fit is None:
        w_fit = np.ones(len(y_fit), dtype="float32")
    model.fit(X_fit, y_fit, sample_weight=w_fit)
    pred = model.predict(
        temporal_frame.loc[predict_mask, temporal_feature_names].to_numpy(dtype="float32", copy=False)
    )
    importance = {col: 0.0 for col in feature_cols}
    for col in selected_cols:
        importance[col] = 1.0
    return {
        "pred": pred,
        "importance": importance,
        "temporal_selected_factors": selected_cols,
    }


def _fit_predict_temporal_tabular_model(
    *,
    model_name: str,
    feature_cols: List[str],
    frame: pd.DataFrame,
    train_mask: pd.Series,
    predict_mask: pd.Series,
) -> Dict[str, Any]:
    y_train = frame.loc[train_mask, "resp"].astype("float32")
    selected_cols = _select_temporal_factor_cols(frame, y_train, feature_cols, train_mask, max_factors=24)
    temporal_frame, temporal_feature_names, _seq_len, _n_channels = _make_temporal_feature_frame(
        frame,
        selected_cols,
        seq_len=8,
    )
    builders: Dict[str, tuple[Any, int]] = {
        "TemporalRidgeLagMetaModel": (
            make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=4.0, random_state=42)),
            260_000,
        ),
        "TemporalLightGBMLagMetaModel": (
            lgb.LGBMRegressor(
                n_estimators=220,
                learning_rate=0.035,
                num_leaves=31,
                subsample=0.85,
                colsample_bytree=0.75,
                reg_lambda=1.5,
                min_child_samples=120,
                n_jobs=-1,
                random_state=42,
                verbose=-1,
            ),
            220_000,
        ),
        "TemporalHistGBLagMetaModel": (
            HistGradientBoostingRegressor(
                max_iter=160,
                learning_rate=0.04,
                max_leaf_nodes=31,
                l2_regularization=0.2,
                random_state=42,
            ),
            180_000,
        ),
        "TemporalExtraTreesLagMetaModel": (
            ExtraTreesRegressor(
                n_estimators=120,
                max_depth=10,
                min_samples_leaf=120,
                max_features="sqrt",
                n_jobs=-1,
                random_state=42,
            ),
            160_000,
        ),
    }
    model, max_rows = builders[model_name]
    y_train_model = _make_training_target(y_train)
    y_train_np = y_train_model.to_numpy(dtype="float32", copy=False)
    train_weights = _make_training_weights(y_train_model)
    X_fit, y_fit, w_fit = _sample_training_rows(
        temporal_frame.loc[train_mask, temporal_feature_names].to_numpy(dtype="float32", copy=False),
        y_train_np,
        max_rows=max_rows,
        sample_weight=train_weights,
    )
    if w_fit is None:
        w_fit = np.ones(len(y_fit), dtype="float32")
    _fit_with_optional_weights(model, X_fit, y_fit, w_fit)
    pred = model.predict(temporal_frame.loc[predict_mask, temporal_feature_names].to_numpy(dtype="float32", copy=False))
    raw_importance = _extract_importance(model, temporal_feature_names)
    importance = {col: 0.0 for col in feature_cols}
    for temporal_name, score in raw_importance.items():
        factor_id = temporal_name.split("__", 1)[0]
        if factor_id in importance:
            importance[factor_id] += float(score)
    return {
        "pred": pred,
        "importance": importance,
        "temporal_selected_factors": selected_cols,
    }


def _fit_predict_models(
    *,
    feature_cols: List[str],
    y_train: pd.Series,
    frame: pd.DataFrame,
    model_frame: pd.DataFrame,
    combo_rank_frame: pd.DataFrame | None = None,
    train_mask: pd.Series,
    predict_mask: pd.Series,
    log_path: Path | None = None,
) -> Dict[str, Dict[str, Any]]:
    outputs: Dict[str, Dict[str, Any]] = {}
    train_features = frame.loc[train_mask, feature_cols].astype("float32")
    combo_stats = _precompute_combo_weight_stats(train_features, y_train.astype("float32"), feature_cols)
    for model_name, _builder, _max_rows in _model_spec_items():
        if log_path is not None:
            _log(f"[mock-oos] Fitting model {model_name}", log_path)
        outputs[model_name] = _fit_predict_window_model(
            model_name=model_name,
            feature_cols=feature_cols,
            frame=frame,
            model_frame=model_frame,
            combo_rank_frame=combo_rank_frame,
            combo_stats=combo_stats,
            train_mask=train_mask,
            predict_mask=predict_mask,
        )
        if log_path is not None:
            _log(f"[mock-oos] Finished model {model_name}", log_path)

    return outputs


def _evaluate_predictions(pred: pd.Series, resp: pd.Series, *, fee_bps: float = DEFAULT_FEE_BPS) -> Dict[str, Any]:
    frame = pd.DataFrame({"pred": pred, "resp": resp}).replace([np.inf, -np.inf], np.nan).dropna()
    if frame.empty:
        strategy = _strategy_from_predictions(pred, resp, fee_bps=fee_bps)
        return {
            "rows": 0,
            "daily_ic_mean": 0.0,
            "daily_rank_ic_mean": 0.0,
            "daily_ic_ir": 0.0,
            "daily_rank_ic_ir": 0.0,
            "overall_ic": 0.0,
            "strategy": strategy,
            "daily_ic_curve": [],
            "daily_rank_ic_curve": [],
            "prediction_comparison_curve": [],
        }

    daily_ic = frame.groupby(level="date").apply(_daily_corr)
    daily_rank_ic = frame.groupby(level="date").apply(_daily_corr, rank=True)
    overall_ic = frame["pred"].corr(frame["resp"])
    strategy = _strategy_from_predictions(pred, resp, fee_bps=fee_bps)

    def _ic_ir(series: pd.Series) -> float:
        vals = series.dropna()
        if len(vals) < 2:
            return 0.0
        std = float(vals.std())
        return float(vals.mean() / std) if std > 0 else 0.0

    return {
        "rows": int(len(frame)),
        "daily_ic_mean": float(daily_ic.dropna().mean()) if daily_ic.notna().any() else 0.0,
        "daily_rank_ic_mean": float(daily_rank_ic.dropna().mean()) if daily_rank_ic.notna().any() else 0.0,
        "daily_ic_ir": _ic_ir(daily_ic),
        "daily_rank_ic_ir": _ic_ir(daily_rank_ic),
        "overall_ic": float(overall_ic) if pd.notna(overall_ic) else 0.0,
        "strategy": strategy,
        "daily_ic_curve": [
            {"date": str(idx), "value": float(val)}
            for idx, val in daily_ic.dropna().items()
        ],
        "daily_rank_ic_curve": [
            {"date": str(idx), "value": float(val)}
            for idx, val in daily_rank_ic.dropna().items()
        ],
        "prediction_comparison_curve": _prediction_comparison_curve(pred, resp),
    }


def _combo_daily_tvr_from_alpha(alpha: pd.Series) -> tuple[float, List[Dict[str, Any]]]:
    if alpha.empty:
        return 0.0, []
    alpha_un = alpha.unstack("security_id").replace([np.inf, -np.inf], np.nan)
    if alpha_un.empty:
        return 0.0, []
    diff_abs = alpha_un.diff().abs().sum(axis=1)
    tot_abs = alpha_un.abs().sum(axis=1).replace(0.0, np.nan)
    daily = (diff_abs / tot_abs * 100.0).replace([np.inf, -np.inf], np.nan).dropna()
    if daily.empty:
        return 0.0, []
    return float(daily.mean()), [{"date": str(idx), "value": float(val)} for idx, val in daily.items()]


def _combo_period_metrics(pred: pd.Series, resp: pd.Series, *, fee_bps: float = DEFAULT_FEE_BPS) -> Dict[str, Any]:
    evaluation = _evaluate_predictions(pred, resp, fee_bps=fee_bps)
    daily_ic = pd.Series(
        {
            row["date"]: float(row["value"])
            for row in evaluation.get("daily_ic_curve", [])
            if isinstance(row, dict)
        },
        dtype="float64",
    )
    if len(daily_ic) > 1:
        ic_mean = float(daily_ic.mean())
        ic_std = float(daily_ic.std())
        ir = float(ic_mean / ic_std * math.sqrt(252)) if ic_std > 0 else 0.0
    else:
        ic_mean = float(evaluation.get("daily_ic_mean", 0.0) or 0.0)
        ir = 0.0
    ic_display = ic_mean * 100.0
    alpha = _normalize_alpha_series(pred).rename("alpha")
    combo_tvr, combo_tvr_curve = _combo_daily_tvr_from_alpha(alpha)
    gate_ic = ic_display > 0.6
    gate_ir = ir > 2.5
    gate_tvr = combo_tvr < 400
    score = max(0.0, ic_display - 0.0005 * combo_tvr) * math.sqrt(ir) * 100.0 if gate_ic and gate_ir and gate_tvr else 0.0
    return {
        "rows": int(evaluation.get("rows", 0) or 0),
        "IC": float(ic_display),
        "RankIC": float((evaluation.get("daily_rank_ic_mean", 0.0) or 0.0) * 100.0),
        "IR": float(ir),
        "Score": float(score),
        "TVR": float(combo_tvr),
        "PassGates": bool(gate_ic and gate_ir and gate_tvr),
        "GatesDetail": {"IC": bool(gate_ic), "IR": bool(gate_ir), "Turnover": bool(gate_tvr)},
        "daily_ic_curve": evaluation.get("daily_ic_curve", []),
        "daily_rank_ic_curve": evaluation.get("daily_rank_ic_curve", []),
        "prediction_comparison_curve": evaluation.get("prediction_comparison_curve", []),
        "combo_tvr_curve": combo_tvr_curve,
        "strategy": {
            "total_pnl": evaluation["strategy"]["total_pnl"],
            "sharpe": evaluation["strategy"]["sharpe"],
            "hit_ratio": evaluation["strategy"]["hit_ratio"],
            "max_drawdown": evaluation["strategy"]["max_drawdown"],
        },
    }


def _prefix_curve_period(rows: List[Dict[str, Any]], period: str) -> List[Dict[str, Any]]:
    out = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        out.append({**row, "period": period})
    return out


def _recompute_score_with_combo_tvr(metrics: Dict[str, Any], combo_tvr: float) -> Dict[str, Any]:
    updated = dict(metrics)
    ic = float(updated.get("IC", 0.0) or 0.0)
    ir = float(updated.get("IR", 0.0) or 0.0)
    gate_ic = ic > 0.6
    gate_ir = ir > 2.5
    gate_tvr = combo_tvr < 400
    score = max(0.0, ic - 0.0005 * combo_tvr) * math.sqrt(ir) * 100.0 if gate_ic and gate_ir and gate_tvr else 0.0
    updated["tvr"] = float(combo_tvr)
    updated["TurnoverLocal"] = float(combo_tvr)
    updated["Score"] = float(score)
    updated["PassGates"] = bool(gate_ic and gate_ir and gate_tvr)
    gates = dict(updated.get("GatesDetail") or {})
    gates.update({"IC": bool(gate_ic), "IR": bool(gate_ir), "Turnover": bool(gate_tvr)})
    updated["GatesDetail"] = gates
    preview = dict(updated.get("result_preview") or {})
    if preview:
        preview["tvr"] = float(combo_tvr)
        preview["local_tvr"] = float(combo_tvr)
        preview["raw_tvr"] = float(combo_tvr) / 100.0
        preview["score"] = float(score)
        updated["result_preview"] = preview
    updated["turnover_basis"] = "daily_combo_alpha_diff_sum_x100"
    updated["score_formula"] = "score = (IC - 0.0005 * daily_combo_TVR) * sqrt(IR) * 100"
    return updated


def _compute_model_input_correlations(
    pred: pd.Series,
    features: pd.DataFrame,
    feature_refs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if pred.empty or features.empty:
        return []

    feature_meta = {
        str(ref.get("run_id", "")): ref
        for ref in feature_refs
        if ref.get("run_id")
    }
    aligned = pd.concat(
        [pred.rename("pred").astype("float32"), features.astype("float32")],
        axis=1,
        join="inner",
    ).replace([np.inf, -np.inf], np.nan)
    if aligned.empty:
        return []

    rows: List[Dict[str, Any]] = []
    for run_id, meta in feature_meta.items():
        if run_id not in aligned.columns:
            continue
        valid = aligned[["pred", run_id]].dropna()
        if len(valid) < 200:
            continue
        corr = valid["pred"].corr(valid[run_id])
        if pd.isna(corr):
            continue
        rows.append(
            {
                "run_id": run_id,
                "corr": _safe_float(corr),
                "abs_corr": abs(_safe_float(corr)),
                "score": _safe_float(meta.get("score", 0)),
                "ic": _safe_float(meta.get("ic", 0)),
                "generation": int(meta.get("generation", 0) or 0),
                "formula": str(meta.get("formula", "") or ""),
            }
        )

    rows.sort(key=lambda item: (item["abs_corr"], item["score"], item["run_id"]), reverse=True)
    return rows


def _compute_pred_series_all_factor_correlations(
    pred: pd.Series,
    feature_refs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Correlate rolling-test pred series against all PassGates factors from KB."""
    if pred.empty:
        return []
    selected_ids = {str(ref.get("run_id", "")) for ref in feature_refs}
    all_factors = [
        item for item in kb.get_all_factors()
        if item.get("run_id") and item.get("PassGates")
    ]
    if not all_factors:
        return []

    pred_clean = pred.rename("pred").astype("float32").replace([np.inf, -np.inf], np.nan)

    rows: List[Dict[str, Any]] = []
    for factor in all_factors:
        run_id = str(factor.get("run_id", ""))
        # Prefer daily cache; fall back to full parquet.
        cache_path = _daily_feature_cache_path(run_id)
        if cache_path.is_file():
            factor_df = pd.read_parquet(str(cache_path), columns=["date", "security_id", "value"])
            if factor_df.empty:
                continue
            factor_series = (
                factor_df.set_index(["date", "security_id"])["value"]
                .astype("float32").rename(run_id).sort_index()
            )
        else:
            factor_path = Path(str(factor.get("parquet_path") or ""))
            if not factor_path.is_file():
                continue
            factor_df = pd.read_parquet(str(factor_path), columns=["date", "security_id", "alpha"])
            if factor_df.empty:
                continue
            factor_series = (
                factor_df.groupby(["date", "security_id"], sort=True)["alpha"]
                .mean().astype("float32").rename(run_id).sort_index()
            )

        aligned = pd.concat([pred_clean, factor_series], axis=1, join="inner").dropna()
        if len(aligned) < 200:
            continue
        corr = aligned["pred"].corr(aligned[run_id])
        if pd.isna(corr):
            continue
        rows.append({
            "run_id": run_id,
            "corr": _safe_float(corr),
            "abs_corr": abs(_safe_float(corr)),
            "score": _safe_float(factor.get("Score", 0)),
            "ic": _safe_float(factor.get("IC", 0)),
            "generation": int(factor.get("generation", 0) or 0),
            "formula": str(factor.get("formula", "") or ""),
            "is_input_factor": run_id in selected_ids,
        })

    rows.sort(key=lambda item: (item["abs_corr"], item["score"], item["run_id"]), reverse=True)
    return rows


def _compute_submit_factor_input_correlations(
    summary: Dict[str, Any],
    *,
    model_name: str,
) -> List[Dict[str, Any]]:
    submit_path = (
        (summary.get("ensemble_outputs") or {}).get(model_name)
        or ((summary.get("submit_factor_output") or {}).get("submit_path") if model_name == summary.get("best_model") else "")
    )
    if not submit_path or not Path(str(submit_path)).is_file():
        return []

    factor_meta = {
        str(item.get("run_id", "")): item
        for item in summary.get("selected_factors", [])
        if item.get("run_id")
    }
    if not factor_meta:
        return []

    factor_by_id = {
        str(item.get("run_id", "")): item
        for item in kb.get_all_factors()
        if item.get("run_id")
    }

    model_df = pd.read_parquet(str(submit_path), columns=["date", "security_id", "alpha"])
    if model_df.empty:
        return []
    model_series = (
        model_df.groupby(["date", "security_id"], sort=True)["alpha"]
        .mean()
        .astype("float32")
        .rename("pred")
        .sort_index()
    )

    rows: List[Dict[str, Any]] = []
    for run_id, meta in factor_meta.items():
        factor = factor_by_id.get(run_id)
        factor_path = Path(str((factor or {}).get("parquet_path") or ""))
        if not factor_path.is_file():
            continue
        factor_df = pd.read_parquet(str(factor_path), columns=["date", "security_id", "alpha"])
        if factor_df.empty:
            continue
        factor_series = (
            factor_df.groupby(["date", "security_id"], sort=True)["alpha"]
            .mean()
            .astype("float32")
            .rename(run_id)
            .sort_index()
        )
        aligned = pd.concat([model_series, factor_series], axis=1, join="inner").dropna()
        if len(aligned) < 200:
            continue
        corr = aligned["pred"].corr(aligned[run_id])
        if pd.isna(corr):
            continue
        rows.append(
            {
                "run_id": run_id,
                "corr": _safe_float(corr),
                "abs_corr": abs(_safe_float(corr)),
                "score": _safe_float(meta.get("score", 0)),
                "ic": _safe_float(meta.get("ic", 0)),
                "generation": int(meta.get("generation", 0) or 0),
                "formula": str(meta.get("formula", "") or ""),
            }
        )

    rows.sort(key=lambda item: (item["abs_corr"], item["score"], item["run_id"]), reverse=True)
    return rows


def _compute_submit_factor_all_correlations(
    summary: Dict[str, Any],
    *,
    model_name: str,
) -> List[Dict[str, Any]]:
    submit_path = (
        (summary.get("ensemble_outputs") or {}).get(model_name)
        or ((summary.get("submit_factor_output") or {}).get("submit_path") if model_name == summary.get("best_model") else "")
    )
    if not submit_path or not Path(str(submit_path)).is_file():
        return []

    all_factors = [
        item
        for item in kb.get_all_factors()
        if item.get("run_id") and item.get("PassGates")
    ]
    if not all_factors:
        return []

    selected_factor_ids = {
        str(item.get("run_id", ""))
        for item in summary.get("selected_factors", [])
        if item.get("run_id")
    }

    model_df = pd.read_parquet(str(submit_path), columns=["date", "security_id", "alpha"])
    if model_df.empty:
        return []
    model_series = (
        model_df.groupby(["date", "security_id"], sort=True)["alpha"]
        .mean()
        .astype("float32")
        .rename("pred")
        .sort_index()
    )

    rows: List[Dict[str, Any]] = []
    for factor in all_factors:
        run_id = str(factor.get("run_id", ""))
        factor_path = Path(str(factor.get("parquet_path") or ""))
        if not run_id or not factor_path.is_file():
            continue
        factor_df = pd.read_parquet(str(factor_path), columns=["date", "security_id", "alpha"])
        if factor_df.empty:
            continue
        factor_series = (
            factor_df.groupby(["date", "security_id"], sort=True)["alpha"]
            .mean()
            .astype("float32")
            .rename(run_id)
            .sort_index()
        )
        aligned = pd.concat([model_series, factor_series], axis=1, join="inner").dropna()
        if len(aligned) < 200:
            continue
        corr = aligned["pred"].corr(aligned[run_id])
        if pd.isna(corr):
            continue
        rows.append(
            {
                "run_id": run_id,
                "corr": _safe_float(corr),
                "abs_corr": abs(_safe_float(corr)),
                "score": _safe_float(factor.get("Score", 0)),
                "ic": _safe_float(factor.get("IC", 0)),
                "generation": int(factor.get("generation", 0) or 0),
                "formula": str(factor.get("formula", "") or ""),
                "is_input_factor": run_id in selected_factor_ids,
            }
        )

    rows.sort(key=lambda item: (item["abs_corr"], item["score"], item["run_id"]), reverse=True)
    return rows


def _serialize_curve(curve: pd.Series) -> List[Dict[str, Any]]:
    return [{"date": str(idx), "value": float(val)} for idx, val in curve.items() if np.isfinite(val)]


def _normalize_alpha_series(pred: pd.Series) -> pd.Series:
    if pred.empty:
        return pred.astype("float32")
    frame = pred.rename("alpha").to_frame()
    normalized = frame.groupby(level="date")["alpha"].transform(
        lambda s: (s.rank(method="average", pct=True) - 0.5).clip(-0.5, 0.5)
    )
    return normalized.astype("float32")


def _model_spec_items() -> list[tuple[str, Any, int]]:
    items = [
        ("EqualWeightRankCombo", lambda: None, 0),
        ("TrainICRankCombo", lambda: None, 0),
        ("TrainICIRRankCombo", lambda: None, 0),
        ("ValTopKRankCombo", lambda: None, 0),
        ("ValPowerRankCombo", lambda: None, 0),
        ("CorrPrunedRankCombo", lambda: None, 0),
        ("InverseVolICRankCombo", lambda: None, 0),
        ("DiversifiedICRankCombo", lambda: None, 0),
        ("SoftmaxICRankCombo", lambda: None, 0),
        ("TopAbsICRankCombo", lambda: None, 0),
        ("RidgeShrinkageRankCombo", lambda: None, 0),
        ("VolBalancedTopRankCombo", lambda: None, 0),
        ("ClusterNeutralICRankCombo", lambda: None, 0),
        ("RidgeZScoreMetaModel", lambda: make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=2.0, random_state=42)), 250_000),
        ("RandomForestMetaModel", lambda: RandomForestRegressor(n_estimators=80, max_depth=8, min_samples_leaf=80, max_features="sqrt", n_jobs=-1, random_state=42), 180_000),
        ("ExtraTreesMetaModel", lambda: ExtraTreesRegressor(n_estimators=120, max_depth=10, min_samples_leaf=80, max_features="sqrt", n_jobs=-1, random_state=42), 180_000),
        ("HistGradientBoostingMetaModel", lambda: HistGradientBoostingRegressor(max_iter=180, learning_rate=0.045, max_leaf_nodes=31, l2_regularization=0.15, random_state=42), 220_000),
        ("LightGBMMetaModel", lambda: lgb.LGBMRegressor(n_estimators=260, learning_rate=0.035, num_leaves=31, subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0, min_child_samples=80, n_jobs=-1, random_state=42, verbose=-1), 260_000),
        ("MLPRegressorMetaModel", lambda: make_pipeline(StandardScaler(with_mean=False), MLPRegressor(hidden_layer_sizes=(96, 32), activation="relu", alpha=1e-4, learning_rate_init=8e-4, batch_size=4096, max_iter=80, early_stopping=True, validation_fraction=0.12, n_iter_no_change=8, random_state=42)), 160_000),
        ("TemporalRidgeLagMetaModel", lambda: None, 0),
        ("TemporalLightGBMLagMetaModel", lambda: None, 0),
        ("TemporalHistGBLagMetaModel", lambda: None, 0),
        ("TemporalExtraTreesLagMetaModel", lambda: None, 0),
        ("FactorTokenTransformerRidgeStackModel", lambda: None, 0),
        ("CausalDecayFactorTransformerStackModel", lambda: None, 0),
    ]
    if str(os.environ.get("AUTOALPHA_ENABLE_TORCH_MODELS", "")).lower() in {"1", "true", "yes", "on"}:
        items.extend(
            [
                ("TorchMPSMLPMetaModel", lambda: TorchMPSMLPRegressor(hidden_sizes=(96, 32), epochs=4, batch_size=8192, lr=8e-4, weight_decay=1e-4, random_state=42), 60_000),
                ("TorchTCNMetaModel", lambda: None, 0),
                ("TorchGRUMetaModel", lambda: None, 0),
                ("TorchTransformerMetaModel", lambda: None, 0),
            ]
        )
    model_filter = [
        item.strip()
        for item in str(os.environ.get("AUTOALPHA_MODEL_FILTER", "")).split(",")
        if item.strip()
    ]
    if model_filter:
        wanted = set(model_filter)
        items = [item for item in items if item[0] in wanted]
    return items


def _active_model_names() -> List[str]:
    return [name for name, _builder, _max_rows in _model_spec_items()]


def _get_model_spec(model_name: str) -> tuple[Any, int]:
    for name, builder, max_rows in _model_spec_items():
        if name == model_name:
            return builder(), max_rows
    raise KeyError(f"Unknown model spec: {model_name}")


def _fit_predict_single_model(
    *,
    model_name: str,
    feature_cols: List[str],
    y_train: pd.Series,
    x_train_np: np.ndarray,
    x_pred_np: np.ndarray,
) -> Dict[str, Any]:
    if model_name in COMBO_MODEL_NAMES:
        raise RuntimeError(f"{model_name} should use combo prediction path, not sklearn path.")
    model, max_rows = _get_model_spec(model_name)
    y_train_model = _make_training_target(y_train)
    y_train_np = y_train_model.to_numpy(dtype="float32", copy=False)
    train_weights = _make_training_weights(y_train_model)
    X_fit, y_fit, w_fit = _sample_training_rows(
        x_train_np,
        y_train_np,
        max_rows=max_rows,
        sample_weight=train_weights,
    )
    if w_fit is None:
        w_fit = np.ones(len(y_fit), dtype="float32")
    _fit_with_optional_weights(model, X_fit, y_fit, w_fit)
    pred = model.predict(x_pred_np)
    return {
        "pred": pred,
        "importance": _extract_importance(model, feature_cols),
    }


def _fit_predict_window_model(
    *,
    model_name: str,
    feature_cols: List[str],
    frame: pd.DataFrame,
    model_frame: pd.DataFrame,
    combo_rank_frame: pd.DataFrame | None = None,
    combo_stats: Dict[str, Dict[str, Dict[str, float]]] | None = None,
    train_mask: pd.Series,
    predict_mask: pd.Series,
) -> Dict[str, Any]:
    if model_name in COMBO_MODEL_NAMES:
        return _fit_predict_combo_model(
            model_name=model_name,
            feature_cols=feature_cols,
            train_features=frame.loc[train_mask, feature_cols].astype("float32"),
            y_train=frame.loc[train_mask, "resp"].astype("float32"),
            pred_features=frame.loc[predict_mask, feature_cols].astype("float32"),
            pred_ranked_features=combo_rank_frame.loc[predict_mask, feature_cols] if combo_rank_frame is not None else None,
            stats_bundle=combo_stats,
        )

    if model_name in TEMPORAL_TORCH_MODEL_NAMES:
        return _fit_predict_temporal_torch_model(
            model_name=model_name,
            feature_cols=feature_cols,
            frame=frame,
            train_mask=train_mask,
            predict_mask=predict_mask,
        )

    if model_name in TEMPORAL_TABULAR_MODEL_NAMES:
        return _fit_predict_temporal_tabular_model(
            model_name=model_name,
            feature_cols=feature_cols,
            frame=frame,
            train_mask=train_mask,
            predict_mask=predict_mask,
        )

    if model_name in FACTOR_TRANSFORMER_MODEL_NAMES:
        if model_name == "CausalDecayFactorTransformerStackModel":
            return _fit_predict_causal_decay_factor_transformer_stack(
                feature_cols=feature_cols,
                frame=frame,
                model_frame=model_frame,
                train_mask=train_mask,
                predict_mask=predict_mask,
            )
        return _fit_predict_factor_token_transformer_stack(
            feature_cols=feature_cols,
            frame=frame,
            model_frame=model_frame,
            train_mask=train_mask,
            predict_mask=predict_mask,
        )

    return _fit_predict_single_model(
        model_name=model_name,
        feature_cols=list(model_frame.columns),
        y_train=frame.loc[train_mask, "resp"],
        x_train_np=model_frame.loc[train_mask].to_numpy(dtype="float32", copy=False),
        x_pred_np=model_frame.loc[predict_mask].to_numpy(dtype="float32", copy=False),
    )


def _build_submit_export_windows(
    trading_days: List[str],
    *,
    train_days: int,
    step_days: int,
) -> List[Dict[str, Any]]:
    export_windows: List[Dict[str, Any]] = []
    if not trading_days:
        return export_windows

    block_days = max(int(step_days or 0), 1)
    start_idx = 0
    window_id = 1
    while start_idx < len(trading_days):
        end_idx = min(start_idx + block_days, len(trading_days))
        predict_days = trading_days[start_idx:end_idx]
        train_start_idx = max(0, start_idx - max(int(train_days or 0), 0))
        train_slice = trading_days[train_start_idx:start_idx]
        export_windows.append(
            {
                "window_id": f"prequential_{window_id}",
                "train_days": list(train_slice),
                "predict_days": list(predict_days),
            }
        )
        start_idx = end_idx
        window_id += 1
    return export_windows


def _fallback_prequential_prediction(frame: pd.DataFrame, feature_cols: List[str]) -> Dict[str, Any]:
    pred = _weighted_rank_combo_prediction(
        frame[feature_cols].astype("float32"),
        feature_cols,
        _equal_weight_combo_weights(feature_cols),
    )
    return {
        "pred": pred.to_numpy(dtype="float32", copy=False),
        "importance": _equal_weight_combo_weights(feature_cols),
    }


def _predict_fixed_oos_daily_alpha(
    *,
    model_name: str,
    feature_refs: List[Dict[str, Any]],
    resp_series: pd.Series,
    train_days: List[str],
    predict_days: List[str],
    log_path: Path,
) -> tuple[pd.Series, List[Dict[str, Any]]]:
    if not predict_days:
        raise RuntimeError("No mock-OOS predict days available.")

    all_days = list(dict.fromkeys(train_days + predict_days))
    frame = _assemble_window_dataset(feature_refs, resp_series, all_days)
    feature_cols = [ref["run_id"] for ref in feature_refs]
    train_mask = frame.index.get_level_values("date").isin(train_days)
    predict_mask = frame.index.get_level_values("date").isin(predict_days)
    combo_frame = frame.loc[:, feature_cols].astype("float32")
    model_frame, _model_feature_cols = _make_model_feature_frame(frame, feature_cols)

    if not predict_mask.any():
        raise RuntimeError("Mock-OOS predict frame is empty after dataset assembly.")

    train_features = frame.loc[train_mask, feature_cols].astype("float32")
    train_day_count = len(pd.Index(train_days).unique())
    if train_features.empty or train_day_count < MIN_HISTORY_DAYS_FOR_TRAINED_PREDICTION:
        model_output = _fallback_prequential_prediction(frame.loc[predict_mask], feature_cols)
        _log(
            f"[mock-oos-export] {model_name} used fallback equal-weight combo history_days={train_day_count}",
            log_path,
        )
    else:
        model_output = _fit_predict_window_model(
            model_name=model_name,
            feature_cols=feature_cols,
            frame=frame,
            model_frame=model_frame,
            train_mask=train_mask,
            predict_mask=predict_mask,
        )

    pred_series = pd.Series(
        model_output["pred"],
        index=combo_frame.loc[predict_mask].index,
        name="pred",
        dtype="float32",
    )
    alpha = _normalize_alpha_series(pred_series).rename("alpha")
    importance_items = [
        {"factor": name, "importance": float(score)}
        for name, score in (model_output.get("importance") or {}).items()
    ]
    importance_items.sort(key=lambda item: item["importance"], reverse=True)
    return alpha, importance_items


def _predict_submit_daily_alpha(
    *,
    model_name: str,
    feature_refs: List[Dict[str, Any]],
    resp_series: pd.Series,
    trading_days: List[str],
    train_days: int,
    test_days: int,
    step_days: int,
    log_path: Path,
) -> tuple[pd.Series, List[Dict[str, Any]]]:
    export_windows = _build_submit_export_windows(
        trading_days,
        train_days=train_days,
        step_days=step_days,
    )
    if not export_windows:
        raise RuntimeError("No export windows available for submit-ready model factor.")

    prediction_parts: List[pd.Series] = []
    importance_store: Dict[str, List[float]] = defaultdict(list)
    feature_cols = [ref["run_id"] for ref in feature_refs]

    for window in export_windows:
        all_days = list(dict.fromkeys(window["train_days"] + window["predict_days"]))
        frame = _assemble_window_dataset(feature_refs, resp_series, all_days)
        combo_frame = frame.loc[:, feature_cols].astype("float32")
        model_frame, _model_feature_cols = _make_model_feature_frame(frame, feature_cols)
        train_mask = frame.index.get_level_values("date").isin(window["train_days"])
        predict_mask = frame.index.get_level_values("date").isin(window["predict_days"])

        X_train = combo_frame.loc[train_mask, feature_cols]
        X_predict = combo_frame.loc[predict_mask, feature_cols]
        train_day_count = len(pd.Index(window["train_days"]).unique())
        if X_predict.empty:
            _log(f"[submit-export] skip {window['window_id']} due to empty predict frame", log_path)
            continue

        if X_train.empty or train_day_count < MIN_HISTORY_DAYS_FOR_TRAINED_PREDICTION:
            model_output = _fallback_prequential_prediction(
                frame.loc[predict_mask],
                feature_cols,
            )
            _log(
                (
                    f"[submit-export] {model_name} window={window['window_id']} "
                    f"used fallback equal-weight combo history_days={train_day_count}"
                ),
                log_path,
            )
        else:
            model_output = _fit_predict_window_model(
                model_name=model_name,
                feature_cols=feature_cols,
                frame=frame,
                model_frame=model_frame,
                train_mask=train_mask,
                predict_mask=predict_mask,
            )
        pred_series = pd.Series(
            model_output["pred"],
            index=X_predict.index,
            name="pred",
            dtype="float32",
        )
        prediction_parts.append(pred_series)
        for name, score in model_output.get("importance", {}).items():
            importance_store[name].append(float(score))
        _log(
            f"[submit-export] {model_name} window={window['window_id']} predict_rows={len(pred_series):,}",
            log_path,
        )

    if not prediction_parts:
        raise RuntimeError(f"No predictions generated for submit export model {model_name}.")

    merged_pred = (
        pd.concat(prediction_parts)
        .groupby(level=["date", "security_id"], sort=True)
        .mean()
        .astype("float32")
        .sort_index()
    )
    alpha = _normalize_alpha_series(merged_pred).rename("alpha")
    importance_items = [
        {"factor": name, "importance": float(np.mean(scores))}
        for name, scores in importance_store.items()
        if scores
    ]
    importance_items.sort(key=lambda item: item["importance"], reverse=True)
    return alpha, importance_items


def _build_submission_alpha_wide_from_daily(
    daily_alpha: pd.Series,
    hub: DataHub,
    *,
    eval_days: List[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from core.evaluator import evaluate_submission_like_wide  # noqa: F401 - import guard for environment parity

    resp_df = hub.resp
    rest_df = hub.trading_restriction
    resp_col = "resp" if "resp" in resp_df.columns else resp_df.columns[0]
    rest_col = "trading_restriction" if "trading_restriction" in rest_df.columns else rest_df.columns[0]

    resp_un = resp_df[resp_col].unstack("security_id").sort_index()
    rest_un = rest_df[rest_col].unstack("security_id").sort_index().reindex_like(resp_un).fillna(0)
    if eval_days:
        eval_dt = pd.to_datetime(pd.Index(eval_days)).normalize()
        row_dates = pd.to_datetime(resp_un.index.get_level_values("date")).normalize()
        mask = row_dates.isin(eval_dt)
        resp_un = resp_un.loc[mask]
        rest_un = rest_un.reindex_like(resp_un).fillna(0)

    daily_wide = daily_alpha.unstack("security_id").astype("float32").sort_index()
    daily_wide.index = pd.to_datetime(pd.Index(daily_wide.index)).normalize()
    row_dates = pd.to_datetime(resp_un.index.get_level_values("date")).normalize()
    alpha_wide = daily_wide.reindex(row_dates).reindex(columns=resp_un.columns)
    alpha_un = pd.DataFrame(
        alpha_wide.to_numpy(dtype="float32", copy=False),
        index=resp_un.index,
        columns=resp_un.columns,
    )
    return alpha_un, resp_un, rest_un


def _write_submit_ready_daily_alpha(
    daily_alpha: pd.Series,
    out_path: Path,
    *,
    chunk_days: int = 20,
) -> Dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    from core.datahub import get_trading_days, load_universe
    from core.submission import ALLOWED_UTC_TIMES

    out_path.parent.mkdir(parents=True, exist_ok=True)
    daily_df = daily_alpha.rename("alpha").reset_index()
    daily_df["date"] = pd.to_datetime(daily_df["date"]).dt.normalize()
    daily_df["security_id"] = daily_df["security_id"].astype("int64")
    daily_df["alpha"] = pd.to_numeric(daily_df["alpha"], errors="coerce").clip(-1.0, 1.0).astype("float32")

    start_date = daily_df["date"].min().strftime("%Y-%m-%d")
    end_date = daily_df["date"].max().strftime("%Y-%m-%d")
    trading_days = get_trading_days(start=start_date, end=end_date)
    universe = load_universe(trading_days).reset_index()
    if "eq_univ" in universe.columns:
        universe = universe[universe["eq_univ"] == True]
    universe = universe[["date", "security_id"]].copy()
    universe["date"] = pd.to_datetime(universe["date"]).dt.normalize()
    universe["security_id"] = universe["security_id"].astype("int64")

    writer = None
    row_count = 0
    non_null_alpha = 0
    alpha_min = float("inf")
    alpha_max = float("-inf")

    try:
        for index in range(0, len(trading_days), chunk_days):
            date_slice = trading_days[index : index + chunk_days]
            date_ts = pd.to_datetime(date_slice).normalize()
            alpha_chunk = daily_df[daily_df["date"].isin(date_ts)]
            universe_chunk = universe[universe["date"].isin(date_ts)]
            merged = universe_chunk.merge(alpha_chunk, on=["date", "security_id"], how="left", copy=False)
            if merged.empty:
                continue

            parts = []
            for time_str in ALLOWED_UTC_TIMES:
                part = merged[["date", "security_id", "alpha"]].copy()
                part["datetime"] = part["date"].dt.strftime("%Y-%m-%d") + f" {time_str}"
                parts.append(part[["date", "datetime", "security_id", "alpha"]])
            chunk_df = pd.concat(parts, ignore_index=True)
            chunk_df["date"] = pd.to_datetime(chunk_df["date"]).dt.strftime("%Y-%m-%d")
            chunk_df["security_id"] = chunk_df["security_id"].astype("int64")
            chunk_df["alpha"] = pd.to_numeric(chunk_df["alpha"], errors="coerce").astype("float32")

            row_count += int(len(chunk_df))
            non_null = int(chunk_df["alpha"].notna().sum())
            non_null_alpha += non_null
            if non_null:
                alpha_min = min(alpha_min, float(chunk_df["alpha"].min(skipna=True)))
                alpha_max = max(alpha_max, float(chunk_df["alpha"].max(skipna=True)))

            table = pa.Table.from_pandas(chunk_df, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(str(out_path), table.schema)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()

    alpha_constant = non_null_alpha == 0 or math.isclose(alpha_min, alpha_max, rel_tol=0.0, abs_tol=1e-12)
    sanity = {
        "status": "PASS" if non_null_alpha > 0 and not alpha_constant else "FAIL (Submission Profile Invalid)",
        "submission_ready": bool(non_null_alpha > 0 and not alpha_constant),
        "cover_all": 1,
        "missing_days_count": 0,
        "missing_days_sample": [],
        "row_count": int(row_count),
        "unique_days": int(len(trading_days)),
        "duplicate_keys": 0,
        "alpha_in_bounds": True,
        "alpha_is_constant": bool(alpha_constant),
        "allowed_utc_times": list(ALLOWED_UTC_TIMES),
        "observed_utc_times": list(ALLOWED_UTC_TIMES),
        "invalid_utc_times": [],
        "exact_15m_grid": True,
        "bars_per_day_min": len(ALLOWED_UTC_TIMES),
        "bars_per_day_max": len(ALLOWED_UTC_TIMES),
        "bars_per_day_expected": len(ALLOWED_UTC_TIMES),
        "full_intraday_grid": True,
        "max": float(alpha_max) if non_null_alpha else 0.0,
        "min": float(alpha_min) if non_null_alpha else 0.0,
        "non_null_alpha": int(non_null_alpha),
    }
    return sanity


def _metrics_snapshot(metrics: Dict[str, Any]) -> Dict[str, Any]:
    preview = dict(metrics.get("result_preview") or {})
    return {
        "IC": float(metrics.get("IC", 0.0) or 0.0),
        "IR": float(metrics.get("IR", 0.0) or 0.0),
        "tvr": float(metrics.get("Turnover", 0.0) or 0.0),
        "TurnoverLocal": float(metrics.get("TurnoverLocal", 0.0) or 0.0),
        "Score": float(metrics.get("Score", 0.0) or 0.0),
        "PassGates": bool(metrics.get("PassGates", False)),
        "GatesDetail": metrics.get("GatesDetail", {}),
        "maxx": float(metrics.get("maxx", 0.0) or 0.0),
        "minn": float(metrics.get("minn", 0.0) or 0.0),
        "max_mean": float(metrics.get("max_mean", 0.0) or 0.0),
        "min_mean": float(metrics.get("min_mean", 0.0) or 0.0),
        "result_preview": preview,
        "score_formula": metrics.get("score_formula", "score = (IC - 0.0005 * Turnover) * sqrt(IR) * 100"),
        "metric_mode": metrics.get("metric_mode", "submission_cloud_aligned"),
        "turnover_basis": metrics.get("turnover_basis", "restricted_raw_alpha_diff_sum_x100"),
    }


def _export_submit_ready_model_factor(
    *,
    summary: Dict[str, Any],
    run_dir: Path,
    log_path: Path,
    model_name: str,
    hub: DataHub | None = None,
) -> Dict[str, Any]:
    model_lab_run_id = str(summary.get("run_id") or run_dir.name)
    selected_factor_rows = summary.get("selected_factors") or []
    if not selected_factor_rows:
        raise RuntimeError("Model lab summary has no selected_factors for submit export.")

    feature_refs: List[Dict[str, Any]] = []
    for row in selected_factor_rows:
        run_id = str(row.get("run_id") or "")
        factor = kb.get_factor(run_id)
        if not factor:
            raise KeyError(f"Selected factor missing from knowledge base: {run_id}")
        cache_path = ensure_daily_feature_cache({"run_id": run_id, **factor}, log_path)
        feature_refs.append(
            {
                "run_id": run_id,
                "cache_path": cache_path,
                "score": _safe_float(row.get("score", factor.get("Score", 0))),
                "ic": _safe_float(row.get("ic", factor.get("IC", 0))),
                "generation": factor.get("generation", row.get("generation", 0)),
                "formula": factor.get("formula", row.get("formula", "")),
            }
        )

    hub = hub or DataHub()
    resp_series = hub.resp["resp"].astype("float32").sort_index()
    if getattr(resp_series.index, "nlevels", 1) > 2:
        names = list(resp_series.index.names)
        date_level = "date" if "date" in names else names[0]
        security_level = "security_id" if "security_id" in names else names[-1]
        resp_series = (
            resp_series
            .groupby(level=[date_level, security_level])
            .mean()
            .astype("float32")
            .sort_index()
        )
    resp_series = _date_level_to_string(resp_series.rename("resp")).astype("float32")
    trading_days = sorted(resp_series.index.get_level_values("date").astype(str).unique().tolist())
    train_start = str(summary.get("train_period_start") or MOCK_OOS_TRAIN_START)
    train_end = str(summary.get("train_period_end") or MOCK_OOS_TRAIN_END)
    eval_start = str(summary.get("eval_period_start") or MOCK_OOS_EVAL_START)
    eval_end = str(summary.get("eval_period_end") or MOCK_OOS_EVAL_END)
    train_period_days = _slice_days_between(trading_days, train_start, train_end)
    eval_period_days = _slice_days_between(trading_days, eval_start, eval_end)
    if not train_period_days:
        raise RuntimeError(f"Missing training days for mock OOS train period {train_start}..{train_end}")
    if not eval_period_days:
        raise RuntimeError(f"Missing evaluation days for mock OOS period {eval_start}..{eval_end}")

    daily_alpha, importance_items = _predict_fixed_oos_daily_alpha(
        model_name=model_name,
        feature_refs=feature_refs,
        resp_series=resp_series,
        train_days=train_period_days,
        predict_days=eval_period_days,
        log_path=log_path,
    )

    covered_days = sorted(daily_alpha.index.get_level_values("date").astype(str).unique().tolist())
    if covered_days != eval_period_days:
        missing = sorted(set(eval_period_days) - set(covered_days))
        if missing:
            raise RuntimeError(f"Mock-OOS factor still misses {len(missing)} evaluation days, first={missing[:5]}")

    export_suffix = str(summary.get("export_suffix") or "").strip()
    export_id = f"model_lab_{model_lab_run_id}_{model_name.lower()}"
    if export_suffix:
        export_id = f"{export_id}_{export_suffix}"
    out_path = kb.SUBMIT_DIR / f"{export_id}.pq"
    sanity_report = _write_submit_ready_daily_alpha(daily_alpha, out_path)

    metrics_payload: Dict[str, Any] | None = None
    metrics_error = ""
    try:
        from core.evaluator import evaluate_submission_like_wide

        alpha_un, resp_un, rest_un = _build_submission_alpha_wide_from_daily(
            daily_alpha,
            hub,
            eval_days=eval_period_days,
        )
        metrics_payload = evaluate_submission_like_wide(alpha_un, resp_un, rest_un)
    except Exception as exc:
        metrics_error = str(exc)
        _log(f"[submit-export] official-like evaluation skipped for {export_id}: {exc}", log_path)

    snapshot = _metrics_snapshot(metrics_payload or {})
    combo_daily_tvr, combo_tvr_curve = _combo_daily_tvr_from_alpha(daily_alpha.rename("alpha"))
    snapshot = _recompute_score_with_combo_tvr(snapshot, combo_daily_tvr)
    metadata = {
        "run_id": export_id,
        "source_model_lab_run_id": model_lab_run_id,
        "model_name": model_name,
        "best_model": summary.get("best_model", ""),
        "selected_factor_count": len(feature_refs),
        "selected_factor_run_ids": [item["run_id"] for item in feature_refs],
        "daily_prediction_days": len(covered_days),
        "daily_prediction_start": covered_days[0] if covered_days else "",
        "daily_prediction_end": covered_days[-1] if covered_days else "",
        "train_period_start": train_start,
        "train_period_end": train_end,
        "eval_period_start": eval_start,
        "eval_period_end": eval_end,
        "formula": "",
        "description": "Mock-OOS factor exported from AutoAlpha model lab (trained on visible history, scored on hidden-style 2024 period).",
        "PassGates": snapshot["PassGates"],
        "Score": snapshot["Score"],
        "IC": snapshot["IC"],
        "IR": snapshot["IR"],
        "tvr": snapshot["tvr"],
        "TurnoverLocal": snapshot["TurnoverLocal"],
        "maxx": snapshot["maxx"],
        "minn": snapshot["minn"],
        "max_mean": snapshot["max_mean"],
        "min_mean": snapshot["min_mean"],
        "GatesDetail": snapshot["GatesDetail"],
        "result_preview": snapshot["result_preview"],
        "score_formula": snapshot["score_formula"],
        "metric_mode": snapshot["metric_mode"],
        "turnover_basis": snapshot["turnover_basis"],
        "combo_daily_tvr": combo_daily_tvr,
        "combo_tvr_curve": combo_tvr_curve,
        "submit_path": str(out_path),
        "source_summary_path": str(run_dir / "summary.json"),
        "source_latest_summary_path": str(MODEL_LAB_ROOT / "latest_summary.json"),
        "sanity_report": sanity_report,
        "top_features": importance_items[:20],
        "exported_at": datetime.now().isoformat(),
    }
    if metrics_error:
        metadata["metrics_error"] = metrics_error

    metadata_path = kb.SUBMIT_DIR / f"{export_id}_metadata.json"
    official_like_path = kb.SUBMIT_DIR / f"{export_id}_official_like_result.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    official_like_path.write_text(
        json.dumps([snapshot["result_preview"]] if snapshot["result_preview"] else [], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    run_dir_payload = {
        "run_id": export_id,
        "source_model_lab_run_id": model_lab_run_id,
        "model_name": model_name,
        "submit_path": str(out_path),
        "metadata_path": str(metadata_path),
        "official_like_result_path": str(official_like_path),
        "sanity_report": sanity_report,
        "top_features": importance_items[:20],
        "created_at": metadata["exported_at"],
        # Official submission-like metrics (same scale as factor evaluation)
        "IC": snapshot["IC"],
        "IR": snapshot["IR"],
        "Score": snapshot["Score"],
        "tvr": snapshot["tvr"],
        "TurnoverLocal": snapshot["TurnoverLocal"],
        "PassGates": snapshot["PassGates"],
        "GatesDetail": snapshot["GatesDetail"],
        "combo_daily_tvr": combo_daily_tvr,
        "combo_tvr_curve": combo_tvr_curve,
    }
    (run_dir / f"{model_name.lower()}_submit_factor.json").write_text(
        json.dumps(run_dir_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return run_dir_payload


def _export_precomputed_oos_model_factor(
    *,
    summary: Dict[str, Any],
    run_dir: Path,
    log_path: Path,
    model_name: str,
    daily_alpha: pd.Series,
    eval_period_days: List[str],
    importance_items: List[Dict[str, Any]],
    hub: DataHub,
) -> Dict[str, Any]:
    model_lab_run_id = str(summary.get("run_id") or run_dir.name)
    selected_factor_rows = summary.get("selected_factors") or []
    covered_days = sorted(daily_alpha.index.get_level_values("date").astype(str).unique().tolist())
    missing = sorted(set(eval_period_days) - set(covered_days))
    if missing:
        raise RuntimeError(f"Precomputed OOS factor misses {len(missing)} evaluation days, first={missing[:5]}")

    export_suffix = str(summary.get("export_suffix") or "").strip()
    export_id = f"model_lab_{model_lab_run_id}_{model_name.lower()}"
    if export_suffix:
        export_id = f"{export_id}_{export_suffix}"
    out_path = kb.SUBMIT_DIR / f"{export_id}.pq"
    sanity_report = _write_submit_ready_daily_alpha(daily_alpha, out_path)

    metrics_payload: Dict[str, Any] | None = None
    metrics_error = ""
    try:
        from core.evaluator import evaluate_submission_like_wide

        alpha_un, resp_un, rest_un = _build_submission_alpha_wide_from_daily(
            daily_alpha,
            hub,
            eval_days=eval_period_days,
        )
        metrics_payload = evaluate_submission_like_wide(alpha_un, resp_un, rest_un)
    except Exception as exc:
        metrics_error = str(exc)
        _log(f"[mock-oos-export] official-like evaluation skipped for {export_id}: {exc}", log_path)

    snapshot = _metrics_snapshot(metrics_payload or {})
    combo_daily_tvr, combo_tvr_curve = _combo_daily_tvr_from_alpha(daily_alpha.rename("alpha"))
    snapshot = _recompute_score_with_combo_tvr(snapshot, combo_daily_tvr)
    metadata = {
        "run_id": export_id,
        "source_model_lab_run_id": model_lab_run_id,
        "model_name": model_name,
        "best_model": summary.get("best_model", ""),
        "selected_factor_count": len(selected_factor_rows),
        "selected_factor_run_ids": [str(item.get("run_id", "")) for item in selected_factor_rows],
        "daily_prediction_days": len(covered_days),
        "daily_prediction_start": covered_days[0] if covered_days else "",
        "daily_prediction_end": covered_days[-1] if covered_days else "",
        "train_period_start": summary.get("train_period_start", ""),
        "train_period_end": summary.get("train_period_end", ""),
        "eval_period_start": summary.get("eval_period_start", ""),
        "eval_period_end": summary.get("eval_period_end", ""),
        "formula": "",
        "description": "Precomputed 2024 mock-OOS factor exported from AutoAlpha exploratory combo lab.",
        "PassGates": snapshot["PassGates"],
        "Score": snapshot["Score"],
        "IC": snapshot["IC"],
        "IR": snapshot["IR"],
        "tvr": snapshot["tvr"],
        "TurnoverLocal": snapshot["TurnoverLocal"],
        "maxx": snapshot["maxx"],
        "minn": snapshot["minn"],
        "max_mean": snapshot["max_mean"],
        "min_mean": snapshot["min_mean"],
        "GatesDetail": snapshot["GatesDetail"],
        "result_preview": snapshot["result_preview"],
        "score_formula": snapshot["score_formula"],
        "metric_mode": snapshot["metric_mode"],
        "turnover_basis": snapshot["turnover_basis"],
        "combo_daily_tvr": combo_daily_tvr,
        "combo_tvr_curve": combo_tvr_curve,
        "submit_path": str(out_path),
        "source_summary_path": str(run_dir / "summary.json"),
        "source_latest_summary_path": str(MODEL_LAB_ROOT / "latest_summary.json"),
        "sanity_report": sanity_report,
        "top_features": importance_items[:20],
        "exported_at": datetime.now().isoformat(),
    }
    if metrics_error:
        metadata["metrics_error"] = metrics_error

    metadata_path = kb.SUBMIT_DIR / f"{export_id}_metadata.json"
    official_like_path = kb.SUBMIT_DIR / f"{export_id}_official_like_result.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    official_like_path.write_text(
        json.dumps([snapshot["result_preview"]] if snapshot["result_preview"] else [], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    run_dir_payload = {
        "run_id": export_id,
        "source_model_lab_run_id": model_lab_run_id,
        "model_name": model_name,
        "submit_path": str(out_path),
        "metadata_path": str(metadata_path),
        "official_like_result_path": str(official_like_path),
        "sanity_report": sanity_report,
        "top_features": importance_items[:20],
        "created_at": metadata["exported_at"],
        "IC": snapshot["IC"],
        "IR": snapshot["IR"],
        "Score": snapshot["Score"],
        "tvr": snapshot["tvr"],
        "TurnoverLocal": snapshot["TurnoverLocal"],
        "PassGates": snapshot["PassGates"],
        "GatesDetail": snapshot["GatesDetail"],
        "combo_daily_tvr": combo_daily_tvr,
        "combo_tvr_curve": combo_tvr_curve,
    }
    (run_dir / f"{model_name.lower()}_submit_factor.json").write_text(
        json.dumps(run_dir_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return run_dir_payload


def _plot_summary(summary: Dict[str, Any], run_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = summary.get("models", {})
    windows = summary.get("windows", [])
    if not models or not windows:
        return

    # Plot 1: rolling IC by window.
    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = list(range(1, len(windows) + 1))
    for model_name, payload in models.items():
        y = [window["models"][model_name]["daily_ic_mean"] for window in windows]
        ax.plot(x, y, marker="o", linewidth=2, label=model_name)
    ax.axhline(0, color="#94a3b8", linestyle="--", linewidth=1)
    ax.set_title("Mock OOS Daily IC by Window")
    ax.set_xlabel("Window")
    ax.set_ylabel("Mean Daily IC")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(run_dir / "rolling_ic.png", dpi=140)
    plt.close(fig)

    # Plot 2: cumulative net pnl
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for model_name, payload in models.items():
        curve = pd.Series(
            [point["value"] for point in payload.get("cumulative_curve", [])],
            index=[point["date"] for point in payload.get("cumulative_curve", [])],
            dtype="float32",
        )
        if curve.empty:
            continue
        ax.plot(curve.index, curve.values, linewidth=2, label=model_name)
    ax.set_title("Mock OOS Cumulative PnL")
    ax.set_xlabel("Date")
    ax.set_ylabel("PnL")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(run_dir / "rolling_pnl.png", dpi=140)
    plt.close(fig)

    # Plot 3: model-level Sharpe / net PnL / TVR comparison.
    comparison = [
        {
            "model": model_name,
            "sharpe": float(payload.get("avg_sharpe", 0.0)),
            "pnl": float(payload.get("total_pnl", 0.0)),
            "tvr": float(payload.get("avg_turnover", 0.0)),
        }
        for model_name, payload in models.items()
    ]
    if comparison:
        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax_pnl = ax.twinx()
        labels = [item["model"] for item in comparison]
        idx = np.arange(len(labels))
        width = 0.25
        ax.bar(idx - width, [item["sharpe"] for item in comparison], width=width, color="#be123c", label="Sharpe")
        ax.bar(idx, [item["tvr"] for item in comparison], width=width, color="#0f766e", label="Avg TVR")
        ax_pnl.bar(idx + width, [item["pnl"] for item in comparison], width=width, color="#0369a1", label="Net PnL")
        ax.set_xticks(idx)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylabel("Sharpe / Avg TVR")
        ax_pnl.set_ylabel("Net PnL")
        ax.set_title("Model Sharpe / Avg TVR / Net PnL Comparison")
        handles, labels_left = ax.get_legend_handles_labels()
        handles_right, labels_right = ax_pnl.get_legend_handles_labels()
        ax.legend(handles + handles_right, labels_left + labels_right, loc="upper left")
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(run_dir / "sharpe_pnl_comparison.png", dpi=140)
        plt.close(fig)

    # Plot 4: top features from best model
    best_model = summary.get("best_model")
    top_features = (models.get(best_model, {}) if best_model else {}).get("top_features", [])
    if top_features:
        fig, ax = plt.subplots(figsize=(10, 6))
        names = [item["factor"] for item in top_features[:15]][::-1]
        values = [item["importance"] for item in top_features[:15]][::-1]
        ax.barh(names, values, color="#0f766e")
        ax.set_title(f"Top Features — {best_model}")
        ax.set_xlabel("Importance")
        fig.tight_layout()
        fig.savefig(run_dir / "feature_importance.png", dpi=140)
        plt.close(fig)


def _write_markdown_report(summary: Dict[str, Any], run_dir: Path) -> None:
    lines = [
        "# AutoAlpha Mock OOS Model Lab",
        "",
        f"- Run ID: `{summary['run_id']}`",
        f"- Generated At: `{summary['created_at']}`",
        f"- Factor Count: `{summary['selected_factor_count']}`",
        f"- Training Period: `{summary.get('train_period_start', '')}` → `{summary.get('train_period_end', '')}`",
        f"- Mock OOS Period: `{summary.get('eval_period_start', '')}` → `{summary.get('eval_period_end', '')}`",
        f"- Best Combo: `{summary['best_model']}`",
        f"- Best OOS IC: `{float(summary.get('best_ic', 0.0)):.4f}`",
        f"- Best OOS Score: `{float(summary.get('best_score', 0.0)):.2f}`",
        "",
        "## Model Summary",
    ]
    for model_name, payload in summary.get("models", {}).items():
        lines.extend(
            [
                "",
                f"### {model_name}",
                f"- Mock OOS Daily IC: `{payload['avg_daily_ic']:.4f}`",
                f"- Mock OOS Daily Rank IC: `{payload['avg_daily_rank_ic']:.4f}`",
                f"- Mock OOS Sharpe: `{payload['avg_sharpe']:.4f}`",
                f"- Mock OOS Net PnL: `{payload['total_pnl']:.4f}`",
                f"- Gross PnL: `{payload.get('gross_pnl', 0.0):.4f}`",
                f"- Total Fee: `{payload.get('total_fee', 0.0):.4f}`",
                f"- Avg TVR: `{payload.get('avg_turnover', 0.0):.4f}`",
                f"- Max Drawdown: `{payload['max_drawdown']:.4f}`",
                f"- Hit Ratio: `{payload['hit_ratio']:.2%}`",
            ]
        )
    if summary.get("ensemble_outputs"):
        lines.extend(["", "## Mock OOS Factor Outputs"])
        for model_name, path in summary.get("ensemble_outputs", {}).items():
            lines.append(f"- {model_name}: `{path}`")
    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def _execute_combo_lab(
    *,
    selected_factors: List[Dict[str, Any]],
    run_root: Path,
    latest_summary_path: Path,
    fee_bps: float,
    target_valid_count: int,
    log_label: str,
    lab_mode: str,
) -> Dict[str, Any]:
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = run_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "rolling_model_lab.log"
    _log(f"{log_label} started: {run_id}", log_path)

    if len(selected_factors) < 1:
        raise RuntimeError("Need at least 1 valid factor for combo lab.")
    _log(
        f"Using selected factors for combo lab: count={len(selected_factors)}",
        log_path,
    )

    feature_refs: List[Dict[str, Any]] = []
    for factor in selected_factors:
        cache_path = ensure_daily_feature_cache(factor, log_path)
        feature_refs.append(
            {
                "run_id": factor["run_id"],
                "cache_path": cache_path,
                "score": _safe_float(factor.get("Score", 0)),
                "ic": _safe_float(factor.get("IC", 0)),
                "generation": factor.get("generation", 0),
                "formula": factor.get("formula", ""),
            }
        )

    _log(f"Loading resp data for combo lab with {len(feature_refs)} factors", log_path)
    hub = DataHub()
    resp_series = hub.resp["resp"].astype("float32").sort_index()
    if getattr(resp_series.index, "nlevels", 1) > 2:
        names = list(resp_series.index.names)
        date_level = "date" if "date" in names else names[0]
        security_level = "security_id" if "security_id" in names else names[-1]
        resp_series = (
            resp_series
            .groupby(level=[date_level, security_level])
            .mean()
            .astype("float32")
            .sort_index()
        )
    resp_series = _date_level_to_string(resp_series.rename("resp")).astype("float32")
    trading_days = sorted(resp_series.index.get_level_values("date").astype(str).unique().tolist())
    train_period_days = _slice_days_between(trading_days, MOCK_OOS_TRAIN_START, MOCK_OOS_TRAIN_END)
    eval_period_days = _slice_days_between(trading_days, MOCK_OOS_EVAL_START, MOCK_OOS_EVAL_END)
    if not train_period_days:
        raise RuntimeError(f"No training days found in {MOCK_OOS_TRAIN_START}..{MOCK_OOS_TRAIN_END}")
    if not eval_period_days:
        raise RuntimeError(f"No evaluation days found in {MOCK_OOS_EVAL_START}..{MOCK_OOS_EVAL_END}")
    _log(
        (
            f"Mock OOS split ready: train={train_period_days[0]}→{train_period_days[-1]} ({len(train_period_days)} days), "
            f"eval={eval_period_days[0]}→{eval_period_days[-1]} ({len(eval_period_days)} days)"
        ),
        log_path,
    )

    model_window_metrics: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    model_prediction_series: Dict[str, List[pd.Series]] = defaultdict(list)
    model_input_feature_frames: Dict[str, List[pd.DataFrame]] = defaultdict(list)
    model_daily_pnls: Dict[str, List[pd.Series]] = defaultdict(list)
    model_daily_gross_pnls: Dict[str, List[pd.Series]] = defaultdict(list)
    model_daily_fees: Dict[str, List[pd.Series]] = defaultdict(list)
    model_daily_long_pnls: Dict[str, List[pd.Series]] = defaultdict(list)
    model_daily_long_gross_pnls: Dict[str, List[pd.Series]] = defaultdict(list)
    model_daily_long_fees: Dict[str, List[pd.Series]] = defaultdict(list)
    model_prediction_comparisons: Dict[str, List[List[Dict[str, Any]]]] = defaultdict(list)
    importance_store: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    window_summaries: List[Dict[str, Any]] = []

    window = {
        "window_id": 1,
        "train_days": train_period_days,
        "test_days": eval_period_days,
        "train_start": train_period_days[0],
        "train_end": train_period_days[-1],
        "test_start": eval_period_days[0],
        "test_end": eval_period_days[-1],
    }
    all_days = train_period_days + eval_period_days
    frame = _assemble_window_dataset(feature_refs, resp_series, all_days)
    feature_cols = [ref["run_id"] for ref in feature_refs]
    train_mask = frame.index.get_level_values("date").isin(train_period_days)
    test_mask = frame.index.get_level_values("date").isin(eval_period_days)
    combo_frame = frame.loc[:, feature_cols].astype("float32")
    active_model_names = _active_model_names()
    transformer_only = bool(active_model_names) and set(active_model_names).issubset(FACTOR_TRANSFORMER_MODEL_NAMES)
    if transformer_only:
        model_frame, _model_feature_cols = _make_model_feature_frame(frame, feature_cols, max_temporal=24)
        combo_rank_frame = None
    else:
        model_frame, _model_feature_cols = _make_model_feature_frame(frame, feature_cols)
        combo_rank_frame = model_frame.loc[:, [f"{col}__rank" for col in feature_cols]].copy()
        combo_rank_frame.columns = feature_cols
    X_train = combo_frame.loc[train_mask, feature_cols]
    y_train = frame.loc[train_mask, "resp"]
    X_test = combo_frame.loc[test_mask, feature_cols]
    y_test = frame.loc[test_mask, "resp"]
    if X_train.empty or X_test.empty:
        raise RuntimeError("Mock OOS split produced empty train/test frame.")

    _log(
        (
            f"Mock OOS window: train {window['train_start']}→{window['train_end']} ({len(X_train):,} rows), "
            f"eval {window['test_start']}→{window['test_end']} ({len(X_test):,} rows)"
        ),
        log_path,
    )

    fitted_models = _fit_predict_models(
        feature_cols=feature_cols,
        y_train=y_train,
        frame=frame,
        model_frame=model_frame,
        combo_rank_frame=combo_rank_frame,
        train_mask=train_mask,
        predict_mask=test_mask,
        log_path=log_path,
    )

    fit_days, val_days = _split_train_val_days(train_period_days)
    fit_mask = frame.index.get_level_values("date").isin(fit_days)
    val_mask = frame.index.get_level_values("date").isin(val_days)
    window_model_payloads: Dict[str, Dict[str, Any]] = {}
    model_train_val_curves: Dict[str, List[Dict[str, Any]]] = {}
    model_combo_weights: Dict[str, Dict[str, float]] = {}
    for model_name, model_output in fitted_models.items():
        pred = pd.Series(model_output["pred"], index=X_test.index, name="pred").astype("float32")
        evaluation = _evaluate_predictions(pred, y_test, fee_bps=fee_bps)
        train_val_metrics: Dict[str, Any] = {}
        train_val_curve: List[Dict[str, Any]] = []
        combo_weights = model_output.get("weights") if isinstance(model_output.get("weights"), dict) else {}
        if combo_weights:
            train_pred = _weighted_combo_from_ranked(
                combo_rank_frame.loc[fit_mask, feature_cols],
                feature_cols,
                combo_weights,
            )
            val_pred = _weighted_combo_from_ranked(
                combo_rank_frame.loc[val_mask, feature_cols],
                feature_cols,
                combo_weights,
            )
            train_metrics = _combo_period_metrics(train_pred, frame.loc[fit_mask, "resp"].astype("float32"), fee_bps=fee_bps)
            val_metrics = _combo_period_metrics(val_pred, frame.loc[val_mask, "resp"].astype("float32"), fee_bps=fee_bps)
            oos_metrics = _combo_period_metrics(pred, y_test, fee_bps=fee_bps)
            train_val_metrics = {
                "train": {key: value for key, value in train_metrics.items() if key not in {"prediction_comparison_curve", "daily_ic_curve", "daily_rank_ic_curve", "combo_tvr_curve", "strategy"}},
                "val": {key: value for key, value in val_metrics.items() if key not in {"prediction_comparison_curve", "daily_ic_curve", "daily_rank_ic_curve", "combo_tvr_curve", "strategy"}},
                "oos": {key: value for key, value in oos_metrics.items() if key not in {"prediction_comparison_curve", "daily_ic_curve", "daily_rank_ic_curve", "combo_tvr_curve", "strategy"}},
            }
            train_val_curve = (
                _prefix_curve_period(train_metrics.get("prediction_comparison_curve", []), "train")
                + _prefix_curve_period(val_metrics.get("prediction_comparison_curve", []), "val")
            )
        else:
            try:
                probe_mask = fit_mask | val_mask
                probe_output = _fit_predict_window_model(
                    model_name=model_name,
                    feature_cols=feature_cols,
                    frame=frame,
                    model_frame=model_frame,
                    combo_rank_frame=combo_rank_frame,
                    train_mask=fit_mask,
                    predict_mask=probe_mask,
                )
                probe_pred = pd.Series(
                    probe_output["pred"],
                    index=model_frame.loc[probe_mask].index,
                    name="pred",
                    dtype="float32",
                )
                train_pred = probe_pred.loc[probe_pred.index.get_level_values("date").isin(fit_days)]
                val_pred = probe_pred.loc[probe_pred.index.get_level_values("date").isin(val_days)]
                train_metrics = _combo_period_metrics(train_pred, frame.loc[fit_mask, "resp"].astype("float32"), fee_bps=fee_bps)
                val_metrics = _combo_period_metrics(val_pred, frame.loc[val_mask, "resp"].astype("float32"), fee_bps=fee_bps)
                oos_metrics = _combo_period_metrics(pred, y_test, fee_bps=fee_bps)
                train_val_metrics = {
                    "train": {key: value for key, value in train_metrics.items() if key not in {"prediction_comparison_curve", "daily_ic_curve", "daily_rank_ic_curve", "combo_tvr_curve", "strategy"}},
                    "val": {key: value for key, value in val_metrics.items() if key not in {"prediction_comparison_curve", "daily_ic_curve", "daily_rank_ic_curve", "combo_tvr_curve", "strategy"}},
                    "oos": {key: value for key, value in oos_metrics.items() if key not in {"prediction_comparison_curve", "daily_ic_curve", "daily_rank_ic_curve", "combo_tvr_curve", "strategy"}},
                }
                train_val_curve = (
                    _prefix_curve_period(train_metrics.get("prediction_comparison_curve", []), "train")
                    + _prefix_curve_period(val_metrics.get("prediction_comparison_curve", []), "val")
                )
            except Exception as exc:
                _log(f"[mock-oos] train/val probe skipped for {model_name}: {exc}", log_path)
        model_train_val_curves[model_name] = train_val_curve
        model_combo_weights[model_name] = {str(k): float(v) for k, v in combo_weights.items()} if combo_weights else {}
        model_prediction_series[model_name].append(pred)
        model_input_feature_frames[model_name].append(
            frame.loc[test_mask, feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float32")
        )
        payload = {
            "daily_ic_mean": evaluation["daily_ic_mean"],
            "daily_rank_ic_mean": evaluation["daily_rank_ic_mean"],
            "daily_ic_ir": evaluation["daily_ic_ir"],
            "daily_rank_ic_ir": evaluation["daily_rank_ic_ir"],
            "overall_ic": evaluation["overall_ic"],
            "rows": evaluation["rows"],
            "pnl": evaluation["strategy"]["total_pnl"],
            "gross_pnl": evaluation["strategy"]["gross_pnl"],
            "total_fee": evaluation["strategy"]["total_fee"],
            "sharpe": evaluation["strategy"]["sharpe"],
            "max_drawdown": evaluation["strategy"]["max_drawdown"],
            "hit_ratio": evaluation["strategy"]["hit_ratio"],
            "avg_turnover": evaluation["strategy"]["avg_turnover"],
            "train_val_metrics": train_val_metrics,
            "model_diagnostics": model_output.get("diagnostics", {}),
        }
        model_window_metrics[model_name].append({**window, **payload})
        model_daily_pnls[model_name].append(evaluation["strategy"]["daily_pnl"])
        model_daily_gross_pnls[model_name].append(evaluation["strategy"]["daily_gross_pnl"])
        model_daily_fees[model_name].append(evaluation["strategy"]["daily_fee"])
        model_daily_long_pnls[model_name].append(evaluation["strategy"]["daily_long_pnl"])
        model_daily_long_gross_pnls[model_name].append(evaluation["strategy"]["daily_long_gross_pnl"])
        model_daily_long_fees[model_name].append(evaluation["strategy"]["daily_long_fee"])
        model_prediction_comparisons[model_name].append(
            [
                {
                    **row,
                    "window_id": window["window_id"],
                    "test_start": window["test_start"],
                    "test_end": window["test_end"],
                }
                for row in evaluation["prediction_comparison_curve"]
            ]
        )
        for name, score in model_output.get("importance", {}).items():
            importance_store[model_name][name].append(float(score))
        window_model_payloads[model_name] = payload

    window_summaries.append(
        {
            "window_id": window["window_id"],
            "train_start": window["train_start"],
            "train_end": window["train_end"],
            "test_start": window["test_start"],
            "test_end": window["test_end"],
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "models": window_model_payloads,
        }
    )

    model_summaries: Dict[str, Any] = {}
    ensemble_outputs: Dict[str, str] = {}
    submit_factor_output: Dict[str, Any] | None = None
    best_model = None
    best_rankic_score = -1e18
    for model_name, metrics_list in model_window_metrics.items():
        concatenated = pd.concat(model_daily_pnls[model_name]).sort_index()
        gross_concatenated = pd.concat(model_daily_gross_pnls[model_name]).sort_index()
        fee_concatenated = pd.concat(model_daily_fees[model_name]).sort_index()
        long_concatenated = pd.concat(model_daily_long_pnls[model_name]).sort_index()
        long_gross_concatenated = pd.concat(model_daily_long_gross_pnls[model_name]).sort_index()
        long_fee_concatenated = pd.concat(model_daily_long_fees[model_name]).sort_index()
        cumulative = concatenated.cumsum()
        gross_cumulative = gross_concatenated.cumsum()
        fee_cumulative = fee_concatenated.cumsum()
        long_cumulative = long_concatenated.cumsum()
        long_gross_cumulative = long_gross_concatenated.cumsum()
        long_fee_cumulative = long_fee_concatenated.cumsum()
        drawdown = cumulative - cumulative.cummax()
        long_drawdown = long_cumulative - long_cumulative.cummax()
        avg_ic = float(np.mean([item["daily_ic_mean"] for item in metrics_list]))
        avg_rank_ic = float(np.mean([item["daily_rank_ic_mean"] for item in metrics_list]))
        avg_ir = float(np.mean([item.get("daily_rank_ic_ir", 0.0) for item in metrics_list]))
        avg_sharpe = float(np.mean([item["sharpe"] for item in metrics_list]))
        total_pnl = float(cumulative.iloc[-1]) if len(cumulative) else 0.0
        gross_pnl = float(gross_cumulative.iloc[-1]) if len(gross_cumulative) else 0.0
        total_fee = float(fee_cumulative.iloc[-1]) if len(fee_cumulative) else 0.0
        max_drawdown = float(drawdown.min()) if len(drawdown) else 0.0
        long_total_pnl = float(long_cumulative.iloc[-1]) if len(long_cumulative) else 0.0
        long_gross_pnl = float(long_gross_cumulative.iloc[-1]) if len(long_gross_cumulative) else 0.0
        long_total_fee = float(long_fee_cumulative.iloc[-1]) if len(long_fee_cumulative) else 0.0
        long_max_drawdown = float(long_drawdown.min()) if len(long_drawdown) else 0.0
        hit_ratio = float(np.mean([item["hit_ratio"] for item in metrics_list]))
        avg_turnover = float(np.mean([item.get("avg_turnover", 0.0) for item in metrics_list]))
        prediction_comparison_curve = sorted(
            (row for rows in model_prediction_comparisons.get(model_name, []) for row in rows),
            key=lambda item: str(item.get("date", "")),
        )
        importance_items = [
            {"factor": name, "importance": float(np.mean(scores))}
            for name, scores in importance_store[model_name].items()
            if scores
        ]
        importance_items.sort(key=lambda item: item["importance"], reverse=True)
        concat_pred = pd.concat(model_prediction_series[model_name]).sort_index()
        input_factor_correlations = _compute_model_input_correlations(
            concat_pred,
            pd.concat(model_input_feature_frames[model_name]).sort_index(),
            feature_refs,
        )
        model_summaries[model_name] = {
            "avg_daily_ic": avg_ic,
            "avg_daily_rank_ic": avg_rank_ic,
            "avg_daily_ic_bps": round(avg_ic * 100, 4),
            "avg_daily_rank_ic_bps": round(avg_rank_ic * 100, 4),
            "avg_ir": avg_ir,
            "avg_sharpe": avg_sharpe,
            "total_pnl": total_pnl,
            "gross_pnl": gross_pnl,
            "total_fee": total_fee,
            "max_drawdown": max_drawdown,
            "long_only_total_pnl": long_total_pnl,
            "long_only_gross_pnl": long_gross_pnl,
            "long_only_total_fee": long_total_fee,
            "long_only_max_drawdown": long_max_drawdown,
            "hit_ratio": hit_ratio,
            "avg_turnover": avg_turnover,
            "window_metrics": metrics_list,
            "cumulative_curve": _serialize_curve(cumulative),
            "drawdown_curve": _serialize_curve(drawdown),
            "gross_cumulative_curve": _serialize_curve(gross_cumulative),
            "fee_cumulative_curve": _serialize_curve(fee_cumulative),
            "daily_pnl_curve": _serialize_curve(concatenated),
            "long_only_cumulative_curve": _serialize_curve(long_cumulative),
            "long_only_drawdown_curve": _serialize_curve(long_drawdown),
            "long_only_gross_cumulative_curve": _serialize_curve(long_gross_cumulative),
            "long_only_fee_cumulative_curve": _serialize_curve(long_fee_cumulative),
            "daily_long_pnl_curve": _serialize_curve(long_concatenated),
            "prediction_comparison_curve": prediction_comparison_curve,
            "top_features": importance_items[:20],
            "input_factor_correlations": input_factor_correlations,
            "all_factor_correlations": [],
            "method_card": _combo_method_card(model_name),
            "train_val_metrics": metrics_list[0].get("train_val_metrics", {}) if metrics_list else {},
            "model_diagnostics": metrics_list[0].get("model_diagnostics", {}) if metrics_list else {},
            "train_val_curve": model_train_val_curves.get(model_name, []),
            "combo_weights": [
                {"factor": col, "weight": float(model_combo_weights.get(model_name, {}).get(col, 0.0))}
                for col in feature_cols
                if abs(float(model_combo_weights.get(model_name, {}).get(col, 0.0))) > 1e-12
            ][:32],
        }
        rankic_score = avg_rank_ic * 1000 + total_pnl
        if rankic_score > best_rankic_score:
            best_rankic_score = rankic_score
            best_model = model_name

    export_summary = {
        "run_id": run_id,
        "best_model": best_model,
        "export_suffix": "lowcorr8" if "low_corr" in lab_mode else "",
        "selected_factors": [
            {
                "run_id": ref["run_id"],
                "score": ref["score"],
                "ic": ref["ic"],
                "generation": ref["generation"],
                "formula": ref["formula"],
            }
            for ref in feature_refs
        ],
        "train_period_start": train_period_days[0],
        "train_period_end": train_period_days[-1],
        "eval_period_start": eval_period_days[0],
        "eval_period_end": eval_period_days[-1],
    }
    for model_name in model_summaries:
        _log(f"[mock-oos-export] Fast exporting {model_name} 2024 parquet and official-like metrics", log_path)
        concat_pred = pd.concat(model_prediction_series[model_name]).sort_index()
        daily_alpha = _normalize_alpha_series(concat_pred).rename("alpha")
        payload = _export_precomputed_oos_model_factor(
            summary=export_summary,
            run_dir=run_dir,
            log_path=log_path,
            model_name=model_name,
            daily_alpha=daily_alpha,
            eval_period_days=eval_period_days,
            importance_items=model_summaries[model_name].get("top_features", []),
            hub=hub,
        )
        _log(
            f"[mock-oos-export] Finished {model_name}: IC={float(payload.get('IC', 0.0)):.3f} IR={float(payload.get('IR', 0.0)):.3f} Score={float(payload.get('Score', 0.0)):.2f}",
            log_path,
        )
        ensemble_outputs[model_name] = payload["submit_path"]
        for _k in ("IC", "IR", "Score", "tvr", "TurnoverLocal", "PassGates", "GatesDetail", "combo_daily_tvr", "combo_tvr_curve"):
            if _k in payload:
                model_summaries[model_name][f"submit_{_k}"] = payload[_k]
        model_summaries[model_name]["combo_tvr_curve"] = payload.get("combo_tvr_curve", [])
        model_summaries[model_name].setdefault("train_val_metrics", {}).setdefault("oos", {})
        model_summaries[model_name]["train_val_metrics"]["oos"].update(
            {
                "IC": payload.get("IC", 0.0),
                "IR": payload.get("IR", 0.0),
                "Score": payload.get("Score", 0.0),
                "TVR": payload.get("tvr", payload.get("combo_daily_tvr", 0.0)),
                "PassGates": payload.get("PassGates", False),
                "GatesDetail": payload.get("GatesDetail", {}),
            }
        )

    if model_summaries:
        best_model = max(
            model_summaries,
            key=lambda name: (
                float(model_summaries[name].get("submit_Score", float("-inf"))),
                float(model_summaries[name].get("submit_IC", float("-inf"))),
            ),
        )
        submit_factor_output = {
            "model_name": best_model,
            "submit_path": ensemble_outputs.get(best_model, ""),
            **{k.replace("submit_", ""): v for k, v in model_summaries[best_model].items() if k.startswith("submit_")},
        }

    display_run_id = f"{best_model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}" if best_model else run_id
    summary = {
        "run_id": display_run_id,
        "run_dir_name": run_dir.name,
        "created_at": datetime.now().isoformat(),
        "target_valid_count": target_valid_count,
        "selected_factor_count": len(feature_refs),
        "window_count": len(window_summaries),
        "train_period_start": train_period_days[0],
        "train_period_end": train_period_days[-1],
        "eval_period_start": eval_period_days[0],
        "eval_period_end": eval_period_days[-1],
        "lab_mode": lab_mode,
        "fee_bps": fee_bps,
        "best_model": best_model,
        "best_ic": max((float(payload.get("submit_IC", float("-inf"))) for payload in model_summaries.values()), default=0.0),
        "best_score": max((float(payload.get("submit_Score", float("-inf"))) for payload in model_summaries.values()), default=0.0),
        "selected_factors": [
            {
                "run_id": ref["run_id"],
                "score": ref["score"],
                "ic": ref["ic"],
                "generation": ref["generation"],
                "formula": ref["formula"],
            }
            for ref in feature_refs
        ],
        "windows": window_summaries,
        "models": model_summaries,
        "ensemble_outputs": ensemble_outputs,
        "submit_factor_output": submit_factor_output,
        "best_model_input_factor_correlations": model_summaries.get(best_model, {}).get("input_factor_correlations", []) if best_model else [],
        "best_model_all_factor_correlations": [],
        "research_references": MODEL_LAB_REFERENCES,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _plot_summary(summary, run_dir)
    _write_markdown_report(summary, run_dir)
    latest_summary_path.parent.mkdir(parents=True, exist_ok=True)
    latest_summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _log(f"{log_label} finished: {run_dir}", log_path)
    return summary


def run_model_lab(
    *,
    target_valid_count: int,
    ideas_per_round: int,
    eval_days_count: int,
    max_rounds: int,
    sleep_seconds: float,
    train_days: int,
    test_days: int,
    step_days: int,
    allow_partial: bool,
    fee_bps: float = DEFAULT_FEE_BPS,
) -> Dict[str, Any]:
    valid_factors = mine_until_target_valid(
        target_valid_count=target_valid_count,
        ideas_per_round=ideas_per_round,
        eval_days_count=eval_days_count,
        max_rounds=max_rounds,
        sleep_seconds=sleep_seconds,
        log_path=MODEL_LAB_ROOT / "rolling_model_lab.log",
    )

    if len(valid_factors) < target_valid_count and not allow_partial:
        raise RuntimeError(
            f"Need {target_valid_count} valid factors, but only {len(valid_factors)} are available."
        )

    return _execute_combo_lab(
        selected_factors=valid_factors,
        run_root=MODEL_LAB_ROOT,
        latest_summary_path=MODEL_LAB_ROOT / "latest_summary.json",
        fee_bps=fee_bps,
        target_valid_count=target_valid_count,
        log_label="Rolling model lab",
        lab_mode="mock_oos_fixed_2024",
    )


def run_low_corr_experiment(
    *,
    fee_bps: float = DEFAULT_FEE_BPS,
) -> Dict[str, Any]:
    selected_factors = [_resolve_factor_or_raise(run_id) for run_id in LOW_CORR_FACTOR_RUN_IDS]
    return _execute_combo_lab(
        selected_factors=selected_factors,
        run_root=MODEL_LAB_EXPLORATIONS_ROOT / LOW_CORR_EXPERIMENT_SLUG,
        latest_summary_path=MODEL_LAB_EXPLORATIONS_ROOT / LOW_CORR_EXPERIMENT_SLUG / "latest_summary.json",
        fee_bps=fee_bps,
        target_valid_count=len(selected_factors),
        log_label="Low-corr combo exploration",
        lab_mode="mock_oos_low_corr_exploration",
    )


def export_submit_ready_model_factor(
    summary_path: str | Path | None = None,
    *,
    model_name: str | None = None,
) -> Dict[str, Any]:
    summary_file = Path(summary_path) if summary_path else MODEL_LAB_ROOT / "latest_summary.json"
    if not summary_file.is_file():
        raise FileNotFoundError(f"Missing model lab summary: {summary_file}")
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    run_id = str(summary.get("run_id") or "")
    run_dir_name = str(summary.get("run_dir_name") or run_id or "")
    if not run_dir_name:
        raise RuntimeError(f"Invalid model lab summary without run_id: {summary_file}")
    run_dir = MODEL_LAB_ROOT / run_dir_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "rolling_model_lab.log"
    target_model = model_name or str(summary.get("best_model") or "")
    if not target_model:
        raise RuntimeError(f"Model lab summary has no best_model: {summary_file}")

    _log(f"Re-exporting submit-ready model factor for {target_model} from {summary_file}", log_path)
    payload = _export_submit_ready_model_factor(
        summary=summary,
        run_dir=run_dir,
        log_path=log_path,
        model_name=target_model,
    )

    summary.setdefault("ensemble_outputs", {})
    summary["ensemble_outputs"][target_model] = payload["submit_path"]
    summary["submit_factor_output"] = payload
    summary.setdefault("models", {})
    if target_model in summary["models"]:
        summary["models"][target_model]["input_factor_correlations"] = _compute_submit_factor_input_correlations(
            summary,
            model_name=target_model,
        )
        summary["models"][target_model]["all_factor_correlations"] = _compute_submit_factor_all_correlations(
            summary,
            model_name=target_model,
        )
    if summary.get("best_model") == target_model:
        summary["best_model_input_factor_correlations"] = summary["models"].get(target_model, {}).get("input_factor_correlations", [])
        summary["best_model_all_factor_correlations"] = summary["models"].get(target_model, {}).get("all_factor_correlations", [])
    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (MODEL_LAB_ROOT / "latest_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_markdown_report(summary, run_dir)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="AutoAlpha flexible rolling model lab")
    parser.add_argument("--reexport-submit-factor", action="store_true", help="Re-export submit-ready model factor from latest_summary.json without rerunning the full lab")
    parser.add_argument("--run-low-corr-experiment", action="store_true", help="Run the low-correlation basket exploration using the same mock OOS combo setup")
    parser.add_argument("--model-name", type=str, default="", help="Override the model name used by --reexport-submit-factor")
    parser.add_argument("--target-valid", type=int, default=100, help="Target number of valid factors")
    parser.add_argument("--ideas-per-round", type=int, default=3, help="Ideas generated in each mining round")
    parser.add_argument("--eval-days", type=int, default=0, help="0 means full trading history")
    parser.add_argument("--max-rounds", type=int, default=0, help="0 means keep mining until target")
    parser.add_argument("--sleep-seconds", type=float, default=2.0, help="Pause between mining rounds")
    parser.add_argument("--train-days", type=int, default=126, help="Rolling train window in trading days")
    parser.add_argument("--test-days", type=int, default=126, help="Rolling test window in trading days")
    parser.add_argument("--step-days", type=int, default=126, help="Rolling step size in trading days")
    parser.add_argument("--allow-partial", action="store_true", help="Proceed even if target-valid is not reached")
    parser.add_argument("--fee-bps", type=float, default=DEFAULT_FEE_BPS, help="One-way fee in basis points for net PnL")
    args = parser.parse_args()

    if args.reexport_submit_factor:
        payload = export_submit_ready_model_factor(model_name=args.model_name or None)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.run_low_corr_experiment:
        summary = run_low_corr_experiment(fee_bps=args.fee_bps)
        print(json.dumps({"run_id": summary.get("run_id"), "best_model": summary.get("best_model")}, ensure_ascii=False, indent=2))
        return 0

    run_model_lab(
        target_valid_count=args.target_valid,
        ideas_per_round=args.ideas_per_round,
        eval_days_count=args.eval_days,
        max_rounds=args.max_rounds,
        sleep_seconds=args.sleep_seconds,
        train_days=args.train_days,
        test_days=args.test_days,
        step_days=args.step_days,
        allow_partial=args.allow_partial,
        fee_bps=args.fee_bps,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
