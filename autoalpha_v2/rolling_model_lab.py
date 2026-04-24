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
FEATURE_CACHE_DIR = MODEL_LAB_ROOT / "feature_cache_daily"
ENSEMBLE_OUTPUT_DIR = MODEL_LAB_ROOT / "ensemble_factors"
DEFAULT_FEE_BPS = 2.0

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


def _strategy_from_predictions(pred: pd.Series, resp: pd.Series, *, fee_bps: float = DEFAULT_FEE_BPS) -> Dict[str, Any]:
    frame = pd.DataFrame({"pred": pred, "resp": resp}).replace([np.inf, -np.inf], np.nan).dropna()
    if frame.empty:
        return {
            "daily_pnl": pd.Series(dtype="float32"),
            "daily_gross_pnl": pd.Series(dtype="float32"),
            "daily_fee": pd.Series(dtype="float32"),
            "total_pnl": 0.0,
            "gross_pnl": 0.0,
            "total_fee": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "hit_ratio": 0.0,
            "avg_turnover": 0.0,
        }

    frame["weight"] = frame.groupby(level="date")["pred"].transform(_signal_to_weights)
    frame["gross_pnl"] = frame["weight"] * frame["resp"]
    daily_gross_pnl = frame.groupby(level="date")["gross_pnl"].sum().astype("float32")
    daily_weights = frame["weight"].unstack("security_id").fillna(0.0)
    turnover = daily_weights.diff().abs().sum(axis=1) / 2.0
    if len(turnover):
        turnover.iloc[0] = daily_weights.iloc[0].abs().sum() / 2.0
    fee_rate = max(float(fee_bps), 0.0) / 10_000.0
    daily_fee = (turnover.reindex(daily_gross_pnl.index).fillna(0.0) * fee_rate).astype("float32")
    daily_pnl = (daily_gross_pnl - daily_fee).astype("float32")
    cum_pnl = daily_pnl.cumsum()
    running_max = cum_pnl.cummax()
    drawdown = cum_pnl - running_max
    pnl_std = daily_pnl.std()
    sharpe = float(daily_pnl.mean() / pnl_std * np.sqrt(252)) if pnl_std and pnl_std > 0 else 0.0
    return {
        "daily_pnl": daily_pnl,
        "daily_gross_pnl": daily_gross_pnl,
        "daily_fee": daily_fee,
        "drawdown_curve": drawdown.astype("float32"),
        "total_pnl": float(cum_pnl.iloc[-1]) if len(cum_pnl) else 0.0,
        "gross_pnl": float(daily_gross_pnl.cumsum().iloc[-1]) if len(daily_gross_pnl) else 0.0,
        "total_fee": float(daily_fee.sum()) if len(daily_fee) else 0.0,
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
            })
        q = max(1, int(n * 0.2))
        ordered = group.sort_values("pred")
        bottom = ordered.iloc[:q]
        top = ordered.iloc[-q:]
        return pd.Series({
            "mean_prediction": float(group["pred"].mean()),
            "mean_return": float(group["resp"].mean()),
            "predicted_spread": float(top["pred"].mean() - bottom["pred"].mean()),
            "realized_spread": float(top["resp"].mean() - bottom["resp"].mean()),
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


def _fit_predict_models(
    *,
    feature_cols: List[str],
    y_train: pd.Series,
    x_train_np: np.ndarray,
    x_test_np: np.ndarray,
) -> Dict[str, Dict[str, Any]]:
    outputs: Dict[str, Dict[str, Any]] = {}
    for model_name, _builder, _max_rows in _model_spec_items():
        outputs[model_name] = _fit_predict_single_model(
            model_name=model_name,
            feature_cols=feature_cols,
            y_train=y_train,
            x_train_np=x_train_np,
            x_pred_np=x_test_np,
        )

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
    return [
        ("LinearRegression", lambda: LinearRegression(), 0),
        ("Ridge", lambda: make_pipeline(StandardScaler(), Ridge(alpha=5.0, random_state=42)), 0),
        (
            "LightGBM",
            lambda: lgb.LGBMRegressor(
                objective="regression",
                learning_rate=0.035,
                n_estimators=260,
                num_leaves=63,
                max_depth=7,
                min_child_samples=120,
                reg_alpha=0.05,
                reg_lambda=0.25,
                subsample=0.88,
                colsample_bytree=0.85,
                random_state=42,
                n_jobs=-1,
                verbosity=-1,
            ),
            0,
        ),
        (
            "RandomForest",
            lambda: RandomForestRegressor(
                n_estimators=80,
                max_depth=7,
                min_samples_leaf=80,
                max_features="sqrt",
                random_state=42,
                n_jobs=-1,
            ),
            250_000,
        ),
        (
            "ExtraTrees",
            lambda: ExtraTreesRegressor(
                n_estimators=100,
                max_depth=8,
                min_samples_leaf=80,
                max_features="sqrt",
                random_state=42,
                n_jobs=-1,
            ),
            250_000,
        ),
        (
            "HistGradientBoosting",
            lambda: HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_iter=140,
                max_leaf_nodes=31,
                l2_regularization=0.02,
                random_state=42,
            ),
            350_000,
        ),
        (
            "MLP",
            lambda: make_pipeline(
                StandardScaler(),
                MLPRegressor(
                    hidden_layer_sizes=(64, 16),
                    activation="relu",
                    alpha=1e-4,
                    batch_size=8192,
                    learning_rate_init=0.001,
                    max_iter=45,
                    early_stopping=True,
                    validation_fraction=0.08,
                    random_state=42,
                    verbose=False,
                ),
            ),
            300_000,
        ),
    ]


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


def _build_submit_export_windows(
    trading_days: List[str],
    *,
    train_days: int,
    test_days: int,
    step_days: int,
) -> List[Dict[str, Any]]:
    rolling_windows = _build_rolling_windows(
        trading_days,
        train_days=train_days,
        test_days=test_days,
        step_days=step_days,
    )
    if not rolling_windows:
        return []

    export_windows: List[Dict[str, Any]] = []
    first = rolling_windows[0]
    export_windows.append(
        {
            "window_id": "initial",
            "train_days": list(first["train_days"]),
            "predict_days": list(dict.fromkeys(first["train_days"] + first["test_days"])),
        }
    )
    for window in rolling_windows[1:]:
        export_windows.append(
            {
                "window_id": f"test_{window['window_id']}",
                "train_days": list(window["train_days"]),
                "predict_days": list(window["test_days"]),
            }
        )

    last_regular_day = export_windows[-1]["predict_days"][-1]
    last_regular_idx = trading_days.index(last_regular_day)
    if last_regular_idx < len(trading_days) - 1:
        tail_days = trading_days[last_regular_idx + 1 :]
        train_start_idx = max(0, last_regular_idx + 1 - train_days)
        tail_train_days = trading_days[train_start_idx : last_regular_idx + 1]
        export_windows.append(
            {
                "window_id": "tail",
                "train_days": tail_train_days,
                "predict_days": tail_days,
            }
        )
    return export_windows


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
        test_days=test_days,
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
        model_frame, model_feature_cols = _make_model_feature_frame(frame, feature_cols)
        train_mask = frame.index.get_level_values("date").isin(window["train_days"])
        predict_mask = frame.index.get_level_values("date").isin(window["predict_days"])

        X_train = model_frame.loc[train_mask, model_feature_cols]
        y_train = frame.loc[train_mask, "resp"]
        X_predict = model_frame.loc[predict_mask, model_feature_cols]
        if X_train.empty or X_predict.empty:
            _log(f"[submit-export] skip {window['window_id']} due to empty train/predict frame", log_path)
            continue

        model_output = _fit_predict_single_model(
            model_name=model_name,
            feature_cols=model_feature_cols,
            y_train=y_train,
            x_train_np=X_train.to_numpy(dtype="float32", copy=False),
            x_pred_np=X_predict.to_numpy(dtype="float32", copy=False),
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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from core.evaluator import evaluate_submission_like_wide  # noqa: F401 - import guard for environment parity

    resp_df = hub.resp
    rest_df = hub.trading_restriction
    resp_col = "resp" if "resp" in resp_df.columns else resp_df.columns[0]
    rest_col = "trading_restriction" if "trading_restriction" in rest_df.columns else rest_df.columns[0]

    resp_un = resp_df[resp_col].unstack("security_id").sort_index()
    rest_un = rest_df[rest_col].unstack("security_id").sort_index().reindex_like(resp_un).fillna(0)

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

    daily_alpha, importance_items = _predict_submit_daily_alpha(
        model_name=model_name,
        feature_refs=feature_refs,
        resp_series=resp_series,
        trading_days=trading_days,
        train_days=int(summary.get("train_days", 126) or 126),
        test_days=int(summary.get("test_days", 126) or 126),
        step_days=int(summary.get("step_days", 126) or 126),
        log_path=log_path,
    )

    covered_days = sorted(daily_alpha.index.get_level_values("date").astype(str).unique().tolist())
    if covered_days != trading_days:
        missing = sorted(set(trading_days) - set(covered_days))
        if missing:
            raise RuntimeError(f"Submit-ready model factor still misses {len(missing)} trading days, first={missing[:5]}")

    export_id = f"model_lab_{model_lab_run_id}_{model_name.lower()}"
    out_path = kb.SUBMIT_DIR / f"{export_id}.pq"
    sanity_report = _write_submit_ready_daily_alpha(daily_alpha, out_path)

    metrics_payload: Dict[str, Any] | None = None
    metrics_error = ""
    try:
        from core.evaluator import evaluate_submission_like_wide

        alpha_un, resp_un, rest_un = _build_submission_alpha_wide_from_daily(daily_alpha, hub)
        metrics_payload = evaluate_submission_like_wide(alpha_un, resp_un, rest_un)
    except Exception as exc:
        metrics_error = str(exc)
        _log(f"[submit-export] official-like evaluation skipped for {export_id}: {exc}", log_path)

    snapshot = _metrics_snapshot(metrics_payload or {})
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
        "formula": "",
        "description": "Submit-ready model factor exported from AutoAlpha rolling model lab.",
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
    ax.set_title("Rolling Test Daily IC by Window")
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
    ax.set_title("Rolling Test Cumulative PnL")
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
        "# AutoAlpha Rolling Model Lab",
        "",
        f"- Run ID: `{summary['run_id']}`",
        f"- Generated At: `{summary['created_at']}`",
        f"- Target Valid Factors: `{summary['target_valid_count']}`",
        f"- Selected Factors: `{summary['selected_factor_count']}`",
        f"- Rolling Windows: `{summary['window_count']}`",
        f"- Best Model: `{summary['best_model']}`",
        "",
        "## Model Summary",
    ]
    for model_name, payload in summary.get("models", {}).items():
        lines.extend(
            [
                "",
                f"### {model_name}",
                f"- Mean Daily IC: `{payload['avg_daily_ic']:.4f}`",
                f"- Mean Daily Rank IC: `{payload['avg_daily_rank_ic']:.4f}`",
                f"- Mean Sharpe: `{payload['avg_sharpe']:.4f}`",
                f"- Net PnL: `{payload['total_pnl']:.4f}`",
                f"- Gross PnL: `{payload.get('gross_pnl', 0.0):.4f}`",
                f"- Total Fee: `{payload.get('total_fee', 0.0):.4f}`",
                f"- Avg TVR: `{payload.get('avg_turnover', 0.0):.4f}`",
                f"- Max Drawdown: `{payload['max_drawdown']:.4f}`",
                f"- Hit Ratio: `{payload['hit_ratio']:.2%}`",
            ]
        )
    if summary.get("ensemble_outputs"):
        lines.extend(["", "## Overall Factor Outputs"])
        for model_name, path in summary.get("ensemble_outputs", {}).items():
            lines.append(f"- {model_name}: `{path}`")
    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


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
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = MODEL_LAB_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "rolling_model_lab.log"
    _log(f"Rolling model lab started: {run_id}", log_path)

    valid_factors = mine_until_target_valid(
        target_valid_count=target_valid_count,
        ideas_per_round=ideas_per_round,
        eval_days_count=eval_days_count,
        max_rounds=max_rounds,
        sleep_seconds=sleep_seconds,
        log_path=log_path,
    )

    if len(valid_factors) < target_valid_count and not allow_partial:
        raise RuntimeError(
            f"Need {target_valid_count} valid factors, but only {len(valid_factors)} are available."
        )

    selected_factors = valid_factors
    if len(selected_factors) < 1:
        raise RuntimeError("Need at least 1 valid factor for rolling model lab.")
    _log(
        f"Using all currently valid factors for model lab: selected={len(selected_factors)} threshold={target_valid_count}",
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

    _log(f"Loading resp data for rolling experiment with {len(feature_refs)} factors", log_path)
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
    windows = _build_rolling_windows(
        trading_days,
        train_days=train_days,
        test_days=test_days,
        step_days=step_days,
    )
    if not windows:
        raise RuntimeError("Not enough trading days for the requested rolling windows.")
    _log(f"Constructed {len(windows)} rolling windows", log_path)

    model_window_metrics: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    model_prediction_series: Dict[str, List[pd.Series]] = defaultdict(list)
    model_input_feature_frames: Dict[str, List[pd.DataFrame]] = defaultdict(list)
    model_daily_pnls: Dict[str, List[pd.Series]] = defaultdict(list)
    model_daily_gross_pnls: Dict[str, List[pd.Series]] = defaultdict(list)
    model_daily_fees: Dict[str, List[pd.Series]] = defaultdict(list)
    model_prediction_comparisons: Dict[str, List[List[Dict[str, Any]]]] = defaultdict(list)
    importance_store: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    window_summaries: List[Dict[str, Any]] = []

    for window in windows:
        all_days = window["train_days"] + window["test_days"]
        frame = _assemble_window_dataset(feature_refs, resp_series, all_days)
        feature_cols = [ref["run_id"] for ref in feature_refs]
        train_mask = frame.index.get_level_values("date").isin(window["train_days"])
        test_mask = frame.index.get_level_values("date").isin(window["test_days"])
        model_frame, model_feature_cols = _make_model_feature_frame(frame, feature_cols)

        X_train = model_frame.loc[train_mask, model_feature_cols]
        y_train = frame.loc[train_mask, "resp"]
        X_test = model_frame.loc[test_mask, model_feature_cols]
        y_test = frame.loc[test_mask, "resp"]
        if X_train.empty or X_test.empty:
            _log(f"Skip window {window['window_id']} due to empty train/test frame", log_path)
            continue

        _log(
            (
                f"Window {window['window_id']}: train {window['train_start']}→{window['train_end']} "
                f"({len(X_train):,} rows), test {window['test_start']}→{window['test_end']} ({len(X_test):,} rows)"
            ),
            log_path,
        )

        X_train_np = X_train.to_numpy(dtype="float32", copy=False)
        X_test_np = X_test.to_numpy(dtype="float32", copy=False)
        fitted_models = _fit_predict_models(
            feature_cols=model_feature_cols,
            y_train=y_train,
            x_train_np=X_train_np,
            x_test_np=X_test_np,
        )

        window_model_payloads: Dict[str, Dict[str, Any]] = {}
        for model_name, model_output in fitted_models.items():
            pred = pd.Series(model_output["pred"], index=X_test.index, name="pred").astype("float32")
            evaluation = _evaluate_predictions(pred, y_test, fee_bps=fee_bps)
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
            }
            model_window_metrics[model_name].append({**window, **payload})
            model_daily_pnls[model_name].append(evaluation["strategy"]["daily_pnl"])
            model_daily_gross_pnls[model_name].append(evaluation["strategy"]["daily_gross_pnl"])
            model_daily_fees[model_name].append(evaluation["strategy"]["daily_fee"])
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

    if not window_summaries:
        raise RuntimeError("No rolling windows produced valid train/test results.")

    model_summaries: Dict[str, Any] = {}
    ensemble_outputs: Dict[str, str] = {}
    submit_factor_output: Dict[str, Any] | None = None
    best_model = None
    best_score = -1e18
    for model_name, metrics_list in model_window_metrics.items():
        if not metrics_list:
            continue
        concatenated = pd.concat(model_daily_pnls[model_name]).sort_index()
        gross_concatenated = pd.concat(model_daily_gross_pnls[model_name]).sort_index()
        fee_concatenated = pd.concat(model_daily_fees[model_name]).sort_index()
        cumulative = concatenated.cumsum()
        gross_cumulative = gross_concatenated.cumsum()
        fee_cumulative = fee_concatenated.cumsum()
        drawdown = cumulative - cumulative.cummax()
        avg_ic = float(np.mean([item["daily_ic_mean"] for item in metrics_list]))
        avg_rank_ic = float(np.mean([item["daily_rank_ic_mean"] for item in metrics_list]))
        avg_ir = float(np.mean([item.get("daily_rank_ic_ir", 0.0) for item in metrics_list]))
        avg_sharpe = float(np.mean([item["sharpe"] for item in metrics_list]))
        total_pnl = float(cumulative.iloc[-1]) if len(cumulative) else 0.0
        gross_pnl = float(gross_cumulative.iloc[-1]) if len(gross_cumulative) else 0.0
        total_fee = float(fee_cumulative.iloc[-1]) if len(fee_cumulative) else 0.0
        max_drawdown = float(drawdown.min()) if len(drawdown) else 0.0
        hit_ratio = float(np.mean([item["hit_ratio"] for item in metrics_list]))
        avg_turnover = float(np.mean([item.get("avg_turnover", 0.0) for item in metrics_list]))
        prediction_comparison_curve = sorted(
            (
                row
                for rows in model_prediction_comparisons.get(model_name, [])
                for row in rows
            ),
            key=lambda item: str(item.get("date", "")),
        )
        importance_items = [
            {
                "factor": name,
                "importance": float(np.mean(scores)),
            }
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
        all_factor_correlations = _compute_pred_series_all_factor_correlations(
            concat_pred,
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
            "hit_ratio": hit_ratio,
            "avg_turnover": avg_turnover,
            "window_metrics": metrics_list,
            "cumulative_curve": _serialize_curve(cumulative),
            "drawdown_curve": _serialize_curve(drawdown),
            "gross_cumulative_curve": _serialize_curve(gross_cumulative),
            "fee_cumulative_curve": _serialize_curve(fee_cumulative),
            "daily_pnl_curve": _serialize_curve(concatenated),
            "prediction_comparison_curve": prediction_comparison_curve,
            "top_features": importance_items[:20],
            "input_factor_correlations": input_factor_correlations,
            "all_factor_correlations": all_factor_correlations,
        }

        score = avg_ic * 1000 + total_pnl
        if score > best_score:
            best_score = score
            best_model = model_name

    if best_model:
        submit_factor_output = _export_submit_ready_model_factor(
            summary={
                "run_id": run_id,
                "best_model": best_model,
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
                "train_days": train_days,
                "test_days": test_days,
                "step_days": step_days,
            },
            run_dir=run_dir,
            log_path=log_path,
            model_name=best_model,
            hub=hub,
        )
        ensemble_outputs[best_model] = submit_factor_output["submit_path"]
        # Attach official submission-like metrics to the best model's summary.
        if best_model in model_summaries:
            for _k in ("IC", "IR", "Score", "tvr", "TurnoverLocal", "PassGates", "GatesDetail"):
                if _k in submit_factor_output:
                    model_summaries[best_model][f"submit_{_k}"] = submit_factor_output[_k]

    summary = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(),
        "target_valid_count": target_valid_count,
        "selected_factor_count": len(feature_refs),
        "window_count": len(window_summaries),
        "train_days": train_days,
        "test_days": test_days,
        "step_days": step_days,
        "fee_bps": fee_bps,
        "best_model": best_model,
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
        "best_model_input_factor_correlations": (
            model_summaries.get(best_model, {}).get("input_factor_correlations", [])
            if best_model
            else []
        ),
        "best_model_all_factor_correlations": (
            model_summaries.get(best_model, {}).get("all_factor_correlations", [])
            if best_model
            else []
        ),
    }

    if best_model and best_model in model_summaries and ensemble_outputs.get(best_model):
        all_factor_correlations = _compute_submit_factor_all_correlations(
            summary,
            model_name=best_model,
        )
        model_summaries[best_model]["all_factor_correlations"] = all_factor_correlations
        summary["best_model_all_factor_correlations"] = all_factor_correlations

        pq_input_corrs = _compute_submit_factor_input_correlations(summary, model_name=best_model)
        if pq_input_corrs:
            model_summaries[best_model]["input_factor_correlations"] = pq_input_corrs
            summary["best_model_input_factor_correlations"] = pq_input_corrs

    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _plot_summary(summary, run_dir)
    _write_markdown_report(summary, run_dir)
    (MODEL_LAB_ROOT / "latest_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _log(f"Rolling model lab finished: {run_dir}", log_path)
    return summary


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
    if not run_id:
        raise RuntimeError(f"Invalid model lab summary without run_id: {summary_file}")
    run_dir = MODEL_LAB_ROOT / run_id
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
