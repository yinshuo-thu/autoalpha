"""
autoalpha_v1/rolling_model_lab.py

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
    from sklearn.linear_model import LinearRegression
    import lightgbm as lgb
except ImportError as exc:  # pragma: no cover - environment guard
    raise RuntimeError("rolling_model_lab requires scikit-learn and lightgbm") from exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_LAB_ROOT = Path(__file__).resolve().parent / "model_lab"
FEATURE_CACHE_DIR = MODEL_LAB_ROOT / "feature_cache_daily"
ENSEMBLE_OUTPUT_DIR = MODEL_LAB_ROOT / "ensemble_factors"

from autoalpha_v1 import knowledge_base as kb
from autoalpha_v1.error_utils import humanize_error
from autoalpha_v1.pipeline import run as run_pipeline
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
    frame = pd.concat(frames, axis=1, join="left")
    feature_cols = [ref["run_id"] for ref in feature_refs]
    frame[feature_cols] = frame[feature_cols].fillna(0.0).astype("float32")
    frame = frame.dropna(subset=["resp"])
    return frame


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


def _signal_to_weights(signal: pd.Series) -> pd.Series:
    if signal.notna().sum() < 2:
        return pd.Series(0.0, index=signal.index, dtype="float32")
    ranked = signal.rank(method="average")
    demeaned = ranked - ranked.mean()
    denom = demeaned.abs().sum()
    if not np.isfinite(denom) or denom == 0:
        return pd.Series(0.0, index=signal.index, dtype="float32")
    return (demeaned / denom).astype("float32")


def _strategy_from_predictions(pred: pd.Series, resp: pd.Series) -> Dict[str, Any]:
    frame = pd.DataFrame({"pred": pred, "resp": resp}).replace([np.inf, -np.inf], np.nan).dropna()
    if frame.empty:
        return {
            "daily_pnl": pd.Series(dtype="float32"),
            "total_pnl": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "hit_ratio": 0.0,
            "avg_turnover": 0.0,
        }

    frame["weight"] = frame.groupby(level="date")["pred"].transform(_signal_to_weights)
    frame["pnl"] = frame["weight"] * frame["resp"]
    daily_pnl = frame.groupby(level="date")["pnl"].sum().astype("float32")
    cum_pnl = daily_pnl.cumsum()
    running_max = cum_pnl.cummax()
    drawdown = cum_pnl - running_max
    pnl_std = daily_pnl.std()
    sharpe = float(daily_pnl.mean() / pnl_std * np.sqrt(252)) if pnl_std and pnl_std > 0 else 0.0
    daily_weights = frame["weight"].unstack("security_id").fillna(0.0)
    turnover = daily_weights.diff().abs().sum(axis=1) / 2.0
    return {
        "daily_pnl": daily_pnl,
        "total_pnl": float(cum_pnl.iloc[-1]) if len(cum_pnl) else 0.0,
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()) if len(drawdown) else 0.0,
        "hit_ratio": float((daily_pnl > 0).mean()) if len(daily_pnl) else 0.0,
        "avg_turnover": float(turnover.mean()) if len(turnover) else 0.0,
    }


def _evaluate_predictions(pred: pd.Series, resp: pd.Series) -> Dict[str, Any]:
    frame = pd.DataFrame({"pred": pred, "resp": resp}).replace([np.inf, -np.inf], np.nan).dropna()
    if frame.empty:
        strategy = _strategy_from_predictions(pred, resp)
        return {
            "rows": 0,
            "daily_ic_mean": 0.0,
            "daily_rank_ic_mean": 0.0,
            "overall_ic": 0.0,
            "strategy": strategy,
            "daily_ic_curve": [],
        }

    daily_ic = frame.groupby(level="date").apply(_daily_corr)
    daily_rank_ic = frame.groupby(level="date").apply(_daily_corr, rank=True)
    overall_ic = frame["pred"].corr(frame["resp"])
    strategy = _strategy_from_predictions(pred, resp)
    return {
        "rows": int(len(frame)),
        "daily_ic_mean": float(daily_ic.dropna().mean()) if daily_ic.notna().any() else 0.0,
        "daily_rank_ic_mean": float(daily_rank_ic.dropna().mean()) if daily_rank_ic.notna().any() else 0.0,
        "overall_ic": float(overall_ic) if pd.notna(overall_ic) else 0.0,
        "strategy": strategy,
        "daily_ic_curve": [
            {"date": str(idx), "value": float(val)}
            for idx, val in daily_ic.dropna().items()
        ],
    }


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


def _export_ensemble_factor(model_name: str, pred: pd.Series, run_dir: Path, run_id: str) -> str:
    ENSEMBLE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    alpha = _normalize_alpha_series(pred).rename("alpha")
    df = alpha.reset_index()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    out_path = ENSEMBLE_OUTPUT_DIR / f"{run_id}_{model_name.lower()}_overall_factor.pq"
    df.to_parquet(out_path, index=False)
    meta = {
        "run_id": run_id,
        "model_name": model_name,
        "path": str(out_path),
        "rows": int(len(df)),
        "created_at": datetime.now().isoformat(),
        "description": "Rolling aggregation output exported as an overall factor parquet.",
    }
    (run_dir / f"{model_name.lower()}_overall_factor.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return str(out_path)


def _plot_summary(summary: Dict[str, Any], run_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = summary.get("models", {})
    windows = summary.get("windows", [])
    if not models or not windows:
        return

    # Plot 1: rolling IC by window
    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = list(range(1, len(windows) + 1))
    for model_name, payload in models.items():
        y = [window["models"][model_name]["daily_ic_mean"] for window in windows]
        ax.plot(x, y, marker="o", linewidth=2, label=model_name)
    ax.axhline(0, color="#94a3b8", linestyle="--", linewidth=1)
    ax.set_title("Rolling Test Daily IC by Window")
    ax.set_xlabel("Window")
    ax.set_ylabel("Mean Daily IC")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(run_dir / "rolling_ic.png", dpi=140)
    plt.close(fig)

    # Plot 2: cumulative pnl
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

    # Plot 3: top features from best model
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
                f"- Total PnL: `{payload['total_pnl']:.4f}`",
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

    selected_factors = valid_factors[: min(target_valid_count, len(valid_factors))]
    if len(selected_factors) < 1:
        raise RuntimeError("Need at least 1 valid factor for rolling model lab.")

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
    model_daily_pnls: Dict[str, List[pd.Series]] = defaultdict(list)
    model_predictions: Dict[str, List[pd.Series]] = defaultdict(list)
    importance_store: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    window_summaries: List[Dict[str, Any]] = []

    for window in windows:
        all_days = window["train_days"] + window["test_days"]
        frame = _assemble_window_dataset(feature_refs, resp_series, all_days)
        feature_cols = [ref["run_id"] for ref in feature_refs]
        train_mask = frame.index.get_level_values("date").isin(window["train_days"])
        test_mask = frame.index.get_level_values("date").isin(window["test_days"])

        X_train = frame.loc[train_mask, feature_cols]
        y_train = frame.loc[train_mask, "resp"]
        X_test = frame.loc[test_mask, feature_cols]
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
        y_train_np = y_train.to_numpy(dtype="float32", copy=False)

        linear = LinearRegression()
        linear.fit(X_train_np, y_train_np)
        linear_pred = pd.Series(linear.predict(X_test_np), index=X_test.index, name="pred")
        linear_eval = _evaluate_predictions(linear_pred, y_test)
        linear_payload = {
            "daily_ic_mean": linear_eval["daily_ic_mean"],
            "daily_rank_ic_mean": linear_eval["daily_rank_ic_mean"],
            "overall_ic": linear_eval["overall_ic"],
            "rows": linear_eval["rows"],
            "pnl": linear_eval["strategy"]["total_pnl"],
            "sharpe": linear_eval["strategy"]["sharpe"],
            "max_drawdown": linear_eval["strategy"]["max_drawdown"],
            "hit_ratio": linear_eval["strategy"]["hit_ratio"],
        }
        model_window_metrics["LinearRegression"].append(
            {
                **window,
                **linear_payload,
            }
        )
        model_daily_pnls["LinearRegression"].append(linear_eval["strategy"]["daily_pnl"])
        model_predictions["LinearRegression"].append(linear_pred.astype("float32"))
        for name, coef in zip(feature_cols, np.abs(linear.coef_), strict=False):
            importance_store["LinearRegression"][name].append(float(coef))

        lightgbm = lgb.LGBMRegressor(
            objective="regression",
            learning_rate=0.05,
            n_estimators=120,
            num_leaves=31,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )
        lightgbm.fit(X_train_np, y_train_np)
        lightgbm_pred = pd.Series(lightgbm.predict(X_test_np), index=X_test.index, name="pred")
        lightgbm_eval = _evaluate_predictions(lightgbm_pred, y_test)
        lightgbm_payload = {
            "daily_ic_mean": lightgbm_eval["daily_ic_mean"],
            "daily_rank_ic_mean": lightgbm_eval["daily_rank_ic_mean"],
            "overall_ic": lightgbm_eval["overall_ic"],
            "rows": lightgbm_eval["rows"],
            "pnl": lightgbm_eval["strategy"]["total_pnl"],
            "sharpe": lightgbm_eval["strategy"]["sharpe"],
            "max_drawdown": lightgbm_eval["strategy"]["max_drawdown"],
            "hit_ratio": lightgbm_eval["strategy"]["hit_ratio"],
        }
        model_window_metrics["LightGBM"].append(
            {
                **window,
                **lightgbm_payload,
            }
        )
        model_daily_pnls["LightGBM"].append(lightgbm_eval["strategy"]["daily_pnl"])
        model_predictions["LightGBM"].append(lightgbm_pred.astype("float32"))
        for name, score in zip(feature_cols, lightgbm.feature_importances_, strict=False):
            importance_store["LightGBM"][name].append(float(score))

        window_summaries.append(
            {
                "window_id": window["window_id"],
                "train_start": window["train_start"],
                "train_end": window["train_end"],
                "test_start": window["test_start"],
                "test_end": window["test_end"],
                "train_rows": int(len(X_train)),
                "test_rows": int(len(X_test)),
                "models": {
                    "LinearRegression": linear_payload,
                    "LightGBM": lightgbm_payload,
                },
            }
        )

    if not window_summaries:
        raise RuntimeError("No rolling windows produced valid train/test results.")

    model_summaries: Dict[str, Any] = {}
    ensemble_outputs: Dict[str, str] = {}
    best_model = None
    best_score = -1e18
    for model_name, metrics_list in model_window_metrics.items():
        if not metrics_list:
            continue
        concatenated = pd.concat(model_daily_pnls[model_name]).sort_index()
        cumulative = concatenated.cumsum()
        drawdown = cumulative - cumulative.cummax()
        avg_ic = float(np.mean([item["daily_ic_mean"] for item in metrics_list]))
        avg_rank_ic = float(np.mean([item["daily_rank_ic_mean"] for item in metrics_list]))
        avg_sharpe = float(np.mean([item["sharpe"] for item in metrics_list]))
        total_pnl = float(cumulative.iloc[-1]) if len(cumulative) else 0.0
        max_drawdown = float(drawdown.min()) if len(drawdown) else 0.0
        hit_ratio = float(np.mean([item["hit_ratio"] for item in metrics_list]))
        importance_items = [
            {
                "factor": name,
                "importance": float(np.mean(scores)),
            }
            for name, scores in importance_store[model_name].items()
            if scores
        ]
        importance_items.sort(key=lambda item: item["importance"], reverse=True)

        model_summaries[model_name] = {
            "avg_daily_ic": avg_ic,
            "avg_daily_rank_ic": avg_rank_ic,
            "avg_sharpe": avg_sharpe,
            "total_pnl": total_pnl,
            "max_drawdown": max_drawdown,
            "hit_ratio": hit_ratio,
            "window_metrics": metrics_list,
            "cumulative_curve": _serialize_curve(cumulative),
            "daily_pnl_curve": _serialize_curve(concatenated),
            "top_features": importance_items[:20],
        }

        merged_pred = pd.concat(model_predictions[model_name]).sort_index() if model_predictions[model_name] else pd.Series(dtype="float32")
        if not merged_pred.empty:
            ensemble_outputs[model_name] = _export_ensemble_factor(model_name, merged_pred, run_dir, run_id)

        score = avg_ic * 1000 + total_pnl
        if score > best_score:
            best_score = score
            best_model = model_name

    summary = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(),
        "target_valid_count": target_valid_count,
        "selected_factor_count": len(feature_refs),
        "window_count": len(window_summaries),
        "train_days": train_days,
        "test_days": test_days,
        "step_days": step_days,
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
    }

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


def main() -> int:
    parser = argparse.ArgumentParser(description="AutoAlpha flexible rolling model lab")
    parser.add_argument("--target-valid", type=int, default=100, help="Target number of valid factors")
    parser.add_argument("--ideas-per-round", type=int, default=3, help="Ideas generated in each mining round")
    parser.add_argument("--eval-days", type=int, default=0, help="0 means full trading history")
    parser.add_argument("--max-rounds", type=int, default=0, help="0 means keep mining until target")
    parser.add_argument("--sleep-seconds", type=float, default=2.0, help="Pause between mining rounds")
    parser.add_argument("--train-days", type=int, default=126, help="Rolling train window in trading days")
    parser.add_argument("--test-days", type=int, default=126, help="Rolling test window in trading days")
    parser.add_argument("--step-days", type=int, default=126, help="Rolling step size in trading days")
    parser.add_argument("--allow-partial", action="store_true", help="Proceed even if target-valid is not reached")
    args = parser.parse_args()

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
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
