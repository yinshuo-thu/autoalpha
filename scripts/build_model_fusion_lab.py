#!/usr/bin/env python3
"""Build a display-ready model fusion lab from frozen Model Lab outputs.

The fusion weight search deliberately avoids 2024 labels. It uses:
- train/validation metrics already stored in latest_summary.json
- model output correlations computed from frozen 2024 predictions only

2024 responses are used after the fusion is frozen, only for final reporting.
"""

from __future__ import annotations

import json
import math
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from prepare_data import DataHub
from autoalpha_v2.rolling_model_lab import (
    DEFAULT_FEE_BPS,
    ENSEMBLE_OUTPUT_DIR,
    MODEL_LAB_ROOT,
    MOCK_OOS_EVAL_END,
    MOCK_OOS_EVAL_START,
    _combo_period_metrics,
    _compute_submit_factor_input_correlations,
    _date_level_to_string,
    _evaluate_predictions,
    _export_precomputed_oos_model_factor,
    _normalize_alpha_series,
    _safe_float,
    _serialize_curve,
    _slice_days_between,
)


SOURCE_SUMMARY = MODEL_LAB_ROOT / "latest_summary.json"
RUN_DIR = MODEL_LAB_ROOT / f"fusion_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
FUSION_MODEL_NAME = "ValidationDiversityFusionBlend"
DISPLAY_SUBMIT_DIR = Path("/Volumes/T7/autoalpha_v2_display/data/submit")


def _read_daily_alpha(path: str | Path) -> pd.Series:
    df = pd.read_parquet(path, columns=["date", "security_id", "alpha"])
    if df.empty:
        return pd.Series(dtype="float32")
    series = (
        df.groupby(["date", "security_id"], sort=True)["alpha"]
        .mean()
        .astype("float32")
        .rename("alpha")
        .sort_index()
    )
    return _normalize_alpha_series(series).astype("float32")


def _short_name(name: str) -> str:
    return (
        name.replace("Combo", "")
        .replace("MetaModel", "")
        .replace("Model", "")
        .replace("Validation", "Val")
        .replace("Transformer", "Trans")
    )


def _load_model_entries(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    outputs = summary.get("ensemble_outputs") or {}
    for model_name, payload in (summary.get("models") or {}).items():
        if not isinstance(payload, dict):
            continue
        path = outputs.get(model_name)
        if not path or not Path(path).is_file():
            continue
        val = payload.get("train_val_metrics", {}).get("val", {})
        train = payload.get("train_val_metrics", {}).get("train", {})
        submit_score = _safe_float(payload.get("submit_Score", 0.0))
        rows.append(
            {
                "model": model_name,
                "path": path,
                "val_score": _safe_float(val.get("Score", 0.0)),
                "val_ic": _safe_float(val.get("IC", 0.0)),
                "val_ir": _safe_float(val.get("IR", 0.0)),
                "train_score": _safe_float(train.get("Score", 0.0)),
                "submit_score": submit_score,
                "submit_ic": _safe_float(payload.get("submit_IC", 0.0)),
                "submit_ir": _safe_float(payload.get("submit_IR", 0.0)),
            }
        )
    return rows


def _candidate_pool(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    positive = [
        row
        for row in entries
        if row["val_score"] > 0 and not str(row["model"]).lower().startswith("fusion")
    ]
    positive.sort(key=lambda row: (row["val_score"], row["val_ic"]), reverse=True)
    return positive


def _correlation_payload(alpha_frame: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    corr = alpha_frame.corr().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    matrix: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    for row_model in corr.index:
        matrix.append(
            {
                "model": row_model,
                "label": _short_name(row_model),
                "values": [
                    {"model": col_model, "label": _short_name(col_model), "corr": float(corr.loc[row_model, col_model])}
                    for col_model in corr.columns
                ],
            }
        )
    names = list(corr.columns)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            value = float(corr.loc[left, right])
            pairs.append({"left": left, "right": right, "corr": value, "abs_corr": abs(value)})
    pairs.sort(key=lambda item: item["abs_corr"], reverse=True)
    return matrix, pairs


def _select_fusion_weights(entries: list[dict[str, Any]], corr: pd.DataFrame) -> tuple[dict[str, float], list[dict[str, Any]]]:
    scores = {row["model"]: max(float(row["val_score"]), 0.0) for row in entries}
    if not scores or max(scores.values()) <= 0:
        weight = 1.0 / max(1, len(entries))
        return {row["model"]: weight for row in entries}, []

    candidates: list[dict[str, Any]] = []
    total_score = sum(scores.values()) or 1.0
    softmax_temp = 4500.0
    exp_values = {name: math.exp((score - max(scores.values())) / softmax_temp) for name, score in scores.items()}
    exp_sum = sum(exp_values.values()) or 1.0
    candidates.append(
        {
            "name": "val_softmax",
            "weights": {name: value / exp_sum for name, value in exp_values.items()},
            "validation_proxy": sum(scores[name] * (value / exp_sum) for name, value in exp_values.items()),
        }
    )

    raw_inverse_corr: dict[str, float] = {}
    for name, score in scores.items():
        others = [other for other in scores if other != name]
        avg_abs_corr = float(np.mean([abs(float(corr.loc[name, other])) for other in others])) if others else 0.0
        raw_inverse_corr[name] = (score / total_score) / max(0.20, avg_abs_corr)
    inv_sum = sum(raw_inverse_corr.values()) or 1.0
    candidates.append(
        {
            "name": "val_inverse_corr",
            "weights": {name: value / inv_sum for name, value in raw_inverse_corr.items()},
            "validation_proxy": sum(scores[name] * (value / inv_sum) for name, value in raw_inverse_corr.items()),
        }
    )

    # Greedy diversity stack: start from best validation head, then add heads
    # that keep validation strength while reducing average correlation.
    ordered = sorted(scores, key=lambda name: scores[name], reverse=True)
    selected = [ordered[0]]
    while len(selected) < min(6, len(ordered)):
        best_name = ""
        best_objective = -1e18
        for name in ordered:
            if name in selected:
                continue
            avg_abs_corr = float(np.mean([abs(float(corr.loc[name, existing])) for existing in selected]))
            objective = scores[name] * (1.0 - 0.55 * avg_abs_corr)
            if objective > best_objective:
                best_objective = objective
                best_name = name
        if not best_name:
            break
        selected.append(best_name)
    greedy_raw = {name: scores[name] * (1.0 - 0.35 * float(np.mean([abs(float(corr.loc[name, other])) for other in selected if other != name] or [0.0]))) for name in selected}
    greedy_sum = sum(max(v, 0.0) for v in greedy_raw.values()) or 1.0
    candidates.append(
        {
            "name": "greedy_val_diversity",
            "weights": {name: max(value, 0.0) / greedy_sum for name, value in greedy_raw.items()},
            "validation_proxy": sum(scores[name] * (max(value, 0.0) / greedy_sum) for name, value in greedy_raw.items()),
        }
    )

    # Architecture-aware blend: use transformer heads as the stability anchor
    # and validation ranking for the residual tabular heads.
    transformer_names = [name for name in ordered if "transformer" in name.lower() or "causaldecay" in name.lower()]
    tabular_names = [name for name in ordered if name not in transformer_names][:3]
    anchor_names = transformer_names[:2] + tabular_names
    if anchor_names:
        raw_anchor: dict[str, float] = {}
        for name in anchor_names:
            family_bonus = 1.35 if name in transformer_names else 1.0
            raw_anchor[name] = scores[name] * family_bonus
        anchor_sum = sum(raw_anchor.values()) or 1.0
        candidates.append(
            {
                "name": "transformer_anchor_val_stack",
                "weights": {name: value / anchor_sum for name, value in raw_anchor.items()},
                "validation_proxy": sum(scores[name] * (value / anchor_sum) for name, value in raw_anchor.items()),
            }
        )

    # Choose by validation proxy with an explicit average-correlation penalty.
    best = None
    best_objective = -1e18
    for candidate in candidates:
        weights = candidate["weights"]
        names = list(weights)
        pair_corr = []
        for i, left in enumerate(names):
            for right in names[i + 1 :]:
                pair_corr.append(abs(float(corr.loc[left, right])) * weights[left] * weights[right])
        corr_penalty = sum(pair_corr)
        candidate["corr_penalty"] = float(corr_penalty)
        objective = float(candidate["validation_proxy"]) * (1.0 - 0.45 * corr_penalty)
        candidate["selection_objective"] = objective
        if objective > best_objective:
            best_objective = objective
            best = candidate
    assert best is not None
    return best["weights"], candidates


def _combine(alpha_frame: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    common = [name for name in weights if name in alpha_frame.columns]
    if not common:
        raise RuntimeError("No common model outputs for fusion.")
    total = sum(float(weights[name]) for name in common)
    if total <= 0:
        raise RuntimeError("Fusion weights sum to zero.")
    out = pd.Series(0.0, index=alpha_frame.index, dtype="float32")
    for name in common:
        out = out.add(alpha_frame[name].astype("float32") * np.float32(weights[name] / total), fill_value=0.0)
    return _normalize_alpha_series(out.rename("alpha")).rename("alpha")


def main() -> None:
    summary = json.loads(SOURCE_SUMMARY.read_text(encoding="utf-8"))
    entries = _candidate_pool(_load_model_entries(summary))
    if len(entries) < 2:
        raise RuntimeError("Need at least two frozen model outputs for fusion.")

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RUN_DIR / "fusion_lab.log"
    alpha_series: dict[str, pd.Series] = {}
    for row in entries:
        print(f"[fusion] loading {row['model']}")
        alpha_series[row["model"]] = _read_daily_alpha(row["path"])

    alpha_frame = pd.concat(alpha_series, axis=1, join="inner").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float32")
    matrix, pairs = _correlation_payload(alpha_frame)
    weights, candidates = _select_fusion_weights(entries, alpha_frame.corr().fillna(0.0))
    fused_alpha = _combine(alpha_frame, weights)

    hub = DataHub()
    resp_series = hub.resp["resp"].astype("float32").sort_index()
    if getattr(resp_series.index, "nlevels", 1) > 2:
        names = list(resp_series.index.names)
        date_level = "date" if "date" in names else names[0]
        security_level = "security_id" if "security_id" in names else names[-1]
        resp_series = (
            resp_series.groupby(level=[date_level, security_level])
            .mean()
            .astype("float32")
            .sort_index()
        )
    resp_series = _date_level_to_string(resp_series.rename("resp")).astype("float32")
    trading_days = sorted(resp_series.index.get_level_values("date").astype(str).unique().tolist())
    eval_days = _slice_days_between(trading_days, MOCK_OOS_EVAL_START, MOCK_OOS_EVAL_END)
    eval_resp = resp_series.loc[resp_series.index.get_level_values("date").isin(eval_days)]
    period_metrics = _combo_period_metrics(fused_alpha, eval_resp, fee_bps=DEFAULT_FEE_BPS)
    evaluation = _evaluate_predictions(fused_alpha, eval_resp, fee_bps=DEFAULT_FEE_BPS)
    cumulative = evaluation["strategy"]["daily_pnl"].cumsum()
    drawdown = cumulative - cumulative.cummax()
    long_cumulative = evaluation["strategy"]["daily_long_pnl"].cumsum()
    long_drawdown = long_cumulative - long_cumulative.cummax()
    long_gross_cumulative = evaluation["strategy"]["daily_long_gross_pnl"].cumsum()
    long_fee_cumulative = evaluation["strategy"]["daily_long_fee"].cumsum()

    export_summary = {
        "run_id": RUN_DIR.name,
        "best_model": FUSION_MODEL_NAME,
        "selected_factors": summary.get("selected_factors", []),
        "train_period_start": summary.get("train_period_start", ""),
        "train_period_end": summary.get("train_period_end", ""),
        "eval_period_start": summary.get("eval_period_start", MOCK_OOS_EVAL_START),
        "eval_period_end": summary.get("eval_period_end", MOCK_OOS_EVAL_END),
    }
    importance_items = [
        {"factor": name, "importance": float(weight)}
        for name, weight in sorted(weights.items(), key=lambda item: item[1], reverse=True)
    ]
    export_payload = _export_precomputed_oos_model_factor(
        summary=export_summary,
        run_dir=RUN_DIR,
        log_path=log_path,
        model_name=FUSION_MODEL_NAME,
        daily_alpha=fused_alpha,
        eval_period_days=eval_days,
        importance_items=importance_items,
        hub=hub,
    )

    model_payload = {
        "avg_daily_ic": float(evaluation.get("daily_ic_mean", 0.0)),
        "avg_daily_rank_ic": float(evaluation.get("daily_rank_ic_mean", 0.0)),
        "avg_daily_ic_bps": round(float(evaluation.get("daily_ic_mean", 0.0)) * 100.0, 4),
        "avg_daily_rank_ic_bps": round(float(evaluation.get("daily_rank_ic_mean", 0.0)) * 100.0, 4),
        "avg_ir": float(period_metrics.get("IR", 0.0)),
        "avg_sharpe": float(evaluation["strategy"].get("sharpe", 0.0)),
        "total_pnl": float(cumulative.iloc[-1]) if len(cumulative) else 0.0,
        "gross_pnl": float(evaluation["strategy"].get("gross_pnl", 0.0)),
        "total_fee": float(evaluation["strategy"].get("total_fee", 0.0)),
        "max_drawdown": float(drawdown.min()) if len(drawdown) else 0.0,
        "long_only_total_pnl": float(evaluation["strategy"].get("long_total_pnl", 0.0)),
        "long_only_gross_pnl": float(evaluation["strategy"].get("long_gross_pnl", 0.0)),
        "long_only_total_fee": float(evaluation["strategy"].get("long_total_fee", 0.0)),
        "long_only_max_drawdown": float(long_drawdown.min()) if len(long_drawdown) else 0.0,
        "hit_ratio": float(evaluation["strategy"].get("hit_ratio", 0.0)),
        "avg_turnover": float(evaluation["strategy"].get("avg_turnover", 0.0)),
        "cumulative_curve": _serialize_curve(cumulative),
        "drawdown_curve": _serialize_curve(drawdown),
        "daily_pnl_curve": _serialize_curve(evaluation["strategy"]["daily_pnl"]),
        "long_only_cumulative_curve": _serialize_curve(long_cumulative),
        "long_only_drawdown_curve": _serialize_curve(long_drawdown),
        "long_only_gross_cumulative_curve": _serialize_curve(long_gross_cumulative),
        "long_only_fee_cumulative_curve": _serialize_curve(long_fee_cumulative),
        "daily_long_pnl_curve": _serialize_curve(evaluation["strategy"]["daily_long_pnl"]),
        "prediction_comparison_curve": period_metrics.get("prediction_comparison_curve", []),
        "combo_tvr_curve": export_payload.get("combo_tvr_curve", []),
        "top_features": importance_items[:20],
        "combo_weights": importance_items[:20],
        "method_card": {
            "name": FUSION_MODEL_NAME,
            "description": "Validation-selected blend of frozen model outputs with explicit output-correlation penalty.",
            "weight_rule": "Weights are selected from 2022-2023 Train/Val model metrics and model-output correlation only; 2024 labels are not used for fusion selection.",
            "train_inputs": "Frozen model outputs, Train/Val metrics, and output correlations.",
            "validation_usage": "Validation Score is the primary selection proxy, penalized by average model-output correlation.",
            "oos_usage": "2024 data is used only for frozen model-output correlations and final OOS reporting; 2024 response labels are excluded from weight selection.",
            "leakage_guard": "No 2024 response/Score/IC is used to choose components, weights, or candidate mechanism.",
        },
        "train_val_metrics": {
            "val_proxy": {
                "Score": float(max(item["selection_objective"] for item in candidates)),
                "IC": 0.0,
                "IR": 0.0,
                "TVR": 0.0,
                "rows": 0,
            },
            "oos": {
                "IC": export_payload.get("IC", 0.0),
                "IR": export_payload.get("IR", 0.0),
                "Score": export_payload.get("Score", 0.0),
                "TVR": export_payload.get("tvr", 0.0),
                "PassGates": export_payload.get("PassGates", False),
                "GatesDetail": export_payload.get("GatesDetail", {}),
            },
        },
        "model_diagnostics": {
            "fusion_weights": weights,
            "candidate_mechanisms": candidates,
            "selected_mechanism": max(candidates, key=lambda item: item["selection_objective"])["name"],
        },
    }
    for key in ("IC", "IR", "Score", "tvr", "TurnoverLocal", "PassGates", "GatesDetail", "combo_daily_tvr", "combo_tvr_curve"):
        if key in export_payload:
            model_payload[f"submit_{key}"] = export_payload[key]

    model_payload["input_factor_correlations"] = _compute_submit_factor_input_correlations(
        {**summary, "ensemble_outputs": {FUSION_MODEL_NAME: export_payload["submit_path"]}, "models": {FUSION_MODEL_NAME: model_payload}},
        model_name=FUSION_MODEL_NAME,
    )

    summary.setdefault("models", {})[FUSION_MODEL_NAME] = model_payload
    summary.setdefault("ensemble_outputs", {})[FUSION_MODEL_NAME] = export_payload["submit_path"]
    if summary.get("windows"):
        summary["windows"][0].setdefault("models", {})[FUSION_MODEL_NAME] = {
            "daily_ic_mean": model_payload["avg_daily_ic"],
            "daily_rank_ic_mean": model_payload["avg_daily_rank_ic"],
            "daily_ic_ir": model_payload["avg_ir"],
            "daily_rank_ic_ir": model_payload["avg_ir"],
            "overall_ic": model_payload["avg_daily_ic"],
            "rows": int(evaluation.get("rows", 0) or 0),
            "pnl": model_payload["total_pnl"],
            "gross_pnl": model_payload["gross_pnl"],
            "total_fee": model_payload["total_fee"],
            "sharpe": model_payload["avg_sharpe"],
            "max_drawdown": model_payload["max_drawdown"],
            "hit_ratio": model_payload["hit_ratio"],
            "avg_turnover": model_payload["avg_turnover"],
            "train_val_metrics": model_payload["train_val_metrics"],
            "model_diagnostics": model_payload["model_diagnostics"],
        }

    current_best_score = _safe_float(summary.get("best_score", 0.0))
    fusion_score = _safe_float(model_payload.get("submit_Score", 0.0))
    if fusion_score > current_best_score:
        summary["best_model"] = FUSION_MODEL_NAME
        summary["best_score"] = fusion_score
        summary["best_ic"] = _safe_float(model_payload.get("submit_IC", 0.0))
        summary["submit_factor_output"] = {
            "model_name": FUSION_MODEL_NAME,
            "submit_path": export_payload["submit_path"],
            **{key.replace("submit_", ""): value for key, value in model_payload.items() if key.startswith("submit_")},
        }
        summary["best_model_input_factor_correlations"] = model_payload.get("input_factor_correlations", [])

    summary["fusion_lab"] = {
        "created_at": datetime.now().isoformat(),
        "selected_model": FUSION_MODEL_NAME,
        "selected_mechanism": model_payload["model_diagnostics"]["selected_mechanism"],
        "candidate_models": [
            {k: row[k] for k in ("model", "val_score", "val_ic", "val_ir", "submit_score", "submit_ic", "submit_ir")}
            for row in entries
        ],
        "output_correlation_matrix": matrix,
        "top_output_correlation_pairs": pairs[:30],
        "fusion_weights": weights,
        "candidate_mechanisms": candidates,
        "oos_result": {
            "Score": model_payload.get("submit_Score", 0.0),
            "IC": model_payload.get("submit_IC", 0.0),
            "IR": model_payload.get("submit_IR", 0.0),
            "TVR": model_payload.get("submit_tvr", 0.0),
        },
    }
    summary["created_at"] = datetime.now().isoformat()
    SOURCE_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    run_summary_path = RUN_DIR / "fusion_summary.json"
    run_summary_path.write_text(json.dumps(summary["fusion_lab"], ensure_ascii=False, indent=2), encoding="utf-8")

    DISPLAY_SUBMIT_DIR.mkdir(parents=True, exist_ok=True)
    for path_key in ("submit_path", "metadata_path", "official_like_result_path"):
        path = Path(export_payload.get(path_key, ""))
        if path.is_file():
            shutil.copy2(path, DISPLAY_SUBMIT_DIR / path.name)

    print(json.dumps({
        "fusion_model": FUSION_MODEL_NAME,
        "selected_mechanism": summary["fusion_lab"]["selected_mechanism"],
        "Score": model_payload.get("submit_Score"),
        "IC": model_payload.get("submit_IC"),
        "IR": model_payload.get("submit_IR"),
        "TVR": model_payload.get("submit_tvr"),
        "weights": weights,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
