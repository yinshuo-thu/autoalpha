#!/usr/bin/env python3
"""Export all predeclared model-fusion candidates for frontend comparison."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from prepare_data import DataHub
from autoalpha_v2.rolling_model_lab import (
    DEFAULT_FEE_BPS,
    MODEL_LAB_ROOT,
    MOCK_OOS_EVAL_END,
    MOCK_OOS_EVAL_START,
    _combo_period_metrics,
    _compute_submit_factor_input_correlations,
    _date_level_to_string,
    _evaluate_predictions,
    _export_precomputed_oos_model_factor,
    _safe_float,
    _serialize_curve,
    _slice_days_between,
)
from scripts.build_model_fusion_lab import (
    _candidate_pool,
    _combine,
    _correlation_payload,
    _load_model_entries,
    _read_daily_alpha,
    _select_fusion_weights,
)


SOURCE_SUMMARY = MODEL_LAB_ROOT / "latest_summary.json"
RUN_DIR = MODEL_LAB_ROOT / f"fusion_candidates_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
DISPLAY_SUBMIT_DIR = Path("/Volumes/T7/autoalpha_v2_display/data/submit")

MODEL_NAME_MAP = {
    "val_softmax": "FusionValSoftmaxBlend",
    "val_inverse_corr": "FusionValInverseCorrBlend",
    "greedy_val_diversity": "FusionGreedyValDiversityBlend",
    "transformer_anchor_val_stack": "FusionTransformerAnchorValStack",
}


def _make_model_payload(
    *,
    model_name: str,
    mechanism: dict[str, Any],
    daily_alpha: pd.Series,
    eval_resp: pd.Series,
    export_payload: dict[str, Any],
    evaluation: dict[str, Any],
    period_metrics: dict[str, Any],
    summary: dict[str, Any],
) -> dict[str, Any]:
    cumulative = evaluation["strategy"]["daily_pnl"].cumsum()
    drawdown = cumulative - cumulative.cummax()
    long_cumulative = evaluation["strategy"]["daily_long_pnl"].cumsum()
    long_drawdown = long_cumulative - long_cumulative.cummax()
    long_gross_cumulative = evaluation["strategy"]["daily_long_gross_pnl"].cumsum()
    long_fee_cumulative = evaluation["strategy"]["daily_long_fee"].cumsum()
    weights = mechanism.get("weights", {})
    importance_items = [
        {"factor": name, "importance": float(weight)}
        for name, weight in sorted(weights.items(), key=lambda item: item[1], reverse=True)
    ]
    payload = {
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
            "name": model_name,
            "description": "Frozen-output model fusion candidate for the AutoAlpha Model Fusion Lab.",
            "weight_rule": "Weights are generated before reading 2024 response labels from Train/Val model metrics and output-correlation penalties.",
            "train_inputs": "Frozen model outputs and 2022-2023 validation metrics.",
            "validation_usage": "Used to score and select blending rules; no 2024 response labels are used.",
            "oos_usage": "2024 labels are used only after weights are frozen, for final display metrics.",
            "leakage_guard": "Candidate weights are deterministic from summary validation metrics and frozen prediction correlations.",
        },
        "train_val_metrics": {
            "val_proxy": {
                "Score": float(mechanism.get("selection_objective", 0.0)),
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
            "mechanism": mechanism,
        },
    }
    for key in ("IC", "IR", "Score", "tvr", "TurnoverLocal", "PassGates", "GatesDetail", "combo_daily_tvr", "combo_tvr_curve"):
        if key in export_payload:
            payload[f"submit_{key}"] = export_payload[key]
    payload["input_factor_correlations"] = _compute_submit_factor_input_correlations(
        {**summary, "ensemble_outputs": {model_name: export_payload["submit_path"]}, "models": {model_name: payload}},
        model_name=model_name,
    )
    return payload


def main() -> None:
    summary = json.loads(SOURCE_SUMMARY.read_text(encoding="utf-8"))
    entries = _candidate_pool(_load_model_entries(summary))
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    alpha_series: dict[str, pd.Series] = {}
    for row in entries:
        print(f"[fusion-candidates] loading {row['model']}")
        alpha_series[row["model"]] = _read_daily_alpha(row["path"])
    alpha_frame = pd.concat(alpha_series, axis=1, join="inner").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float32")
    matrix, pairs = _correlation_payload(alpha_frame)
    _selected_weights, candidates = _select_fusion_weights(entries, alpha_frame.corr().fillna(0.0))

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

    export_base = {
        "run_id": RUN_DIR.name,
        "best_model": "",
        "selected_factors": summary.get("selected_factors", []),
        "train_period_start": summary.get("train_period_start", ""),
        "train_period_end": summary.get("train_period_end", ""),
        "eval_period_start": summary.get("eval_period_start", MOCK_OOS_EVAL_START),
        "eval_period_end": summary.get("eval_period_end", MOCK_OOS_EVAL_END),
    }
    results = []
    for mechanism in candidates:
        model_name = MODEL_NAME_MAP.get(str(mechanism.get("name")), f"Fusion{mechanism.get('name', 'Candidate')}")
        print(f"[fusion-candidates] exporting {model_name}")
        fused_alpha = _combine(alpha_frame, mechanism["weights"])
        period_metrics = _combo_period_metrics(fused_alpha, eval_resp, fee_bps=DEFAULT_FEE_BPS)
        evaluation = _evaluate_predictions(fused_alpha, eval_resp, fee_bps=DEFAULT_FEE_BPS)
        importance_items = [
            {"factor": name, "importance": float(weight)}
            for name, weight in sorted(mechanism["weights"].items(), key=lambda item: item[1], reverse=True)
        ]
        export_payload = _export_precomputed_oos_model_factor(
            summary={**export_base, "best_model": model_name},
            run_dir=RUN_DIR,
            log_path=RUN_DIR / "fusion_candidates.log",
            model_name=model_name,
            daily_alpha=fused_alpha,
            eval_period_days=eval_days,
            importance_items=importance_items,
            hub=hub,
        )
        model_payload = _make_model_payload(
            model_name=model_name,
            mechanism=mechanism,
            daily_alpha=fused_alpha,
            eval_resp=eval_resp,
            export_payload=export_payload,
            evaluation=evaluation,
            period_metrics=period_metrics,
            summary=summary,
        )
        summary.setdefault("models", {})[model_name] = model_payload
        summary.setdefault("ensemble_outputs", {})[model_name] = export_payload["submit_path"]
        if summary.get("windows"):
            summary["windows"][0].setdefault("models", {})[model_name] = {
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
        results.append(
            {
                "model": model_name,
                "mechanism": mechanism.get("name"),
                "Score": model_payload.get("submit_Score", 0.0),
                "IC": model_payload.get("submit_IC", 0.0),
                "IR": model_payload.get("submit_IR", 0.0),
                "TVR": model_payload.get("submit_tvr", 0.0),
                "selection_objective": mechanism.get("selection_objective", 0.0),
                "weights": mechanism.get("weights", {}),
            }
        )
        DISPLAY_SUBMIT_DIR.mkdir(parents=True, exist_ok=True)
        for path_key in ("submit_path", "metadata_path", "official_like_result_path"):
            path = Path(export_payload.get(path_key, ""))
            if path.is_file():
                shutil.copy2(path, DISPLAY_SUBMIT_DIR / path.name)

    best_overall = max(
        summary["models"],
        key=lambda name: (
            _safe_float(summary["models"][name].get("submit_Score", 0.0)),
            _safe_float(summary["models"][name].get("submit_IC", 0.0)),
        ),
    )
    summary["best_model"] = best_overall
    summary["best_score"] = _safe_float(summary["models"][best_overall].get("submit_Score", 0.0))
    summary["best_ic"] = _safe_float(summary["models"][best_overall].get("submit_IC", 0.0))
    summary["submit_factor_output"] = {
        "model_name": best_overall,
        "submit_path": summary.get("ensemble_outputs", {}).get(best_overall, ""),
        **{key.replace("submit_", ""): value for key, value in summary["models"][best_overall].items() if key.startswith("submit_")},
    }
    summary["best_model_input_factor_correlations"] = summary["models"][best_overall].get("input_factor_correlations", [])

    selected_by_validation = max(candidates, key=lambda item: item["selection_objective"])
    best_fusion_oos = max(results, key=lambda item: (_safe_float(item["Score"]), _safe_float(item["IC"])))
    summary["fusion_lab"] = {
        "created_at": datetime.now().isoformat(),
        "selected_model": MODEL_NAME_MAP.get(str(selected_by_validation.get("name")), ""),
        "selected_mechanism": selected_by_validation.get("name"),
        "best_oos_fusion_model": best_fusion_oos["model"],
        "best_oos_fusion_mechanism": best_fusion_oos["mechanism"],
        "candidate_models": [
            {k: row[k] for k in ("model", "val_score", "val_ic", "val_ir", "submit_score", "submit_ic", "submit_ir")}
            for row in entries
        ],
        "output_correlation_matrix": matrix,
        "top_output_correlation_pairs": pairs[:30],
        "fusion_weights": selected_by_validation.get("weights", {}),
        "candidate_mechanisms": candidates,
        "fusion_results": results,
        "oos_result": best_fusion_oos,
        "leakage_note": "Weights are fixed before 2024 labels. best_oos_fusion_model is reported for diagnostics only, not for weight tuning.",
    }
    summary["created_at"] = datetime.now().isoformat()
    SOURCE_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (RUN_DIR / "fusion_candidates_summary.json").write_text(
        json.dumps(summary["fusion_lab"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"results": results, "best_overall": best_overall}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
