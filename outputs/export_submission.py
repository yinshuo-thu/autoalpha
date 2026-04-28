import os
import pandas as pd
import json
from datetime import datetime
from core.submission import SubmissionBuilder
from paths import SUBMISSIONS_ROOT


def _safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _metric_snapshot(metrics_dict):
    return {
        "PassGates": bool(metrics_dict.get("PassGates", False)),
        "Score": _safe_float(metrics_dict.get("Score", 0.0)),
        "IC": _safe_float(metrics_dict.get("IC", 0.0)),
        "rank_ic": _safe_float(metrics_dict.get("rank_ic", 0.0)),
        "IR": _safe_float(metrics_dict.get("IR", 0.0)),
        "Turnover": _safe_float(metrics_dict.get("Turnover", 0.0)),
        "maxx": _safe_float(metrics_dict.get("maxx", 0.0)),
        "minn": _safe_float(metrics_dict.get("minn", 0.0)),
        "max_mean": _safe_float(metrics_dict.get("max_mean", 0.0)),
        "min_mean": _safe_float(metrics_dict.get("min_mean", 0.0)),
    }


def export_to_parquet(
    alpha_series,
    factor_name,
    output_dir=None,
    metrics=None,
    description="Auto-generated factor via pipeline",
    sanity_report=None,
    hypothesis=None,
):
    """
    Exports the alpha factor to a parquet file following Scientech competition rules.
    Also saves a JSON file with metadata, evaluation data, pass/fail status, and score.
    Saves in structured folders: submit/{basename}_{timestamp}_{y|n}/
    For submission_ready (_y), basename uses alpha_{seq}_{slug} from core.submission_registry (no submit_ prefix).
    """
    if output_dir is None:
        output_dir = SUBMISSIONS_ROOT

    if metrics is None:
        metrics = {}

    overall_metrics = metrics.get("overall", metrics)
    submission_metrics = metrics.get("official_metrics", overall_metrics)
    research_metrics = _metric_snapshot(overall_metrics)
    pass_gates = submission_metrics.get("PassGates", overall_metrics.get("PassGates", metrics.get("PassGates", False)))
    formula = (metrics.get("formula") or overall_metrics.get("formula") or "").strip()

    alpha_for_export = SubmissionBuilder._ensure_frame(alpha_series)
    if not alpha_for_export.empty:
        date_idx = pd.to_datetime(alpha_for_export.index.get_level_values("date"))
        export_start = date_idx.min().strftime("%Y-%m-%d")
        export_end = date_idx.max().strftime("%Y-%m-%d")
        alpha_for_export = SubmissionBuilder.expand_to_full_grid(alpha_for_export, export_start, export_end)
        sanity_report = SubmissionBuilder.pre_submit_sanity_check(alpha_for_export, export_start, export_end)
    else:
        sanity_report = sanity_report or {}

    submission_ready = bool(pass_gates and sanity_report.get("submission_ready", False))
    suffix_flag = "y" if submission_ready else "n"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    storage_basename = factor_name
    display_title = factor_name
    sequence_for_card = None
    skip_feishu_duplicate = False

    if submission_ready and formula:
        from core.submission_registry import resolve_ready_submission

        storage_basename, sequence_for_card, display_title, skip_feishu_duplicate = resolve_ready_submission(
            output_dir, formula
        )
    else:
        from core.submission_registry import sanitize_display_name

        storage_basename = sanitize_display_name(factor_name) if factor_name else "alpha_export"
        display_title = storage_basename

    folder_name = f"{storage_basename}_{timestamp}_{suffix_flag}"

    base_dir = output_dir
    target_dir = os.path.join(base_dir, folder_name)
    os.makedirs(target_dir, exist_ok=True)

    # Reset index to extract date, datetime, security_id
    df = alpha_for_export.reset_index()

    # Bound alpha between -1.0 and 1.0
    df["alpha"] = df["alpha"].clip(-1.0, 1.0)

    # Ensure correct data types matching the Data Specification
    df["date"] = df["date"].astype(str)

    # Formatting datetime to UTC string if it's a timestamp
    if pd.api.types.is_datetime64_any_dtype(df["datetime"]):
        # If it's already timezone-aware, we convert to UTC, then strip TZ
        if df["datetime"].dt.tz is not None:
            df["datetime"] = df["datetime"].dt.tz_convert("UTC").dt.tz_localize(None)
        df["datetime"] = df["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")

    df["security_id"] = df["security_id"].astype(int)

    # Keep only the required columns
    df = df[["date", "datetime", "security_id", "alpha"]]

    # Save to Parquet
    out_path = os.path.join(target_dir, f"{storage_basename}_submission.pq")
    SubmissionBuilder.build(alpha_for_export, out_path)

    ai_hypothesis = hypothesis or description
    # Save JSON metadata (single source of truth for Feishu + humans)
    metadata_path = os.path.join(target_dir, f"{storage_basename}_metadata.json")
    metadata = {
        "factor_name": storage_basename,
        "legacy_factor_hint": factor_name,
        "display_name": display_title,
        "sequence_index": sequence_for_card,
        "description": description,
        "hypothesis": ai_hypothesis,
        "PassGates": pass_gates,
        "submission_ready_flag": submission_ready,
        "Score": float(submission_metrics.get("Score", overall_metrics.get("Score", metrics.get("Score", 0.0)))),
        "IC": float(submission_metrics.get("IC", overall_metrics.get("IC", metrics.get("IC", 0.0)))),
        "rank_ic": float(overall_metrics.get("rank_ic", metrics.get("rank_ic", 0.0))),
        "IR": float(submission_metrics.get("IR", overall_metrics.get("IR", metrics.get("IR", 0.0)))),
        "Turnover": float(submission_metrics.get("Turnover", overall_metrics.get("Turnover", metrics.get("Turnover", 0.0)))),
        "metric_mode": submission_metrics.get("metric_mode", "research"),
        "classification": metrics.get("classification", "Unknown"),
        "formula": formula,
        "score_formula": submission_metrics.get(
            "score_formula",
            metrics.get(
            "score_formula",
            "score = (IC - 0.0005 * Turnover) * sqrt(IR) * 100",
        )),
        "research_metrics": research_metrics,
        "official_metrics": _metric_snapshot(submission_metrics),
        "submission_path": out_path,
        "submission_dir": target_dir,
        "metadata_path": metadata_path,
        "row_count": int(len(df)),
        "timestamp": timestamp,
        "sanity_report": sanity_report or {},
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    result_preview = submission_metrics.get("result_preview")
    if result_preview:
        preview_path = os.path.join(target_dir, f"{storage_basename}_official_like_result.json")
        preview_payload = [dict(result_preview, cover_all=int((sanity_report or {}).get("cover_all", 0)))]
        with open(preview_path, "w", encoding="utf-8") as f:
            json.dump(preview_payload, f, indent=2, ensure_ascii=False)

    print(f"[OK] Exported submission parquet to: {target_dir} ({len(df)} rows)")

    # Send Feishu notification for submission-ready factors only; de-dupe same formula
    if submission_ready:
        import sys

        base_dir_for_import = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if base_dir_for_import not in sys.path:
            sys.path.append(base_dir_for_import)

        try:
            # Reload metadata from disk (authoritative)
            with open(metadata_path, "r", encoding="utf-8") as f:
                meta_loaded = json.load(f)

            if skip_feishu_duplicate:
                print("[Feishu] Same formula already notified earlier — skipping duplicate webhook.")
            else:
                from core.feishu_bot import default_notifier
                from core.submission_registry import mark_formula_notified

                default_notifier.send_factor_notification_from_metadata(meta_loaded)
                if meta_loaded.get("formula"):
                    mark_formula_notified(output_dir, meta_loaded["formula"])
        except Exception as e:
            print(f"[Feishu] Could not send notification: {e}")
    else:
        print(f"[INFO] Factor {factor_name} is not submission-ready yet. Feishu push skipped.")

    return out_path
