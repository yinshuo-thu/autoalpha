from __future__ import annotations

import argparse
import json
import shutil
import sys
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
from prepare_data import DataHub
from runtime_config import load_runtime_config

from scripts.v3_systematic_miner import (
    _corr_info_from_vectors,
    _metric_value,
    _oos_comparison,
    _period_row,
)


CORR_SAMPLE_ROWS = 1_000_000


def _date_mask(index: pd.MultiIndex, start: str, end: str) -> np.ndarray:
    dates = pd.to_datetime(index.get_level_values("date"))
    return np.asarray((dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end)))


def _slice_series(series: pd.Series, start: str, end: str) -> pd.Series:
    if series.empty:
        return series
    return series.loc[_date_mask(series.index, start, end)].astype("float32")


def _normalize_alpha_index(series: pd.Series) -> pd.Series:
    if series.empty:
        return series.rename("alpha").astype("float32")
    frame = series.rename("alpha").reset_index()
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    if "datetime" in frame.columns:
        frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True)
    names = list(series.index.names)
    if "date" not in names or "security_id" not in names:
        return frame.set_index(names)["alpha"].astype("float32").sort_index()
    group_cols = ["date", "security_id"]
    if "datetime" in frame.columns:
        group_cols = ["date", "datetime", "security_id"]
    normalized = (
        frame.groupby(group_cols, sort=True)["alpha"]
        .mean()
        .astype("float32")
        .sort_index()
    )
    return normalized


def _sample_index(index: pd.MultiIndex, max_rows: int = CORR_SAMPLE_ROWS) -> pd.MultiIndex:
    if len(index) <= max_rows:
        return index
    positions = np.linspace(0, len(index) - 1, max_rows, dtype=np.int64)
    return index[positions]


def _series_to_train_vector(series: pd.Series, sample_index: pd.MultiIndex) -> np.ndarray:
    return series.reindex(sample_index).to_numpy(dtype="float32", copy=True)


def _seed_sampled_accepted_vectors(sample_index: pd.MultiIndex) -> list[tuple[str, np.ndarray]]:
    vectors: list[tuple[str, np.ndarray]] = []
    for item in kb.list_valid_factors():
        parquet_path = str(item.get("parquet_path", "") or "")
        series = _read_alpha_series_from_parquet(parquet_path)
        if series is None or series.empty:
            continue
        vectors.append((str(item.get("run_id", "")), _series_to_train_vector(_slice_series(series, DISCOVERY_START, DISCOVERY_END), sample_index)))
        print(f"[v2import] seeded sampled corr reference {item.get('run_id')} from {parquet_path}", flush=True)
    return vectors


def _load_metadata(path: Path) -> dict[str, Any]:
    meta_path = path.with_name(f"{path.stem}_metadata.json")
    if meta_path.is_file():
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _source_paths(root: Path) -> list[Path]:
    return sorted(path for path in root.glob("*.pq") if path.is_file())


def _v2_priority_scores(v2_submit_dir: Path) -> dict[str, float]:
    model_lab_root = v2_submit_dir.parent / "model_lab"
    scores: dict[str, float] = {}
    for summary_path in [
        model_lab_root / "latest_summary.json",
        model_lab_root / "explorations" / "low_corr_submit_combo" / "latest_summary.json",
    ]:
        if not summary_path.is_file():
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for rank, row in enumerate(summary.get("selected_factors") or []):
            run_id = str(row.get("run_id") or "")
            if not run_id:
                continue
            try:
                score = float(row.get("score", 0.0) or 0.0)
            except Exception:
                score = 0.0
            scores[run_id] = max(scores.get(run_id, 0.0), score + max(0, 1000 - rank) * 1e-6)
    return scores


def _already_imported_sources() -> tuple[set[str], set[str]]:
    sources: set[str] = set()
    formulas: set[str] = set()
    for factor in kb.get_all_factors():
        source = str(((factor.get("v2_source") or {}).get("parquet_path") if isinstance(factor.get("v2_source"), dict) else "") or "")
        if source:
            sources.add(str(Path(source).resolve()))
        if str(factor.get("target_source", "")) == "v2_submit_revalidated":
            formula = str(factor.get("formula", "") or "").strip()
            if formula:
                formulas.add(formula)
    return sources, formulas


def _copy_report(source_path: Path, run_id: str, record: dict[str, Any]) -> None:
    report_dir = PROJECT_ROOT / "systematic" / "v2_import_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / f"{run_id}.json").write_text(
        json.dumps(
            {
                "source_path": str(source_path),
                "record": record,
                "created_at": datetime.now().isoformat(),
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


def import_v2_factors(v2_submit_dir: Path, target_valid: int, max_candidates: int = 0) -> dict[str, Any]:
    cfg = load_runtime_config()
    hub = DataHub()
    all_days = sorted(hub.resp.index.get_level_values("date").astype(str).unique().tolist())
    train_days = [day for day in all_days if DISCOVERY_START <= day <= DISCOVERY_END]
    oos_days = [day for day in all_days if OOS_START <= day <= OOS_END]
    full_days = [day for day in all_days if DISCOVERY_START <= day <= OOS_END]
    if not train_days or not full_days:
        raise RuntimeError("Missing train/full day calendar for v2 import.")

    train_index = hub.resp.loc[_date_mask(hub.resp.index, DISCOVERY_START, DISCOVERY_END)].index
    sample_index = _sample_index(train_index)
    accepted_vectors = _seed_sampled_accepted_vectors(sample_index)
    imported_sources, imported_formulas = _already_imported_sources()
    priority_scores = _v2_priority_scores(v2_submit_dir)
    candidates = sorted(
        _source_paths(v2_submit_dir),
        key=lambda path: (-priority_scores.get(path.stem, 0.0), path.name),
    )
    if max_candidates > 0:
        candidates = candidates[:max_candidates]

    existing_valid = len(kb.list_valid_factors())
    accepted_needed = max(0, target_valid - existing_valid)
    if accepted_needed <= 0:
        return {"accepted_count": 0, "total_valid": existing_valid, "message": "target already satisfied"}

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    accepted: list[dict[str, Any]] = []
    checked: list[dict[str, Any]] = []
    max_corr_threshold = float(cfg.get("AUTOALPHA_MAX_LIBRARY_CORR", 0.72) or 0.72)

    print(
        f"[v2import] source={v2_submit_dir} candidates={len(candidates)} "
        f"existing_valid={existing_valid} target_valid={target_valid}",
        flush=True,
    )

    for idx, path in enumerate(candidates, start=1):
        source_key = str(path.resolve())
        if source_key in imported_sources:
            continue
        meta = _load_metadata(path)
        formula = str(meta.get("formula") or meta.get("description") or f"v2_import::{path.stem}").strip()
        if formula in imported_formulas:
            continue
        series = _read_alpha_series_from_parquet(str(path))
        if series is None or series.empty:
            continue
        series = _normalize_alpha_index(series)
        train_series = _slice_series(series, DISCOVERY_START, DISCOVERY_END)
        oos_series = _slice_series(series, OOS_START, OOS_END)
        full_series = _slice_series(series, DISCOVERY_START, OOS_END)
        if train_series.empty or full_series.empty:
            continue

        corr_vector = _series_to_train_vector(train_series, sample_index)
        corr_info = _corr_info_from_vectors(corr_vector, accepted_vectors)
        if float(corr_info.get("max_abs_corr", 0.0) or 0.0) > max_corr_threshold:
            metrics = {
                "IC": 0.0,
                "IR": 0.0,
                "Turnover": 0.0,
                "Score": 0.0,
                "PassGates": False,
                "correlation": {**corr_info, "threshold": max_corr_threshold, "PassLowCorrelation": False},
            }
            oos_metrics: dict[str, Any] = {}
            full_metrics: dict[str, Any] = {}
        else:
            metrics = evaluate_alpha(train_series, hub, train_days)
            metrics = _apply_low_correlation_gate(metrics, corr_info, cfg)
            oos_metrics = evaluate_alpha(oos_series, hub, oos_days) if metrics.get("PassGates") else {}
            full_metrics = evaluate_alpha(full_series, hub, full_days) if metrics.get("PassGates") else {}

        row = {
            "path": str(path),
            "IC": _metric_value(metrics, "IC"),
            "IR": _metric_value(metrics, "IR"),
            "Turnover": _metric_value(metrics, "tvr"),
            "Score": _metric_value(metrics, "Score"),
            "PassGates": bool(metrics.get("PassGates", False)),
            "max_abs_corr": float(corr_info.get("max_abs_corr", 0.0) or 0.0),
            "oos_IC": _metric_value(oos_metrics, "IC") if oos_metrics else 0.0,
            "oos_Score": _metric_value(oos_metrics, "Score") if oos_metrics else 0.0,
        }
        checked.append(row)
        print(
            f"[v2import] {idx:03d}/{len(candidates)} {path.name} | "
            f"IC={row['IC']:.4f} IR={row['IR']:.2f} TVR={row['Turnover']:.2f} "
            f"Score={row['Score']:.3f} Corr={row['max_abs_corr']:.3f} Pass={row['PassGates']} "
            f"| OOS_IC={row['oos_IC']:.4f}",
            flush=True,
        )
        if not row["PassGates"]:
            continue

        run_id = f"v3v2_{run_stamp}_{existing_valid + len(accepted) + 1:02d}"
        parquet_path = export_parquet(
            full_series,
            run_id,
            AUTOALPHA_OUT,
            start_date=full_days[0],
            end_date=full_days[-1],
        )
        period_metrics = [
            _period_row(period="discovery", label="Discovery 2022-2023", start=DISCOVERY_START, end=DISCOVERY_END, days=len(train_days), metrics=metrics, used_for_discovery=True),
            _period_row(period="oos_2024", label="OOS 2024", start=OOS_START, end=OOS_END, days=len(oos_days), metrics=oos_metrics, used_for_discovery=False),
            _period_row(period="full_2022_2024", label="Full 2022-2024", start=full_days[0], end=full_days[-1], days=len(full_days), metrics=full_metrics, used_for_discovery=False),
        ]
        result = {
            "run_id": run_id,
            "formula": formula,
            "thought_process": "Imported from v2 submit candidates, then re-screened by v3 train-only 2022-2023 metrics and low-correlation gate. 2024 is report-only.",
            "IC": _metric_value(metrics, "IC"),
            "IR": _metric_value(metrics, "IR"),
            "tvr": _metric_value(metrics, "tvr"),
            "Turnover": _metric_value(metrics, "tvr"),
            "Score": _metric_value(metrics, "Score"),
            "PassGates": bool(metrics.get("PassGates", False)),
            "correlation": metrics.get("correlation", corr_info),
            "oss_2024": {"start": OOS_START, "end": OOS_END, "days": len(oos_days), "metrics": oos_metrics, "used_for_feedback": False},
            "full_2022_2024": {"start": full_days[0], "end": full_days[-1], "days": len(full_days), "metrics": full_metrics, "used_for_discovery": False},
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
            "postprocess": "v2_import_reexport",
            "lookback_days": 0,
            "status": "ok",
            "inspiration_source_type": "v2_import",
            "inspiration_source_types": ["v2_import"],
            "generation_mode": "v3_v2_import_train_only_lowcorr",
            "target_source": "v2_submit_revalidated",
            "prompt_version": "v3-v2-import-oos-lowcorr-20260427",
            "parquet_path": str(parquet_path),
            "eval_days": len(train_days),
            "screening": {"method": "v2 parquet candidate revalidation", "days": len(train_days), "used_2024": False, "exported_days": len(full_days)},
            "systematic_spec": {"family": "v2_import", "params": {"source_stem": path.stem}, "direction": 0, "expression": formula},
            "v2_source": {"parquet_path": str(path), "metadata": meta},
        }
        kb.add_factor(result, parent_run_ids=[])
        accepted.append(result)
        accepted_vectors.append((run_id, corr_vector))
        imported_sources.add(source_key)
        imported_formulas.add(formula)
        _copy_report(path, run_id, result)
        print(f"[v2import] accepted {run_id} | total_valid={existing_valid + len(accepted)}/{target_valid}", flush=True)
        if len(accepted) >= accepted_needed:
            break

    report_path = PROJECT_ROOT / "systematic" / "v2_import_reports" / f"v2_import_search_{run_stamp}.csv"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(checked).to_csv(report_path, index=False)
    return {
        "run_stamp": run_stamp,
        "accepted_count": len(accepted),
        "total_valid": len(kb.list_valid_factors()),
        "report_path": str(report_path),
        "accepted": [{"run_id": item["run_id"], "Score": item["Score"], "IC": item["IC"], "IR": item["IR"]} for item in accepted],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Import v2 submit parquets as v3 train-only candidates.")
    parser.add_argument("--v2-submit-dir", type=Path, default=Path("/Volumes/T7/autoalpha_v2/autoalpha_v2/submit"))
    parser.add_argument("--target-valid", type=int, default=10)
    parser.add_argument("--max-candidates", type=int, default=0)
    args = parser.parse_args()
    summary = import_v2_factors(args.v2_submit_dir, args.target_valid, max_candidates=args.max_candidates)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    if int(summary.get("total_valid", 0) or 0) < int(args.target_valid):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
