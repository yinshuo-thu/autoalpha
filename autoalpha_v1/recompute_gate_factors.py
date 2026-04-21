#!/usr/bin/env python3
"""
Recompute previously gate-passing AutoAlpha factors with the current official-like
metric and submission export path.

This script is intentionally conservative:
  - candidates come from knowledge.json entries that previously had PassGates=true;
  - every candidate is recomputed from formula on the full DataHub history;
  - metrics are recalculated by core.evaluator.evaluate_submission_like_wide;
  - output/<run_id>.pq is regenerated with the normalized full-grid exporter;
  - autoalpha_v1/submit is rebuilt with only factors that still pass the corrected gates.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from autoalpha_v1 import factor_research
from autoalpha_v1.pipeline import AUTOALPHA_OUT, compute_alpha, evaluate_alpha, export_parquet
from prepare_data import DataHub

AUTOALPHA_DIR = Path(__file__).resolve().parent
KB_PATH = AUTOALPHA_DIR / "knowledge.json"
SUBMIT_DIR = AUTOALPHA_DIR / "submit"
REPORT_DIR = AUTOALPHA_DIR / "recompute_reports"


def _load_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _metric_snapshot(metrics: dict[str, Any]) -> dict[str, Any]:
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


def _submit_metadata(run_id: str, entry: dict[str, Any], metrics: dict[str, Any], src_pq: Path, dst_pq: Path) -> dict[str, Any]:
    snapshot = _metric_snapshot(metrics)
    preview = dict(snapshot["result_preview"])
    return {
        "run_id": run_id,
        "formula": entry.get("formula", ""),
        "thought_process": entry.get("thought_process", ""),
        "postprocess": entry.get("postprocess", "rank_clip"),
        "lookback_days": int(entry.get("lookback_days", 20) or 20),
        "eval_days": int(entry.get("eval_days", 0) or 0),
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
        "result_preview": preview,
        "score_formula": snapshot["score_formula"],
        "metric_mode": snapshot["metric_mode"],
        "turnover_basis": snapshot["turnover_basis"],
        "source_parquet_path": str(src_pq),
        "submit_path": str(dst_pq),
        "recomputed_at": datetime.now().isoformat(),
    }


def _refresh_kb_summary(kb: dict[str, Any]) -> None:
    factors = kb.get("factors", {})
    kb["updated_at"] = datetime.now().isoformat()
    kb["total_tested"] = len(factors)
    kb["total_passing"] = sum(1 for item in factors.values() if item.get("PassGates"))
    kb["best_score"] = max((float(item.get("Score", 0) or 0) for item in factors.values()), default=0.0)


def _write_progress(kb: dict[str, Any], rows: list[dict[str, Any]], report_path: Path, backup_path: Path, archive_dir: Path, done: bool = False) -> None:
    _refresh_kb_summary(kb)
    _save_json(KB_PATH, kb)
    summary = {
        "recomputed_at": datetime.now().isoformat(),
        "status": "done" if done else "running",
        "candidate_count": len(rows),
        "passing_count": sum(1 for row in rows if row.get("PassGates")),
        "knowledge_backup": str(backup_path),
        "submit_backup": str(archive_dir),
        "rows": rows,
    }
    _save_json(report_path, summary)


def recompute(previous_only: bool = True, write_research: bool = True) -> dict[str, Any]:
    kb = _load_json(KB_PATH, {"factors": {}})
    factors = kb.setdefault("factors", {})
    candidates = [
        (run_id, entry)
        for run_id, entry in sorted(factors.items())
        if entry.get("formula") and (entry.get("PassGates") or not previous_only)
    ]
    if not candidates:
        raise SystemExit("No candidate factors found.")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = KB_PATH.with_name(f"knowledge_backup_before_recompute_{stamp}.json")
    shutil.copy2(KB_PATH, backup_path)
    report_path = REPORT_DIR / f"gate_recompute_{stamp}.json"

    SUBMIT_DIR.mkdir(parents=True, exist_ok=True)
    archive_dir = SUBMIT_DIR / f"_backup_before_recompute_{stamp}"
    archive_dir.mkdir(parents=True, exist_ok=True)
    for child in list(SUBMIT_DIR.iterdir()):
        if child.name.startswith("_backup_before_recompute_"):
            continue
        shutil.move(str(child), str(archive_dir / child.name))

    print(f"[recompute] candidates={len(candidates)}")
    print(f"[recompute] knowledge backup={backup_path}")
    print(f"[recompute] submit backup={archive_dir}")
    print("[recompute] loading DataHub once ...", flush=True)
    hub = DataHub()
    days = hub.get_trading_days_list()

    rows: list[dict[str, Any]] = []
    for idx, (run_id, entry) in enumerate(candidates, start=1):
        formula = entry.get("formula", "")
        postprocess = entry.get("postprocess", "rank_clip")
        lookback = int(entry.get("lookback_days", 20) or 20)
        print(f"\n[{idx}/{len(candidates)}] {run_id}", flush=True)
        try:
            alpha = compute_alpha(
                formula=formula,
                pv=hub.pv_15m,
                days=days,
                lookback_days=lookback,
                postprocess_mode=postprocess,
            )
            metrics = evaluate_alpha(alpha, hub, days)
            out_path = export_parquet(alpha, run_id, AUTOALPHA_OUT)
            research_path = entry.get("research_path", "")
            if write_research:
                research_path = factor_research.analyze_factor(
                    run_id=run_id,
                    formula=formula,
                    alpha=alpha,
                    metrics=metrics,
                    hub=hub,
                    eval_days=days,
                    thought_process=entry.get("thought_process", ""),
                )

            snapshot = _metric_snapshot(metrics)
            updated = {
                **entry,
                "IC": snapshot["IC"],
                "IR": snapshot["IR"],
                "tvr": snapshot["tvr"],
                "Turnover": snapshot["tvr"],
                "TurnoverLocal": snapshot["TurnoverLocal"],
                "Score": snapshot["Score"],
                "PassGates": snapshot["PassGates"],
                "gates_detail": snapshot["GatesDetail"],
                "maxx": snapshot["maxx"],
                "minn": snapshot["minn"],
                "max_mean": snapshot["max_mean"],
                "min_mean": snapshot["min_mean"],
                "result_preview": snapshot["result_preview"],
                "score_formula": snapshot["score_formula"],
                "metric_mode": snapshot["metric_mode"],
                "turnover_basis": snapshot["turnover_basis"],
                "status": "ok",
                "eval_days": len(days),
                "parquet_path": str(out_path),
                "research_path": research_path,
                "factor_card_path": str(Path(research_path) / "factor_card.json") if research_path and (Path(research_path) / "factor_card.json").is_file() else "",
                "recomputed_at": datetime.now().isoformat(),
            }
            for key in ("submit_path", "submit_metadata_path", "submit_copied_at"):
                updated.pop(key, None)

            if snapshot["PassGates"]:
                dst = SUBMIT_DIR / f"{run_id}.pq"
                shutil.copy2(out_path, dst)
                meta = _submit_metadata(run_id, updated, metrics, out_path, dst)
                meta_path = SUBMIT_DIR / f"{run_id}_metadata.json"
                result_path = SUBMIT_DIR / f"{run_id}_official_like_result.json"
                _save_json(meta_path, meta)
                _save_json(result_path, [dict(snapshot["result_preview"], cover_all=1)])
                updated["submit_path"] = str(dst)
                updated["submit_metadata_path"] = str(meta_path)
                updated["submit_copied_at"] = meta["recomputed_at"]

            factors[run_id] = updated
            row = {
                "run_id": run_id,
                "PassGates": snapshot["PassGates"],
                "IC": snapshot["IC"],
                "IR": snapshot["IR"],
                "tvr": snapshot["tvr"],
                "Score": snapshot["Score"],
                "parquet_path": str(out_path),
                "submit_path": updated.get("submit_path", ""),
            }
            rows.append(row)
            _write_progress(kb, rows, report_path, backup_path, archive_dir)
            print(
                f"  IC={row['IC']:.4f} IR={row['IR']:.4f} tvr={row['tvr']:.2f} "
                f"PassGates={row['PassGates']} Score={row['Score']:.4f}",
                flush=True,
            )
        except Exception as exc:
            entry = dict(entry)
            entry["status"] = "recompute_error"
            entry["errors"] = str(exc)
            entry["PassGates"] = False
            entry["Score"] = 0.0
            factors[run_id] = entry
            rows.append({"run_id": run_id, "PassGates": False, "error": str(exc)})
            _write_progress(kb, rows, report_path, backup_path, archive_dir)
            print(f"  [ERROR] {exc}", flush=True)

    summary = {
        "recomputed_at": datetime.now().isoformat(),
        "status": "done",
        "candidate_count": len(candidates),
        "passing_count": sum(1 for row in rows if row.get("PassGates")),
        "knowledge_backup": str(backup_path),
        "submit_backup": str(archive_dir),
        "rows": rows,
    }
    _write_progress(kb, rows, report_path, backup_path, archive_dir, done=True)
    _save_json(report_path, summary)
    print(f"\n[recompute] report={report_path}")
    print(f"[recompute] passing={summary['passing_count']}/{summary['candidate_count']}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Recompute AutoAlpha Gate-passing factors.")
    parser.add_argument("--all", action="store_true", help="Recompute all factors instead of previous PassGates=true candidates.")
    parser.add_argument("--skip-research", action="store_true", help="Do not regenerate research report files.")
    args = parser.parse_args()
    recompute(previous_only=not args.all, write_research=not args.skip_research)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
