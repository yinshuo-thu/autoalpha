#!/usr/bin/env python3
"""
Recalculate submission-like metrics for all manual/submit factors, update metadata,
set folder suffix y/n from Gate + submission sanity, renumber folders by Score (desc).
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.evaluator import evaluate_submission_like_wide
from core.submission import SubmissionBuilder
from prepare_data import DataHub

SUBMIT_ROOT = PROJECT_ROOT / "manual" / "submit"


def normalize_mi_datetime_utc_naive(mi: pd.MultiIndex) -> pd.MultiIndex:
    """与 DataHub 中 resp 一致：datetime 层可为 UTC tz-aware，提交 pq 为 naive，统一成 naive UTC 墙钟再对齐。"""
    names = mi.names
    d = mi.get_level_values(0)
    t = mi.get_level_values(1)
    if getattr(t, "tz", None) is not None:
        t = pd.DatetimeIndex(t).tz_convert("UTC").tz_localize(None)
    return pd.MultiIndex.from_arrays([d, t], names=names)


def load_eval_frames() -> tuple[pd.DataFrame, pd.DataFrame, str, str]:
    """Only load resp + restriction (no full PV unstack)."""
    hub = DataHub()
    resp_wide = hub.resp["resp"].unstack("security_id").astype("float32")
    rest_wide = (
        hub.trading_restriction["trading_restriction"]
        .unstack("security_id")
        .reindex_like(resp_wide)
        .fillna(0)
        .astype("float32")
    )
    resp_wide.index = normalize_mi_datetime_utc_naive(resp_wide.index)
    rest_wide.index = normalize_mi_datetime_utc_naive(rest_wide.index)
    days = hub.get_trading_days_list()
    return resp_wide, rest_wide, days[0], days[-1]


def read_submission_parquet(pq_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(pq_path)
    df["date"] = pd.to_datetime(df["date"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    if getattr(df["datetime"].dt, "tz", None) is not None:
        df["datetime"] = df["datetime"].dt.tz_localize(None)
    return df


def df_to_alpha_un(df: pd.DataFrame, resp_un: pd.DataFrame) -> pd.DataFrame:
    alpha_long = df.set_index(["date", "datetime", "security_id"]).sort_index()
    alpha_un = alpha_long["alpha"].astype("float32").unstack("security_id")
    alpha_un.index = normalize_mi_datetime_utc_naive(alpha_un.index)
    return alpha_un.reindex(index=resp_un.index, columns=resp_un.columns)


def collect_submit_dirs() -> list[Path]:
    out: list[Path] = []
    for p in sorted(SUBMIT_ROOT.iterdir()):
        if not p.is_dir() or p.name.startswith((".", "_recalc")):
            continue
        if re.match(r"^manual_alpha_\d+_", p.name):
            out.append(p)
    return out


def main() -> None:
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dirs = collect_submit_dirs()
    if not dirs:
        print("No manual_alpha_* folders under", SUBMIT_ROOT)
        return

    print(f"[recalc] loading resp + restriction (once, no PV) …", flush=True)
    resp_un, rest_un, start_date, end_date = load_eval_frames()

    rows: list[dict[str, Any]] = []
    shared_sanity: dict[str, Any] | None = None
    for d in dirs:
        meta_files = list(d.glob("*_metadata.json"))
        if not meta_files:
            print(f"[skip] no metadata in {d}")
            continue
        meta_path = meta_files[0]
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        factor_name = old.get("factor_name")
        if not factor_name:
            print(f"[skip] missing factor_name in {meta_path}")
            continue
        pq = d / f"{factor_name}_submission.pq"
        if not pq.is_file():
            print(f"[skip] missing {pq}")
            continue

        df = read_submission_parquet(pq)
        alpha_un = df_to_alpha_un(df, resp_un)
        submission_metrics = evaluate_submission_like_wide(alpha_un, resp_un, rest_un)

        # 网格与覆盖在所有导出 submission 上一致，只跑一次完整 sanity（~5700 万行）。
        if shared_sanity is None:
            alpha_submit = df.set_index(["date", "datetime", "security_id"])[["alpha"]]
            shared_sanity = SubmissionBuilder.pre_submit_sanity_check(
                alpha_submit, start_date, end_date
            )
        sanity_report = shared_sanity

        suffix = (
            "y"
            if submission_metrics.get("PassGates", False)
            and sanity_report.get("submission_ready", False)
            else "n"
        )

        rows.append(
            {
                "old_dir": d,
                "old_meta": old,
                "submission_metrics": submission_metrics,
                "sanity_report": sanity_report,
                "suffix": suffix,
                "pq_path": pq,
            }
        )
        print(
            f"  {factor_name} | Score={submission_metrics.get('Score', 0):.4f} "
            f"PassGates={submission_metrics.get('PassGates')} suffix={suffix}",
            flush=True,
        )

    rows.sort(
        key=lambda r: (
            -float(r["submission_metrics"].get("Score") or 0),
            -float(r["submission_metrics"].get("IC") or 0),
            -float(r["submission_metrics"].get("IR") or 0),
        )
    )

    backup = SUBMIT_ROOT / f"_recalc_backup_{run_stamp}"
    backup.mkdir(parents=True, exist_ok=False)
    for r in rows:
        dest = backup / r["old_dir"].name
        shutil.move(str(r["old_dir"]), str(dest))
        r["backup_path"] = dest

    for rank, r in enumerate(rows, start=1):
        sm = r["submission_metrics"]
        old = r["old_meta"]
        new_name = f"manual_alpha_{rank:03d}"
        out_dir = SUBMIT_ROOT / f"{new_name}_{run_stamp}_{r['suffix']}"
        out_dir.mkdir(parents=True, exist_ok=True)
        new_pq = out_dir / f"{new_name}_submission.pq"
        shutil.copy2(r["backup_path"] / f"{old['factor_name']}_submission.pq", new_pq)

        meta_new = {
            "factor_name": new_name,
            "display_name": f"{new_name}_{old.get('family', '')}",
            "family": old.get("family"),
            "family_label": old.get("family_label"),
            "params": old.get("params"),
            "direction": old.get("direction"),
            "expression": old.get("expression"),
            "description": old.get("description"),
            "PassGates": bool(sm.get("PassGates", False)),
            "Score": float(sm.get("Score", 0.0)),
            "IC": float(sm.get("IC", 0.0)),
            "IR": float(sm.get("IR", 0.0)),
            "Turnover": float(sm.get("Turnover", 0.0)),
            "maxx": float(sm.get("maxx", 0.0)),
            "minn": float(sm.get("minn", 0.0)),
            "max_mean": float(sm.get("max_mean", 0.0)),
            "min_mean": float(sm.get("min_mean", 0.0)),
            "metric_mode": sm.get("metric_mode", "submission_like"),
            "GatesDetail": sm.get("GatesDetail"),
            "score_formula": sm.get("score_formula"),
            "positive_ic_ratio": old.get("positive_ic_ratio")
            or (old.get("research_metrics") or {}).get("positive_ic_ratio"),
            "submission_path": str(new_pq),
            "submission_dir": str(out_dir),
            "metadata_path": str(out_dir / f"{new_name}_metadata.json"),
            "timestamp": run_stamp,
            "sanity_report": r["sanity_report"],
            "future_info_check": old.get("future_info_check"),
            "research_metrics": old.get("research_metrics"),
            "recalc": {
                "from_dir": str(r["backup_path"]),
                "old_factor_name": old.get("factor_name"),
                "rank_by_score": rank,
            },
        }
        if meta_new.get("research_metrics") is None:
            meta_new.pop("research_metrics", None)
        if meta_new.get("future_info_check") is None:
            meta_new.pop("future_info_check", None)

        meta_path = out_dir / f"{new_name}_metadata.json"
        meta_path.write_text(json.dumps(meta_new, indent=2, ensure_ascii=False), encoding="utf-8")

        rp = sm.get("result_preview")
        if rp:
            preview_path = out_dir / f"{new_name}_official_like_result.json"
            preview_payload = [dict(rp, cover_all=int(r["sanity_report"].get("cover_all", 0)))]
            preview_path.write_text(
                json.dumps(preview_payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )

        for extra in r["backup_path"].glob("*"):
            if extra.suffix not in {".pq", ".parquet"} and "metadata" not in extra.name:
                if extra.is_file():
                    new_fn = extra.name.replace(old["factor_name"], new_name, 1)
                    shutil.copy2(extra, out_dir / new_fn)

        print(
            f"[ok] {new_name} (rank {rank}) <- {old.get('factor_name')} "
            f"Score={meta_new['Score']:.4f} {out_dir.name}",
            flush=True,
        )

    print(f"\nDone. Backup: {backup}", flush=True)
    print(f"New timestamp: {run_stamp}", flush=True)


if __name__ == "__main__":
    main()
