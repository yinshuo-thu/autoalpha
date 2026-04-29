from __future__ import annotations

import glob
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from paths import FUTURE_ALPHA_ROOT, OFR_ROOT, OUTPUTS_ROOT


PRODUCTS = ("C", "LH", "M")
PRODUCT_ALIAS = {"C": "C0001", "LH": "LH0001", "M": "M0001"}


def product_from_contract(contract: str) -> str:
    token = str(contract or "").lower()
    if token.startswith("lh"):
        return "LH"
    if token.startswith("c"):
        return "C"
    if token.startswith("m"):
        return "M"
    return token.upper()


def product_from_alpha_path(path: str) -> Optional[str]:
    name = os.path.basename(path)
    match = re.search(r"@([A-Z]+)0001_", name)
    return match.group(1) if match else None


def factor_name_from_alpha_path(path: str) -> str:
    name = os.path.basename(path)
    return name.split("@", 1)[0]


def _date_key(value: Any) -> str:
    return pd.to_datetime(value).strftime("%Y%m%d")


def _series_dates(alpha: pd.Series) -> List[str]:
    if alpha.empty:
        return []
    dates = pd.to_datetime(alpha.index.get_level_values("date")).strftime("%Y%m%d")
    return sorted(pd.unique(dates).tolist())


def product_series(alpha: pd.Series, product: str, *, how: str = "active") -> pd.Series:
    """Collapse a contract-indexed alpha series to one product-level time series."""
    if alpha is None or alpha.empty:
        return pd.Series(dtype="float32")
    frame = alpha.rename("alpha").reset_index()
    frame["product"] = frame["security_id"].map(product_from_contract)
    frame = frame[frame["product"] == product].copy()
    if frame.empty:
        return pd.Series(dtype="float32")
    frame["datetime"] = pd.to_datetime(frame["datetime"]).dt.floor("15min")
    if how == "active":
        std_by_contract = frame.groupby("security_id")["alpha"].std().replace([np.inf, -np.inf], np.nan).dropna()
        if not std_by_contract.empty and float(std_by_contract.max()) > 0:
            chosen = str(std_by_contract.sort_values(ascending=False).index[0])
            frame = frame[frame["security_id"].astype(str) == chosen]
        grouped = frame.groupby(["date", "datetime"], sort=True)["alpha"]
        out = grouped.mean()
    else:
        grouped = frame.groupby(["date", "datetime"], sort=True)["alpha"]
        out = grouped.mean() if how == "mean" else grouped.last()
    return pd.to_numeric(out, errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float32")


def active_contract_alpha(alpha: pd.Series, product: str) -> tuple[str, pd.Series]:
    """Return the most active contract's 15m alpha series for a product."""
    if alpha is None or alpha.empty:
        return "", pd.Series(dtype="float32")
    frame = alpha.rename("alpha").reset_index()
    frame["product"] = frame["security_id"].map(product_from_contract)
    frame = frame[frame["product"] == product].copy()
    if frame.empty:
        return "", pd.Series(dtype="float32")
    frame["datetime"] = pd.to_datetime(frame["datetime"]).dt.floor("15min")
    std_by_contract = frame.groupby("security_id")["alpha"].std().replace([np.inf, -np.inf], np.nan).dropna()
    if not std_by_contract.empty and float(std_by_contract.max()) > 0:
        chosen = str(std_by_contract.sort_values(ascending=False).index[0])
    else:
        chosen = str(frame["security_id"].astype(str).iloc[0])
    frame = frame[frame["security_id"].astype(str) == chosen]
    series = frame.groupby(["date", "datetime"], sort=True)["alpha"].mean()
    return chosen, pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float32")


def _read_ofr_ticks(contract: str, dates: Iterable[str]) -> pd.DataFrame:
    product = product_from_contract(contract)
    path = os.path.join(OFR_ROOT, product, f"{str(contract).lower()}.parquet")
    if not os.path.exists(path):
        return pd.DataFrame()
    date_keys = sorted({_date_key(d) for d in dates})
    if not date_keys:
        return pd.DataFrame()
    columns = ["trading_date", "timestamp", "contract", "price", "mid"]
    try:
        df = pd.read_parquet(
            path,
            columns=columns,
            filters=[("trading_date", ">=", date_keys[0]), ("trading_date", "<=", date_keys[-1])],
        )
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return df
    df = df[df["trading_date"].astype(str).isin(date_keys)].copy()
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["trading_date"].astype(str), format="%Y%m%d", errors="coerce").dt.strftime("%Y-%m-%d")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["datetime"] = df["timestamp"].dt.floor("15min")
    df["security_id"] = df["contract"].astype(str)
    price_col = "mid" if "mid" in df.columns and df["mid"].notna().any() else "price"
    df["h60_price"] = pd.to_numeric(df[price_col], errors="coerce")
    return df.dropna(subset=["date", "timestamp", "h60_price"]).sort_values(["date", "timestamp"])


def product_tick_signal(alpha: pd.Series, product: str) -> pd.DataFrame:
    """Broadcast a 15m product alpha to the OFR tick grid used by existing alpha files."""
    contract, bar_alpha = active_contract_alpha(alpha, product)
    if not contract or bar_alpha.empty:
        return pd.DataFrame()
    dates = pd.to_datetime(bar_alpha.index.get_level_values("date")).strftime("%Y-%m-%d").unique().tolist()
    ticks = _read_ofr_ticks(contract, dates)
    if ticks.empty:
        return ticks
    bars = bar_alpha.rename("alpha").reset_index()
    bars["date"] = pd.to_datetime(bars["date"]).dt.strftime("%Y-%m-%d")
    bars["datetime"] = pd.to_datetime(bars["datetime"]).dt.floor("15min")
    merged = ticks.merge(bars, on=["date", "datetime"], how="inner")
    if merged.empty:
        return merged
    merged["ext"] = merged["timestamp"].astype("int64").astype("float64")
    return merged[["date", "timestamp", "datetime", "security_id", "ext", "h60_price", "alpha"]].sort_values(["date", "timestamp"])


def _attach_h60_response(ticks: pd.DataFrame, *, horizon_seconds: int = 15) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    delta = np.timedelta64(int(horizon_seconds), "s")
    tolerance = np.timedelta64(2, "s")
    for (_, _), grp in ticks.groupby(["date", "security_id"], sort=True):
        grp = grp.sort_values("timestamp").copy()
        ts = grp["timestamp"].to_numpy(dtype="datetime64[ns]")
        px = pd.to_numeric(grp["h60_price"], errors="coerce").to_numpy(dtype=float)
        target = ts + delta
        pos = np.searchsorted(ts, target, side="left")
        resp = np.full(len(grp), np.nan, dtype=float)
        ok = pos < len(grp)
        if ok.any():
            pos_ok = pos[ok]
            close_enough = (ts[pos_ok] - target[ok]) <= tolerance
            idx = np.flatnonzero(ok)[close_enough]
            fut_pos = pos[idx]
            base = px[idx]
            fut = px[fut_pos]
            valid = np.isfinite(base) & np.isfinite(fut) & (base != 0)
            resp[idx[valid]] = fut[valid] / base[valid] - 1.0
        grp["resp"] = resp
        rows.append(grp)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def evaluate_tick_h60_alpha(alpha: pd.Series, *, products: Iterable[str] = PRODUCTS, horizon_seconds: int = 15) -> Dict[str, Any]:
    """Evaluate a 15m DSL alpha on the existing OFR tick grid with h60=15s forward return."""
    product_rows: list[dict[str, Any]] = []
    daily_values: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []
    for product in products:
        ticks = product_tick_signal(alpha, product)
        if ticks.empty:
            product_rows.append({"available": False, "effective": False, "product": product, "reason": "no tick overlap"})
            continue
        ticks = _attach_h60_response(ticks, horizon_seconds=horizon_seconds)
        clean = ticks[["date", "alpha", "resp"]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(clean) < 200:
            product_rows.append({"available": True, "effective": False, "product": product, "reason": "too few h60 ticks", "observations": int(len(clean))})
            continue
        product_daily = clean.groupby("date").apply(lambda g: g["alpha"].corr(g["resp"]) if g["alpha"].nunique() > 1 and g["resp"].nunique() > 1 else np.nan).dropna()
        mean_ic = float(product_daily.mean()) if not product_daily.empty else 0.0
        std_ic = float(product_daily.std()) if len(product_daily) > 1 else 0.0
        icir = mean_ic / std_ic if std_ic > 0 else 0.0
        effective = abs(mean_ic) > 0.02 and abs(icir) > 1.0
        product_rows.append({
            "available": True,
            "effective": bool(effective),
            "product": product,
            "IC": mean_ic,
            "IR": icir,
            "ICIR": icir,
            "daily_ic_std": std_ic,
            "observations": int(len(clean)),
            "days": int(product_daily.shape[0]),
        })
        for date, val in product_daily.items():
            daily_values.append({"date": str(date), "product": product, "IC": float(val)})
        frames.append(ticks)
    daily_df = pd.DataFrame(daily_values)
    if daily_df.empty:
        mean_ic = daily_std = icir = score = 0.0
        daily_ic = pd.Series(dtype=float)
    else:
        daily_ic = daily_df.groupby("date")["IC"].mean()
        mean_ic = float(daily_ic.mean()) if not daily_ic.empty else 0.0
        daily_std = float(daily_ic.std()) if len(daily_ic) > 1 else 0.0
        icir = mean_ic / daily_std if daily_std > 0 else 0.0
        score = abs(mean_ic) * math.sqrt(min(abs(icir), 6.0)) * 100.0 * 100.0
    pass_gates = abs(mean_ic) > 0.02 and abs(icir) > 1.0 and any(row.get("effective") for row in product_rows)
    return {
        "IC": mean_ic,
        "IR": icir,
        "ICIR": icir,
        "daily_ic_std": daily_std,
        "Score": score,
        "PassGates": bool(pass_gates),
        "GatesDetail": {"IC": abs(mean_ic) > 0.02, "ICIR": abs(icir) > 1.0, "EffectiveMarket": any(row.get("effective") for row in product_rows)},
        "metric_mode": f"futures_tick_h60_{horizon_seconds}s",
        "market_metrics": {row["product"]: row for row in product_rows},
        "daily_ic": daily_ic,
        "tick_frames": frames,
    }


def read_existing_alpha_15m(path: str) -> pd.DataFrame:
    """Read an existing futures alpha file and resample tick ext rows to 15-minute bars."""
    df = pd.read_parquet(path)
    if df.empty or "ext" not in df.columns:
        return pd.DataFrame()
    value_cols = [c for c in df.columns if c != "ext" and pd.api.types.is_numeric_dtype(df[c])]
    if not value_cols:
        return pd.DataFrame()
    out = df[["ext", *value_cols]].copy()
    out["datetime"] = pd.to_datetime(out["ext"].astype("int64"), errors="coerce").dt.floor("15min")
    out = out.dropna(subset=["datetime"])
    if out.empty:
        return pd.DataFrame()
    return out.groupby("datetime", sort=True)[value_cols].mean()


def read_existing_alpha_tick(path: str) -> pd.DataFrame:
    """Read an existing futures alpha file on its native ext/tick grid."""
    df = pd.read_parquet(path)
    if df.empty or "ext" not in df.columns:
        return pd.DataFrame()
    value_cols = [c for c in df.columns if c != "ext" and pd.api.types.is_numeric_dtype(df[c])]
    if not value_cols:
        return pd.DataFrame()
    out = df[["ext", *value_cols]].copy()
    out["ext_i"] = pd.to_numeric(out["ext"], errors="coerce").round().astype("Int64")
    out = out.dropna(subset=["ext_i"])
    return out.groupby("ext_i", sort=True)[value_cols].mean()


def existing_alpha_files(
    *,
    alpha_root: str = FUTURE_ALPHA_ROOT,
    dates: Optional[Iterable[str]] = None,
    products: Iterable[str] = PRODUCTS,
) -> List[str]:
    products = {p.upper() for p in products}
    date_keys = list(dates or [])
    if not date_keys:
        date_dirs = sorted(glob.glob(os.path.join(alpha_root, "[0-9]" * 8)))
    else:
        date_dirs = [os.path.join(alpha_root, d.replace("-", "")) for d in date_keys]
    files: List[str] = []
    for day_dir in date_dirs:
        if not os.path.isdir(day_dir):
            continue
        for product in products:
            files.extend(glob.glob(os.path.join(day_dir, f"*@{PRODUCT_ALIAS[product]}_*.parquet")))
    return sorted(set(files))


def compute_existing_alpha_correlations(
    alpha: pd.Series,
    factor_name: str,
    *,
    alpha_root: str = FUTURE_ALPHA_ROOT,
    products: Iterable[str] = PRODUCTS,
    max_existing: Optional[int] = None,
    top_n: int = 30,
    out_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Compare a new factor against existing futures alpha files on the native tick grid."""
    dates = _series_dates(alpha)
    files = existing_alpha_files(alpha_root=alpha_root, dates=dates, products=products)
    if max_existing is None:
        raw = os.environ.get("AUTOALPHA_CORR_MAX_EXISTING", "0")
        try:
            max_existing = int(raw)
        except ValueError:
            max_existing = 0
    if max_existing and max_existing > 0:
        if len(files) > max_existing:
            idx = np.linspace(0, len(files) - 1, num=max_existing, dtype=int)
            files = [files[int(i)] for i in idx]

    by_product_tick = {p: product_tick_signal(alpha, p) for p in products}
    by_product_15m = {p: product_series(alpha, p) for p in products}
    rows: List[Dict[str, Any]] = []
    for path in files:
        product = product_from_alpha_path(path)
        if product not in by_product_tick and product not in by_product_15m:
            continue
        candidate_tick = by_product_tick.get(product, pd.DataFrame())
        existing = read_existing_alpha_tick(path)
        if not candidate_tick.empty and not existing.empty:
            cand_s = candidate_tick.assign(ext_i=lambda x: pd.to_numeric(x["ext"], errors="coerce").round().astype("Int64")).groupby("ext_i", sort=True)["alpha"].mean()
            joined = existing.join(cand_s.rename("__new_alpha__"), how="inner")
            overlap_key = "overlap_ticks"
        else:
            existing_15m = read_existing_alpha_15m(path)
            candidate = by_product_15m.get(product, pd.Series(dtype=float))
            if existing_15m.empty or candidate.empty or not isinstance(candidate.index, pd.MultiIndex):
                continue
            cand_dt = candidate.reset_index()
            cand_dt["datetime"] = pd.to_datetime(cand_dt["datetime"]).dt.floor("15min")
            cand_s = cand_dt.groupby("datetime", sort=True)["alpha"].mean()
            joined = existing_15m.join(cand_s.rename("__new_alpha__"), how="inner")
            existing = existing_15m
            overlap_key = "overlap_bars"
        if len(joined) < 20:
            continue
        for col in existing.columns:
            corr = joined[col].corr(joined["__new_alpha__"])
            if pd.isna(corr):
                continue
            rows.append({
                "product": product,
                "existing_factor": factor_name_from_alpha_path(path),
                "existing_column": col,
                "date": os.path.basename(os.path.dirname(path)),
                "correlation": float(corr),
                "abs_correlation": float(abs(corr)),
                "overlap_bars": int(joined[[col, "__new_alpha__"]].dropna().shape[0]) if overlap_key == "overlap_bars" else 0,
                "overlap_ticks": int(joined[[col, "__new_alpha__"]].dropna().shape[0]) if overlap_key == "overlap_ticks" else 0,
                "path": path,
            })

    detail = pd.DataFrame(rows)
    if detail.empty:
        summary = pd.DataFrame(columns=["product", "existing_factor", "existing_column", "mean_corr", "max_abs_corr", "observations", "overlap_bars"])
    else:
        summary = (
            detail.groupby(["product", "existing_factor", "existing_column"], sort=True)
            .agg(
                mean_corr=("correlation", "mean"),
                max_abs_corr=("abs_correlation", "max"),
                observations=("correlation", "count"),
                overlap_bars=("overlap_bars", "sum"),
                overlap_ticks=("overlap_ticks", "sum"),
            )
            .reset_index()
            .sort_values("max_abs_corr", ascending=False)
        )

    max_abs_corr = float(summary["max_abs_corr"].max()) if not summary.empty else 0.0
    mean_top_abs_corr = float(summary["max_abs_corr"].head(max(1, min(top_n, len(summary)))).mean()) if not summary.empty else 0.0
    product_max = {
        p: float(summary.loc[summary["product"] == p, "max_abs_corr"].max())
        for p in products
        if not summary.loc[summary["product"] == p].empty
    }
    report = {
        "factor_name": factor_name,
        "alpha_root": alpha_root,
        "dates": dates,
        "files_scanned": len(files),
        "pairs_evaluated": int(len(detail)),
        "max_abs_corr": max_abs_corr,
        "mean_top_abs_corr": mean_top_abs_corr,
        "product_max_abs_corr": product_max,
        "top": summary.head(top_n).to_dict(orient="records"),
    }
    if out_dir:
        paths = save_correlation_report(report, detail, summary, out_dir=out_dir)
        report.update(paths)
    return report


def save_correlation_report(report: Dict[str, Any], detail: pd.DataFrame, summary: pd.DataFrame, *, out_dir: str) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "existing_alpha_correlation.json")
    detail_path = os.path.join(out_dir, "existing_alpha_correlation_detail.csv")
    summary_path = os.path.join(out_dir, "existing_alpha_correlation_summary.csv")
    png_path = os.path.join(out_dir, "existing_alpha_correlation_top.png")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, default=str)
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        top = summary.head(25).copy()
        if not top.empty:
            top["label"] = top["product"] + " " + top["existing_factor"] + ":" + top["existing_column"]
            fig_h = max(4.0, 0.24 * len(top) + 1.2)
            fig, ax = plt.subplots(figsize=(10, fig_h))
            colors = np.where(top["mean_corr"].to_numpy() >= 0, "#2563eb", "#f97316")
            ax.barh(top["label"][::-1], top["max_abs_corr"][::-1], color=colors[::-1])
            ax.set_xlabel("max |corr| vs new factor")
            ax.set_title(f"Existing alpha correlation: {report.get('factor_name', '')}")
            ax.set_xlim(0, min(1.0, max(0.2, float(top["max_abs_corr"].max()) * 1.15)))
            fig.tight_layout()
            fig.savefig(png_path, dpi=160)
            plt.close(fig)
    except Exception as exc:
        report["plot_error"] = str(exc)
    return {
        "correlation_json_path": json_path,
        "correlation_detail_csv_path": detail_path,
        "correlation_summary_csv_path": summary_path,
        "correlation_plot_path": png_path if os.path.exists(png_path) else "",
    }


def export_future_alpha_format(
    alpha: pd.Series,
    factor_name: str,
    *,
    out_root: str = FUTURE_ALPHA_ROOT,
    products: Iterable[str] = PRODUCTS,
    column_name: Optional[str] = None,
) -> List[str]:
    """Save alpha in existing futures-alpha layout on the native OFR tick ext grid."""
    fallback_root = os.path.join(OUTPUTS_ROOT, "future_alpha")
    target_root = out_root
    try:
        os.makedirs(target_root, exist_ok=True)
        probe = os.path.join(target_root, ".autoalpha_write_probe")
        with open(probe, "w", encoding="utf-8") as handle:
            handle.write("ok")
        os.remove(probe)
    except Exception:
        target_root = fallback_root
        os.makedirs(target_root, exist_ok=True)

    safe_factor = re.sub(r"[^A-Za-z0-9_]+", "_", factor_name).strip("_") or "AutoAlpha"
    value_col = column_name or safe_factor[:48]
    if alpha.empty:
        return []
    written: List[str] = []
    for product in products:
        tick_signal = product_tick_signal(alpha, product)
        if tick_signal.empty:
            continue
        tick_signal[value_col] = pd.to_numeric(tick_signal["alpha"], errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float64")
        for date, day_frame in tick_signal.groupby("date", sort=True):
            date_key = _date_key(date)
            day_dir = os.path.join(target_root, date_key)
            os.makedirs(day_dir, exist_ok=True)
            day_frame = day_frame[["ext", value_col]]
            out_path = os.path.join(day_dir, f"{safe_factor}@{PRODUCT_ALIAS[product]}_{date_key}.parquet")
            day_frame.to_parquet(out_path, engine="pyarrow", index=False)
            written.append(out_path)
    return written


def futures_research_score(metrics: Dict[str, Any], corr_report: Optional[Dict[str, Any]] = None, market_metrics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Rank factors by out-of-sample robustness, market breadth and novelty instead of legacy Score."""
    ic = abs(float(metrics.get("IC", 0.0) or 0.0))
    rank_ic = abs(float(metrics.get("rank_ic", 0.0) or 0.0) * 100.0)
    ir = max(0.0, float(metrics.get("IR", 0.0) or 0.0))
    tvr = max(0.0, float(metrics.get("Turnover", 0.0) or 0.0))
    stability = max(0.0, float(metrics.get("stability_score", 0.0) or 0.0))
    novelty_penalty = float((corr_report or {}).get("max_abs_corr", 0.0) or 0.0)
    market_count = 0
    if market_metrics:
        market_count = sum(1 for m in market_metrics.values() if m.get("effective"))
    breadth = market_count / float(len(PRODUCTS))
    turnover_penalty = math.log1p(tvr) / math.log1p(500.0) if tvr > 0 else 0.0
    raw = (
        0.30 * min(ic / 1.0, 2.0)
        + 0.20 * min(rank_ic / 1.0, 2.0)
        + 0.18 * min(ir / 2.0, 2.0)
        + 0.14 * stability
        + 0.12 * breadth
        + 0.06 * max(0.0, 1.0 - turnover_penalty)
    )
    novelty = max(0.0, 1.0 - novelty_penalty)
    score = 100.0 * raw * novelty
    return {
        "futures_score": float(score),
        "score_mode": "futures_oos_novelty_v1",
        "components": {
            "ic_abs_bps": ic,
            "rank_ic_abs_bps": rank_ic,
            "ir": ir,
            "turnover": tvr,
            "stability": stability,
            "effective_market_count": market_count,
            "breadth": breadth,
            "max_existing_abs_corr": novelty_penalty,
            "novelty_multiplier": novelty,
        },
        "formula": "100 * weighted(IC,RankIC,IR,stability,breadth,low_turnover) * (1 - max_existing_abs_corr)",
    }
