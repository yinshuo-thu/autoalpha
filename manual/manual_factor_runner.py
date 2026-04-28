from __future__ import annotations

import argparse
import ast
import gc
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.submission import ALLOWED_UTC_TIMES, SubmissionBuilder
from core.evaluator import evaluate_submission_like_wide
from prepare_data import DataHub


EPS = 1e-8
MANUAL_ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = MANUAL_ROOT / "artifacts"
REPORTS_DIR = MANUAL_ROOT / "reports"
SUBMIT_ROOT = MANUAL_ROOT / "submit"
FORBIDDEN_MANUAL_TOKENS = ("resp", "trading_restriction", "lead(", "future_")


@dataclass(frozen=True)
class CandidateSpec:
    family: str
    family_label: str
    params: dict[str, Any]
    direction: int
    expression: str
    description: str

    @property
    def direction_label(self) -> str:
        return "pro" if self.direction > 0 else "anti"

    @property
    def key(self) -> str:
        param_bits = [f"{name}{self.params[name]}" for name in sorted(self.params)]
        suffix = "_".join(param_bits) if param_bits else "base"
        return f"{self.family}__{suffix}__{self.direction_label}"


class ManualFactorDataset:
    def __init__(self) -> None:
        self.hub = DataHub()
        self._load()

    def _load(self) -> None:
        load_start = time.time()
        pv = self.hub.pv_15m[
            [
                "open_trade_px",
                "high_trade_px",
                "low_trade_px",
                "close_trade_px",
                "volume",
                "dvolume",
                "trade_count",
                "vwap",
            ]
        ]
        self.load_seconds_pv = time.time() - load_start

        unstack_start = time.time()
        self.open_ = pv["open_trade_px"].unstack("security_id").astype("float32")
        self.high = pv["high_trade_px"].unstack("security_id").astype("float32")
        self.low = pv["low_trade_px"].unstack("security_id").astype("float32")
        self.close = pv["close_trade_px"].unstack("security_id").astype("float32")
        self.volume = pv["volume"].unstack("security_id").astype("float32")
        self.dvolume = pv["dvolume"].unstack("security_id").astype("float32")
        self.trade_count = pv["trade_count"].unstack("security_id").astype("float32")
        self.vwap = pv["vwap"].unstack("security_id").astype("float32")
        self.load_seconds_unstack = time.time() - unstack_start

        align_start = time.time()
        resp_wide = self.hub.resp["resp"].unstack("security_id")
        self.resp = resp_wide.reindex(self.close.index).astype("float32")
        rest_wide = self.hub.trading_restriction["trading_restriction"].unstack("security_id")
        self.restriction = rest_wide.reindex(self.close.index).fillna(0).astype("float32")
        self.load_seconds_labels = time.time() - align_start

        self.index = self.close.index
        self.columns = self.close.columns
        self.trading_days = self.hub.get_trading_days_list()
        self.start_date = self.trading_days[0]
        self.end_date = self.trading_days[-1]
        self.security_count = len(self.columns)
        self.bar_count = len(self.index)
        self._index_frame = self.index.to_frame(index=False)


def audit_manual_spec(spec: CandidateSpec) -> dict[str, Any]:
    expression = (spec.expression or "").replace(" ", "").lower()
    errors: list[str] = []

    supported_families = {
        "close_zscore",
        "range_location",
        "body_fraction",
        "wick_imbalance",
        "vwap_gap",
        "open_close_return",
        "gap_return",
        "bar_return",
        "volume_conditioned_return",
        "dvolume_conditioned_return",
        "trade_conditioned_return",
        "avg_trade_conditioned_return",
        "volatility_conditioned_return",
        "range_conditioned_body",
        "range_conditioned_location",
        "vwap_gap_with_dvol",
        "ema_spread",
        "multi_horizon_mix",
        "zscore_vwap_gap",
        "zscore_body_fraction",
    }
    if spec.family not in supported_families:
        errors.append(f"Unsupported manual family for causality audit: {spec.family}")

    for token in FORBIDDEN_MANUAL_TOKENS:
        if token in expression:
            errors.append(f"Forbidden token in manual factor expression: {token}")

    if re.search(r"delay\([^)]*,-\d+", expression):
        errors.append("Negative delay detected in manual factor expression")

    for key, value in (spec.params or {}).items():
        if isinstance(value, (int, float)) and value <= 0:
            errors.append(f"Parameter '{key}' must be > 0, got {value}")

    notes = [
        "manual factors are restricted to fixed price-volume families implemented in compute_raw()",
        "only contemporaneous bars, positive lags, rolling windows, and positive-span EMA are allowed",
        "resp and trading_restriction are touched only after construction during evaluation",
    ]
    return {
        "passed": not errors,
        "errors": errors,
        "notes": notes,
    }


def rolling_mean(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    return frame.rolling(window, min_periods=2).mean()


def rolling_std(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    return frame.rolling(window, min_periods=2).std()


def ts_zscore(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    mean = rolling_mean(frame, window)
    std = rolling_std(frame, window)
    return (frame - mean) / (std + EPS)


def safe_div(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    return left / right.replace(0, np.nan)


def cs_rank(frame: pd.DataFrame) -> pd.DataFrame:
    counts = frame.notna().sum(axis=1)
    ranked = frame.rank(axis=1, method="average")
    return (ranked - 1).div((counts - 1).replace(0, np.nan), axis=0)


def apply_restriction(alpha: pd.DataFrame, restriction: pd.DataFrame) -> pd.DataFrame:
    blocked = (
        ((restriction == 1) & (alpha < 0))
        | ((restriction == 2) & (alpha > 0))
        | (restriction == 3)
    ).fillna(False)
    return alpha.where(~blocked)


def evaluate_alpha(alpha: pd.DataFrame, dataset: ManualFactorDataset) -> dict[str, Any]:
    alpha_tradeable = apply_restriction(alpha, dataset.restriction)
    bar_ic = alpha_tradeable.corrwith(dataset.resp, axis=1, method="pearson")
    daily_ic = bar_ic.groupby(level="date").mean()
    ic_raw = float(daily_ic.mean()) if not daily_ic.empty else 0.0
    ic_std = float(daily_ic.std()) if not daily_ic.empty else 0.0
    ir = 0.0 if ic_std == 0 or math.isnan(ic_std) else float(ic_raw / ic_std * math.sqrt(252))

    turnover_bar = alpha.diff().abs().sum(axis=1) / alpha.abs().sum(axis=1).replace(0, np.nan)
    daily_turnover = turnover_bar.groupby(level="date").sum()
    tvr = float(daily_turnover.mean()) if not daily_turnover.empty else 0.0

    abs_sum = alpha.abs().sum(axis=1).replace(0, np.nan)
    weights = alpha.div(abs_sum, axis=0) * 10_000
    max_weight = weights.max(axis=1).dropna()
    min_weight = weights.min(axis=1).dropna()

    maxx = float(max_weight.max()) if not max_weight.empty else 0.0
    minn = float(min_weight.min()) if not min_weight.empty else 0.0
    max_mean = float(max_weight.groupby(level="date").max().mean()) if not max_weight.empty else 0.0
    min_mean = float(min_weight.groupby(level="date").min().mean()) if not min_weight.empty else 0.0
    positive_ic_ratio = float((daily_ic > 0).mean()) if not daily_ic.empty else 0.0

    pass_gates = (
        ic_raw > 0.006
        and ir > 2.5
        and tvr < 400
        and maxx < 50
        and abs(minn) < 50
        and max_mean < 20
        and abs(min_mean) < 20
    )
    score = max(0.0, ic_raw - 0.0005 * tvr) * math.sqrt(ir) * 100 if pass_gates else 0.0

    return {
        "IC": ic_raw,
        "IR": ir,
        "Turnover": tvr,
        "Score": float(score),
        "PassGates": bool(pass_gates),
        "maxx": maxx,
        "minn": minn,
        "max_mean": max_mean,
        "min_mean": min_mean,
        "positive_ic_ratio": positive_ic_ratio,
        "daily_ic_mean_bps": ic_raw * 100,
    }


def generate_candidates() -> list[CandidateSpec]:
    candidates: list[CandidateSpec] = []

    def add_signed(
        family: str,
        family_label: str,
        params_list: list[dict[str, Any]],
        expression_builder,
        description_builder,
    ) -> None:
        for params in params_list:
            for direction in (-1, 1):
                candidates.append(
                    CandidateSpec(
                        family=family,
                        family_label=family_label,
                        params=params,
                        direction=direction,
                        expression=expression_builder(params, direction),
                        description=description_builder(params, direction),
                    )
                )

    add_signed(
        "close_zscore",
        "Close Z-Score Stretch",
        [{"window": window} for window in (8, 15, 24)],
        lambda p, d: f"rank({d:+d} * zscore(close_trade_px, {p['window']}))",
        lambda p, d: (
            f"{'Follow' if d > 0 else 'Fade'} rolling {p['window']}-bar close z-score stretch; "
            "captures short-horizon intraday price extension."
        ),
    )
    add_signed(
        "range_location",
        "Close Location In Range",
        [{}],
        lambda _p, d: f"rank({d:+d} * ((close-low)/(high-low+eps)))",
        lambda _p, d: (
            f"{'Reward' if d > 0 else 'Fade'} closes near the intrabar high relative to the low-high range."
        ),
    )
    add_signed(
        "body_fraction",
        "Body Fraction",
        [{}],
        lambda _p, d: f"rank({d:+d} * ((close-open)/(high-low+eps)))",
        lambda _p, d: (
            f"{'Follow' if d > 0 else 'Fade'} strong candle bodies after scaling by current bar range."
        ),
    )
    add_signed(
        "wick_imbalance",
        "Wick Imbalance",
        [{}],
        lambda _p, d: f"rank({d:+d} * ((lower_wick-upper_wick)/(high-low+eps)))",
        lambda _p, d: (
            f"{'Reward' if d > 0 else 'Fade'} bars with stronger lower wick support than upper wick pressure."
        ),
    )
    add_signed(
        "vwap_gap",
        "Close-VWAP Gap",
        [{}],
        lambda _p, d: f"rank({d:+d} * (close_trade_px/vwap - 1))",
        lambda _p, d: (
            f"{'Follow' if d > 0 else 'Fade'} the instantaneous premium or discount of close versus VWAP."
        ),
    )
    add_signed(
        "open_close_return",
        "Open-Close Return",
        [{}],
        lambda _p, d: f"rank({d:+d} * (close_trade_px/open_trade_px - 1))",
        lambda _p, d: (
            f"{'Follow' if d > 0 else 'Fade'} same-bar open-to-close move."
        ),
    )
    add_signed(
        "gap_return",
        "Open To Previous Close Gap",
        [{}],
        lambda _p, d: f"rank({d:+d} * (open_trade_px/delay(close_trade_px,1) - 1))",
        lambda _p, d: (
            f"{'Follow' if d > 0 else 'Fade'} the gap between current open and previous close."
        ),
    )
    add_signed(
        "bar_return",
        "Close To Previous Close Return",
        [{}],
        lambda _p, d: f"rank({d:+d} * (close_trade_px/delay(close_trade_px,1) - 1))",
        lambda _p, d: (
            f"{'Follow' if d > 0 else 'Fade'} the last-bar close-to-close return."
        ),
    )
    add_signed(
        "volume_conditioned_return",
        "Return x Volume Surprise",
        [{"window": window} for window in (8, 16)],
        lambda p, d: (
            f"rank({d:+d} * close_ret * (volume/mean(volume,{p['window']})))"
        ),
        lambda p, d: (
            f"{'Follow' if d > 0 else 'Fade'} close-to-close return when volume is elevated against a "
            f"{p['window']}-bar baseline."
        ),
    )
    add_signed(
        "dvolume_conditioned_return",
        "Return x Dollar-Volume Surprise",
        [{"window": window} for window in (8, 16)],
        lambda p, d: (
            f"rank({d:+d} * close_ret * (dvolume/mean(dvolume,{p['window']})))"
        ),
        lambda p, d: (
            f"{'Follow' if d > 0 else 'Fade'} return intensity when dollar volume is unusually large over "
            f"{p['window']} bars."
        ),
    )
    add_signed(
        "trade_conditioned_return",
        "Return x Trade-Count Surprise",
        [{"window": window} for window in (8, 16)],
        lambda p, d: (
            f"rank({d:+d} * close_ret * (trade_count/mean(trade_count,{p['window']})))"
        ),
        lambda p, d: (
            f"{'Follow' if d > 0 else 'Fade'} short return moves when quote/print activity spikes versus "
            f"a {p['window']}-bar average."
        ),
    )
    add_signed(
        "avg_trade_conditioned_return",
        "Return x Average Trade Value Surprise",
        [{"window": window} for window in (8, 16)],
        lambda p, d: (
            f"rank({d:+d} * close_ret * ((dvolume/trade_count)/mean(dvolume/trade_count,{p['window']})))"
        ),
        lambda p, d: (
            f"{'Follow' if d > 0 else 'Fade'} moves when average trade value is stretched over a "
            f"{p['window']}-bar context."
        ),
    )
    add_signed(
        "volatility_conditioned_return",
        "Return / Recent Volatility",
        [{"window": window} for window in (16, 32)],
        lambda p, d: f"rank({d:+d} * close_ret / std(close_ret,{p['window']}))",
        lambda p, d: (
            f"{'Follow' if d > 0 else 'Fade'} bar return after normalizing by {p['window']}-bar realized volatility."
        ),
    )
    add_signed(
        "range_conditioned_body",
        "Body x Range Surprise",
        [{"window": window} for window in (8, 16)],
        lambda p, d: (
            f"rank({d:+d} * body_frac * (range_pct/mean(range_pct,{p['window']})))"
        ),
        lambda p, d: (
            f"{'Follow' if d > 0 else 'Fade'} candle body when the contemporaneous range is a "
            f"{p['window']}-bar outlier."
        ),
    )
    add_signed(
        "range_conditioned_location",
        "Range Location x Range Surprise",
        [{"window": window} for window in (8, 16)],
        lambda p, d: (
            f"rank({d:+d} * range_loc * (range_pct/mean(range_pct,{p['window']})))"
        ),
        lambda p, d: (
            f"{'Follow' if d > 0 else 'Fade'} close location in the bar when range expansion exceeds its "
            f"{p['window']}-bar norm."
        ),
    )
    add_signed(
        "vwap_gap_with_dvol",
        "VWAP Gap x Dollar-Volume Surprise",
        [{"window": window} for window in (8, 16)],
        lambda p, d: (
            f"rank({d:+d} * (close_trade_px/vwap-1) * (dvolume/mean(dvolume,{p['window']})))"
        ),
        lambda p, d: (
            f"{'Follow' if d > 0 else 'Fade'} the close-VWAP gap when dollar turnover is hot versus "
            f"{p['window']}-bar history."
        ),
    )
    add_signed(
        "ema_spread",
        "Short-Long EMA Spread",
        [{"short": 4, "long": 16}, {"short": 8, "long": 32}],
        lambda p, d: (
            f"rank({d:+d} * (ema(close,{p['short']})/ema(close,{p['long']}) - 1))"
        ),
        lambda p, d: (
            f"{'Follow' if d > 0 else 'Fade'} the short/long EMA spread using {p['short']} vs {p['long']} bars."
        ),
    )
    add_signed(
        "multi_horizon_mix",
        "1/4/16-Bar Return Mix",
        [{}],
        lambda _p, d: "rank({:+d} * (ret1 + 0.5*ret4 + 0.25*ret16))".format(d),
        lambda _p, d: (
            f"{'Follow' if d > 0 else 'Fade'} a blended 1/4/16-bar return stack."
        ),
    )
    add_signed(
        "zscore_vwap_gap",
        "VWAP Gap Z-Score",
        [{"window": window} for window in (8, 16)],
        lambda p, d: f"rank({d:+d} * zscore(close_trade_px/vwap - 1,{p['window']}))",
        lambda p, d: (
            f"{'Follow' if d > 0 else 'Fade'} the rolling {p['window']}-bar z-score of the close-VWAP gap."
        ),
    )
    add_signed(
        "zscore_body_fraction",
        "Body Fraction Z-Score",
        [{"window": window} for window in (8, 16)],
        lambda p, d: f"rank({d:+d} * zscore((close-open)/(high-low+eps),{p['window']}))",
        lambda p, d: (
            f"{'Follow' if d > 0 else 'Fade'} the rolling {p['window']}-bar z-score of candle body fraction."
        ),
    )
    return candidates


def compute_raw(spec: CandidateSpec, dataset: ManualFactorDataset) -> pd.DataFrame:
    future_guard = audit_manual_spec(spec)
    if not future_guard["passed"]:
        raise ValueError(f"Manual future-info audit failed for {spec.key}: {'; '.join(future_guard['errors'])}")

    close = dataset.close
    open_ = dataset.open_
    high = dataset.high
    low = dataset.low
    volume = dataset.volume
    dvolume = dataset.dvolume
    trade_count = dataset.trade_count
    vwap = dataset.vwap

    prev_close = close.shift(1)
    close_ret = safe_div(close, prev_close) - 1.0
    range_pct = safe_div(high - low, close.abs() + EPS)
    range_loc = safe_div(close - low, (high - low) + EPS)
    body_frac = safe_div(close - open_, (high - low) + EPS)
    upper_wick = high - np.maximum(open_, close)
    lower_wick = np.minimum(open_, close) - low
    wick_imbalance = safe_div(lower_wick - upper_wick, (high - low) + EPS)

    family = spec.family
    params = spec.params

    if family == "close_zscore":
        raw = ts_zscore(close, params["window"])
    elif family == "range_location":
        raw = range_loc
    elif family == "body_fraction":
        raw = body_frac
    elif family == "wick_imbalance":
        raw = wick_imbalance
    elif family == "vwap_gap":
        raw = safe_div(close, vwap) - 1.0
    elif family == "open_close_return":
        raw = safe_div(close, open_) - 1.0
    elif family == "gap_return":
        raw = safe_div(open_, prev_close) - 1.0
    elif family == "bar_return":
        raw = close_ret
    elif family == "volume_conditioned_return":
        raw = close_ret * safe_div(volume, rolling_mean(volume, params["window"]) + EPS)
    elif family == "dvolume_conditioned_return":
        raw = close_ret * safe_div(dvolume, rolling_mean(dvolume, params["window"]) + EPS)
    elif family == "trade_conditioned_return":
        raw = close_ret * safe_div(trade_count, rolling_mean(trade_count, params["window"]) + EPS)
    elif family == "avg_trade_conditioned_return":
        avg_trade = safe_div(dvolume, trade_count + EPS)
        raw = close_ret * safe_div(avg_trade, rolling_mean(avg_trade, params["window"]) + EPS)
    elif family == "volatility_conditioned_return":
        raw = safe_div(close_ret, rolling_std(close_ret, params["window"]) + EPS)
    elif family == "range_conditioned_body":
        range_ratio = safe_div(range_pct, rolling_mean(range_pct, params["window"]) + EPS)
        raw = body_frac * range_ratio
    elif family == "range_conditioned_location":
        range_ratio = safe_div(range_pct, rolling_mean(range_pct, params["window"]) + EPS)
        raw = range_loc * range_ratio
    elif family == "vwap_gap_with_dvol":
        dvol_ratio = safe_div(dvolume, rolling_mean(dvolume, params["window"]) + EPS)
        raw = (safe_div(close, vwap) - 1.0) * dvol_ratio
    elif family == "ema_spread":
        ema_short = close.ewm(span=params["short"], adjust=False, min_periods=params["short"]).mean()
        ema_long = close.ewm(span=params["long"], adjust=False, min_periods=params["long"]).mean()
        raw = safe_div(ema_short, ema_long + EPS) - 1.0
    elif family == "multi_horizon_mix":
        ret1 = safe_div(close, close.shift(1)) - 1.0
        ret4 = safe_div(close, close.shift(4)) - 1.0
        ret16 = safe_div(close, close.shift(16)) - 1.0
        raw = ret1 + 0.5 * ret4 + 0.25 * ret16
    elif family == "zscore_vwap_gap":
        gap = safe_div(close, vwap) - 1.0
        raw = ts_zscore(gap, params["window"])
    elif family == "zscore_body_fraction":
        raw = ts_zscore(body_frac, params["window"])
    else:
        raise ValueError(f"Unsupported family: {family}")

    return spec.direction * raw


def select_candidates(results: pd.DataFrame, top_k: int) -> pd.DataFrame:
    passing = results[results["PassGates"]].copy()
    if passing.empty:
        return passing

    passing = passing.sort_values(["Score", "IC", "IR"], ascending=False).reset_index(drop=True)
    family_bests = (
        passing.sort_values(["family", "Score", "IC"], ascending=[True, False, False])
        .groupby("family", as_index=False)
        .head(1)
        .sort_values(["Score", "IC"], ascending=False)
    )

    selected_keys = []
    for key in family_bests["key"]:
        if key not in selected_keys:
            selected_keys.append(key)
        if len(selected_keys) >= top_k:
            break

    if len(selected_keys) < top_k:
        for key in passing["key"]:
            if key not in selected_keys:
                selected_keys.append(key)
            if len(selected_keys) >= top_k:
                break

    return passing[passing["key"].isin(selected_keys)].copy().sort_values(
        ["Score", "IC", "IR"], ascending=False
    )


def ensure_dirs() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    SUBMIT_ROOT.mkdir(parents=True, exist_ok=True)


def format_number(value: Any, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return f"{value:.{digits}f}" if isinstance(value, (float, np.floating)) else str(value)


def build_markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(title for title, _ in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(key, "")) for _, key in columns) + " |")
    return "\n".join([header, divider, *body])


def export_submission(
    alpha: pd.DataFrame,
    spec: CandidateSpec,
    metrics: dict[str, Any],
    export_rank: int,
    run_stamp: str,
    dataset: ManualFactorDataset,
) -> dict[str, Any]:
    factor_name = f"manual_alpha_{export_rank:03d}"
    submission_metrics = evaluate_submission_like_wide(alpha, dataset.resp, dataset.restriction)

    alpha_long = alpha.stack(dropna=False).rename("alpha").to_frame().reset_index()
    alpha_long.columns = ["date", "datetime", "security_id", "alpha"]
    alpha_long["date"] = pd.to_datetime(alpha_long["date"])
    alpha_long["datetime"] = pd.to_datetime(alpha_long["datetime"], utc=True).dt.tz_localize(None)
    alpha_long = alpha_long[
        alpha_long["datetime"].dt.strftime("%H:%M:%S").isin(ALLOWED_UTC_TIMES)
    ]
    alpha_long = alpha_long.set_index(["date", "datetime", "security_id"]).sort_index()
    alpha_expanded = SubmissionBuilder.expand_to_full_grid(alpha_long, dataset.start_date, dataset.end_date)
    alpha_expanded["alpha"] = alpha_expanded["alpha"].fillna(0.0).clip(-1.0, 1.0)

    sanity_report = SubmissionBuilder.pre_submit_sanity_check(alpha_expanded, dataset.start_date, dataset.end_date)
    suffix_flag = "y" if (submission_metrics.get("PassGates", False) and sanity_report.get("submission_ready", False)) else "n"
    output_dir = SUBMIT_ROOT / f"{factor_name}_{run_stamp}_{suffix_flag}"
    output_dir.mkdir(parents=True, exist_ok=True)
    submission_path = output_dir / f"{factor_name}_submission.pq"
    SubmissionBuilder.build(alpha_expanded, str(submission_path))

    metadata_path = output_dir / f"{factor_name}_metadata.json"
    metadata = {
        "factor_name": factor_name,
        "display_name": f"{factor_name}_{spec.family}",
        "family": spec.family,
        "family_label": spec.family_label,
        "params": spec.params,
        "direction": spec.direction,
        "expression": spec.expression,
        "description": spec.description,
        "PassGates": bool(submission_metrics.get("PassGates", False)),
        "Score": float(submission_metrics.get("Score", 0.0)),
        "IC": float(submission_metrics.get("IC", 0.0)),
        "IR": float(submission_metrics.get("IR", 0.0)),
        "Turnover": float(submission_metrics.get("Turnover", 0.0)),
        "maxx": float(submission_metrics.get("maxx", 0.0)),
        "minn": float(submission_metrics.get("minn", 0.0)),
        "max_mean": float(submission_metrics.get("max_mean", 0.0)),
        "min_mean": float(submission_metrics.get("min_mean", 0.0)),
        "metric_mode": submission_metrics.get("metric_mode", "research"),
        "research_metrics": {
            "PassGates": bool(metrics["PassGates"]),
            "Score": float(metrics["Score"]),
            "IC": float(metrics["IC"]),
            "IR": float(metrics["IR"]),
            "Turnover": float(metrics["Turnover"]),
            "maxx": float(metrics["maxx"]),
            "minn": float(metrics["minn"]),
            "max_mean": float(metrics["max_mean"]),
            "min_mean": float(metrics["min_mean"]),
            "positive_ic_ratio": float(metrics["positive_ic_ratio"]),
        },
        "submission_path": str(submission_path),
        "submission_dir": str(output_dir),
        "metadata_path": str(metadata_path),
        "timestamp": run_stamp,
        "sanity_report": sanity_report,
        "future_info_check": audit_manual_spec(spec),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    result_preview = submission_metrics.get("result_preview")
    if result_preview:
        result_preview_path = output_dir / f"{factor_name}_official_like_result.json"
        preview_payload = [dict(result_preview, cover_all=int(sanity_report.get("cover_all", 0)))]
        result_preview_path.write_text(json.dumps(preview_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    return metadata


def rows_to_specs(rows: pd.DataFrame) -> list[tuple[CandidateSpec, dict[str, Any]]]:
    specs: list[tuple[CandidateSpec, dict[str, Any]]] = []
    for row in rows.to_dict(orient="records"):
        params = row["params"]
        if not isinstance(params, dict):
            params = ast.literal_eval(str(params))
        spec = CandidateSpec(
            family=row["family"],
            family_label=row["family_label"],
            params=params,
            direction=int(row["direction"]),
            expression=row["expression"],
            description=row["description"],
        )
        specs.append((spec, row))
    return specs


def export_from_existing_results(
    csv_path: Path,
    dataset: ManualFactorDataset,
    passing_only: bool = True,
    export_limit: int | None = None,
    row_offset: int = 0,
    rank_offset: int = 0,
    run_stamp_override: str | None = None,
) -> tuple[list[dict[str, Any]], str]:
    results_df = pd.read_csv(csv_path)
    if passing_only:
        results_df = results_df[results_df["PassGates"]].copy()
    results_df = results_df.sort_values(["Score", "IC", "IR"], ascending=False).reset_index(drop=True)
    if row_offset:
        results_df = results_df.iloc[row_offset:].reset_index(drop=True)
    if export_limit is not None:
        results_df = results_df.head(export_limit).copy()
    if results_df.empty:
        raise RuntimeError(f"No rows to export from {csv_path}")

    run_stamp = run_stamp_override or datetime.now().strftime("%Y%m%d_%H%M%S")
    exported: list[dict[str, Any]] = []
    specs = rows_to_specs(results_df)

    print(
        f"[manual] export-only mode | source={csv_path} rows={len(specs)} "
        f"passing_only={passing_only} run_stamp={run_stamp}"
    )

    for rank, (spec, row) in enumerate(specs, start=1 + rank_offset):
        started = time.time()
        raw = compute_raw(spec, dataset)
        alpha = cs_rank(raw).astype("float32").replace([np.inf, -np.inf], np.nan)
        metadata = export_submission(alpha, spec, row, rank, run_stamp, dataset)
        elapsed = time.time() - started
        exported.append(metadata)
        print(
            f"[manual] exported {rank:02d}/{len(specs)} {metadata['factor_name']} "
            f"| family={metadata['family']} score={metadata['Score']:.3f} "
            f"| ready={metadata['sanity_report'].get('submission_ready')} "
            f"| elapsed={elapsed:.1f}s"
        )
        del raw, alpha
        gc.collect()

    export_json = ARTIFACTS_DIR / f"manual_factor_export_manifest_{run_stamp}.json"
    export_csv = ARTIFACTS_DIR / f"manual_factor_export_manifest_{run_stamp}.csv"
    export_json.write_text(json.dumps(exported, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(exported).to_csv(export_csv, index=False)
    print(f"[manual] wrote export manifest -> {export_json}")
    print(f"[manual] wrote export manifest -> {export_csv}")
    return exported, run_stamp


def build_report(
    run_stamp: str,
    dataset: ManualFactorDataset,
    candidates: list[CandidateSpec],
    results: pd.DataFrame,
    selected: pd.DataFrame,
    exported: list[dict[str, Any]],
) -> str:
    candidate_count = len(candidates)
    passing_count = int(results["PassGates"].sum()) if not results.empty else 0
    family_count = results["family"].nunique() if not results.empty else 0
    exported_by_key = {
        f"{item['family']}__{item['params']}__{item['direction']}": item["submission_dir"]
        for item in exported
    }

    selected_rows = []
    for rank, row in enumerate(selected.itertuples(index=False), start=1):
        selected_rows.append(
            {
                "Rank": rank,
                "Factor": row.export_name,
                "Family": row.family_label,
                "Params": json.dumps(row.params, ensure_ascii=False),
                "Dir": row.direction_label,
                "IC": format_number(row.IC, 4),
                "IR": format_number(row.IR, 2),
                "Turnover": format_number(row.Turnover, 2),
                "Score": format_number(row.Score, 3),
                "Expr": row.expression,
            }
        )

    exploration_rows = []
    for row in results.sort_values(["Score", "IC"], ascending=False).itertuples(index=False):
        exploration_rows.append(
            {
                "Key": row.key,
                "Family": row.family_label,
                "Params": json.dumps(row.params, ensure_ascii=False),
                "Dir": row.direction_label,
                "Pass": "Y" if row.PassGates else "N",
                "IC": format_number(row.IC, 4),
                "IR": format_number(row.IR, 2),
                "Turnover": format_number(row.Turnover, 2),
                "Score": format_number(row.Score, 3),
            }
        )

    top5 = results.sort_values(["Score", "IC"], ascending=False).head(5)
    top5_text = []
    for row in top5.itertuples(index=False):
        top5_text.append(
            f"- `{row.key}` | Score={row.Score:.3f} | IC={row.IC:.4f} | "
            f"IR={row.IR:.2f} | Turnover={row.Turnover:.2f}"
        )

    report = f"""# Manual Factor Research Report

## Objective
- Build a dedicated `manual/` research path for directly computed, hand-designed factors.
- Keep the construction competition-safe: only price/volume style inputs, no `resp`, no `trading_restriction`, no future information.
- Prioritize official gate passage and higher `Score = (IC - 0.0005 * Turnover) * sqrt(IR) * 100`.

## Dataset And Evaluation Setup
- Data source: cached 15-minute bars from `prepare_data.py` / `DataHub`.
- Coverage: `{dataset.start_date}` to `{dataset.end_date}`.
- Intraday grid loaded for research: `{dataset.bar_count}` bars x `{dataset.security_count}` securities.
- Base fields used: `open_trade_px`, `high_trade_px`, `low_trade_px`, `close_trade_px`, `vwap`, `volume`, `dvolume`, `trade_count`.
- Labels / constraints were used strictly for evaluation after factor construction.
- Final alpha post-process: every candidate is converted into a cross-sectional rank in `[0, 1]` before official metrics.

## Research Process
1. Load the full cached 15-minute panel once, then unstack to wide matrices for faster repeated evaluation.
2. Explore multiple manual families: short-term price stretch, bar-shape structure, VWAP gaps, gap/return continuation-vs-reversal, and liquidity-conditioned variants.
3. For each family/parameter set, test both orientations (`pro` and `anti`) because some signals work as continuation and some work as reversal.
4. Keep all candidate metrics in `manual/artifacts/`, then select the final submission set from gate-passing candidates.
5. Recompute each selected factor and export a submission-ready parquet plus metadata under `manual/submit/`.
6. Before each compute/export step, run `audit_manual_spec()` to reject forbidden labels, negative delays, or non-positive windows.

## Search Summary
- Candidate count: `{candidate_count}`
- Distinct factor families: `{family_count}`
- Gate-passing candidates: `{passing_count}`
- Selection target: `{len(selected)}`

### Top 5 Candidates During Search
{chr(10).join(top5_text) if top5_text else '- No passing candidates found.'}

## Final Selected Factors
{build_markdown_table(
    selected_rows,
    [
        ("Rank", "Rank"),
        ("Factor", "Factor"),
        ("Family", "Family"),
        ("Params", "Params"),
        ("Dir", "Dir"),
        ("IC", "IC"),
        ("IR", "IR"),
        ("Turnover", "Turnover"),
        ("Score", "Score"),
        ("Expr", "Expr"),
    ],
)}

## Candidate Exploration Log
{build_markdown_table(
    exploration_rows,
    [
        ("Key", "Key"),
        ("Family", "Family"),
        ("Params", "Params"),
        ("Dir", "Dir"),
        ("Pass", "Pass"),
        ("IC", "IC"),
        ("IR", "IR"),
        ("Turnover", "Turnover"),
        ("Score", "Score"),
    ],
)}

## Notes
- `pro` means we keep the raw signal direction before ranking; `anti` means we flip it first.
- Exported submissions live under `manual/submit/manual_alpha_*_{run_stamp}_y/`.
- Every exported metadata JSON now includes a `future_info_check` section for audit traceability.
- The report is auto-generated from the actual run outputs, so it matches the saved metrics and metadata.
"""
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run manual factor search and export top candidates.")
    parser.add_argument("--top-k", type=int, default=20, help="Number of final gate-passing factors to export.")
    parser.add_argument(
        "--export-csv",
        type=str,
        help="Export factors from an existing result CSV instead of running a fresh search.",
    )
    parser.add_argument(
        "--export-limit",
        type=int,
        help="Optional maximum number of rows to export in --export-csv mode.",
    )
    parser.add_argument(
        "--export-row-offset",
        type=int,
        default=0,
        help="Skip the first N sorted rows in --export-csv mode.",
    )
    parser.add_argument(
        "--export-rank-offset",
        type=int,
        default=0,
        help="Start exported factor numbering after this many rows in --export-csv mode.",
    )
    parser.add_argument(
        "--run-stamp",
        type=str,
        help="Optional run stamp override for export directories, e.g. 20260410_165637.",
    )
    args = parser.parse_args()

    ensure_dirs()
    dataset = ManualFactorDataset()

    if args.export_csv:
        exported, run_stamp = export_from_existing_results(
            csv_path=Path(args.export_csv),
            dataset=dataset,
            passing_only=True,
            export_limit=args.export_limit,
            row_offset=args.export_row_offset,
            rank_offset=args.export_rank_offset,
            run_stamp_override=args.run_stamp,
        )
        print(f"[manual] export-only completed | count={len(exported)} run_stamp={run_stamp}")
        return

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidates = generate_candidates()
    print(
        "[manual] loaded dataset | "
        f"pv={dataset.load_seconds_pv:.1f}s, unstack={dataset.load_seconds_unstack:.1f}s, "
        f"labels={dataset.load_seconds_labels:.1f}s, candidates={len(candidates)}"
    )

    progress_path = ARTIFACTS_DIR / f"manual_factor_search_{run_stamp}.csv"
    results: list[dict[str, Any]] = []

    for idx, spec in enumerate(candidates, start=1):
        started = time.time()
        future_info_check = audit_manual_spec(spec)
        raw = compute_raw(spec, dataset)
        alpha = cs_rank(raw).astype("float32")
        alpha = alpha.replace([np.inf, -np.inf], np.nan)
        metrics = evaluate_alpha(alpha, dataset)

        record = {
            "key": spec.key,
            "family": spec.family,
            "family_label": spec.family_label,
            "params": spec.params,
            "direction": spec.direction,
            "direction_label": spec.direction_label,
            "expression": spec.expression,
            "description": spec.description,
            "future_info_check": future_info_check,
            "future_info_check_passed": future_info_check["passed"],
            "elapsed_seconds": round(time.time() - started, 3),
            **metrics,
        }
        results.append(record)
        pd.DataFrame(results).to_csv(progress_path, index=False)
        print(
            f"[manual] {idx:02d}/{len(candidates)} {spec.key} | "
            f"IC={record['IC']:.4f} IR={record['IR']:.2f} "
            f"Turnover={record['Turnover']:.2f} Score={record['Score']:.3f} "
            f"Pass={record['PassGates']}"
        )
        del raw, alpha
        gc.collect()

    results_df = pd.DataFrame(results)
    selected_df = select_candidates(results_df, args.top_k)
    if selected_df.empty:
        raise RuntimeError("No gate-passing manual factors were found.")

    selected_df = selected_df.reset_index(drop=True)
    selected_df["export_name"] = [f"manual_alpha_{idx:03d}" for idx in range(1, len(selected_df) + 1)]

    selected_path = ARTIFACTS_DIR / f"manual_factor_selected_{run_stamp}.csv"
    selected_df.to_csv(selected_path, index=False)

    exported: list[dict[str, Any]] = []
    for rank, row in enumerate(selected_df.itertuples(index=False), start=1):
        spec = CandidateSpec(
            family=row.family,
            family_label=row.family_label,
            params=row.params if isinstance(row.params, dict) else json.loads(row.params.replace("'", '"')),
            direction=int(row.direction),
            expression=row.expression,
            description=row.description,
        )
        raw = compute_raw(spec, dataset)
        alpha = cs_rank(raw).astype("float32").replace([np.inf, -np.inf], np.nan)
        metadata = export_submission(alpha, spec, row._asdict(), rank, run_stamp, dataset)
        metadata["params"] = spec.params
        metadata["direction"] = spec.direction
        exported.append(metadata)
        print(f"[manual] exported {metadata['factor_name']} -> {metadata['submission_dir']}")
        del raw, alpha
        gc.collect()

    exported_path = ARTIFACTS_DIR / f"manual_factor_exported_{run_stamp}.json"
    exported_path.write_text(json.dumps(exported, indent=2, ensure_ascii=False), encoding="utf-8")

    report_text = build_report(run_stamp, dataset, candidates, results_df, selected_df, exported)
    report_path = REPORTS_DIR / f"manual_factor_report_{run_stamp}.md"
    report_path.write_text(report_text, encoding="utf-8")
    (MANUAL_ROOT / "MANUAL_FACTOR_REPORT.md").write_text(report_text, encoding="utf-8")
    print(f"[manual] wrote report -> {report_path}")


if __name__ == "__main__":
    main()
