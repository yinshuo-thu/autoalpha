"""
autoalpha_v2/pipeline.py

End-to-end pipeline:
  1. LLM generates formula idea (Claude via third-party relay)
  2. formula_validator checks syntax + future-leakage
  3. DataHub provides 15m OHLCV data
  4. FormulaEngine evaluates the DSL expression
  5. evaluator computes IC / IR / turnover metrics
  6. SubmissionBuilder exports alpha to parquet

No mocks — all steps use real data and a real API call.
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# ── project root on sys.path ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from autoalpha_v2 import factor_research
from autoalpha_v2 import knowledge_base as kb
from autoalpha_v2.error_utils import AutoAlphaRuntimeError, humanize_error
from autoalpha_v2.llm_client import generate_idea
from core.evaluator import evaluate_submission_like_wide
from core.feishu_bot import FeishuNotifier
from formula_validator import validate_formula
from prepare_data import DataHub
from factors import operators as _ops
from runtime_config import load_runtime_config

# Feishu Webhook for factor notifications
_feishu = FeishuNotifier(
    webhook_url="https://open.feishu.cn/open-apis/bot/v2/hook/b4cd233b-5185-4135-8a08-4ffda6305877"
)

AUTOALPHA_OUT = PROJECT_ROOT / "autoalpha_v2" / "output"
AUTOALPHA_TRACE_DIR = PROJECT_ROOT / "autoalpha_v2" / "process_logs"

# Extended operator registry: superset of core/formula_engine.py OPS_REGISTRY
# Adds neg, clip, ts_corr, ts_cov, and infix aliases that the LLM may emit.
_EXT_OPS = {
    "lag":           _ops.lag,
    "delay":         _ops.lag,
    "delta":         _ops.delta,
    "ts_mean":       _ops.ts_mean,
    "ts_std":        _ops.ts_std,
    "ts_sum":        _ops.ts_sum,
    "ts_max":        _ops.ts_max,
    "ts_min":        _ops.ts_min,
    "ts_median":     _ops.ts_median,
    "ts_quantile":   _ops.ts_quantile,
    "ts_skew":       _ops.ts_skew,
    "ts_kurt":       _ops.ts_kurt,
    "ts_ema":        _ops.ts_ema,
    "ts_argmax":     _ops.ts_argmax,
    "ts_argmin":     _ops.ts_argmin,
    "ts_pct_change": _ops.ts_pct_change,
    "ts_minmax_norm": _ops.ts_minmax_norm,
    "ts_zscore":     _ops.ts_zscore,
    "ts_rank":       _ops.ts_rank,
    "ts_decay_linear": _ops.ts_decay_linear,
    "decay_linear":  _ops.ts_decay_linear,
    "ts_corr":       _ops.ts_corr,
    "ts_cov":        _ops.ts_cov,
    "cs_rank":       _ops.cs_rank,
    "rank":          _ops.cs_rank,
    "cs_demean":     _ops.cs_demean,
    "demean":        _ops.cs_demean,
    "cs_zscore":     _ops.cs_zscore,
    "zscore":        _ops.cs_zscore,
    "cs_scale":      _ops.cs_scale,
    "scale":         _ops.cs_scale,
    "cs_winsorize":  _ops.cs_winsorize,
    "winsorize":     _ops.cs_winsorize,
    "cs_quantile":   _ops.cs_quantile,
    "cs_neutralize": _ops.cs_neutralize,
    "safe_div":      _ops.safe_div,
    "div":           _ops.safe_div,
    "signed_power":  _ops.signed_power,
    "pow":           _ops.signed_power,
    "abs":           np.abs,
    "sign":          np.sign,
    "log":           _ops.safe_log,
    "signed_log":    _ops.signed_log,
    "sqrt":          _ops.safe_sqrt,
    "sigmoid":       _ops.sigmoid,
    "tanh":          np.tanh,
    "neg":           lambda x: -x,
    "clip":          _ops.clamp,
    "clamp":         _ops.clamp,
    "min_of":        _ops.min_of,
    "max_of":        _ops.max_of,
    "ifelse":        _ops.ifelse,
    "gt":            _ops.gt,
    "ge":            _ops.ge,
    "lt":            _ops.lt,
    "le":            _ops.le,
    "eq":            _ops.eq,
    "and_op":        _ops.and_op,
    "or_op":         _ops.or_op,
    "not_op":        _ops.not_op,
    "mean_of":       _ops.mean_of,
    "weighted_sum":  _ops.weighted_sum,
    "combine_rank":  _ops.combine_rank,
    "np":            np,
}


def _notify_pipeline_error(
    title: str,
    error: Any,
    *,
    stage: str,
    run_id: str = "",
    formula: str = "",
) -> None:
    """Send a concise Feishu alert for actionable pipeline failures."""
    friendly, suggestion, error_code, raw = humanize_error(error)
    try:
        _feishu.send_error_notification(
            title=title,
            summary=friendly,
            stage=stage,
            error_code=error_code,
            suggestion=suggestion,
            raw_detail=raw,
            run_id=run_id,
            formula=formula,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
    except Exception as notify_error:
        print(f"  [feishu] ⚠ Failed to send error notification: {notify_error}")


def _eval_formula(formula_str: str, df: pd.DataFrame) -> pd.Series:
    """Evaluate DSL formula with the extended operator registry."""
    local_env = {col: df[col] for col in df.columns}
    try:
        code = compile(formula_str, "<formula>", "eval")
        result = eval(code, _EXT_OPS, local_env)  # noqa: S307
        return result
    except SyntaxError as e:
        raise ValueError(f"Syntax error in formula: {e}")
    except Exception as e:
        raise RuntimeError(f"Runtime error evaluating formula: {e}")


def _append_trace(trace_path: Path, stage: str, payload: dict[str, Any]) -> None:
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "ts": datetime.now().isoformat(),
        "stage": stage,
        **payload,
    }
    with open(trace_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Post-processing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _postprocess(series: pd.Series, mode: str) -> pd.Series:
    """
    Light post-processing: cross-sectional rank then clip to [-1, 1].
    Keeps alpha bounded (required by SubmissionBuilder) and reduces extreme
    position weights, helping pass the concentration gates.
    """
    if series.empty:
        return series
    try:
        un = series.unstack("security_id")
    except Exception:
        return series.clip(-1.0, 1.0)

    if mode in ("rank_clip", "stable_low_turnover"):
        # Percentile rank → [0,1] → shift to [-0.5, 0.5]
        ranked = un.rank(axis=1, pct=True) - 0.5
    else:  # zscore_clip / aggressive_high_ic
        mu  = un.mean(axis=1)
        std = un.std(axis=1).replace(0, np.nan)
        ranked = un.sub(mu, axis=0).div(std, axis=0).clip(-3, 3) / 3

    stacked = ranked.stack("security_id").reorder_levels(series.index.names).sort_index()
    return stacked.clip(-1.0, 1.0).astype("float32")


# ─────────────────────────────────────────────────────────────────────────────
# Core alpha computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_alpha(
    formula: str,
    pv: pd.DataFrame,
    days: list[str],
    lookback_days: int = 20,
    postprocess_mode: str = "rank_clip",
) -> pd.Series:
    """
    Compute alpha values for the given date range.

    Args:
        formula:          DSL expression string
        pv:               full 15m OHLCV DataFrame (MultiIndex: date/datetime/security_id)
        days:             list of trading-day strings to EVALUATE (without warmup)
        lookback_days:    how many extra days before `days[0]` to load as warmup
        postprocess_mode: 'rank_clip' or 'zscore_clip'

    Returns:
        pd.Series with MultiIndex (date, datetime, security_id), alpha in [-1, 1].
    """
    all_days = sorted(pv.index.get_level_values("date").unique().astype(str).tolist())
    eval_start = days[0]
    eval_end   = days[-1]

    # Find warmup start
    try:
        idx0 = all_days.index(eval_start)
    except ValueError:
        idx0 = 0
    warmup_idx = max(0, idx0 - lookback_days)
    warmup_start = all_days[warmup_idx]

    # Slice: warmup + eval window
    eval_ts      = pd.to_datetime(eval_start)
    warmup_ts    = pd.to_datetime(warmup_start)
    eval_end_ts  = pd.to_datetime(eval_end)

    sub_pv = pv.loc[
        (pv.index.get_level_values("date") >= warmup_ts) &
        (pv.index.get_level_values("date") <= eval_end_ts)
    ]

    # Compute alpha
    raw_alpha = _eval_formula(formula, sub_pv)
    if not isinstance(raw_alpha, pd.Series):
        raw_alpha = pd.Series(raw_alpha)

    # Post-process (cross-section rank / zscore)
    alpha_pp = _postprocess(raw_alpha, postprocess_mode)

    # Trim to eval window only (drop warmup)
    alpha_eval = alpha_pp.loc[
        alpha_pp.index.get_level_values("date") >= eval_ts
    ]
    return alpha_eval


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_alpha(alpha: pd.Series, hub: DataHub, days: list[str]) -> dict[str, Any]:
    """Compute IC / IR / turnover / gates using evaluate_submission_like_wide."""
    day_ts = [pd.to_datetime(d) for d in days]

    resp_df = hub.resp
    rest_df = hub.trading_restriction

    resp_slice = resp_df.loc[resp_df.index.get_level_values("date").isin(day_ts)]
    rest_slice = rest_df.loc[rest_df.index.get_level_values("date").isin(day_ts)]

    resp_col = "resp" if "resp" in resp_df.columns else resp_df.columns[0]
    rest_col = "trading_restriction" if "trading_restriction" in rest_df.columns else rest_df.columns[0]

    alpha_un = alpha.unstack("security_id")
    resp_un  = resp_slice[resp_col].unstack("security_id").reindex_like(alpha_un)
    rest_un  = rest_slice[rest_col].unstack("security_id").reindex_like(alpha_un).fillna(0)

    return evaluate_submission_like_wide(alpha_un, resp_un, rest_un)


def _cfg_int(cfg: dict[str, str], key: str, default: int) -> int:
    try:
        return int(str(cfg.get(key, default) or default))
    except (TypeError, ValueError):
        return default


def _cfg_float(cfg: dict[str, str], key: str, default: float) -> float:
    try:
        return float(str(cfg.get(key, default) or default))
    except (TypeError, ValueError):
        return default


def _evaluate_with_optional_flip(
    alpha: pd.Series,
    formula: str,
    hub: DataHub,
    days: list[str],
) -> tuple[pd.Series, str, dict[str, Any]]:
    metrics = evaluate_alpha(alpha, hub, days)
    ic = float(metrics.get("IC", 0) or 0)
    if ic < -0.3:
        print("  [flip] IC is negative — flipping alpha direction")
        alpha = -alpha
        formula = f"neg({formula})"
        metrics = evaluate_alpha(alpha, hub, days)
    return alpha, formula, metrics


def _should_promote_from_screen(metrics: dict[str, Any], cfg: dict[str, str]) -> bool:
    ic = float(metrics.get("IC", 0) or 0)
    ir = float(metrics.get("IR", 0) or 0)
    score = float(metrics.get("Score", 0) or 0)
    tvr = float(metrics.get("Turnover", 0) or 0)
    min_ic = _cfg_float(cfg, "AUTOALPHA_SCREEN_MIN_IC", 0.12)
    min_ir = _cfg_float(cfg, "AUTOALPHA_SCREEN_MIN_IR", 0.6)
    max_tvr = _cfg_float(cfg, "AUTOALPHA_SCREEN_MAX_TVR", 420.0)
    return bool(metrics.get("PassGates")) or score > 0 or (ic >= min_ic and ir >= min_ir and tvr <= max_tvr)


def _should_materialize_artifacts(metrics: dict[str, Any], cfg: dict[str, str]) -> bool:
    ic = float(metrics.get("IC", 0) or 0)
    ir = float(metrics.get("IR", 0) or 0)
    score = float(metrics.get("Score", 0) or 0)
    min_ic = _cfg_float(cfg, "AUTOALPHA_RESEARCH_MIN_IC", 0.3)
    min_ir = _cfg_float(cfg, "AUTOALPHA_RESEARCH_MIN_IR", 1.2)
    return bool(metrics.get("PassGates")) or score > 0 or (ic >= min_ic and ir >= min_ir)


# ─────────────────────────────────────────────────────────────────────────────
# Parquet export
# ─────────────────────────────────────────────────────────────────────────────

ALLOWED_UTC_TIMES = {
    "01:45:00", "02:00:00", "02:15:00", "02:30:00",
    "02:45:00", "03:00:00", "03:15:00", "03:30:00",
    "05:15:00", "05:30:00", "05:45:00", "06:00:00",
    "06:15:00", "06:30:00", "06:45:00", "07:00:00",
}


def export_parquet(alpha: pd.Series, run_id: str, out_dir: Path) -> Path:
    """
    Save alpha to parquet in submission format with a complete (datetime × security_id)
    cartesian grid per date, as required by the simulator's cartesian_unstack.

    Formulas with lookback warmup produce sparse early dates (fewer bars, uneven
    security sets). expand_to_full_grid fills the missing cells with NaN so every
    date has exactly 16 bars × universe_size rows.
    """
    from core.submission import SubmissionBuilder

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.pq"

    # Build MultiIndex frame expected by SubmissionBuilder
    alpha_frame = SubmissionBuilder._ensure_frame(alpha)

    start_date = alpha_frame.index.get_level_values("date").min()
    end_date   = alpha_frame.index.get_level_values("date").max()
    if hasattr(start_date, "strftime"):
        start_date = start_date.strftime("%Y-%m-%d")
        end_date   = end_date.strftime("%Y-%m-%d")

    expanded = SubmissionBuilder.expand_to_full_grid(
        alpha_frame, str(start_date), str(end_date), chunk_days=30
    )
    alpha_col = SubmissionBuilder._alpha_col(expanded)
    non_null = int(expanded[alpha_col].notna().sum()) if not expanded.empty else 0
    if non_null == 0:
        raise ValueError(
            "Expanded submission grid contains zero non-null alpha values; "
            "check date/datetime/security_id index alignment before export."
        )
    SubmissionBuilder.build(expanded, str(out_path))

    row_count = len(expanded)
    print(f"  [export] Saved {row_count:,} rows (full grid, non-null alpha={non_null:,}) → {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(
    n_ideas: int = 3,
    eval_days_count: int = 0,
    parents: Optional[List] = None,
) -> List[dict]:
    """
    Full pipeline: generate → validate → compute → evaluate → export.

    Args:
        n_ideas:        number of ideas to request from the LLM
        eval_days_count: how many of the most recent trading days to evaluate;
                         0 or below means use the full history
        parents:        optional list of prior factor results to feed the LLM

    Returns:
        list of result dicts (one per valid, successfully computed idea)
    """
    AUTOALPHA_OUT.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trace_path = AUTOALPHA_TRACE_DIR / f"run_{timestamp}.jsonl"
    cfg = load_runtime_config()
    idea_pause_sec = max(0.0, _cfg_float(cfg, "AUTOALPHA_IDEA_PAUSE_MS", 250.0) / 1000.0)

    # ── 1. Load data ONCE ────────────────────────────────────────────────────
    print("[pipeline] Loading DataHub …")
    try:
        hub = DataHub()
        pv = hub.pv_15m
        all_days = hub.get_trading_days_list()
    except Exception as e:
        _notify_pipeline_error("AutoAlpha 数据加载失败", e, stage="DataHub 初始化")
        raise

    if not all_days:
        error = AutoAlphaRuntimeError(
            "交易日列表为空，当前无法执行因子评估。",
            raw_message="DataHub.get_trading_days_list() returned no trading days.",
            suggestion="检查数据目录和 DataHub 读取配置。",
            error_code="empty_trading_days",
        )
        _notify_pipeline_error("AutoAlpha 数据加载失败", error, stage="交易日读取")
        raise error

    eval_days = all_days if eval_days_count <= 0 else all_days[-eval_days_count:]
    screen_days_count = min(
        len(eval_days),
        max(60, _cfg_int(cfg, "AUTOALPHA_SCREEN_DAYS", 160)),
    )
    screen_days = eval_days[-screen_days_count:]
    eval_mode = "ALL" if eval_days_count <= 0 else str(eval_days_count)
    print(f"[pipeline] Eval window: {eval_days[0]} → {eval_days[-1]} ({len(eval_days)} days, requested={eval_mode})")
    print(f"[pipeline] Screen window: {screen_days[0]} → {screen_days[-1]} ({len(screen_days)} days)")
    _append_trace(
        trace_path,
        "run_start",
        {
            "eval_days": len(eval_days),
            "screen_days": len(screen_days),
            "parent_run_ids": [item.get("run_id", "") for item in parents or []],
        },
    )

    # ── 2. Generate ideas (parallel with concurrency limit + cache pop) ────────
    from concurrent.futures import ThreadPoolExecutor, as_completed as _as_completed
    from autoalpha_v2.idea_cache import get_default_cache

    idea_cache = get_default_cache()
    idea_concurrency = _cfg_int(cfg, "AUTOALPHA_IDEA_CONCURRENCY", 3)

    ideas: list[dict] = []
    generation_errors: list[dict[str, str]] = []

    def _fetch_one_idea(i: int) -> tuple[int, Optional[dict], Optional[dict]]:
        # Try cache first
        cached = idea_cache.pop()
        if cached:
            print(f"\n[pipeline] Idea {i+1}/{n_ideas} from cache: {cached.get('formula','')[:60]}")
            return i, cached, None
        print(f"\n[pipeline] Generating idea {i+1}/{n_ideas} via LLM …")
        try:
            idea = generate_idea(parents=parents, idea_index=i, total_ideas=n_ideas)
            return i, idea, None
        except Exception as e:
            friendly, suggestion, _, raw = humanize_error(e)
            return i, None, {"friendly": friendly, "suggestion": suggestion, "raw": raw}

    with ThreadPoolExecutor(max_workers=idea_concurrency) as pool:
        futures = {pool.submit(_fetch_one_idea, i): i for i in range(n_ideas)}
        for fut in _as_completed(futures):
            i, idea, err = fut.result()
            if idea:
                print(f"  formula    : {idea.get('formula')}")
                print(f"  thought    : {str(idea.get('thought_process',''))[:120]}")
                ideas.append(idea)
                _append_trace(
                    trace_path,
                    "idea_generated",
                    {
                        "idea_index": i + 1,
                        "formula": idea.get("formula", ""),
                        "thought_process": idea.get("thought_process", ""),
                    },
                )
            elif err:
                generation_errors.append(err)
                extra = f" 建议: {err.get('suggestion')}" if err.get("suggestion") else ""
                print(f"  [WARN] LLM call failed: {err.get('friendly')}{extra}")
                _append_trace(
                    trace_path,
                    "idea_generation_error",
                    {
                        "idea_index": i + 1,
                        "friendly": err.get("friendly", ""),
                        "suggestion": err.get("suggestion", ""),
                        "raw": err.get("raw", ""),
                    },
                )

    if idea_pause_sec > 0 and ideas:
        time.sleep(idea_pause_sec)

    if not ideas:
        primary = generation_errors[0] if generation_errors else {
            "friendly": "LLM 没有返回任何可用因子。",
            "suggestion": "检查 API Key、模型和网关状态。",
            "raw": "No ideas generated.",
        }
        error = AutoAlphaRuntimeError(
            primary["friendly"],
            raw_message=primary["raw"],
            suggestion=primary["suggestion"],
            error_code="idea_generation_failed",
        )
        _notify_pipeline_error("AutoAlpha 因子生成失败", error, stage="LLM 生成")
        raise error

    # ── 3. Validate + compute ─────────────────────────────────────────────────
    results = []
    existing_formula_keys = kb.get_existing_formula_keys()
    batch_formula_keys: set[str] = set()
    for idx, idea in enumerate(ideas):
        formula  = idea.get("formula", "").strip()
        postmode = idea.get("postprocess", "rank_clip")
        lookback = int(idea.get("lookback_days", 20))
        run_id   = f"autoalpha_{timestamp}_{idx+1:02d}"

        print(f"\n[pipeline] Processing idea {idx+1}: {formula}")

        formula_key = kb.canonical_formula(formula)
        if formula_key in existing_formula_keys or formula_key in batch_formula_keys:
            print("  [SKIP] Duplicate formula against KB or current batch")
            _append_trace(trace_path, "duplicate_formula", {"run_id": run_id, "formula": formula})
            results.append({
                "run_id": run_id,
                "formula": formula,
                "thought_process": idea.get("thought_process", ""),
                "status": "duplicate",
                "errors": "Duplicate formula signature already exists in knowledge base or current batch.",
            })
            continue
        batch_formula_keys.add(formula_key)

        # 3a. Validate (syntax + whitelist + future-leakage)
        vr = validate_formula(formula)
        if not vr.valid:
            print(f"  [SKIP] Validation failed: {vr.errors}")
            _append_trace(
                trace_path,
                "validation_failed",
                {"run_id": run_id, "formula": formula, "errors": list(vr.errors or [])},
            )
            results.append({
                "run_id": run_id,
                "formula": formula,
                "thought_process": idea.get("thought_process", ""),
                "status": "invalid",
                "errors": vr.errors,
            })
            continue
        if vr.warnings:
            print(f"  [WARN] {vr.warnings}")
            _append_trace(
                trace_path,
                "validation_warning",
                {"run_id": run_id, "formula": formula, "warnings": list(vr.warnings or [])},
            )

        # 3b. Compute alpha on a shorter screening window first
        try:
            alpha_screen = compute_alpha(
                formula      = formula,
                pv           = pv,
                days         = screen_days,
                lookback_days= lookback,
                postprocess_mode = postmode,
            )
            print(f"  [screen-compute] alpha shape={alpha_screen.shape}, "
                  f"non-null={alpha_screen.notna().sum()}, "
                  f"range=[{alpha_screen.min():.3f}, {alpha_screen.max():.3f}]")
        except Exception as e:
            print(f"  [SKIP] Computation error: {e}")
            _append_trace(
                trace_path,
                "screen_compute_error",
                {"run_id": run_id, "formula": formula, "error": str(e)},
            )
            results.append({
                "run_id": run_id,
                "formula": formula,
                "thought_process": idea.get("thought_process", ""),
                "status": "compute_error",
                "error": str(e),
            })
            continue

        # 3c. Fast screen metrics on recent subset
        try:
            alpha_screen, formula, screen_metrics = _evaluate_with_optional_flip(
                alpha_screen, formula, hub, screen_days
            )
            sic = float(screen_metrics.get("IC", 0) or 0)
            sir = float(screen_metrics.get("IR", 0) or 0)
            stvr = float(screen_metrics.get("Turnover", 0) or 0)
            spass = bool(screen_metrics.get("PassGates", False))
            sscore = float(screen_metrics.get("Score", 0) or 0)
            print(f"  [screen-metrics] IC={sic:.4f}  IR={sir:.4f}  tvr={stvr:.2f}  "
                  f"PassGates={spass}  score={sscore:.4f}")
            _append_trace(
                trace_path,
                "screen_metrics",
                {
                    "run_id": run_id,
                    "formula": formula,
                    "IC": sic,
                    "IR": sir,
                    "Turnover": stvr,
                    "PassGates": spass,
                    "Score": sscore,
                },
            )
        except Exception as e:
            print(f"  [WARN] Screen evaluation error: {e}")
            screen_metrics = {}
            _append_trace(
                trace_path,
                "screen_eval_error",
                {"run_id": run_id, "formula": formula, "error": str(e)},
            )

        if not _should_promote_from_screen(screen_metrics, cfg):
            print("  [screen] Low-signal candidate — skipping full-history evaluation and heavy artifacts")
            _append_trace(
                trace_path,
                "screen_rejected",
                {"run_id": run_id, "formula": formula, "screen_metrics": screen_metrics},
            )
            results.append({
                "run_id": run_id,
                "formula": formula,
                "thought_process": idea.get("thought_process", ""),
                "postprocess": postmode,
                "lookback_days": lookback,
                "status": "screened_out",
                "IC": screen_metrics.get("IC", 0),
                "IR": screen_metrics.get("IR", 0),
                "tvr": screen_metrics.get("Turnover", 0),
                "PassGates": False,
                "Score": screen_metrics.get("Score", 0),
                "gates_detail": screen_metrics.get("GatesDetail", {}),
                "parquet_path": None,
                "eval_days": len(screen_days),
                "research_path": None,
                "screening": {
                    "stage": "recent_subset",
                    "days": len(screen_days),
                    "promoted": False,
                },
                "artifact_policy": "skipped_low_signal",
            })
            gc.collect()
            continue

        # 3d. Full-history compute + evaluation for promoted ideas
        try:
            alpha = compute_alpha(
                formula=formula,
                pv=pv,
                days=eval_days,
                lookback_days=lookback,
                postprocess_mode=postmode,
            )
            print(f"  [full-compute] alpha shape={alpha.shape}, "
                  f"non-null={alpha.notna().sum()}, "
                  f"range=[{alpha.min():.3f}, {alpha.max():.3f}]")
            alpha, formula, metrics = _evaluate_with_optional_flip(alpha, formula, hub, eval_days)
            ic = float(metrics.get("IC", 0) or 0)
            ir = float(metrics.get("IR", 0) or 0)
            tvr = float(metrics.get("Turnover", 0) or 0)
            passes = bool(metrics.get("PassGates", False))
            score = float(metrics.get("Score", 0) or 0)
            print(f"  [full-metrics] IC={ic:.4f}  IR={ir:.4f}  tvr={tvr:.2f}  "
                  f"PassGates={passes}  score={score:.4f}")
            _append_trace(
                trace_path,
                "full_metrics",
                {
                    "run_id": run_id,
                    "formula": formula,
                    "IC": ic,
                    "IR": ir,
                    "Turnover": tvr,
                    "PassGates": passes,
                    "Score": score,
                },
            )
        except Exception as e:
            print(f"  [WARN] Full evaluation error: {e}")
            _append_trace(
                trace_path,
                "full_eval_error",
                {"run_id": run_id, "formula": formula, "error": str(e)},
            )
            results.append({
                "run_id": run_id,
                "formula": formula,
                "thought_process": idea.get("thought_process", ""),
                "status": "compute_error",
                "error": str(e),
            })
            gc.collect()
            continue

        # 3e. Export parquet + research only for promising full-eval outputs
        out_path = None
        research_path = None
        artifact_policy = "skipped_low_signal"
        if _should_materialize_artifacts(metrics, cfg):
            artifact_policy = "materialized"
            try:
                out_path = export_parquet(alpha, run_id, AUTOALPHA_OUT)
            except Exception as e:
                print(f"  [WARN] Export error: {e}")
                out_path = None

            try:
                research_path = factor_research.analyze_factor(
                    run_id=run_id,
                    formula=formula,
                    alpha=alpha,
                    metrics=metrics,
                    hub=hub,
                    eval_days=eval_days,
                    thought_process=idea.get("thought_process", ""),
                )
            except Exception as e:
                print(f"  [WARN] Research analysis error: {e}")
        else:
            print("  [artifacts] Skipping parquet/research for low-signal full-history result")
            _append_trace(
                trace_path,
                "artifact_skipped",
                {"run_id": run_id, "formula": formula, "policy": "skipped_low_signal"},
            )

        result = {
            "run_id":         run_id,
            "formula":        formula,
            "thought_process": idea.get("thought_process", ""),
            "postprocess":    postmode,
            "lookback_days":  lookback,
            "status":         "ok",
            "IC":             metrics.get("IC", 0),
            "IR":             metrics.get("IR", 0),
            "tvr":            metrics.get("Turnover", 0),
            "PassGates":      metrics.get("PassGates", False),
            "Score":          metrics.get("Score", 0),
            "gates_detail":   metrics.get("GatesDetail", {}),
            "parquet_path":   str(out_path) if out_path else None,
            "eval_days":      len(eval_days),
            "research_path":  research_path,
            "screening": {
                "stage": "recent_subset",
                "days": len(screen_days),
                "promoted": True,
                "IC": screen_metrics.get("IC", 0),
                "IR": screen_metrics.get("IR", 0),
                "Turnover": screen_metrics.get("Turnover", 0),
                "Score": screen_metrics.get("Score", 0),
            },
            "artifact_policy": artifact_policy,
        }
        results.append(result)
        _append_trace(
            trace_path,
            "result_saved",
            {
                "run_id": run_id,
                "formula": formula,
                "status": result.get("status"),
                "PassGates": result.get("PassGates"),
                "Score": result.get("Score"),
                "artifact_policy": artifact_policy,
            },
        )

        # 3f. Feishu notification for passing factors
        if metrics.get("PassGates"):
            try:
                # Compute ranking among all known factors
                from autoalpha_v2 import knowledge_base as _kb
                all_known = _kb.get_all_factors()
                all_scores = sorted(
                    [f.get("Score", 0) for f in all_known if f.get("Score", 0) > 0],
                    reverse=True,
                )
                rank_pos = 1
                for i, s in enumerate(all_scores):
                    if score <= s:
                        rank_pos = i + 1
                rank_text = f"{rank_pos}/{len(all_scores)+1}" if all_scores else "1/1"

                description = (
                    f"{idea.get('thought_process', '')}\n\n"
                    f"📊 排名: {rank_text} (全部有效因子中)\n"
                    f"📅 评估窗口: {len(eval_days)} 交易日\n"
                    f"🔧 后处理: {postmode} | 回看: {lookback}天"
                )
                _feishu.send_factor_notification(
                    factor_name=run_id,
                    description=description,
                    metrics={
                        "Score": score,
                        "PassGates": True,
                        "IC": ic,
                        "IR": ir,
                        "Turnover": tvr,
                    },
                    formula=formula,
                    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                )
                print(f"  [feishu] ✅ Notification sent for {run_id}")
            except Exception as e:
                print(f"  [feishu] ⚠ Failed to send notification: {e}")

        gc.collect()

    # ── 4. Save run manifest ─────────────────────────────────────────────────
    manifest_path = AUTOALPHA_OUT / f"run_{timestamp}.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[pipeline] Manifest saved → {manifest_path}")
    _append_trace(trace_path, "run_complete", {"manifest_path": str(manifest_path), "result_count": len(results)})

    return results
