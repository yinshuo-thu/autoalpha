"""
server.py — AutoAlpha Backend API (Flask)

Provides a frontend-compatible API for:
- factor library browsing
- manual factor generation / evaluation
- autonomous research loop control
- lightweight backtest execution
- runtime system configuration
"""
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
import ast
from datetime import datetime
from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from autoalpha_v2.error_utils import humanize_error
from autoalpha_v2.inspiration_db import (
    DB_PATH as AUTOALPHA_DB_PATH,
    PROMPT_DIR as AUTOALPHA_PROMPT_DIR,
    compose_inspiration_context,
    delete_inspiration,
    list_inspirations_paginated,
    list_inspiration_source_counts,
    list_recent_inspirations,
    prepare_inspiration,
    save_inspiration,
    sync_prompt_directory,
    toggle_inspiration_status,
    update_inspiration_summary,
)
from autoalpha_v2.llm_client import summarize_inspiration_text
from asset_registry import get_all_assets
from core.factor_experience import append_factor_experience, build_factor_experience_record
from data_catalog import get_full_catalog
from factor_idea_generator import generate_from_prompt, generate_ideas_with_prompt
from fit_models import fit_and_evaluate_models
from core.llm_mining_log import read_recent_llm_mining_records
from leaderboard import add_or_update_factor, get_all_factors, load_leaderboard
from operator_catalog import get_operators_by_category
from paths import LLM_MINING_JSONL, OUTPUTS_ROOT, RESEARCH_ARTIFACTS_ROOT, RESEARCH_LOG_PATH
from prepare_data import DataHub
from quick_test import compute_formula, quick_test
from runtime_config import get_llm_routing, load_runtime_config, masked_runtime_config, save_runtime_config
from simulate_strategy import run_strategy_simulation

app = Flask(__name__, static_folder=os.path.join("frontend", "dist"))
CORS(app)

_data_hub = None
_research_process = None
_autoalpha_loop_process = None
_backtest_tasks: Dict[str, Dict[str, Any]] = {}
_backtest_threads: Dict[str, threading.Thread] = {}
_backtest_lock = threading.Lock()

AUTOALPHA_LOG_PATH = os.path.join(os.path.dirname(__file__), "autoalpha_v2", "loop.log")
AUTOALPHA_LOOP_PID_PATH = os.path.join(os.path.dirname(__file__), "autoalpha_v2", "loop.pid")
AUTOALPHA_LOOP_META_PATH = os.path.join(os.path.dirname(__file__), "autoalpha_v2", "loop_state.json")
AUTOALPHA_KB_PATH  = os.path.join(os.path.dirname(__file__), "autoalpha_v2", "knowledge.json")
AUTOALPHA_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "autoalpha_v2", "output")
AUTOALPHA_RESEARCH_DIR = os.path.join(os.path.dirname(__file__), "autoalpha_v2", "research")
AUTOALPHA_MODEL_LAB_DIR = os.path.join(os.path.dirname(__file__), "autoalpha_v2", "model_lab")
AUTOALPHA_QUOTA_DISPLAY_FX = float(os.environ.get("AUTOALPHA_QUOTA_DISPLAY_FX", "7.3"))
AUTOALPHA_BILLING_ENDPOINTS = {
    "subscription": [
        "https://vip.aipro.love/v1/dashboard/billing/subscription",
        "http://free.aipro.love/v1/dashboard/billing/subscription",
    ],
    "usage": [
        "https://vip.aipro.love/v1/dashboard/billing/usage?start_date=2020-01-01&end_date=2030-12-31",
        "http://free.aipro.love/v1/dashboard/billing/usage?start_date=2020-01-01&end_date=2030-12-31",
    ],
}


def ok(data: Optional[Dict[str, Any]] = None, message: Optional[str] = None):
    payload: Dict[str, Any] = {"success": True}
    if data is not None:
        payload["data"] = data
    if message:
        payload["message"] = message
    return jsonify(payload)


def fail(message: str, status_code: int = 400):
    return jsonify({"success": False, "error": message}), status_code


def now_iso() -> str:
    return datetime.now().isoformat()


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _read_autoalpha_loop_meta() -> Dict[str, Any]:
    try:
        with open(AUTOALPHA_LOOP_META_PATH, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _tracked_autoalpha_loop_pid() -> Optional[int]:
    global _autoalpha_loop_process
    if _autoalpha_loop_process is not None and _autoalpha_loop_process.poll() is None:
        return int(_autoalpha_loop_process.pid)
    try:
        with open(AUTOALPHA_LOOP_PID_PATH, "r", encoding="utf-8") as handle:
            pid = int((handle.read() or "0").strip())
        if _pid_is_running(pid):
            return pid
    except Exception:
        pass
    return None


def _write_autoalpha_loop_state(pid: int, params: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(AUTOALPHA_LOOP_PID_PATH), exist_ok=True)
    with open(AUTOALPHA_LOOP_PID_PATH, "w", encoding="utf-8") as handle:
        handle.write(str(pid))
    with open(AUTOALPHA_LOOP_META_PATH, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "pid": pid,
                "started_at": now_iso(),
                "params": params,
                "log_path": AUTOALPHA_LOG_PATH,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )


def get_data_hub():
    global _data_hub
    if _data_hub is None:
        _data_hub = DataHub()
    return _data_hub


def get_frontend_dist_ready() -> bool:
    return os.path.exists(os.path.join(app.static_folder, "index.html"))


def normalize_factor_quality(factor: Dict[str, Any]) -> str:
    if factor.get("submission_ready_flag") or factor.get("classification") == "Submission Ready":
        return "high"
    if factor.get("PassGates") or factor.get("classification") == "Research Candidate":
        return "medium"
    return "low"


def factor_sort_key(factor: Dict[str, Any]):
    return (
        float(factor.get("submission_ready_flag", factor.get("classification") == "Submission Ready")),
        float(factor.get("PassGates", False)),
        float(factor.get("Score", 0) or 0),
        float(factor.get("rank_ic", 0) or 0),
        float(factor.get("IC", 0) or 0),
        factor.get("timestamp", ""),
    )


def _top_leaderboard_factors(limit: int = 8) -> List[Dict[str, Any]]:
    """Top factors by quality/score for LLM context (experience sharing)."""
    lb = load_leaderboard()
    rows = list(lb.get("factors", []))
    rows.sort(key=factor_sort_key, reverse=True)
    out: List[Dict[str, Any]] = []
    for f in rows[:limit]:
        out.append(
            {
                "factor_name": f.get("factor_name"),
                "formula": f.get("formula"),
                "Score": f.get("Score", 0),
                "IC": f.get("IC", 0),
                "PassGates": f.get("PassGates", False),
                "classification": f.get("classification", ""),
            }
        )
    return out


def serialize_factor(factor: Dict[str, Any]) -> Dict[str, Any]:
    factor_name = factor.get("factor_name", "unknown")
    submission_ready = bool(
        factor.get("submission_ready_flag") or factor.get("classification") == "Submission Ready"
    )
    sanity_report = factor.get("sanity_report") or {}
    gates_detail = factor.get("gates_detail") or {}
    return {
        "factorId": factor_name,
        "factorName": factor_name,
        "factorExpression": factor.get("formula", ""),
        "factorDescription": factor.get("recommendation")
        or factor.get("rationale")
        or factor.get("classification", ""),
        "factorFormulation": factor.get("formula", ""),
        "quality": normalize_factor_quality(factor),
        "ic": float(factor.get("IC", 0) or 0),
        "icir": float(factor.get("IR", 0) or 0),
        "rankIc": float(factor.get("rank_ic", 0) or 0),
        "rankIcir": float(factor.get("IR", 0) or 0),
        "annualReturn": float(factor.get("IC", 0) or 0),
        "maxDrawdown": float(-(factor.get("Turnover", 0) or 0) / 1000.0),
        "sharpeRatio": float(factor.get("IR", 0) or 0),
        "round": int(factor.get("round", 0) or 0),
        "direction": factor.get("family", ""),
        "createdAt": factor.get("timestamp", now_iso()),
        "score": float(factor.get("Score", 0) or 0),
        "turnover": float(factor.get("Turnover", 0) or 0),
        "turnoverLocal": float(factor.get("TurnoverLocal", factor.get("Turnover", 0)) or 0),
        "passGates": bool(factor.get("PassGates", False)),
        "submissionReadyFlag": submission_ready,
        "classification": factor.get("classification", "Drop"),
        "recommendation": factor.get("recommendation", ""),
        "reason": factor.get("reason", ""),
        "submissionPath": factor.get("submission_path", ""),
        "submissionDir": factor.get("submission_dir", ""),
        "metadataPath": factor.get("metadata_path", ""),
        "sanityReport": sanity_report,
        "gatesDetail": gates_detail,
        "scoreFormula": factor.get(
            "score_formula",
            "score = (IC - 0.0005 * Turnover) * sqrt(IR) * 100",
        ),
        "scoreComponents": factor.get("score_components", {}),
        "metricMode": factor.get("metric_mode", "cloud_aligned_preferred"),
        "turnoverBasis": factor.get("turnover_basis", "cloud_aligned_preferred"),
        "backtestResults": {
            "Pass Gates": bool(factor.get("PassGates", False)),
            "Submission Ready": submission_ready,
            "IC": float(factor.get("IC", 0) or 0),
            "ICIR": float(factor.get("IR", 0) or 0),
            "Rank IC": float(factor.get("rank_ic", 0) or 0),
            "Rank ICIR": float(factor.get("IR", 0) or 0),
            "Turnover": float(factor.get("Turnover", 0) or 0),
            "Turnover Local": float(factor.get("TurnoverLocal", factor.get("Turnover", 0)) or 0),
            "Score": float(factor.get("Score", 0) or 0),
            "Coverage": bool(sanity_report.get("cover_all", 0)),
            "Exact Grid": bool(sanity_report.get("exact_15m_grid", False)),
        },
    }


def load_factor_detail(factor_name: str) -> Optional[Dict[str, Any]]:
    artifact_path = os.path.join(RESEARCH_ARTIFACTS_ROOT, f"{factor_name}.json")
    if os.path.exists(artifact_path):
        with open(artifact_path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    for factor in get_all_factors():
        if factor.get("factor_name") == factor_name:
            return factor
    return None


def get_available_libraries() -> List[str]:
    return ["leaderboard"]


def detect_formula_input(user_input: str) -> bool:
    return "(" in user_input and ")" in user_input


def build_factor_name(prefix: str = "manual_factor") -> str:
    return f"{prefix}_{int(time.time())}"


def load_runtime_env_payload() -> Dict[str, Any]:
    config = masked_runtime_config()
    raw = load_runtime_config()
    return {
        "env": {
            "OPENAI_API_KEY": config.get("OPENAI_API_KEY", ""),
            "OPENAI_BASE_URL": raw.get("OPENAI_BASE_URL", ""),
            "CHAT_MODEL": raw.get("CHAT_MODEL", ""),
            "REASONING_MODEL": raw.get("REASONING_MODEL", ""),
            "CHEAP_MODEL": raw.get("CHEAP_MODEL", ""),
            "AUTOALPHA_DEFAULT_ROUNDS": raw.get("AUTOALPHA_DEFAULT_ROUNDS", ""),
            "AUTOALPHA_DEFAULT_IDEAS": raw.get("AUTOALPHA_DEFAULT_IDEAS", ""),
            "AUTOALPHA_DEFAULT_DAYS": raw.get("AUTOALPHA_DEFAULT_DAYS", ""),
            "AUTOALPHA_DEFAULT_TARGET_VALID": raw.get("AUTOALPHA_DEFAULT_TARGET_VALID", ""),
            "AUTOALPHA_PROMPT_CONTEXT_LIMIT": raw.get("AUTOALPHA_PROMPT_CONTEXT_LIMIT", ""),
            "AUTOALPHA_QUOTA_DISPLAY_FX": raw.get("AUTOALPHA_QUOTA_DISPLAY_FX", ""),
            "AUTOALPHA_ROLLING_TARGET_VALID": raw.get("AUTOALPHA_ROLLING_TARGET_VALID", ""),
            "AUTOALPHA_ROLLING_TRAIN_DAYS": raw.get("AUTOALPHA_ROLLING_TRAIN_DAYS", ""),
            "AUTOALPHA_ROLLING_TEST_DAYS": raw.get("AUTOALPHA_ROLLING_TEST_DAYS", ""),
            "AUTOALPHA_ROLLING_STEP_DAYS": raw.get("AUTOALPHA_ROLLING_STEP_DAYS", ""),
        },
        "factorLibraries": get_available_libraries(),
        "paths": {
            "autoalphaRoot": os.path.join(os.path.dirname(__file__), "autoalpha_v2"),
            "promptDir": str(AUTOALPHA_PROMPT_DIR),
            "databasePath": str(AUTOALPHA_DB_PATH),
            "outputDir": AUTOALPHA_OUTPUT_DIR,
            "researchDir": AUTOALPHA_RESEARCH_DIR,
            "modelLabDir": AUTOALPHA_MODEL_LAB_DIR,
        },
    }


def pick_best_factor() -> Optional[Dict[str, Any]]:
    factors = get_all_factors()
    if not factors:
        return None
    return max(factors, key=factor_sort_key)


def append_research_log(message: str) -> None:
    os.makedirs(os.path.dirname(RESEARCH_LOG_PATH), exist_ok=True)
    with open(RESEARCH_LOG_PATH, "a", encoding="utf-8") as handle:
        handle.write(message.rstrip() + "\n")


def create_backtest_task(config: Dict[str, Any]) -> Dict[str, Any]:
    task_id = str(uuid.uuid4())[:8]
    return {
        "taskId": task_id,
        "status": "running",
        "type": "backtest",
        "config": config,
        "progress": {
            "phase": "backtesting",
            "currentRound": 0,
            "totalRounds": 1,
            "progress": 0,
            "message": "正在初始化回测任务...",
            "timestamp": now_iso(),
        },
        "logs": [],
        "metrics": {},
        "createdAt": now_iso(),
        "updatedAt": now_iso(),
        "_cancelled": False,
    }


def add_task_log(task: Dict[str, Any], message: str, level: str = "info") -> None:
    task["logs"].append(
        {
            "id": str(uuid.uuid4())[:8],
            "timestamp": now_iso(),
            "level": level,
            "message": message,
        }
    )
    task["logs"] = task["logs"][-500:]
    task["updatedAt"] = now_iso()


def run_backtest_task(task_id: str) -> None:
    with _backtest_lock:
        task = _backtest_tasks.get(task_id)
    if not task:
        return

    started_at = time.time()
    try:
        task["progress"].update(
            {"progress": 10, "message": "正在选择最优因子...", "timestamp": now_iso()}
        )
        add_task_log(task, "Scanning current factor library...")
        best_factor = pick_best_factor()
        if not best_factor:
            raise RuntimeError("当前因子库为空，请先生成或挖掘因子")
        if task.get("_cancelled"):
            return

        factor_name = best_factor["factor_name"]
        factor_detail = load_factor_detail(factor_name) or best_factor
        formula = factor_detail.get("formula")
        if not formula:
            raise RuntimeError(f"因子 {factor_name} 缺少公式，无法回测")

        task["progress"].update(
            {"progress": 35, "message": f"正在计算 {factor_name} 因子值...", "timestamp": now_iso()}
        )
        add_task_log(task, f"Selected factor: {factor_name}")
        add_task_log(task, f"Formula: {formula}")

        hub = get_data_hub()
        series = compute_formula(formula, hub)
        if task.get("_cancelled"):
            return

        task["progress"].update(
            {"progress": 70, "message": "正在执行策略回测...", "timestamp": now_iso()}
        )
        strategy = run_strategy_simulation(series, hub, factor_name)
        if strategy.get("status") != "success":
            raise RuntimeError(strategy.get("reason", "回测计算失败"))

        pnl_curve = strategy.get("time_series", {}).get("cum_pnl", [])
        cumulative_curve = [{"date": item["date"], "value": item["pnl"]} for item in pnl_curve]
        sharpe = float(strategy["metrics"].get("sharpe", 0) or 0)
        max_dd = float(strategy["metrics"].get("max_drawdown", 0) or 0)
        total_pnl = float(strategy["metrics"].get("total_pnl", 0) or 0)
        calmar = total_pnl / abs(max_dd) if max_dd else 0.0

        task["metrics"] = {
            "IC": float(best_factor.get("IC", 0) or 0),
            "ICIR": float(best_factor.get("IR", 0) or 0),
            "Rank IC": float(best_factor.get("rank_ic", 0) or 0),
            "Rank ICIR": float(best_factor.get("IR", 0) or 0),
            "annualized_return": total_pnl,
            "max_drawdown": max_dd,
            "information_ratio": sharpe,
            "calmar_ratio": calmar,
            "cumulative_curve": cumulative_curve,
            "__num_factors": len(get_all_factors()),
            "__elapsed_seconds": time.time() - started_at,
            "__best_factor": factor_name,
        }
        task["status"] = "completed"
        task["progress"].update(
            {"phase": "completed", "progress": 100, "message": "回测完成", "timestamp": now_iso()}
        )
        add_task_log(task, f"Backtest completed for {factor_name}", "success")
    except Exception as exc:
        task["status"] = "failed"
        task["progress"].update(
            {
                "phase": "backtesting",
                "progress": 100,
                "message": f"回测失败: {exc}",
                "timestamp": now_iso(),
            }
        )
        add_task_log(task, str(exc), "error")
    finally:
        task["updatedAt"] = now_iso()


# ── Serve Frontend ──
@app.route("/")
def index():
    if get_frontend_dist_ready():
        return send_from_directory(app.static_folder, "index.html")
    return jsonify(
        {
            "message": "Frontend build not found.",
            "hint": "Run `cd frontend && npm run build` or use the Vite dev server on http://localhost:3000.",
        }
    )


@app.route("/<path:path>")
def static_files(path):
    if get_frontend_dist_ready() and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    if get_frontend_dist_ready():
        return send_from_directory(app.static_folder, "index.html")
    return fail("Frontend build not found", 404)


# ── Summary / Health ──
@app.route("/api/summary", methods=["GET"])
def get_summary():
    factors = get_all_factors()
    total = len(factors)
    passed = sum(1 for f in factors if f.get("PassGates"))
    sub_ready = sum(
        1 for f in factors if f.get("submission_ready_flag") or f.get("classification") == "Submission Ready"
    )
    valid_ic = [f["IC"] for f in factors if f.get("IC") is not None]
    valid_score = [f["Score"] for f in factors if f.get("Score") is not None]
    valid_tvr = [
        f["Turnover"]
        for f in factors
        if f.get("Turnover") is not None and f.get("Turnover") > 0
    ]
    return ok(
        {
            "total_factors": total,
            "passed_gates": passed,
            "submission_ready": sub_ready,
            "best_ic": max(valid_ic) if valid_ic else 0.0,
            "best_score": max(valid_score) if valid_score else 0.0,
            "avg_tvr": sum(valid_tvr) / len(valid_tvr) if valid_tvr else 0.0,
        }
    )


@app.route("/api/health", methods=["GET"])
def get_health():
    return ok({"status": "healthy", "timestamp": now_iso()})


# ── Factor Library ──
@app.route("/api/factors", methods=["GET"])
def list_factors():
    search = request.args.get("search", request.args.get("q", "")).lower()
    quality = request.args.get("quality", "")
    classification = request.args.get("classification", "")
    limit = int(request.args.get("limit", "500"))
    offset = int(request.args.get("offset", "0"))

    raw_factor_rows = sorted(get_all_factors(), key=factor_sort_key, reverse=True)
    raw_factors = [serialize_factor(item) for item in raw_factor_rows]
    if search:
        raw_factors = [
            item
            for item in raw_factors
            if search in item["factorName"].lower()
            or search in item["factorExpression"].lower()
            or search in item["factorDescription"].lower()
        ]
    if quality:
        raw_factors = [item for item in raw_factors if item["quality"] == quality]
    if classification and classification != "All":
        factor_names = {
            item["factor_name"]
            for item in get_all_factors()
            if item.get("classification") == classification
        }
        raw_factors = [item for item in raw_factors if item["factorName"] in factor_names]

    metadata = load_leaderboard()
    total = len(raw_factors)
    paginated = raw_factors[offset : offset + limit]
    return ok(
        {
            "factors": paginated,
            "total": total,
            "limit": limit,
            "offset": offset,
            "metadata": {
                "last_updated": metadata.get("last_updated"),
                "total_factors": total,
            },
            "libraries": get_available_libraries(),
        }
    )


@app.route("/api/factors/libraries", methods=["GET"])
def list_factor_libraries():
    return ok({"libraries": get_available_libraries()})


@app.route("/api/factors/cache-status", methods=["GET"])
def get_cache_status():
    factors = [serialize_factor(item) for item in get_all_factors()]
    payload = {
        "total": len(factors),
        "h5_cached": len(factors),
        "md5_cached": 0,
        "need_compute": 0,
        "factors": [
            {
                "factor_id": factor["factorId"],
                "factor_name": factor["factorName"],
                "status": "h5_cached",
            }
            for factor in factors
        ],
    }
    return ok(payload)


@app.route("/api/factors/warm-cache", methods=["POST"])
def warm_cache():
    factors = get_all_factors()
    return ok(
        {
            "total": len(factors),
            "synced": len(factors),
            "skipped": 0,
            "failed": 0,
        },
        message=f"缓存状态已同步，共 {len(factors)} 个因子可直接使用",
    )


@app.route("/api/factors/<name>", methods=["GET"])
def get_factor_detail(name):
    detail = load_factor_detail(name)
    if not detail:
        return fail("Factor not found", 404)
    return ok({"factor": {**serialize_factor(detail), **detail}})


# ── Catalog / Idea / Formula ──
@app.route("/api/catalog/data", methods=["GET"])
def catalog_data():
    return ok(get_full_catalog())


@app.route("/api/catalog/operators", methods=["GET"])
def catalog_operators():
    return ok(get_operators_by_category())


@app.route("/api/catalog/assets", methods=["GET"])
def catalog_assets():
    return ok({"assets": get_all_assets()})


@app.route("/api/idea/generate", methods=["POST"])
def generate_ideas():
    data = request.get_json(silent=True) or {}
    prompt = (data.get("prompt") or "").strip()
    if not prompt:
        return fail("Prompt is required")
    ideas = generate_ideas_with_prompt(prompt, parents=_top_leaderboard_factors(), num_ideas=3)
    if not ideas:
        ideas = generate_from_prompt(prompt)
    return ok({"ideas": ideas})


@app.route("/api/formula/test", methods=["POST"])
def run_quick_formula_test():
    data = request.get_json(silent=True) or {}
    formula = (data.get("formula") or "").strip()
    if not formula:
        return fail("Formula is required")
    factor_name = data.get("name") or build_factor_name("formula")
    postprocess = data.get("postprocess")
    get_data_hub()
    try:
        result = quick_test(formula, factor_name, postprocess=postprocess)
        if result.get("status") == "success":
            add_or_update_factor(result)
        return ok({"factor": result})
    except Exception as exc:
        return fail(str(exc), 500)


@app.route("/api/formula/execute", methods=["POST"])
def execute_formula_or_prompt():
    data = request.get_json(silent=True) or {}
    user_input = (data.get("input") or data.get("prompt") or data.get("formula") or "").strip()
    if not user_input:
        return fail("Input is required")

    manual_name = data.get("name") or build_factor_name("specified_factor")
    postprocess = data.get("postprocess")
    ideas: List[Dict[str, Any]] = []
    if detect_formula_input(user_input):
        formula = user_input
        source_mode = "formula"
    else:
        ideas = generate_ideas_with_prompt(user_input, parents=_top_leaderboard_factors(), num_ideas=3)
        if not ideas:
            return fail("No factor ideas could be generated", 500)
        formula = ideas[0]["formula"]
        source_mode = ideas[0].get("source", "prompt")

    get_data_hub()
    try:
        hyp = (ideas[0].get("rationale") or user_input) if ideas else None
        result = quick_test(formula, manual_name, postprocess=postprocess, hypothesis=hyp)
        result["source_input"] = user_input
        result["source_mode"] = source_mode
        result["formula"] = formula
        if result.get("status") == "success":
            add_or_update_factor(result)
        if source_mode != "formula":
            try:
                append_factor_experience(
                    build_factor_experience_record(
                        result=result,
                        prompt=user_input,
                        rationale=hyp or "",
                        source=ideas[0].get("source", source_mode) if ideas else source_mode,
                        generation_mode=source_mode,
                        parents=_top_leaderboard_factors(),
                    )
                )
            except Exception:
                pass
        return ok({"factor": result, "ideas": ideas})
    except Exception as exc:
        return fail(str(exc), 500)


# ── Model / Strategy ──
@app.route("/api/model/<factor_name>", methods=["GET"])
def run_fit_model(factor_name):
    hub = get_data_hub()
    detail = load_factor_detail(factor_name)
    if not detail or not detail.get("formula"):
        return fail("Factor metric result not found", 404)
    try:
        series = compute_formula(detail["formula"], hub)
        result = fit_and_evaluate_models(series, hub, factor_name)
        return ok({"result": result})
    except Exception as exc:
        return fail(str(exc), 500)


@app.route("/api/strategy/<factor_name>", methods=["GET"])
def run_strategy(factor_name):
    hub = get_data_hub()
    detail = load_factor_detail(factor_name)
    if not detail or not detail.get("formula"):
        return fail("Factor metric result not found", 404)
    try:
        series = compute_formula(detail["formula"], hub)
        result = run_strategy_simulation(series, hub, factor_name)
        return ok({"result": result})
    except Exception as exc:
        return fail(str(exc), 500)


# ── System Config / LLM ──
@app.route("/api/system/config", methods=["GET"])
def get_system_config():
    return ok(load_runtime_env_payload())


@app.route("/api/system/config", methods=["PUT"])
def update_system_config():
    data = request.get_json(silent=True) or {}
    allowed = {
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "CHAT_MODEL",
        "REASONING_MODEL",
        "CHEAP_MODEL",
        "AUTOALPHA_DEFAULT_ROUNDS",
        "AUTOALPHA_DEFAULT_IDEAS",
        "AUTOALPHA_DEFAULT_DAYS",
        "AUTOALPHA_PROMPT_CONTEXT_LIMIT",
        "AUTOALPHA_QUOTA_DISPLAY_FX",
        "AUTOALPHA_ROLLING_TARGET_VALID",
        "AUTOALPHA_ROLLING_TRAIN_DAYS",
        "AUTOALPHA_ROLLING_TEST_DAYS",
        "AUTOALPHA_ROLLING_STEP_DAYS",
    }
    updates = {k: v for k, v in data.items() if k in allowed and v is not None}
    if not updates:
        return fail("No valid settings provided")
    save_runtime_config(updates)
    return ok(load_runtime_env_payload(), message="配置已保存")


@app.route("/api/system/llm-test", methods=["POST"])
def llm_test():
    from research.auto_agent import test_llm_connection

    result = test_llm_connection()
    if result.get("ok"):
        return ok(result, message="LLM API 可用")
    return jsonify({"success": False, "error": result.get("reason", "LLM test failed"), "data": result}), 400


# ── Lightweight Backtest Tasks ──
@app.route("/api/backtest/start", methods=["POST"])
def start_backtest():
    data = request.get_json(silent=True) or {}
    factor_json = data.get("factorJson") or "leaderboard"
    factor_source = data.get("factorSource") or "custom"
    task = create_backtest_task(
        {
            "factorJson": factor_json,
            "factorSource": factor_source,
            "configPath": data.get("configPath"),
        }
    )
    with _backtest_lock:
        _backtest_tasks[task["taskId"]] = task
    thread = threading.Thread(target=run_backtest_task, args=(task["taskId"],), daemon=True)
    _backtest_threads[task["taskId"]] = thread
    thread.start()
    return ok({"taskId": task["taskId"], "task": task}, message="回测任务已启动")


@app.route("/api/backtest/<task_id>", methods=["GET"])
def get_backtest_status(task_id):
    with _backtest_lock:
        task = _backtest_tasks.get(task_id)
    if not task:
        return fail("Task not found", 404)
    return ok({"task": task})


@app.route("/api/backtest/<task_id>", methods=["DELETE"])
def cancel_backtest(task_id):
    with _backtest_lock:
        task = _backtest_tasks.get(task_id)
    if not task:
        return fail("Task not found", 404)
    task["_cancelled"] = True
    task["status"] = "cancelled"
    task["progress"].update({"message": "回测已取消", "timestamp": now_iso()})
    task["updatedAt"] = now_iso()
    add_task_log(task, "Backtest cancelled by user", "warning")
    return ok({"task": task}, message="回测任务已取消")


# ── Factory / Auto Research ──
@app.route("/api/factory/status", methods=["GET"])
def get_factory_status():
    global _research_process
    lb = load_leaderboard()
    factors = lb.get("factors", [])
    best = max(factors, key=factor_sort_key, default={})
    recent_fail = next((f for f in reversed(factors) if not f.get("PassGates")), {})
    is_running = _research_process is not None and _research_process.poll() is None

    agents = [
        {
            "id": "planner",
            "name": "Research Planner (EA)",
            "status": "running" if is_running else "idle",
            "task": "Iterating evolutionary population",
        },
        {
            "id": "miner",
            "name": "Formula Miner",
            "status": "running" if is_running else "idle",
            "task": f"Mined {len(factors)} formulas",
        },
        {
            "id": "compliance",
            "name": "Compliance Guard",
            "status": "running" if is_running else "idle",
            "task": "Checking leakages on real data",
        },
        {
            "id": "evaluator",
            "name": "Evaluator",
            "status": "running" if is_running else "idle",
            "task": "Evaluating metrics on cached real data",
        },
    ]

    cmd_logs: List[str] = []
    if os.path.exists(RESEARCH_LOG_PATH):
        try:
            with open(RESEARCH_LOG_PATH, "r", encoding="utf-8") as handle:
                cmd_logs = [line.strip() for line in handle.readlines()[-400:] if line.strip()]
        except Exception:
            cmd_logs = []

    from runtime_config import get_llm_config

    llm_cfg = get_llm_config()
    llm_enabled = bool(llm_cfg.get("api_key"))

    return ok(
        {
            "global_state": {
                "is_running": is_running,
                "best_factor": best.get("factor_name", "None"),
                "best_score": best.get("Score", 0),
                "submission_ready_count": sum(
                    1
                    for f in factors
                    if f.get("submission_ready_flag") or f.get("classification") == "Submission Ready"
                ),
                "total_factors": len(factors),
                "recent_fail_reason": recent_fail.get("reason", "N/A"),
                "llm_enabled": llm_enabled,
                "research_log_path": RESEARCH_LOG_PATH,
                "llm_mining_log_path": LLM_MINING_JSONL,
            },
            "agents": agents,
            "cmd_logs": cmd_logs,
            "llm_mining_recent": read_recent_llm_mining_records(limit=60),
        }
    )


@app.route("/api/factory/start", methods=["POST"])
def start_factory():
    global _research_process
    if _research_process is not None and _research_process.poll() is None:
        return ok({"status": "already_running"}, message="研究循环已在运行")

    try:
        import pyarrow  # noqa: F401 — research_loop / DataHub 读 parquet 必需
    except ImportError:
        return fail(
            "当前运行 Flask 的 Python 未安装 pyarrow，无法启动研究循环（DataHub 无法读缓存 parquet）。"
            "请使用已安装依赖的 conda 环境启动：例如 `conda activate autoalpha && python server.py`，"
            "或执行 `pip install pyarrow`。",
            503,
        )

    data = request.get_json(silent=True) or {}
    direction = (data.get("direction") or "").strip()
    max_rounds = int(data.get("maxRounds") or 5)
    max_loops = int(data.get("maxLoops") or 5)
    max_iters = max(1, max_rounds * max_loops)
    batch_size = int(data.get("factorsPerHypothesis") or data.get("numDirections") or 4)

    from runtime_config import get_llm_config

    llm_config = get_llm_config()
    env = os.environ.copy()
    if llm_config.get("api_key"):
        env["OPENAI_API_KEY"] = llm_config["api_key"]
        env["OPENAI_BASE_URL"] = llm_config.get("api_base") or env.get("OPENAI_BASE_URL", "")
        env["CHAT_MODEL"] = llm_config.get("model") or env.get("CHAT_MODEL", "")
        env["CHEAP_MODEL"] = load_runtime_config().get("CHEAP_MODEL", env.get("CHAT_MODEL", ""))
        env["AUTOALPHA_USE_LLM"] = "1"
        env["ALPHACLAW_USE_LLM"] = "1"  # legacy alias for subprocess tools
    else:
        env["AUTOALPHA_USE_LLM"] = "0"
        env["ALPHACLAW_USE_LLM"] = "0"

    os.makedirs(os.path.dirname(RESEARCH_LOG_PATH), exist_ok=True)
    with open(RESEARCH_LOG_PATH, "w", encoding="utf-8") as handle:
        handle.write(f"[{now_iso()}] Starting research loop\n")
        if direction:
            handle.write(f"[{now_iso()}] Seed prompt: {direction}\n")

    # Use standard proxy settings
    env["NO_PROXY"] = "localhost,127.0.0.1"
    env["no_proxy"] = "localhost,127.0.0.1"

    log_file = open(RESEARCH_LOG_PATH, "a", encoding="utf-8")
    cmd = [
        sys.executable,
        "-u",  # Unbuffered stdout for real-time log streaming
        "research_loop.py",
        "--max-iters",
        str(max_iters),
        "--batch-size",
        str(max(1, batch_size)),
    ]
    if direction:
        cmd.extend(["--seed-prompt", direction])

    _research_process = subprocess.Popen(
        cmd,
        cwd=os.path.dirname(__file__),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=env,
    )
    return ok(
        {
            "status": "started",
            "llm_enabled": bool(llm_config.get("api_key")),
            "max_iters": max_iters,
            "batch_size": max(1, batch_size),
        },
        message="自动挖掘已启动（已配置 LLM 时将驱动因子生成）"
        if llm_config.get("api_key")
        else "自动挖掘已启动（未检测到 API Key，将使用进化算法离线生成）",
    )


@app.route("/api/factory/stop", methods=["POST"])
def stop_factory():
    global _research_process
    if _research_process is not None and _research_process.poll() is None:
        _research_process.terminate()
        _research_process = None
        append_research_log(f"[{now_iso()}] Research loop stopped by user")
        return ok({"status": "stopped"}, message="自动挖掘已停止")
    return ok({"status": "not_running"}, message="当前没有运行中的自动挖掘")


# ── AutoAlpha Loop Endpoints ──

def _load_autoalpha_kb() -> Dict[str, Any]:
    if os.path.exists(AUTOALPHA_KB_PATH):
        try:
            with open(AUTOALPHA_KB_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"total_tested": 0, "total_passing": 0, "best_score": 0, "factors": {}}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _compress_points(points: List[Dict[str, Any]], max_points: int = 36) -> List[Dict[str, Any]]:
    if len(points) <= max_points:
        return points
    if max_points <= 1:
        return [points[-1]]

    # Keep the whole timeline represented, especially the newest cumulative
    # point. The previous step-based sampler could append the latest point and
    # then slice it away, leaving charts stuck on older factor counts.
    last_index = len(points) - 1
    indices = {
        round((last_index * index) / (max_points - 1))
        for index in range(max_points)
    }
    indices.add(0)
    indices.add(last_index)
    return [points[index] for index in sorted(indices)]


def _list_autoalpha_output_files(limit: int = 20) -> List[Dict[str, Any]]:
    if not os.path.isdir(AUTOALPHA_OUTPUT_DIR):
        return []
    files: List[Dict[str, Any]] = []
    for entry in os.scandir(AUTOALPHA_OUTPUT_DIR):
        if not entry.is_file():
            continue
        kind = "parquet" if entry.name.endswith(".pq") else "manifest" if entry.name.endswith(".json") else "other"
        stat = entry.stat()
        files.append(
            {
                "name": entry.name,
                "path": entry.path,
                "relative_path": os.path.relpath(entry.path, os.path.dirname(__file__)),
                "kind": kind,
                "size_bytes": stat.st_size,
                "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            }
        )
    files.sort(key=lambda item: item["modified_at"], reverse=True)
    return files[:limit]


def _list_autoalpha_research_reports(limit: int = 20) -> List[Dict[str, Any]]:
    if not os.path.isdir(AUTOALPHA_RESEARCH_DIR):
        return []
    reports: List[Dict[str, Any]] = []
    for run_id in os.listdir(AUTOALPHA_RESEARCH_DIR):
        report_path = os.path.join(AUTOALPHA_RESEARCH_DIR, run_id, "report.json")
        if not os.path.isfile(report_path):
            continue
        stat = os.stat(report_path)
        card_path = os.path.join(AUTOALPHA_RESEARCH_DIR, run_id, "factor_card.json")
        reports.append(
            {
                "run_id": run_id,
                "path": report_path,
                "factor_card_path": card_path if os.path.isfile(card_path) else "",
                "relative_path": os.path.relpath(report_path, os.path.dirname(__file__)),
                "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "size_bytes": stat.st_size,
            }
        )
    reports.sort(key=lambda item: item["modified_at"], reverse=True)
    return reports[:limit]


def _read_json_if_exists(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _list_model_lab_runs(limit: int = 10) -> List[Dict[str, Any]]:
    if not os.path.isdir(AUTOALPHA_MODEL_LAB_DIR):
        return []

    runs: List[Dict[str, Any]] = []
    for entry in os.scandir(AUTOALPHA_MODEL_LAB_DIR):
        if not entry.is_dir() or not entry.name.startswith("run_"):
            continue
        summary_path = os.path.join(entry.path, "summary.json")
        summary = _read_json_if_exists(summary_path) or {}
        stat = os.stat(summary_path) if os.path.isfile(summary_path) else entry.stat()
        runs.append(
            {
                "run_id": entry.name,
                "path": entry.path,
                "relative_path": os.path.relpath(entry.path, os.path.dirname(__file__)),
                "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "target_valid_count": summary.get("target_valid_count"),
                "selected_factor_count": summary.get("selected_factor_count"),
                "window_count": summary.get("window_count"),
                "best_model": summary.get("best_model"),
                "models": summary.get("models", {}),
                "summary": summary,
            }
        )

    runs.sort(key=lambda item: item["modified_at"], reverse=True)
    return runs[:limit]


def _build_autoalpha_progress_points(all_factors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ordered = sorted(
        all_factors,
        key=lambda item: item.get("created_at", ""),
    )
    points: List[Dict[str, Any]] = []
    best_score = 0.0
    passing = 0
    for idx, factor in enumerate(ordered, start=1):
        if factor.get("PassGates"):
            passing += 1
        best_score = max(best_score, _safe_float(factor.get("Score", 0)))
        created_at = factor.get("created_at") or factor.get("updated_at") or ""
        points.append(
            {
                "index": idx,
                "timestamp": created_at,
                "label": created_at.replace("T", " ")[:16] if created_at else f"#{idx}",
                "tested": idx,
                "passing": passing,
                "best_score": round(best_score, 2),
            }
        )
    return _compress_points(points)


def _build_autoalpha_generation_summary(all_factors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    buckets: Dict[int, Dict[str, Any]] = {}
    for factor in all_factors:
        generation = _safe_int(factor.get("generation", 0))
        bucket = buckets.setdefault(
            generation,
            {"generation": generation, "total": 0, "passing": 0, "best_score": 0.0},
        )
        bucket["total"] += 1
        if factor.get("PassGates"):
            bucket["passing"] += 1
        bucket["best_score"] = max(bucket["best_score"], _safe_float(factor.get("Score", 0)))
    return [buckets[key] for key in sorted(buckets.keys())]


def _build_autoalpha_inspiration_stats(all_factors: List[Dict[str, Any]]) -> Dict[str, Any]:
    tracked_sources = ["arxiv", "llm", "future"]
    source_counts = {source: 0 for source in tracked_sources}
    try:
        for row in list_inspiration_source_counts():
            source_type = str(row.get("source_type") or "manual")
            if source_type in source_counts:
                source_counts[source_type] += _safe_int(row.get("count", 0))
    except Exception:
        pass

    tested = {source: 0 for source in tracked_sources}
    passing = {source: 0 for source in tracked_sources}
    ordered = sorted(all_factors, key=lambda item: item.get("created_at", ""))
    timeline_counts = {source: {"tested": 0, "passing": 0} for source in tracked_sources}
    timeline: List[Dict[str, Any]] = []
    for idx, factor in enumerate(ordered, start=1):
        source_type = str(factor.get("inspiration_source_type") or "none")
        if source_type not in tested:
            continue
        tested[source_type] += 1
        timeline_counts[source_type]["tested"] += 1
        if factor.get("PassGates"):
            passing[source_type] += 1
            timeline_counts[source_type]["passing"] += 1
        point = {"index": idx, "label": f"#{idx}"}
        for source in tracked_sources:
            t = timeline_counts[source]["tested"]
            p = timeline_counts[source]["passing"]
            point[f"{source}_tested"] = t
            point[f"{source}_passing"] = p
            point[f"{source}_pass_rate"] = round((p / max(t, 1)) * 100, 2)
        timeline.append(point)

    total_passing = sum(passing.values()) or sum(1 for factor in all_factors if factor.get("PassGates"))
    by_source = []
    for source in tracked_sources:
        prompt_count = source_counts[source]
        passing_count = passing[source]
        tested_count = tested[source]
        by_source.append({
            "source": source,
            "prompt_count": prompt_count,
            "tested_count": tested_count,
            "passing_count": passing_count,
            "pass_rate": round((passing_count / max(tested_count, 1)) * 100, 2),
            "valid_per_prompt": round(passing_count / max(prompt_count, 1), 4),
            "valid_share": round((passing_count / max(total_passing, 1)) * 100, 2),
        })

    return {
        "by_source": by_source,
        "timeline": _compress_points(timeline, max_points=48),
        "total_passing_attributed": total_passing,
    }


def _fetch_billing_payload(kind: str, headers: Dict[str, str]) -> tuple[Dict[str, Any], str, str]:
    import requests as rq

    last_error = ""
    for url in AUTOALPHA_BILLING_ENDPOINTS.get(kind, []):
        try:
            resp = rq.get(url, headers=headers, timeout=10, verify=False)
            if resp.status_code >= 400:
                friendly, _, _, raw = humanize_error(resp.text, status_code=resp.status_code)
                last_error = f"{friendly} 接口响应: {raw[:240]}".strip()
                continue
            data = resp.json()
            if isinstance(data, dict) and data.get("error"):
                friendly, _, _, raw = humanize_error(data.get("error"))
                last_error = f"{friendly} 接口响应: {str(raw)[:240]}".strip()
                continue
            return data if isinstance(data, dict) else {}, url, ""
        except Exception as exc:
            friendly, _, _, raw = humanize_error(exc)
            last_error = f"{friendly} 接口错误: {raw[:240]}".strip()
    return {}, "", last_error


def _quota_status(used_pct: float, remaining: float) -> str:
    if remaining <= 0.01:
        return "exhausted"
    if used_pct >= 90:
        return "critical"
    if used_pct >= 75:
        return "warning"
    return "healthy"


def _parse_live_result_text(text: str) -> Any:
    raw = (text or "").strip()
    if not raw:
        raise ValueError("Result file 内容为空")
    try:
        return json.loads(raw)
    except Exception:
        pass
    try:
        return ast.literal_eval(raw)
    except Exception as exc:
        raise ValueError(f"无法解析 Result file：{exc}") from exc


def _write_autoalpha_kb(kb: Dict[str, Any]) -> None:
    kb["updated_at"] = now_iso()
    with open(AUTOALPHA_KB_PATH, "w", encoding="utf-8") as handle:
        json.dump(kb, handle, indent=2, ensure_ascii=False, default=str)


@app.route("/api/autoalpha/knowledge", methods=["GET"])
def autoalpha_knowledge():
    """Return knowledge base overview + full factor list."""
    kb = _load_autoalpha_kb()
    all_factors = [
        {"run_id": rid, **info}
        for rid, info in kb.get("factors", {}).items()
    ]
    all_factors.sort(key=lambda x: x.get("Score", 0), reverse=True)
    for idx, factor in enumerate(all_factors, start=1):
        factor["rank"] = idx

    status_breakdown: Dict[str, int] = {}
    for factor in all_factors:
        status = factor.get("status", "unknown")
        status_breakdown[status] = status_breakdown.get(status, 0) + 1

    return ok({
        "total_tested": kb.get("total_tested", 0),
        "total_passing": kb.get("total_passing", 0),
        "best_score": kb.get("best_score", 0),
        "updated_at": kb.get("updated_at", ""),
        "factors": all_factors,
        "pass_rate": round((kb.get("total_passing", 0) / max(kb.get("total_tested", 1), 1)) * 100, 1),
        "status_breakdown": status_breakdown,
        "progress_points": _build_autoalpha_progress_points(all_factors),
        "generation_summary": _build_autoalpha_generation_summary(all_factors),
        "inspiration_stats": _build_autoalpha_inspiration_stats(all_factors),
        "artifacts": {
            "output_files": _list_autoalpha_output_files(),
            "research_reports": _list_autoalpha_research_reports(),
        },
    })


@app.route("/api/autoalpha/factors/<run_id>/live-result", methods=["POST"])
def autoalpha_factor_live_result(run_id: str):
    data = request.get_json(silent=True) or {}
    raw_text = data.get("result_text") or data.get("result") or data.get("raw") or ""
    try:
        parsed = _parse_live_result_text(str(raw_text))
    except ValueError as exc:
        return fail(str(exc), 400)

    kb = _load_autoalpha_kb()
    factors = kb.setdefault("factors", {})
    if run_id not in factors:
        return fail("Factor not found", 404)

    record = {
        "raw": str(raw_text).strip(),
        "data": parsed,
        "submitted_at": now_iso(),
    }
    factors[run_id]["live_test_result"] = record
    factors[run_id]["live_submitted"] = True
    factors[run_id]["live_result_updated_at"] = record["submitted_at"]
    _write_autoalpha_kb(kb)

    return ok({"run_id": run_id, "live_test_result": record, "live_submitted": True})


@app.route("/api/autoalpha_v2/submit/sync", methods=["POST"])
def autoalpha_submit_sync():
    try:
        from autoalpha_v2 import knowledge_base as kb

        return ok(kb.sync_submit_artifacts(), message="可提交因子已同步到 autoalpha_v2/submit")
    except Exception as exc:
        friendly, _, _, raw = humanize_error(exc)
        return fail(friendly or raw, 500)


@app.route("/api/autoalpha/balance", methods=["GET"])
def autoalpha_balance():
    """Read quota package usage and compute per-factor costs."""
    routing = get_llm_routing()
    api_key = routing.get("api_key", "")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    source_total_quota = 0.0
    source_used = 0.0
    warnings_list: List[str] = []
    sub_data, sub_source, sub_warning = _fetch_billing_payload("subscription", headers)
    usage_data, usage_source, usage_warning = _fetch_billing_payload("usage", headers)
    if sub_warning:
        warnings_list.append(f"订阅额度获取失败: {sub_warning}")
    if usage_warning:
        warnings_list.append(f"已用额度获取失败: {usage_warning}")

    source_total_quota = _safe_float(sub_data.get("hard_limit_usd", 0))
    source_used = _safe_float(usage_data.get("total_usage", 0)) / 100.0
    source_remaining = max(0.0, source_total_quota - source_used)

    # Convert provider units to the quota-pack display unit used by the frontend
    # (e.g. 438 -> 60.00, 94.53 -> 12.95).
    runtime_cfg = load_runtime_config()
    fx_rate = _safe_float(runtime_cfg.get("AUTOALPHA_QUOTA_DISPLAY_FX", AUTOALPHA_QUOTA_DISPLAY_FX), AUTOALPHA_QUOTA_DISPLAY_FX)
    fx_rate = fx_rate if fx_rate > 0 else 1.0
    total_quota = source_total_quota / fx_rate
    used = source_used / fx_rate
    remaining = source_remaining / fx_rate

    # Per-factor cost from knowledge base
    kb = _load_autoalpha_kb()
    all_factors = [
        {"run_id": rid, **info}
        for rid, info in kb.get("factors", {}).items()
    ]
    total_factors = len(all_factors)
    passing_factors = sum(1 for f in all_factors if f.get("PassGates"))

    avg_cost_per_factor = used / total_factors if total_factors > 0 else 0.0
    avg_cost_per_valid = used / passing_factors if passing_factors > 0 else 0.0

    # Estimate tokens from cost (Claude Sonnet 4: ~$3/Mtok input, ~$15/Mtok output, rough avg ~$6/Mtok)
    est_total_tokens = int(used / 6.0 * 1_000_000) if used > 0 else 0
    avg_tokens_per_factor = est_total_tokens // total_factors if total_factors > 0 else 0
    avg_tokens_per_valid = est_total_tokens // passing_factors if passing_factors > 0 else 0
    used_pct = round(used / total_quota * 100, 1) if total_quota > 0 else 0.0
    remaining_pct = round(remaining / total_quota * 100, 1) if total_quota > 0 else 0.0

    return ok({
        "total_quota": round(total_quota, 2),
        "used": round(used, 2),
        "remaining": round(remaining, 2),
        "used_pct": used_pct,
        "remaining_pct": remaining_pct,
        "quota_status": _quota_status(used_pct, remaining),
        "total_factors": total_factors,
        "passing_factors": passing_factors,
        "pass_rate": round((passing_factors / max(total_factors, 1)) * 100, 1),
        "avg_cost_per_factor": round(avg_cost_per_factor, 4),
        "avg_cost_per_valid_factor": round(avg_cost_per_valid, 4),
        "est_total_tokens": est_total_tokens,
        "avg_tokens_per_factor": avg_tokens_per_factor,
        "avg_tokens_per_valid_factor": avg_tokens_per_valid,
        "warnings": warnings_list,
    })


@app.route("/api/autoalpha/inspirations", methods=["GET"])
def autoalpha_inspirations():
    sync_prompt_directory(limit=80)
    page = request.args.get("page")
    per_page = request.args.get("per_page")
    source_type = request.args.get("source_type")
    search = request.args.get("search")
    include_inactive = str(request.args.get("include_inactive", "")).lower() in {"1", "true", "yes", "on"}

    if page is not None or per_page is not None or source_type or search or include_inactive:
        payload = list_inspirations_paginated(
            page=max(1, _safe_int(page, 1)),
            per_page=min(max(1, _safe_int(per_page, 20)), 100),
            source_type=source_type,
            search=search,
            include_inactive=include_inactive,
        )
        payload.update(
            {
                "count": payload.get("total", 0),
                "prompt_dir": str(AUTOALPHA_PROMPT_DIR),
                "database_path": str(AUTOALPHA_DB_PATH),
                "prompt_context_preview": compose_inspiration_context(limit=4, max_chars=900),
            }
        )
        return ok(payload)

    items = list_recent_inspirations(limit=12)
    return ok(
        {
            "items": items,
            "count": len(items),
            "prompt_dir": str(AUTOALPHA_PROMPT_DIR),
            "database_path": str(AUTOALPHA_DB_PATH),
            "prompt_context_preview": compose_inspiration_context(limit=4, max_chars=900),
        }
    )


@app.route("/api/autoalpha/inspirations/browse", methods=["GET"])
def autoalpha_inspirations_browse():
    """Paginated inspiration browser endpoint used by the dedicated inspiration page."""
    sync_prompt_directory(limit=80)
    payload = list_inspirations_paginated(
        page=max(1, _safe_int(request.args.get("page"), 1)),
        per_page=min(max(1, _safe_int(request.args.get("per_page"), 20)), 100),
        source_type=request.args.get("source_type"),
        search=request.args.get("search"),
        include_inactive=str(request.args.get("include_inactive", "")).lower() in {"1", "true", "yes", "on"},
    )
    payload.update(
        {
            "count": payload.get("total", 0),
            "prompt_dir": str(AUTOALPHA_PROMPT_DIR),
            "database_path": str(AUTOALPHA_DB_PATH),
            "prompt_context_preview": compose_inspiration_context(limit=4, max_chars=900),
        }
    )
    return ok(payload)


@app.route("/api/autoalpha/inspirations", methods=["POST"])
def autoalpha_add_inspiration():
    data = request.get_json(silent=True) or {}
    user_input = (
        data.get("input")
        or data.get("raw_input")
        or data.get("prompt")
        or data.get("url")
        or ""
    ).strip()
    title = (data.get("title") or "").strip()
    source_type = (data.get("source_type") or "").strip()
    if not user_input:
        return fail("Prompt / URL is required")

    record = prepare_inspiration(user_input, title=title)
    if source_type:
        record["source_type"] = source_type
    summary = summarize_inspiration_text(record.get("content", ""), source_hint=record.get("source", ""))
    if summary:
        record["summary"] = summary
    saved = save_inspiration(record)
    if summary and saved.get("id"):
        update_inspiration_summary(int(saved["id"]), summary)
        saved["summary"] = summary

    items = list_recent_inspirations(limit=12)
    return ok(
        {
            "item": saved,
            "items": items,
            "count": len(items),
            "prompt_dir": str(AUTOALPHA_PROMPT_DIR),
            "database_path": str(AUTOALPHA_DB_PATH),
        },
        message="灵感已写入 AutoAlpha 灵感库",
    )


@app.route("/api/autoalpha/inspirations/fetch", methods=["POST"])
def autoalpha_fetch_inspirations():
    """Fetch fresh inspirations from ArXiv, optional URLs, and optional LLM brainstorm."""
    data = request.get_json(silent=True) or {}
    llm_ideas = max(0, min(_safe_int(request.args.get("llm_ideas", data.get("llm_ideas", 6)), 6), 20))
    arxiv_per_query = max(
        0,
        min(_safe_int(request.args.get("arxiv_per_query", data.get("arxiv_per_query", 5)), 5), 20),
    )
    raw_urls = data.get("urls") or data.get("url") or request.args.get("url") or []
    if isinstance(raw_urls, str):
        raw_urls = [item.strip() for item in re.split(r"[\n,]+", raw_urls) if item.strip()]
    extra_urls = [url for url in raw_urls if isinstance(url, str) and url.startswith(("http://", "https://"))]
    query = (data.get("arxiv_query") or request.args.get("arxiv_query") or "").strip()

    try:
        from autoalpha_v2.inspiration_fetcher import run_fetch_cycle

        added = run_fetch_cycle(
            arxiv_queries=[query] if query else None,
            extra_urls=extra_urls,
            llm_ideas=llm_ideas,
            arxiv_per_query=arxiv_per_query,
        )
        sync_payload = sync_prompt_directory(limit=120)
        items = list_recent_inspirations(limit=12)
        return ok(
            {
                "added": added,
                "items": items,
                "count": len(items),
                "total": sync_payload.get("total", len(items)),
                "prompt_dir": str(AUTOALPHA_PROMPT_DIR),
                "database_path": str(AUTOALPHA_DB_PATH),
                "prompt_context_preview": compose_inspiration_context(limit=4, max_chars=900),
            },
            message=(
                f"抓取完成：ArXiv {added.get('arxiv', 0)}，URL {added.get('url', 0)}，"
                f"LLM {added.get('llm', 0)}，期货 {added.get('future', 0)}，重复 {added.get('skipped', 0)}"
            ),
        )
    except Exception as exc:
        friendly, _, _, raw = humanize_error(exc)
        return fail(f"抓取新灵感失败: {friendly or raw}", 500)


@app.route("/api/autoalpha/inspirations/sync", methods=["POST"])
def autoalpha_sync_inspirations():
    payload = sync_prompt_directory(limit=120)
    payload["count"] = len(payload.get("items", []))
    payload["prompt_context_preview"] = compose_inspiration_context(limit=4, max_chars=900)
    return ok(payload, message="AutoAlpha 目录灵感文件已同步")


@app.route("/api/autoalpha/inspirations/<int:entry_id>/toggle", methods=["PUT", "POST"])
def autoalpha_toggle_inspiration(entry_id: int):
    try:
        return ok({"item": toggle_inspiration_status(entry_id)})
    except Exception as exc:
        return fail(str(exc), 404)


@app.route("/api/autoalpha/inspirations/<int:entry_id>", methods=["DELETE"])
def autoalpha_delete_inspiration(entry_id: int):
    if delete_inspiration(entry_id):
        return ok({"deleted": True})
    return fail("Inspiration not found", 404)


@app.route("/api/autoalpha/idea-cache/status", methods=["GET"])
def autoalpha_idea_cache_status():
    try:
        from autoalpha_v2.idea_cache import get_default_cache

        return ok(get_default_cache().status())
    except Exception as exc:
        return fail(str(exc), 500)


@app.route("/api/autoalpha/loop/start", methods=["POST"])
def autoalpha_loop_start():
    global _autoalpha_loop_process
    running_pid = _tracked_autoalpha_loop_pid()
    if running_pid is not None:
        return ok({"status": "already_running"}, message="挖掘循环已在运行")

    data = request.get_json(silent=True) or {}
    cfg = load_runtime_config()
    rounds = int(data.get("rounds", cfg.get("AUTOALPHA_DEFAULT_ROUNDS", 0) or 0))
    ideas  = int(data.get("ideas", cfg.get("AUTOALPHA_DEFAULT_IDEAS", 4) or 4))
    days   = int(data.get("days", cfg.get("AUTOALPHA_DEFAULT_DAYS", 0) or 0))
    target_valid = int(data.get("target_valid", cfg.get("AUTOALPHA_DEFAULT_TARGET_VALID", 100) or 100))
    run_model_lab = bool(data.get("run_model_lab", False))
    rounds = max(0, rounds)
    ideas = max(1, min(ideas, 20))
    if days < 0:
        days = 0

    loop_script = os.path.join(os.path.dirname(__file__), "autoalpha_v2", "loop.py")
    os.makedirs(os.path.dirname(AUTOALPHA_LOG_PATH), exist_ok=True)
    log_file = open(AUTOALPHA_LOG_PATH, "w", encoding="utf-8")
    params = {
        "rounds": rounds,
        "ideas": ideas,
        "days": days,
        "target_valid": target_valid,
        "run_model_lab": run_model_lab,
    }
    log_file.write(
        f"[{now_iso()}] Loop started (rounds={rounds} ideas={ideas} days={days} target_valid={target_valid} run_model_lab={run_model_lab})\n"
    )
    log_file.flush()

    _autoalpha_loop_process = subprocess.Popen(
        [sys.executable, "-u", loop_script,
         "--rounds", str(rounds),
         "--ideas", str(ideas),
         "--days", str(days),
         "--target-valid", str(target_valid),
         *([] if not run_model_lab else ["--run-model-lab"])],
        cwd=os.path.dirname(__file__),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    _write_autoalpha_loop_state(int(_autoalpha_loop_process.pid), params)
    return ok({
        "status": "started",
        "pid": int(_autoalpha_loop_process.pid),
        "rounds": rounds,
        "ideas": ideas,
        "days": days,
        "target_valid": target_valid,
        "run_model_lab": run_model_lab,
    }, message=(
        f"已启动挖掘循环：{rounds} 轮，每轮 {ideas} 个因子，评估窗口="
        f"{'全量交易日' if days <= 0 else f'{days} 个交易日'}"
        f"{'' if target_valid <= 0 else f'，目标有效因子={target_valid}'}"
    ))


@app.route("/api/autoalpha/loop/stop", methods=["POST"])
def autoalpha_loop_stop():
    global _autoalpha_loop_process
    pid = _tracked_autoalpha_loop_pid()
    if pid is not None:
        try:
            if _autoalpha_loop_process is not None and _autoalpha_loop_process.poll() is None:
                _autoalpha_loop_process.terminate()
            else:
                os.kill(pid, 15)
        finally:
            _autoalpha_loop_process = None
        return ok({"status": "stopped", "pid": pid}, message="挖掘循环已停止")
    return ok({"status": "not_running"}, message="当前没有运行中的挖掘循环")


@app.route("/api/autoalpha/loop/status", methods=["GET"])
def autoalpha_loop_status():
    global _autoalpha_loop_process
    running_pid = _tracked_autoalpha_loop_pid()
    is_running = running_pid is not None

    # Read recent log lines
    logs: List[str] = []
    if os.path.exists(AUTOALPHA_LOG_PATH):
        try:
            with open(AUTOALPHA_LOG_PATH, "r", encoding="utf-8") as f:
                logs = [
                    line.replace("\x00", "").rstrip()
                    for line in f.readlines()[-200:]
                    if line.replace("\x00", "").strip()
                ]
        except Exception:
            pass

    # Knowledge base summary
    kb = _load_autoalpha_kb()
    return ok({
        "is_running": is_running,
        "pid": running_pid,
        "run_state": _read_autoalpha_loop_meta(),
        "total_tested": kb.get("total_tested", 0),
        "total_passing": kb.get("total_passing", 0),
        "best_score": kb.get("best_score", 0),
        "updated_at": kb.get("updated_at", ""),
        "logs": logs,
    })


@app.route("/api/autoalpha/research/<run_id>", methods=["GET"])
def autoalpha_research(run_id: str):
    """Return factor research report JSON for a specific run_id."""
    report_path = os.path.join(
        os.path.dirname(__file__), "autoalpha_v2", "research", run_id, "report.json"
    )
    if not os.path.exists(report_path):
        return fail("Research report not found", 404)
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        card_path = os.path.join(
            os.path.dirname(__file__), "autoalpha_v2", "research", run_id, "factor_card.json"
        )
        if "factor_card" not in report and os.path.exists(card_path):
            with open(card_path, "r", encoding="utf-8") as f:
                report["factor_card"] = json.load(f)
        if os.path.exists(card_path):
            report["factor_card_path"] = card_path
        return ok({"report": report})
    except Exception as e:
        return fail(str(e), 500)


@app.route("/api/autoalpha/model-lab", methods=["GET"])
def autoalpha_model_lab():
    """Return latest rolling model-lab summary plus recent run list."""
    runs = _list_model_lab_runs(limit=8)
    latest = runs[0]["summary"] if runs else None
    return ok(
        {
            "latest": latest,
            "runs": [
                {
                    "run_id": run["run_id"],
                    "relative_path": run["relative_path"],
                    "modified_at": run["modified_at"],
                    "target_valid_count": run.get("target_valid_count"),
                    "selected_factor_count": run.get("selected_factor_count"),
                    "window_count": run.get("window_count"),
                    "best_model": run.get("best_model"),
                    "models": run.get("models", {}),
                }
                for run in runs
            ],
        }
    )


if __name__ == "__main__":
    print("AutoAlpha Server (Real Data Engine) starting...")
    app.run(host="127.0.0.1", port=8080, debug=False, threaded=True)
