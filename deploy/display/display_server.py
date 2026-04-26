#!/usr/bin/env python3
"""Read-only AutoAlpha v2 display server.

This server is intentionally small: it serves the built frontend and precomputed
JSON snapshots only. It does not import the research package, start loops, or
touch parquet/raw data.
"""

from __future__ import annotations

import json
import math
import os
import re
import gzip
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, redirect, request, send_from_directory


ROOT = Path(os.environ.get("AUTOALPHA_DISPLAY_ROOT") or Path(__file__).resolve().parent)
DIST_DIR = ROOT / "frontend" / "dist"
SNAPSHOT_DIR = ROOT / "data" / "snapshots"
RESEARCH_DIR = ROOT / "data" / "research"
GENERATION_NOTES_DIR = ROOT / "data" / "generation_notes"

app = Flask(__name__, static_folder=None)


@app.after_request
def add_display_cache_headers(response):
    path = request.path or ""
    if path.startswith("/v2/assets/"):
        response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
    elif path == "/v2/st-logo.png" or path == "/st-logo.png":
        response.headers["Cache-Control"] = "public, max-age=86400"
    elif path in {"/v2", "/v2/"} or path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    if (
        "gzip" in (request.headers.get("Accept-Encoding") or "").lower()
        and "Content-Encoding" not in response.headers
        and not response.direct_passthrough
        and response.status_code < 300
        and response.content_length is not None
        and response.content_length > 1024
        and (response.mimetype or "").split(";")[0] in {"application/json", "text/html", "text/css", "application/javascript"}
    ):
        compressed = gzip.compress(response.get_data(), compresslevel=6)
        response.set_data(compressed)
        response.headers["Content-Encoding"] = "gzip"
        response.headers["Content-Length"] = str(len(compressed))
        response.headers["Vary"] = "Accept-Encoding"
    return response


def ok(data: Any = None, message: str | None = None):
    payload: dict[str, Any] = {"success": True, "data": data if data is not None else {}}
    if message:
        payload["message"] = message
    return jsonify(payload)


def fail(error: str, status: int = 400):
    return jsonify({"success": False, "error": error}), status


def read_json(name: str, default: Any = None) -> Any:
    path = SNAPSHOT_DIR / name
    if not path.is_file():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def snapshot_response(name: str):
    payload = read_json(name)
    if payload is None:
        return fail(f"Snapshot not found: {name}", 404)
    return jsonify(payload)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_run_id(run_id: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9_.-]+", run_id or ""))


def paginate_items(items: list[dict[str, Any]], page: int, per_page: int) -> dict[str, Any]:
    total = len(items)
    pages = max(1, math.ceil(total / per_page)) if per_page else 1
    start = (page - 1) * per_page
    return {
        "items": items[start : start + per_page],
        "total": total,
        "count": total,
        "page": page,
        "per_page": per_page,
        "pages": pages,
    }


@app.get("/")
def root():
    return redirect("/v2/", code=302)


@app.get("/v2")
@app.get("/v2/")
def frontend_index():
    return send_from_directory(DIST_DIR, "index.html")


@app.get("/st-logo.png")
def frontend_logo_alias():
    return send_from_directory(DIST_DIR, "st-logo.png")


@app.get("/v2/<path:path>")
def frontend_asset(path: str):
    candidate = DIST_DIR / path
    if candidate.is_file():
        return send_from_directory(DIST_DIR, path)
    return send_from_directory(DIST_DIR, "index.html")


@app.get("/api/health")
def health():
    return ok(
        {
            "status": "ok",
            "mode": "display-only",
            "timestamp": utc_now(),
            "root": str(ROOT),
        }
    )


@app.route("/api/system/config", methods=["GET", "POST"])
def system_config():
    if request.method == "POST":
        return fail("Display-only snapshot is read-only.", 403)
    return snapshot_response("system_config.json")


@app.post("/api/system/llm-test")
def system_llm_test():
    return fail("Display-only snapshot does not call LLM services.", 403)


@app.get("/api/autoalpha/knowledge")
def autoalpha_knowledge():
    compact = str(request.args.get("compact_factors", "")).lower() in {"1", "true", "yes"}
    return snapshot_response("knowledge_compact.json" if compact else "knowledge_table.json")


@app.get("/api/autoalpha/model-lab")
def autoalpha_model_lab():
    return snapshot_response("model_lab.json")


@app.route("/api/autoalpha/loop/status", methods=["GET"])
def autoalpha_loop_status():
    return snapshot_response("loop_status.json")


@app.route("/api/autoalpha/loop/start", methods=["POST"])
@app.route("/api/autoalpha/loop/stop", methods=["POST"])
def autoalpha_loop_readonly():
    return fail("Display-only snapshot cannot start or stop mining loops.", 403)


@app.get("/api/autoalpha/balance")
def autoalpha_balance():
    return snapshot_response("balance.json")


@app.route("/api/autoalpha/inspirations", methods=["GET", "POST"])
def autoalpha_inspirations():
    if request.method == "POST":
        return fail("Display-only snapshot cannot add inspirations.", 403)
    return snapshot_response("inspirations.json")


@app.route("/api/autoalpha/inspirations/sync", methods=["POST"])
@app.route("/api/autoalpha/inspirations/fetch", methods=["POST"])
@app.route("/api/autoalpha/inspirations/<int:inspiration_id>/toggle", methods=["PUT"])
@app.route("/api/autoalpha/inspirations/<int:inspiration_id>", methods=["DELETE"])
def autoalpha_inspiration_readonly(inspiration_id: int | None = None):
    return fail("Display-only snapshot cannot modify inspirations.", 403)


@app.get("/api/autoalpha/inspirations/browse")
def autoalpha_inspirations_browse():
    payload = read_json("inspirations_browse.json", {"success": True, "data": {"items": []}})
    data = payload.get("data", payload) if isinstance(payload, dict) else {"items": []}
    items = data.get("items", []) if isinstance(data, dict) else []
    if not isinstance(items, list):
        items = []

    source_type = (request.args.get("source_type") or "all").strip()
    search = (request.args.get("search") or "").strip().lower()
    include_inactive = str(request.args.get("include_inactive", "")).lower() in {"1", "true", "yes", "on"}
    page = max(1, int(request.args.get("page") or 1))
    per_page = min(max(1, int(request.args.get("per_page") or 20)), 200)

    filtered: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if not include_inactive and item.get("status") == "inactive":
            continue
        if source_type and source_type != "all" and item.get("source_type") != source_type:
            continue
        if search:
            haystack = " ".join(str(item.get(k, "")) for k in ("title", "summary", "content", "tags", "source")).lower()
            if search not in haystack:
                continue
        filtered.append(item)

    result = paginate_items(filtered, page=page, per_page=per_page)
    for key in ("prompt_dir", "database_path", "prompt_context_preview"):
        if isinstance(data, dict) and key in data:
            result[key] = data[key]
    return ok(result)


@app.get("/api/autoalpha/idea-cache/status")
def autoalpha_idea_cache_status():
    inspiration_payload = read_json("inspirations_browse.json", {"data": {"total": 0}})
    data = inspiration_payload.get("data", {}) if isinstance(inspiration_payload, dict) else {}
    total = data.get("total") or len(data.get("items", [])) if isinstance(data, dict) else 0
    return ok({"pending": 0, "consumed": int(total or 0), "total": int(total or 0), "fill_running": False})


@app.get("/api/autoalpha/factor-correlations")
def autoalpha_factor_correlations():
    return snapshot_response("factor_correlations.json")


@app.get("/api/autoalpha/research/<run_id>")
def autoalpha_research(run_id: str):
    if not safe_run_id(run_id):
        return fail("Invalid run_id", 400)
    report_path = RESEARCH_DIR / run_id / "report.json"
    if not report_path.is_file():
        return fail("Research report not found", 404)
    with report_path.open("r", encoding="utf-8") as f:
        report = json.load(f)
    card_path = RESEARCH_DIR / run_id / "factor_card.json"
    if card_path.is_file() and not isinstance(report.get("factor_card"), dict):
        with card_path.open("r", encoding="utf-8") as f:
            report["factor_card"] = json.load(f)

    if str(request.args.get("full", "")).lower() not in {"1", "true", "yes"}:
        report = {
            "run_id": report.get("run_id", run_id),
            "formula": report.get("formula", ""),
            "metrics": report.get("metrics", {}),
            "factor_card": report.get("factor_card"),
            "factor_card_path": str(card_path) if card_path.is_file() else "",
            "created_at": report.get("created_at", ""),
            "n_daily_ic": report.get("n_daily_ic", 0),
        }
    return ok({"report": report})


@app.route("/api/autoalpha/generation-experience/<int:generation>", methods=["GET", "POST"])
def autoalpha_generation_experience(generation: int):
    if request.method == "POST":
        return fail("Display-only snapshot cannot regenerate generation notes.", 403)
    note_path = GENERATION_NOTES_DIR / f"generation_{generation:03d}.md"
    if not note_path.is_file():
        note_path = GENERATION_NOTES_DIR / f"generation_{generation}.md"
    if not note_path.is_file():
        return fail("Generation note not found in display snapshot.", 404)
    markdown = note_path.read_text(encoding="utf-8")
    summary = "\n".join(line for line in markdown.splitlines() if line.strip())[:500]
    return ok(
        {
            "generation": generation,
            "created_at": datetime.fromtimestamp(note_path.stat().st_mtime).isoformat(),
            "path": str(note_path),
            "relative_path": str(note_path.relative_to(ROOT)),
            "summary": summary,
            "markdown": markdown,
        }
    )


@app.route("/api/autoalpha/factors/<run_id>/live-result", methods=["POST"])
def autoalpha_factor_live_result(run_id: str):
    return fail("Display-only snapshot cannot update live results.", 403)


@app.get("/api/autoalpha/factors/<run_id>/download")
def autoalpha_factor_download(run_id: str):
    return fail("Display bundle excludes parquet/download artifacts by design.", 404)


@app.post("/api/formula/execute")
def formula_execute():
    return fail("Display-only snapshot cannot execute formulas.", 403)


if __name__ == "__main__":
    host = os.environ.get("AUTOALPHA_DISPLAY_HOST", "127.0.0.1")
    port = int(os.environ.get("AUTOALPHA_DISPLAY_PORT", "8080"))
    app.run(host=host, port=port)
