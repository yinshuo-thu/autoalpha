from __future__ import annotations

import json
import math
import os
import re
from datetime import datetime
from typing import Any, Dict, List

from paths import LLM_EXPERIENCE_DOC_PATH, LLM_EXPERIENCE_JSONL


STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "this",
    "that",
    "from",
    "into",
    "rank",
    "score",
    "factor",
    "alpha",
    "bars",
    "bar",
    "close",
    "open",
    "high",
    "low",
    "volume",
    "trade",
    "price",
    "using",
    "used",
    "over",
    "when",
    "then",
    "keep",
    "generate",
    "current",
    "prior",
    "idea",
    "ideas",
    "llm",
    "json",
    "only",
    "one",
    "new",
    "dsl",
    "need",
}


def _now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M")


def _ensure_parent(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _tokenize(text: str) -> set[str]:
    tokens = set()
    for token in re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{1,}", (text or "").lower()):
        if token in STOPWORDS or len(token) <= 2:
            continue
        tokens.add(token)
    return tokens


def _safe_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(parsed):
        return 0.0
    return parsed


def _failed_gate_names(result: Dict[str, Any]) -> List[str]:
    detail = result.get("gates_detail") or {}
    failed = []
    for name, value in detail.items():
        if value in (False, 0, "false", "False"):
            failed.append(str(name))
    return failed


def _truncate(text: str, limit: int = 220) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _build_lesson(result: Dict[str, Any], rationale: str) -> str:
    status = result.get("status", "")
    classification = result.get("classification", "")
    if status and status != "success":
        reason = result.get("reason") or "evaluation failed"
        return _truncate(f"{classification or status}: {reason}")
    if result.get("PassGates"):
        return _truncate(
            "Passed gates with "
            f"Score={_safe_float(result.get('Score')):.3f}, "
            f"IC={_safe_float(result.get('IC')):.4f}, "
            f"IR={_safe_float(result.get('IR')):.2f}, "
            f"Turnover={_safe_float(result.get('Turnover')):.2f}. "
            f"{rationale or 'This structure is worth reusing as a parent or reference.'}"
        )
    failed_gates = _failed_gate_names(result)
    if failed_gates:
        return _truncate(
            f"Did not pass gates. Failed: {', '.join(failed_gates)}. "
            f"{result.get('reason') or rationale or 'Use as inspiration, not as a template to copy directly.'}"
        )
    return _truncate(
        f"{classification or 'Evaluated'} with "
        f"Score={_safe_float(result.get('Score')):.3f}, "
        f"IC={_safe_float(result.get('IC')):.4f}, "
        f"IR={_safe_float(result.get('IR')):.2f}. "
        f"{result.get('reason') or rationale or 'Mixed result.'}"
    )


def build_factor_experience_record(
    *,
    result: Dict[str, Any],
    prompt: str = "",
    rationale: str = "",
    source: str = "",
    generation_mode: str = "",
    depth_hint: str = "",
    parents: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    formula = (result.get("formula") or "").strip()
    parent_formulas = []
    for parent in parents or []:
        formula_text = (parent.get("formula") or "").strip()
        if formula_text:
            parent_formulas.append(formula_text)
    return {
        "logged_at": _now_iso(),
        "factor_name": result.get("factor_name", ""),
        "formula": formula,
        "prompt": prompt,
        "rationale": rationale,
        "source": source,
        "generation_mode": generation_mode,
        "depth_hint": depth_hint,
        "status": result.get("status", ""),
        "classification": result.get("classification", ""),
        "pass_gates": bool(result.get("PassGates")),
        "submission_ready": bool(result.get("submission_ready_flag")),
        "score": _safe_float(result.get("Score")),
        "ic": _safe_float(result.get("IC")),
        "ir": _safe_float(result.get("IR")),
        "turnover": _safe_float(result.get("Turnover")),
        "reason": result.get("reason", ""),
        "failed_gates": _failed_gate_names(result),
        "validation_errors": list((result.get("validation") or {}).get("errors", [])),
        "parent_formulas": parent_formulas[:4],
        "lesson": _build_lesson(result, rationale),
    }


def append_factor_experience(record: Dict[str, Any], *, path: str | None = None) -> str:
    out = path or LLM_EXPERIENCE_JSONL
    _ensure_parent(out)
    row = dict(record)
    row.setdefault("logged_at", _now_iso())
    with open(out, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    refresh_experience_doc()
    return out


def load_factor_experiences(limit: int | None = None, *, path: str | None = None) -> List[Dict[str, Any]]:
    rows = _read_jsonl(path or LLM_EXPERIENCE_JSONL)
    if limit is not None and limit > 0:
        return rows[-limit:]
    return rows


def retrieve_relevant_experiences(query_text: str, limit: int = 4) -> List[Dict[str, Any]]:
    rows = load_factor_experiences(limit=300)
    if not rows:
        return []
    query_tokens = _tokenize(query_text)
    scored: List[tuple[float, Dict[str, Any]]] = []
    for recency_rank, row in enumerate(reversed(rows), start=1):
        searchable = " ".join(
            [
                row.get("prompt", ""),
                row.get("formula", ""),
                row.get("rationale", ""),
                row.get("lesson", ""),
                row.get("classification", ""),
                " ".join(row.get("failed_gates", [])),
            ]
        )
        tokens = _tokenize(searchable)
        overlap = len(query_tokens & tokens) if query_tokens else 0
        quality_bonus = 3.0 if row.get("pass_gates") else (1.0 if row.get("status") == "success" else 0.0)
        score_bonus = max(min(_safe_float(row.get("score")), 10.0), 0.0) * 0.2
        recency_bonus = max(0.2, 2.5 / recency_rank)
        failure_bonus = 0.4 if row.get("failed_gates") else 0.0
        score = overlap * 4.0 + quality_bonus + score_bonus + recency_bonus + failure_bonus
        if score <= 0:
            continue
        scored.append((score, row))

    seen_formulas = set()
    picked: List[Dict[str, Any]] = []
    for _, row in sorted(scored, key=lambda item: item[0], reverse=True):
        formula = row.get("formula", "")
        if formula and formula in seen_formulas:
            continue
        if formula:
            seen_formulas.add(formula)
        picked.append(row)
        if len(picked) >= limit:
            break
    return picked


def format_experiences_for_prompt(experiences: List[Dict[str, Any]]) -> str:
    if not experiences:
        return ""
    lines = []
    for index, item in enumerate(experiences, start=1):
        outcome = "PASS" if item.get("pass_gates") else (item.get("classification") or item.get("status") or "record")
        formula = _truncate(item.get("formula", ""), 140)
        lesson = _truncate(item.get("lesson", ""), 180)
        lines.append(
            f"{index}. outcome={outcome} | formula={formula} | lesson={lesson}"
        )
    return "\n".join(lines)


def refresh_experience_doc(limit: int = 120) -> str:
    rows = load_factor_experiences(limit=limit)
    _ensure_parent(LLM_EXPERIENCE_DOC_PATH)

    lines = [
        "# LLM 因子经验库",
        "",
        f"更新时间：{_now_iso()}",
        "",
        "这些经验用于检索和参考，不是硬约束。后续生成可以参考它们，也可以明确偏离它们去做更正交的探索。",
        "",
    ]

    passed = sorted(
        [row for row in rows if row.get("pass_gates")],
        key=lambda row: (_safe_float(row.get("score")), _safe_float(row.get("ic"))),
        reverse=True,
    )[:10]
    failed = [row for row in reversed(rows) if row.get("failed_gates") or row.get("status") != "success"][:10]
    recent = list(reversed(rows[-12:]))

    lines.extend(["## 高价值经验", ""])
    if passed:
        for row in passed:
            lines.append(
                f"- `{row.get('factor_name') or 'unnamed'}` | "
                f"Score={_safe_float(row.get('score')):.3f} | "
                f"formula=`{_truncate(row.get('formula', ''), 120)}` | "
                f"{row.get('lesson', '')}"
            )
    else:
        lines.append("- 还没有通过 Gate 的经验记录。")

    lines.extend(["", "## 常见失败模式", ""])
    if failed:
        for row in failed:
            failed_gates = ", ".join(row.get("failed_gates", [])) or row.get("status", "unknown")
            lines.append(
                f"- `{row.get('factor_name') or 'unnamed'}` | "
                f"failed=`{failed_gates}` | "
                f"formula=`{_truncate(row.get('formula', ''), 120)}` | "
                f"{row.get('lesson', '')}"
            )
    else:
        lines.append("- 暂无失败记录。")

    lines.extend(["", "## 最近生成记录", ""])
    if recent:
        for row in recent:
            lines.append(
                f"- `{row.get('logged_at', '')}` | "
                f"source={row.get('source') or row.get('generation_mode') or 'unknown'} | "
                f"`{row.get('factor_name') or 'unnamed'}` | "
                f"Score={_safe_float(row.get('score')):.3f} | "
                f"{_truncate(row.get('lesson', ''), 180)}"
            )
    else:
        lines.append("- 暂无记录。")

    with open(LLM_EXPERIENCE_DOC_PATH, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines).rstrip() + "\n")
    return LLM_EXPERIENCE_DOC_PATH
