"""
LLM 因子挖掘审计日志：追加写入 JSONL（每行一条 JSON），便于保留完整思路与模型输出。
时间戳字段统一精确到分钟（与 submit 目录时间戳一致）。
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from paths import LLM_MINING_JSONL, LLM_MINING_LOG_DIR


def _now_minute_iso() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M")


def append_llm_mining_record(
    record: Dict[str, Any],
    *,
    path: Optional[str] = None,
) -> str:
    """
    追加一条挖掘记录。自动写入 logged_at（分钟精度）。
    返回写入的文件路径。
    """
    out = path or LLM_MINING_JSONL
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    row = dict(record)
    row.setdefault("logged_at", _now_minute_iso())
    line = json.dumps(row, ensure_ascii=False, default=str) + "\n"
    with open(out, "a", encoding="utf-8") as f:
        f.write(line)
    return out


def read_recent_llm_mining_records(
    path: Optional[str] = None,
    *,
    limit: int = 40,
) -> List[Dict[str, Any]]:
    """读取 JSONL 末尾若干条，供前端与 API 展示 LLM 审计记录。"""
    out = path or LLM_MINING_JSONL
    if not out or not os.path.isfile(out):
        return []
    rows: List[Dict[str, Any]] = []
    try:
        with open(out, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except Exception:
        return []
    if limit <= 0:
        return rows
    return rows[-limit:]
