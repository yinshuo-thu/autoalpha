from __future__ import annotations

import hashlib
import os
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

AUTOALPHA_DIR = Path(__file__).resolve().parent
DB_PATH = AUTOALPHA_DIR / "autoalpha_lab.db"
PROMPT_DIR = AUTOALPHA_DIR / "inspirations"
MANUAL_PROMPT_DIR = AUTOALPHA_DIR.parent / "manual" / "prompts"

_URL_RE = re.compile(r"^https?://", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_source_type(source_type: Any) -> str:
    value = str(source_type or "manual").strip().lower()
    if value in {"url", "wechat", "manual_url", "human", "prompt", "custom"}:
        return "manual"
    if value in {"arxiv", "openalex", "ssrn", "nber", "scholar"}:
        return "paper"
    if value in {"factor", "factors", "future"}:
        return "manual"
    if value in {"llm", "paper"}:
        return value
    return "manual"


def _configured_context_sources(preferred_source: Optional[str] = None) -> List[str]:
    configured = os.environ.get("AUTOALPHA_INSPIRATION_SOURCES", "paper,llm,manual")
    allowed: List[str] = []
    for raw in configured.split(","):
        source = normalize_source_type(raw.strip())
        if source in {"paper", "llm", "manual"} and source not in allowed:
            allowed.append(source)
    if not allowed:
        allowed = ["paper", "llm", "manual"]
    preferred = normalize_source_type(preferred_source) if preferred_source else ""
    if preferred in allowed:
        return [preferred] + [source for source in allowed if source != preferred]
    return allowed


def _now_iso() -> str:
    return datetime.now().isoformat()


def _ensure_storage() -> None:
    PROMPT_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS inspirations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                kind TEXT NOT NULL,
                title TEXT NOT NULL,
                source TEXT NOT NULL,
                content TEXT NOT NULL,
                summary TEXT NOT NULL,
                tags TEXT NOT NULL,
                relative_path TEXT NOT NULL,
                source_hash TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                source_type TEXT NOT NULL DEFAULT 'manual',
                published_date TEXT NOT NULL DEFAULT '',
                quality_score REAL NOT NULL DEFAULT 0.0,
                usage_count INTEGER NOT NULL DEFAULT 0,
                pass_count INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        # Migrate existing tables that may not have new columns
        for col, definition in [
            ("source_type", "TEXT NOT NULL DEFAULT 'manual'"),
            ("published_date", "TEXT NOT NULL DEFAULT ''"),
            ("quality_score", "REAL NOT NULL DEFAULT 0.0"),
            ("usage_count", "INTEGER NOT NULL DEFAULT 0"),
            ("pass_count", "INTEGER NOT NULL DEFAULT 0"),
        ]:
            try:
                conn.execute(f"ALTER TABLE inspirations ADD COLUMN {col} {definition}")
            except sqlite3.OperationalError:
                pass  # column already exists

        # One-way compatibility migration: historical factor/future notes are
        # now treated as user-provided manual inspirations.
        conn.execute(
            """
            UPDATE inspirations
            SET source_type = 'manual'
            WHERE lower(trim(source_type)) IN ('future', 'factors', 'factor')
            """
        )
        conn.execute(
            """
            UPDATE inspirations
            SET source_type = 'paper'
            WHERE lower(trim(source_type)) IN ('arxiv', 'openalex', 'ssrn', 'nber', 'scholar')
            """
        )
        conn.commit()


def _source_type_where_clause(source: str) -> tuple[str, list[Any]]:
    normalized = normalize_source_type(source)
    if normalized == "manual":
        return "source_type IN ('manual', 'url', 'wechat')", []
    if normalized == "paper":
        return "source_type IN ('paper', 'arxiv', 'openalex', 'ssrn', 'nber', 'scholar')", []
    if normalized == "factors":
        return "source_type IN ('factors', 'future')", []
    return "source_type = ?", [normalized]


def get_effective_score(row: Dict[str, Any]) -> float:
    base = float(row.get("quality_score", 0.0) or 0.0)
    usage = int(row.get("usage_count", 0) or 0)
    passed = int(row.get("pass_count", 0) or 0)
    success_rate = passed / max(usage, 1)
    exploration_bonus = 0.08 if usage == 0 else 0.0
    return base * 0.5 + success_rate * 0.5 + exploration_bonus


def _row_to_dict(cursor: sqlite3.Cursor, row: tuple[Any, ...]) -> Dict[str, Any]:
    payload = {col[0]: row[idx] for idx, col in enumerate(cursor.description)}
    if "source_type" in payload:
        payload["source_type"] = normalize_source_type(payload.get("source_type"))
    payload["effective_score"] = get_effective_score(payload)
    return payload


def _get_conn() -> sqlite3.Connection:
    _ensure_storage()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = _row_to_dict
    return conn


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]+", "-", (value or "").strip())
    cleaned = cleaned.strip("-").lower()
    return cleaned[:48] or "note"


def _trim_text(text: str, limit: int = 6000) -> str:
    text = _WHITESPACE_RE.sub(" ", (text or "").strip())
    return text[:limit]


def _heuristic_summary(text: str) -> str:
    if not text:
        return ""
    snippets = [seg.strip() for seg in re.split(r"[。！？\n]", text) if seg.strip()]
    summary = "；".join(snippets[:2])
    return summary[:180]


def _extract_html_text(html: str) -> str:
    html = re.sub(r"(?is)<script.*?>.*?</script>", " ", html)
    html = re.sub(r"(?is)<style.*?>.*?</style>", " ", html)
    title_match = re.search(r"(?is)<title.*?>(.*?)</title>", html)
    title = _trim_text(re.sub(r"<[^>]+>", " ", title_match.group(1))) if title_match else ""
    body = _trim_text(re.sub(r"<[^>]+>", " ", html), limit=9000)
    return title, body


def _fetch_url_content(url: str) -> tuple[str, str]:
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": "AutoAlpha/1.0"},
            timeout=12,
            verify=False,
        )
        resp.raise_for_status()
    except Exception:
        return url, url

    content_type = resp.headers.get("Content-Type", "")
    if "text/html" in content_type:
        return _extract_html_text(resp.text)
    text = resp.text or url
    return url, _trim_text(text, limit=9000)


def _build_source_hash(relative_path: str, content: str) -> str:
    return hashlib.sha256(f"{relative_path}\n{content}".encode("utf-8")).hexdigest()


def _write_note_file(title: str, source: str, content: str, kind: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{_slugify(title)}.md"
    path = PROMPT_DIR / filename
    payload = (
        f"# {title}\n\n"
        f"- source: {source}\n"
        f"- kind: {kind}\n"
        f"- created_at: {_now_iso()}\n\n"
        f"{content}\n"
    )
    path.write_text(payload, encoding="utf-8")
    return str(path.relative_to(AUTOALPHA_DIR))


def prepare_inspiration(raw_input: str, title: str = "") -> Dict[str, Any]:
    _ensure_storage()
    raw_input = (raw_input or "").strip()
    if not raw_input:
        raise ValueError("Inspiration input is empty.")

    kind = "url" if _URL_RE.match(raw_input) else "prompt"
    source = raw_input
    resolved_title = title.strip()
    content = raw_input

    if kind == "url":
        fetched_title, fetched_content = _fetch_url_content(raw_input)
        if not resolved_title:
            resolved_title = fetched_title[:80]
        content = fetched_content

    if not resolved_title:
        resolved_title = _trim_text(raw_input, limit=48)

    content = _trim_text(content, limit=9000)
    relative_path = _write_note_file(resolved_title, source, content, kind)
    return {
        "kind": kind,
        "title": resolved_title,
        "source": source,
        "content": content,
        "summary": _heuristic_summary(content),
        "tags": "",
        "relative_path": relative_path,
        "source_hash": _build_source_hash(relative_path, content),
        "source_type": "manual",
        "published_date": "",
        "quality_score": 0.0,
        "usage_count": 0,
        "pass_count": 0,
    }


def save_inspiration(record: Dict[str, Any]) -> Dict[str, Any]:
    _ensure_storage()
    payload = dict(record)
    payload["summary"] = _trim_text(payload.get("summary", ""), limit=280)
    payload["tags"] = _trim_text(payload.get("tags", ""), limit=180)
    now = _now_iso()

    with _get_conn() as conn:
        existing = conn.execute(
            "SELECT * FROM inspirations WHERE source_hash = ?",
            (payload["source_hash"],),
        ).fetchone()
        if existing:
            existing["was_new"] = False
            return existing

        conn.execute(
            """
            INSERT INTO inspirations (
                kind, title, source, content, summary, tags,
                relative_path, source_hash, created_at, updated_at, status,
                source_type, published_date, quality_score, usage_count, pass_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload["kind"],
                payload["title"],
                payload["source"],
                payload["content"],
                payload["summary"],
                payload["tags"],
                payload.get("relative_path", ""),
                payload["source_hash"],
                now,
                now,
                "active",
                normalize_source_type(payload.get("source_type", "manual")),
                payload.get("published_date", ""),
                float(payload.get("quality_score", 0.0)),
                int(payload.get("usage_count", 0) or 0),
                int(payload.get("pass_count", 0) or 0),
            ),
        )
        conn.commit()
        created = conn.execute(
            "SELECT * FROM inspirations WHERE source_hash = ?", (payload["source_hash"],)
        ).fetchone()
        created["was_new"] = True
        return created


def record_usage(inspiration_ids: List[int]) -> None:
    ids = [int(item) for item in inspiration_ids if item is not None]
    if not ids:
        return
    placeholders = ",".join("?" for _ in ids)
    with _get_conn() as conn:
        conn.execute(
            f"""
            UPDATE inspirations
            SET usage_count = usage_count + 1,
                updated_at = ?
            WHERE id IN ({placeholders})
            """,
            [_now_iso(), *ids],
        )
        conn.commit()


def record_pass(inspiration_ids: List[int]) -> None:
    ids = [int(item) for item in inspiration_ids if item is not None]
    if not ids:
        return
    placeholders = ",".join("?" for _ in ids)
    with _get_conn() as conn:
        conn.execute(
            f"""
            UPDATE inspirations
            SET pass_count = pass_count + 1,
                updated_at = ?
            WHERE id IN ({placeholders})
            """,
            [_now_iso(), *ids],
        )
        conn.commit()


def _effective_score_sql(alias: str = "inspirations") -> str:
    return (
        f"(({alias}.quality_score * 0.5) + "
        f"((1.0 * {alias}.pass_count / CASE WHEN {alias}.usage_count > 0 THEN {alias}.usage_count ELSE 1 END) * 0.5) + "
        f"(CASE WHEN {alias}.usage_count = 0 THEN 0.08 ELSE 0.0 END))"
    )


def update_inspiration_summary(entry_id: int, summary: str) -> None:
    with _get_conn() as conn:
        conn.execute(
            "UPDATE inspirations SET summary = ?, updated_at = ? WHERE id = ?",
            (_trim_text(summary, limit=280), _now_iso(), entry_id),
        )
        conn.commit()


def toggle_inspiration_status(entry_id: int) -> Dict[str, Any]:
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT status FROM inspirations WHERE id = ?", (entry_id,)
        ).fetchone()
        if not row:
            raise ValueError(f"Inspiration {entry_id} not found")
        new_status = "inactive" if row["status"] == "active" else "active"
        conn.execute(
            "UPDATE inspirations SET status = ?, updated_at = ? WHERE id = ?",
            (new_status, _now_iso(), entry_id),
        )
        conn.commit()
        return conn.execute(
            "SELECT * FROM inspirations WHERE id = ?", (entry_id,)
        ).fetchone()


def delete_inspiration(entry_id: int) -> bool:
    with _get_conn() as conn:
        cursor = conn.execute("DELETE FROM inspirations WHERE id = ?", (entry_id,))
        conn.commit()
        return cursor.rowcount > 0


def _source_type_where_clause(source: str) -> tuple[str, list[Any]]:
    normalized = normalize_source_type(source)
    if normalized == "manual":
        return "source_type IN ('manual', 'url', 'wechat', 'factors', 'future', 'factor')", []
    if normalized == "paper":
        return "source_type IN ('paper', 'arxiv', 'openalex', 'ssrn', 'nber', 'scholar')", []
    return "source_type = ?", [normalized]


def list_recent_inspirations(limit: int = 12) -> List[Dict[str, Any]]:
    with _get_conn() as conn:
        rows = conn.execute(
            """
            SELECT * FROM inspirations
            WHERE status = 'active'
            ORDER BY CASE WHEN source_type IN ('manual', 'url', 'wechat') THEN 0 ELSE 1 END,
                     created_at DESC, id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return rows


def get_recent_inspiration_context_rows(limit: int = 6) -> List[Dict[str, Any]]:
    """Rows used to build the LLM inspiration context."""
    effective = _effective_score_sql("inspirations")
    with _get_conn() as conn:
        rows = conn.execute(
            """
            SELECT * FROM inspirations
            WHERE status = 'active'
            ORDER BY CASE WHEN source_type IN ('manual', 'url', 'wechat') THEN 0 ELSE 1 END,
                     """ + effective + """ DESC,
                     created_at DESC, id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return rows


def _usable_context_row(row: Dict[str, Any]) -> bool:
    source = normalize_source_type(row.get("source_type") or row.get("kind") or "manual")
    if source != "paper":
        return True
    quality = float(row.get("effective_score", get_effective_score(row)) or 0.0)
    text = " ".join(
        str(row.get(key) or "")
        for key in ("title", "summary", "content", "tags")
    ).lower()
    finance_terms = (
        "stock", "return", "trading", "market", "portfolio", "asset", "equity",
        "alpha", "factor", "microstructure", "order flow", "price impact",
        "momentum", "reversal", "liquidity", "volatility",
    )
    signal_terms = (
        "predict", "forecast", "signal", "anomaly", "feature", "factor",
        "cross-sectional", "intraday", "price", "volume", "trade",
    )
    return quality >= 0.55 and any(t in text for t in finance_terms) and any(t in text for t in signal_terms)


def get_balanced_inspiration_context_rows(
    limit: int = 8,
    preferred_source: Optional[str] = None,
    prefer_unused: bool = False,
) -> List[Dict[str, Any]]:
    """Return active inspiration rows balanced across source types."""
    sources = _configured_context_sources(preferred_source)
    effective = _effective_score_sql("inspirations")

    selected: List[Dict[str, Any]] = []
    seen: set[int] = set()
    per_source = max(1, limit // max(len(sources), 1))
    with _get_conn() as conn:
        preferred = normalize_source_type(preferred_source) if preferred_source else ""
        if prefer_unused and preferred:
            where, where_params = _source_type_where_clause(preferred)
            fresh_rows = conn.execute(
                f"""
                SELECT * FROM inspirations
                WHERE status = 'active' AND {where}
                ORDER BY CASE WHEN usage_count = 0 THEN 0 ELSE 1 END,
                         updated_at DESC, created_at DESC, id DESC
                LIMIT ?
                """,
                (*where_params, max(2, per_source)),
            ).fetchall()
            for row in fresh_rows:
                if not _usable_context_row(dict(row)):
                    continue
                rid = int(row["id"])
                if rid in seen:
                    continue
                selected.append(row)
                seen.add(rid)
                if len(selected) >= min(limit, 2):
                    break

        for source in sources:
            where, where_params = _source_type_where_clause(source)
            rows = conn.execute(
                f"""
                SELECT * FROM inspirations
                WHERE status = 'active' AND {where}
                ORDER BY {effective} DESC, created_at DESC, id DESC
                LIMIT ?
                """,
                [*where_params, per_source + 1],
            ).fetchall()
            for row in rows:
                if not _usable_context_row(dict(row)):
                    continue
                rid = int(row["id"])
                if rid in seen:
                    continue
                selected.append(row)
                seen.add(rid)
                if len(selected) >= limit:
                    return selected

        if len(selected) < limit:
            rows = conn.execute(
                """
                SELECT * FROM inspirations
                WHERE status = 'active'
                ORDER BY """ + effective + """ DESC, created_at DESC, id DESC
                LIMIT ?
                """,
                (limit * 2,),
            ).fetchall()
            for row in rows:
                if not _usable_context_row(dict(row)):
                    continue
                rid = int(row["id"])
                if rid in seen:
                    continue
                selected.append(row)
                seen.add(rid)
                if len(selected) >= limit:
                    break
    return selected


def list_inspiration_source_counts() -> List[Dict[str, Any]]:
    with _get_conn() as conn:
        rows = conn.execute(
            """
            SELECT CASE
                       WHEN source_type IN ('manual', 'url', 'wechat', 'factors', 'future', 'factor') THEN 'manual'
                       WHEN source_type IN ('paper', 'arxiv', 'openalex', 'ssrn', 'nber', 'scholar') THEN 'paper'
                       ELSE source_type
                   END AS source_type,
                   COUNT(*) AS count
            FROM inspirations
            WHERE status = 'active'
            GROUP BY CASE
                       WHEN source_type IN ('manual', 'url', 'wechat', 'factors', 'future', 'factor') THEN 'manual'
                       WHEN source_type IN ('paper', 'arxiv', 'openalex', 'ssrn', 'nber', 'scholar') THEN 'paper'
                       ELSE source_type
                     END
            """
        ).fetchall()
    return rows


def list_inspirations_paginated(
    page: int = 1,
    per_page: int = 20,
    source_type: Optional[str] = None,
    search: Optional[str] = None,
    include_inactive: bool = False,
) -> Dict[str, Any]:
    conditions = []
    params: list = []

    if not include_inactive:
        conditions.append("status = 'active'")

    normalized_source = normalize_source_type(source_type) if source_type and source_type != "all" else ""
    if normalized_source:
        where, where_params = _source_type_where_clause(normalized_source)
        conditions.append(where)
        params.extend(where_params)

    if search:
        conditions.append("(title LIKE ? OR summary LIKE ? OR tags LIKE ?)")
        like = f"%{search}%"
        params.extend([like, like, like])

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    offset = (max(1, page) - 1) * per_page

    with _get_conn() as conn:
        total = conn.execute(
            f"SELECT COUNT(*) as cnt FROM inspirations {where}", params
        ).fetchone()["cnt"]
        effective = _effective_score_sql("inspirations")
        rows = conn.execute(
            f"""
            SELECT * FROM inspirations {where}
            ORDER BY CASE WHEN source_type IN ('manual', 'url', 'wechat') THEN 0 ELSE 1 END,
                     {effective} DESC, created_at DESC, id DESC
            LIMIT ? OFFSET ?
            """,
            params + [per_page, offset],
        ).fetchall()

    return {
        "items": rows,
        "total": total,
        "page": page,
        "per_page": per_page,
        "pages": max(1, (total + per_page - 1) // per_page),
    }


def sync_prompt_directory(limit: int = 80) -> Dict[str, Any]:
    _ensure_storage()
    imported = 0
    scanned = 0
    supported = {".md", ".txt", ".url"}

    scan_dirs = [
        MANUAL_PROMPT_DIR,
        AUTOALPHA_DIR.parent / "factors" / "prompts",
        AUTOALPHA_DIR.parent / "fut_feat",
        PROMPT_DIR,
    ]
    seen: set[Path] = set()

    for base_dir in scan_dirs:
        if not base_dir.is_dir():
            continue
        for path in sorted(base_dir.glob("*")):
            if scanned >= limit:
                break
            if not path.is_file() or path.suffix.lower() not in supported:
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            scanned += 1
            try:
                raw = path.read_text(encoding="utf-8").strip()
            except Exception:
                continue
            if not raw:
                continue

            kind = "url" if path.suffix.lower() == ".url" or _URL_RE.match(raw) else "prompt"
            source = raw if kind == "url" else str(path)
            content = raw
            title = path.stem
            if kind == "url":
                fetched_title, fetched_content = _fetch_url_content(raw)
                title = fetched_title[:80] or title
                content = fetched_content

            try:
                relative_path = str(path.relative_to(AUTOALPHA_DIR.parent))
            except Exception:
                relative_path = str(path.relative_to(AUTOALPHA_DIR))
            record = {
                "kind": kind,
                "title": title,
                "source": source,
                "content": _trim_text(content, limit=9000),
                "summary": _heuristic_summary(content),
                "tags": "",
                "relative_path": relative_path,
                "source_hash": _build_source_hash(relative_path, _trim_text(content, limit=9000)),
                "source_type": "manual",
                "published_date": "",
                "quality_score": 0.0,
                "usage_count": 0,
                "pass_count": 0,
            }
            saved = save_inspiration(record)
            if saved.get("was_new"):
                imported += 1

    return {
        "scanned": scanned,
        "imported": imported,
        "prompt_dir": str(MANUAL_PROMPT_DIR),
        "db_path": str(DB_PATH),
        "items": list_recent_inspirations(limit=12),
    }


def compose_inspiration_context(limit: int = 6, max_chars: int = 2400) -> str:
    sync_prompt_directory(limit=80)
    rows = get_balanced_inspiration_context_rows(limit=limit)
    if not rows:
        return ""

    lines: List[str] = []
    total = 0
    for row in rows:
        source = row.get("source") or row.get("relative_path") or ""
        summary = row.get("summary") or _heuristic_summary(row.get("content", ""))
        line = f"- [{row.get('source_type', row.get('kind', 'manual'))}] {row.get('title')}: {summary} (source: {source})"
        total += len(line)
        if total > max_chars:
            break
        lines.append(line)
    return "\n".join(lines)


def compose_inspiration_context_with_sources(
    limit: int = 6,
    max_chars: int = 2400,
    preferred_source: Optional[str] = None,
    prefer_unused: bool = False,
) -> tuple[str, List[Dict[str, Any]]]:
    sync_prompt_directory(limit=80)
    rows = get_balanced_inspiration_context_rows(
        limit=limit,
        preferred_source=preferred_source,
        prefer_unused=prefer_unused,
    )
    if not rows:
        return "", []

    lines: List[str] = []
    used: List[Dict[str, Any]] = []
    total = 0
    for row in rows:
        source = row.get("source") or row.get("relative_path") or ""
        summary = row.get("summary") or _heuristic_summary(row.get("content", ""))
        line = f"- [{row.get('source_type', row.get('kind', 'manual'))}] {row.get('title')}: {summary} (source: {source})"
        total += len(line)
        if total > max_chars:
            break
        lines.append(line)
        used.append(row)
    return "\n".join(lines), used
