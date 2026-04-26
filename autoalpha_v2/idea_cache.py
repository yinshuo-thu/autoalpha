"""
autoalpha_v2/idea_cache.py

SQLite-backed pre-generation cache for LLM-created alpha ideas.

Why this exists:
  LLM idea generation is the slowest step (~10-30s per idea).  By pre-generating
  ideas in the background while the current batch evaluates, we can keep all
  CPU cores busy without blocking on LLM calls during the hot path.

  A configurable concurrency limit (default 2) prevents hammering the API.

Usage:
    from autoalpha_v2.idea_cache import IdeaCache
    cache = IdeaCache()
    cache.start_fill(n=10, parents=parents, guidance=guidance)
    idea = cache.pop()   # None if cache is still empty
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

AUTOALPHA_DIR = Path(__file__).resolve().parent
DB_PATH = AUTOALPHA_DIR / "autoalpha_lab.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS idea_cache (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    formula     TEXT NOT NULL,
    thought     TEXT NOT NULL DEFAULT '',
    postprocess TEXT NOT NULL DEFAULT 'rank_clip',
    lookback    INTEGER NOT NULL DEFAULT 20,
    archetype   TEXT NOT NULL DEFAULT '',
    generation_mode TEXT NOT NULL DEFAULT '',
    target_source TEXT NOT NULL DEFAULT '',
    prompt_version TEXT NOT NULL DEFAULT '',
    outcome     TEXT DEFAULT NULL,
    meta_json   TEXT NOT NULL DEFAULT '{}',
    created_at  TEXT NOT NULL,
    consumed    INTEGER NOT NULL DEFAULT 0
)
"""


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute(_SCHEMA)
    for col, definition in [
        ("generation_mode", "TEXT NOT NULL DEFAULT ''"),
        ("target_source", "TEXT NOT NULL DEFAULT ''"),
        ("prompt_version", "TEXT NOT NULL DEFAULT ''"),
        ("outcome", "TEXT DEFAULT NULL"),
    ]:
        try:
            conn.execute(f"ALTER TABLE idea_cache ADD COLUMN {col} {definition}")
        except sqlite3.OperationalError:
            pass
    conn.commit()
    return conn


# Module-level connection pool (one per thread via threading.local)
_local = threading.local()


def _conn() -> sqlite3.Connection:
    if not hasattr(_local, "conn") or _local.conn is None:
        _local.conn = _get_conn()
    try:
        _local.conn.execute("SELECT 1")
    except sqlite3.ProgrammingError:
        _local.conn = _get_conn()
    return _local.conn


class IdeaCache:
    """
    Thread-safe cache for pre-generated alpha ideas.

    Typical usage inside loop.py:
        cache = IdeaCache()
        # kick off background fill while previous batch evaluates
        cache.start_fill(n=10, parents=parents, guidance=guidance)
        ...
        # in pipeline, pop() instead of calling generate_idea directly
        idea = cache.pop()
    """

    def __init__(self, max_size: int = 30, concurrency: int = 1):
        self.max_size = max_size
        self.concurrency = concurrency
        self._fill_lock = threading.Lock()
        self._fill_thread: Optional[threading.Thread] = None
        _get_conn()  # ensure table exists

    # ── public API ────────────────────────────────────────────────────────────

    def size(self) -> int:
        """Number of unconsumed ideas in cache."""
        c = _conn()
        row = c.execute(
            "SELECT COUNT(*) AS cnt FROM idea_cache WHERE consumed = 0"
        ).fetchone()
        return row["cnt"] if row else 0

    def pop(self) -> Optional[Dict[str, Any]]:
        """
        Remove and return the oldest unconsumed idea, or None if cache is empty.
        """
        c = _conn()
        c.execute("BEGIN IMMEDIATE")
        try:
            row = c.execute(
                "SELECT * FROM idea_cache WHERE consumed = 0 ORDER BY id ASC LIMIT 1"
            ).fetchone()
            if row is None:
                c.commit()
                return None
            cur = c.execute(
                "UPDATE idea_cache SET consumed = 1 WHERE id = ? AND consumed = 0",
                (row["id"],),
            )
            if int(cur.rowcount or 0) <= 0:
                c.commit()
                return None
            c.commit()
        except Exception:
            c.rollback()
            raise
        try:
            meta = json.loads(row["meta_json"] or "{}")
        except Exception:
            meta = {}
        return {
            **meta,
            "idea_cache_id": int(row["id"]),
            "formula": row["formula"],
            "thought_process": row["thought"],
            "postprocess": row["postprocess"],
            "lookback_days": row["lookback"],
            "archetype": row["archetype"],
            "generation_mode": row["generation_mode"] or meta.get("generation_mode", ""),
            "target_source": row["target_source"] or meta.get("target_source", ""),
            "prompt_version": row["prompt_version"] or meta.get("prompt_version", ""),
        }

    def push(self, idea: Dict[str, Any]) -> bool:
        """Insert a pre-generated idea into the cache. Returns False if formula is a duplicate."""
        formula = idea.get("formula", "")
        if not formula:
            return False
        c = _conn()
        existing = c.execute(
            "SELECT id FROM idea_cache WHERE formula = ? AND consumed = 0 LIMIT 1", (formula,)
        ).fetchone()
        if existing:
            return False
        meta = {k: v for k, v in idea.items()
                if k not in ("formula", "thought_process", "postprocess", "lookback_days", "archetype")}
        c.execute(
            """
            INSERT INTO idea_cache (
                formula, thought, postprocess, lookback, archetype,
                generation_mode, target_source, prompt_version,
                meta_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                formula,
                idea.get("thought_process", ""),
                idea.get("postprocess", "rank_clip"),
                int(idea.get("lookback_days", 20)),
                idea.get("archetype", ""),
                idea.get("generation_mode", ""),
                idea.get("target_source", idea.get("inspiration_source_type", "")),
                idea.get("prompt_version", ""),
                json.dumps(meta),
                datetime.now().isoformat(),
            ),
        )
        c.commit()
        return True

    def register_generated_idea(self, idea: Dict[str, Any], consumed: bool = True) -> Optional[int]:
        """
        Persist a generated idea so later pipeline outcomes can be written back,
        even when the idea did not originate from the prefill cache.
        """
        formula = idea.get("formula", "")
        if not formula:
            return None
        c = _conn()
        meta = {
            k: v for k, v in idea.items()
            if k not in ("formula", "thought_process", "postprocess", "lookback_days", "archetype", "idea_cache_id")
        }
        cur = c.execute(
            """
            INSERT INTO idea_cache (
                formula, thought, postprocess, lookback, archetype,
                generation_mode, target_source, prompt_version,
                outcome, meta_json, created_at, consumed
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                formula,
                idea.get("thought_process", ""),
                idea.get("postprocess", "rank_clip"),
                int(idea.get("lookback_days", 20)),
                idea.get("archetype", ""),
                idea.get("generation_mode", ""),
                idea.get("target_source", idea.get("inspiration_source_type", "")),
                idea.get("prompt_version", ""),
                None,
                json.dumps(meta),
                datetime.now().isoformat(),
                1 if consumed else 0,
            ),
        )
        c.commit()
        return int(cur.lastrowid)

    def record_outcome(self, idea_cache_id: int | None, outcome: str) -> None:
        if not idea_cache_id or not outcome:
            return
        c = _conn()
        c.execute(
            "UPDATE idea_cache SET outcome = ?, consumed = 1 WHERE id = ?",
            (str(outcome), int(idea_cache_id)),
        )
        c.commit()

    def recent_archetype_outcomes(self, limit: int = 80) -> Dict[str, Dict[str, int]]:
        c = _conn()
        rows = c.execute(
            """
            SELECT archetype, outcome
            FROM idea_cache
            WHERE consumed = 1
              AND archetype != ''
              AND outcome IS NOT NULL
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        summary: Dict[str, Dict[str, int]] = {}
        for row in rows:
            archetype = str(row["archetype"] or "").strip()
            outcome = str(row["outcome"] or "").strip()
            if not archetype or not outcome:
                continue
            bucket = summary.setdefault(archetype, {})
            bucket[outcome] = bucket.get(outcome, 0) + 1
        return summary

    def clear_stale(self, prompt_version: str) -> int:
        """Consume pending ideas generated under an older prompt/version contract."""
        prompt_version = str(prompt_version or "").strip()
        if not prompt_version:
            return 0
        c = _conn()
        cur = c.execute(
            """
            UPDATE idea_cache
            SET consumed = 1
            WHERE consumed = 0
              AND (prompt_version = '' OR prompt_version IS NULL OR prompt_version != ?)
            """,
            (prompt_version,),
        )
        c.commit()
        return cur.rowcount

    def clear(self) -> int:
        """Mark all unconsumed ideas as consumed.  Returns count cleared."""
        c = _conn()
        cur = c.execute("UPDATE idea_cache SET consumed = 1 WHERE consumed = 0")
        c.commit()
        return cur.rowcount

    def prune_old(self, keep: int = 200) -> int:
        """Delete consumed entries beyond the last `keep` rows to bound DB size."""
        c = _conn()
        cur = c.execute(
            """
            DELETE FROM idea_cache WHERE consumed = 1
            AND id NOT IN (SELECT id FROM idea_cache WHERE consumed = 1 ORDER BY id DESC LIMIT ?)
            """,
            (keep,),
        )
        c.commit()
        return cur.rowcount

    def status(self) -> Dict[str, Any]:
        c = _conn()
        total = c.execute("SELECT COUNT(*) AS n FROM idea_cache").fetchone()["n"]
        pending = c.execute(
            "SELECT COUNT(*) AS n FROM idea_cache WHERE consumed = 0"
        ).fetchone()["n"]
        consumed = total - pending
        return {
            "pending": pending,
            "consumed": consumed,
            "total": total,
            "fill_running": self._fill_thread is not None and self._fill_thread.is_alive(),
        }

    def join_fill(self) -> None:
        """Block until any running background fill completes."""
        with self._fill_lock:
            t = self._fill_thread
        if t is not None and t.is_alive():
            t.join()

    # ── background fill ───────────────────────────────────────────────────────

    def start_fill(
        self,
        n: int,
        parents: Optional[List[Dict]] = None,
        guidance: Optional[Dict] = None,
        idea_index_offset: int = 0,
    ) -> None:
        """
        Spawn a background thread that generates `n` ideas and pushes them to
        cache.  Respects self.concurrency as the ThreadPoolExecutor limit.
        Skips if a fill is already running or cache already has enough ideas.
        """
        with self._fill_lock:
            if self._fill_thread is not None and self._fill_thread.is_alive():
                return
            current = self.size()
            if current >= self.max_size:
                return
            actual_n = min(n, self.max_size - current)

        def _fill_worker():
            print(f"[idea_cache] Filling {actual_n} ideas (concurrency={self.concurrency})…")
            try:
                from autoalpha_v2.llm_client import generate_idea
            except ImportError:
                print("[idea_cache] Cannot import generate_idea")
                return

            def _gen_one(idx: int) -> Optional[Dict[str, Any]]:
                try:
                    return generate_idea(
                        parents=parents,
                        idea_index=idea_index_offset + idx,
                        total_ideas=actual_n,
                    )
                except Exception as exc:
                    print(f"[idea_cache] generate_idea({idx}) failed: {exc}")
                    return None

            with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
                futures = {pool.submit(_gen_one, i): i for i in range(actual_n)}
                for fut in as_completed(futures):
                    idea = fut.result()
                    if idea and idea.get("formula"):
                        if self.push(idea):
                            print(f"[idea_cache] Cached idea: {idea['formula'][:60]}…")
                        else:
                            print(f"[idea_cache] Skipped duplicate: {idea['formula'][:60]}…")

            self.prune_old()
            print(f"[idea_cache] Fill done — pending={self.size()}")

        t = threading.Thread(target=_fill_worker, name="idea-cache-fill", daemon=True)
        with self._fill_lock:
            self._fill_thread = t
        t.start()

    def fill_sync(
        self,
        n: int,
        parents: Optional[List[Dict]] = None,
        guidance: Optional[Dict] = None,
        idea_index_offset: int = 0,
    ) -> int:
        """
        Synchronous fill (blocks until done).  Used in tests / manual calls.
        Returns number of ideas added.
        """
        from autoalpha_v2.llm_client import generate_idea

        added = 0

        def _gen_one(idx: int) -> Optional[Dict[str, Any]]:
            try:
                return generate_idea(
                    parents=parents,
                    idea_index=idea_index_offset + idx,
                    total_ideas=n,
                )
            except Exception as exc:
                print(f"[idea_cache] generate_idea({idx}) failed: {exc}")
                return None

        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            futures = [pool.submit(_gen_one, i) for i in range(n)]
            for fut in as_completed(futures):
                idea = fut.result()
                if idea and idea.get("formula"):
                    self.push(idea)
                    added += 1
        self.prune_old()
        return added


# Module-level singleton — importers can share the same cache instance
_default_cache: Optional[IdeaCache] = None
_cache_lock = threading.Lock()


def get_default_cache(max_size: int = 30, concurrency: int = 1) -> IdeaCache:
    global _default_cache
    with _cache_lock:
        if _default_cache is None:
            _default_cache = IdeaCache(max_size=max_size, concurrency=concurrency)
    return _default_cache
