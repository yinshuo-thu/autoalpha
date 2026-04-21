"""
autoalpha_v1/idea_cache.py

SQLite-backed pre-generation cache for LLM-created alpha ideas.

Why this exists:
  LLM idea generation is the slowest step (~10-30s per idea).  By pre-generating
  ideas in the background while the current batch evaluates, we can keep all
  CPU cores busy without blocking on LLM calls during the hot path.

  A configurable concurrency limit (default 3) prevents hammering the API.

Usage:
    from autoalpha_v1.idea_cache import IdeaCache
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
    meta_json   TEXT NOT NULL DEFAULT '{}',
    created_at  TEXT NOT NULL,
    consumed    INTEGER NOT NULL DEFAULT 0
)
"""


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute(_SCHEMA)
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

    def __init__(self, max_size: int = 30, concurrency: int = 3):
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
        row = c.execute(
            "SELECT * FROM idea_cache WHERE consumed = 0 ORDER BY id ASC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        c.execute(
            "UPDATE idea_cache SET consumed = 1 WHERE id = ?", (row["id"],)
        )
        c.commit()
        try:
            meta = json.loads(row["meta_json"] or "{}")
        except Exception:
            meta = {}
        return {
            "formula": row["formula"],
            "thought_process": row["thought"],
            "postprocess": row["postprocess"],
            "lookback_days": row["lookback"],
            "archetype": row["archetype"],
            **meta,
        }

    def push(self, idea: Dict[str, Any]) -> None:
        """Insert a pre-generated idea into the cache."""
        c = _conn()
        meta = {k: v for k, v in idea.items()
                if k not in ("formula", "thought_process", "postprocess", "lookback_days", "archetype")}
        c.execute(
            """
            INSERT INTO idea_cache (formula, thought, postprocess, lookback, archetype, meta_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                idea.get("formula", ""),
                idea.get("thought_process", ""),
                idea.get("postprocess", "rank_clip"),
                int(idea.get("lookback_days", 20)),
                idea.get("archetype", ""),
                json.dumps(meta),
                datetime.now().isoformat(),
            ),
        )
        c.commit()

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
                from autoalpha_v1.llm_client import generate_idea
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
                        self.push(idea)
                        print(f"[idea_cache] Cached idea: {idea['formula'][:60]}…")

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
        from autoalpha_v1.llm_client import generate_idea

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


def get_default_cache(max_size: int = 30, concurrency: int = 3) -> IdeaCache:
    global _default_cache
    with _cache_lock:
        if _default_cache is None:
            _default_cache = IdeaCache(max_size=max_size, concurrency=concurrency)
    return _default_cache
