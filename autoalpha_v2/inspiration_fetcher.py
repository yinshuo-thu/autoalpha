"""
autoalpha_v2/inspiration_fetcher.py

Multi-source inspiration fetcher.  Supported sources:
  - arxiv   : query the public ArXiv API for quant-finance papers
  - url     : fetch any HTTP URL (WeChat articles, blog posts, …)
  - llm     : ask the cheap LLM model to brainstorm new alpha directions
  - future  : distill local futures-factor Markdown notes from fut_feat/

The module exposes a lightweight background-thread scheduler so loop.py
can call `start_background_fetcher()` once and forget about it.
"""

from __future__ import annotations

import hashlib
import re
import threading
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from autoalpha_v2.inspiration_db import (
    AUTOALPHA_DIR,
    _heuristic_summary,
    _trim_text,
    save_inspiration,
)

# ─── ArXiv query template ────────────────────────────────────────────────────
_ARXIV_API = "https://export.arxiv.org/api/query"
_ARXIV_QUERIES = [
    "intraday alpha factor finance microstructure",
    "quantitative trading factor momentum reversal",
    "high frequency price volume signal machine learning",
    "cross-sectional stock return prediction neural",
    "order flow imbalance price impact short-term",
    "intrabar range volatility breakout signal",
    "trade count participation signal alpha",
]

# ─── Curated URL sources ─────────────────────────────────────────────────────
# WeChat / blog articles the user may want to regularly harvest
_DEFAULT_URL_SOURCES: List[str] = []   # populated by add_url_source() or config
_FUTURE_MD_DIR = AUTOALPHA_DIR.parent / "fut_feat"

# ─── LLM-generated brainstorm prompt ─────────────────────────────────────────
_LLM_BRAINSTORM_SYSTEM = """\
You are a quantitative researcher.  Your job is to brainstorm short alpha-factor
ideas that are novel and implementable using 15-minute bar data (OHLCV + vwap +
trade_count + trade_amount).  Each idea should capture a specific, named market
micro-mechanism (e.g. exhaustion reversal, participation divergence, range
compression, etc.).
"""

_LLM_BRAINSTORM_USER = """\
Generate {n} brief alpha-factor research inspirations.  Each entry must be a
JSON object on its own line with keys:
  title        – one-line name (≤ 60 chars)
  mechanism    – 1-2 sentences describing the market mechanism
  key_fields   – comma-separated list of relevant data fields
  time_horizon – "intraday" | "short-term" | "medium-term"

Return ONLY the JSON lines, no extra commentary.
"""

# ─────────────────────────────────────────────────────────────────────────────

_url_sources: List[str] = list(_DEFAULT_URL_SOURCES)
_url_sources_lock = threading.Lock()


def add_url_source(url: str) -> None:
    with _url_sources_lock:
        if url not in _url_sources:
            _url_sources.append(url)


# ─── ArXiv fetch ─────────────────────────────────────────────────────────────

def _arxiv_entry_to_record(entry: ET.Element, ns: str) -> Optional[Dict[str, Any]]:
    def tag(name: str) -> Optional[ET.Element]:
        return entry.find(f"{ns}{name}")

    id_el = tag("id")
    title_el = tag("title")
    summary_el = tag("summary")
    published_el = tag("published")

    if title_el is None or id_el is None:
        return None

    arxiv_url = (id_el.text or "").strip()
    arxiv_id = arxiv_url.split("/abs/")[-1] if "/abs/" in arxiv_url else arxiv_url
    title = _trim_text((title_el.text or "").replace("\n", " "), limit=120)
    abstract = _trim_text((summary_el.text or "").replace("\n", " "), limit=600) if summary_el else ""
    published = (published_el.text or "")[:10] if published_el else ""

    content = f"Title: {title}\nArXiv ID: {arxiv_id}\nPublished: {published}\nAbstract: {abstract}"
    source_hash = hashlib.sha256(f"arxiv:{arxiv_id}".encode()).hexdigest()

    return {
        "kind": "url",
        "title": title[:80],
        "source": arxiv_url,
        "content": content,
        "summary": abstract[:280],
        "tags": "arxiv,quant-finance",
        "relative_path": f"inspirations/arxiv_{arxiv_id.replace('/', '_')}.md",
        "source_hash": source_hash,
        "source_type": "arxiv",
        "arxiv_id": arxiv_id,
        "published_date": published,
        "quality_score": 0.0,
    }


def fetch_arxiv(query: str, max_results: int = 8) -> List[Dict[str, Any]]:
    try:
        resp = requests.get(
            _ARXIV_API,
            params={
                "search_query": f"all:{query}",
                "max_results": max_results,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            },
            timeout=20,
        )
        resp.raise_for_status()
    except Exception as exc:
        print(f"[fetcher] ArXiv query failed: {exc}")
        return []

    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError as exc:
        print(f"[fetcher] ArXiv XML parse error: {exc}")
        return []

    ns_match = re.match(r"\{[^}]+\}", root.tag)
    ns = ns_match.group(0) if ns_match else ""
    records = []
    for entry in root.findall(f"{ns}entry"):
        rec = _arxiv_entry_to_record(entry, ns)
        if rec:
            records.append(rec)
    return records


# ─── URL fetch ────────────────────────────────────────────────────────────────

def fetch_url_inspiration(url: str) -> Optional[Dict[str, Any]]:
    """Fetch a single URL and convert to an inspiration record."""
    from autoalpha_v2.inspiration_db import _fetch_url_content, _build_source_hash, PROMPT_DIR
    try:
        title, content = _fetch_url_content(url)
        if not content or content == url:
            return None
        content = _trim_text(content, limit=9000)
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", url[8:])[:40]
        relative_path = f"inspirations/url_{slug}.md"
        return {
            "kind": "url",
            "title": (title or url)[:80],
            "source": url,
            "content": content,
            "summary": _heuristic_summary(content),
            "tags": "url,wechat",
            "relative_path": relative_path,
            "source_hash": _build_source_hash(relative_path, content),
            "source_type": "wechat" if "mp.weixin.qq.com" in url else "url",
            "arxiv_id": "",
            "published_date": "",
            "quality_score": 0.0,
        }
    except Exception as exc:
        print(f"[fetcher] URL fetch failed ({url}): {exc}")
        return None


# ─── LLM brainstorm ───────────────────────────────────────────────────────────

def generate_llm_inspirations(n: int = 6) -> List[Dict[str, Any]]:
    """Ask the cheap model to brainstorm n alpha ideas; return inspiration records."""
    import json as _json
    try:
        from autoalpha_v2.llm_client import _call_cheap_model
    except ImportError:
        return []

    user_msg = _LLM_BRAINSTORM_USER.format(n=n)
    try:
        raw = _call_cheap_model(system=_LLM_BRAINSTORM_SYSTEM, user=user_msg)
    except Exception as exc:
        print(f"[fetcher] LLM brainstorm failed: {exc}")
        return []

    records = []
    for line in (raw or "").splitlines():
        line = line.strip().lstrip("- ")
        if not line.startswith("{"):
            continue
        try:
            obj = _json.loads(line)
        except _json.JSONDecodeError:
            continue
        title = str(obj.get("title", "LLM idea"))[:80]
        mechanism = str(obj.get("mechanism", ""))
        key_fields = str(obj.get("key_fields", ""))
        time_horizon = str(obj.get("time_horizon", "intraday"))
        content = f"Mechanism: {mechanism}\nKey fields: {key_fields}\nTime horizon: {time_horizon}"
        source_hash = hashlib.sha256(f"llm:{title}:{mechanism}".encode()).hexdigest()
        records.append({
            "kind": "prompt",
            "title": title,
            "source": "llm-brainstorm",
            "content": content,
            "summary": mechanism[:280],
            "tags": f"llm,{time_horizon}",
            "relative_path": f"inspirations/llm_{source_hash[:12]}.md",
            "source_hash": source_hash,
            "source_type": "llm",
            "arxiv_id": "",
            "published_date": datetime.now().strftime("%Y-%m-%d"),
            "quality_score": 0.0,
        })
    return records


# ─── Local futures-factor notes ──────────────────────────────────────────────

def fetch_future_factor_inspirations(max_files: int = 10) -> List[Dict[str, Any]]:
    """Read local fut_feat Markdown notes and convert them to inspiration records."""
    if not _FUTURE_MD_DIR.is_dir():
        return []

    records: List[Dict[str, Any]] = []
    for path in sorted(_FUTURE_MD_DIR.glob("*.md"))[:max_files]:
        try:
            raw = path.read_text(encoding="utf-8").strip()
        except Exception as exc:
            print(f"[fetcher] Future-factor note read failed ({path}): {exc}")
            continue
        if not raw:
            continue

        title = raw.splitlines()[0].lstrip("# ").strip() if raw.splitlines() else path.stem
        title = title or path.stem
        content = _trim_text(raw, limit=9000)
        source_hash = hashlib.sha256(f"future:{path.name}:{content}".encode("utf-8")).hexdigest()
        records.append({
            "kind": "prompt",
            "title": f"Future factor: {title[:64]}",
            "source": str(path),
            "content": content,
            "summary": _heuristic_summary(content),
            "tags": "future,futures-factor,local-markdown",
            "relative_path": f"inspirations/future_{path.stem}.md",
            "source_hash": source_hash,
            "source_type": "future",
            "arxiv_id": "",
            "published_date": datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d"),
            "quality_score": 0.6,
        })
    return records


# ─── Full fetch cycle ─────────────────────────────────────────────────────────

def run_fetch_cycle(
    arxiv_queries: Optional[List[str]] = None,
    extra_urls: Optional[List[str]] = None,
    llm_ideas: int = 6,
    arxiv_per_query: int = 5,
    future_files: int = 10,
) -> Dict[str, Any]:
    """
    Run one full inspiration-fetch cycle.
    Returns a summary dict of what was added.
    """
    queries = arxiv_queries or _ARXIV_QUERIES
    added = {"arxiv": 0, "url": 0, "llm": 0, "future": 0, "skipped": 0}

    # 1. ArXiv — rotate through queries
    query = queries[int(time.time()) % len(queries)]
    print(f"[fetcher] ArXiv query: {query!r}")
    for rec in fetch_arxiv(query, max_results=arxiv_per_query):
        result = save_inspiration(rec)
        if result.get("was_new"):
            added["arxiv"] += 1
        else:
            added["skipped"] += 1

    # 2. URL sources (WeChat articles etc.)
    all_urls = list(extra_urls or [])
    with _url_sources_lock:
        all_urls += list(_url_sources)
    for url in all_urls:
        rec = fetch_url_inspiration(url)
        if rec:
            result = save_inspiration(rec)
            if result.get("was_new"):
                added["url"] += 1
            else:
                added["skipped"] += 1

    # 3. Local futures-factor Markdown notes
    for rec in fetch_future_factor_inspirations(max_files=future_files):
        result = save_inspiration(rec)
        if result.get("was_new"):
            added["future"] += 1
        else:
            added["skipped"] += 1

    # 4. LLM brainstorm
    if llm_ideas > 0:
        for rec in generate_llm_inspirations(n=llm_ideas):
            result = save_inspiration(rec)
            if result.get("was_new"):
                added["llm"] += 1
            else:
                added["skipped"] += 1

    print(
        f"[fetcher] Cycle done — arxiv={added['arxiv']} url={added['url']} "
        f"llm={added['llm']} future={added['future']} skipped={added['skipped']}"
    )
    return added


# ─── Background scheduler ────────────────────────────────────────────────────

_bg_thread: Optional[threading.Thread] = None
_bg_stop = threading.Event()


def start_background_fetcher(
    interval_seconds: int = 1800,
    llm_ideas: int = 6,
    arxiv_per_query: int = 5,
    run_immediately: bool = True,
) -> threading.Thread:
    """
    Start a background daemon thread that calls run_fetch_cycle() every
    `interval_seconds`.  Safe to call multiple times — only one thread runs.
    """
    global _bg_thread

    if _bg_thread is not None and _bg_thread.is_alive():
        return _bg_thread

    _bg_stop.clear()

    def _worker():
        if run_immediately:
            try:
                run_fetch_cycle(llm_ideas=llm_ideas, arxiv_per_query=arxiv_per_query)
            except Exception as exc:
                print(f"[fetcher-bg] Initial cycle error: {exc}")
        while not _bg_stop.wait(timeout=interval_seconds):
            try:
                run_fetch_cycle(llm_ideas=llm_ideas, arxiv_per_query=arxiv_per_query)
            except Exception as exc:
                print(f"[fetcher-bg] Cycle error: {exc}")

    _bg_thread = threading.Thread(target=_worker, name="inspiration-fetcher", daemon=True)
    _bg_thread.start()
    print(f"[fetcher] Background fetcher started (interval={interval_seconds}s)")
    return _bg_thread


def stop_background_fetcher() -> None:
    _bg_stop.set()
