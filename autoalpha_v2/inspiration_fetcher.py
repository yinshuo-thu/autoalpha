"""
autoalpha_v2/inspiration_fetcher.py

Multi-source inspiration fetcher.  Supported sources:
  - paper   : query broader scholarly/web indices plus curated quant papers
  - manual  : user-provided URLs / notes, plus local Markdown docs from manual/prompts/
  - llm     : ask the cheap LLM model to brainstorm new alpha directions

The module exposes a lightweight background-thread scheduler so loop.py
can call `start_background_fetcher()` once and forget about it.
"""

from __future__ import annotations

import hashlib
import re
import threading
import time
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

# Broad scholarly search API. OpenAlex is not arXiv-specific and works without
# an API key, so it gives the fetcher a wider outside-world idea surface.
_OPENALEX_API = "https://api.openalex.org/works"
_OPENALEX_QUERIES = [
    "intraday stock return predictability order flow volume reversal",
    "cross sectional stock returns momentum reversal liquidity factor",
    "stock return prediction technical indicators volume price factor",
    "market microstructure price impact order imbalance return prediction",
]

_QUANT_PAPER_KEYWORDS = (
    "stock return", "returns", "cross-section", "cross-sectional", "intraday",
    "high-frequency", "momentum", "reversal", "liquidity", "volume",
    "order flow", "order imbalance", "price impact", "volatility", "factor",
    "anomaly", "predictability", "prediction", "forecast",
)
_QUANT_PAPER_REJECT_KEYWORDS = (
    "cryptocurrency", "bitcoin", "option pricing", "credit risk", "textual",
    "sentiment", "news", "esg", "climate", "macro", "fundamental only",
)

# High-signal papers found through broad web/scholarly search.  These are kept as
# a deterministic floor so the library has useful outside ideas even if a live
# search endpoint is temporarily unavailable.
_CURATED_QUANT_PAPERS: List[Dict[str, str]] = [
    {
        "title": "Intraday Patterns in the Cross-Section of Stock Returns",
        "source": "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1509466",
        "published_date": "2010-05-26",
        "summary": "Half-hour return continuation at daily lags, plus short-run reversal tied to temporary liquidity imbalance.",
        "mechanism": "Test same-clock intraday continuation and sub-hour reversal using lagged 15-minute returns, volume, volatility and liquidity proxies.",
    },
    {
        "title": "How and When are High-Frequency Stock Returns Predictable?",
        "source": "https://www.nber.org/papers/w30366",
        "published_date": "2022-08-01",
        "summary": "Ultra high-frequency stock returns and durations are predictable from recent price, volume and transaction events.",
        "mechanism": "Build event-recency factors from short-horizon returns, trade counts, volume bursts and quote/trade activity.",
    },
    {
        "title": "Intraday Market Return Predictability Culled from the Factor Zoo",
        "source": "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4388560",
        "published_date": "2023-03-14",
        "summary": "Lagged high-frequency cross-sectional factor-zoo returns predict intraday aggregate market returns.",
        "mechanism": "Compress cross-sectional lagged factor returns into regularized intraday predictors and separate continuous from jump-like moves.",
    },
    {
        "title": "Liquidity Risk and Expected Stock Returns",
        "source": "https://www.nber.org/papers/w8462",
        "published_date": "2001-09-01",
        "summary": "Expected returns relate to liquidity sensitivity; liquidity is measured through order-flow-induced reversals.",
        "mechanism": "Use return reversal after volume shocks as an illiquidity/liquidity-risk signal for cross-sectional ranking.",
    },
    {
        "title": "Evaporating Liquidity",
        "source": "https://www.nber.org/papers/w17653",
        "published_date": "2011-12-01",
        "summary": "Short-term reversal strategy returns proxy time-varying liquidity provision returns.",
        "mechanism": "Condition reversal strength on recent volatility/liquidity stress using intraday range, turnover and short-lag return reversal.",
    },
    {
        "title": "Liquidity and Autocorrelations in Individual Stock Returns",
        "source": "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=555968",
        "published_date": "2005-01-12",
        "summary": "Short-run reversals are strongest in high-turnover, low-liquidity stocks after controlling for trading volume.",
        "mechanism": "Rank names by turnover-adjusted illiquidity and recent return autocorrelation to capture transient price pressure.",
    },
    {
        "title": "Persistence or Reversal? The Effects of Abnormal Trading Volume on Stock Returns",
        "source": "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4346340",
        "published_date": "2023-02-02",
        "summary": "Abnormal trading volume predicts short-run persistence and longer-run reversal through volume persistence.",
        "mechanism": "Measure abnormal volume persistence and interact it with recent return direction for drift-versus-reversal timing.",
    },
    {
        "title": "Overnight Returns and the Timing of Trading Volume",
        "source": "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5004991",
        "published_date": "2024-10-30",
        "summary": "U-shaped intraday trading activity predicts higher overnight returns and links to overnight momentum.",
        "mechanism": "Compare open/close volume concentration versus close-only concentration as an overnight continuation signal.",
    },
    {
        "title": "Machine Learning Techniques for Cross-Sectional Equity Returns' Prediction",
        "source": "https://link.springer.com/article/10.1007/s00291-022-00693-w",
        "published_date": "2022-09-28",
        "summary": "Machine learning improves cross-sectional equity return forecasts using lagged stock-level predictors.",
        "mechanism": "Use nonlinear combinations of technical return, volatility, liquidity and volume features for cross-sectional ranking.",
    },
    {
        "title": "Machine Learning Goes Global: Cross-Sectional Return Predictability in International Stock Markets",
        "source": "https://www.sciencedirect.com/science/article/pii/S0165188923001318",
        "published_date": "2023-10-01",
        "summary": "Return predictability across global equities comes largely from momentum, reversal, value and size-style predictors.",
        "mechanism": "Prioritize simple technical factor families such as momentum, reversal and liquidity before complex interactions.",
    },
    {
        "title": "Machine Learning and The Cross-Section of Emerging Market Stock Returns",
        "source": "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4287550",
        "published_date": "2023-03-13",
        "summary": "Nonlinear and interaction-aware models outperform linear models for emerging-market stock return prediction.",
        "mechanism": "Search interactions between recent return, volatility, turnover and liquidity constraints for underreaction signals.",
    },
    {
        "title": "The Momentum Gap and Return Predictability",
        "source": "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2318858",
        "published_date": "2019-02-28",
        "summary": "The gap between winner and loser formation returns predicts future momentum profitability.",
        "mechanism": "Use dispersion between recent winners and losers to gate momentum versus reversal factor exposure.",
    },
]

# ─── Curated manual URL sources ──────────────────────────────────────────────
# User-provided article URLs that should be harvested during fetch cycles.
_DEFAULT_URL_SOURCES: List[str] = []   # populated by add_url_source() or config
_MANUAL_PROMPT_DIR = AUTOALPHA_DIR.parent / "manual" / "prompts"
_LEGACY_FACTOR_PROMPT_DIR = AUTOALPHA_DIR.parent / "factors" / "prompts"
_LEGACY_FUTURE_MD_DIR = AUTOALPHA_DIR.parent / "fut_feat"

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


# ─── Broader paper search / curated quant papers ─────────────────────────────

def _paper_relevance_score(title: str, summary: str) -> tuple[bool, float, str]:
    text = f"{title} {summary}".lower()
    hits = [kw for kw in _QUANT_PAPER_KEYWORDS if kw in text]
    rejects = [kw for kw in _QUANT_PAPER_REJECT_KEYWORDS if kw in text]
    has_return_signal = any(kw in text for kw in ("return", "predict", "forecast", "anomaly", "momentum", "reversal"))
    has_market_micro = any(kw in text for kw in ("stock", "equity", "intraday", "volume", "liquidity", "order", "price"))
    score = min(0.98, 0.25 + 0.07 * len(hits) - 0.12 * len(rejects))
    keep = has_return_signal and has_market_micro and len(hits) >= 3 and score >= 0.62
    reason = f"hits: {', '.join(hits[:8])}" if hits else "no quant-factor keyword hits"
    if rejects:
        reason += f"; rejects: {', '.join(rejects[:4])}"
    return keep, max(0.0, min(1.0, score)), reason


def _quant_paper_record(
    *,
    title: str,
    source: str,
    summary: str,
    mechanism: str = "",
    published_date: str = "",
    tags: str = "paper,external-search,quant-factor",
    quality_score: float = 0.86,
) -> Dict[str, Any]:
    content = (
        f"Title: {title}\n"
        f"Source: {source}\n"
        f"Published: {published_date}\n"
        f"Summary: {summary}\n"
        f"Factor idea: {mechanism or summary}\n"
    )
    source_hash = hashlib.sha256(f"paper:{source}".encode("utf-8")).hexdigest()
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", title)[:48].strip("_").lower() or source_hash[:12]
    return {
        "kind": "url",
        "title": title[:80],
        "source": source,
        "content": _trim_text(content, limit=4000),
        "summary": _trim_text(summary, limit=280),
        "tags": tags,
        "relative_path": f"inspirations/paper_{slug}.md",
        "source_hash": source_hash,
        "source_type": "paper",
        "published_date": published_date,
        "quality_score": quality_score,
    }


def fetch_openalex_quant_papers(query: str, max_results: int = 8) -> List[Dict[str, Any]]:
    """Fetch broader scholarly-paper ideas through OpenAlex and strict keyword gating."""
    try:
        resp = requests.get(
            _OPENALEX_API,
            params={
                "search": query,
                "filter": "from_publication_date:2000-01-01",
                "sort": "relevance_score:desc",
                "per-page": min(max(max_results * 4, max_results), 50),
            },
            timeout=18,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        print(f"[fetcher] OpenAlex query failed: {exc}")
        return []

    records: List[Dict[str, Any]] = []
    for item in data.get("results", []):
        title = _trim_text(item.get("title") or "", limit=140)
        if not title:
            continue
        abstract_index = item.get("abstract_inverted_index") or {}
        abstract_words: list[tuple[int, str]] = []
        for word, positions in abstract_index.items():
            for pos in positions:
                abstract_words.append((int(pos), word))
        abstract = " ".join(word for _, word in sorted(abstract_words)) if abstract_words else ""
        summary = _trim_text(abstract, limit=500) or title
        keep, score, reason = _paper_relevance_score(title, summary)
        if not keep:
            print(f"[fetcher] OpenAlex screened out: {title} ({reason})")
            continue
        source = (
            (item.get("primary_location") or {}).get("landing_page_url")
            or item.get("doi")
            or item.get("id")
            or ""
        )
        if source.startswith("https://doi.org/"):
            pass
        elif source.startswith("10."):
            source = f"https://doi.org/{source}"
        if not source:
            continue
        published_date = str(item.get("publication_date") or "")[:10]
        records.append(_quant_paper_record(
            title=title,
            source=source,
            summary=summary,
            mechanism=f"Broad scholarly search match for: {query}. {reason}",
            published_date=published_date,
            tags="paper,openalex,external-search,quant-factor",
            quality_score=score,
        ))
        if len(records) >= max_results:
            break
    return records


def fetch_curated_quant_papers(limit: int = 12) -> List[Dict[str, Any]]:
    """Return deterministic high-quality quant-factor papers with preserved links."""
    records: List[Dict[str, Any]] = []
    for item in _CURATED_QUANT_PAPERS[:limit]:
        records.append(_quant_paper_record(
            title=item["title"],
            source=item["source"],
            summary=item["summary"],
            mechanism=item["mechanism"],
            published_date=item.get("published_date", ""),
            tags="paper,curated,external-search,quant-factor,ohlcv",
            quality_score=0.9,
        ))
    return records


def fetch_quant_papers(max_results: int = 12) -> List[Dict[str, Any]]:
    """Combine broad live search with curated high-signal papers."""
    records: List[Dict[str, Any]] = []
    seen_sources: set[str] = set()
    for query in _OPENALEX_QUERIES:
        for rec in fetch_openalex_quant_papers(query, max_results=4):
            source = str(rec.get("source") or "")
            if source in seen_sources:
                continue
            seen_sources.add(source)
            records.append(rec)
            if len(records) >= max_results:
                return records
    for rec in fetch_curated_quant_papers(limit=max_results):
        source = str(rec.get("source") or "")
        if source in seen_sources:
            continue
        seen_sources.add(source)
        records.append(rec)
        if len(records) >= max_results:
            break
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
            "tags": "manual,url",
            "relative_path": relative_path,
            "source_hash": _build_source_hash(relative_path, content),
            "source_type": "manual",
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
            "published_date": datetime.now().strftime("%Y-%m-%d"),
            "quality_score": 0.0,
        })
    return records


# ─── Local manual prompt notes ───────────────────────────────────────────────

def _iter_manual_prompt_paths() -> List[Path]:
    seen: set[Path] = set()
    ordered: List[Path] = []
    for base_dir in (_MANUAL_PROMPT_DIR, _LEGACY_FACTOR_PROMPT_DIR, _LEGACY_FUTURE_MD_DIR):
        if not base_dir.is_dir():
            continue
        for path in sorted(base_dir.glob("*.md")):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            ordered.append(path)
    return ordered


def fetch_manual_file_inspirations(max_files: int = 12) -> List[Dict[str, Any]]:
    """Read local manual Markdown prompt notes and convert them to inspiration records."""
    paths = _iter_manual_prompt_paths()
    if not paths:
        return []

    records: List[Dict[str, Any]] = []
    for path in paths[:max_files]:
        try:
            raw = path.read_text(encoding="utf-8").strip()
        except Exception as exc:
            print(f"[fetcher] Manual prompt read failed ({path}): {exc}")
            continue
        if not raw:
            continue

        title = raw.splitlines()[0].lstrip("# ").strip() if raw.splitlines() else path.stem
        title = title or path.stem
        content = _trim_text(raw, limit=9000)
        source_hash = hashlib.sha256(f"manual:{path.name}:{content}".encode("utf-8")).hexdigest()
        records.append({
            "kind": "prompt",
            "title": f"Manual prompt: {title[:64]}",
            "source": str(path),
            "content": content,
            "summary": _heuristic_summary(content),
            "tags": "manual,file-prompt,local-markdown",
            "relative_path": f"inspirations/manual_{path.stem}.md",
            "source_hash": source_hash,
            "source_type": "manual",
            "published_date": datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d"),
            "quality_score": 0.6,
        })
    return records


# ─── Full fetch cycle ─────────────────────────────────────────────────────────

def run_fetch_cycle(
    extra_urls: Optional[List[str]] = None,
    llm_ideas: int = 6,
    paper_results: int = 12,
    manual_files: int = 12,
) -> Dict[str, Any]:
    """
    Run one full inspiration-fetch cycle.
    Returns a summary dict of what was added.
    """
    added = {"paper": 0, "manual": 0, "llm": 0, "skipped": 0, "screened_out": 0}

    # 1. Broader external paper search plus curated quant-factor papers
    for rec in fetch_quant_papers(max_results=paper_results):
        result = save_inspiration(rec)
        if result.get("was_new"):
            added["paper"] += 1
        else:
            added["skipped"] += 1

    # 2. Manual URL sources
    all_urls = list(extra_urls or [])
    with _url_sources_lock:
        all_urls += list(_url_sources)
    for url in all_urls:
        rec = fetch_url_inspiration(url)
        if rec:
            result = save_inspiration(rec)
            if result.get("was_new"):
                added["manual"] += 1
            else:
                added["skipped"] += 1

    # 3. Local manual Markdown notes
    for rec in fetch_manual_file_inspirations(max_files=manual_files):
        result = save_inspiration(rec)
        if result.get("was_new"):
            added["manual"] += 1
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
        f"[fetcher] Cycle done — paper={added['paper']} manual={added['manual']} "
        f"llm={added['llm']} skipped={added['skipped']}"
    )
    return added


# ─── Background scheduler ────────────────────────────────────────────────────

_bg_thread: Optional[threading.Thread] = None
_bg_stop = threading.Event()


def start_background_fetcher(
    interval_seconds: int = 1800,
    llm_ideas: int = 6,
    paper_results: int = 12,
    manual_files: int = 12,
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
                run_fetch_cycle(llm_ideas=llm_ideas, paper_results=paper_results, manual_files=manual_files)
            except Exception as exc:
                print(f"[fetcher-bg] Initial cycle error: {exc}")
        while not _bg_stop.wait(timeout=interval_seconds):
            try:
                run_fetch_cycle(llm_ideas=llm_ideas, paper_results=paper_results, manual_files=manual_files)
            except Exception as exc:
                print(f"[fetcher-bg] Cycle error: {exc}")

    _bg_thread = threading.Thread(target=_worker, name="inspiration-fetcher", daemon=True)
    _bg_thread.start()
    print(f"[fetcher] Background fetcher started (interval={interval_seconds}s)")
    return _bg_thread


def stop_background_fetcher() -> None:
    _bg_stop.set()
