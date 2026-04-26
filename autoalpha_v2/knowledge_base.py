"""
autoalpha_v2/knowledge_base.py

Persistent JSON knowledge base for all autoalpha experiments.

Design lineage:
  - Ratchet loop (karpathy/autoresearch): every tested factor is logged; top
    passing factors become parents for the next LLM round.
  - Experience memory (FactorMiner 2026): operator-pair win-rates and productive
    search patterns are accumulated so the LLM avoids repeating known dead ends.
  - Family-aware deduplication (Hubble 2026): a structural fingerprint strips
    field names and numeric params so near-duplicate formulas (same operator
    skeleton, different fields/lookbacks) are detected and suppressed.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import threading
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import urllib3

from runtime_config import get_embedding_routing, load_runtime_config, openai_embeddings_url

KB_PATH = Path(__file__).resolve().parent / "knowledge.json"
SUBMIT_DIR = Path(__file__).resolve().parent / "submit"
GENERATION_NOTES_DIR = Path(__file__).resolve().parent / "generation_notes"
EMBEDDINGS_PATH = Path(__file__).resolve().parent / "knowledge_embeddings.json"

_EMPTY_KB: Dict[str, Any] = {
    "version": 3,
    "updated_at": "",
    "total_tested": 0,
    "total_passing": 0,
    "best_score": 0.0,
    "factors": {},
    # FactorMiner-inspired: accumulated search experience
    "skill_memory": {
        "operator_pairs": {},   # "opA|opB" → {attempts, wins, total_score}
        "family_records": {},   # fingerprint → {attempts, wins, avg_score, example}
        "productive_ops": {},   # single op → {attempts, wins}
    },
    "generation_experiences": {},
}

_MOTIF_TOKENS = [
    "volume", "trade_count", "dvolume", "vwap",
    "close_trade_px", "high_trade_px", "low_trade_px",
    "ts_mean", "ts_zscore", "ts_rank", "ts_decay_linear",
    "ts_ema", "ts_median", "ts_minmax_norm", "safe_div",
    "cs_rank", "cs_zscore", "cs_neutralize", "delta", "lag",
    "ts_corr", "ts_cov", "ifelse", "gt", "lt",
]

_PRICE_FIELDS = {
    "open_trade_px",
    "high_trade_px",
    "low_trade_px",
    "close_trade_px",
    "vwap",
}
_VOLUME_FIELDS = {
    "trade_count",
    "volume",
    "dvolume",
}
_FIELD_GROUPS = {
    **{field: "_price" for field in _PRICE_FIELDS},
    **{field: "_vol" for field in _VOLUME_FIELDS},
}

_ALL_OPS = [
    "ts_decay_linear", "ts_corr", "ts_cov", "ts_zscore", "ts_rank",
    "ts_mean", "ts_std", "ts_sum", "ts_max", "ts_min", "ts_median",
    "ts_quantile", "ts_skew", "ts_kurt", "ts_ema", "ts_argmax", "ts_argmin",
    "ts_pct_change", "ts_minmax_norm",
    "lag", "delta", "cs_rank", "cs_zscore", "cs_demean",
    "cs_scale", "cs_winsorize", "cs_quantile", "cs_neutralize",
    "safe_div", "signed_power", "abs", "sign", "neg", "log", "signed_log",
    "sqrt", "clip", "clamp", "min_of", "max_of", "sigmoid", "tanh",
    "ifelse", "gt", "ge", "lt", "le", "eq", "and_op", "or_op", "not_op",
    "mean_of", "weighted_sum", "combine_rank",
]

_EMBED_FAIL_UNTIL = 0.0
_EMBED_FAIL_LOCK = threading.Lock()
_EMBED_QUEUE_LOCK = threading.Lock()
_EMBED_PENDING_RUN_IDS: set[str] = set()
_EMBED_WORKER: threading.Thread | None = None

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# ─────────────────────────────────────────────────────────────────────────────
# Structural fingerprint (Hubble family-aware deduplication)
# ─────────────────────────────────────────────────────────────────────────────

def formula_structural_fingerprint(formula: str) -> str:
    """
    Replace field names with semantic field-class placeholders and numeric
    literals with placeholders so that near-duplicate formulas map to the same
    family without conflating economically different price and volume signals.

    Example:
        ts_decay_linear(cs_zscore((close_trade_px-vwap)/open_trade_px),6)
        ts_decay_linear(cs_zscore(volume/dvolume),10)
    → "ts_decay_linear(cs_zscore((_price-_price)/_price),_n)"
    → "ts_decay_linear(cs_zscore(_vol/_vol),_n)"
    """
    s = re.sub(r"\s+", "", formula or "").lower()
    for field in sorted(_FIELD_GROUPS, key=len, reverse=True):
        s = s.replace(field, _FIELD_GROUPS[field])
    s = re.sub(r"\b\d+\b", "_n", s)
    return s


def _ops_in_formula(formula: str) -> list[str]:
    """Return sorted list of operator names found in a formula."""
    lowered = re.sub(r"\s+", "", formula or "").lower()
    return sorted({op for op in _ALL_OPS if op in lowered})


def _embedding_model() -> str:
    cfg = load_runtime_config()
    return str(cfg.get("EMBEDDING_MODEL", "text-embedding-3-small") or "text-embedding-3-small").strip()


def _empty_embedding_store() -> Dict[str, Any]:
    return {
        "version": 1,
        "updated_at": "",
        "vectors": {},
    }


def _load_embedding_store() -> Dict[str, Any]:
    if EMBEDDINGS_PATH.is_file():
        try:
            payload = json.loads(EMBEDDINGS_PATH.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                payload.setdefault("vectors", {})
                return payload
        except Exception:
            pass
    return _empty_embedding_store()


def _save_embedding_store(payload: Dict[str, Any]) -> None:
    payload["updated_at"] = datetime.now().isoformat()
    EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    EMBEDDINGS_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _candidate_embedding_urls(api_base: str) -> list[str]:
    primary = openai_embeddings_url(api_base)
    urls = [primary]
    if "vip.aipro.love" in primary:
        urls.append("https://free.aipro.love/v1/embeddings")
    return [url for url in dict.fromkeys(urls) if url]


def _embedding_cooldown_active() -> bool:
    with _EMBED_FAIL_LOCK:
        return time.time() < _EMBED_FAIL_UNTIL


def _mark_embedding_failure(cooldown_sec: float = 300.0) -> None:
    global _EMBED_FAIL_UNTIL
    with _EMBED_FAIL_LOCK:
        _EMBED_FAIL_UNTIL = time.time() + max(30.0, cooldown_sec)


def _normalize_vector(vector: list[float]) -> list[float]:
    norm = sum(float(x) * float(x) for x in vector) ** 0.5
    if norm <= 0:
        return [0.0 for _ in vector]
    return [float(x) / norm for x in vector]


def _dot_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    return float(sum(float(a) * float(b) for a, b in zip(vec_a, vec_b)))


def _text_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def _factor_embedding_text(entry: Dict[str, Any]) -> str:
    formula = str(entry.get("formula", "") or "").strip()
    thought = str(entry.get("thought_process", "") or "").strip()
    motif = formula_motif(formula)
    family = formula_structural_fingerprint(formula)
    ops = ", ".join(_ops_in_formula(formula)[:8])
    parts = [
        f"thought: {thought}",
        f"formula: {formula}",
        f"motif: {motif}",
        f"family: {family}",
        f"operators: {ops}",
    ]
    return "\n".join(part for part in parts if part.strip())


def _trace_rag(tag: str, message: str) -> None:
    print(f"{tag} {message}", flush=True)


def _embed_texts(texts: list[str], *, model: str | None = None) -> list[list[float]] | None:
    if not texts:
        return None
    if _embedding_cooldown_active():
        _trace_rag("[Embeding]", "skip reason=cooldown_active")
        return None
    routing = get_embedding_routing()
    api_key = routing.get("api_key", "")
    api_base = routing.get("api_base", "")
    model_name = (model or _embedding_model()).strip()
    if not api_key or not api_base or not model_name:
        _trace_rag(
            "[Embeding]",
            f"skip reason=missing_route api_base={'yes' if api_base else 'no'} api_key={'yes' if api_key else 'no'} model={model_name or '--'}",
        )
        return None

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name,
        "input": texts,
    }
    _trace_rag("[Embeding]", f"request model={model_name} texts={len(texts)}")
    last_error: str = ""
    for url in _candidate_embedding_urls(api_base):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=20, verify=False)
            if resp.status_code >= 400:
                last_error = f"status={resp.status_code} body={resp.text[:300]}"
                continue
            data = resp.json()
            rows = data.get("data") if isinstance(data, dict) else None
            if not isinstance(rows, list):
                last_error = f"bad_embedding_payload={str(data)[:300]}"
                continue
            vectors = []
            for row in rows:
                emb = row.get("embedding") if isinstance(row, dict) else None
                if not isinstance(emb, list):
                    vectors = []
                    break
                vectors.append(_normalize_vector([float(x) for x in emb]))
            if len(vectors) == len(texts):
                dims = len(vectors[0]) if vectors and isinstance(vectors[0], list) else 0
                _trace_rag("[Embeding]", f"success model={model_name} texts={len(texts)} dims={dims} url={url}")
                return vectors
            last_error = "embedding_count_mismatch"
        except Exception as exc:
            last_error = str(exc)
            continue
    if last_error:
        _trace_rag("[Embeding]", f"failed model={model_name} reason={last_error[:220]}")
    cooldown = 3600.0 if ("model_not_found" in last_error or "无可用渠道" in last_error) else 300.0
    _mark_embedding_failure(cooldown_sec=cooldown)
    return None


def _upsert_factor_embeddings(records: list[tuple[str, Dict[str, Any]]]) -> int:
    candidates: list[tuple[str, str, Dict[str, Any]]] = []
    model_name = _embedding_model()
    store = _load_embedding_store()
    vectors = store.setdefault("vectors", {})
    for run_id, entry in records:
        text = _factor_embedding_text(entry)
        if not text.strip():
            continue
        text_hash = _text_hash(text)
        existing = vectors.get(run_id, {})
        if (
            existing.get("text_hash") == text_hash
            and existing.get("model") == model_name
            and isinstance(existing.get("vector"), list)
        ):
            continue
        candidates.append((run_id, text, entry))
    if not candidates:
        _trace_rag("[Embeding]", f"cache_hit records={len(records)} updated=0")
        return 0
    total_updated = 0
    batch_size = 10
    for start in range(0, len(candidates), batch_size):
        chunk = candidates[start:start + batch_size]
        embedded = _embed_texts([item[1] for item in chunk], model=model_name)
        if not embedded:
            continue
        for (run_id, text, entry), vector in zip(chunk, embedded):
            vectors[run_id] = {
                "model": model_name,
                "text_hash": _text_hash(text),
                "vector": vector,
                "formula": entry.get("formula", ""),
                "updated_at": datetime.now().isoformat(),
            }
            total_updated += 1
    if total_updated <= 0:
        _trace_rag("[Embeding]", f"no_update candidates={len(candidates)}")
        return 0
    _save_embedding_store(store)
    _trace_rag("[Embeding]", f"upserted updated={total_updated} total_candidates={len(candidates)} model={model_name}")
    return total_updated


def _ensure_passing_factor_embeddings(passing: list[Dict[str, Any]]) -> int:
    records = [(str(item.get("run_id", "")), item) for item in passing if item.get("run_id")]
    return _upsert_factor_embeddings(records)


def _factor_family_key(entry: Dict[str, Any]) -> str:
    formula = str(entry.get("formula", "") or "").strip()
    if not formula:
        return str(entry.get("run_id", "") or "")
    return formula_structural_fingerprint(formula) or formula_motif(formula)


def _lexical_similarity_scores(
    passing: list[Dict[str, Any]],
    query_text: str,
) -> dict[str, float]:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except Exception:
        return {}

    corpus = [_factor_embedding_text(item) for item in passing]
    if not any(text.strip() for text in corpus):
        return {}

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    matrix = vectorizer.fit_transform(corpus + [query_text])
    query_vec = matrix[-1]
    doc_mat = matrix[:-1]
    similarities = (doc_mat @ query_vec.T).toarray().ravel().tolist()
    scores: dict[str, float] = {}
    for similarity, item in zip(similarities, passing):
        run_id = str(item.get("run_id", ""))
        if not run_id:
            continue
        scores[run_id] = float(similarity)
    return scores


def _select_query_guided_rows(
    ranked: list[tuple[float, Dict[str, Any], float | None]],
    passing: list[Dict[str, Any]],
    *,
    primary_kind: str,
    semantic_k: int,
    anchor_count: int,
    max_factors: int,
) -> list[tuple[str, Dict[str, Any], float | None]]:
    selected: list[tuple[str, Dict[str, Any], float | None]] = []
    seen_run_ids: set[str] = set()
    seen_families: set[str] = set()
    primary_limit = min(max_factors, max(1, semantic_k))

    def _append(kind: str, item: Dict[str, Any], score: float | None, *, enforce_family_diversity: bool) -> bool:
        run_id = str(item.get("run_id", ""))
        if not run_id or run_id in seen_run_ids:
            return False
        family_key = _factor_family_key(item)
        if enforce_family_diversity and family_key and family_key in seen_families:
            return False
        selected.append((kind, item, score))
        seen_run_ids.add(run_id)
        if family_key:
            seen_families.add(family_key)
        return True

    for combined_score, item, display_score in ranked:
        if _append(primary_kind, item, display_score if display_score is not None else combined_score, enforce_family_diversity=True):
            if len(selected) >= primary_limit:
                break

    if len(selected) < primary_limit:
        for combined_score, item, display_score in ranked:
            if _append(primary_kind, item, display_score if display_score is not None else combined_score, enforce_family_diversity=False):
                if len(selected) >= primary_limit:
                    break

    for item in passing[: max(1, anchor_count)]:
        if _append("anchor", item, None, enforce_family_diversity=False) and len(selected) >= max_factors:
            break

    return selected[:max_factors]


def _lexical_similarity_selection(
    passing: list[Dict[str, Any]],
    query_text: str,
    *,
    semantic_k: int,
    anchor_count: int,
    max_factors: int,
) -> list[tuple[str, Dict[str, Any], float | None]]:
    score_map = _lexical_similarity_scores(passing, query_text)
    if not score_map:
        return [("top", item, None) for item in passing[:max_factors]]
    ranked = sorted(
        ((score_map.get(str(item.get("run_id", "")), 0.0), item, score_map.get(str(item.get("run_id", "")), 0.0)) for item in passing),
        key=lambda pair: pair[0],
        reverse=True,
    )
    return _select_query_guided_rows(
        ranked,
        passing,
        primary_kind="lexical",
        semantic_k=semantic_k,
        anchor_count=anchor_count,
        max_factors=max_factors,
    )


def _background_embed_loop() -> None:
    global _EMBED_WORKER
    while True:
        with _EMBED_QUEUE_LOCK:
            run_ids = list(_EMBED_PENDING_RUN_IDS)
            _EMBED_PENDING_RUN_IDS.clear()
        if not run_ids:
            break
        kb = _load()
        factors = kb.get("factors", {})
        records = [
            (run_id, factors.get(run_id, {}))
            for run_id in run_ids
            if factors.get(run_id, {}).get("PassGates") and factors.get(run_id, {}).get("formula")
        ]
        try:
            _upsert_factor_embeddings(records)
        except Exception as exc:
            print(f"[knowledge_base] Background embedding refresh failed: {exc}")
        time.sleep(0.1)
    with _EMBED_QUEUE_LOCK:
        _EMBED_WORKER = None
        if _EMBED_PENDING_RUN_IDS and not _embedding_cooldown_active():
            worker = threading.Thread(target=_background_embed_loop, name="autoalpha-embed", daemon=True)
            _EMBED_WORKER = worker
            worker.start()


def _schedule_factor_embedding(run_id: str, entry: Dict[str, Any]) -> None:
    global _EMBED_WORKER
    if not run_id or not entry.get("PassGates") or not entry.get("formula") or _embedding_cooldown_active():
        return
    with _EMBED_QUEUE_LOCK:
        _EMBED_PENDING_RUN_IDS.add(run_id)
        if _EMBED_WORKER is not None and _EMBED_WORKER.is_alive():
            return
        worker = threading.Thread(target=_background_embed_loop, name="autoalpha-embed", daemon=True)
        _EMBED_WORKER = worker
        worker.start()


def _rebuild_family_records(kb: Dict[str, Any]) -> None:
    """Recompute structural family records after fingerprint schema changes."""
    family_records: Dict[str, Dict[str, Any]] = {}
    for run_id, info in kb.get("factors", {}).items():
        formula = info.get("formula", "")
        status = info.get("status", "")
        passed = bool(info.get("PassGates", False))
        if status not in ("ok", "screened_out") and not passed:
            continue
        if not formula:
            continue
        score = float(info.get("Score", 0) or 0)
        fingerprint = formula_structural_fingerprint(formula)
        info["fingerprint"] = fingerprint
        rec = family_records.setdefault(
            fingerprint,
            {
                "attempts": 0,
                "wins": 0,
                "avg_score": 0.0,
                "example": formula,
                "_best_example_score": float("-inf"),
            },
        )
        rec["attempts"] += 1
        if passed:
            rec["wins"] += 1
            n = rec["attempts"]
            rec["avg_score"] = (float(rec.get("avg_score", 0) or 0) * (n - 1) + score) / n
        if score > float(rec.get("_best_example_score", float("-inf"))):
            rec["example"] = formula
            rec["_best_example_score"] = score

    for rec in family_records.values():
        rec.pop("_best_example_score", None)
    kb.setdefault("skill_memory", _EMPTY_KB["skill_memory"].copy())["family_records"] = family_records


def _migrate_kb_schema(kb: Dict[str, Any]) -> bool:
    """Apply one-time KB migrations. Returns True when the payload changed."""
    changed = False
    version = int(kb.get("version", 1) or 1)
    if version < 3:
        _rebuild_family_records(kb)
        kb["version"] = 3
        changed = True
    return changed


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load() -> Dict[str, Any]:
    if KB_PATH.exists():
        try:
            with open(KB_PATH, "r", encoding="utf-8") as f:
                kb = json.load(f)
            # Back-fill skill_memory section for older KB files
            if "skill_memory" not in kb:
                kb["skill_memory"] = _EMPTY_KB["skill_memory"].copy()
            if "generation_experiences" not in kb:
                kb["generation_experiences"] = {}
            if _migrate_kb_schema(kb):
                kb["updated_at"] = datetime.now().isoformat()
                with open(KB_PATH, "w", encoding="utf-8") as f:
                    json.dump(kb, f, indent=2, ensure_ascii=False, default=str)
            return kb
        except Exception:
            pass
    return _EMPTY_KB.copy()


def _save(kb: Dict[str, Any]) -> None:
    KB_PATH.parent.mkdir(parents=True, exist_ok=True)
    kb["updated_at"] = datetime.now().isoformat()
    kb["total_tested"] = len(kb.get("factors", {}))
    kb["total_passing"] = sum(
        1 for f in kb.get("factors", {}).values() if f.get("PassGates")
    )
    kb["best_score"] = max(
        (f.get("Score", 0) for f in kb.get("factors", {}).values()),
        default=0.0,
    )
    with open(KB_PATH, "w", encoding="utf-8") as f:
        json.dump(kb, f, indent=2, ensure_ascii=False, default=str)


def _copy_submit_artifact(run_id: str, entry: Dict[str, Any]) -> Dict[str, Any]:
    """Copy passing factor parquet into autoalpha_v2/submit and return metadata fields."""
    if not entry.get("PassGates"):
        return {}

    parquet_path = entry.get("parquet_path") or ""
    if not parquet_path:
        return {}

    src = Path(str(parquet_path))
    if not src.is_file():
        return {}

    SUBMIT_DIR.mkdir(parents=True, exist_ok=True)
    dst = SUBMIT_DIR / f"{run_id}{src.suffix or '.pq'}"
    shutil.copy2(src, dst)

    meta_path = SUBMIT_DIR / f"{run_id}_metadata.json"
    meta = {
        "run_id": run_id,
        "formula": entry.get("formula", ""),
        "Score": entry.get("Score", 0),
        "IC": entry.get("IC", 0),
        "IR": entry.get("IR", 0),
        "tvr": entry.get("tvr", 0),
        "PassGates": bool(entry.get("PassGates")),
        "source_parquet_path": str(src),
        "submit_path": str(dst),
        "copied_at": datetime.now().isoformat(),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "submit_path": str(dst),
        "submit_metadata_path": str(meta_path),
        "submit_copied_at": meta["copied_at"],
    }


def _factor_card_path(research_path: Any) -> str:
    if not research_path:
        return ""
    path = Path(str(research_path)) / "factor_card.json"
    return str(path) if path.is_file() else ""


# ─────────────────────────────────────────────────────────────────────────────
# Skill memory update (FactorMiner-inspired)
# ─────────────────────────────────────────────────────────────────────────────

def _update_skill_memory(kb: Dict[str, Any], result: Dict[str, Any]) -> None:
    """
    After each evaluation, update operator-pair win-rates and family records
    so the LLM can be steered away from structurally exhausted search regions.
    """
    formula = result.get("formula", "")
    passed = bool(result.get("PassGates", False))
    score = float(result.get("Score", 0) or 0)
    status = result.get("status", "")

    if status not in ("ok", "screened_out") or not formula:
        return

    sm = kb.setdefault("skill_memory", _EMPTY_KB["skill_memory"].copy())

    # Operator-pair win rates
    ops = _ops_in_formula(formula)
    pair_records = sm.setdefault("operator_pairs", {})
    for i, opa in enumerate(ops):
        for opb in ops[i + 1:]:
            key = f"{opa}|{opb}"
            rec = pair_records.setdefault(key, {"attempts": 0, "wins": 0, "total_score": 0.0})
            rec["attempts"] += 1
            if passed:
                rec["wins"] += 1
                rec["total_score"] += score

    # Single-op productive rates
    prod = sm.setdefault("productive_ops", {})
    for op in ops:
        rec = prod.setdefault(op, {"attempts": 0, "wins": 0})
        rec["attempts"] += 1
        if passed:
            rec["wins"] += 1

    # Family fingerprint records
    fingerprint = formula_structural_fingerprint(formula)
    family_records = sm.setdefault("family_records", {})
    frec = family_records.setdefault(
        fingerprint,
        {"attempts": 0, "wins": 0, "avg_score": 0.0, "example": formula},
    )
    frec["attempts"] += 1
    if passed:
        frec["wins"] += 1
        n = frec["attempts"]
        frec["avg_score"] = (frec["avg_score"] * (n - 1) + score) / n
    # Keep the highest-scoring formula as the representative example
    if score > frec.get("avg_score", 0):
        frec["example"] = formula


# ─────────────────────────────────────────────────────────────────────────────
# Public write API
# ─────────────────────────────────────────────────────────────────────────────

def add_factor(
    result: Dict[str, Any],
    parent_run_ids: Optional[List[str]] = None,
) -> None:
    """Persist one factor result (pass or fail) into the knowledge base."""
    run_id = result.get("run_id", "")
    if not run_id:
        return

    kb = _load()
    factors = kb.setdefault("factors", {})
    existing = factors.get(run_id, {})

    generation = 0
    if parent_run_ids:
        # Include leaderboard parents (not in KB) as generation 0 so children advance to gen 1+
        parent_gens = [factors.get(pid, {}).get("generation", 0) for pid in parent_run_ids]
        generation = max(parent_gens) + 1

    entry: Dict[str, Any] = {
        "formula": result.get("formula", ""),
        "thought_process": result.get("thought_process", ""),
        "IC": float(result.get("IC", 0) or 0),
        "IR": float(result.get("IR", 0) or 0),
        "tvr": float(result.get("tvr", result.get("Turnover", 0)) or 0),
        "Score": float(result.get("Score", 0) or 0),
        "PassGates": bool(result.get("PassGates", False)),
        "postprocess": result.get("postprocess", "rank_clip"),
        "lookback_days": int(result.get("lookback_days", 20) or 20),
        "status": result.get("status", "ok"),
        "errors": result.get("errors", result.get("error", "")),
        "screen_fail_reason": result.get("screen_fail_reason", ""),
        "screen_fail_details": result.get("screen_fail_details", []),
        "screening": result.get("screening", {}),
        "parent_run_ids": parent_run_ids or [],
        "generation": generation,
        "inspiration_source_type": result.get("inspiration_source_type", "none"),
        "inspiration_source_types": result.get("inspiration_source_types", []),
        "inspiration_ids": result.get("inspiration_ids", []),
        "generation_mode": result.get("generation_mode", ""),
        "target_source": result.get("target_source", result.get("inspiration_source_type", "")),
        "prompt_version": result.get("prompt_version", ""),
        "created_at": datetime.now().isoformat(),
        "parquet_path": result.get("parquet_path", ""),
        "eval_days": int(result.get("eval_days", 0) or 0),
        "research_path": result.get("research_path", ""),
        "factor_card_path": _factor_card_path(result.get("research_path", "")),
        "fingerprint": formula_structural_fingerprint(result.get("formula", "")),
    }

    for key in ("live_test_result", "live_submitted", "live_result_updated_at"):
        if key in existing:
            entry[key] = existing[key]

    entry.update(_copy_submit_artifact(run_id, entry))

    factors[run_id] = entry
    _update_skill_memory(kb, result)
    _save(kb)
    _schedule_factor_embedding(run_id, entry)
    if entry.get("PassGates") and (entry.get("submit_path") or entry.get("parquet_path")):
        try:
            from autoalpha_v2.factor_research import (
                DEFAULT_CORRELATION_MATRIX_FACTORS,
                schedule_factor_correlation_refresh,
            )

            schedule_factor_correlation_refresh(
                max_factors=DEFAULT_CORRELATION_MATRIX_FACTORS,
                update_cards=True,
            )
        except Exception as exc:
            print(f"[knowledge_base] Factor correlation refresh failed for {run_id}: {exc}")


def sync_submit_artifacts() -> Dict[str, Any]:
    """Backfill autoalpha_v2/submit with all currently passing parquet factors."""
    kb = _load()
    factors = kb.setdefault("factors", {})
    copied = 0
    skipped = 0
    for run_id, entry in factors.items():
        fields = _copy_submit_artifact(run_id, entry)
        if fields:
            entry.update(fields)
            copied += 1
        else:
            skipped += 1
    _save(kb)
    return {"submit_dir": str(SUBMIT_DIR), "copied": copied, "skipped": skipped}


# ─────────────────────────────────────────────────────────────────────────────
# Generation experience notes
# ─────────────────────────────────────────────────────────────────────────────

def _failure_reason(item: Dict[str, Any]) -> str:
    if item.get("PassGates"):
        return "passed"
    status = item.get("status", "")
    if status == "invalid":
        return "invalid_formula"
    if status == "compute_error":
        return "compute_error"
    if status == "duplicate":
        return "duplicate_formula"
    if status == "screened_out":
        return "screened_out"
    if float(item.get("tvr", 0) or 0) >= 330:
        return "high_tvr"
    if float(item.get("IR", 0) or 0) <= 2.5:
        return "low_ir"
    if float(item.get("IC", 0) or 0) <= 0.6:
        return "low_ic"
    if float(item.get("Score", 0) or 0) <= 0:
        return "zero_score"
    return "failed_gate"


def build_generation_experience_payload(generation: int) -> Dict[str, Any]:
    kb = _load()
    factors = [
        {"run_id": rid, **info}
        for rid, info in kb.get("factors", {}).items()
        if int(info.get("generation", 0) or 0) == int(generation)
    ]
    factors.sort(key=lambda item: item.get("created_at", ""))
    reason_counts = Counter(_failure_reason(item) for item in factors)
    top = sorted(factors, key=lambda item: (item.get("Score", 0), item.get("IC", 0)), reverse=True)[:6]
    weak = sorted(factors, key=lambda item: (item.get("PassGates", False), item.get("Score", 0), item.get("IC", 0)))[:8]
    return {
        "generation": int(generation),
        "created_at": datetime.now().isoformat(),
        "total": len(factors),
        "passing": sum(1 for item in factors if item.get("PassGates")),
        "best_score": max((float(item.get("Score", 0) or 0) for item in factors), default=0.0),
        "failure_counts": dict(reason_counts),
        "top_examples": [
            {
                "run_id": item.get("run_id", ""),
                "formula": item.get("formula", ""),
                "thought_process": item.get("thought_process", ""),
                "IC": float(item.get("IC", 0) or 0),
                "IR": float(item.get("IR", 0) or 0),
                "tvr": float(item.get("tvr", 0) or 0),
                "Score": float(item.get("Score", 0) or 0),
                "PassGates": bool(item.get("PassGates")),
                "status": item.get("status", ""),
                "failure_reason": _failure_reason(item),
                "motif": formula_motif(item.get("formula", "")),
            }
            for item in top
        ],
        "weak_examples": [
            {
                "run_id": item.get("run_id", ""),
                "formula": item.get("formula", ""),
                "IC": float(item.get("IC", 0) or 0),
                "IR": float(item.get("IR", 0) or 0),
                "tvr": float(item.get("tvr", 0) or 0),
                "Score": float(item.get("Score", 0) or 0),
                "PassGates": bool(item.get("PassGates")),
                "status": item.get("status", ""),
                "failure_reason": _failure_reason(item),
                "motif": formula_motif(item.get("formula", "")),
            }
            for item in weak
        ],
    }


def save_generation_experience(generation: int, markdown: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    generation = int(generation)
    GENERATION_NOTES_DIR.mkdir(parents=True, exist_ok=True)
    note_path = GENERATION_NOTES_DIR / f"generation_{generation:03d}.md"
    note_path.write_text(markdown.strip() + "\n", encoding="utf-8")

    kb = _load()
    record = {
        "generation": generation,
        "created_at": datetime.now().isoformat(),
        "path": str(note_path),
        "relative_path": str(note_path.relative_to(Path(__file__).resolve().parents[1])),
        "summary": _markdown_summary(markdown),
        "stats": {
            "total": payload.get("total", 0),
            "passing": payload.get("passing", 0),
            "best_score": payload.get("best_score", 0.0),
            "failure_counts": payload.get("failure_counts", {}),
        },
    }
    kb.setdefault("generation_experiences", {})[str(generation)] = record
    _save(kb)
    return record


def _markdown_summary(markdown: str, limit: int = 220) -> str:
    text = re.sub(r"`+", "", markdown or "")
    text = re.sub(r"#+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def get_generation_experience(generation: int) -> Optional[Dict[str, Any]]:
    kb = _load()
    record = kb.get("generation_experiences", {}).get(str(int(generation)))
    if not record:
        return None
    markdown = ""
    path = record.get("path")
    if path and Path(str(path)).is_file():
        markdown = Path(str(path)).read_text(encoding="utf-8")
    return {**record, "markdown": markdown}


def list_generation_experiences() -> List[Dict[str, Any]]:
    kb = _load()
    records = list(kb.get("generation_experiences", {}).values())
    records.sort(key=lambda item: int(item.get("generation", 0) or 0))
    return records


def compose_recent_generation_experience_context(limit: int = 3, max_chars: int = 2200) -> str:
    records = list_generation_experiences()[-max(1, limit):]
    lines: List[str] = []
    for record in records:
        stats = record.get("stats", {})
        failures = stats.get("failure_counts", {})
        failure_text = ", ".join(f"{k}={v}" for k, v in failures.items()) or "none"
        lines.append(
            f"- Generation {record.get('generation')}: tested={stats.get('total', 0)}, "
            f"passing={stats.get('passing', 0)}, best_score={float(stats.get('best_score', 0) or 0):.2f}, "
            f"failures=({failure_text}). Notes: {record.get('summary', '')}"
        )
    text = "\n".join(lines)
    return text[:max_chars]


def _experience_query_tokens(archetype: str = "", failure_mode: str = "") -> set[str]:
    text = f"{archetype} {failure_mode}".lower()
    aliases = {
        "mean_reversion": ["mean", "reversion", "vwap", "reversal", "baseline"],
        "momentum": ["momentum", "continuation", "trend", "persistence"],
        "volatility": ["volatility", "range", "compression", "release"],
        "volume_signal": ["volume", "dvolume", "trade_count", "participation", "liquidity"],
        "range_location": ["range", "location", "bar", "high", "low", "close"],
        "exhaustion": ["exhaustion", "reversal", "follow-through", "failed"],
        "high_tvr": ["tvr", "turnover", "smooth", "smoother"],
        "low_ir": ["ir", "unstable", "stability"],
        "low_ic": ["ic", "signal", "noisy"],
        "compute_error": ["syntax", "compute", "dsl", "error"],
        "screened_out": ["screen", "gate", "failed"],
    }
    tokens = {
        token
        for token in re.split(r"[^a-z0-9_]+", text)
        if len(token) >= 3
    }
    for key, values in aliases.items():
        if key in text or key.replace("_", " ") in text:
            tokens.update(values)
    return tokens


def find_relevant_experience(
    archetype: str = "",
    failure_mode: str = "",
    *,
    exclude_recent: int = 2,
    max_chars: int = 360,
) -> str:
    """
    Return one older generation summary relevant to the current archetype/failure
    keywords. This complements the recency context without adding embeddings.
    """
    records = list_generation_experiences()
    if len(records) <= exclude_recent:
        return ""
    query_tokens = _experience_query_tokens(archetype, failure_mode)
    if not query_tokens:
        return ""

    recent_generations = {
        int(record.get("generation", 0) or 0)
        for record in records[-max(0, exclude_recent):]
    }
    best: tuple[int, int, Dict[str, Any]] | None = None
    for record in records:
        generation = int(record.get("generation", 0) or 0)
        if generation in recent_generations:
            continue
        stats = record.get("stats", {})
        failures = " ".join((stats.get("failure_counts") or {}).keys())
        haystack = f"{record.get('summary', '')} {failures}".lower()
        score = sum(1 for token in query_tokens if token in haystack)
        if score <= 0:
            continue
        # Prefer more relevant records, then more recent historical records.
        candidate = (score, generation, record)
        if best is None or candidate[:2] > best[:2]:
            best = candidate

    if best is None:
        return ""
    record = best[2]
    stats = record.get("stats", {})
    failures = stats.get("failure_counts", {})
    failure_text = ", ".join(f"{k}={v}" for k, v in failures.items()) or "none"
    text = (
        f"- Generation {record.get('generation')}: tested={stats.get('total', 0)}, "
        f"passing={stats.get('passing', 0)}, best_score={float(stats.get('best_score', 0) or 0):.2f}, "
        f"failures=({failure_text}). Notes: {record.get('summary', '')}"
    )
    return text[:max_chars]


# ─────────────────────────────────────────────────────────────────────────────
# Formula utilities
# ─────────────────────────────────────────────────────────────────────────────

def canonical_formula(formula: str) -> str:
    return re.sub(r"\s+", "", formula or "").strip().lower()


def formula_motif(formula: str) -> str:
    lowered = canonical_formula(formula)
    hits = [token for token in _MOTIF_TOKENS if token in lowered]
    return " + ".join(hits[:6]) if hits else "generic"


def get_existing_formula_keys() -> set[str]:
    kb = _load()
    return {
        canonical_formula(info.get("formula", ""))
        for info in kb.get("factors", {}).values()
        if info.get("formula")
    }


def get_existing_fingerprints() -> Dict[str, Dict[str, Any]]:
    """Return family fingerprint → record for all tested formulas."""
    kb = _load()
    return dict(kb.get("skill_memory", {}).get("family_records", {}))


# ─────────────────────────────────────────────────────────────────────────────
# Guidance for LLM generation
# ─────────────────────────────────────────────────────────────────────────────

def get_generation_guidance(
    recent_limit: int = 20,
    crowded_threshold: int = 5,
) -> Dict[str, Any]:
    kb = _load()
    all_factors = [
        {"run_id": rid, **info}
        for rid, info in kb.get("factors", {}).items()
        if info.get("formula")
    ]
    all_factors.sort(key=lambda item: item.get("created_at", ""))

    recent = all_factors[-recent_limit:]
    recent_failed = [
        item for item in recent
        if item.get("status") == "ok" and not item.get("PassGates")
    ]
    passing = [item for item in all_factors if item.get("PassGates")]

    # Token-level crowding from recent failures
    motif_counter: Counter[str] = Counter()
    for item in recent_failed:
        motif_counter.update(
            token for token in _MOTIF_TOKENS
            if token in canonical_formula(item.get("formula", ""))
        )
    crowded_tokens = [t for t, cnt in motif_counter.items() if cnt >= crowded_threshold]

    low_quality = sorted(
        recent_failed, key=lambda item: (item.get("Score", 0), item.get("IC", 0))
    )[:4]
    strong = sorted(
        passing, key=lambda item: (item.get("Score", 0), item.get("IC", 0)), reverse=True
    )[:4]

    # FactorMiner: productive operator pairs (win_rate ≥ 0.4, ≥3 attempts)
    sm = kb.get("skill_memory", {})
    pair_records = sm.get("operator_pairs", {})
    productive_pairs = sorted(
        [
            (k, v)
            for k, v in pair_records.items()
            if v.get("attempts", 0) >= 3
            and v.get("wins", 0) / v.get("attempts", 1) >= 0.4
        ],
        key=lambda kv: kv[1].get("wins", 0) / max(kv[1].get("attempts", 1), 1),
        reverse=True,
    )[:5]

    # Hubble: saturated families (≥3 attempts, 0 wins)
    family_records = sm.get("family_records", {})
    exhausted_families = [
        {"fingerprint": fp, "attempts": rec["attempts"], "example": rec.get("example", "")}
        for fp, rec in family_records.items()
        if rec.get("attempts", 0) >= 3 and rec.get("wins", 0) == 0
    ][:6]

    # TVR dominance alert — check screened_out + ok failures for high TVR
    all_non_passing = [
        item for item in recent
        if item.get("status") in ("ok", "screened_out") and not item.get("PassGates")
    ]
    tvr_failures = [f for f in all_non_passing if float(f.get("tvr", 0) or 0) >= 330]
    failure_modes = Counter(_failure_reason(item) for item in all_non_passing)
    dominant_failure_mode = failure_modes.most_common(1)[0][0] if failure_modes else ""
    duplicate_count = sum(1 for item in recent if item.get("status") == "duplicate")
    tvr_alert: str = ""
    if all_non_passing and len(tvr_failures) / len(all_non_passing) >= 0.4:
        avg_tvr = sum(float(f.get("tvr", 0) or 0) for f in tvr_failures) / len(tvr_failures)
        tvr_alert = (
            f"CRITICAL TVR ALERT: {len(tvr_failures)}/{len(all_non_passing)} recent non-passing factors "
            f"failed due to TVR >= 330 (avg TVR={avg_tvr:.0f}). "
            "Root cause: outer smoother window too short. "
            "REQUIRED: wrap the final signal with ts_mean(x,10), ts_decay_linear(x,15), or ts_ema(x,10). "
            "Any formula with ts_decay_linear window < 8 will produce TVR > 500."
        )
        if duplicate_count >= 3:
            tvr_alert += (
                f" Additionally {duplicate_count}/{len(recent)} recent attempts were duplicates — "
                "generate structurally novel formulas."
            )

    return {
        "existing_formula_keys": get_existing_formula_keys(),
        "crowded_tokens": crowded_tokens,
        "recent_failed_examples": [
            {
                "run_id": item.get("run_id", ""),
                "formula": item.get("formula", ""),
                "IC": float(item.get("IC", 0) or 0),
                "IR": float(item.get("IR", 0) or 0),
                "Score": float(item.get("Score", 0) or 0),
                "motif": formula_motif(item.get("formula", "")),
            }
            for item in low_quality
        ],
        "strong_examples": [
            {
                "run_id": item.get("run_id", ""),
                "formula": item.get("formula", ""),
                "IC": float(item.get("IC", 0) or 0),
                "IR": float(item.get("IR", 0) or 0),
                "Score": float(item.get("Score", 0) or 0),
                "motif": formula_motif(item.get("formula", "")),
            }
            for item in strong
        ],
        # FactorMiner: experience memory
        "productive_operator_pairs": [
            f"{k} (win_rate={v['wins']}/{v['attempts']})"
            for k, v in productive_pairs
        ],
        # Hubble: exhausted structural families
        "exhausted_families": exhausted_families,
        "generation_experience_context": compose_recent_generation_experience_context(),
        "dominant_failure_mode": dominant_failure_mode,
        "tvr_alert": tvr_alert,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Parent selection
# ─────────────────────────────────────────────────────────────────────────────

def list_valid_factors(
    min_eval_days: int = 0,
    exclude_prefixes: tuple[str, ...] = ("test_",),
) -> List[Dict[str, Any]]:
    kb = _load()
    valid: List[Dict[str, Any]] = []
    for run_id, info in kb.get("factors", {}).items():
        if exclude_prefixes and any(str(run_id).startswith(p) for p in exclude_prefixes):
            continue
        if not info.get("PassGates"):
            continue
        if int(info.get("eval_days", 0) or 0) < int(min_eval_days):
            continue
        valid.append({"run_id": run_id, **info})
    valid.sort(key=lambda item: (item.get("Score", 0), item.get("IC", 0)), reverse=True)
    return valid


def get_top_parents(k: int = 4, min_ic: float = 0.2) -> List[Dict[str, Any]]:
    """Return top-K parents while preserving diversity beyond the current winners."""
    kb = _load()
    # Include screened_out so low-TVR partial successes can serve as parents
    all_factors = [
        {"run_id": rid, **info}
        for rid, info in kb.get("factors", {}).items()
        if info.get("status") in ("ok", "screened_out")
    ]

    passing = [f for f in all_factors if f.get("PassGates") and f.get("IC", 0) >= min_ic]
    selected: List[Dict[str, Any]] = []
    seen_formulas: set[str] = set()

    def _add(item: Dict[str, Any]) -> bool:
        formula = canonical_formula(item.get("formula", ""))
        if not formula or formula in seen_formulas:
            return False
        selected.append(item)
        seen_formulas.add(formula)
        return len(selected) >= k

    if passing:
        passing.sort(key=lambda x: (x.get("Score", 0), x.get("IC", 0)), reverse=True)
        # Keep the best anchors, but leave slots for exploratory parents.
        for item in passing[: max(1, min(2, k))]:
            if _add(item):
                return selected[:k]
        # Fill remaining parent slots with other passing factors from different
        # structural/source neighborhoods when available. This keeps the old
        # high-score anchors, but reduces over-conditioning on one formula family.
        seen_fingerprints = {
            formula_structural_fingerprint(item.get("formula", ""))
            for item in selected
        }
        seen_sources = {
            str(item.get("inspiration_source_type", ""))
            for item in selected
        }
        diverse_passing = sorted(
            passing,
            key=lambda item: (
                formula_structural_fingerprint(item.get("formula", "")) not in seen_fingerprints,
                str(item.get("inspiration_source_type", "")) not in seen_sources,
                float(item.get("Score", 0) or 0),
                float(item.get("IC", 0) or 0),
            ),
            reverse=True,
        )
        for item in diverse_passing:
            if formula_structural_fingerprint(item.get("formula", "")) in seen_fingerprints and len(selected) < max(2, k - 1):
                continue
            if _add(item):
                return selected[:k]
            seen_fingerprints.add(formula_structural_fingerprint(item.get("formula", "")))
            seen_sources.add(str(item.get("inspiration_source_type", "")))
    else:
        # Prefer low-TVR factors as fallback — high-TVR failures teach LLM the wrong patterns.
        # Never use factors with TVR > 400 as parents: they reinforce bad smoothing habits.
        low_tvr = [f for f in all_factors if float(f.get("tvr", 0) or 0) < 400]
        for item in sorted(low_tvr, key=lambda x: x.get("IC", 0), reverse=True)[: max(1, k // 2)]:
            if _add(item):
                return selected[:k]

    # Add promising but non-passing structures, ranked for low TVR and positive IC.
    exploratory = [
        f for f in all_factors
        if not f.get("PassGates")
        and float(f.get("IC", 0) or 0) >= min_ic
        and float(f.get("tvr", 0) or 0) < 400
    ]
    exploratory.sort(key=lambda x: (x.get("IC", 0), -float(x.get("tvr", 0) or 0)), reverse=True)
    for item in exploratory:
        if _add(item):
            return selected[:k]

    if len(selected) < k:
        try:
            from leaderboard import load_leaderboard

            for factor in sorted(
                load_leaderboard().get("factors", []),
                key=lambda item: (item.get("PassGates", False), item.get("Score", 0), item.get("IC", 0)),
                reverse=True,
            ):
                if not factor.get("formula"):
                    continue
                if canonical_formula(factor.get("formula", "")) in seen_formulas:
                    continue
                if not (factor.get("PassGates") or factor.get("submission_ready_flag")):
                    continue
                if _add({
                    "run_id": factor.get("factor_name", ""),
                    "formula": factor.get("formula", ""),
                    "thought_process": factor.get("recommendation")
                    or factor.get("classification", ""),
                    "IC": float(factor.get("IC", 0) or 0),
                    "IR": float(factor.get("IR", 0) or 0),
                    "tvr": float(factor.get("Turnover", 0) or 0),
                    "Score": float(factor.get("Score", 0) or 0),
                    "PassGates": bool(factor.get("PassGates", False)),
                    "generation": 0,
                    "inspiration_source_type": "leaderboard",
                }):
                    break
        except Exception:
            pass

    return selected[:k]


# ─────────────────────────────────────────────────────────────────────────────
# Summary / utility
# ─────────────────────────────────────────────────────────────────────────────

def get_summary() -> Dict[str, Any]:
    kb = _load()
    all_entries = [{"run_id": rid, **info} for rid, info in kb.get("factors", {}).items()]
    top = sorted(all_entries, key=lambda x: x.get("Score", 0), reverse=True)
    return {
        "total_tested": kb.get("total_tested", 0),
        "total_passing": kb.get("total_passing", 0),
        "best_score": kb.get("best_score", 0.0),
        "updated_at": kb.get("updated_at", ""),
        "top_factors": top[:10],
    }


def get_all_factors() -> List[Dict[str, Any]]:
    kb = _load()
    all_entries = [{"run_id": rid, **info} for rid, info in kb.get("factors", {}).items()]
    return sorted(all_entries, key=lambda x: x.get("Score", 0), reverse=True)


def _query_guided_passing_factors(
    passing: list[Dict[str, Any]],
    query_text: str,
    *,
    semantic_k: int,
    anchor_count: int,
    max_factors: int,
) -> tuple[list[tuple[str, Dict[str, Any], float | None]], str]:
    normalized_query = str(query_text or "").strip()
    if not normalized_query:
        _trace_rag("[Recall]", f"mode=top reason=empty_query selected={min(len(passing), max_factors)}")
        return [("top", item, None) for item in passing[:max_factors]], "top"

    updated = _ensure_passing_factor_embeddings(passing)
    store = _load_embedding_store()
    model_name = _embedding_model()
    query_vectors = _embed_texts([normalized_query], model=model_name)
    if not query_vectors:
        _trace_rag(
            "[Recall]",
            f"mode=lexical reason=no_query_embedding passing={len(passing)} updated_embeddings={updated} semantic_k={semantic_k} anchors={anchor_count}",
        )
        return _lexical_similarity_selection(
            passing,
            normalized_query,
            semantic_k=semantic_k,
            anchor_count=anchor_count,
            max_factors=max_factors,
        ), "lexical"
    query_vec = query_vectors[0]
    vectors = store.get("vectors", {})
    lexical_scores = _lexical_similarity_scores(passing, normalized_query)
    semantic_scored: list[tuple[float, Dict[str, Any], float | None]] = []
    semantic_raw_values: list[float] = []
    total = max(len(passing), 1)
    for index, item in enumerate(passing):
        record = vectors.get(str(item.get("run_id", "")), {})
        vector = record.get("vector")
        if record.get("model") != model_name or not isinstance(vector, list):
            continue
        semantic_score = _dot_similarity(query_vec, vector)
        semantic_raw_values.append(semantic_score)
        quality_prior = 1.0 - (index / total)
        lexical_score = lexical_scores.get(str(item.get("run_id", "")), 0.0)
        semantic_scored.append((semantic_score, item, lexical_score + quality_prior))
    if not semantic_scored:
        _trace_rag(
            "[Recall]",
            f"mode=lexical reason=no_factor_vectors passing={len(passing)} model={model_name} updated_embeddings={updated}",
        )
        return _lexical_similarity_selection(
            passing,
            normalized_query,
            semantic_k=semantic_k,
            anchor_count=anchor_count,
            max_factors=max_factors,
        ), "lexical"
    sem_min = min(semantic_raw_values)
    sem_max = max(semantic_raw_values)
    ranked: list[tuple[float, Dict[str, Any], float | None]] = []
    for semantic_score, item, lexical_plus_prior in semantic_scored:
        if sem_max > sem_min:
            semantic_norm = (semantic_score - sem_min) / (sem_max - sem_min)
        else:
            semantic_norm = 1.0
        lexical_score = lexical_scores.get(str(item.get("run_id", "")), 0.0)
        quality_prior = max(0.0, lexical_plus_prior - lexical_score)
        combined = (semantic_norm * 0.72) + (lexical_score * 0.22) + (quality_prior * 0.06)
        ranked.append((combined, item, semantic_score))
    ranked.sort(key=lambda pair: pair[0], reverse=True)
    selected = _select_query_guided_rows(
        ranked,
        passing,
        primary_kind="semantic",
        semantic_k=semantic_k,
        anchor_count=anchor_count,
        max_factors=max_factors,
    )
    top_sim = selected[0][2] if selected and selected[0][2] is not None else None
    _trace_rag(
        "[Recall]",
        f"mode=semantic passing={len(passing)} candidates={len(semantic_scored)} selected={len(selected)} "
        f"anchors={anchor_count} semantic_k={semantic_k} top_sim={f'{top_sim:.3f}' if top_sim is not None else '--'}",
    )
    return selected, "semantic"


def compose_passing_factors_rag(
    query_text: str = "",
    max_factors: int = 12,
    semantic_k: int = 6,
    anchor_count: int = 2,
    max_chars: int = 3600,
    include_formulas: bool = True,
    include_template: bool = True,
) -> str:
    """
    Build a RAG context string from passing factors.
    When query_text is available, use semantic retrieval against passing-factor
    embeddings and keep a couple of score-based anchors as guard rails.
    """
    kb = _load()
    passing = [
        {"run_id": rid, **info}
        for rid, info in kb.get("factors", {}).items()
        if info.get("PassGates") and info.get("formula")
    ]
    if not passing:
        return ""
    passing.sort(key=lambda x: x.get("Score", 0), reverse=True)
    selected_rows, retrieval_mode = _query_guided_passing_factors(
        passing,
        query_text=query_text,
        semantic_k=min(max_factors, max(1, semantic_k)),
        anchor_count=min(max_factors, max(1, anchor_count)),
        max_factors=max_factors,
    )

    lines: List[str] = [
        "## Verified Passing Factors (RAG)",
        "These formulas have ALL passed IC>0.6, IR>2.5, TVR<330 on full 2022-2024 history.",
        (
            "Semantic matches for the current mechanism hypothesis, plus a couple of global anchors."
            if retrieval_mode == "semantic"
            else (
                "Query-guided fallback retrieval is active: lexical similarity over passing-factor research text, plus score anchors."
                if retrieval_mode == "lexical"
                else "Study their structure: price signal core * sigmoid/tanh volume gate, strong outer smoother."
            )
        ),
        (
            "Query-guided retrieval is active here: prioritize the semantic matches first, then use anchors to stay grounded."
            if include_formulas and retrieval_mode == "semantic"
            else (
                "Embedding endpoint is unavailable or incomplete; lexical retrieval keeps the prompt query-aware until embeddings recover."
                if include_formulas and retrieval_mode == "lexical"
                else (
                    "Use these only as performance anchors and exclusion references; do not reuse their operator skeletons."
                    if not include_formulas
                    else "These are the current highest-signal passing references."
                )
            )
        ),
        "",
    ]
    total = sum(len(l) + 1 for l in lines)
    for i, (selection_kind, f, similarity) in enumerate(selected_rows):
        ic = float(f.get("IC", 0) or 0)
        ir = float(f.get("IR", 0) or 0)
        tvr = float(f.get("tvr", 0) or 0)
        score = float(f.get("Score", 0) or 0)
        formula = f.get("formula", "")
        run_id = f.get("run_id", "")
        prefix = selection_kind
        if include_formulas:
            line = (
                f"[{i+1}] {prefix} run_id={run_id} IC={ic:.3f} IR={ir:.2f} TVR={tvr:.0f} Score={score:.0f}"
                + (f" sim={similarity:.3f}" if similarity is not None else "")
                + "\n"
                f"    formula: {formula}"
            )
        else:
            source = f.get("inspiration_source_type", "unknown")
            fingerprint = formula_structural_fingerprint(formula)[:120]
            line = (
                f"[{i+1}] {prefix} run_id={run_id} source={source} motif={formula_motif(formula)} "
                f"IC={ic:.3f} IR={ir:.2f} TVR={tvr:.0f} Score={score:.0f}"
                + (f" sim={similarity:.3f}" if similarity is not None else "")
                + "\n"
                f"    avoid-skeleton: {fingerprint}"
            )
        if total + len(line) + 2 > max_chars:
            break
        lines.append(line)
        total += len(line) + 2

    if include_template:
        lines += [
            "",
            "## Structural Pattern (extracted from ALL passing factors above)",
            "neg(  outer_smoother(  cs_zscore(  ts_mean(  price_core * sigmoid_volume_gate, 3-4  )  )  )  )",
            "Where:",
            "  outer_smoother = ts_decay_linear(x,15) OR ts_ema(x,10-12) OR ts_mean(x,10)",
            "  price_core = tanh(safe_div(close-lag_baseline, ts_median(range,20))) OR ts_minmax_norm(close,20) OR ts_pct_change(close,4)",
            "  sigmoid_volume_gate = sigmoid(ts_zscore(safe_div(volume, ts_median(volume,20)), 12-20))",
            "  Often: mean_of(vol_gate, trade_count_gate) for dual confirmation",
            "CRITICAL: outer_smoother window >= 10 is MANDATORY — window < 10 causes TVR > 500.",
        ]
    else:
        lines += [
            "",
            "Exploration note: keep the turnover-safe outer smoothing lesson, but choose a different market mechanism and operator skeleton.",
        ]
    return "\n".join(lines)


def compose_failure_pattern_summary(recent_n: int = 60, max_chars: int = 900) -> str:
    """
    Summarise dominant failure modes from the most recent N tested factors.
    """
    kb = _load()
    all_factors = [
        {"run_id": rid, **info}
        for rid, info in kb.get("factors", {}).items()
        if info.get("status") in ("ok", "screened_out") and info.get("formula")
    ]
    all_factors.sort(key=lambda x: x.get("created_at", ""))
    recent = all_factors[-recent_n:]
    non_passing = [f for f in recent if not f.get("PassGates")]
    if not non_passing:
        return ""

    n = len(non_passing)
    tvr_fails   = sum(1 for f in non_passing if float(f.get("tvr", 0) or 0) > 330)
    low_ic      = sum(1 for f in non_passing if abs(float(f.get("IC", 0) or 0)) < 0.3)
    neg_ic      = sum(1 for f in non_passing if float(f.get("IC", 0) or 0) < 0)
    low_ir      = sum(1 for f in non_passing if float(f.get("IR", 0) or 0) < 1.0)

    lines = [f"## Recent Failure Analysis (last {n} non-passing factors out of {len(recent)} tested)"]
    lines.append(f"  TVR > 330: {tvr_fails}/{n} ({100*tvr_fails//max(n,1)}%) — smoother window too short")
    lines.append(f"  |IC| < 0.3 (no signal): {low_ic}/{n} ({100*low_ic//max(n,1)}%) — wrong mechanism or noisy formula")
    lines.append(f"  IC < 0 (inverted): {neg_ic}/{n} ({100*neg_ic//max(n,1)}%) — missing neg() wrapper or flipped logic")
    lines.append(f"  IR < 1 (unstable): {low_ir}/{n} ({100*low_ir//max(n,1)}%) — signal too noisy across time")
    lines.append("  Fix priorities: (1) wrap outer layer in ts_decay_linear/ts_ema window>=10, "
                 "(2) ensure neg() wrapper for reversal, (3) use median/quantile baselines not raw series")
    text = "\n".join(lines)
    return text[:max_chars]


def get_factor(run_id: str) -> Optional[Dict[str, Any]]:
    kb = _load()
    info = kb.get("factors", {}).get(run_id)
    if info is None:
        return None
    return {"run_id": run_id, **info}
