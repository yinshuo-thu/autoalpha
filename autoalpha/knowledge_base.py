"""
autoalpha/knowledge_base.py

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

import json
import re
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

KB_PATH = Path(__file__).resolve().parent / "knowledge.json"
SUBMIT_DIR = Path(__file__).resolve().parent / "submit"

_EMPTY_KB: Dict[str, Any] = {
    "version": 2,
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
}

_MOTIF_TOKENS = [
    "volume", "trade_count", "dvolume", "vwap",
    "close_trade_px", "high_trade_px", "low_trade_px",
    "ts_mean", "ts_zscore", "ts_rank", "ts_decay_linear",
    "safe_div", "cs_rank", "cs_zscore", "delta", "lag",
    "ts_corr", "ts_cov",
]

_FIELD_NAMES = [
    "open_trade_px", "high_trade_px", "low_trade_px", "close_trade_px",
    "trade_count", "volume", "dvolume", "vwap",
]

_ALL_OPS = [
    "ts_decay_linear", "ts_corr", "ts_cov", "ts_zscore", "ts_rank",
    "ts_mean", "ts_std", "ts_sum", "ts_max", "ts_min",
    "lag", "delta", "cs_rank", "cs_zscore", "cs_demean",
    "safe_div", "signed_power", "abs", "sign", "neg", "log", "sqrt",
]


# ─────────────────────────────────────────────────────────────────────────────
# Structural fingerprint (Hubble family-aware deduplication)
# ─────────────────────────────────────────────────────────────────────────────

def formula_structural_fingerprint(formula: str) -> str:
    """
    Replace field names and numeric literals with placeholders so that two
    formulas with the same operator skeleton but different fields/lookbacks
    map to the same fingerprint (same "family").

    Example:
        ts_decay_linear(cs_zscore((close_trade_px-vwap)/open_trade_px),6)
        ts_decay_linear(cs_zscore(volume/dvolume),10)
    Both → "ts_decay_linear(cs_zscore((_f-_f)/_f),_n)"
    """
    s = re.sub(r"\s+", "", formula or "").lower()
    for field in sorted(_FIELD_NAMES, key=len, reverse=True):
        s = s.replace(field, "_f")
    s = re.sub(r"\b\d+\b", "_n", s)
    return s


def _ops_in_formula(formula: str) -> list[str]:
    """Return sorted list of operator names found in a formula."""
    lowered = re.sub(r"\s+", "", formula or "").lower()
    return sorted({op for op in _ALL_OPS if op in lowered})


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
    """Copy passing factor parquet into autoalpha/submit and return metadata fields."""
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
        parent_gens = [
            factors.get(pid, {}).get("generation", 0)
            for pid in parent_run_ids
            if pid in factors
        ]
        if parent_gens:
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
        "parent_run_ids": parent_run_ids or [],
        "generation": generation,
        "created_at": datetime.now().isoformat(),
        "parquet_path": result.get("parquet_path", ""),
        "eval_days": int(result.get("eval_days", 0) or 0),
        "research_path": result.get("research_path", ""),
        "fingerprint": formula_structural_fingerprint(result.get("formula", "")),
    }

    for key in ("live_test_result", "live_submitted", "live_result_updated_at"):
        if key in existing:
            entry[key] = existing[key]

    entry.update(_copy_submit_artifact(run_id, entry))

    factors[run_id] = entry
    _update_skill_memory(kb, result)
    _save(kb)


def sync_submit_artifacts() -> Dict[str, Any]:
    """Backfill autoalpha/submit with all currently passing parquet factors."""
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
    """Return top-K passing factors to use as parents for next round."""
    kb = _load()
    all_factors = [
        {"run_id": rid, **info}
        for rid, info in kb.get("factors", {}).items()
        if info.get("status") == "ok"
    ]

    passing = [f for f in all_factors if f.get("PassGates") and f.get("IC", 0) >= min_ic]
    if not passing:
        passing = [f for f in all_factors if f.get("IC", 0) >= min_ic]
    if not passing:
        passing = sorted(all_factors, key=lambda x: x.get("IC", 0), reverse=True)

    passing.sort(key=lambda x: (x.get("Score", 0), x.get("IC", 0)), reverse=True)

    if len(passing) < k:
        try:
            from leaderboard import load_leaderboard

            seen_formulas = {item.get("formula", "") for item in passing if item.get("formula")}
            for factor in sorted(
                load_leaderboard().get("factors", []),
                key=lambda item: (item.get("PassGates", False), item.get("Score", 0), item.get("IC", 0)),
                reverse=True,
            ):
                if not factor.get("formula"):
                    continue
                if factor.get("formula") in seen_formulas:
                    continue
                if not (factor.get("PassGates") or factor.get("submission_ready_flag")):
                    continue
                passing.append(
                    {
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
                    }
                )
                seen_formulas.add(factor.get("formula", ""))
                if len(passing) >= k:
                    break
        except Exception:
            pass

    return passing[:k]


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


def get_factor(run_id: str) -> Optional[Dict[str, Any]]:
    kb = _load()
    info = kb.get("factors", {}).get(run_id)
    if info is None:
        return None
    return {"run_id": run_id, **info}
