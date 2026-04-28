"""
Submission identity: alpha_###_slug basenames, sequence counter, Feishu de-dupe by formula hash.
Registry updates only apply to submission_ready (_y) exports.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any, Dict, Tuple

STATE_FILENAME = ".submission_state.json"


def _norm_formula(formula: str) -> str:
    return re.sub(r"\s+", "", (formula or "").strip())


def formula_fingerprint(formula: str) -> str:
    return hashlib.sha256(_norm_formula(formula).encode("utf-8")).hexdigest()


def formula_to_slug(formula: str, max_len: int = 26) -> str:
    """Short ASCII slug from formula for folder names (no 'submit' prefix)."""
    s = _norm_formula(formula).lower()
    repl = (
        ("close_trade_px", "cpx"),
        ("open_trade_px", "opx"),
        ("high_trade_px", "hpx"),
        ("low_trade_px", "lpx"),
        ("trade_count", "tcnt"),
        ("volume", "vol"),
        ("dvolume", "dvol"),
        ("vwap", "vwap"),
        ("ts_zscore", "tz"),
        ("ts_rank", "tr"),
        ("ts_mean", "tm"),
        ("ts_std", "tstd"),
        ("ts_decay_linear", "tdl"),
        ("ts_corr", "tcorr"),
        ("cs_rank", "cr"),
        ("cs_zscore", "cz"),
        ("cs_demean", "cdm"),
        ("delta", "d"),
        ("lag", "lg"),
        ("safe_div", "div"),
        ("signed_power", "sp"),
    )
    for a, b in repl:
        s = s.replace(a, b)
    s = re.sub(r"[^a-z0-9_+-]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        return "f_" + hashlib.sha1((formula or "").encode()).hexdigest()[:10]
    if len(s) > max_len:
        s = s[: max_len - 7] + "_" + hashlib.sha1((formula or "").encode()).hexdigest()[:6]
    return s


def _load_state(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_state(path: str, state: Dict[str, Any]) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _parse_ts_from_dirname(dirname: str) -> str:
    """支持目录后缀时间戳：YYYYMMDD_HHMM（分钟）或 YYYYMMDD_HHMMSS（旧版秒）。"""
    m = re.search(r"_(\d{8})_(\d{4,6})_[yn]$", dirname)
    if not m:
        return dirname
    d, t = m.group(1), m.group(2)
    if len(t) == 4:
        t = t + "00"
    return f"{d}_{t}"


def bootstrap_state_from_submit_dir(submit_root: str) -> Dict[str, Any]:
    """Scan submit/*_y/ folders with metadata; build by_hash and next_seq."""
    path = os.path.join(submit_root, STATE_FILENAME)
    state = _load_state(path)
    if state.get("by_hash") and state.get("next_seq", 0) >= 1:
        return state

    by_hash: Dict[str, Any] = {}
    rows = []
    if not os.path.isdir(submit_root):
        return {"next_seq": 1, "by_hash": {}}

    for name in os.listdir(submit_root):
        if name == "backup" or name.startswith("."):
            continue
        if not name.endswith("_y") or not os.path.isdir(os.path.join(submit_root, name)):
            continue
        meta = None
        dpath = os.path.join(submit_root, name)
        for fn in os.listdir(dpath):
            if fn.endswith("_metadata.json"):
                try:
                    with open(os.path.join(dpath, fn), "r", encoding="utf-8") as f:
                        meta = json.load(f)
                except Exception:
                    pass
                break
        if not meta:
            continue
        formula = meta.get("formula") or ""
        fp = formula_fingerprint(formula)
        ts = _parse_ts_from_dirname(name)
        rows.append((ts, fp, formula, name))

    rows.sort(key=lambda x: x[0])
    seq = 0
    for _ts, fp, formula, dirname in rows:
        seq += 1
        slug = formula_to_slug(formula)
        by_hash[fp] = {
            "seq": seq,
            "slug": slug,
            "notified": True,
            "last_dir": dirname,
            "formula": _norm_formula(formula),
        }

    next_seq = seq + 1 if seq else 1
    state = {"next_seq": next_seq, "by_hash": by_hash}
    _save_state(path, state)
    return state


def sanitize_display_name(factor_name: str) -> str:
    """Strip legacy 'submit_' prefix for UI."""
    fn = (factor_name or "").strip()
    if fn.lower().startswith("submit_"):
        return fn[7:]
    return fn


def resolve_ready_submission(
    submit_root: str,
    formula: str,
) -> Tuple[str, int, str, bool]:
    """
    For submission_ready exports only.

    Returns:
      storage_basename — e.g. alpha_002_cr_tz_cpx15
      sequence — N for 第 N 个
      display_title — same as storage_basename (no submit_)
      skip_feishu — True if this formula hash was already Feishu-notified
    """
    state = bootstrap_state_from_submit_dir(submit_root)
    by_hash = state.setdefault("by_hash", {})
    next_seq = int(state.get("next_seq", 1))
    path = os.path.join(submit_root, STATE_FILENAME)

    fp = formula_fingerprint(formula)
    slug = formula_to_slug(formula)

    if fp in by_hash:
        entry = by_hash[fp]
        seq = int(entry["seq"])
        use_slug = entry.get("slug") or slug
        storage = f"alpha_{seq:03d}_{use_slug}"
        entry["slug"] = use_slug
        skip_feishu = bool(entry.get("notified", False))
        return storage, seq, storage, skip_feishu

    seq = next_seq
    storage = f"alpha_{seq:03d}_{slug}"
    by_hash[fp] = {
        "seq": seq,
        "slug": slug,
        "notified": False,
        "last_dir": None,
        "formula": _norm_formula(formula),
    }
    state["next_seq"] = seq + 1
    _save_state(path, state)
    return storage, seq, storage, False


def mark_formula_notified(submit_root: str, formula: str) -> None:
    path = os.path.join(submit_root, STATE_FILENAME)
    state = _load_state(path)
    if not state.get("by_hash"):
        state = bootstrap_state_from_submit_dir(submit_root)
    by_hash = state.setdefault("by_hash", {})
    fp = formula_fingerprint(formula)
    if fp in by_hash:
        by_hash[fp]["notified"] = True
    _save_state(path, state)
