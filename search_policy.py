from __future__ import annotations

import math
from typing import Any


def _bucket(stats: dict[str, dict[str, int]], key: str) -> dict[str, int]:
    raw = stats.get(key) or {}
    return {k: int(v or 0) for k, v in raw.items()}


def choose_archetype_ucb(
    options: list[tuple[str, str]],
    stats: dict[str, dict[str, int]],
    *,
    idea_index: int = 0,
    exploration: bool = False,
) -> tuple[str, str]:
    if not options:
        return "", ""
    total_attempts = 1 + sum(sum(_bucket(stats, key).values()) for key, _ in options)
    scored: list[tuple[float, int, tuple[str, str]]] = []
    for idx, option in enumerate(options):
        key, _ = option
        bucket = _bucket(stats, key)
        attempts = max(1, sum(bucket.values()))
        wins = bucket.get("passing", 0)
        screened = bucket.get("screened_out", 0)
        severe = bucket.get("invalid", 0) + bucket.get("compute_error", 0)
        reward = wins / attempts - 0.25 * screened / attempts - 0.35 * severe / attempts
        bonus = math.sqrt(2.0 * math.log(max(total_attempts, 2)) / attempts)
        round_robin = 0.03 if idx == idea_index % len(options) else 0.0
        explore_bonus = 0.12 / attempts if exploration else 0.0
        scored.append((reward + bonus + round_robin + explore_bonus, -attempts, option))
    scored.sort(key=lambda row: row[:2], reverse=True)
    return scored[0][2]


def choose_mutation_type(
    *,
    exploration: bool = False,
    source_type: str = "",
    recent_failures: str = "",
) -> str:
    text = f"{source_type} {recent_failures}".lower()
    if exploration:
        return "orthogonal_mechanism"
    if "correlation" in text or "duplicate" in text:
        return "decorrelate_structure"
    if "tvr" in text or "turnover" in text:
        return "turnover_safe_smoothing"
    if "ir" in text or "unstable" in text:
        return "stability_filter"
    if source_type in {"paper", "llm", "manual"}:
        return f"source_grounded_{source_type}"
    return "balanced_iteration"

