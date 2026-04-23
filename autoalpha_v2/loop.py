"""
autoalpha_v2/loop.py

Multi-round iterative factor mining loop.
Inspired by karpathy/autoresearch's ratchet pattern:
  - Each round, top passing factors feed the next LLM call as "parents"
  - All results (pass or fail) are persisted in knowledge_base.py
  - Loop logs to autoalpha_v2/loop.log for frontend streaming

Usage:
    python autoalpha_v2/loop.py --rounds 5 --ideas 3 --days 0
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from autoalpha_v2 import knowledge_base as kb
from autoalpha_v2.error_utils import AutoAlphaRuntimeError, humanize_error
from autoalpha_v2.pipeline import run
from autoalpha_v2.idea_cache import get_default_cache
from autoalpha_v2.inspiration_fetcher import start_background_fetcher
from autoalpha_v2.llm_client import summarize_generation_experience
from core.feishu_bot import FeishuNotifier
from runtime_config import get_llm_routing, load_runtime_config

LOG_PATH = Path(__file__).resolve().parent / "loop.log"
_feishu = FeishuNotifier(
    webhook_url="https://open.feishu.cn/open-apis/bot/v2/hook/b4cd233b-5185-4135-8a08-4ffda6305877"
)

_BILLING_ENDPOINTS = {
    "subscription": [
        "https://vip.aipro.love/v1/dashboard/billing/subscription",
        "http://free.aipro.love/v1/dashboard/billing/subscription",
    ],
    "usage": [
        "https://vip.aipro.love/v1/dashboard/billing/usage?start_date=2020-01-01&end_date=2030-12-31",
        "http://free.aipro.love/v1/dashboard/billing/usage?start_date=2020-01-01&end_date=2030-12-31",
    ],
}


def _log(msg: str) -> None:
    """Write timestamped message to both stdout and loop.log."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def _notify_unexpected_loop_error(round_i: int, error: Exception) -> None:
    """Avoid duplicate alerts for already-humanized pipeline errors."""
    if isinstance(error, AutoAlphaRuntimeError):
        return
    friendly, suggestion, error_code, raw = humanize_error(error)
    try:
        _feishu.send_error_notification(
            title="AutoAlpha 挖掘循环异常",
            summary=friendly,
            stage=f"Loop Round {round_i}",
            error_code=error_code,
            suggestion=suggestion,
            raw_detail=raw,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
    except Exception:
        pass


def _fetch_billing(kind: str, headers: dict[str, str]) -> dict:
    import requests

    for url in _BILLING_ENDPOINTS.get(kind, []):
        try:
            resp = requests.get(url, headers=headers, timeout=10, verify=False)
            if resp.status_code >= 400:
                continue
            data = resp.json()
            if isinstance(data, dict) and not data.get("error"):
                return data
        except Exception:
            continue
    return {}


def _quota_snapshot() -> dict[str, float | str]:
    routing = get_llm_routing()
    api_key = routing.get("api_key", "")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    subscription = _fetch_billing("subscription", headers)
    usage = _fetch_billing("usage", headers)
    if not subscription and not usage:
        return {"total": 0.0, "used": 0.0, "remaining": 0.0, "known": 0.0}
    total_source = float(subscription.get("hard_limit_usd", 0) or 0)
    used_source = float(usage.get("total_usage", 0) or 0) / 100.0
    cfg = load_runtime_config()
    fx_rate = float(cfg.get("AUTOALPHA_QUOTA_DISPLAY_FX", "7.3") or 7.3)
    fx_rate = fx_rate if fx_rate > 0 else 1.0
    total = total_source / fx_rate
    used = used_source / fx_rate
    remaining = max(0.0, total - used)
    return {"total": total, "used": used, "remaining": remaining, "known": 1.0 if total > 0 else 0.0}


def _notify_quota_exhausted(snapshot: dict[str, float | str], stage: str) -> None:
    try:
        _feishu.send_error_notification(
            title="AutoAlpha 额度耗尽，挖掘已停止",
            summary=(
                f"剩余额度 {float(snapshot.get('remaining', 0) or 0):.2f}，"
                f"已用 {float(snapshot.get('used', 0) or 0):.2f} / "
                f"{float(snapshot.get('total', 0) or 0):.2f}。"
            ),
            stage=stage,
            error_code="quota_exhausted",
            suggestion="补充额度或更换可用 API Key 后再启动挖掘。",
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
    except Exception:
        pass


def _run_model_lab_snapshot(cfg: dict, n_ideas_per_round: int, eval_days_count: int, target_valid_count: int, min_full_eval_days: int) -> None:
    try:
        from autoalpha_v2.rolling_model_lab import run_model_lab

        full_valid_count = len(kb.list_valid_factors(min_eval_days=min_full_eval_days))
        if full_valid_count <= 0:
            _log("[model-lab] Skip update: no full-window valid factors yet")
            return
        target_for_lab = max(1, min(target_valid_count or full_valid_count or 1, full_valid_count))
        _log(f"[model-lab] Updating rolling model lab with {target_for_lab} valid factor(s)")
        summary = run_model_lab(
            target_valid_count=target_for_lab,
            ideas_per_round=n_ideas_per_round,
            eval_days_count=eval_days_count,
            max_rounds=0,
            sleep_seconds=0.0,
            train_days=int(cfg.get("AUTOALPHA_ROLLING_TRAIN_DAYS", "126") or 126),
            test_days=int(cfg.get("AUTOALPHA_ROLLING_TEST_DAYS", "126") or 126),
            step_days=int(cfg.get("AUTOALPHA_ROLLING_STEP_DAYS", "126") or 126),
            allow_partial=True,
        )
        _log(f"[model-lab] Finished: best_model={summary.get('best_model')} selected={summary.get('selected_factor_count')}")
    except Exception as exc:
        _log(f"[WARN] Model lab update failed: {exc}")


def mining_loop(
    n_rounds: int = 5,
    n_ideas_per_round: int = 3,
    eval_days_count: int = 0,
    target_valid_count: int = 0,
    run_model_lab_on_finish: bool = False,
) -> None:
    """
    Run iterative factor mining loop with accumulated experience.

    Each round:
      1. Load top passing factors from KB as "parents" (context for LLM)
      2. Generate n_ideas_per_round new factors via LLM
      3. Validate → Compute → Evaluate each factor
      4. Persist ALL results to knowledge base
      5. Print round summary
    """
    cfg = load_runtime_config()
    source_config = str(cfg.get("AUTOALPHA_INSPIRATION_SOURCES", "paper,llm,future,manual"))
    source_set = {
        source.strip().lower()
        for source in source_config.split(",")
        if source.strip()
    }
    try:
        round_pause_sec = max(0.0, float(cfg.get("AUTOALPHA_ROUND_PAUSE_SEC", "1") or 1))
    except (TypeError, ValueError):
        round_pause_sec = 1.0
    try:
        min_full_eval_days = int(cfg.get("AUTOALPHA_MIN_FULL_EVAL_DAYS", "700") or 700)
    except (TypeError, ValueError):
        min_full_eval_days = 700
    try:
        cache_prefill = int(cfg.get("AUTOALPHA_CACHE_PREFILL", str(n_ideas_per_round * 2)) or n_ideas_per_round * 2)
    except (TypeError, ValueError):
        cache_prefill = n_ideas_per_round * 2
    try:
        inspiration_interval = int(cfg.get("AUTOALPHA_INSPIRATION_INTERVAL_SEC", "1800") or 1800)
    except (TypeError, ValueError):
        inspiration_interval = 1800

    # Start background inspiration fetcher (Paper + LLM brainstorm)
    try:
        start_background_fetcher(
            interval_seconds=inspiration_interval,
            llm_ideas=1 if "llm" in source_set else 0,
            paper_results=12,
            future_files=10 if "future" in source_set else 0,
            run_immediately=True,
        )
    except Exception as exc:
        _log(f"[WARN] Could not start inspiration fetcher: {exc}")

    idea_cache = get_default_cache()
    prompt_version = str(cfg.get("AUTOALPHA_PROMPT_VERSION", "v2-diversity-20260423") or "v2-diversity-20260423")
    stale_cache_count = 0
    stale_cache_error = ""
    try:
        stale_cache_count = idea_cache.clear_stale(prompt_version)
    except Exception as exc:
        stale_cache_error = str(exc)

    # Truncate log for this run
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_PATH, "w", encoding="utf-8") as f:
            f.write(f"[{datetime.now().isoformat()}] Loop started\n")
    except Exception:
        pass

    _log("=" * 60)
    _log(f"AutoAlpha Mining Loop")
    eval_label = "ALL" if eval_days_count <= 0 else str(eval_days_count)
    target_label = target_valid_count if target_valid_count > 0 else "off"
    _log(f"rounds={n_rounds}  ideas={n_ideas_per_round}  eval_days={eval_label}  target_valid={target_label}")
    _log(f"inspiration_sources={source_config}")
    if stale_cache_count:
        _log(f"[cache] Cleared {stale_cache_count} stale pending ideas for prompt_version={prompt_version}")
    if stale_cache_error:
        _log(f"[WARN] Could not clear stale idea cache: {stale_cache_error}")
    _log("=" * 60)

    init = kb.get_summary()
    _log(f"Knowledge base: {init['total_tested']} tested / {init['total_passing']} passing / best={init['best_score']:.2f}")
    last_model_lab_tested = int(init.get("total_tested", 0) or 0)

    round_i = 0
    while True:
        if n_rounds > 0 and round_i >= n_rounds:
            break
        try:
            quota = _quota_snapshot()
            if float(quota.get("known", 0) or 0) > 0 and float(quota.get("remaining", 0) or 0) <= 0.01:
                _log(
                    "[quota] Exhausted before next round — "
                    f"remaining={float(quota.get('remaining', 0) or 0):.2f}, "
                    f"used={float(quota.get('used', 0) or 0):.2f}/{float(quota.get('total', 0) or 0):.2f}. "
                    "Stopping mining loop."
                )
                _notify_quota_exhausted(quota, stage=f"Loop Round {round_i + 1}")
                break
        except Exception as exc:
            _log(f"[WARN] Quota preflight failed; continue cautiously: {exc}")
        round_i += 1
        _log(f"")
        round_label = f"{round_i}/{n_rounds}" if n_rounds > 0 else f"{round_i}/target"
        _log(f"--- ROUND {round_label} ---")

        # Select parents
        parents = kb.get_top_parents(k=4)
        parent_run_ids = [p["run_id"] for p in parents]
        if parents:
            _log(f"Parents ({len(parents)}): " + ", ".join(p.get("run_id", "") for p in parents))
            for p in parents:
                _log(f"  {p['run_id']}: IC={p.get('IC',0):.3f} Score={p.get('Score',0):.1f}")
        else:
            _log("No parents yet — generating from scratch")

        # Run pipeline
        round_start = time.time()
        try:
            results = run(
                n_ideas=n_ideas_per_round,
                eval_days_count=eval_days_count,
                parents=parents if parents else None,
            )
        except Exception as e:
            _log(f"[ERROR] Pipeline failed in round {round_i}: {e}")
            if isinstance(e, AutoAlphaRuntimeError) and e.error_code == "quota_exhausted":
                snapshot = {"remaining": 0.0, "used": 0.0, "total": 0.0}
                try:
                    snapshot = _quota_snapshot()
                except Exception:
                    pass
                _notify_quota_exhausted(snapshot, stage=f"Loop Round {round_i}")
                break
            _notify_unexpected_loop_error(round_i, e)
            results = []

        round_elapsed = time.time() - round_start

        # Persist to knowledge base
        for result in results:
            try:
                kb.add_factor(result, parent_run_ids=parent_run_ids)
            except Exception as e:
                _log(f"[WARN] Failed to save factor to KB: {e}")

        # Write a generation-level research memory note for newly completed generations only.
        # A generation is "new" when it appears for the first time this round and has no
        # existing summary yet — avoids re-calling the LLM every round for the same gen.
        touched_generations: set[int] = set()
        for result in results:
            try:
                saved = kb.get_factor(result.get("run_id", ""))
                if saved:
                    touched_generations.add(int(saved.get("generation", 0) or 0))
            except Exception:
                pass
        for generation in sorted(touched_generations):
            try:
                if kb.get_generation_experience(generation):
                    continue  # already summarised — skip to avoid redundant LLM calls
                payload = kb.build_generation_experience_payload(generation)
                if payload.get("total", 0) <= 0:
                    continue
                previous_context = kb.compose_recent_generation_experience_context(limit=3)
                markdown = summarize_generation_experience(payload, previous_context=previous_context)
                record = kb.save_generation_experience(generation, markdown, payload)
                _log(f"[experience] Generation {generation} summary saved → {record.get('relative_path')}")
            except Exception as exc:
                _log(f"[WARN] Generation experience summary failed for gen {generation}: {exc}")

        # Round summary
        passing = [r for r in results if r.get("PassGates")]
        _log(f"Round {round_i} done in {round_elapsed:.0f}s — tested={len(results)} passing={len(passing)}")
        if passing:
            best = max(passing, key=lambda r: r.get("Score", 0))
            _log(f"  Best: {best.get('run_id','')}  IC={best.get('IC',0):.3f}  "
                 f"IR={best.get('IR',0):.3f}  tvr={best.get('tvr',0):.0f}  score={best.get('Score',0):.1f}")

        summary = kb.get_summary()
        _log(f"Cumulative KB: {summary['total_tested']} tested / {summary['total_passing']} passing / best={summary['best_score']:.2f}")
        full_valid_count = len(kb.list_valid_factors(min_eval_days=min_full_eval_days))
        _log(f"Full-window valid factors (>= {min_full_eval_days} days): {full_valid_count}")
        tested_now = int(summary.get("total_tested", 0) or 0)
        model_lab_interval = int(cfg.get("AUTOALPHA_MODEL_LAB_INTERVAL_FACTORS", "10") or 10)
        if model_lab_interval > 0 and tested_now - last_model_lab_tested >= model_lab_interval:
            _run_model_lab_snapshot(cfg, n_ideas_per_round, eval_days_count, target_valid_count, min_full_eval_days)
            last_model_lab_tested = tested_now
        if target_valid_count > 0 and full_valid_count >= target_valid_count:
            _log(f"Target reached: {full_valid_count} valid full-window factors >= requested {target_valid_count}")
            break

        # Pre-fill idea cache for next round in background
        next_parents = kb.get_top_parents(k=4)
        try:
            idea_cache.start_fill(n=cache_prefill, parents=next_parents if next_parents else None)
            _log(f"[cache] Pre-filling {cache_prefill} ideas in background (current={idea_cache.size()})")
        except Exception as exc:
            _log(f"[WARN] Idea cache fill failed: {exc}")

        # Pause between rounds
        if n_rounds <= 0 or round_i < n_rounds:
            _log(f"Waiting {round_pause_sec:.1f}s before next round...")
            time.sleep(round_pause_sec)

    # Final summary
    final = kb.get_summary()
    _log("")
    _log("=" * 60)
    _log("LOOP COMPLETE")
    _log(f"Total tested: {final['total_tested']}")
    _log(f"Total passing: {final['total_passing']}")
    _log(f"Best score: {final['best_score']:.2f}")
    if final["top_factors"]:
        best = final["top_factors"][0]
        _log(f"Best factor: {best['run_id']}")
        _log(f"  IC={best['IC']:.3f}  IR={best['IR']:.3f}  Score={best['Score']:.1f}")
        _log(f"  {best['formula'][:80]}")
    _log("=" * 60)

    if run_model_lab_on_finish:
        _run_model_lab_snapshot(cfg, n_ideas_per_round, eval_days_count, target_valid_count, min_full_eval_days)


def main() -> int:
    parser = argparse.ArgumentParser(description="AutoAlpha iterative mining loop")
    parser.add_argument("--rounds", type=int, default=5, help="Number of rounds; 0 means keep going until target-valid")
    parser.add_argument("--ideas", type=int, default=3, help="Ideas per round")
    parser.add_argument("--days", type=int, default=0, help="Eval window (trading days); 0 means full history")
    parser.add_argument("--target-valid", type=int, default=0, help="Stop once this many full-window valid factors are reached")
    parser.add_argument("--run-model-lab", action="store_true", help="Run model lab after mining finishes")
    args = parser.parse_args()

    mining_loop(
        n_rounds=args.rounds,
        n_ideas_per_round=args.ideas,
        eval_days_count=args.days,
        target_valid_count=args.target_valid,
        run_model_lab_on_finish=args.run_model_lab,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
