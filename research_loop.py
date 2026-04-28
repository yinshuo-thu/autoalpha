"""
research_loop.py — Autonomous Alpha Factor Research Engineer

Orchestrates the continuous EA-based generation and testing of factors.
Loops infinitely until max_iters reached.
Real-data only. No mock functionality allowed.
"""
import os
import sys
import json
import time
import argparse
from datetime import datetime

# Local modules
from prepare_data import DataHub
from leaderboard import load_leaderboard, get_all_factors, add_or_update_factor
from core.factor_experience import append_factor_experience, build_factor_experience_record
from factor_idea_generator import generate_ideas_from_parents, generate_ideas_with_llm, generate_ideas_with_prompt
from quick_test import quick_test
from paths import LLM_MINING_JSONL, RESEARCH_ARTIFACTS_ROOT
from core.llm_mining_log import append_llm_mining_record


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"[{ts}] {msg}", flush=True)

def run_research_loop(max_iters=50, resume=True, seed_prompt="", batch_size=4):
    log("=== AutoAlpha Autonomous Research Loop ===")

    use_mock = (
        os.environ.get("AUTOALPHA_MOCK") == "1"
        or os.environ.get("ALPHACLAW_MOCK") == "1"
    )
    _llm = os.environ.get("AUTOALPHA_USE_LLM")
    if _llm is None:
        _llm = os.environ.get("ALPHACLAW_USE_LLM", "1")
    use_llm = _llm == "1"
    
    # 1. Initialize Real/Mock Data Hub
    mode = "MOCK" if use_mock else "REAL"
    log(f"Loading DataHub ({mode} 3-year data with 15-minute bars)...")
    try:
        data_hub = DataHub(use_mock=use_mock)
        log(f"DataHub ready. Loaded {len(data_hub.pv_15m):,} rows.")
    except Exception as e:
        log(f"FATAL: DataHub initialization failed: {e}")
        return

    # 2. Check resume state
    lb = load_leaderboard()
    factors = lb.get('factors', [])
    log(f"Leaderboard loaded. Current library size: {len(factors)} factors.")
    
    iters_run = 0
    total_iters = max_iters

    # Infinite Loop Engine
    while iters_run < total_iters:
        iters_run += 1
        log(f"\n--- Iteration {iters_run}/{total_iters} ---")
        
        # A. Filter top parents from leaderboard (EA Selection)
        # Select best 5 factors that pass gates
        top_parents = [f for f in factors if f.get('PassGates', False)]
        top_parents = sorted(top_parents, key=lambda x: x.get('Score', 0), reverse=True)[:5]
        
        # B. Generation (LLM or EA)
        if iters_run <= max_iters // 3:
            depth_hint = "2 (simple, robust)"
        elif iters_run <= 2 * max_iters // 3:
            depth_hint = "3 (moderate complexity)"
        else:
            depth_hint = "3-4 (professional, sophisticated)"

        if seed_prompt and iters_run == 1:
            log(f"Agent generating {batch_size} prompt-conditioned ideas with depth_hint: {depth_hint}...")
            candidate_ideas = generate_ideas_with_prompt(seed_prompt, top_parents, num_ideas=batch_size, depth_hint=depth_hint)
        elif use_llm:
            log(f"Agent generating {batch_size} new ideas via LLM with depth_hint: {depth_hint}...")
            candidate_ideas = generate_ideas_with_llm(top_parents, num_ideas=batch_size, depth_hint=depth_hint)
        else:
            log(f"Agent generating {batch_size} new ideas via EA...")
            candidate_ideas = generate_ideas_from_parents(top_parents, num_ideas=batch_size)

        gen_mode = (
            "prompt"
            if (seed_prompt and iters_run == 1)
            else ("llm" if use_llm else "ea")
        )
        log(
            f"[Batch] mode={gen_mode} use_llm_env={use_llm} "
            f"ideas={len(candidate_ideas)} (detail also in {LLM_MINING_JSONL})"
        )
        for idx, idea in enumerate(candidate_ideas):
            src = idea.get("source", "?")
            log(f"[Idea {idx + 1}/{len(candidate_ideas)}] source={src}")
            log(f"[Idea {idx + 1}] formula: {idea.get('formula', '')}")
            note = (idea.get("rationale") or idea.get("thought_process") or "").strip()
            if note:
                one_line = " ".join(note.split())
                if len(one_line) > 720:
                    one_line = one_line[:720] + "..."
                log(f"[Idea {idx + 1}] AI_note: {one_line}")
        try:
            append_llm_mining_record(
                {
                    "event": "research_loop_batch",
                    "iteration": iters_run,
                    "max_iters": total_iters,
                    "generation_mode": gen_mode,
                    "seed_prompt_excerpt": (seed_prompt or "")[:600],
                    "ideas": [
                        {
                            "formula": i.get("formula"),
                            "rationale": (i.get("rationale") or i.get("thought_process") or "")[:1200],
                            "source": i.get("source"),
                        }
                        for i in candidate_ideas
                    ],
                }
            )
        except Exception as _e:
            log(f"[MiningLog] append failed: {_e}")

        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        results = []
        with ThreadPoolExecutor(max_workers=batch_size) as pool:
            futures = {}
            for i, idea in enumerate(candidate_ideas):
                f_name = f"evol_factor_{len(factors) + i + 1:04d}"
                idea['factor_name'] = f_name
                log(f"[Agent] Proposing {f_name}: {idea['formula']}")
                rationale = idea.get("rationale") or idea.get("thought_process", "")
                futures[pool.submit(quick_test, idea["formula"], f_name, None, rationale, data_hub)] = idea
                
            for future in as_completed(futures):
                idea = futures[future]
                try:
                    res = future.result()
                    res["rationale"] = idea.get("rationale") or idea.get("thought_process", "")
                    res["_source"] = idea.get("source", gen_mode)
                    res["_generation_mode"] = gen_mode
                    res["_depth_hint"] = depth_hint
                    results.append(res)
                except Exception as e:
                    log(f"[Eval] Critical error on {idea['factor_name']}: {e}")
                    
        for result in results:
            factor_name = result.get('factor_name', 'unknown')
            if result['status'] == 'success':
                log(f"[Eval] {factor_name} Success! IC: {result['IC']:.4f} | IR: {result['IR']:.2f} | TVR: {result['Turnover']:.1f} | Score: {result.get('Score',0):.2f}")
                log(f"[Gates] {'✅ PASS' if result['PassGates'] else '❌ REJECT'} | Class: {result['classification']}")
            else:
                log(f"[Eval] {factor_name} Failed | Reason: {result.get('reason', 'Unknown')}")
                
            metrics_entry = {
                'factor_name': result.get('factor_name', factor_name),
                'family': 'Evolutionary',
                'formula': result.get('formula', ''),
                'IC': result.get('IC', 0),
                'rank_ic': result.get('rank_ic', 0),
                'IR': result.get('IR', 0),
                'Turnover': result.get('Turnover', 0),
                'Score': result.get('Score', 0),
                'PassGates': result.get('PassGates', False),
                'classification': result.get('classification', 'Drop'),
                'cover_all': result.get('cover_all', 0),
                'maxx': result.get('maxx', 0),
                'minn': result.get('minn', 0),
                'stability_score': result.get('stability_score', 0),
                'missing_days': result.get('missing_days', 0),
                'reason': result.get('reason', ''),
                'recommendation': result.get('recommendation', ''),
                'submission_ready_flag': result.get('submission_ready_flag', False),
                'submission_path': result.get('submission_path', ''),
                'submission_dir': result.get('submission_dir', ''),
                'metadata_path': result.get('metadata_path', ''),
                'gates_detail': result.get('gates_detail', {}),
                'sanity_report': result.get('sanity_report', {}),
                'score_formula': result.get('score_formula', ''),
                'score_components': result.get('score_components', {}),
            }
            add_or_update_factor(metrics_entry)

            try:
                append_factor_experience(
                    build_factor_experience_record(
                        result=result,
                        prompt=seed_prompt if gen_mode == "prompt" else "",
                        rationale=result.get("rationale", ""),
                        source=result.get("_source", gen_mode),
                        generation_mode=result.get("_generation_mode", gen_mode),
                        depth_hint=result.get("_depth_hint", depth_hint),
                        parents=top_parents,
                    )
                )
            except Exception as exp_err:
                log(f"[Experience] append failed: {exp_err}")
            
            if result['status'] == 'success':
                artifact_dir = RESEARCH_ARTIFACTS_ROOT
                os.makedirs(artifact_dir, exist_ok=True)
                with open(os.path.join(artifact_dir, f"{factor_name}.json"), 'w') as f:
                    json.dump(result, f, indent=2, default=str)
                    
        factors = get_all_factors()

    log("\n=== Research Loop Terminated ===")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-iters", type=int, default=10, help="Max iterations")
    parser.add_argument("--seed-prompt", type=str, default="", help="Optional prompt used to seed the first iteration")
    parser.add_argument("--batch-size", type=int, default=4, help="Ideas generated per iteration")
    args = parser.parse_args()
    
    run_research_loop(
        max_iters=args.max_iters,
        resume=True,
        seed_prompt=args.seed_prompt.strip(),
        batch_size=max(1, args.batch_size),
    )
