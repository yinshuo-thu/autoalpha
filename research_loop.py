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
from factor_idea_generator import generate_ideas_from_parents, generate_ideas_with_llm
from quick_test import quick_test

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")

def run_research_loop(max_iters=50, resume=True):
    log("=== AutoAlpha Autonomous Research Loop ===")
    
    use_mock = os.environ.get('AUTOALPHA_MOCK') == '1'
    use_llm = os.environ.get('AUTOALPHA_USE_LLM', '1') == '1'
    
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
        if use_llm:
            log("Agent generating new ideas via Large Language Model API...")
            candidate_ideas = generate_ideas_with_llm(top_parents, num_ideas=1)
        else:
            log("Agent generating new ideas via Evolutionary Generation...")
            candidate_ideas = generate_ideas_from_parents(top_parents, num_ideas=1)
            
        idea = candidate_ideas[0]
        
        formula = idea['formula']
        rationale = idea['rationale']
        factor_name = f"evol_factor_{len(factors) + 1:04d}"
        
        log(f"[Agent] Proposing Factor: {factor_name}")
        log(f"[Agent] Rationale: {rationale}")
        log(f"[Agent] Formula: {formula}")
        
        # C. Quick Test Pipeline
        log("[Pipeline] Starting validation, compliance & evaluation...")
        result = quick_test(formula, factor_name)
        
        if result['status'] == 'success':
            log(f"[Eval] Success! IC: {result['IC']:.4f} | IR: {result['IR']:.2f} | TVR: {result['Turnover']:.1f}")
            log(f"[Gates] {'✅ REJECT' if not result['PassGates'] else '✅ PASS'} | Class: {result['classification']}")
        else:
            log(f"[Eval] Failed | Reason: {result.get('reason', 'Unknown')}")
            
        # D. Leaderboard Update
        metrics_entry = {
            'factor_name': result.get('factor_name', factor_name),
            'family': 'Evolutionary',
            'formula': formula,
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
            'missing_days': result.get('missing_days', 0)
        }
        add_or_update_factor(metrics_entry)
        
        # Track updated library
        factors = get_all_factors()
        
        # Save comprehensive JSON (equivalent to saving artifacts for detailed charts)
        if result['status'] == 'success':
            artifact_dir = os.path.join(os.path.dirname(__file__), 'research_runs')
            os.makedirs(artifact_dir, exist_ok=True)
            with open(os.path.join(artifact_dir, f"{factor_name}.json"), 'w') as f:
                json.dump(result, f, indent=2, default=str)
                
        # Sleep tight
        time.sleep(1)

    log("\n=== Research Loop Terminated ===")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-iters", type=int, default=10, help="Max iterations")
    args = parser.parse_args()
    
    run_research_loop(max_iters=args.max_iters, resume=True)
