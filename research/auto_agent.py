import os
import json
import time
import random
import yaml
import traceback
from datetime import datetime

# Direct API endpoint and credentials
API_BASE = "https://api.gmncode.cn/v1/chat/completions"
API_KEY = "sk-c85bf4ef4b1f52b4cb10e48cbf1fa8010865f3c19025147beb67102760d997e6"
MODEL_NAME = "gpt-4o"

SYSTEM_PROMPT = """You are a Quantitative Alpha Researcher. Your task is to generate Python-based factor formulas using our proprietary DSL.
Available inputs: open_trade_px, high_trade_px, low_trade_px, close_trade_px, trade_count, volume, dvolume, vwap
Available Time-Series operators: lag(x, d), delta(x, d), ts_mean(x, d), ts_std(x, d), ts_sum(x, d), ts_max(x, d), ts_min(x, d), ts_rank(x, d), ts_zscore(x, d), ts_decay_linear(x, d)
Available Cross-Sectional operators: cs_rank(x), cs_zscore(x), cs_demean(x)
Available Math operators: safe_div(a, b), signed_power(a, b), abs(x), sign(x), log(x)

Goal: Generate a formula that predicts the 'resp' label.
Gates to pass: IC > 0.6, IR > 2.5, Turnover < 400.

Output MUST be valid JSON matching this schema exactly:
{
  "thought_process": "Your internal logic finding the signal",
  "formula": "The mathematical formula string",
  "postprocess": "stable_low_turnover" or "aggressive_high_ic",
  "lookback_days": 20
}
DO NOT output markdown formatting like ```json, strictly the naked object.
"""

# ===== LOCAL OFFLINE FORMULA BANK (used when API is unreachable) =====
OFFLINE_FORMULAS = [
    {"thought": "Short-term mean reversion: if price deviates from its 10-bar mean, it will revert.",
     "formula": "cs_rank(-ts_zscore(close_trade_px, 10))", "post": "stable_low_turnover", "lb": 10},
    {"thought": "Volume-weighted momentum: higher volume confirms direction, rank cross-sectionally.",
     "formula": "cs_rank(ts_decay_linear(delta(close_trade_px, 5) * volume, 10))", "post": "stable_low_turnover", "lb": 15},
    {"thought": "VWAP deviation reversal: securities far below VWAP tend to bounce.",
     "formula": "cs_rank(ts_decay_linear(-(close_trade_px / vwap - 1), 10))", "post": "stable_low_turnover", "lb": 10},
    {"thought": "Multi-horizon mean reversion combining 5 and 20 bar lookbacks.",
     "formula": "cs_rank(-ts_zscore(close_trade_px, 5)) + cs_rank(-ts_zscore(close_trade_px, 20))", "post": "stable_low_turnover", "lb": 20},
    {"thought": "Intraday range compression signals breakout direction via volume.",
     "formula": "cs_rank(safe_div(ts_std(close_trade_px, 10), ts_mean(close_trade_px, 10)) * sign(delta(volume, 5)))", "post": "stable_low_turnover", "lb": 10},
    {"thought": "Trade count anomaly: unusual trade count relative to recent history signals informed trading.",
     "formula": "cs_rank(ts_zscore(trade_count, 15))", "post": "stable_low_turnover", "lb": 15},
    {"thought": "Price-volume divergence: rising price with falling volume signals weakness.",
     "formula": "cs_rank(-delta(close_trade_px, 5) * ts_rank(-volume, 10))", "post": "stable_low_turnover", "lb": 10},
    {"thought": "Smooth momentum using decay-weighted delta, penalizing erratic changes.",
     "formula": "cs_rank(ts_decay_linear(delta(vwap, 3), 15))", "post": "stable_low_turnover", "lb": 15},
    {"thought": "High-low range contraction predicts continuation, weighted by recent volume trend.",
     "formula": "cs_rank(safe_div(high_trade_px - low_trade_px, ts_mean(high_trade_px - low_trade_px, 10)) * sign(delta(dvolume, 5)))", "post": "aggressive_high_ic", "lb": 10},
    {"thought": "Cross-sectional ranking of smoothed close-to-open gap captures overnight drift.",
     "formula": "cs_rank(ts_decay_linear(close_trade_px - open_trade_px, 10))", "post": "stable_low_turnover", "lb": 10},
]

def log_agent_thought(generation_id, status, thought, metrics=None):
    log_file = r"e:\Scientech\outputs\autoresearch_log.json"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logs = []
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                logs = json.load(f)
        except: pass
        
    entry = {
        "timestamp": datetime.now().isoformat(),
        "generation": generation_id,
        "status": status,
        "thought": thought,
        "metrics": metrics
    }
    logs.insert(0, entry)
    
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(logs, f, indent=2, ensure_ascii=False)
    print(f"[AutoAgent] {status} -> {thought[:120]}...")

def query_llm(history):
    """Try the remote LLM API first. On ANY failure, fall back to the offline formula bank."""
    try:
        import requests
        headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": MODEL_NAME,
            "messages": history,
            "temperature": 0.7,
            "max_tokens": 1000
        }
        resp = requests.post(API_BASE, headers=headers, json=payload, verify=False, timeout=30)
        resp.raise_for_status()
        content = resp.json()['choices'][0]['message']['content'].strip()
        # Clean markdown fence if present
        if content.startswith('```json'):
            content = content.replace('```json', '', 1)
        if content.startswith('```'):
            content = content[3:]
        if content.endswith('```'):
            content = content[:-3]
        return json.loads(content.strip())
    except Exception as e:
        print(f"[AutoAgent] API unreachable ({type(e).__name__}: {e}). Using Offline Formula Bank...")
        # Pick a random formula from the bank
        pick = random.choice(OFFLINE_FORMULAS)
        return {
            "thought_process": f"[OFFLINE MODE] {pick['thought']}",
            "formula": pick['formula'],
            "postprocess": pick['post'],
            "lookback_days": pick['lb']
        }

def run_auto_loop(max_iters=5):
    import pandas as pd
    from core.genalpha import GenAlpha
    from core.datahub import get_trading_days, load_pv_days, load_universe, load_resp_days, load_restriction_days
    
    print("\n[AutoAgent] Booting up and preloading 2022-01-01 to 2024-12-31...")
    start_date, end_date = "2022-01-01", "2024-12-31"
    all_days = get_trading_days(start=None, end=end_date)
    target_start_idx = all_days.index(start_date) if start_date in all_days else 0
    eval_slice = all_days[target_start_idx:]
    
    shared_ram = {
        'pv': load_pv_days(all_days),  
        'univ': load_universe(all_days),
        'resp': load_resp_days(eval_slice),
        'rest': load_restriction_days(eval_slice)
    }
    
    conversation = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Generate the baseline alpha 001. Keep it simple, e.g., using moving average crossover or price reversal."}
    ]
    
    for i in range(1, max_iters + 1):
        run_id = f"auto_alpha_{i:03d}"
        log_agent_thought(i, "THINKING", "Querying LLM API (with offline fallback)...")
        
        try:
            res_json = query_llm(conversation)
            thought = res_json['thought_process']
            formula = res_json['formula']
            post = res_json.get('postprocess', 'stable_low_turnover')
            lb = res_json.get('lookback_days', 20)
            
            log_agent_thought(i, "WRITING_CODE", f"Formula: {formula}\nThought: {thought}")
            
            # Save YAML config
            yaml_path = f"e:/Scientech/research/configs/{run_id}.yaml"
            os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
            with open(yaml_path, 'w') as f:
                yaml.dump({
                    'formula': formula,
                    'preprocess': {'lookback_days': lb},
                    'postprocess': post,
                    'export_submission': True
                }, f)
                
            # Execute evaluation
            log_agent_thought(i, "EVALUATING", f"Running 3-year backtest engine on '{formula}'...")
            eval_res = GenAlpha.run(
                formula=formula,
                start=start_date, end=end_date,
                preprocess={'lookback_days': lb}, postprocess=post,
                export_submission=True, run_id=run_id, preloaded_data=shared_ram
            )
            
            overall = eval_res.get('metrics', {}).get('overall', {})
            score = overall.get('Score', 0)
            ic = overall.get('IC', 0)
            tvr = overall.get('Turnover', 0)
            ir = overall.get('IR', 0)
            passed = overall.get('PassGates', False)
            
            metrics_dict = {
                'IC': round(ic,4), 'IR': round(ir,4), 'Turnover': round(tvr,4), 
                'Score': round(score,2), 'Passed': passed,
                'Formula': formula, 'PostProcess': post
            }
            
            status_emoji = "✅ PASS" if passed else "❌ FAIL"
            log_agent_thought(i, "RESULT", 
                f"{status_emoji} | Score: {score:.2f} | IC: {ic:.4f} | IR: {ir:.4f} | Tvr: {tvr:.2f}\nFormula: {formula}", 
                metrics_dict)
            
            # Append to auto_leaderboard.csv
            summary_path = r"e:\Scientech\outputs\research_runs\auto_leaderboard.csv"
            os.makedirs(os.path.dirname(summary_path), exist_ok=True)
            df_new = pd.DataFrame([metrics_dict], index=[run_id])
            if os.path.exists(summary_path):
                df_old = pd.read_csv(summary_path, index_col=0)
                df_all = pd.concat([df_old, df_new])
                df_all = df_all[~df_all.index.duplicated(keep='last')]
                df_all = df_all.sort_values(by='Score', ascending=False)
            else:
                df_all = df_new
            df_all.to_csv(summary_path)
            
            # Build next iteration prompt
            conversation.append({"role": "assistant", "content": json.dumps(res_json)})
            if passed:
                msg = f"Excellent! The factor passed with Score={score:.2f}, IC={ic:.4f}, Turnover={tvr:.2f}. Now, generate a completely orthogonal, distinct factor using different features to diversify our pipeline."
            else:
                msg = f"Factor Failed. Score={score:.2f}, IC={ic:.4f}, IR={ir:.4f}, Turnover={tvr:.2f}. "
                if ic < 0.6: msg += "IC is too low. Try a stronger momentum or volume indicator. "
                if tvr >= 400: msg += "Turnover is too high! Use ts_decay_linear or ts_mean to smooth the signal. "
                if ir <= 2.5: msg += "IR is low, it means the signal is not consistent over time. Adjust rolling periods."
                msg += " Please mutate the formula to fix these bottlenecks."
                
            conversation.append({"role": "user", "content": msg})
            
        except Exception as e:
            trace_str = traceback.format_exc()
            log_agent_thought(i, "ERROR", f"Engine Error: {str(e)}\n{trace_str}")
            conversation.append({"role": "user", "content": f"Your last JSON caused an error: {str(e)}. Fix the syntax and return strictly raw JSON."})
            
        import gc
        gc.collect()

if __name__ == "__main__":
    import urllib3
    urllib3.disable_warnings()
    run_auto_loop()
