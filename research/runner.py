import os
import glob
import gc
import yaml
import pandas as pd
from core.genalpha import GenAlpha
from core.datahub import get_trading_days, load_pv_days, load_universe, load_resp_days, load_restriction_days
from paths import OUTPUTS_ROOT, PROJECT_ROOT

class BatchRunner:
    @staticmethod
    def execute_configs(configs_dir="research/configs", start_date="2022-01-01", end_date="2024-12-31"):
        yaml_files = glob.glob(os.path.join(configs_dir, "*.yaml"))
        results = {}
        
        # 1. Determine max lookback to preload cache correctly
        max_lookback = 0
        configs_to_run = {}
        for config_path in yaml_files:
            run_id = os.path.basename(config_path).split('.')[0]
            with open(config_path, 'r') as f:
                cfg = yaml.safe_load(f)
            if not cfg.get('formula'):
                print(f"Skipping {run_id}: Missing formula")
                continue
            configs_to_run[run_id] = cfg
            lb = cfg.get('preprocess', {}).get('lookback_days', 20)
            if lb > max_lookback:
                max_lookback = lb
                
        if not configs_to_run:
            print("No valid configs found. Exiting.")
            return
            
        # 2. Preload Data into Memory (THE HUGE SPEEDUP)
        print(f"\n--- [Phase 1: Shared Memory Loading] ({len(configs_to_run)} Alphas) ---")
        print(f"Global Start: {start_date}, End: {end_date}, Max Lookback: {max_lookback} days")
        
        all_days = get_trading_days(start=None, end=end_date)
        try:
            target_start_idx = all_days.index(start_date)
        except ValueError:
            target_start_idx = 0
            start_date = all_days[0] if all_days else None
            
        warmup_start_idx = max(0, target_start_idx - max_lookback)
        eval_days = all_days[warmup_start_idx : ]
        
        print(f"Aggregating {len(eval_days)} trading days of 1-minute Parquet arrays into RAM via Multi-Threading...")
        shared_ram = {
            'pv': load_pv_days(eval_days),
            'univ': load_universe(eval_days)
        }
        
        eval_slice = all_days[target_start_idx : ]
        print(f"Loading Evaluator labels for {len(eval_slice)} days...")
        shared_ram['resp'] = load_resp_days(eval_slice)
        shared_ram['rest'] = load_restriction_days(eval_slice)
        
        print("Data preloading complete! Launching Factor Engine Sweep...\n")

        # 3. Iterate through alphas using shared cache
        for run_id, cfg in configs_to_run.items():
            print(f"\n{'='*40}\nExecuting Batch: {run_id}\n{'='*40}")
            try:
                res = GenAlpha.run(
                    formula=cfg['formula'],
                    start=start_date,
                    end=end_date,
                    preprocess=cfg.get('preprocess', {}),
                    postprocess=cfg.get('postprocess', {}),
                    export_submission=cfg.get('export_submission', True),
                    run_id=run_id,
                    preloaded_data=shared_ram
                )
                
                # Check metrics & gate
                overall = res.get('metrics', {}).get('overall', {})
                valid = res.get('sanity_report', {})
                results[run_id] = {
                    'IC': overall.get('IC', 0),
                    'IR': overall.get('IR', 0),
                    'Turnover': overall.get('Turnover', 0),
                    'Score': overall.get('Score', 0),
                    'PassGates': overall.get('PassGates', False),
                    'SanityStatus': valid.get('status', 'FAIL')
                }
                
                print(f"[{run_id}] Finished! Scorecard:")
                print(f"   -> IC: {results[run_id]['IC']:.4f} | IR: {results[run_id]['IR']:.4f} | Turnover: {results[run_id]['Turnover']:.2f}")
                
                if results[run_id]['PassGates']:
                    print(f"   -> Gate Pass: True ✅ | Final Score: {results[run_id]['Score']:.2f} 🏆")
                else:
                    print(f"   -> Gate Pass: False ❌ | Final Score: 0.00 (Did not pass quality gates)")
                
                # Simple evaluation text
                eval_texts = []
                if results[run_id]['IC'] > 0.03: eval_texts.append("⭐⭐⭐ High predictive power.")
                elif results[run_id]['IC'] > 0.015: eval_texts.append("⭐⭐ Moderate predictive power.")
                else: eval_texts.append("⭐ Weak IC.")
                
                if results[run_id]['Turnover'] > 0.4: eval_texts.append("⚠️ High turnover, execution cost risk.")
                elif results[run_id]['Turnover'] < 0.15: eval_texts.append("✅ Excellent low turnover capacity.")
                
                if not results[run_id]['PassGates']: eval_texts.append("❌ Fails official gates.")
                
                print(f"   -> Eval: {' '.join(eval_texts)}\n")

            except Exception as e:
                print(f"FAILED to execute {run_id}: {e}")
                results[run_id] = {'Error': str(e)}
                
            # Forcibly collect garbage to collapse evaluation buffers
            gc.collect()
            
        if results:
            df_res = pd.DataFrame.from_dict(results, orient='index')
            df_res.index.name = 'run_id'
            
            # 1. Update overall batch summary
            summary_path = os.path.join(OUTPUTS_ROOT, "research_runs", "batch_summary.csv")
            os.makedirs(os.path.dirname(summary_path), exist_ok=True)
            df_res.to_csv(summary_path)
            
            # 2. Output dedicated Leaderboard (Sorted by Score descending)
            leaderboard_path = os.path.join(OUTPUTS_ROOT, "research_runs", "leaderboard.csv")
            lb_df = df_res[['Score', 'PassGates', 'IC', 'IR', 'Turnover']].sort_values(by='Score', ascending=False)
            lb_df.to_csv(leaderboard_path)
            
            print("\n*** Batch Sweep Complete. Leaderboard: ***")
            print(lb_df.to_string())

if __name__ == "__main__":
    BatchRunner.execute_configs(configs_dir=os.path.join(PROJECT_ROOT, "research", "configs"), start_date="2022-01-01", end_date="2024-12-31")
