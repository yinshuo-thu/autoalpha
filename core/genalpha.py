import os
import pandas as pd
import yaml
from core.datahub import get_trading_days, load_pv_days, load_universe, load_resp_days, load_restriction_days, align_to_universe
from core.formula_engine import FormulaEngine
from core.postprocess import apply_postprocess, TEMPLATES
from core.evaluator import Evaluator
from core.submission import SubmissionBuilder
from core.diagnostics import Diagnostics
from paths import RESEARCH_ARTIFACTS_ROOT, SUBMISSIONS_ROOT

class GenAlpha:
    @staticmethod
    def run(formula, start, end, preprocess=None, postprocess=None, eval_mode="official_like", export_submission=True, run_id="alpha_base", preloaded_data=None):
        if not preloaded_data:
            print(f"[{run_id}] Loading Data...")
        all_days = get_trading_days(start=None, end=end)
        try:
            target_start_idx = all_days.index(start)
            target_start_date = pd.to_datetime(start)
        except ValueError:
            target_start_idx = 0
            target_start_date = pd.to_datetime(all_days[0]) if all_days else None
            
        lookback_days = 20 # 16 bars/day = ~320 15m bars
        if preprocess and 'lookback_days' in preprocess:
            lookback_days = preprocess['lookback_days']
            
        warmup_start_idx = max(0, target_start_idx - lookback_days)
        eval_days = all_days[warmup_start_idx : ]
        
        if not eval_days:
            raise ValueError("No matching trading days found.")
            
        # Data pipelining
        eval_date_ts = [pd.to_datetime(d) for d in eval_days]
        
        chunk_days = 90
        target_start_dt = pd.to_datetime(start)
        all_eval_days = [d for d in eval_date_ts if d >= target_start_dt]
        
        print(f"[{run_id}] Evaluating Formula via {chunk_days}-Day Streaming Chunks...")
        if isinstance(postprocess, str) and postprocess in TEMPLATES:
            postprocess = TEMPLATES[postprocess]
            
        chunked_alphas = []
        for i in range(0, len(all_eval_days), chunk_days):
            chunk_start = all_eval_days[i]
            chunk_end = all_eval_days[min(i + chunk_days - 1, len(all_eval_days) - 1)]
            
            idx_start = eval_date_ts.index(chunk_start)
            idx_lookback = max(0, idx_start - lookback_days)
            sub_eval_ts = eval_date_ts[idx_lookback : eval_date_ts.index(chunk_end) + 1]
            
            if preloaded_data:
                sub_pv = preloaded_data['pv']
                sub_univ = preloaded_data['univ']
                sub_pv = sub_pv[sub_pv.index.get_level_values('date').isin(sub_eval_ts)]
                sub_univ = sub_univ[sub_univ.index.get_level_values('date').isin(sub_eval_ts)]
            else:
                sub_eval_str = [d.strftime('%Y-%m-%d') for d in sub_eval_ts]
                sub_pv = load_pv_days(sub_eval_str)
                sub_univ = load_universe(sub_eval_str)
                
            raw_alpha = FormulaEngine.evaluate(formula, sub_pv)
            
            merged = align_to_universe(raw_alpha.to_frame('alpha'), sub_univ)
            if 'eq_univ' in merged.columns:
                merged = merged[merged['eq_univ'] == True]
                
            alpha_post = apply_postprocess(merged['alpha'], postprocess)
            
            alpha_valid = alpha_post[
                (alpha_post.index.get_level_values('date') >= chunk_start) & 
                (alpha_post.index.get_level_values('date') <= chunk_end)
            ]
            
            chunked_alphas.append(alpha_valid)
            
            import gc
            del sub_pv, sub_univ, raw_alpha, merged, alpha_post, alpha_valid
            gc.collect()
            
        alpha_final = pd.concat(chunked_alphas) if chunked_alphas else pd.Series(dtype=float)
        
        metrics = {}
        if eval_mode:
            print(f"[{run_id}] Running Evaluation...")
            # We only evaluate the non-warmup test slice
            eval_date_slice = all_days[target_start_idx:]
            if preloaded_data and 'resp' in preloaded_data:
                df_resp = preloaded_data['resp']
                df_rest = preloaded_data['rest']
                # Filter down to eval slice
                eval_slice_ts = [pd.to_datetime(d) for d in eval_date_slice]
                df_resp = df_resp[df_resp.index.get_level_values('date').isin(eval_slice_ts)]
                df_rest = df_rest[df_rest.index.get_level_values('date').isin(eval_slice_ts)]
            else:
                df_resp = load_resp_days(eval_date_slice)
                df_rest = load_restriction_days(eval_date_slice)
            
            resp = df_resp['resp'] if 'resp' in df_resp.columns else pd.Series(dtype=float)
            rest = df_rest['trading_restriction'] if 'trading_restriction' in df_rest.columns else pd.Series(dtype=float)
            
            metrics = Evaluator.run(alpha_final, resp, rest)
            
            # Save diagnostics
            export_dir = os.path.join(RESEARCH_ARTIFACTS_ROOT, run_id)
            Diagnostics.export(metrics, export_dir)
            
        print(f"[{run_id}] Running Sanity Check...")
        valid = SubmissionBuilder.pre_submit_sanity_check(alpha_final, start, end)
        
        out_path = os.path.join(SUBMISSIONS_ROOT, f"{run_id}.pq")
        if export_submission:
            import sys
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if base_dir not in sys.path:
                sys.path.append(base_dir)
            from outputs.export_submission import export_to_parquet
            out_path = export_to_parquet(
                alpha_final, 
                run_id, 
                metrics=metrics, 
                description=f"Generated via GenAlpha for formula: {formula}"
            )
            
        return {
            'alpha': alpha_final,
            'metrics': metrics,
            'sanity_report': valid,
            'export': out_path
        }
