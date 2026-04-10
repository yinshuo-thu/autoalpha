"""
quick_test.py — Single-Factor Quick Evaluation on Real Data

Pipeline: parse → validate → compute → evaluate → classify
Returns a structured JSON result with all metrics.

Usage:
    python quick_test.py "rank(sub(div(close_trade_px, vwap), 1))"
    python quick_test.py "ts_zscore(volume, 20)" --postprocess rank
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from formula_parser import parse_formula, ParseError, ast_to_string
from formula_validator import validate_formula
from compliance_guard import full_compliance_check
from prepare_data import DataHub


def compute_formula(formula_text, data_hub):
    """
    Compute a factor from a DSL formula on real 15m data.
    Returns a pandas Series with the same index as pv_15m.
    """
    from factors.operators import (
        lag, delta, ts_mean, ts_std, ts_sum, ts_max, ts_min, ts_zscore,
        ts_rank, ts_decay_linear, ts_cov, ts_corr, cs_rank, cs_demean, cs_zscore, safe_div, signed_power
    )

    pv = data_hub.pv_15m

    # Build evaluation namespace with allowed fields and operators
    ns = {}
    for col in pv.columns:
        ns[col] = pv[col]

    # Operator mappings
    ns.update({
        'rank': cs_rank, 'cs_rank': cs_rank,
        'zscore': cs_zscore, 'cs_zscore': cs_zscore,
        'demean': cs_demean, 'cs_demean': cs_demean,
        'delay': lag, 'lag': lag,
        'delta': delta,
        'ts_mean': ts_mean, 'ts_std': ts_std, 'ts_sum': ts_sum,
        'ts_max': ts_max, 'ts_min': ts_min,
        'ts_cov': ts_cov, 'ts_corr': ts_corr,
        'ts_zscore': ts_zscore, 'ts_rank': ts_rank,
        'ts_decay_linear': ts_decay_linear, 'decay_linear': ts_decay_linear,
        'safe_div': safe_div, 'div': safe_div,
        'signed_power': signed_power, 'pow': signed_power,
        'neg': lambda x: -x,
        'abs': lambda x: x.abs() if hasattr(x, 'abs') else np.abs(x),
        'log': lambda x: np.log(np.abs(x) + 1) if not hasattr(x, 'apply') else x.abs().add(1).apply(np.log),
        'signed_log': lambda x: np.sign(x) * np.log1p(np.abs(x)),
        'sqrt': lambda x: np.sqrt(np.abs(x)) if not hasattr(x, 'apply') else x.abs().apply(np.sqrt),
        'sub': lambda a, b: a - b,
        'add': lambda a, b: a + b,
        'mul': lambda a, b: a * b,
        'clip': lambda x, a, b: x.clip(a, b) if hasattr(x, 'clip') else np.clip(x, a, b),
        'gt': lambda x, y: (x > y).astype(float),
        'lt': lambda x, y: (x < y).astype(float),
        'ifelse': lambda c, a, b: a.where(c > 0, b) if hasattr(a, 'where') else np.where(c > 0, a, b),
        'sign': lambda x: np.sign(x) if not hasattr(x, 'apply') else x.apply(np.sign),
        'scale': lambda x: cs_demean(x),  # simplified
        'winsorize': lambda x, p=0.01: x.clip(x.quantile(p), x.quantile(1-p)) if hasattr(x, 'quantile') else x,
        'mean_of': lambda *args: sum(args) / len(args),
        'combine_rank': lambda *args: sum(cs_rank(a) for a in args) / len(args),
    })

    # Evaluate using restricted eval with namespace
    try:
        result = eval(formula_text, {"__builtins__": {}}, ns)
    except Exception as e:
        raise RuntimeError(f"Formula computation failed: {e}")

    if isinstance(result, pd.DataFrame):
        if result.shape[1] == 1:
            result = result.iloc[:, 0]
        else:
            raise RuntimeError("Formula returned multiple columns")

    return result


def evaluate_factor(alpha_series, data_hub, factor_name='test'):
    """Evaluate a computed factor against resp."""
    from core.evaluator import Evaluator

    resp = data_hub.resp
    restriction = data_hub.trading_restriction

    # Broadcast daily resp to 15m alpha
    a_df = alpha_series.to_frame("alpha").reset_index()
    r_df = resp.reset_index()[["date", "security_id", "resp"]]
    merged = pd.merge(a_df, r_df, on=["date", "security_id"], how="inner")
    if merged.empty: return {"error": "No overlap"}
    merged = merged.set_index(["date", "datetime", "security_id"]).sort_index()
    alpha_aligned, resp_aligned = merged["alpha"], merged["resp"]
    if restriction is not None and not restriction.empty:
        rest_df = restriction.reset_index()
        m_rest = pd.merge(merged.reset_index(), rest_df, on=["date", "security_id"], how="left")
        restriction_aligned = m_rest.set_index(["date", "datetime", "security_id"]).sort_index().get("trading_restriction", 0).fillna(0)
    else: restriction_aligned = pd.Series(0.0, index=merged.index)

    # Run official evaluator
    try:
        metrics = Evaluator.run(alpha_aligned, resp_aligned, restriction_aligned)
    except Exception as e:
        return {'error': f'Evaluator failed: {e}'}

    # Flatten and map to expected format for frontend/leaderboard
    overall = metrics.get('overall', {})
    daily_ic = metrics.get('daily_ic', pd.Series(dtype=float))
    missing_days = len(set(data_hub.get_trading_days_list()) - set(alpha_aligned.index.get_level_values('date').unique().astype(str)))

    daily_ic_list = []
    if isinstance(daily_ic, pd.Series) and not daily_ic.empty:
        # Convert index to string dates if they aren't already
        daily_ic_list = [{'date': str(d), 'IC': float(v)}
                        for d, v in daily_ic.items() if np.isfinite(v)]

    # Monthly heatmap
    monthly_heatmap = {}
    if not daily_ic.empty:
        daily_ic_df = daily_ic.to_frame('IC')
        daily_ic_df.index = pd.to_datetime(daily_ic_df.index)
        monthly = daily_ic_df.groupby(daily_ic_df.index.to_period('M')).mean()
        for period, row in monthly.iterrows():
            monthly_heatmap[str(period)] = float(row['IC'])

    return {
        'factor_name': factor_name,
        'IC': overall.get('IC', 0) / 100.0,  # Scale back to decimal for consistency
        'rank_ic': metrics.get('rank_ic', 0) / 100.0,
        'IR': overall.get('IR', 0),
        'Turnover': overall.get('Turnover', 0),
        'Score': overall.get('Score', 0),
        'score_raw': overall.get('Score', 0),
        'ic_minus_tvr': metrics.get('ic_minus_tvr', 0) / 100.0,
        'cover_all': 1 if missing_days == 0 else 0,
        'missing_days': missing_days,
        'maxx': overall.get('maxx', 0),
        'minn': overall.get('minn', 0),
        'stability_score': metrics.get('stability_score', 0),
        'positive_ic_ratio': metrics.get('positive_ic_ratio', 0),
        'PassGates': overall.get('PassGates', False),
        'classification': 'Submission Ready' if overall.get('PassGates') else ('Research Candidate' if abs(overall.get('IC', 0)) > 0.1 else 'Drop'),
        'yearly': metrics.get('yearly', {}),
        'monthly_heatmap': monthly_heatmap,
        'time_series': {
            'daily_ic': daily_ic_list,
        },
    }


def quick_test(formula_text, factor_name='quick_test', postprocess=None):
    """
    Full quick test pipeline: parse → validate → compute → evaluate → classify.
    Returns structured JSON result.
    """
    result = {'factor_name': factor_name, 'formula': formula_text, 'status': 'pending'}

    # Stage 1: Validate
    validation = validate_formula(formula_text)
    result['validation'] = validation.to_dict()
    if not validation.valid:
        result['status'] = 'validation_failed'
        result['classification'] = 'Drop'
        result['reason'] = '; '.join(validation.errors)
        return result

    # Stage 2: Compliance
    from compliance_guard import check_formula_compliance
    compliance = check_formula_compliance(formula_text)
    result['compliance'] = compliance.to_dict()
    if not compliance.passed:
        result['status'] = 'compliance_failed'
        result['classification'] = 'Drop'
        result['reason'] = str(compliance)
        return result

    # Stage 3: Load data & compute
    try:
        t0 = time.time()
        hub = DataHub()
        result['data_load_time'] = time.time() - t0

        t1 = time.time()
        alpha = compute_formula(formula_text, hub)
        result['compute_time'] = time.time() - t1

        # Optional postprocess
        if postprocess == 'rank':
            from factors.operators import cs_rank
            alpha = cs_rank(alpha)
        elif postprocess == 'zscore':
            from factors.operators import cs_zscore
            alpha = cs_zscore(alpha)

    except Exception as e:
        result['status'] = 'computation_failed'
        result['classification'] = 'Drop'
        result['reason'] = str(e)
        return result

    # Stage 4: Evaluate
    try:
        t2 = time.time()
        metrics = evaluate_factor(alpha, hub, factor_name)
        result.update(metrics)
        result['eval_time'] = time.time() - t2
        result['status'] = 'success'

        if metrics['PassGates']:
            result['recommendation'] = '✅ Passed gates — consider adding to research queue'
        elif metrics['classification'] == 'Research Candidate':
            result['recommendation'] = '🔬 Research candidate — worth further exploration'
        else:
            result['recommendation'] = '❌ Drop — metrics too poor for submission'
            
        import sys
        if 'outputs.export_submission' not in sys.modules:
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from outputs.export_submission import export_to_parquet
        out_path = export_to_parquet(
            alpha, 
            factor_name, 
            metrics=metrics, 
            description=f"Quick test result for {factor_name}: {formula_text}"
        )
        result['submission_path'] = out_path

    except Exception as e:
        result['status'] = 'evaluation_failed'
        result['classification'] = 'Drop'
        result['reason'] = str(e)

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Quick test a factor formula on real data')
    parser.add_argument('formula', help='DSL formula to test')
    parser.add_argument('--name', default='quick_test', help='Factor name')
    parser.add_argument('--postprocess', choices=['rank', 'zscore'], help='Postprocessing')
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  🧪 Quick Test: {args.formula}")
    print(f"{'='*60}\n")

    result = quick_test(args.formula, args.name, args.postprocess)

    print(f"Status: {result['status']}")
    if result['status'] == 'success':
        print(f"IC:     {result['IC']:.4f}")
        print(f"RankIC: {result['rank_ic']:.4f}")
        print(f"IR:     {result['IR']:.2f}")
        print(f"TVR:    {result['Turnover']:.1f}")
        print(f"Score:  {result['Score']:.2f}")
        print(f"Gates:  {'PASS' if result['PassGates'] else 'FAIL'}")
        print(f"Class:  {result['classification']}")
        print(f"\n{result.get('recommendation', '')}")
    else:
        print(f"Reason: {result.get('reason', 'Unknown error')}")
