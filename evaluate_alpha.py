"""
evaluate_alpha.py — Full batch evaluation pipeline for factors
Called by research_loop.py

Performs comprehensive evaluation:
- IC, return spread, turnover
- PassGates logic
- Save to artifact log
"""
import os
import json
import uuid
from datetime import datetime
import pandas as pd
import numpy as np

# Re-use the new quick_test logic instead of duplicate evaluator
from quick_test import evaluate_factor, compute_formula
from compliance_guard import full_compliance_check


def evaluate_full(formula_text, factor_name, data_hub, family='exploration', postprocess=None):
    """
    Evaluate a formula on real data.
    """
    metrics = {
        'experiment_id': str(uuid.uuid4()),
        'factor_name': factor_name,
        'family': family,
        'formula': formula_text,
        'status': 'pending',
        'timestamp': datetime.now().isoformat(),
    }

    try:
        # Pre-check compliance
        comp = full_compliance_check(formula_text)
        metrics['compliance'] = comp.to_dict()
        if not comp.passed:
            metrics['status'] = 'compliance_failed'
            metrics['classification'] = 'Drop'
            metrics['reason'] = str(comp)
            return metrics
            
        # Compute Alpha
        alpha_series = compute_formula(formula_text, data_hub)
        
        # Postprocess
        if postprocess == 'rank':
            from factors.operators import cs_rank
            alpha_series = cs_rank(alpha_series)
        elif postprocess == 'zscore':
            from factors.operators import cs_zscore
            alpha_series = cs_zscore(alpha_series)

        # Full eval
        eval_metrics = evaluate_factor(alpha_series, data_hub, factor_name)
        
        # Merge metrics
        metrics.update(eval_metrics)
        metrics['status'] = 'success'
        
        # Export submission to parquet and json
        import sys
        if 'outputs.export_submission' not in sys.modules:
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from outputs.export_submission import export_to_parquet
        
        out_path = export_to_parquet(
            alpha_series, 
            factor_name, 
            metrics=metrics, 
            description=f"Auto evaluation for {factor_name}: {formula_text}"
        )
        metrics['submission_path'] = out_path
        
    except Exception as e:
        metrics['status'] = 'computation_failed'
        metrics['classification'] = 'Drop'
        metrics['PassGates'] = False
        metrics['reason'] = str(e)

    # Detailed formatting for the artifacts JSON save
    return metrics
