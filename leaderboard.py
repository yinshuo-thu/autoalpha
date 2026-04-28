"""
leaderboard.py — AutoAlpha Competition Leaderboard (Real Data Version)

Stores, deduplicates, and manages factors evaluated on real data.
Auto-removes any mock data on startup if detected.
"""
import os
import json
from datetime import datetime
import numpy as np
import pandas as pd
from paths import LEADERBOARD_PATH, RESEARCH_ARTIFACTS_ROOT


def _ensure_dir():
    os.makedirs(os.path.dirname(LEADERBOARD_PATH), exist_ok=True)


def load_leaderboard():
    """Load the leaderboard from disk."""
    _ensure_dir()
    if os.path.exists(LEADERBOARD_PATH):
        try:
            with open(LEADERBOARD_PATH, 'r') as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return {'factors': [], 'last_updated': datetime.now().isoformat()}
            factors = list(data.get('factors') or [])
            # 仅剔除「疑似旧 mock」的单条记录，绝不能整表清空；否则下一次 save 会覆盖掉磁盘上全部历史因子。
            kept = []
            dropped = 0
            for factor in factors:
                if not isinstance(factor, dict):
                    continue
                ic = float(factor.get('IC', 0) or 0)
                if 'cover_all' not in factor and ic > 0.5:
                    dropped += 1
                    continue
                kept.append(factor)
            if dropped:
                print(
                    f"[LEADERBOARD] Removed {dropped} legacy mock-like row(s); kept {len(kept)}.",
                    flush=True,
                )
                data['factors'] = kept
                data['last_updated'] = datetime.now().isoformat()
                with open(LEADERBOARD_PATH, 'w') as fh:
                    json.dump(data, fh, indent=2)
            return data
        except json.JSONDecodeError:
            pass
    return {'factors': [], 'last_updated': datetime.now().isoformat()}


def save_leaderboard(data):
    """Save the leaderboard to disk."""
    _ensure_dir()
    data['last_updated'] = datetime.now().isoformat()
    with open(LEADERBOARD_PATH, 'w') as f:
        json.dump(data, f, indent=2)


def get_all_factors():
    """Return list of all evaluated factors."""
    lb = load_leaderboard()
    return lb.get('factors', [])


def _clean_float(val):
    """Ensure JSON serializable."""
    if val is None or pd.isna(val) or not np.isfinite(val):
        return 0.0
    return float(val)


def add_or_update_factor(metrics):
    """Add a new factor evaluation or update an existing one."""
    lb = load_leaderboard()
    factors = lb.get('factors', [])
    
    factor_name = metrics['factor_name']
    official_metrics = metrics.get('official_metrics') or {}
    effective_ic = official_metrics.get('IC', metrics.get('official_IC', metrics.get('IC', 0)))
    effective_ir = official_metrics.get('IR', metrics.get('official_IR', metrics.get('IR', 0)))
    effective_turnover = official_metrics.get('Turnover', metrics.get('official_Turnover', metrics.get('Turnover', 0)))
    effective_score = official_metrics.get('Score', metrics.get('official_Score', metrics.get('Score', 0)))
    effective_pass = official_metrics.get('PassGates', metrics.get('PassGates', False))
    effective_gates = official_metrics.get('GatesDetail', metrics.get('gates_detail', {}))
    
    # Check if duplicate formula exists
    formula = metrics.get('formula', '')
    is_duplicate = False
    for f in factors:
        if f.get('factor_name') == factor_name:
            continue
        if f.get('formula') == formula and formula:
            is_duplicate = True
            break
            
    # Format entry
    entry = {
        'factor_name': factor_name,
        'family': metrics.get('family', 'unclassified'),
        'formula': formula,
        'IC': _clean_float(effective_ic),
        'rank_ic': _clean_float(metrics.get('rank_ic')),
        'IR': _clean_float(effective_ir),
        'Turnover': _clean_float(effective_turnover),
        'TurnoverLocal': _clean_float(metrics.get('TurnoverLocal', metrics.get('Turnover'))),
        'Score': _clean_float(effective_score),
        'cover_all': metrics.get('cover_all', 1),
        'missing_days': metrics.get('missing_days', 0),
        'maxx': _clean_float(official_metrics.get('maxx', metrics.get('maxx'))),
        'minn': _clean_float(official_metrics.get('minn', metrics.get('minn'))),
        'stability_score': _clean_float(metrics.get('stability_score')),
        'classification': metrics.get('classification', 'Drop'),
        'PassGates': effective_pass,
        'model_ready_flag': metrics.get('model_ready_flag', False),
        'submission_ready_flag': metrics.get(
            'submission_ready_flag',
            metrics.get('classification') == 'Submission Ready'
        ),
        'similarity_cluster': metrics.get('similarity_cluster', 0),
        'reason': metrics.get('reason', ''),
        'recommendation': metrics.get('recommendation', ''),
        'submission_path': metrics.get('submission_path', ''),
        'submission_dir': metrics.get('submission_dir', ''),
        'metadata_path': metrics.get('metadata_path', ''),
        'gates_detail': effective_gates,
        'sanity_report': metrics.get('sanity_report', {}),
        'score_formula': metrics.get('score_formula', ''),
        'score_components': metrics.get('score_components', {}),
        'metric_mode': official_metrics.get('metric_mode', metrics.get('metric_mode', 'cloud_aligned_preferred')),
        'turnover_basis': official_metrics.get('turnover_basis', metrics.get('turnover_basis', 'cloud_aligned_preferred')),
        'timestamp': datetime.now().isoformat()
    }
    
    if is_duplicate:
        entry['classification'] = 'Drop'
        entry['reason'] = 'Duplicate'
        
    # Update or append
    idx = next((i for i, f in enumerate(factors) if f['factor_name'] == factor_name), None)
    if idx is not None:
        factors[idx] = entry
    else:
        factors.append(entry)
        
    lb['factors'] = factors
    save_leaderboard(lb)
    
    # Save detailed JSON artifact
    artifact_dir = RESEARCH_ARTIFACTS_ROOT
    os.makedirs(artifact_dir, exist_ok=True)
    with open(os.path.join(artifact_dir, f"{factor_name}.json"), 'w') as f:
        json.dump(metrics, f, indent=2, default=str)


def compute_clusters():
    """Simple offline clustering based on correlation (mocked metric here)."""
    # Requires fully computed time series, so we will assign groups purely based on family for now
    lb = load_leaderboard()
    factors = lb.get('factors', [])
    
    family_map = {}
    cluster_id = 1
    for f in factors:
        fam = f.get('family', 'unknown')
        if fam not in family_map:
            family_map[fam] = cluster_id
            cluster_id += 1
        f['similarity_cluster'] = family_map[fam]
        
    save_leaderboard(lb)
