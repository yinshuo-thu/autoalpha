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

LEADERBOARD_PATH = os.path.join(os.path.dirname(__file__), 'outputs', 'leaderboard.json')


def _ensure_dir():
    os.makedirs(os.path.dirname(LEADERBOARD_PATH), exist_ok=True)


def load_leaderboard():
    """Load the leaderboard from disk."""
    _ensure_dir()
    if os.path.exists(LEADERBOARD_PATH):
        try:
            with open(LEADERBOARD_PATH, 'r') as f:
                data = json.load(f)
                
            # If the loaded data has a mock origin flagged, clean it
            clear_mock = False
            for factor in data.get('factors', []):
                # If these fields from the previous mock state exist or IC is exact mock value
                if 'cover_all' not in factor and factor.get('IC', 0) > 0.5:
                    clear_mock = True
                    break
            if clear_mock:
                print("[LEADERBOARD] Cleaned up legacy mock data from leaderboard.")
                return {'factors': [], 'last_updated': datetime.now().isoformat()}
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
    
    # Check if duplicate formula exists
    formula = metrics.get('formula', '')
    is_duplicate = False
    for f in factors:
        if f.get('formula') == formula and formula:
            f.update({
                'classification': 'Drop',
                'reason': 'Exact formula already exists in library.',
                'Drop': True
            })
            is_duplicate = True
            break
            
    # Format entry
    entry = {
        'factor_name': factor_name,
        'family': metrics.get('family', 'unclassified'),
        'formula': formula,
        'IC': _clean_float(metrics.get('IC')),
        'rank_ic': _clean_float(metrics.get('rank_ic')),
        'IR': _clean_float(metrics.get('IR')),
        'Turnover': _clean_float(metrics.get('Turnover')),
        'Score': _clean_float(metrics.get('Score')),
        'cover_all': metrics.get('cover_all', 1),
        'missing_days': metrics.get('missing_days', 0),
        'maxx': _clean_float(metrics.get('maxx')),
        'minn': _clean_float(metrics.get('minn')),
        'stability_score': _clean_float(metrics.get('stability_score')),
        'classification': metrics.get('classification', 'Drop'),
        'PassGates': metrics.get('PassGates', False),
        'model_ready_flag': metrics.get('model_ready_flag', False),
        'submission_ready_flag': metrics.get('classification') == 'Submission Ready',
        'similarity_cluster': metrics.get('similarity_cluster', 0),
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
    artifact_dir = os.path.join(os.path.dirname(__file__), 'research_runs')
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
