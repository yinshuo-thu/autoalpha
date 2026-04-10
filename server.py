"""
server.py — AutoAlpha Backend API (Flask)
Serves the 5-page frontend and provides all API endpoints.
Uses exclusively real data from DataHub.
"""
import os
import json
import uuid
import collections
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime

import subprocess

# Local Modules
from prepare_data import DataHub
from leaderboard import load_leaderboard, get_all_factors, add_or_update_factor
from formula_validator import validate_formula
from compliance_guard import check_formula_compliance
from quick_test import quick_test, compute_formula
from fit_models import fit_and_evaluate_models
from simulate_strategy import run_strategy_simulation
from data_catalog import get_full_catalog
from operator_catalog import get_operators_by_category, get_operator_list
from asset_registry import get_all_assets
from factor_idea_generator import generate_from_prompt

app = Flask(__name__, static_folder='frontend-react/dist')
CORS(app)

# Global DataHub Reference (Lazy load to save startup time)
_data_hub = None
_research_process = None

def get_data_hub():
    global _data_hub
    if _data_hub is None:
        _data_hub = DataHub()
    return _data_hub

# ── Serve Frontend ──
@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def static_files(path):
    if os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, 'index.html')

# ── Existing API Routes ──
@app.route('/api/summary', methods=['GET'])
def get_summary():
    factors = get_all_factors()
    total = len(factors)
    passed = sum(1 for f in factors if f.get('PassGates'))
    sub_ready = sum(1 for f in factors if f.get('classification') == 'Submission Ready')
    
    valid_ic = [f['IC'] for f in factors if f.get('IC') is not None]
    valid_score = [f['Score'] for f in factors if f.get('Score') is not None]
    valid_tvr = [f['Turnover'] for f in factors if f.get('Turnover') is not None and f.get('Turnover') > 0]
    
    return jsonify({
        'total_factors': total,
        'passed_gates': passed,
        'submission_ready': sub_ready,
        'best_ic': max(valid_ic) if valid_ic else 0.0,
        'best_score': max(valid_score) if valid_score else 0.0,
        'avg_tvr': sum(valid_tvr)/len(valid_tvr) if valid_tvr else 0.0,
    })

@app.route('/api/factors', methods=['GET'])
def list_factors():
    factors = get_all_factors()
    
    # Filtering
    q = request.args.get('q', '').lower()
    if q:
        factors = [f for f in factors if q in f['factor_name'].lower() or q in f.get('formula', '').lower()]
        
    cls = request.args.get('classification')
    if cls and cls != 'All':
        factors = [f for f in factors if f.get('classification') == cls]
        
    return jsonify(factors)

@app.route('/api/factors/<name>', methods=['GET'])
def get_factor_detail(name):
    # Load detailed artifact
    artifact_path = os.path.join(os.path.dirname(__file__), 'research_runs', f"{name}.json")
    if os.path.exists(artifact_path):
        with open(artifact_path, 'r') as f:
            return jsonify(json.load(f))
            
    # Fallback to leaderboard metadata
    factors = get_all_factors()
    for f in factors:
        if f['factor_name'] == name:
            return jsonify(f)
    return jsonify({'error': 'Factor not found'}), 404

@app.route('/api/filters', methods=['GET'])
def get_filters():
    factors = get_all_factors()
    families = set(f.get('family', 'unclassified') for f in factors)
    classes = set(f.get('classification', 'Drop') for f in factors)
    return jsonify({
        'families': sorted(list(families)),
        'classifications': sorted(list(classes))
    })

@app.route('/api/compare', methods=['GET'])
def get_compare_stats():
    names = request.args.get('factors', '').split(',')
    factors = [f for f in get_all_factors() if f['factor_name'] in names]
    return jsonify(factors)

@app.route('/api/clusters', methods=['GET'])
def get_clusters():
    factors = get_all_factors()
    return jsonify({
        'clusters': [{'id': f['similarity_cluster'], 'size': sum(1 for x in factors if x['similarity_cluster'] == f['similarity_cluster'])} for f in factors],
        'nodes': [{'id': f['factor_name'], 'group': f['similarity_cluster'], 'val': f['IC']} for f in factors]
    })


# ── New API Routes: Formula Lab ──
@app.route('/api/catalog/data', methods=['GET'])
def catalog_data():
    return jsonify(get_full_catalog())

@app.route('/api/catalog/operators', methods=['GET'])
def catalog_operators():
    return jsonify(get_operators_by_category())

@app.route('/api/catalog/assets', methods=['GET'])
def catalog_assets():
    return jsonify(get_all_assets())

@app.route('/api/idea/generate', methods=['POST'])
def generate_ideas():
    data = request.json
    prompt = data.get('prompt', '')
    ideas = generate_from_prompt(prompt)
    return jsonify({'ideas': ideas})

@app.route('/api/formula/test', methods=['POST'])
def run_quick_test():
    """Runs quick_test on real data"""
    data = request.json
    formula = data.get('formula')
    factor_name = data.get('name', f"formula_{int(time.time())}")
    postprocess = data.get('postprocess', None)
    
    # 1. First trigger DataHub initialization before locking UI for 30 seconds
    get_data_hub()
    
    # 2. Run Test
    try:
        result = quick_test(formula, factor_name, postprocess=postprocess)
        if result.get('status') == 'success':
            add_or_update_factor(result)
        return jsonify(result)
    except Exception as e:
        return jsonify({'status': 'error', 'reason': str(e)}), 500


# ── New API Routes: Model & Strategy Lab ──
@app.route('/api/model/<factor_name>', methods=['GET'])
def run_fit_model(factor_name):
    """Runs LinearRegression/LightGBM on real Train/Val/Test data."""
    hub = get_data_hub()
    
    # We need the precomputed factor series. Let's load the artifact formula and compute.
    artifact_path = os.path.join(os.path.dirname(__file__), 'research_runs', f"{factor_name}.json")
    if not os.path.exists(artifact_path):
        return jsonify({'status': 'error', 'reason': 'Factor metric result not found'})
        
    with open(artifact_path, 'r') as f:
        meta = json.load(f)
        
    try:
        series = compute_formula(meta['formula'], hub)
        result = fit_and_evaluate_models(series, hub, factor_name)
        return jsonify(result)
    except Exception as e:
        return jsonify({'status': 'error', 'reason': str(e)})

@app.route('/api/strategy/<factor_name>', methods=['GET'])
def run_strategy(factor_name):
    """Runs Portoflio simulation."""
    hub = get_data_hub()
    artifact_path = os.path.join(os.path.dirname(__file__), 'research_runs', f"{factor_name}.json")
    if not os.path.exists(artifact_path):
        return jsonify({'status': 'error', 'reason': 'Factor metric result not found'})
        
    with open(artifact_path, 'r') as f:
        meta = json.load(f)
        
    try:
        series = compute_formula(meta['formula'], hub)
        result = run_strategy_simulation(series, hub, factor_name)
        return jsonify(result)
    except Exception as e:
        return jsonify({'status': 'error', 'reason': str(e)})


# ── New API Routes: Multi-Agent Factory ──
@app.route('/api/factory/status', methods=['GET'])
def get_factory_status():
    """Mock-up status endpoint showing agent statuses based on latest factor."""
    global _research_process
    lb = load_leaderboard()
    factors = lb.get('factors', [])
    best = max(factors, key=lambda x: x.get('Score', 0), default={})
    
    recent_fail = next((f for f in reversed(factors) if not f.get('PassGates')), {})
    
    is_running = _research_process is not None and _research_process.poll() is None
    
    agents = [
        {'id': 'planner', 'name': 'Research Planner (EA)', 'status': 'running' if is_running else 'idle', 'task': 'Iterating Evolutionary Population'},
        {'id': 'miner', 'name': 'Formula Miner', 'status': 'running' if is_running else 'idle', 'task': f'Mined {len(factors)} formulas'},
        {'id': 'compliance', 'name': 'Compliance Guard', 'status': 'running' if is_running else 'idle', 'task': 'Checking leakages on real data'},
        {'id': 'evaluator', 'name': 'Evaluator', 'status': 'running' if is_running else 'idle', 'task': 'Evaluated metrics...'},
        {'id': 'modellab', 'name': 'Model Lab', 'status': 'idle', 'task': 'Waiting for manual trigger'}
    ]
    
    return jsonify({
        'global_state': {
            'is_running': is_running,
            'best_factor': best.get('factor_name', 'None'),
            'best_score': best.get('Score', 0),
            'submission_ready_count': sum(1 for f in factors if f.get('classification') == 'Submission Ready'),
            'total_factors': len(factors),
            'recent_fail_reason': recent_fail.get('reason', 'N/A')
        },
        'agents': agents
    })

@app.route('/api/factory/start', methods=['POST'])
def start_factory():
    global _research_process
    if _research_process is not None and _research_process.poll() is None:
        return jsonify({'status': 'already_running'})
        
    log_file = open(os.path.join(os.path.dirname(__file__), 'outputs', 'research.log'), 'w')
    _research_process = subprocess.Popen(
        ['python', 'research_loop.py', '--max-iters', '100'],
        cwd=os.path.dirname(__file__),
        stdout=log_file,
        stderr=subprocess.STDOUT
    )
    return jsonify({'status': 'started'})

@app.route('/api/factory/stop', methods=['POST'])
def stop_factory():
    global _research_process
    if _research_process is not None and _research_process.poll() is None:
        _research_process.terminate()
        _research_process = None
        return jsonify({'status': 'stopped'})
    return jsonify({'status': 'not_running'})


if __name__ == '__main__':
    print("AutoAlpha Server (Real Data Engine) starting...")
    app.run(host='127.0.0.1', port=8080, debug=False)
