"""
asset_registry.py — Registered Data Assets Manager

Manages pre-computed, validated intermediate results that can be
reused as inputs in new factor formulas.
"""
import os
import json
import time
from datetime import datetime

REGISTRY_PATH = os.path.join(os.path.dirname(__file__), 'artifacts', 'asset_registry.json')


def _ensure_dir():
    os.makedirs(os.path.dirname(REGISTRY_PATH), exist_ok=True)


def load_registry():
    """Load the asset registry from disk."""
    _ensure_dir()
    if os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH, 'r') as f:
            return json.load(f)
    return {}


def save_registry(registry):
    """Save the asset registry to disk."""
    _ensure_dir()
    with open(REGISTRY_PATH, 'w') as f:
        json.dump(registry, f, indent=2, default=str)


def register_asset(name, asset_type='derived', source_formula='', frequency='15m',
                   leakage_checked=True, cache_path='', tags=None, description=''):
    """Register a new data asset."""
    registry = load_registry()
    registry[name] = {
        'asset_name': name,
        'asset_type': asset_type,
        'source_formula': source_formula,
        'frequency': frequency,
        'created_at': datetime.now().isoformat(),
        'valid_for_formula_input': True,
        'leakage_checked': leakage_checked,
        'cache_path': cache_path,
        'tags': tags or [],
        'description': description,
    }
    save_registry(registry)
    return registry[name]


def get_registered_asset_names():
    """Get set of all registered asset names (for formula validation)."""
    return set(load_registry().keys())


def get_all_assets():
    """Get all assets as a list for frontend display."""
    registry = load_registry()
    return [{'name': k, **v} for k, v in registry.items()]


def remove_asset(name):
    """Remove an asset from the registry."""
    registry = load_registry()
    if name in registry:
        del registry[name]
        save_registry(registry)
