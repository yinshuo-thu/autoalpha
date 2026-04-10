"""
data_catalog.py — Registry of all available data fields

Provides a complete catalog of raw and derived data fields
that can be used in factor formulas, with metadata for each.
"""

# ── Raw fields from basic_pv (allowed in formulas) ──
RAW_FIELDS = {
    'open_mid_px':     {'type': 'raw', 'source': 'basic_pv', 'freq': '1m/15m', 'can_use': True,
                        'desc': 'Opening mid-quote price', 'example_range': '5-200'},
    'high_mid_px':     {'type': 'raw', 'source': 'basic_pv', 'freq': '1m/15m', 'can_use': True,
                        'desc': 'Highest mid-quote price in bar', 'example_range': '5-200'},
    'low_mid_px':      {'type': 'raw', 'source': 'basic_pv', 'freq': '1m/15m', 'can_use': True,
                        'desc': 'Lowest mid-quote price in bar', 'example_range': '5-200'},
    'close_mid_px':    {'type': 'raw', 'source': 'basic_pv', 'freq': '1m/15m', 'can_use': True,
                        'desc': 'Closing mid-quote price', 'example_range': '5-200'},
    'open_trade_px':   {'type': 'raw', 'source': 'basic_pv', 'freq': '1m/15m', 'can_use': True,
                        'desc': 'Opening trade price', 'example_range': '5-200'},
    'high_trade_px':   {'type': 'raw', 'source': 'basic_pv', 'freq': '1m/15m', 'can_use': True,
                        'desc': 'Highest trade price in bar', 'example_range': '5-200'},
    'low_trade_px':    {'type': 'raw', 'source': 'basic_pv', 'freq': '1m/15m', 'can_use': True,
                        'desc': 'Lowest trade price in bar', 'example_range': '5-200'},
    'close_trade_px':  {'type': 'raw', 'source': 'basic_pv', 'freq': '1m/15m', 'can_use': True,
                        'desc': 'Closing trade price', 'example_range': '5-200'},
    'trade_count':     {'type': 'raw', 'source': 'basic_pv', 'freq': '1m/15m', 'can_use': True,
                        'desc': 'Number of trades in bar', 'example_range': '0-10000'},
    'volume':          {'type': 'raw', 'source': 'basic_pv', 'freq': '1m/15m', 'can_use': True,
                        'desc': 'Share volume traded', 'example_range': '0-1e8'},
    'dvolume':         {'type': 'raw', 'source': 'basic_pv', 'freq': '1m/15m', 'can_use': True,
                        'desc': 'Dollar volume traded (price × volume)', 'example_range': '0-1e10'},
    'vwap':            {'type': 'raw', 'source': 'basic_pv', 'freq': '1m/15m', 'can_use': True,
                        'desc': 'Volume-weighted average price', 'example_range': '5-200'},
}

# ── Forbidden fields (evaluation-only) ──
FORBIDDEN_FIELDS = {
    'resp':                {'type': 'target', 'source': 'eq_resp_stage1', 'freq': '15m', 'can_use': False,
                            'desc': '⛔ Future return target — FORBIDDEN in factor construction. Eval only.'},
    'trading_restriction': {'type': 'restriction', 'source': 'eq_trading_restriction_stage1', 'freq': '15m', 'can_use': False,
                            'desc': '⛔ Trading restriction flag — FORBIDDEN in factor construction. Eval only.'},
}

# ── Derived fields (computed from raw, registered as reusable assets) ──
DERIVED_FIELD_TEMPLATES = {
    'ret_1bar':        {'type': 'derived', 'formula': 'close_trade_px / delay(close_trade_px, 1) - 1',
                        'freq': '15m', 'can_use': True, 'desc': '1-bar return'},
    'vwap_dev':        {'type': 'derived', 'formula': 'close_trade_px / vwap - 1',
                        'freq': '15m', 'can_use': True, 'desc': 'VWAP deviation ratio'},
    'hl_range':        {'type': 'derived', 'formula': 'high_trade_px - low_trade_px',
                        'freq': '15m', 'can_use': True, 'desc': 'High-low range'},
    'hl_range_pct':    {'type': 'derived', 'formula': 'div(sub(high_trade_px, low_trade_px), close_trade_px)',
                        'freq': '15m', 'can_use': True, 'desc': 'High-low range as % of close'},
    'volume_ratio':    {'type': 'derived', 'formula': 'div(volume, ts_mean(volume, 20))',
                        'freq': '15m', 'can_use': True, 'desc': 'Volume relative to 20-bar MA'},
    'dollar_volume_ratio': {'type': 'derived', 'formula': 'div(dvolume, ts_mean(dvolume, 20))',
                        'freq': '15m', 'can_use': True, 'desc': 'Dollar volume relative to 20-bar MA'},
    'mid_spread':      {'type': 'derived', 'formula': 'sub(close_trade_px, close_mid_px)',
                        'freq': '15m', 'can_use': True, 'desc': 'Trade vs mid price spread'},
}


def get_all_allowed_fields():
    """Return set of field names allowed in formula construction."""
    return set(RAW_FIELDS.keys()) | set(DERIVED_FIELD_TEMPLATES.keys())


def get_full_catalog():
    """Return complete catalog for frontend display."""
    catalog = []
    for name, meta in RAW_FIELDS.items():
        catalog.append({'name': name, **meta, 'category': 'Raw Price-Volume'})
    for name, meta in DERIVED_FIELD_TEMPLATES.items():
        catalog.append({'name': name, **meta, 'category': 'Derived'})
    for name, meta in FORBIDDEN_FIELDS.items():
        catalog.append({'name': name, **meta, 'category': '⛔ Forbidden'})
    return catalog


def search_catalog(query):
    """Search catalog by name or description."""
    query = query.lower()
    return [f for f in get_full_catalog()
            if query in f['name'].lower() or query in f['desc'].lower()]
