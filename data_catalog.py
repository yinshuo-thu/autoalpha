"""
data_catalog.py — Registry of all available data fields

Provides a complete catalog of raw and derived data fields
that can be used in factor formulas, with metadata for each.
"""

# ── Raw fields from futures OFR 15m bars (allowed in formulas) ──
RAW_FIELDS = {
    'open_mid_px':     {'type': 'raw', 'source': 'future_ofr', 'freq': '15m', 'can_use': True,
                        'desc': 'Opening mid-quote price', 'example_range': '5-200'},
    'high_mid_px':     {'type': 'raw', 'source': 'future_ofr', 'freq': '15m', 'can_use': True,
                        'desc': 'Highest mid-quote price in bar', 'example_range': '5-200'},
    'low_mid_px':      {'type': 'raw', 'source': 'future_ofr', 'freq': '15m', 'can_use': True,
                        'desc': 'Lowest mid-quote price in bar', 'example_range': '5-200'},
    'close_mid_px':    {'type': 'raw', 'source': 'future_ofr', 'freq': '15m', 'can_use': True,
                        'desc': 'Closing mid-quote price', 'example_range': '5-200'},
    'open_trade_px':   {'type': 'raw', 'source': 'future_ofr', 'freq': '15m', 'can_use': True,
                        'desc': 'Opening trade price', 'example_range': '5-200'},
    'high_trade_px':   {'type': 'raw', 'source': 'future_ofr', 'freq': '15m', 'can_use': True,
                        'desc': 'Highest trade price in bar', 'example_range': '5-200'},
    'low_trade_px':    {'type': 'raw', 'source': 'future_ofr', 'freq': '15m', 'can_use': True,
                        'desc': 'Lowest trade price in bar', 'example_range': '5-200'},
    'close_trade_px':  {'type': 'raw', 'source': 'future_ofr', 'freq': '15m', 'can_use': True,
                        'desc': 'Closing trade price', 'example_range': '5-200'},
    'trade_count':     {'type': 'raw', 'source': 'future_ofr', 'freq': '15m', 'can_use': True,
                        'desc': 'Number of trades in bar', 'example_range': '0-10000'},
    'volume':          {'type': 'raw', 'source': 'future_ofr', 'freq': '15m', 'can_use': True,
                        'desc': 'Contract volume traded', 'example_range': '0-1e8'},
    'dvolume':         {'type': 'raw', 'source': 'future_ofr', 'freq': '15m', 'can_use': True,
                        'desc': 'Notional amount traded (price × volume)', 'example_range': '0-1e10'},
    'vwap':            {'type': 'raw', 'source': 'future_ofr', 'freq': '15m', 'can_use': True,
                        'desc': 'Volume-weighted average price', 'example_range': '5-200'},
    'open_interest':  {'type': 'raw', 'source': 'future_ofr', 'freq': '15m', 'can_use': True,
                        'desc': 'Open interest at bar end', 'example_range': '0-1e7'},
    'delta_oi':       {'type': 'raw', 'source': 'future_ofr', 'freq': '15m', 'can_use': True,
                        'desc': 'Open interest change in bar', 'example_range': '-1e5-1e5'},
    'buy_volume':     {'type': 'raw', 'source': 'future_ofr', 'freq': '15m', 'can_use': True,
                        'desc': 'Aggressor buy volume', 'example_range': '0-1e8'},
    'sell_volume':    {'type': 'raw', 'source': 'future_ofr', 'freq': '15m', 'can_use': True,
                        'desc': 'Aggressor sell volume', 'example_range': '0-1e8'},
    'open_volume':    {'type': 'raw', 'source': 'future_ofr', 'freq': '15m', 'can_use': True,
                        'desc': 'Opening-position volume proxy', 'example_range': '0-1e8'},
    'close_volume':   {'type': 'raw', 'source': 'future_ofr', 'freq': '15m', 'can_use': True,
                        'desc': 'Closing-position volume proxy', 'example_range': '0-1e8'},
    'market_ofi':     {'type': 'raw', 'source': 'future_ofr', 'freq': '15m', 'can_use': True,
                        'desc': 'Trade-side order-flow imbalance', 'example_range': '-1e8-1e8'},
    'add_ofi':        {'type': 'raw', 'source': 'future_ofr', 'freq': '15m', 'can_use': True,
                        'desc': 'Added-limit order flow imbalance', 'example_range': '-1e8-1e8'},
    'cancel_ofi':     {'type': 'raw', 'source': 'future_ofr', 'freq': '15m', 'can_use': True,
                        'desc': 'Cancelled-limit order flow imbalance', 'example_range': '-1e8-1e8'},
    'book_ofi':       {'type': 'raw', 'source': 'future_ofr', 'freq': '15m', 'can_use': True,
                        'desc': 'Book-level order-flow imbalance', 'example_range': '-1e8-1e8'},
    'book_imbalance': {'type': 'raw', 'source': 'future_ofr', 'freq': '15m', 'can_use': True,
                        'desc': 'Bid/ask book imbalance', 'example_range': '-1-1'},
    'spread':         {'type': 'raw', 'source': 'future_ofr', 'freq': '15m', 'can_use': True,
                        'desc': 'Quoted spread proxy', 'example_range': '0-100'},
    'cvd':            {'type': 'raw', 'source': 'future_ofr', 'freq': '15m', 'can_use': True,
                        'desc': 'Cumulative volume delta', 'example_range': '-1e9-1e9'},
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
    'order_flow_imbalance': {'type': 'derived', 'formula': 'div(sub(buy_volume, sell_volume), add(buy_volume, sell_volume))',
                        'freq': '15m', 'can_use': True, 'desc': 'Aggressor buy-sell volume imbalance'},
    'oi_pressure':     {'type': 'derived', 'formula': 'div(delta_oi, ts_mean(volume, 20))',
                        'freq': '15m', 'can_use': True, 'desc': 'Open-interest change normalized by recent activity'},
    'book_pressure':   {'type': 'derived', 'formula': 'mean_of(book_imbalance, div(book_ofi, ts_mean(volume, 20)))',
                        'freq': '15m', 'can_use': True, 'desc': 'Depth imbalance blended with normalized book OFI'},
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
