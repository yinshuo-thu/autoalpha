"""
prepare_data.py — Real Data Loader & 15-Minute Resampler

Loads raw 1-minute parquet data from the competition dataset,
resamples to 15-minute bars, and caches the result.
Also loads universe, resp (eval-only), and trading_restriction (eval-only).

Usage:
    python prepare_data.py                    # Precompute & cache
    python prepare_data.py --force            # Force re-cache
"""
import os
import sys
import time
import glob
import json
import hashlib
import argparse
import re
import warnings
import pandas as pd
import numpy as np
from datetime import datetime
from paths import DATA_ROOT, CACHE_ROOT

warnings.filterwarnings('ignore', category=FutureWarning)

CACHE_DIR = CACHE_ROOT
os.makedirs(CACHE_DIR, exist_ok=True)

# ── Allowed fields (competition-safe for factor construction) ──
ALLOWED_FIELDS = [
    'open_mid_px', 'high_mid_px', 'low_mid_px', 'close_mid_px',
    'open_trade_px', 'high_trade_px', 'low_trade_px', 'close_trade_px',
    'trade_count', 'volume', 'dvolume', 'vwap',
]

# ── FORBIDDEN for factor construction (eval-only) ──
FORBIDDEN_FIELDS = ['resp', 'trading_restriction']


def get_trading_days(start='2022-01-04', end='2024-12-31'):
    """Get all trading days from basic_pv directory structure."""
    pv_root = os.path.join(DATA_ROOT, 'eq_data_stage1', 'basic_pv')
    days = []
    for year in sorted(os.listdir(pv_root)):
        year_dir = os.path.join(pv_root, year)
        if not os.path.isdir(year_dir):
            continue
        for month in sorted(os.listdir(year_dir)):
            month_dir = os.path.join(year_dir, month)
            if not os.path.isdir(month_dir):
                continue
            for day in sorted(os.listdir(month_dir)):
                day_dir = os.path.join(month_dir, day)
                if not os.path.isdir(day_dir):
                    continue
                date_str = f"{year}-{month}-{day}"
                if start <= date_str <= end:
                    days.append(date_str)
    return sorted(days)


def load_single_day_pv(date_str):
    parts = date_str.split('-')
    path = os.path.join(DATA_ROOT, 'eq_data_stage1', 'basic_pv',
                        parts[0], parts[1], parts[2], 'data.pq')
    if not os.path.exists(path):
        return None
    try:
        return pd.read_parquet(path)
    except Exception as e:
        print(f"Warning: Failed to read pv parquet {path}: {e}")
        return None


def resample_1m_to_15m(df_1m):
    """Resample 1-minute OHLCV data to 15-minute bars."""
    if df_1m is None or df_1m.empty:
        return pd.DataFrame()

    # Ensure datetime index
    df = df_1m.reset_index()
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)

    # Group by date + security, resample to 15min
    agg_map = {
        'open_mid_px': 'first', 'high_mid_px': 'max', 'low_mid_px': 'min', 'close_mid_px': 'last',
        'open_trade_px': 'first', 'high_trade_px': 'max', 'low_trade_px': 'min', 'close_trade_px': 'last',
        'trade_count': 'sum', 'volume': 'sum', 'dvolume': 'sum',
    }

    results = []
    for (date_val, sec_id), grp in df.groupby(['date', 'security_id']):
        grp = grp.set_index('datetime').sort_index()
        resampled = grp[list(agg_map.keys())].resample('15min', closed='left', label='left').agg(agg_map)
        # VWAP: dollar-volume-weighted average
        vol_sum = grp['volume'].resample('15min', closed='left', label='left').sum()
        dvol_sum = grp['dvolume'].resample('15min', closed='left', label='left').sum()
        resampled['vwap'] = np.where(vol_sum > 0, dvol_sum / vol_sum, np.nan)
        resampled = resampled.dropna(subset=['close_trade_px'])
        resampled['date'] = date_val
        resampled['security_id'] = sec_id
        resampled = resampled.reset_index()
        results.append(resampled)

    if not results:
        return pd.DataFrame()

    combined = pd.concat(results, ignore_index=True)
    combined = combined.set_index(['date', 'datetime', 'security_id']).sort_index()
    return combined


def load_universe():
    """Load universe data (all years)."""
    uni_root = os.path.join(DATA_ROOT, 'eq_data_stage1', 'universe')
    dfs = []
    for year in sorted(os.listdir(uni_root)):
        path = os.path.join(uni_root, year, 'data.pq')
        if os.path.exists(path):
            dfs.append(pd.read_parquet(path))
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs).sort_index()


def load_resp(date_str):
    parts = date_str.split('-')
    path = os.path.join(DATA_ROOT, 'eq_resp_stage1', 'resp',
                        parts[0], parts[1], parts[2], 'data.pq')
    if not os.path.exists(path):
        return None
    try:
        return pd.read_parquet(path)
    except Exception as e:
        print(f"Warning: Failed to read resp parquet {path}: {e}")
        return None


def load_trading_restriction(date_str):
    """Load trading restriction for a single day (EVAL ONLY)."""
    parts = date_str.split('-')
    path = os.path.join(DATA_ROOT, 'eq_trading_restriction_stage1', 'trading_restriction',
                        parts[0], parts[1], parts[2], 'data.pq')
    if os.path.exists(path):
        return pd.read_parquet(path)
    return None


def _cache_path(prefix, start, end):
    return os.path.join(CACHE_DIR, f'{prefix}_{start}_{end}.parquet')


def _parse_cache_window(path, prefix):
    """Return (start_ts, end_ts) for cache files named prefix_YYYY-MM-DD_YYYY-MM-DD.parquet."""
    name = os.path.basename(path)
    expected = f"{prefix}_"
    if not name.startswith(expected) or not name.endswith(".parquet"):
        return None
    body = name[len(expected):-len(".parquet")]
    try:
        start_s, end_s = body.rsplit("_", 1)
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", start_s):
            return None
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", end_s):
            return None
        return pd.Timestamp(start_s), pd.Timestamp(end_s)
    except Exception:
        return None


def _find_superset_cache(prefix, start, end):
    """Find the smallest existing cache whose date window covers start/end."""
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    candidates = []
    for path in glob.glob(os.path.join(CACHE_DIR, f"{prefix}_*.parquet")):
        parsed = _parse_cache_window(path, prefix)
        if parsed is None:
            continue
        cache_start, cache_end = parsed
        if cache_start <= start_ts and cache_end >= end_ts:
            span_days = (cache_end - cache_start).days
            candidates.append((span_days, path, cache_start, cache_end))
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: (item[0], item[1]))[0]


def _load_cache_window(prefix, label, start, end, force=False):
    exact_path = _cache_path(prefix, start, end)
    if os.path.exists(exact_path) and not force:
        print(f"[CACHE] Loading cached {label} from {exact_path}")
        t0 = time.time()
        df = pd.read_parquet(exact_path)
        print(f"[CACHE] Loaded {len(df):,} {label} rows in {time.time()-t0:.1f}s")
        return df

    if force:
        return None

    superset = _find_superset_cache(prefix, start, end)
    if superset is None:
        return None
    _, path, cache_start, cache_end = superset
    print(
        f"[CACHE] Loading {label} window {start} → {end} from superset "
        f"{os.path.basename(path)} ({cache_start.date()} → {cache_end.date()})"
    )
    t0 = time.time()
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    try:
        df = pd.read_parquet(path, filters=[("date", ">=", start_ts), ("date", "<=", end_ts)])
    except Exception as exc:
        print(f"[CACHE] Filtered parquet read failed for {label}: {exc}; falling back to in-memory slice")
        df = pd.read_parquet(path)
        if "date" in df.index.names:
            dates = pd.to_datetime(df.index.get_level_values("date"))
            df = df.loc[(dates >= start_ts) & (dates <= end_ts)]
    print(f"[CACHE] Loaded {len(df):,} {label} rows in {time.time()-t0:.1f}s")
    return df


def precompute_15m_cache(start='2022-01-04', end='2024-12-31', force=False):
    """
    Precompute 15-minute resampled data and cache to disk.
    Returns the cached DataFrame.
    """
    cache_path = _cache_path('pv_15m', start, end)
    cached = _load_cache_window('pv_15m', '15m data', start, end, force)
    if cached is not None:
        return cached

    print(f"[PREPARE] Precomputing 15m data from {start} to {end}...")
    days = get_trading_days(start, end)
    print(f"[PREPARE] Found {len(days)} trading days")

    all_15m = []
    t0 = time.time()
    for i, day in enumerate(days):
        df_1m = load_single_day_pv(day)
        if df_1m is not None:
            df_15m = resample_1m_to_15m(df_1m)
            if not df_15m.empty:
                all_15m.append(df_15m)
        if (i + 1) % 50 == 0 or i == len(days) - 1:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(days) - i - 1)
            print(f"  [{i+1}/{len(days)}] {day} | elapsed={elapsed:.0f}s | ETA={eta:.0f}s")

    if not all_15m:
        raise RuntimeError("No data loaded. Check data directory.")

    df_combined = pd.concat(all_15m)
    df_combined = df_combined.sort_index()

    print(f"[PREPARE] Saving cache: {len(df_combined):,} rows → {cache_path}")
    df_combined.to_parquet(cache_path, engine='pyarrow')
    print(f"[PREPARE] Done in {time.time()-t0:.1f}s")

    return df_combined


def precompute_resp_cache(start='2022-01-04', end='2024-12-31', force=False):
    """Cache all resp data (for evaluation)."""
    cache_path = _cache_path('resp', start, end)
    cached = _load_cache_window('resp', 'resp', start, end, force)
    if cached is not None:
        return cached

    print(f"[PREPARE] Loading resp data...")
    days = get_trading_days(start, end)
    dfs = []
    for day in days:
        r = load_resp(day)
        if r is not None:
            dfs.append(r)
    if not dfs:
        raise RuntimeError("No resp data found.")
    df = pd.concat(dfs).sort_index()
    df.to_parquet(cache_path, engine='pyarrow')
    print(f"[PREPARE] Cached {len(df):,} resp rows")
    return df


def precompute_tr_cache(start='2022-01-04', end='2024-12-31', force=False):
    """Cache all trading restriction data (for evaluation)."""
    cache_path = _cache_path('tr', start, end)
    cached = _load_cache_window('tr', 'trading restriction', start, end, force)
    if cached is not None:
        return cached

    days = get_trading_days(start, end)
    dfs = []
    for day in days:
        t = load_trading_restriction(day)
        if t is not None:
            dfs.append(t)
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs).sort_index()
    df.to_parquet(cache_path, engine='pyarrow')
    return df


class DataHub:
    """Central data access point. Loads from cache or precomputes."""

    def __init__(self, start='2022-01-04', end='2024-12-31', force=False, use_mock=False):
        self.start = start
        self.end = end
        self._pv_15m = None
        self._resp = None
        self._tr = None
        self._universe = None
        self._force = force
        self._use_mock = use_mock or (
            os.environ.get("AUTOALPHA_MOCK") == "1"
            or os.environ.get("ALPHACLAW_MOCK") == "1"
        )

    @property
    def pv_15m(self):
        if self._pv_15m is None:
            if self._use_mock:
                self._pv_15m = self._generate_mock_pv()
            else:
                self._pv_15m = precompute_15m_cache(self.start, self.end, self._force)
        return self._pv_15m

    @property
    def resp(self):
        if self._resp is None:
            if self._use_mock:
                self._resp = self._generate_mock_resp()
            else:
                self._resp = precompute_resp_cache(self.start, self.end, self._force)
        return self._resp

    @property
    def trading_restriction(self):
        if self._tr is None:
            if self._use_mock:
                self._tr = pd.DataFrame() # Mock no restrictions
            else:
                self._tr = precompute_tr_cache(self.start, self.end, self._force)
        return self._tr

    @property
    def universe(self):
        if self._universe is None:
            if self._use_mock:
                self._universe = self._generate_mock_universe()
            else:
                self._universe = load_universe()
        return self._universe

    def _generate_mock_pv(self):
        print(f"[MOCK] Generating dummy 15m PV data...")
        dates = pd.date_range(self.start, self.end, freq='B')[:60]  # Increased to 60 days
        secs = ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH', '000858.SZ']
        
        rows = []
        for d in dates:
            d_str = d.strftime('%Y-%m-%d')
            times = pd.date_range(f"{d_str} 09:30:00", f"{d_str} 15:00:00", freq='15min', inclusive='left')
            for t in times:
                for s in secs:
                    rows.append({
                        'date': d_str,
                        'datetime': t,
                        'security_id': s,
                        'open_mid_px': np.random.uniform(9, 11),
                        'high_mid_px': np.random.uniform(10, 11),
                        'low_mid_px': np.random.uniform(9, 10),
                        'close_mid_px': np.random.uniform(9, 11),
                        'open_trade_px': np.random.uniform(9, 11),
                        'high_trade_px': np.random.uniform(10, 11),
                        'low_trade_px': np.random.uniform(9, 10),
                        'close_trade_px': np.random.uniform(9, 11),
                        'trade_count': np.random.randint(100, 1000),
                        'volume': np.random.randint(10000, 100000),
                        'dvolume': np.random.uniform(1e5, 1e6),
                        'vwap': np.random.uniform(9, 11)
                    })
        df = pd.DataFrame(rows)
        return df.set_index(['date', 'datetime', 'security_id']).sort_index()

    def _generate_mock_resp(self):
        print(f"[MOCK] Generating dummy resp data...")
        dates = pd.date_range(self.start, self.end, freq='B')[:60] # Increased to 60 days
        secs = ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH', '000858.SZ']
        rows = []
        for d in dates:
            d_str = d.strftime('%Y-%m-%d')
            for s in secs:
                rows.append({'date': d_str, 'security_id': s, 'resp': np.random.uniform(-0.05, 0.05)})
        df = pd.DataFrame(rows)
        return df.set_index(['date', 'security_id']).sort_index()

    def _generate_mock_universe(self):
        print(f"[MOCK] Generating dummy universe...")
        dates = pd.date_range(self.start, self.end, freq='B')[:60] # Increased to 60 days
        secs = ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH', '000858.SZ']
        rows = []
        for d in dates:
            d_str = d.strftime('%Y-%m-%d')
            for s in secs:
                rows.append({'date': d_str, 'security_id': s, 'is_universe': True})
        df = pd.DataFrame(rows)
        return df.set_index(['date', 'security_id']).sort_index()

    def get_field(self, field_name):
        """Get a single field from 15m data as a Series."""
        if field_name in FORBIDDEN_FIELDS:
            raise ValueError(f"COMPLIANCE VIOLATION: '{field_name}' is forbidden for factor construction. "
                           f"It can only be used for evaluation.")
        if field_name not in self.pv_15m.columns:
            raise KeyError(f"Field '{field_name}' not found in data. Available: {list(self.pv_15m.columns)}")
        return self.pv_15m[field_name]

    def get_trading_days_list(self):
        return get_trading_days(self.start, self.end)

    def summary(self):
        """Print data summary."""
        pv = self.pv_15m
        dates = pv.index.get_level_values('date').unique()
        secs = pv.index.get_level_values('security_id').unique()
        return {
            'pv_rows': len(pv),
            'trading_days': len(dates),
            'securities': len(secs),
            'date_range': f"{dates.min()} → {dates.max()}",
            'fields': list(pv.columns),
        }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Precompute and cache 15-minute data')
    parser.add_argument('--force', action='store_true', help='Force re-cache')
    parser.add_argument('--start', default='2022-01-04')
    parser.add_argument('--end', default='2024-12-31')
    args = parser.parse_args()

    hub = DataHub(args.start, args.end, args.force)
    print("\n=== Data Summary ===")
    for k, v in hub.summary().items():
        print(f"  {k}: {v}")

    print("\n=== Loading Resp (eval-only) ===")
    resp = hub.resp
    print(f"  Resp rows: {len(resp):,}")

    print("\n=== Loading Universe ===")
    uni = hub.universe
    print(f"  Universe rows: {len(uni):,}")

    print("\n✅ All data loaded and cached successfully.")
