"""
prepare_data.py — Futures Real Data Loader & 15-Minute Resampler

Loads DCE futures order-flow reconstruction parquet files from
AUTOALPHA_OFR_ROOT, aggregates tick-level contract records to 15-minute bars,
and caches the result. Raw DCE CSV snapshots under AUTOALPHA_FUTURE_RAW_ROOT are
kept as the original-data reference. Also builds a contract universe, next-day
return resp (eval-only), and no-op trading_restriction (eval-only).

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
import warnings
import pandas as pd
import numpy as np
from datetime import datetime
from paths import DATA_ROOT, CACHE_ROOT, FUTURE_RAW_ROOT, OFR_ROOT

warnings.filterwarnings('ignore', category=FutureWarning)

CACHE_DIR = CACHE_ROOT
os.makedirs(CACHE_DIR, exist_ok=True)

# ── Allowed fields (competition-safe for factor construction) ──
ALLOWED_FIELDS = [
    'open_mid_px', 'high_mid_px', 'low_mid_px', 'close_mid_px',
    'open_trade_px', 'high_trade_px', 'low_trade_px', 'close_trade_px',
    'trade_count', 'volume', 'dvolume', 'vwap',
    'open_interest', 'delta_oi', 'buy_volume', 'sell_volume', 'open_volume',
    'close_volume', 'market_ofi', 'add_ofi', 'cancel_ofi', 'book_ofi',
    'book_imbalance', 'spread', 'cvd',
]

# ── FORBIDDEN for factor construction (eval-only) ──
FORBIDDEN_FIELDS = ['resp', 'trading_restriction']


def futures_mode():
    return os.environ.get("AUTOALPHA_ASSET_CLASS", "futures").strip().lower() in {"future", "futures"}


def _future_products():
    raw = os.environ.get("AUTOALPHA_FUTURE_PRODUCTS", "C,LH,M").strip()
    if not raw or raw == "*":
        return None
    return [p.strip().upper() for p in raw.split(",") if p.strip()]


def _future_summary():
    summary_path = os.path.join(OFR_ROOT, "reports", "checks", "contract_summary.csv")
    if not os.path.exists(summary_path):
        files = glob.glob(os.path.join(OFR_ROOT, "*", "_contract_summary.csv"))
        if not files:
            return pd.DataFrame()
        dfs = []
        for path in files:
            try:
                dfs.append(pd.read_csv(path))
            except Exception:
                pass
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    try:
        return pd.read_csv(summary_path)
    except Exception:
        return pd.DataFrame()


def _normalize_yyyymmdd(value):
    text = str(value)
    if len(text) == 8 and text.isdigit():
        return f"{text[:4]}-{text[4:6]}-{text[6:8]}"
    return pd.to_datetime(text).strftime("%Y-%m-%d")


def _future_available_dates():
    dates = set()
    summary = _future_summary()
    products = _future_products()
    if not summary.empty and {"product", "first_date", "last_date"}.issubset(summary.columns):
        if products:
            summary = summary[summary["product"].astype(str).str.upper().isin(products)]
        for _, row in summary.iterrows():
            try:
                rng = pd.date_range(_normalize_yyyymmdd(row["first_date"]), _normalize_yyyymmdd(row["last_date"]), freq="D")
                dates.update(d.strftime("%Y-%m-%d") for d in rng)
            except Exception:
                continue

    raw_day_dirs = glob.glob(os.path.join(FUTURE_RAW_ROOT, "[0-9]" * 8))
    for path in raw_day_dirs:
        name = os.path.basename(path)
        if len(name) == 8 and name.isdigit():
            dates.add(_normalize_yyyymmdd(name))
    return sorted(dates)


def _future_default_start_end(start=None, end=None):
    dates = _future_available_dates()
    if not dates:
        return start or "2025-09-01", end or "2026-03-31"
    if start is None or start == "2022-01-04":
        days = int(os.environ.get("AUTOALPHA_FUTURE_DEFAULT_DAYS", "20") or 20)
        start = dates[max(0, len(dates) - days)]
    if end is None or end == "2024-12-31":
        end = dates[-1]
    return start, end


def _future_contract_files(start, end):
    summary = _future_summary()
    products = _future_products()
    if not summary.empty and "path" in summary.columns:
        df = summary.copy()
        if products and "product" in df.columns:
            df = df[df["product"].astype(str).str.upper().isin(products)]
        if "status" in df.columns:
            df = df[df["status"].astype(str).str.lower().eq("ok")]
        if {"first_date", "last_date"}.issubset(df.columns):
            start_i = int(str(start).replace("-", ""))
            end_i = int(str(end).replace("-", ""))
            df = df[(df["last_date"].astype(int) >= start_i) & (df["first_date"].astype(int) <= end_i)]
        paths = [p for p in df["path"].astype(str).tolist() if os.path.exists(p)]
        return sorted(paths)
    pattern = os.path.join(OFR_ROOT, "*", "*.parquet")
    paths = [p for p in glob.glob(pattern) if not os.path.basename(p).startswith("_")]
    if products:
        paths = [p for p in paths if os.path.basename(os.path.dirname(p)).upper() in products]
    return sorted(paths)


def _future_bar_from_ofr(path, start, end):
    columns = [
        "trading_date", "timestamp", "contract", "price", "volume", "amount",
        "vwap", "open_interest", "delta_oi", "mid", "spread", "buy_volume",
        "sell_volume", "open_volume", "close_volume", "market_ofi", "add_ofi",
        "cancel_ofi", "book_ofi", "book_imbalance", "cvd",
    ]
    start_key = str(start).replace("-", "")
    end_key = str(end).replace("-", "")
    try:
        df = pd.read_parquet(
            path,
            columns=columns,
            filters=[("trading_date", ">=", start_key), ("trading_date", "<=", end_key)],
        )
    except Exception as exc:
        try:
            df = pd.read_parquet(path, columns=columns)
        except Exception as exc2:
            print(f"Warning: Failed to read OFR parquet {path}: {exc2 or exc}")
            return pd.DataFrame()
    if df.empty:
        return df
    date_s = df["trading_date"].astype(str).map(_normalize_yyyymmdd)
    mask = (date_s >= start) & (date_s <= end)
    df = df.loc[mask].copy()
    if df.empty:
        return df
    df["date"] = date_s.loc[mask].values
    df["datetime"] = pd.to_datetime(df["timestamp"]).dt.floor("15min")
    df["security_id"] = df["contract"].astype(str)
    df["dvolume"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    if "mid" not in df or df["mid"].isna().all():
        df["mid"] = df["price"]

    sum_cols = [
        "volume", "dvolume", "buy_volume", "sell_volume", "open_volume",
        "close_volume", "market_ofi", "add_ofi", "cancel_ofi", "book_ofi",
        "delta_oi",
    ]
    last_cols = ["open_interest", "spread", "book_imbalance", "cvd"]
    agg = {
        "price": ["first", "max", "min", "last"],
        "mid": ["first", "max", "min", "last"],
    }
    for col in sum_cols:
        if col in df.columns:
            agg[col] = "sum"
    for col in last_cols:
        if col in df.columns:
            agg[col] = "last"
    grouped = df.groupby(["date", "datetime", "security_id"], sort=True).agg(agg)
    grouped.columns = [
        "_".join(c).rstrip("_") if isinstance(c, tuple) else c for c in grouped.columns
    ]
    grouped = grouped.rename(columns={
        "price_first": "open_trade_px",
        "price_max": "high_trade_px",
        "price_min": "low_trade_px",
        "price_last": "close_trade_px",
        "mid_first": "open_mid_px",
        "mid_max": "high_mid_px",
        "mid_min": "low_mid_px",
        "mid_last": "close_mid_px",
    })
    for col in sum_cols + last_cols:
        sum_name = f"{col}_sum"
        last_name = f"{col}_last"
        if sum_name in grouped.columns and col not in grouped.columns:
            grouped = grouped.rename(columns={sum_name: col})
        if last_name in grouped.columns and col not in grouped.columns:
            grouped = grouped.rename(columns={last_name: col})
    grouped["trade_count"] = df.groupby(["date", "datetime", "security_id"], sort=True).size().astype("float32")
    vol = grouped["volume"].replace(0, np.nan)
    grouped["vwap"] = (grouped["dvolume"] / vol).where(vol.notna(), grouped["close_trade_px"])
    return grouped[[c for c in ALLOWED_FIELDS if c in grouped.columns]].sort_index()


def precompute_future_15m_cache(start=None, end=None, force=False):
    start, end = _future_default_start_end(start, end)
    products = ",".join(_future_products() or ["ALL"]).replace(",", "_")
    cache_path = os.path.join(CACHE_DIR, f"future_ofr_15m_{products}_{start}_{end}.parquet")
    if os.path.exists(cache_path) and not force:
        print(f"[CACHE] Loading cached futures 15m data from {cache_path}")
        return pd.read_parquet(cache_path)
    paths = _future_contract_files(start, end)
    print(f"[PREPARE] Futures OFR 15m cache {start} to {end}; contracts={len(paths)}")
    frames = []
    for i, path in enumerate(paths, 1):
        frame = _future_bar_from_ofr(path, start, end)
        if not frame.empty:
            frames.append(frame)
        if i % 5 == 0 or i == len(paths):
            print(f"  [{i}/{len(paths)}] {os.path.basename(path)}")
    if not frames:
        raise RuntimeError(f"No futures OFR data loaded from {OFR_ROOT}. Check AUTOALPHA_OFR_ROOT/AUTOALPHA_FUTURE_PRODUCTS.")
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df.astype({c: "float32" for c in df.select_dtypes(include=["float64"]).columns})
    os.makedirs(CACHE_DIR, exist_ok=True)
    df.to_parquet(cache_path, engine="pyarrow")
    print(f"[PREPARE] Cached futures bars: {len(df):,} rows -> {cache_path}")
    return df


def precompute_future_resp_cache(start=None, end=None, force=False):
    start, end = _future_default_start_end(start, end)
    products = ",".join(_future_products() or ["ALL"]).replace(",", "_")
    cache_path = os.path.join(CACHE_DIR, f"future_resp_{products}_{start}_{end}.parquet")
    if os.path.exists(cache_path) and not force:
        return pd.read_parquet(cache_path)
    bars = precompute_future_15m_cache(start, end, force=False)
    daily = (
        bars["close_trade_px"]
        .groupby(["date", "security_id"], sort=True)
        .last()
        .unstack("security_id")
        .sort_index()
    )
    resp = daily.shift(-1).div(daily).sub(1.0).stack().rename("resp").to_frame()
    resp.index = resp.index.set_names(["date", "security_id"])
    resp.to_parquet(cache_path, engine="pyarrow")
    return resp


def precompute_future_tr_cache(start=None, end=None, force=False):
    start, end = _future_default_start_end(start, end)
    products = ",".join(_future_products() or ["ALL"]).replace(",", "_")
    cache_path = os.path.join(CACHE_DIR, f"future_tr_{products}_{start}_{end}.parquet")
    if os.path.exists(cache_path) and not force:
        return pd.read_parquet(cache_path)
    resp = precompute_future_resp_cache(start, end, force=False)
    tr = pd.DataFrame({"trading_restriction": 0.0}, index=resp.index)
    tr.to_parquet(cache_path, engine="pyarrow")
    return tr


def load_future_universe(start=None, end=None):
    bars = precompute_future_15m_cache(start, end, force=False)
    idx = bars.reset_index()[["date", "security_id"]].drop_duplicates()
    idx["is_universe"] = True
    return idx.set_index(["date", "security_id"]).sort_index()


def get_trading_days(start='2022-01-04', end='2024-12-31'):
    """Get all trading days from basic_pv directory structure."""
    if futures_mode():
        start, end = _future_default_start_end(start, end)
        return [d for d in _future_available_dates() if start <= d <= end]

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


def precompute_15m_cache(start='2022-01-04', end='2024-12-31', force=False):
    """
    Precompute 15-minute resampled data and cache to disk.
    Returns the cached DataFrame.
    """
    if futures_mode():
        return precompute_future_15m_cache(start, end, force)

    cache_path = os.path.join(CACHE_DIR, f'pv_15m_{start}_{end}.parquet')

    if os.path.exists(cache_path) and not force:
        print(f"[CACHE] Loading cached 15m data from {cache_path}")
        t0 = time.time()
        df = pd.read_parquet(cache_path)
        print(f"[CACHE] Loaded {len(df):,} rows in {time.time()-t0:.1f}s")
        return df

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
    if futures_mode():
        return precompute_future_resp_cache(start, end, force)

    cache_path = os.path.join(CACHE_DIR, f'resp_{start}_{end}.parquet')

    if os.path.exists(cache_path) and not force:
        print(f"[CACHE] Loading cached resp from {cache_path}")
        return pd.read_parquet(cache_path)

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
    if futures_mode():
        return precompute_future_tr_cache(start, end, force)

    cache_path = os.path.join(CACHE_DIR, f'tr_{start}_{end}.parquet')

    if os.path.exists(cache_path) and not force:
        return pd.read_parquet(cache_path)

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
        if futures_mode():
            start, end = _future_default_start_end(start, end)
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
            elif futures_mode():
                self._universe = load_future_universe(self.start, self.end)
            else:
                self._universe = load_universe()
        return self._universe

    def _generate_mock_pv(self):
        print(f"[MOCK] Generating dummy 15m PV data...")
        dates = pd.date_range(self.start, self.end, freq='B')[:60]  # Increased to 60 days
        secs = ['c2601', 'c2603', 'm2601', 'm2603', 'lh2601'] if futures_mode() else ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH', '000858.SZ']
        
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
                        'vwap': np.random.uniform(9, 11),
                        'open_interest': np.random.randint(100000, 500000),
                        'delta_oi': np.random.randint(-1000, 1000),
                        'buy_volume': np.random.randint(1000, 50000),
                        'sell_volume': np.random.randint(1000, 50000),
                        'open_volume': np.random.randint(1000, 50000),
                        'close_volume': np.random.randint(1000, 50000),
                        'market_ofi': np.random.uniform(-1e5, 1e5),
                        'add_ofi': np.random.uniform(-1e5, 1e5),
                        'cancel_ofi': np.random.uniform(-1e5, 1e5),
                        'book_ofi': np.random.uniform(-1e5, 1e5),
                        'book_imbalance': np.random.uniform(-1, 1),
                        'spread': np.random.uniform(1, 3),
                        'cvd': np.random.uniform(-1e6, 1e6),
                    })
        df = pd.DataFrame(rows)
        return df.set_index(['date', 'datetime', 'security_id']).sort_index()

    def _generate_mock_resp(self):
        print(f"[MOCK] Generating dummy resp data...")
        dates = pd.date_range(self.start, self.end, freq='B')[:60] # Increased to 60 days
        secs = ['c2601', 'c2603', 'm2601', 'm2603', 'lh2601'] if futures_mode() else ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH', '000858.SZ']
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
        secs = ['c2601', 'c2603', 'm2601', 'm2603', 'lh2601'] if futures_mode() else ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH', '000858.SZ']
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
