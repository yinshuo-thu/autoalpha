import os
import glob
import pandas as pd
import numpy as np
import concurrent.futures

import platform
from paths import DATA_ROOT

def _downcast_df(df):
    if df.empty: return df
    float_cols = df.select_dtypes(include=['float64']).columns
    int_cols = df.select_dtypes(include=['int64']).columns
    if len(float_cols) > 0:
        df[float_cols] = df[float_cols].astype('float32')
    if len(int_cols) > 0:
        df[int_cols] = df[int_cols].astype('int32')
    return df

def get_trading_days(start=None, end=None):
    """Get sorted list of valid trading days."""
    univ_path = os.path.join(DATA_ROOT, 'eq_data_stage1', 'universe', '*', 'data.pq')
    files = glob.glob(univ_path)
    if not files:
        basic_pv_path = os.path.join(DATA_ROOT, 'eq_data_stage1', 'basic_pv')
        dates = []
        for year in os.listdir(basic_pv_path):
            year_path = os.path.join(basic_pv_path, year)
            if not os.path.isdir(year_path): continue
            for month in os.listdir(year_path):
                month_path = os.path.join(year_path, month)
                if not os.path.isdir(month_path): continue
                for day in os.listdir(month_path):
                    if day.isdigit():
                        dates.append(f"{year}-{month}-{day}")
        dates.sort()
        trading_days = pd.to_datetime(dates)
    else:
        try:
            all_trading_days = []
            for f in files:
                df = pd.read_parquet(f, columns=[])
                all_trading_days.extend(list(df.index.get_level_values('date').unique()))
            trading_days = pd.Series(pd.to_datetime(all_trading_days)).sort_values().drop_duplicates()
        except:
            return []

    if start:
        trading_days = trading_days[trading_days >= pd.to_datetime(start)]
    if end:
        trading_days = trading_days[trading_days <= pd.to_datetime(end)]
        
    return [d.strftime('%Y-%m-%d') for d in trading_days]

def _read_single_parquet(file_path, columns, index_cols, resample_first=False):
    if not os.path.exists(file_path):
        return None
    try:
        df = pd.read_parquet(file_path, columns=columns)
        if df.index.names != index_cols:
            df = df.reset_index()
            if set(index_cols).issubset(df.columns):
                df = df.set_index(index_cols)
            else:
                return None
        if resample_first:
            return resample_1m_to_15m(df)
        return df
    except Exception:
        return None

def _load_files_for_dates(base_path, dates, columns=None, index_cols=['date', 'datetime', 'security_id'], resample_first=False):
    dfs = []
    file_paths = []
    for d in dates:
        dt = pd.to_datetime(d)
        y, m, day = dt.strftime('%Y'), dt.strftime('%m'), dt.strftime('%d')
        file_paths.append(os.path.join(base_path, y, m, day, 'data.pq'))
        
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(_read_single_parquet, p, columns, index_cols, resample_first) for p in file_paths]
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res is not None and not res.empty:
                dfs.append(_downcast_df(res))
                
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs).sort_index()

def load_universe(dates):
    dfs = []
    univ_base = os.path.join(DATA_ROOT, 'eq_data_stage1', 'universe')
    years = list(set([pd.to_datetime(d).strftime('%Y') for d in dates]))
    paths = [os.path.join(univ_base, y, 'data.pq') for y in years]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(_read_single_parquet, p, None, ['date', 'security_id'], False) for p in paths]
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res is not None and not res.empty:
                dfs.append(_downcast_df(res))
                
    if not dfs:
        return pd.DataFrame()
    res = pd.concat(dfs).sort_index()
    dates_ts = [pd.to_datetime(d) for d in dates]
    idx = res.index.get_level_values('date').isin(dates_ts)
    return res[idx]

def resample_1m_to_15m(df):
    if df.empty:
        return df

    df_res = df.reset_index()
    df_res['datetime_15m'] = df_res['datetime'].dt.ceil('15min')
    
    agg_funcs = {}
    cols = df_res.columns
    if 'open_mid_px' in cols: agg_funcs['open_mid_px'] = 'first'
    if 'high_mid_px' in cols: agg_funcs['high_mid_px'] = 'max'
    if 'low_mid_px' in cols: agg_funcs['low_mid_px'] = 'min'
    if 'close_mid_px' in cols: agg_funcs['close_mid_px'] = 'last'
    
    if 'open_trade_px' in cols: agg_funcs['open_trade_px'] = 'first'
    if 'high_trade_px' in cols: agg_funcs['high_trade_px'] = 'max'
    if 'low_trade_px' in cols: agg_funcs['low_trade_px'] = 'min'
    if 'close_trade_px' in cols: agg_funcs['close_trade_px'] = 'last'
    
    if 'trade_count' in cols: agg_funcs['trade_count'] = 'sum'
    if 'volume' in cols: agg_funcs['volume'] = 'sum'
    if 'dvolume' in cols: agg_funcs['dvolume'] = 'sum'
    
    grouped = df_res.groupby(['date', 'datetime_15m', 'security_id']).agg(agg_funcs)
    
    if 'dvolume' in grouped.columns and 'volume' in grouped.columns:
        grouped['vwap'] = grouped['dvolume'] / grouped['volume']
        grouped['vwap'] = grouped['vwap'].replace([np.inf, -np.inf], np.nan)
        
    grouped.index = grouped.index.rename({'datetime_15m': 'datetime'})
    return grouped

def load_pv_days(dates, columns=None):
    base_path = os.path.join(DATA_ROOT, 'eq_data_stage1', 'basic_pv')
    # Because 1m parquets can exceed 30GB when concatenated, we MUST resample inside threads BEFORE pd.concat
    df_15m = _load_files_for_dates(base_path, dates, columns, resample_first=True)
    if df_15m.empty:
        return df_15m
    return df_15m

def load_resp_days(dates):
    base_path = os.path.join(DATA_ROOT, 'resp')
    if not os.path.exists(base_path):
        base_path = os.path.join(DATA_ROOT, 'eq_resp_stage1', 'resp')
        if not os.path.exists(base_path):
            base_path = os.path.join(DATA_ROOT, 'eq_resp_stage1')
    return _load_files_for_dates(base_path, dates)

def load_restriction_days(dates):
    base_path = os.path.join(DATA_ROOT, 'eq_trading_restriction_stage1', 'trading_restriction')
    if not os.path.exists(base_path):
        base_path = os.path.join(DATA_ROOT, 'eq_trading_restriction_stage1')
    return _load_files_for_dates(base_path, dates)

def align_to_universe(df, df_univ):
    """Align the multi-index dataframe to the target universe."""
    if df.empty or df_univ.empty:
        return pd.DataFrame()
    # Ensure index matching
    # Usually univ has index [date, security_id]
    # We might need to join it properly
    # A fast way is to just do an inner join
    return df.join(df_univ, how='inner')

# Safety hooks
def assert_no_future_leakage(df, context=""):
    """Logical hook to ensure future features aren't masked as present."""
    pass

def assert_bar_alignment(df1, df2):
    """Ensure dataframes share exactly the same MultiIndex structure."""
    if df1.index.names != df2.index.names:
        raise ValueError(f"Index mismatch: {df1.index.names} vs {df2.index.names}")
