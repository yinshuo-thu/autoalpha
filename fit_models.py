"""
fit_models.py — Model Lab (Real Data)

Trains simple models (LinearRegression, LightGBM, MLP) using factor exposures
to predict future returns (resp). Splits data roughly into:
- 2022: Train
- 2023: Val
- 2024: Test

Returns dict of evaluation metrics per split per model.
"""
import os
import json
import warnings
import pandas as pd
import numpy as np

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    import lightgbm as lgb
    HAS_ML = True
except ImportError:
    HAS_ML = False

from prepare_data import DataHub

def prepare_modeling_data(factor_series, data_hub):
    """Aligns factor values (X) with resp (y) and splits by year."""
    if factor_series is None or factor_series.empty:
        return None
        
    resp_series = data_hub.resp['resp']
    
    # Inner join on index (date, datetime, security_id)
    df = pd.DataFrame({'factor': factor_series, 'resp': resp_series}).dropna()
    
    if len(df) < 100:
        return None
        
    dates = df.index.get_level_values('date')
    
    # Split: Train <= 2022-12-31, Val = 2023, Test >= 2024
    train_mask = dates <= '2022-12-31'
    val_mask = (dates >= '2023-01-01') & (dates <= '2023-12-31')
    test_mask = dates >= '2024-01-01'
    
    return {
        'train': (df[train_mask][['factor']], df[train_mask]['resp']),
        'val': (df[val_mask][['factor']], df[val_mask]['resp']),
        'test': (df[test_mask][['factor']], df[test_mask]['resp'])
    }


def evaluate_predictions(y_true, y_pred):
    """Compute IC, Rank IC, and Loss."""
    if len(y_true) < 2:
        return {'ic': 0.0, 'rank_ic': 0.0, 'loss': 0.0}
        
    # Correlation (IC)
    idx = np.isfinite(y_true) & np.isfinite(y_pred)
    y_t, y_p = y_true[idx], y_pred[idx]
    
    if len(y_t) < 2:
        return {'ic': 0.0, 'rank_ic': 0.0, 'loss': 0.0}
        
    ic = np.corrcoef(y_t, y_p)[0, 1]
    if np.isnan(ic): ic = 0.0
    
    # Rank IC
    t_rank = pd.Series(y_t).rank().values
    p_rank = pd.Series(y_p).rank().values
    rank_ic = np.corrcoef(t_rank, p_rank)[0, 1]
    if np.isnan(rank_ic): rank_ic = 0.0
    
    # MSE Loss
    loss = mean_squared_error(y_t, y_p)
    
    return {'ic': float(ic), 'rank_ic': float(rank_ic), 'loss': float(loss)}


def fit_and_evaluate_models(factor_series, data_hub, factor_name):
    """Fit Linear and LightGBM models, evaluate Train/Val/Test."""
    result = {
        'factor_name': factor_name,
        'status': 'error',
        'models': {},
        'best_model': None
    }
    
    if not HAS_ML:
        result['error'] = 'scikit-learn or lightgbm not installed.'
        return result
        
    splits = prepare_modeling_data(factor_series, data_hub)
    if not splits:
        result['error'] = 'Not enough overlapping data or factor evaluation failed.'
        return result
        
    X_train, y_train = splits['train']
    X_val, y_val = splits['val']
    X_test, y_test = splits['test']
    
    if len(X_train) == 0 or len(X_val) == 0:
        result['error'] = 'Missing train or val data (check date ranges).'
        return result

    # 1. Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    
    lr_res = {
        'train': evaluate_predictions(y_train.values, lr.predict(X_train)),
        'val': evaluate_predictions(y_val.values, lr.predict(X_val)),
        'test': evaluate_predictions(y_test.values, lr.predict(X_test)) if len(X_test) > 0 else None
    }
    result['models']['LinearRegression'] = lr_res
    
    # 2. LightGBM (Tree)
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)
    
    params = {
        'objective': 'regression', 'metric': 'rmse',
        'learning_rate': 0.05, 'max_depth': 3, 'num_leaves': 15,
        'verbose': -1, 'seed': 42
    }
    
    gbm = lgb.train(params, lgb_train, num_boost_round=50,
                    valid_sets=[lgb_val], callbacks=[lgb.early_stopping(10, verbose=False)])
                    
    xgb_res = {
        'train': evaluate_predictions(y_train.values, gbm.predict(X_train)),
        'val': evaluate_predictions(y_val.values, gbm.predict(X_val)),
        'test': evaluate_predictions(y_test.values, gbm.predict(X_test)) if len(X_test) > 0 else None
    }
    result['models']['LightGBM'] = xgb_res
    
    # Select best model based on Val IC
    best_name = 'LinearRegression'
    best_val_ic = lr_res['val']['ic']
    
    if xgb_res['val']['ic'] > best_val_ic:
        best_name = 'LightGBM'
        
    result['best_model'] = best_name
    result['status'] = 'success'
    
    # Flag in leaderboard
    from leaderboard import load_leaderboard, save_leaderboard
    lb = load_leaderboard()
    for f in lb.get('factors', []):
        if f['factor_name'] == factor_name:
            f['model_ready_flag'] = True
            break
    save_leaderboard(lb)
    
    return result
