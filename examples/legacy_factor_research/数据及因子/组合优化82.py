#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
商品期货量化策略系统 - 完全无未来函数版本 + 扩展因子库
整合autodl-tmp/factors/factors.py中的所有因子
所有潜在的未来函数已被识别并修复
"""
# 在文件开头的导入部分添加
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner, HyperbandPruner
import json
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import sys
from typing import List, Dict, Any, Optional, Tuple
import catboost as cb
from catboost import CatBoostRegressor, Pool
import lightgbm as lgb
from scipy import stats
import pickle
import hashlib
import time
from collections import defaultdict
from multiprocessing import Pool as MPPool, cpu_count, Manager
from concurrent.futures import ProcessPoolExecutor, as_completed
import psutil
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from enum import Enum
import joblib
import traceback
import functools
import logging
import pyarrow.parquet as pq
from tqdm import tqdm
import talib
    
# 动态导入factors模块
sys.path.append(os.path.expanduser('~/autodl-tmp/factors/'))
try:
   import factors
   FACTORS_AVAILABLE = True
   logger_info = f"Successfully imported factors module with {len(factors.func_list)} factors"
except ImportError as e:
   FACTORS_AVAILABLE = False
   logger_info = f"Warning: Could not import factors module: {e}"
    
# 设置日志
logging.basicConfig(
   level=logging.INFO,
   format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.info(logger_info)
    
# ================== 内存监控装饰器 ==================
def monitor_memory(func):
   """监控函数内存使用"""
   @functools.wraps(func)
   def wrapper(*args, **kwargs):
       gc.collect()
       process = psutil.Process()
       mem_before = process.memory_info().rss / (1024 * 1024)
           
       try:
           result = func(*args, **kwargs)
       except Exception as e:
           logger.error(f"{func.__name__} 执行失败: {e}")
           raise
           
       mem_after = process.memory_info().rss / (1024 * 1024)
       mem_used = mem_after - mem_before
           
       if mem_used > 100:
           logger.info(f"{func.__name__} 内存使用: {mem_used:.1f} MB")
           
       if mem_used > 1000:
           gc.collect()
           logger.info("执行垃圾回收...")
           
       return result
   return wrapper
    
# ================== 数据结构定义 ==================
@dataclass
class TradeSignal:
   """交易信号"""
   symbol: str
   datetime: pd.Timestamp
   signal: float
   position: float
   confidence: float
    
@dataclass
class Position:
   """持仓信息"""
   symbol: str
   quantity: float
   entry_price: float
   entry_time: pd.Timestamp
   current_price: float
   unrealized_pnl: float
   stop_loss: float
   take_profit: float
   trailing_stop: float
   highest_price: float
   lowest_price: float
    
@dataclass
class Trade:
   """交易记录"""
   symbol: str
   entry_time: pd.Timestamp
   exit_time: pd.Timestamp
   entry_price: float
   exit_price: float
   quantity: float
   pnl: float
   commission: float
   net_pnl: float
   slippage: float
   exit_reason: str
    
class SignalType(Enum):
   """信号类型"""
   LONG = 1
   SHORT = -1
   NEUTRAL = 0
# ================== Optuna超参数优化器 ==================
class OptunaHyperparameterOptimizer:
    """使用Optuna进行超参数优化，确保无未来函数"""
        
    def __init__(self, use_gpu: bool = True, n_trials: int = 50):
        self.use_gpu = use_gpu
        self.n_trials = n_trials
        self.best_params = {}
        self.study = None
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None
            
    def prepare_validation_split(self, X_train, y_train, val_ratio=0.2):
        """时间序列验证集分割，保持时间顺序"""
        val_size = int(len(X_train) * val_ratio)
        train_size = len(X_train) - val_size
            
        # 时间序列分割 - 验证集在训练集之后
        self.X_train = X_train[:train_size]
        self.X_val = X_train[train_size:]
        self.y_train = y_train[:train_size]
        self.y_val = y_train[train_size:]
            
        logger.info(f"Optuna数据分割: 训练集 {len(self.X_train)}, 验证集 {len(self.X_val)}")
        
    def objective_xgboost(self, trial):
        """XGBoost目标函数"""
        params = {
            'objective': 'reg:squarederror',
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 1.0, log=True),
            'gamma': trial.suggest_float('gamma', 0.01, 1.0, log=True),
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0
        }
            
        if self.use_gpu:
            params.update({
                'tree_method': 'gpu_hist',
                'predictor': 'gpu_predictor'
            })
            
        # 训练模型
        dtrain = xgb.DMatrix(self.X_train, label=self.y_train)
        dval = xgb.DMatrix(self.X_val, label=self.y_val)
            
        # 使用早停进行剪枝
        pruning_callback = optuna.integration.XGBoostPruningCallback(trial, 'eval-rmse')
            
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=params['n_estimators'],
            evals=[(dval, 'eval')],
            early_stopping_rounds=20,
            verbose_eval=False,
            callbacks=[pruning_callback]
        )
            
        # 预测验证集
        pred = model.predict(dval)
            
        # 计算多个指标
        rmse = np.sqrt(np.mean((self.y_val - pred) ** 2))
        correlation = np.corrcoef(pred, self.y_val)[0, 1] if len(pred) > 1 else 0
        direction_acc = np.mean(np.sign(pred) == np.sign(self.y_val))
            
        # 综合评分（可调整权重）
        score = direction_acc
            
        return score
        
    def objective_lightgbm(self, trial):
        """LightGBM目标函数"""
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 10, 100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 50),
            'lambda_l1': trial.suggest_float('lambda_l1', 0.001, 1.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 0.001, 1.0, log=True),
            'verbose': -1,
            'random_state': 42,
            'n_jobs': -1
        }
    
        train_data = lgb.Dataset(self.X_train, label=self.y_train)
        val_data = lgb.Dataset(self.X_val, label=self.y_val, reference=train_data)
    
        # 移除pruning_callback，或者创建一个自定义的回调
        model = lgb.train(
            params,
            train_data,
            num_boost_round=300,
            valid_sets=[val_data],
            valid_names=['valid_0'],
            callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]  # 移除pruning_callback
        )
    
        pred = model.predict(self.X_val)
    
        rmse = np.sqrt(np.mean((self.y_val - pred) ** 2))
        correlation = np.corrcoef(pred, self.y_val)[0, 1] if len(pred) > 1 else 0
        direction_acc = np.mean(np.sign(pred) == np.sign(self.y_val))
    
        # 综合评分
        score = direction_acc
    
        # 手动实现剪枝逻辑
        trial.report(score, model.best_iteration)
        if trial.should_prune():
            raise optuna.TrialPruned()
    
        return score
        
    def objective_catboost(self, trial):
        """CatBoost目标函数"""
        params = {
            'iterations': trial.suggest_int('iterations', 100, 500, step=50),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'depth': trial.suggest_int('depth', 3, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 50),
            'random_strength': trial.suggest_float('random_strength', 0.5, 2.0),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.5, 1.0),
            'verbose': False,
            'random_state': 42,
            'thread_count': -1,
            'early_stopping_rounds': 20
        }
            
        if self.use_gpu:
            params.update({
                'task_type': 'GPU',
                'devices': '0'
            })
            
        train_pool = Pool(self.X_train, self.y_train)
        val_pool = Pool(self.X_val, self.y_val)
            
        model = CatBoostRegressor(**params)
        model.fit(train_pool, eval_set=val_pool, verbose=False)
            
        pred = model.predict(self.X_val)
            
        rmse = np.sqrt(np.mean((self.y_val - pred) ** 2))
        correlation = np.corrcoef(pred, self.y_val)[0, 1] if len(pred) > 1 else 0
        direction_acc = np.mean(np.sign(pred) == np.sign(self.y_val))
            
        score = direction_acc
            
        # 报告中间结果用于剪枝
        trial.report(score, model.get_best_iteration())
        if trial.should_prune():
            raise optuna.TrialPruned()
            
        return score
        
    def optimize(self, X_train, y_train, model_type='all'):
        """执行超参数优化"""
        self.prepare_validation_split(X_train, y_train)
            
        results = {}
            
        if model_type in ['xgboost', 'all']:
            logger.info("优化XGBoost超参数...")
            study_xgb = optuna.create_study(
                direction='maximize',
                sampler=TPESampler(seed=42),
                pruner=HyperbandPruner()
            )
            study_xgb.optimize(self.objective_xgboost, n_trials=self.n_trials, n_jobs=1)
            results['xgboost'] = study_xgb.best_params
            logger.info(f"XGBoost最佳参数: {study_xgb.best_params}")
            logger.info(f"XGBoost最佳得分: {study_xgb.best_value:.4f}")
            
        if model_type in ['lightgbm', 'all']:
            logger.info("优化LightGBM超参数...")
            study_lgb = optuna.create_study(
                direction='maximize',
                sampler=TPESampler(seed=42),
                pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=10)
            )
            study_lgb.optimize(self.objective_lightgbm, n_trials=self.n_trials, n_jobs=1)
            results['lightgbm'] = study_lgb.best_params
            logger.info(f"LightGBM最佳参数: {study_lgb.best_params}")
            logger.info(f"LightGBM最佳得分: {study_lgb.best_value:.4f}")
            
        if model_type in ['catboost', 'all']:
            logger.info("优化CatBoost超参数...")
            study_cb = optuna.create_study(
                direction='maximize',
                sampler=TPESampler(seed=42),
                pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=10)
            )
            study_cb.optimize(self.objective_catboost, n_trials=self.n_trials, n_jobs=1)
            results['catboost'] = study_cb.best_params
            logger.info(f"CatBoost最佳参数: {study_cb.best_params}")
            logger.info(f"CatBoost最佳得分: {study_cb.best_value:.4f}")
            
        self.best_params = results
        return results
        
    def save_best_params(self, filepath):
        """保存最佳参数"""
        with open(filepath, 'w') as f:
            json.dump(self.best_params, f, indent=2)
        logger.info(f"最佳参数已保存至: {filepath}")
        
    def load_best_params(self, filepath):
        """加载最佳参数"""
        with open(filepath, 'r') as f:
            self.best_params = json.load(f)
        logger.info(f"从 {filepath} 加载最佳参数")
        return self.best_params
# ================== 完全安全的技术指标计算（无未来函数）==================
def calculate_safe_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
   """
   计算技术指标，确保不使用未来数据
   所有指标只使用历史数据计算
   所有rolling操作明确设置center=False
   """
   logger.info(f"计算安全技术指标，数据形状: {df.shape}")
       
   # 确保必要的列存在
   required_cols = ['open', 'high', 'low', 'close', 'volume']
   for col in required_cols:
       if col not in df.columns:
           logger.error(f"缺少必需列: {col}")
           return df
       
   # 价格和成交量数据
   open_price = df['open'].values
   high_price = df['high'].values
   low_price = df['low'].values
   close_price = df['close'].values
   volume = df['volume'].values
       
   # 基础价格特征
   df['returns'] = df['close'].pct_change()
   df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
   df['high_low_ratio'] = df['high'] / df['low'] - 1
   df['close_open_ratio'] = df['close'] / df['open'] - 1
       
   # 移动平均线（只使用历史数据，增加min_periods保证稳定性，明确center=False）
   for period in [5, 10, 20, 30, 60]:
       min_periods = max(period // 2, 3)  # 确保有足够的数据点
       df[f'sma_{period}'] = df['close'].rolling(window=period, min_periods=min_periods, center=False).mean()
       df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
       df[f'volume_sma_{period}'] = df['volume'].rolling(window=period, min_periods=min_periods, center=False).mean()
       
   # 价格位置特征
   for period in [10, 20, 30]:
       min_periods = max(period // 2, 5)
       rolling_high = df['high'].rolling(window=period, min_periods=min_periods, center=False).max()
       rolling_low = df['low'].rolling(window=period, min_periods=min_periods, center=False).min()
       df[f'price_position_{period}'] = (df['close'] - rolling_low) / (rolling_high - rolling_low + 1e-10)
       
   # 波动率特征（使用历史数据，增加min_periods，明确center=False）
   for period in [5, 10, 20]:
       min_periods = max(period // 2, 3)
       df[f'volatility_{period}'] = df['returns'].rolling(window=period, min_periods=min_periods, center=False).std()
       df[f'volume_volatility_{period}'] = df['volume'].rolling(window=period, min_periods=min_periods, center=False).std()
       
   # RSI指标（增加min_periods确保稳定性，明确center=False）
   for period in [6, 12, 24]:
       delta = df['close'].diff()
       gain = delta.where(delta > 0, 0)
       loss = -delta.where(delta < 0, 0)
       min_periods = max(period // 2, 3)
       avg_gain = gain.rolling(window=period, min_periods=min_periods, center=False).mean()
       avg_loss = loss.rolling(window=period, min_periods=min_periods, center=False).mean()
       rs = avg_gain / (avg_loss + 1e-10)
       df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
       
   # MACD指标
   ema_12 = df['close'].ewm(span=12, adjust=False).mean()
   ema_26 = df['close'].ewm(span=26, adjust=False).mean()
   df['macd'] = ema_12 - ema_26
   df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
   df['macd_diff'] = df['macd'] - df['macd_signal']
       
   # 布林带（增加min_periods，明确center=False）
   for period in [10, 20]:
       min_periods = max(period // 2, 5)
       sma = df['close'].rolling(window=period, min_periods=min_periods, center=False).mean()
       std = df['close'].rolling(window=period, min_periods=min_periods, center=False).std()
       df[f'bb_upper_{period}'] = sma + 2 * std
       df[f'bb_lower_{period}'] = sma - 2 * std
       df[f'bb_width_{period}'] = (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']) / (sma + 1e-10)
       df[f'bb_position_{period}'] = (df['close'] - df[f'bb_lower_{period}']) / (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'] + 1e-10)
       
   # ATR（平均真实范围，增加min_periods，明确center=False）
   for period in [7, 14, 21]:
       min_periods = max(period // 2, 3)
       high_low = df['high'] - df['low']
       high_close = np.abs(df['high'] - df['close'].shift(1))
       low_close = np.abs(df['low'] - df['close'].shift(1))
       true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
       df[f'atr_{period}'] = true_range.rolling(window=period, min_periods=min_periods, center=False).mean()
       
   # OBV（能量潮指标）
   obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
   df['obv'] = obv
   df['obv_sma'] = obv.rolling(window=20, min_periods=10, center=False).mean()
       
   # 成交量特征（明确center=False）
   df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20, min_periods=10, center=False).mean()
   df['volume_trend'] = df['volume'].rolling(window=5, min_periods=3, center=False).mean() / df['volume'].rolling(window=20, min_periods=10, center=False).mean()
       
   # 价格动量
   for period in [5, 10, 20]:
       df[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1
       df[f'roc_{period}'] = (df['close'] - df['close'].shift(period)) / df['close'].shift(period)
       
   # 市场微观结构特征
   df['spread'] = df['high'] - df['low']
   df['relative_spread'] = df['spread'] / df['close']
   df['mid_price'] = (df['high'] + df['low']) / 2
   df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
   df['weighted_close'] = (df['high'] + df['low'] + 2 * df['close']) / 4
       
   # 现在添加扩展因子库
   df = validate_and_calculate_extended_factors(df)
   print('-----------------------------df', len(df.columns))
       
   # 填充NaN值（使用向前填充，确保不使用未来数据）
   df = df.fillna(method='ffill', limit=5)
       
   # 对于仍然存在的NaN，使用0填充（通常是序列开始的几个值）
   df = df.fillna(0)
       
   logger.info(f"技术指标计算完成，生成特征数: {len(df.columns) - len(required_cols)}")
       
   return df
    
def clean_numeric_data_no_future(df: pd.DataFrame, columns: List[str] = None, 
                                lookback_window: int = 100) -> pd.DataFrame:
    """
    安全的数值数据清理函数，严格不使用未来数据
    逐行处理，确保每个时间点只使用历史信息
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
        
    df_clean = df.copy()
        
    for col in columns:
        if col not in df_clean.columns:
            continue
                
        col_values = df_clean[col].values.copy()
        n_rows = len(col_values)
            
        # 逐行处理，确保不使用未来数据
        for i in range(n_rows):
            # 替换无穷值
            if np.isinf(col_values[i]):
                col_values[i] = np.nan
                
            # 处理缺失值
            if pd.isna(col_values[i]):
                # 修正：确保历史窗口不包括当前点
                hist_start = max(0, i - lookback_window)
                hist_end = i  # 不包括当前点i
                    
                if hist_end > hist_start:
                    hist_data = col_values[hist_start:hist_end]
                    hist_data_clean = hist_data[~pd.isna(hist_data)]
                        
                    if len(hist_data_clean) > 0:
                        # 使用历史数据的中位数填充
                        col_values[i] = np.median(hist_data_clean)
                    elif i > 0 and not pd.isna(col_values[i-1]):
                        # 如果历史数据不足，使用前一个值
                        col_values[i] = col_values[i-1]
                    else:
                        # 最后的手段：使用0
                        col_values[i] = 0
                
            # 异常值处理（基于历史数据）
            if i >= lookback_window and pd.notna(col_values[i]):
                hist_data = col_values[max(0, i-lookback_window):i]
                hist_data_clean = hist_data[~pd.isna(hist_data)]
                    
                if len(hist_data_clean) >= lookback_window // 2:
                    # 使用历史数据的分位数
                    q01 = np.percentile(hist_data_clean, 1)
                    q99 = np.percentile(hist_data_clean, 99)
                        
                    # 如果当前值超出历史范围太多，进行裁剪
                    if col_values[i] < q01 or col_values[i] > q99:
                        q25 = np.percentile(hist_data_clean, 25)
                        q75 = np.percentile(hist_data_clean, 75)
                        iqr = q75 - q25
                            
                        lower_bound = q25 - 3 * iqr
                        upper_bound = q75 + 3 * iqr
                            
                        col_values[i] = np.clip(col_values[i], lower_bound, upper_bound)
            
        df_clean[col] = col_values
        
    return df_clean
# ================== 数据泄露检查器 ==================
class DataLeakageChecker:
    """增强版数据泄露和未来函数检查器"""
        
    def __init__(self):
        self.check_history = []
        
    def check_temporal_consistency(self, df: pd.DataFrame) -> bool:
        """
        检查时间序列一致性
            
        Args:
            df: 待检查的数据框
                
        Returns:
            True if consistent, False otherwise
        """
        if df.empty:
            return False
            
        # 检查索引是否为时间类型
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.warning("Index is not DatetimeIndex")
            return True  # 如果不是时间索引，暂时返回True
            
        # 检查时间序列是否单调递增
        if not df.index.is_monotonic_increasing:
            logger.warning("Time series is not monotonic increasing")
            return False
            
        # 检查是否有重复的时间戳
        if df.index.has_duplicates:
            logger.warning("Time series has duplicate timestamps")
            return False
            
        return True
        
    def validate_signal_timing(self, signal_date, current_date) -> bool:
        """
        验证信号时间的合法性
            
        Args:
            signal_date: 信号日期
            current_date: 当前日期
                
        Returns:
            True if valid, False otherwise
        """
        return signal_date < current_date
        
    def check_feature_target_leakage(self, features: pd.DataFrame, target: pd.Series, 
                                    threshold: float = 0.95) -> List[str]:
        """
        检查特征与目标之间是否存在数据泄露
            
        Args:
            features: 特征数据框
            target: 目标序列
            threshold: 相关性阈值
                
        Returns:
            可疑特征列表
        """
        suspicious_features = []
            
        for col in features.columns:
            if col in features.columns:
                try:
                    # 计算相关性
                    correlation = features[col].corr(target)
                        
                    # 如果相关性过高，标记为可疑
                    if abs(correlation) > threshold:
                        logger.warning(f"Feature {col} has high correlation with target: {correlation:.4f}")
                        suspicious_features.append(col)
                except Exception as e:
                    logger.debug(f"Could not calculate correlation for {col}: {e}")
                    continue
            
        return suspicious_features
        
    def check_prediction_leakage(self, predictions: np.ndarray, targets: np.ndarray, 
                                threshold: float = 0.95) -> bool:
        """
        检查预测是否存在数据泄露
            
        Args:
            predictions: 预测值
            targets: 真实值
            threshold: 相关性阈值
                
        Returns:
            True if leakage suspected, False otherwise
        """
        if len(predictions) != len(targets):
            return False
            
        # 计算相关性
        correlation = np.corrcoef(predictions, targets)[0, 1]
            
        # 如果相关性异常高，可能存在泄露
        if abs(correlation) > threshold:
            logger.warning(f"Predictions show abnormally high correlation with targets: {correlation:.4f}")
            return True
            
        # 检查是否完全相同（严重的数据泄露）
        if np.allclose(predictions, targets):
            logger.error("Predictions are identical to targets - severe data leakage!")
            return True
            
        return False
def validate_and_calculate_extended_factors(df: pd.DataFrame, 
                                           period_id_list: List[int] = None) -> pd.DataFrame:
    """
    计算并验证扩展因子库中的所有因子
    增加验证机制确保不使用未来数据
    """
    if not FACTORS_AVAILABLE:
        logger.warning("Factors module not available, skipping extended factors")
        return df
        
    logger.info(f"Calculating and validating extended factors from factors.py")
        
    # 准备数据格式
    stock_data = df.copy()
    original_columns = stock_data.columns.tolist()
        
    # 确保必要的列存在
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in required_cols:
        if col not in stock_data.columns:
            logger.error(f"Missing required column for factors: {col}")
            return df
        
    # 添加vwap和amount列
    if 'vwap' not in stock_data.columns:
        stock_data['vwap'] = (stock_data['high'] + stock_data['low'] + stock_data['close']) / 3
    if 'amount' not in stock_data.columns:
        stock_data['amount'] = stock_data['volume'] * stock_data['vwap']
        
    # 默认周期参数
    if period_id_list is None:
        period_id_list = [3, 4, 5, 6, 7]
        
    all_factor_names = []
    suspicious_factors = []  # 记录可疑的因子
        
    # 遍历所有可用的因子函数
    for func_name in factors.func_list:
        try:
            if hasattr(factors, func_name):
                func = getattr(factors, func_name)
                    
                # 保存调用前的数据副本用于验证
                pre_call_data = stock_data.copy()
                    
                # 调用因子函数
                stock_data, factor_names = func(stock_data, period_id_list)
                if isinstance(factor_names, str):
                    factor_names = [factor_names]
                    
                # 验证新生成的因子
                for factor_name in factor_names:
                    if factor_name in stock_data.columns and factor_name not in original_columns:
                        # 检查因子是否使用了未来数据
                        if _check_factor_future_leakage(stock_data[factor_name], pre_call_data):
                            logger.warning(f"Factor {factor_name} may contain future information, removing...")
                            suspicious_factors.append(factor_name)
                            stock_data = stock_data.drop(columns=[factor_name])
                        else:
                            all_factor_names.append(factor_name)
                       
        except Exception as e:
            logger.debug(f"Error calculating factor {func_name}: {e}")
            continue
        
    # 清理因子数据 - 严格只使用向前填充
    for col in all_factor_names:
        if col in stock_data.columns:
            # 替换无穷值
            stock_data[col] = stock_data[col].replace([np.inf, -np.inf], np.nan)
                
            # 使用向前填充（确保不使用未来数据）
            stock_data[col] = stock_data[col].fillna(method='ffill', limit=5)
                
            # 对于仍有缺失值的，使用历史均值填充
            for i in range(len(stock_data)):
                if pd.isna(stock_data[col].iloc[i]) and i >= 20:
                    # 只使用i之前的数据计算均值
                    hist_mean = stock_data[col].iloc[max(0, i-50):i].mean()
                    if pd.notna(hist_mean):
                        stock_data.iloc[i, stock_data.columns.get_loc(col)] = hist_mean
                
            # 最后使用0填充
            stock_data[col] = stock_data[col].fillna(0)
        
    logger.info(f"Calculated {len(all_factor_names)} valid extended factors")
    if suspicious_factors:
        logger.warning(f"Removed {len(suspicious_factors)} suspicious factors: {suspicious_factors[:5]}...")
        
    return stock_data
    
def _check_factor_future_leakage(factor_series: pd.Series, original_data: pd.DataFrame, 
                                threshold: float = 0.7) -> bool:
    """
    检查因子是否可能包含未来信息
        
    Args:
        factor_series: 待检查的因子序列
        original_data: 原始数据
        threshold: 相关性阈值
        
    Returns:
        True if suspicious, False otherwise
    """
    # 检查与未来价格的相关性
    if 'close' in original_data.columns:
        future_prices = original_data['close'].shift(-1)  # 未来一期价格
            
        # 计算相关性（忽略NaN）
        valid_mask = ~(factor_series.isna() | future_prices.isna())
        if valid_mask.sum() > 100:
            correlation = factor_series[valid_mask].corr(future_prices[valid_mask])
                
            # 如果与未来价格相关性过高，则可能存在未来函数
            if abs(correlation) > threshold:
                return True
        
    # 检查因子值是否"太完美"（可能使用了未来信息）
    if factor_series.notna().sum() > 100:
        # 计算因子的自相关性
        autocorr = factor_series.autocorr(lag=1)
            
        # 如果自相关性接近1，可能是直接使用了未来值
        if pd.notna(autocorr) and autocorr > 0.999:
            return True
        
    return False
def create_trading_target_variable_no_future(df: pd.DataFrame, horizon: int = 5, 
                                            is_training: bool = True) -> pd.Series:
    """
    完全重新设计的目标变量：
    训练模式：使用未来收益作为标签
    预测模式：不使用任何未来信息，返回零值
    
    Args:
        df: 包含价格数据的DataFrame
        horizon: 预测周期
        is_training: 是否为训练模式
    
    Returns:
        目标变量序列
    """
    close_prices = df['close'].copy()
    target = pd.Series(index=df.index, dtype=float)
    target[:] = 0
    
    # 预测模式：不使用任何未来信息
    if not is_training:
        logger.info("Target creation in prediction mode - returning zeros")
        return target
    
    # 训练模式：可以使用未来收益作为训练标签
    logger.info("Target creation in training mode - using future returns for labels")
    
    # 计算多个时间尺度的未来收益（仅在训练时使用）
    returns_1 = (close_prices.shift(-horizon) / close_prices - 1)
    returns_2 = (close_prices.shift(-horizon*2) / close_prices - 1) 
    returns_avg = (close_prices.shift(-horizon).rolling(horizon, min_periods=1).mean() / close_prices - 1)
    
    # 计算历史波动率（逐点计算，避免未来函数）
    for i in range(len(df)):
        if i < 50:  # 需要足够的历史数据
            target.iloc[i] = 0
            continue
        
        # 只使用i之前的数据计算波动率
        hist_returns = close_prices.iloc[max(0, i-50):i].pct_change().dropna()
        
        if len(hist_returns) < 20:
            target.iloc[i] = 0
            continue
        
        # 历史波动率
        volatility = hist_returns.std()
        
        # 动态阈值：考虑交易成本（0.03%）和滑点
        cost_threshold = 0.0006  # 双边成本
        signal_threshold = max(volatility * 0.8, cost_threshold * 2)
        
        # 获取未来收益（如果存在）- 仅在训练模式使用
        if i < len(df) - horizon*2:
            r1 = returns_1.iloc[i] if pd.notna(returns_1.iloc[i]) else 0
            r2 = returns_2.iloc[i] if pd.notna(returns_2.iloc[i]) else 0
            r_avg = returns_avg.iloc[i] if pd.notna(returns_avg.iloc[i]) else 0
            
            # 综合考虑短期和中期收益
            combined_return = r1 * 0.5 + r_avg * 0.3 + r2 * 0.2
            
            # 生成标签：考虑风险调整
            if combined_return > signal_threshold:
                # 强上涨信号
                if combined_return > signal_threshold * 2:
                    target.iloc[i] = 1.0
                else:
                    target.iloc[i] = 0.5
            elif combined_return < -signal_threshold:
                # 强下跌信号
                if combined_return < -signal_threshold * 2:
                    target.iloc[i] = -1.0
                else:
                    target.iloc[i] = -0.5
            else:
                # 中性，但保留小信号
                target.iloc[i] = combined_return / signal_threshold * 0.3
    
    # 清理最后的horizon*2个数据点
    target.iloc[-horizon*2:] = 0
    
    return target
# ================== 改进的仓位优化器 ==================
class ImprovedPositionOptimizer:
    """
    基于分段线性函数的仓位优化器
    实现 max(signal*w - cost*abs(w-w_pre) - w**2) 的解析解
    """
     
    def __init__(self, commission_rate: float = 0.0002, min_trade_threshold: float = 0.001):
        """
        Args:
            commission_rate: 交易成本率（双边）
            min_trade_threshold: 最小交易阈值，避免频繁小额交易
        """
        self.commission_rate = commission_rate
        self.min_trade_threshold = min_trade_threshold
        self.position_history = {}  # 记录各品种的历史仓位
         
    def calculate_optimal_position(self, signal: float, w_pre: float = 0, 
                                  volatility: float = None) -> float:
        """
        根据信号计算最优仓位（分段线性函数）
         
        Args:
            signal: 预测信号（已经考虑了波动率调整）
            w_pre: 上一期真实仓位
            volatility: 波动率（用于信号调整，如果信号已调整则不需要）
             
        Returns:
            最优目标仓位
        """
        # 交易成本（这里使用简化的固定成本）
        cost = self.commission_rate * 2  # 考虑买入和卖出的双边成本
         
        # 根据解析解计算最优仓位
        if signal < w_pre - cost:
            # 信号显著小于当前仓位，减仓
            w_optimal = signal + cost
        elif signal > w_pre + cost:
            # 信号显著大于当前仓位，加仓
            w_optimal = signal - cost
        else:
            # 信号在成本区间内，保持不变
            w_optimal = w_pre
             
        # 仓位限制
        w_optimal = np.clip(w_optimal, -1.0, 1.0)
         
        # 最小交易阈值检查
        if abs(w_optimal - w_pre) < self.min_trade_threshold:
            return w_pre
             
        return w_optimal
     
    def update_position_history(self, symbol: str, position: float):
        """更新品种的仓位历史"""
        self.position_history[symbol] = position
         
    def get_previous_position(self, symbol: str) -> float:
        """获取品种的上一期仓位"""
        return self.position_history.get(symbol, 0.0)
 
# ================== 改进的风险管理器（整合仓位优化） ==================
# ================== 改进的风险管理器（整合仓位优化） ==================
class NoFutureRiskManager:
    """整合了改进仓位优化的风险管理器"""
     
    def __init__(self, max_position: float = 0.2, max_leverage: float = 1.5):
        self.max_position = max_position
        self.max_leverage = max_leverage
        self.position_optimizer = ImprovedPositionOptimizer()
         
        # 风险参数
        self.stop_loss_multiplier = 2.5
        self.take_profit_multiplier = 3.0
        self.max_daily_trades = 15
        self.min_holding_period = 3
        self.intelligent_position_manager = IntelligentPositionManager()  # 添加智能仓位管理器

    def calculate_position_size(self, signal: float, volatility: float, 
                               portfolio_value: float, recent_returns: list,
                               current_positions: dict) -> float:
        """
        计算仓位大小（兼容旧接口）
         
        Args:
            signal: 交易信号
            volatility: 波动率
            portfolio_value: 组合价值
            recent_returns: 最近收益
            current_positions: 当前持仓
             
        Returns:
            目标仓位
        """
        # 这里我们调用优化版本的方法
        # 为了兼容，我们需要从current_positions获取symbol
        # 但这里没有symbol参数，所以使用默认symbol
        symbol = "default"
         
        return self.calculate_position_size_optimized(
            signal=signal,
            symbol=symbol,
            volatility=volatility,
            portfolio_value=portfolio_value
        )
     
    def calculate_position_size_optimized(self, signal: float, symbol: str, 
                                          volatility: float = None,
                                          portfolio_value: float = None) -> float:
        # 使用智能仓位管理器计算仓位
        if hasattr(self, 'market_data') and symbol in self.market_data:
            df = self.market_data[symbol]
            current_idx = len(df) - 1
            
            # 使用智能管理器计算最优仓位
            optimal_position = self.intelligent_position_manager.calculate_optimal_position(
                signal, symbol, df, current_idx
            )
        else:
            # 回退到原有逻辑
            optimal_position = self.position_optimizer.calculate_optimal_position(
                signal, self.position_optimizer.get_previous_position(symbol)
            )
            
        return optimal_position
     
    def should_exit_position(self, position, current_price: float, 
                           holding_periods: float, exit_signal: float, 
                           volatility: float) -> tuple:
        """
        判断是否应该退出仓位
         
        Args:
            position: 当前持仓对象
            current_price: 当前价格
            holding_periods: 持有周期数
            exit_signal: 退出信号
            volatility: 波动率
             
        Returns:
            (是否退出, 退出原因)
        """
        should_exit = False
        exit_reason = ""
         
        # 止损检查
        if position.quantity > 0:
            if current_price <= position.stop_loss:
                should_exit = True
                exit_reason = "stop_loss"
        else:
            if current_price >= position.stop_loss:
                should_exit = True
                exit_reason = "stop_loss"
         
        # 止盈检查
        if not should_exit:
            if position.quantity > 0:
                if current_price >= position.take_profit:
                    should_exit = True
                    exit_reason = "take_profit"
            else:
                if current_price <= position.take_profit:
                    should_exit = True
                    exit_reason = "take_profit"
         
        # 信号反转检查
        if not should_exit and abs(exit_signal) > 0.01:
            if position.quantity > 0 and exit_signal < -0.1:
                should_exit = True
                exit_reason = "signal_reverse"
            elif position.quantity < 0 and exit_signal > 0.1:
                should_exit = True
                exit_reason = "signal_reverse"
         
        # 最小持有期检查（如果其他条件触发退出，这个不阻止）
        if should_exit and holding_periods < self.min_holding_period:
            # 只有在非止损情况下才考虑最小持有期
            if exit_reason != "stop_loss":
                should_exit = False
                exit_reason = ""
         
        # 最大持有期检查
        if not should_exit and holding_periods > 100:  # 100个30分钟周期
            should_exit = True
            exit_reason = "max_holding_period"
         
        return should_exit, exit_reason
     
    def calculate_stop_loss(self, entry_price: float, volatility: float, 
                          position_size: float, signal_strength: float = 1.0) -> float:
        """计算止损价格"""
        base_stop = max(volatility * self.stop_loss_multiplier, 0.015)
         
        # 根据信号强度调整止损距离
        adjusted_stop = base_stop / max(signal_strength, 0.5)
         
        if position_size > 0:
            stop_loss = entry_price * (1 - adjusted_stop)
        else:
            stop_loss = entry_price * (1 + adjusted_stop)
             
        return stop_loss
     
    def calculate_take_profit(self, entry_price: float, volatility: float, 
                            position_size: float, signal_strength: float = 1.0) -> float:
        """计算止盈价格"""
        base_profit = max(volatility * self.take_profit_multiplier, 0.02)
         
        # 根据信号强度调整止盈距离
        adjusted_profit = base_profit * max(signal_strength, 0.5)
         
        if position_size > 0:
            take_profit = entry_price * (1 + adjusted_profit)
        else:
            take_profit = entry_price * (1 - adjusted_profit)
             
        return take_profit

class NoFutureSignalGenerator:
    """改进的信号生成器，严格避免未来函数"""
    
    def __init__(self, model_trainer):
        self.model_trainer = model_trainer
        self.signal_delay = 2  # 增加信号延迟到5个周期，更保守
        self.base_threshold = 0.0001
        self.signal_scale_factor = 0.1
        self.leakage_checker = DataLeakageChecker()
        
    def generate_signals_from_features(self, features_df, feature_cols, df, symbol, 
                                      is_realtime: bool = False):
        """
        生成交易信号 - 改进版本
        
        Args:
            features_df: 特征DataFrame
            feature_cols: 特征列名列表
            df: 原始数据DataFrame
            symbol: 品种代码
            is_realtime: 是否为实时预测模式
        
        Returns:
            信号序列
        """
        try:
            if len(features_df) < 100:
                logger.warning(f"{symbol}: 数据不足，跳过信号生成")
                return pd.Series(dtype=float)
            
            logger.info(f"{symbol}: 开始生成信号 (实时模式: {is_realtime})...")
            
            # 准备特征
            X = self._prepare_features(features_df, feature_cols)
            if X is None:
                return pd.Series(dtype=float)
            
            # 如果是实时模式，排除最新的几个数据点
            if is_realtime:
                # 实时模式下，不使用最新的signal_delay个数据点
                X = X[:-self.signal_delay] if len(X) > self.signal_delay else X
            
            # 获取预测（预期收益）
            predictions = self.model_trainer.predict_ensemble(X)
            
            # 计算历史波动率（逐点计算，不使用未来数据）
            volatility_series = pd.Series(index=features_df.index, dtype=float)
            for i in range(len(features_df)):
                if i >= 50 and 'returns' in features_df.columns:
                    # 只使用历史数据计算波动率
                    hist_returns = features_df['returns'].iloc[max(0, i-50):i]
                    volatility_series.iloc[i] = hist_returns.std()
                else:
                    volatility_series.iloc[i] = 0.02  # 默认波动率
            
            # 生成考虑了波动率的信号
            signals = pd.Series(index=features_df.index, dtype=float)
            
            for i in range(len(predictions)):
                if i < 20:
                    signals.iloc[i] = 0
                    continue
                
                pred = predictions[i] if i < len(predictions) else 0
                vol = volatility_series.iloc[i]
                
                # 计算风险调整后的信号（预期收益/波动率）
                if vol > 0:
                    # Sharpe ratio style signal
                    risk_adjusted_signal = pred / vol
                else:
                    risk_adjusted_signal = pred / 0.02
                
                # 缩放信号到合理范围（-1到1之间）
                signals.iloc[i] = np.tanh(risk_adjusted_signal * self.signal_scale_factor)
            
            # 应用信号延迟（避免未来函数）
            signals = signals.shift(self.signal_delay).fillna(0)
            
            # 清零最新的信号（额外的安全措施）
            if is_realtime:
                signals.iloc[-self.signal_delay:] = 0
            
            # 平滑信号（只使用历史数据）
            if len(signals) > 3:
                signals = signals.rolling(window=3, min_periods=1, center=False).mean()
            
            # 最终安全检查：确保不会在最新数据点产生信号
            if is_realtime:
                signals.iloc[-self.signal_delay:] = 0
            
            logger.info(f"{symbol}: 生成 {(signals != 0).sum()} 个非零信号")
            
            return signals
            
        except Exception as e:
            logger.error(f"生成信号时出错 {symbol}: {e}")
            return pd.Series(dtype=float)
    
    def _prepare_features(self, features_df, feature_cols):
        """准备特征（与原版相同）"""
        if self.model_trainer.selected_features:
            available_features = [f for f in self.model_trainer.selected_features 
                                if f in features_df.columns]
            X = features_df[available_features].copy()
        else:
            X = features_df[feature_cols].copy()
        
        # 清理数据
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(method='ffill', limit=5)
        X = X.fillna(0)
        
        # 缩放特征
        if self.model_trainer.scaler is not None and self.model_trainer.scaler_fitted:
            try:
                X_scaled = self.model_trainer.scaler.transform(X)
                return np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
            except:
                return X.values
        
        return X.values
# ================== 完全无未来函数的回测引擎 ==================
class NoFutureBacktestEngine:
   """完全无未来函数的回测引擎"""
       
   def __init__(self, initial_capital: float = 1000000, commission_rate: float = 0.0002):
       self.initial_capital = initial_capital
       self.commission_rate = commission_rate
       self.positions = {}
       self.trades = []
       self.portfolio_values = []
       self.daily_returns = []
       self.daily_trades = defaultdict(int)
       self.position_entry_times = {}
       self.cumulative_pnl = 0
       self.periods_per_day = 48
       self.periods_per_year = 252 * self.periods_per_day
       self.leakage_checker = DataLeakageChecker()
       
   def _create_empty_results(self) -> Dict:
       """创建空的回测结果"""
       return {
           'total_return': 0.0,
           'annual_return': 0.0,
           'volatility': 0.0,
           'sharpe_ratio': 0.0,
           'sortino_ratio': 0.0,
           'max_drawdown': 0.0,
           'win_rate': 0.0,
           'profit_factor': 0.0,
           'total_trades': 0,
           'calmar_ratio': 0.0,
           'total_commission': 0.0,
           'total_slippage': 0.0,
           'portfolio_values': pd.DataFrame(),
           'trades': [],
           'daily_returns': [],
           'exit_reasons': {},
           'avg_positions': 0.0,
           'trading_days': 0,
           'years': 0.0
       }
       
   def _calculate_backtest_results(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> Dict:
       """计算回测结果"""
       logger.info(f"开始计算回测结果...")
       logger.info(f"投资组合记录数: {len(self.portfolio_values)}")
       logger.info(f"交易记录数: {len(self.trades)}")
           
       if not self.portfolio_values or not self.daily_returns:
           logger.warning("没有投资组合数据或日收益数据，返回空结果")
           return self._create_empty_results()
           
       portfolio_df = pd.DataFrame(self.portfolio_values)
       portfolio_df.set_index('date', inplace=True)
           
       total_return = (portfolio_df['portfolio_value'].iloc[-1] /
                      portfolio_df['portfolio_value'].iloc[0] - 1)
           
       trading_days = (end_date - start_date).days
       if trading_days <= 0:
           trading_days = 1
           
       years = trading_days / 252
       if years > 0:
           annual_return = (1 + total_return) ** (1 / years) - 1
       else:
           annual_return = 0
           
       daily_returns = np.array(self.daily_returns)
           
       volatility = np.std(daily_returns) * np.sqrt(self.periods_per_year) if len(daily_returns) > 0 else 0
           
       sharpe_ratio = annual_return / volatility if volatility > 0 else 0
           
       if len(daily_returns) > 0:
           cumulative = (1 + daily_returns).cumprod()
           running_max = np.maximum.accumulate(cumulative)
           drawdown = (cumulative - running_max) / running_max
           max_drawdown = np.min(drawdown)
       else:
           max_drawdown = 0
           
       if self.trades:
           winning_trades = [t for t in self.trades if t.net_pnl > 0]
           win_rate = len(winning_trades) / len(self.trades)
           avg_win = np.mean([t.net_pnl for t in winning_trades]) if winning_trades else 0
           losing_trades = [t for t in self.trades if t.net_pnl <= 0]
           avg_loss = np.mean([t.net_pnl for t in losing_trades]) if losing_trades else 0
           profit_factor = abs(avg_win * len(winning_trades) / (avg_loss * len(losing_trades))) if avg_loss != 0 and len(losing_trades) > 0 else 0
               
           total_commission = sum([t.commission for t in self.trades])
           total_slippage = sum([t.slippage for t in self.trades])
               
           exit_reasons = defaultdict(int)
           for trade in self.trades:
               exit_reasons[trade.exit_reason] += 1
       else:
           win_rate = 0
           profit_factor = 0
           total_commission = 0
           total_slippage = 0
           exit_reasons = {}
           
       calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
           
       if len(daily_returns) > 0:
           downside_returns = [r for r in daily_returns if r < 0]
           if downside_returns:
               downside_std = np.std(downside_returns) * np.sqrt(self.periods_per_year)
               sortino_ratio = annual_return / downside_std if downside_std > 0 else 0
           else:
               sortino_ratio = 0
       else:
           sortino_ratio = 0
           
       results = {
           'total_return': total_return,
           'annual_return': annual_return,
           'volatility': volatility,
           'sharpe_ratio': sharpe_ratio,
           'sortino_ratio': sortino_ratio,
           'max_drawdown': max_drawdown,
           'win_rate': win_rate,
           'profit_factor': profit_factor,
           'total_trades': len(self.trades),
           'calmar_ratio': calmar_ratio,
           'total_commission': total_commission,
           'total_slippage': total_slippage,
           'portfolio_values': portfolio_df,
           'trades': self.trades,
           'daily_returns': daily_returns,
           'exit_reasons': dict(exit_reasons),
           'avg_positions': np.mean([pv['num_positions'] for pv in self.portfolio_values]),
           'trading_days': trading_days,
           'years': years
       }
           
       return results
       
   def _execute_trade(self, symbol: str, target_position: float, 
                     price: float, date: pd.Timestamp, cash: float,
                     risk_manager: NoFutureRiskManager, signal: float,
                     volatility: float) -> Optional[float]:
       """执行交易"""
       current_position = self.positions.get(symbol)
           
       if current_position is None:
           current_quantity = 0
       else:
           current_quantity = current_position.quantity
           
       target_value = target_position * self.initial_capital
       target_quantity = target_value / price
           
       quantity_diff = target_quantity - current_quantity
           
       if abs(quantity_diff * price) < 500:
           return None
           
       is_buy = quantity_diff > 0
           
       slippage_rate = 0.0001
       if is_buy:
           execution_price = price * (1 + slippage_rate)
       else:
           execution_price = price * (1 - slippage_rate)
           
       trade_amount = abs(quantity_diff * execution_price)
       commission = trade_amount * self.commission_rate
           
       if is_buy:
           required_cash = trade_amount + commission
           if required_cash > cash * 0.95:
               max_quantity = (cash * 0.95 - commission) / execution_price
               if max_quantity <= 0:
                   return None
               quantity_diff = min(quantity_diff, max_quantity)
               trade_amount = abs(quantity_diff * execution_price)
               commission = trade_amount * self.commission_rate
           
       if abs(quantity_diff) > 0.001:
           if is_buy:
               new_cash = cash - (trade_amount + commission)
           else:
               new_cash = cash + (trade_amount - commission)
               
           slippage_cost = abs(execution_price - price) * abs(quantity_diff)
               
           if symbol not in self.positions:
               if abs(target_quantity) > 0.001:
                   stop_loss = risk_manager.calculate_stop_loss(
                       execution_price, volatility, target_position, abs(signal)
                   )
                   take_profit = risk_manager.calculate_take_profit(
                       execution_price, volatility, target_position, abs(signal)
                   )
                       
                   self.positions[symbol] = Position(
                       symbol=symbol, 
                       quantity=target_quantity, 
                       entry_price=execution_price,
                       entry_time=date, 
                       current_price=execution_price, 
                       unrealized_pnl=0,
                       stop_loss=stop_loss,
                       take_profit=take_profit,
                       trailing_stop=stop_loss,
                       highest_price=execution_price,
                       lowest_price=execution_price
                   )
           else:
               old_position = self.positions[symbol]
               new_quantity = old_position.quantity + quantity_diff
                   
               if abs(new_quantity) < 0.001:
                   pnl = (execution_price - old_position.entry_price) * old_position.quantity
                   trade = Trade(
                       symbol=symbol, 
                       entry_time=old_position.entry_time, 
                       exit_time=date,
                       entry_price=old_position.entry_price, 
                       exit_price=execution_price,
                       quantity=old_position.quantity, 
                       pnl=pnl, 
                       commission=commission,
                       net_pnl=pnl - commission - slippage_cost,
                       slippage=slippage_cost,
                       exit_reason="signal_close"
                   )
                   self.trades.append(trade)
                   del self.positions[symbol]
               else:
                   if quantity_diff * old_position.quantity < 0:
                       partial_pnl = (execution_price - old_position.entry_price) * (-quantity_diff)
                       partial_commission = abs(quantity_diff * execution_price) * self.commission_rate
                       partial_slippage = slippage_cost * abs(quantity_diff) / abs(old_position.quantity)
                           
                       trade = Trade(
                           symbol=symbol,
                           entry_time=old_position.entry_time,
                           exit_time=date,
                           entry_price=old_position.entry_price,
                           exit_price=execution_price,
                           quantity=-quantity_diff,
                           pnl=partial_pnl,
                           commission=partial_commission,
                           net_pnl=partial_pnl - partial_commission - partial_slippage,
                           slippage=partial_slippage,
                           exit_reason="partial_close"
                       )
                       self.trades.append(trade)
                           
                       self.positions[symbol].quantity = new_quantity
                   else:
                       total_cost = old_position.entry_price * abs(old_position.quantity) + execution_price * abs(quantity_diff)
                       self.positions[symbol].entry_price = total_cost / abs(new_quantity)
                       self.positions[symbol].quantity = new_quantity
                           
                       self.positions[symbol].stop_loss = risk_manager.calculate_stop_loss(
                           self.positions[symbol].entry_price, volatility, target_position, abs(signal)
                       )
                       self.positions[symbol].take_profit = risk_manager.calculate_take_profit(
                           self.positions[symbol].entry_price, volatility, target_position, abs(signal)
                       )
                       self.positions[symbol].trailing_stop = self.positions[symbol].stop_loss
               
           return new_cash
           
       return None
       
   @monitor_memory
   def run_backtest(self, signals_dict: Dict[str, pd.Series], 
                   price_data_dict: Dict[str, pd.DataFrame],
                   start_date: str = None, end_date: str = None) -> Dict:
       """运行回测 - 完全无未来函数版本"""
       logger.info("开始运行回测（完全无未来函数版本）...")
           
       # 验证数据时间一致性
       for symbol, data in price_data_dict.items():
           if not self.leakage_checker.check_temporal_consistency(data):
               logger.error(f"{symbol} 数据时间序列不一致")
               return self._create_empty_results()
           
       total_signals = 0
       for symbol, signals in signals_dict.items():
           non_zero = (signals != 0).sum()
           total_signals += non_zero
           logger.info(f"{symbol}: {non_zero} 个非零信号")
           
       logger.info(f"总共有 {total_signals} 个非零信号")
           
       if total_signals == 0:
           logger.warning("没有任何交易信号，回测无法进行")
           return self._create_empty_results()
           
       all_dates = set()
       for symbol, signals in signals_dict.items():
           if symbol in price_data_dict:
               all_dates.update(signals.index)
           
       all_dates = sorted(list(all_dates))
           
       if start_date:
           all_dates = [d for d in all_dates if d >= pd.Timestamp(start_date)]
       if end_date:
           all_dates = [d for d in all_dates if d <= pd.Timestamp(end_date)]
           
       if len(all_dates) < 20:
           logger.error("回测数据不足")
           return self._create_empty_results()
           
       portfolio_value = self.initial_capital
       cash = self.initial_capital
           
       risk_manager = NoFutureRiskManager()
       recent_returns = []
           
       actual_start_date = all_dates[0]
       actual_end_date = all_dates[-1]
           
       trade_attempts = 0
       successful_trades = 0
           
       for i, date in enumerate(all_dates):
           # 跳过初始几个周期，确保有足够的历史数据
           if i < 10:
               continue
               
           daily_pnl = 0
           current_date_str = date.strftime('%Y-%m-%d')
               
           # 使用前一个时间点的信号，避免未来函数
           # 在 run_backtest 方法中添加严格的时间验证
           signal_date = all_dates[i-1] if i >= 1 else all_dates[0]
                
           # 添加严格的时间验证
           if not self.leakage_checker.validate_signal_timing(signal_date, date):
                logger.error(f"检测到未来函数：信号日期 {signal_date} 不早于当前日期 {date}")
                continue
                
           # 额外的安全检查
           if signal_date >= date:
                logger.error(f"未来函数警告：信号日期{signal_date}不早于交易日期{date}")
                continue
               
           # 更新持仓价值并检查退出条件
           positions_to_close = []
           for symbol, position in self.positions.items():
               if symbol in price_data_dict:
                   price_df = price_data_dict[symbol]
                   if date in price_df.index:
                       current_price = price_df.loc[date, 'open']
                       position.current_price = current_price
                       position.unrealized_pnl = (current_price - position.entry_price) * position.quantity
                       daily_pnl += position.unrealized_pnl
                           
                       if position.quantity > 0:
                           if current_price > position.highest_price:
                               position.highest_price = current_price
                       else:
                           if current_price < position.lowest_price:
                               position.lowest_price = current_price
                           
                       holding_periods = (date - position.entry_time).total_seconds() / 1800
                           
                       exit_signal = 0
                       if signal_date in signals_dict[symbol].index:
                           exit_signal = signals_dict[symbol].loc[signal_date]
                           
                       # 计算历史波动率（只使用过去的数据）
                       if i >= 20:
                           hist_dates = all_dates[max(0, i-20):i]  # 不包括当前点
                           hist_prices = []
                           for hist_date in hist_dates:
                               if hist_date in price_df.index:
                                   hist_prices.append(price_df.loc[hist_date, 'close'])
                               
                           if len(hist_prices) > 10:
                               returns = pd.Series(hist_prices).pct_change().dropna()
                               current_vol = returns.std()
                           else:
                               current_vol = 0.02
                       else:
                           current_vol = 0.02
                           
                       should_exit, exit_reason = risk_manager.should_exit_position(
                           position, current_price, holding_periods, exit_signal, current_vol
                       )
                           
                       if should_exit:
                           positions_to_close.append((symbol, exit_reason))
               
           # 执行平仓
           for symbol, exit_reason in positions_to_close:
               if symbol in self.positions:
                   position = self.positions[symbol]
                   price_df = price_data_dict[symbol]
                   if date in price_df.index:
                       exit_price = price_df.loc[date, 'open']
                           
                       slippage_rate = 0.0001
                       if position.quantity > 0:
                           execution_price = exit_price * (1 - slippage_rate)
                       else:
                           execution_price = exit_price * (1 + slippage_rate)
                           
                       pnl = (execution_price - position.entry_price) * position.quantity
                       commission = abs(position.quantity * execution_price) * self.commission_rate
                       slippage = abs(execution_price - exit_price) * abs(position.quantity)
                           
                       trade = Trade(
                           symbol=symbol,
                           entry_time=position.entry_time,
                           exit_time=date,
                           entry_price=position.entry_price,
                           exit_price=execution_price,
                           quantity=position.quantity,
                           pnl=pnl,
                           commission=commission,
                           net_pnl=pnl - commission - slippage,
                           slippage=slippage,
                           exit_reason=exit_reason
                       )
                       self.trades.append(trade)
                       recent_returns.append(trade.net_pnl / self.initial_capital)
                           
                       cash += position.quantity * execution_price - commission
                           
                       del self.positions[symbol]
                           
                       if symbol in self.position_entry_times:
                           del self.position_entry_times[symbol]
               
           # 检查新的交易信号
           for symbol, signals in signals_dict.items():
               if signal_date in signals.index and symbol in price_data_dict:
                   signal = signals.loc[signal_date]
                       
                   if abs(signal) < 0.0001:
                       continue
                       
                   trade_attempts += 1
                       
                   price_df = price_data_dict[symbol]
                       
                   if date in price_df.index:
                       entry_price = price_df.loc[date, 'open']
                           
                       if self.daily_trades[current_date_str] >= risk_manager.max_daily_trades:
                           continue
                           
                       if symbol in self.position_entry_times:
                           holding_periods = (date - self.position_entry_times[symbol]).total_seconds() / 1800
                           if holding_periods < risk_manager.min_holding_period:
                               continue
                           
                       # 计算历史波动率（只使用过去的数据）
                       if i >= 20:
                           hist_dates = all_dates[max(0, i-20):i]  # 不包括当前点
                           hist_prices = []
                           for hist_date in hist_dates:
                               if hist_date in price_df.index:
                                   hist_prices.append(price_df.loc[hist_date, 'close'])
                               
                           if len(hist_prices) > 10:
                               returns = pd.Series(hist_prices).pct_change().dropna()
                               volatility = returns.std()
                           else:
                               volatility = 0.02
                       else:
                           volatility = 0.02
                           
                       if pd.isna(volatility) or volatility <= 0:
                           volatility = 0.02
                           
                       if len(recent_returns) > 20:
                           recent_returns = recent_returns[-20:]
                           
                       target_position = risk_manager.calculate_position_size(
                           signal, volatility, portfolio_value, recent_returns, self.positions
                       )
                           
                       trade_result = self._execute_trade(
                           symbol, target_position, entry_price, date, cash, 
                           risk_manager, signal, volatility
                       )
                           
                       if trade_result is not None:
                           cash = trade_result
                           self.daily_trades[current_date_str] += 1
                           self.position_entry_times[symbol] = date
                           successful_trades += 1
               
           # 更新投资组合价值
           total_position_value = sum([pos.current_price * pos.quantity 
                                     for pos in self.positions.values()])
           portfolio_value = cash + total_position_value
               
           self.portfolio_values.append({
               'date': date,
               'portfolio_value': portfolio_value,
               'cash': cash,
               'position_value': total_position_value,
               'num_positions': len(self.positions)
           })
               
           if len(self.portfolio_values) > 1:
               prev_value = self.portfolio_values[-2]['portfolio_value']
               daily_return = (portfolio_value - prev_value) / prev_value
               self.daily_returns.append(daily_return)
           
       logger.info(f"交易尝试次数: {trade_attempts}")
       logger.info(f"成功交易次数: {successful_trades}")
       logger.info(f"交易成功率: {successful_trades/max(trade_attempts, 1)*100:.1f}%")
           
       # 在方法末尾
       results = self._calculate_backtest_results(actual_start_date, actual_end_date)
           
       logger.info("="*60)
       logger.info(f"回测完成（完全无未来函数版本）!")
           
       return results
 
# ================== 完全无未来函数的模型训练器 ==================
# ================== 完全无未来函数的模型训练器 ==================
class NoFutureModelTrainer:
    """完全无未来函数的模型训练器"""
        
    def __init__(self, use_gpu: bool = True, use_optuna: bool = False, optuna_trials: int = 30):
        self.use_gpu = use_gpu
        self.use_optuna = use_optuna
        self.optuna_trials = optuna_trials
        self.models = {}
        self.scaler = None
        self.feature_selector = None
        self.selected_features = []
        self.scaler_fitted = False
        self.feature_importance = {}
        self.leakage_checker = DataLeakageChecker()
        self.optimizer = None
        self.optimized_params = {}
            
        # 模型参数
        self.xgb_params = {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'learning_rate': 0.05,
            'n_estimators': 300,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'reg_alpha': 0.05,
            'reg_lambda': 0.5,
            'gamma': 0.1,
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0
        }
            
        if use_gpu:
            self.xgb_params.update({
                'tree_method': 'gpu_hist',
                'predictor': 'gpu_predictor'
            })
            
        self.lgb_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_data_in_leaf': 20,
            'lambda_l1': 0.05,
            'lambda_l2': 0.5,
            'verbose': -1,
            'random_state': 42,
            'n_jobs': -1
        }
            
        self.cb_params = {
            'iterations': 300,
            'learning_rate': 0.05,
            'depth': 6,
            'l2_leaf_reg': 3,
            'min_data_in_leaf': 20,
            'random_strength': 1.0,
            'bagging_temperature': 0.8,
            'verbose': False,
            'random_state': 42,
            'thread_count': -1
        }
            
        if use_gpu:
            self.cb_params.update({
                'task_type': 'GPU',
                'devices': '0'
            })
    
    def optimize_hyperparameters(self, X_train, y_train):
        """使用Optuna优化超参数"""
        if not self.use_optuna:
            return
            
        logger.info("开始Optuna超参数优化...")
        self.optimizer = OptunaHyperparameterOptimizer(
            use_gpu=self.use_gpu,
            n_trials=self.optuna_trials
        )
            
        self.optimized_params = self.optimizer.optimize(X_train, y_train, model_type='all')
            
        # 更新模型参数
        if 'xgboost' in self.optimized_params:
            self.xgb_params.update(self.optimized_params['xgboost'])
            logger.info(f"XGBoost参数已更新: {self.optimized_params['xgboost']}")
        if 'lightgbm' in self.optimized_params:
            self.lgb_params.update(self.optimized_params['lightgbm'])
            logger.info(f"LightGBM参数已更新: {self.optimized_params['lightgbm']}")
        if 'catboost' in self.optimized_params:
            self.cb_params.update(self.optimized_params['catboost'])
            logger.info(f"CatBoost参数已更新: {self.optimized_params['catboost']}")
            
        # 保存优化结果
        params_file = os.path.expanduser('~/autodl-tmp/best_params.json')
        self.optimizer.save_best_params(params_file)
        logger.info(f"优化后的参数已保存至: {params_file}")
    
    def predict_ensemble(self, X: np.ndarray) -> np.ndarray:
        """集成预测"""
        predictions = []
        weights = []
            
        if 'xgb' in self.models:
            try:
                dmatrix = xgb.DMatrix(X)
                pred = self.models['xgb'].predict(dmatrix)
                predictions.append(pred)
                weights.append(1.0)
            except Exception as e:
                logger.error(f"XGBoost预测失败: {e}")
            
        if 'lgb' in self.models:
            try:
                pred = self.models['lgb'].predict(X)
                predictions.append(pred)
                weights.append(1.2)
            except Exception as e:
                logger.error(f"LightGBM预测失败: {e}")
            
        if 'cb' in self.models:
            try:
                pred = self.models['cb'].predict(X)
                predictions.append(pred)
                weights.append(1.1)
            except Exception as e:
                logger.error(f"CatBoost预测失败: {e}")
            
        if predictions:
            weights = np.array(weights) / np.sum(weights)
            ensemble_pred = np.average(predictions, axis=0, weights=weights)
            ensemble_pred = ensemble_pred * 1.5
            return ensemble_pred
        else:
            logger.error("所有模型预测都失败了")
            return np.zeros(X.shape[0])
        
    @monitor_memory
    def prepare_data(self, features_df: pd.DataFrame, targets_df: pd.Series, 
                    test_size: float = 0.25) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """准备数据 - 完全无未来函数版本"""
        logger.info("准备训练数据...")
        logger.info(f"原始特征形状: {features_df.shape}")
        logger.info(f"原始目标形状: {targets_df.shape}")
            
        # 检查时间一致性
        if not self.leakage_checker.check_temporal_consistency(features_df):
            raise ValueError("特征数据时间序列不一致")
            
        # 数据清理
        features_clean = features_df.copy()
            
        # 移除常数特征
        feature_vars = features_clean.var()
        constant_features = feature_vars[feature_vars < 1e-10].index
        if len(constant_features) > 0:
            logger.info(f"移除 {len(constant_features)} 个常数特征")
            features_clean = features_clean.drop(columns=constant_features)
            
        # 移除高缺失率特征
        null_ratios = features_clean.isnull().sum() / len(features_clean)
        high_null_features = null_ratios[null_ratios > 0.5].index
        if len(high_null_features) > 0:
            logger.info(f"移除 {len(high_null_features)} 个高缺失率特征")
            features_clean = features_clean.drop(columns=high_null_features)
            
        # 处理剩余缺失值 - 只使用向前填充
        features_clean = features_clean.fillna(method='ffill', limit=5)
            
        # 时间序列分割（保持时间顺序）
        split_idx = int(len(features_clean) * (1 - test_size))
            
        X_train = features_clean.iloc[:split_idx]
        X_test = features_clean.iloc[split_idx:]
        y_train = targets_df.iloc[:split_idx]
        y_test = targets_df.iloc[split_idx:]
            
        # 分别在训练集和测试集上处理（避免数据泄露）
        # 训练集处理
        for col in X_train.columns:
            train_mean = X_train[col].mean()
            X_train.loc[:, col] = X_train[col].fillna(train_mean)
                
            # 使用训练集的统计量处理测试集
            X_test.loc[:, col] = X_test[col].fillna(train_mean)
            
        X_train = X_train.fillna(0)
        X_test = X_test.fillna(0)
            
        # 处理无穷值
        X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
        X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)
            
        # 异常值处理（只使用训练集的分位数）
        if len(X_train) > 100:
            for col in X_train.columns:
                # 只使用训练集计算分位数
                q05 = X_train[col].quantile(0.05)
                q95 = X_train[col].quantile(0.95)
                    
                # 使用训练集的分位数裁剪训练集和测试集
                X_train.loc[:, col] = X_train[col].clip(q05, q95)
                X_test.loc[:, col] = X_test[col].clip(q05, q95)
    
                    
        # 处理目标变量
        # 在 prepare_data 方法中修改目标变量处理
        # 处理目标变量
        y_train_clean = y_train.copy()
        y_test_clean = y_test.copy()
    
        # 使用训练集的分位数处理目标变量
        if len(y_train_clean) > 100:
            target_q05 = y_train_clean.quantile(0.05)
            target_q95 = y_train_clean.quantile(0.95)
            y_train_clean = y_train_clean.clip(target_q05, target_q95)
            y_test_clean = y_test_clean.clip(target_q05, target_q95)
            
        logger.info(f"训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")
            
        # 特征选择（只使用训练集）
        # 动态设置特征数量上限
        total_features = len(X_train.columns)
        if FACTORS_AVAILABLE and total_features > 500:
            # 如果有扩展因子库，选择更多特征
            n_features = min(300, total_features)
        elif total_features > 100:
            n_features = min(100, total_features)
        else:
            n_features = total_features
            
        if total_features > n_features:
            logger.info(f"执行特征选择，从 {total_features} 个特征中选择 {n_features} 个...")
            try:
                # 使用互信息进行特征选择
                mi_scores = mutual_info_regression(
                    X_train.values, 
                    y_train_clean.values, 
                    random_state=42,
                    n_neighbors=5
                )
                    
                feature_importance = pd.Series(mi_scores, index=X_train.columns)
                    
                # 选择前n_features个特征
                top_features = feature_importance.nlargest(n_features).index
                    
                X_train = X_train[top_features]
                X_test = X_test[top_features]
                self.selected_features = top_features.tolist()
                    
                logger.info(f"选择了 {len(self.selected_features)} 个特征")
                    
                # 检查特征泄露
                suspicious_features = self.leakage_checker.check_feature_target_leakage(
                    X_train, y_train_clean, threshold=0.8
                )
                if suspicious_features:
                    logger.warning(f"发现可疑特征（可能存在数据泄露）: {suspicious_features}")
                    # 移除可疑特征
                    X_train = X_train.drop(columns=suspicious_features, errors='ignore')
                    X_test = X_test.drop(columns=suspicious_features, errors='ignore')
                    self.selected_features = [f for f in self.selected_features if f not in suspicious_features]
                # 在特征选择完成后（大约第2386行），可以添加：
                # 对选中的特征进行未来函数检查
                for feature in self.selected_features[:10]:  # 检查前10个重要特征
                    if _check_factor_future_leakage(X_train[feature], X_train):
                        logger.warning(f"Feature {feature} may contain future information")
            except Exception as e:
                logger.error(f"特征选择失败: {e}")
                self.selected_features = X_train.columns.tolist()
        else:
            self.selected_features = X_train.columns.tolist()
            
        # 数据缩放（只在训练集上fit）
        self.scaler = RobustScaler()
        X_train_scaled = self.scaler.fit_transform(X_train.values)
        X_test_scaled = self.scaler.transform(X_test.values)  # 使用训练集的参数transform
        self.scaler_fitted = True
            
        X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)
            
        logger.info(f"最终特征维度: {X_train_scaled.shape[1]}")
            
        return X_train_scaled, X_test_scaled, y_train_clean.values, y_test_clean.values
        
    @monitor_memory
    def train_ensemble_models(self, X_train: np.ndarray, X_test: np.ndarray, 
                            y_train: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """训练集成模型 - 完全无未来函数版本"""
        results = {}
            
        logger.info("开始训练集成模型...")
        logger.info(f"训练数据形状: {X_train.shape}")
        logger.info(f"目标变量统计: 均值={y_train.mean():.6f}, 标准差={y_train.std():.6f}")
            
        # 添加超参数优化
        if self.use_optuna:
            self.optimize_hyperparameters(X_train, y_train)
            
        # 训练XGBoost
        logger.info("训练XGBoost...")
        try:
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dtest = xgb.DMatrix(X_test, label=y_test)
                
            self.models['xgb'] = xgb.train(
                self.xgb_params,
                dtrain,
                num_boost_round=self.xgb_params['n_estimators'],
                evals=[(dtest, 'eval')],
                early_stopping_rounds=30,
                verbose_eval=False
            )
                
            pred = self.models['xgb'].predict(dtest)
                
            # 检查数据泄露
            if self.leakage_checker.check_prediction_leakage(pred, y_test, threshold=0.5):
                logger.warning("XGBoost模型可能存在数据泄露！")
                
            results['xgb_rmse'] = np.sqrt(np.mean((y_test - pred) ** 2))
                
            correlation = np.corrcoef(pred, y_test)[0, 1]
            results['xgb_correlation'] = correlation if not np.isnan(correlation) else 0
                
            direction_accuracy = np.mean(np.sign(pred) == np.sign(y_test))
            results['xgb_direction_acc'] = direction_accuracy
                
            logger.info(f"XGBoost - RMSE: {results['xgb_rmse']:.6f}, "
                       f"相关性: {results['xgb_correlation']:.4f}, "
                       f"方向准确率: {direction_accuracy:.4f}")
                
        except Exception as e:
            logger.error(f"XGBoost训练失败: {e}")
            
        # 训练LightGBM
        logger.info("训练LightGBM...")
        try:
            train_data = lgb.Dataset(X_train, label=y_train)
            valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
                
            self.models['lgb'] = lgb.train(
                self.lgb_params,
                train_data,
                num_boost_round=300,
                valid_sets=[valid_data],
                callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)]
            )
                
            pred = self.models['lgb'].predict(X_test)
                
            # 检查数据泄露
            if self.leakage_checker.check_prediction_leakage(pred, y_test, threshold=0.5):
                logger.warning("LightGBM模型可能存在数据泄露！")
                
            results['lgb_rmse'] = np.sqrt(np.mean((y_test - pred) ** 2))
                
            correlation = np.corrcoef(pred, y_test)[0, 1]
            results['lgb_correlation'] = correlation if not np.isnan(correlation) else 0
                
            direction_accuracy = np.mean(np.sign(pred) == np.sign(y_test))
            results['lgb_direction_acc'] = direction_accuracy
                
            logger.info(f"LightGBM - RMSE: {results['lgb_rmse']:.6f}, "
                       f"相关性: {results['lgb_correlation']:.4f}, "
                       f"方向准确率: {direction_accuracy:.4f}")
                
        except Exception as e:
            logger.error(f"LightGBM训练失败: {e}")
            
        # 训练CatBoost
        logger.info("训练CatBoost...")
        try:
            train_pool = Pool(X_train, y_train)
            test_pool = Pool(X_test, y_test)
                
            self.models['cb'] = CatBoostRegressor(**self.cb_params)
            self.models['cb'].fit(
                train_pool,
                eval_set=test_pool,
                early_stopping_rounds=30,
                verbose=False
            )
                
            pred = self.models['cb'].predict(X_test)
                
            # 检查数据泄露
            if self.leakage_checker.check_prediction_leakage(pred, y_test, threshold=0.5):
                logger.warning("CatBoost模型可能存在数据泄露！")
                
            results['cb_rmse'] = np.sqrt(np.mean((y_test - pred) ** 2))
                
            correlation = np.corrcoef(pred, y_test)[0, 1]
            results['cb_correlation'] = correlation if not np.isnan(correlation) else 0
                
            direction_accuracy = np.mean(np.sign(pred) == np.sign(y_test))
            results['cb_direction_acc'] = direction_accuracy
                
            logger.info(f"CatBoost - RMSE: {results['cb_rmse']:.6f}, "
                       f"相关性: {results['cb_correlation']:.4f}, "
                       f"方向准确率: {direction_accuracy:.4f}")
                
        except Exception as e:
            logger.error(f"CatBoost训练失败: {e}")
            
        # 计算集成预测效果
        if len(self.models) > 1:
            logger.info("评估集成模型效果...")
            ensemble_pred = self.predict_ensemble(X_test)
                
            # 检查集成模型的数据泄露
            if self.leakage_checker.check_prediction_leakage(ensemble_pred, y_test, threshold=0.5):
                logger.warning("集成模型可能存在数据泄露！")
                
            ensemble_correlation = np.corrcoef(ensemble_pred, y_test)[0, 1]
            ensemble_direction_acc = np.mean(np.sign(ensemble_pred) == np.sign(y_test))
                
            results['ensemble_correlation'] = ensemble_correlation if not np.isnan(ensemble_correlation) else 0
            results['ensemble_direction_acc'] = ensemble_direction_acc
                
            logger.info(f"集成模型 - 相关性: {ensemble_correlation:.4f}, "
                       f"方向准确率: {ensemble_direction_acc:.4f}")
            
        return results
    
# ================== 结果分析器 ==================
class ResultAnalyzer:
   """结果分析器"""
       
   def __init__(self):
       pass
       
   def plot_results(self, backtest_results: Dict, save_path: str = None):
       """绘制回测结果"""
       if not backtest_results or backtest_results.get('total_trades', 0) == 0:
           logger.warning("没有交易数据，无法绘制图表")
           return
           
       fig, axes = plt.subplots(2, 2, figsize=(15, 10))
       fig.suptitle('No-Future-Function Strategy Backtest Results (with Extended Factors)', fontsize=16)
           
       portfolio_df = backtest_results['portfolio_values']
           
       if len(portfolio_df) > 0:
           axes[0, 0].plot(portfolio_df.index, portfolio_df['portfolio_value'])
           axes[0, 0].set_title('Portfolio Value Over Time')
           axes[0, 0].set_ylabel('Portfolio Value')
           axes[0, 0].grid(True)
           axes[0, 0].tick_params(axis='x', rotation=45)
               
           if 'daily_returns' in backtest_results and len(backtest_results['daily_returns']) > 0:
               returns = backtest_results['daily_returns']
               cumulative = (1 + np.array(returns)).cumprod()
               running_max = np.maximum.accumulate(cumulative)
               drawdown = (cumulative - running_max) / running_max
                   
               axes[0, 1].fill_between(range(len(drawdown)), drawdown, 0, alpha=0.3, color='red')
               axes[0, 1].set_title('Drawdown')
               axes[0, 1].set_ylabel('Drawdown')
               axes[0, 1].grid(True)
               
           if backtest_results['trades']:
               pnls = [trade.net_pnl for trade in backtest_results['trades']]
               axes[1, 0].hist(pnls, bins=30, alpha=0.7, color='blue', edgecolor='black')
               axes[1, 0].axvline(x=0, color='red', linestyle='--', alpha=0.5)
               axes[1, 0].set_title('PnL Distribution')
               axes[1, 0].set_xlabel('PnL')
               axes[1, 0].set_ylabel('Frequency')
               axes[1, 0].grid(True, alpha=0.3)
               
           if len(portfolio_df) > 60:
               rolling_returns = portfolio_df['portfolio_value'].pct_change().rolling(60, center=False)
               rolling_mean = rolling_returns.mean()
               rolling_std = rolling_returns.std()
               periods_per_year = 252 * 48
               rolling_sharpe = (rolling_mean / (rolling_std + 1e-10)) * np.sqrt(periods_per_year)
                   
               axes[1, 1].plot(portfolio_df.index[60:], rolling_sharpe.iloc[60:])
               axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
               axes[1, 1].set_title('Rolling Sharpe Ratio (60 periods)')
               axes[1, 1].set_ylabel('Sharpe Ratio')
               axes[1, 1].grid(True)
               axes[1, 1].tick_params(axis='x', rotation=45)
           
       plt.tight_layout()
           
       if save_path:
           plt.savefig(save_path, dpi=300, bbox_inches='tight')
           logger.info(f"Results plot saved to {save_path}")
           
       plt.show()
       
   def generate_report(self, backtest_results: Dict) -> str:
       """生成回测报告"""
       trading_days = backtest_results.get('trading_days', 0)
       years = backtest_results.get('years', 0)
           
       report = f"""
=== 完全无未来函数策略回测报告 (含扩展因子) ===
    
基本统计:
- 总收益率: {backtest_results['total_return']:.2%}
- 年化收益率: {backtest_results['annual_return']:.2%}
- 年化波动率: {backtest_results['volatility']:.2%}
- 夏普比率: {backtest_results['sharpe_ratio']:.2f}
- Sortino比率: {backtest_results.get('sortino_ratio', 0):.2f}
- 最大回撤: {backtest_results['max_drawdown']:.2%}
    
交易统计:
- 总交易次数: {backtest_results['total_trades']}
- 胜率: {backtest_results['win_rate']:.2%}
- 盈亏比: {backtest_results['profit_factor']:.2f}
- 平均持仓数: {backtest_results.get('avg_positions', 0):.1f}
    
风险指标:
- 卡尔马比率: {backtest_results['calmar_ratio']:.2f}
    
成本分析:
- 总手续费: {backtest_results['total_commission']:.2f}
- 总滑点成本: {backtest_results['total_slippage']:.2f}
- 手续费占比: {backtest_results['total_commission'] / 1000000 * 100:.2%}
    
时间参数:
- 数据频率: 30分钟K线
- 交易天数: {trading_days}天
- 实际年数: {years:.2f}年
    
特征库信息:
- 扩展因子库状态: {'已加载' if FACTORS_AVAILABLE else '未加载'}
"""
       if FACTORS_AVAILABLE:
           report += f"- 可用因子函数数量: {len(factors.func_list)}\n"
           
       report += "\n退出原因分析:\n"
       if 'exit_reasons' in backtest_results:
           for reason, count in backtest_results['exit_reasons'].items():
               report += f"- {reason}: {count}次\n"
          
       return report
    
# ================== 缓存管理类 ==================
class ProcessedDataCache:
   """处理后数据的缓存管理器"""
       
   def __init__(self, cache_dir: str = '~/autodl-tmp/processed_data_cache'):
       self.cache_dir = os.path.expanduser(cache_dir)
       if not os.path.exists(self.cache_dir):
           os.makedirs(self.cache_dir)
           
       self.metadata_file = os.path.join(self.cache_dir, 'cache_metadata.pkl')
       self.metadata = self._load_metadata()
       
   def _load_metadata(self) -> Dict:
       """加载缓存元数据"""
       if os.path.exists(self.metadata_file):
           try:
               with open(self.metadata_file, 'rb') as f:
                   return pickle.load(f)
           except:
               return {}
       return {}
       
   def _save_metadata(self):
       """保存缓存元数据"""
       with open(self.metadata_file, 'wb') as f:
           pickle.dump(self.metadata, f)
       
   def _get_cache_key(self, symbol: str, data_type: str, params: dict) -> str:
       """生成缓存键"""
       params_str = str(sorted(params.items()))
       hash_str = hashlib.md5(f"{symbol}_{data_type}_{params_str}".encode()).hexdigest()
       return f"{symbol}_{data_type}_{hash_str}"
       
   def is_cached(self, symbol: str, data_type: str, params: dict = None) -> bool:
       """检查数据是否已缓存"""
       if params is None:
           params = {}
       cache_key = self._get_cache_key(symbol, data_type, params)
           
       if cache_key in self.metadata:
           cache_info = self.metadata[cache_key]
           cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
               
           if os.path.exists(cache_file):
               age_days = (datetime.now() - cache_info['timestamp']).days
               if age_days <= 7:
                   return True
           
       return False
       
   def save_data(self, symbol: str, data_type: str, data: Any, params: dict = None):
       """保存处理后的数据"""
       if params is None:
           params = {}
           
       cache_key = self._get_cache_key(symbol, data_type, params)
       cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
           
       with open(cache_file, 'wb') as f:
           pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
           
       self.metadata[cache_key] = {
           'symbol': symbol,
           'data_type': data_type,
           'params': params,
           'timestamp': datetime.now(),
           'file_path': cache_file
       }
       self._save_metadata()
           
       logger.info(f"Cached {data_type} data for {symbol}")
       
   def load_data(self, symbol: str, data_type: str, params: dict = None) -> Optional[Any]:
       """加载缓存的数据"""
       if params is None:
           params = {}
           
       cache_key = self._get_cache_key(symbol, data_type, params)
           
       if cache_key in self.metadata:
           cache_file = self.metadata[cache_key]['file_path']
               
           if os.path.exists(cache_file):
               try:
                   with open(cache_file, 'rb') as f:
                       data = pickle.load(f)
                   logger.info(f"Loaded cached {data_type} data for {symbol}")
                   return data
               except Exception as e:
                   logger.error(f"Error loading cached data for {symbol}: {e}")
           
       return None
       
   def clear_old_cache(self, days: int = 30):
       """清理旧缓存"""
       current_time = datetime.now()
       keys_to_remove = []
           
       for cache_key, info in self.metadata.items():
           age_days = (current_time - info['timestamp']).days
           if age_days > days:
               if os.path.exists(info['file_path']):
                   os.remove(info['file_path'])
               keys_to_remove.append(cache_key)
           
       for key in keys_to_remove:
           del self.metadata[key]
           
       if keys_to_remove:
           self._save_metadata()
           logger.info(f"Cleared {len(keys_to_remove)} old cache entries")
    
# ================== 数据处理函数 ==================
def process_single_symbol_data_no_future(args):
   """完全无未来函数的单品种数据处理函数"""
   symbol, data_path, cache_dir, sample_rate = args
       
   try:
       cache = ProcessedDataCache(cache_dir)
           
       cache_params = {'sample_rate': sample_rate, 'version': 'no_future_extended_v2'}
       cached_data = cache.load_data(symbol, 'min30_data_extended', cache_params)
           
       if cached_data is not None:
           logger.info(f"Loaded cached data for {symbol}")
           return symbol, cached_data, True
           
       file_path = os.path.join(data_path, f"{symbol}.parquet")
           
       if not os.path.exists(file_path):
           logger.warning(f"File not found: {file_path}")
           return symbol, None, False
           
       logger.info(f"Processing {symbol}...")
           
       df = pd.read_parquet(
           file_path,
           columns=['datetime', 'open', 'high', 'low', 'close', 'volume'],
           engine='pyarrow'
       )
           
       df.columns = df.columns.str.lower()
           
       if 'datetime' in df.columns:
           df['datetime'] = pd.to_datetime(df['datetime'])
           df.set_index('datetime', inplace=True)
       else:
           logger.warning(f"No datetime column found for {symbol}")
           return symbol, None, False
           
       # 确保时间序列单调递增
       if not df.index.is_monotonic_increasing:
           logger.warning(f"{symbol}: Sorting time series to ensure monotonic increasing")
           df = df.sort_index()
           
       df = df.dropna(subset=['close', 'volume'])
       df = df[df['volume'] > 0]
           
       min30_data = df.resample('30T').agg({
           'open': 'first',
           'high': 'max',
           'low': 'min',
           'close': 'last',
           'volume': 'sum'
       })
           
       min30_data = min30_data.dropna()
           
       if sample_rate < 1.0 and len(min30_data) > 100:
           original_len = len(min30_data)
           sample_size = int(len(min30_data) * sample_rate)
           sample_indices = np.sort(np.random.choice(len(min30_data), sample_size, replace=False))
           min30_data = min30_data.iloc[sample_indices]
           logger.info(f"  Sampled {sample_rate*100:.0f}% of 30min data: {original_len} -> {len(min30_data)} bars")
           
       min30_data = clean_numeric_data_no_future(min30_data)
           
       logger.info(f"  Final 30min bars: {len(min30_data)}")
           
       if len(min30_data) > 500:
           cache.save_data(symbol, 'min30_data_extended', min30_data, cache_params)
           return symbol, min30_data, False
       else:
           logger.warning(f"  Insufficient data after resampling: {len(min30_data)} bars")
           return symbol, None, False
           
   except Exception as e:
       logger.error(f"Error processing {symbol}: {e}")
       traceback.print_exc()
       return symbol, None, False
    
def process_features_for_symbol_no_future(args):
   """完全无未来函数的特征生成函数（包含扩展因子）"""
   symbol, min30_data, cache_dir = args
       
   try:
       cache = ProcessedDataCache(cache_dir)
           
       cache_params = {'version': 'no_future_extended_v2', 'factors_available': FACTORS_AVAILABLE}
       cached_features = cache.load_data(symbol, 'features_extended', cache_params)
           
       if cached_features is not None:
           return symbol, cached_features['features_df'], cached_features['feature_cols'], True
           
       logger.info(f"Generating features for {symbol} (with extended factors: {FACTORS_AVAILABLE})...")
           
       data = min30_data.copy()
           
       # 使用包含扩展因子的技术指标函数
       data = calculate_safe_technical_indicators(data)
           
       # 选择特征列
       excluded_cols = ['open', 'high', 'low', 'close', 'volume', 'vwap', 'amount']
       feature_cols = [col for col in data.columns if col not in excluded_cols]
           
       logger.info(f"Generated {len(feature_cols)} features for {symbol}")
           
       # 去除前200行（确保所有特征都已稳定）
       if len(data) > 200:
           data = data.iloc[200:]
           logger.info(f"After removing initial rows: {len(data)} samples")
           
       # 特征质量检查和清理
       features_to_remove = []
       for col in feature_cols:
           if col in data.columns:
               factor_data = data[col]
                   
               null_ratio = factor_data.isnull().sum() / len(factor_data)
               if null_ratio > 0.3:
                   features_to_remove.append(col)
                   continue
                   
               if factor_data.var() < 1e-12:
                   features_to_remove.append(col)
                   continue
                   
               inf_ratio = np.isinf(factor_data).sum() / len(factor_data)
               if inf_ratio > 0.1:
                   features_to_remove.append(col)
                   continue
                   
               data[col] = data[col].replace([np.inf, -np.inf], np.nan)
               data[col] = data[col].fillna(method='ffill', limit=5)
                   
               # 使用滚动窗口填充，增加min_periods，明确center=False
               rolling_median = data[col].rolling(window=50, min_periods=20, center=False).median()
               data[col] = data[col].fillna(rolling_median)
               data[col] = data[col].fillna(0)
           
       for col in features_to_remove:
           if col in feature_cols:
               feature_cols.remove(col)
           if col in data.columns:
               data = data.drop(columns=[col])
           
       logger.info(f"After quality check: {len(feature_cols)} valid features for {symbol}")
           
       if len(feature_cols) < 10:
           logger.warning(f"{symbol}: Too few valid features ({len(feature_cols)})")
           return symbol, None, None, False
           
       features_data = {
           'features_df': data,
           'feature_cols': feature_cols
       }
       cache.save_data(symbol, 'features_extended', features_data, cache_params)
           
       return symbol, data, feature_cols, False
           
   except Exception as e:
       logger.error(f"Error generating features for {symbol}: {e}")
       traceback.print_exc()
       return symbol, None, None, False
def prepare_training_data_for_symbol_no_future(args):
    """准备单个品种的训练数据 - 完全无未来函数版本"""
    symbol, features_df, feature_cols, min30_data, look_ahead, min_samples, is_training = args
    
    try:
        if len(features_df) < min_samples:
            logger.warning(f"{symbol}: Insufficient data ({len(features_df)} < {min_samples})")
            return None
        
        common_index = features_df.index.intersection(min30_data.index)
        if len(common_index) < min_samples:
            logger.warning(f"{symbol}: Insufficient aligned data ({len(common_index)})")
            return None
        
        aligned_features = features_df.loc[common_index]
        aligned_prices = min30_data.loc[common_index]
        
        # 创建目标变量时明确指定是否为训练模式
        target = create_trading_target_variable_no_future(
            aligned_prices, 
            look_ahead, 
            is_training=is_training  # 添加训练模式标志
        )
        
        final_index = aligned_features.index.intersection(target.index)
        if len(final_index) < min_samples:
            logger.warning(f"{symbol}: Insufficient final data ({len(final_index)})")
            return None
        
        X = aligned_features.loc[final_index, feature_cols]
        y = target.loc[final_index]
        
        # 训练模式下，排除最后的look_ahead个数据点
        # 预测模式下，不排除
        if is_training and len(X) > look_ahead:
            X = X.iloc[:-look_ahead]
            y = y.iloc[:-look_ahead]
        
        X_values = X.values
        X_values = np.nan_to_num(X_values, nan=0.0, posinf=0.0, neginf=0.0)
        
        X_clean = pd.DataFrame(X_values, columns=feature_cols, index=X.index)
        
        # 训练模式下，只保留有效的训练样本
        # 预测模式下，保留所有样本
        if is_training:
            valid_rows = ~np.any(np.isinf(X_values), axis=1) & ~pd.isna(y.values) & (y.values != 0)
        else:
            valid_rows = ~np.any(np.isinf(X_values), axis=1) & ~pd.isna(y.values)
        
        if valid_rows.sum() > min_samples:
            logger.info(f"{symbol}: Generated {valid_rows.sum()} valid {'training' if is_training else 'prediction'} samples")
            return {
                'symbol': symbol,
                'X': X_clean[valid_rows],
                'y': y[valid_rows],
                'count': valid_rows.sum()
            }
        else:
            logger.warning(f"{symbol}: Too few valid samples ({valid_rows.sum()})")
        
        return None
        
    except Exception as e:
        logger.error(f"Error processing {symbol}: {e}")
        traceback.print_exc()
        return None
    
# ================== 完全无未来函数的数据加载器 ==================
class NoFutureDataLoader:
   """完全无未来函数的数据加载器"""
       
   def __init__(self, data_path: str, info_path: str, cache_dir: str = '~/autodl-tmp/processed_data_cache'):
       self.data_path = os.path.expanduser(data_path)
       self.info_path = os.path.expanduser(info_path)
       self.cache_dir = os.path.expanduser(cache_dir)
       self.cache = ProcessedDataCache(cache_dir)
       self.commodity_info = None
           
       self.n_processes = min(4, cpu_count() - 1)
       logger.info(f"Using {self.n_processes} processes for data loading")
       
   def load_commodity_info(self) -> pd.DataFrame:
       """加载商品信息"""
       try:
           self.commodity_info = pd.read_csv(self.info_path)
           logger.info(f"Loaded info for {len(self.commodity_info)} commodities")
           return self.commodity_info
       except Exception as e:
           logger.error(f"Error loading commodity info: {e}")
           return None
       
   def get_available_symbols(self) -> List[str]:
       """获取可用的品种代码"""
       available_symbols = []
       if os.path.exists(self.data_path):
           for filename in os.listdir(self.data_path):
               if filename.endswith('.parquet'):
                   symbol = filename.replace('.parquet', '')
                   available_symbols.append(symbol)
       return available_symbols
       
   @monitor_memory
   def load_multiple_symbols_parallel(self, symbols: List[str], sample_rate: float = 1.0) -> Dict[str, pd.DataFrame]:
       """并行加载多个品种的数据"""
       logger.info(f"Loading data for {len(symbols)} symbols using {self.n_processes} processes...")
           
       args_list = [(symbol, self.data_path, self.cache_dir, sample_rate) for symbol in symbols]
           
       min30_data = {}
       cached_count = 0
       processed_count = 0
           
       with ProcessPoolExecutor(max_workers=self.n_processes) as executor:
           future_to_symbol = {
               executor.submit(process_single_symbol_data_no_future, args): args[0] 
               for args in args_list
           }
               
           with tqdm(total=len(symbols), desc="Processing symbols") as pbar:
               for future in as_completed(future_to_symbol):
                   symbol = future_to_symbol[future]
                   try:
                       symbol, data, from_cache = future.result()
                       if data is not None:
                           min30_data[symbol] = data
                           if from_cache:
                               cached_count += 1
                           else:
                               processed_count += 1
                       pbar.update(1)
                   except Exception as e:
                       logger.error(f"Error processing {symbol}: {e}")
                       pbar.update(1)
           
       logger.info(f"Successfully loaded {len(min30_data)} symbols")
       logger.info(f"  - From cache: {cached_count}")
       logger.info(f"  - Newly processed: {processed_count}")
           
       return min30_data
       
   @monitor_memory
   def generate_features_parallel(self, min30_data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Tuple[pd.DataFrame, List[str]]]:
       """并行生成特征"""
       logger.info(f"Generating features for {len(min30_data_dict)} symbols using {self.n_processes} processes...")
           
       args_list = [
           (symbol, data, self.cache_dir) 
           for symbol, data in min30_data_dict.items()
       ]
           
       features_dict = {}
       cached_count = 0
       processed_count = 0
           
       with ProcessPoolExecutor(max_workers=self.n_processes) as executor:
           future_to_symbol = {
               executor.submit(process_features_for_symbol_no_future, args): args[0] 
               for args in args_list
           }
               
           with tqdm(total=len(min30_data_dict), desc="Generating features") as pbar:
               for future in as_completed(future_to_symbol):
                   symbol = future_to_symbol[future]
                   try:
                       symbol, features_df, feature_cols, from_cache = future.result()
                       if features_df is not None and feature_cols is not None:
                           features_dict[symbol] = (features_df, feature_cols)
                           if from_cache:
                               cached_count += 1
                           else:
                               processed_count += 1
                       pbar.update(1)
                   except Exception as e:
                       logger.error(f"Error generating features for {symbol}: {e}")
                       pbar.update(1)
           
       logger.info(f"Successfully generated features for {len(features_dict)} symbols")
       logger.info(f"  - From cache: {cached_count}")
       logger.info(f"  - Newly processed: {processed_count}")
           
       return features_dict
    
# ================== 完全无未来函数的主策略类 ==================
class NoFutureCommodityStrategy:
   def __init__(self, data_path: str, info_path: str, use_gpu: bool = True, 
                 use_cache: bool = True, cache_dir: str = '~/autodl-tmp/processed_data_cache',
                 use_optuna: bool = False, optuna_trials: int = 30):
            
        self.data_path = data_path
        self.info_path = info_path
        self.use_gpu = use_gpu
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        self.use_optuna = use_optuna
        self.optuna_trials = optuna_trials
            
        self.data_loader = NoFutureDataLoader(data_path, info_path, cache_dir)
        self.model_trainer = NoFutureModelTrainer(
            use_gpu=use_gpu, 
            use_optuna=use_optuna,
            optuna_trials=optuna_trials
        )
        self.signal_generator = None
        self.backtest_engine = NoFutureBacktestEngine(commission_rate=0.0002)
        self.result_analyzer = ResultAnalyzer()
            
        self.min30_data = {}
        self.features_data = {}
        self.signals = {}
   @monitor_memory
   def load_and_process_data(self, symbols: List[str] = None, 
                            max_symbols: int = 10,
                            sample_rate: float = 1.0) -> bool:
       """加载和处理数据"""
       logger.info("="*60)
       logger.info("Loading commodity data (Complete No Future Function Version with Extended Factors)...")
           
       commodity_info = self.data_loader.load_commodity_info()
       if commodity_info is None:
           return False
           
       available_symbols = self.data_loader.get_available_symbols()
           
       if symbols is None:
           symbols = available_symbols[:max_symbols]
       else:
           symbols = [s for s in symbols if s in available_symbols]
           
       if self.use_cache:
           self.data_loader.cache.clear_old_cache(days=30)
           
       self.min30_data = self.data_loader.load_multiple_symbols_parallel(symbols, sample_rate)
           
       if not self.min30_data:
           logger.error("No data loaded successfully")
           return False
           
       self.features_data = self.data_loader.generate_features_parallel(self.min30_data)
           
       if not self.features_data:
           logger.error("No features generated successfully")
           return False
           
       logger.info(f"Data processing completed! Processed {len(self.features_data)} symbols")
       return True
       
   @monitor_memory
   def prepare_and_train_models(self, look_ahead: int = 3, min_samples: int = 300) -> bool:
        """准备训练数据并训练模型 - 明确区分训练模式"""
        logger.info("="*60)
        logger.info("Preparing training data (Complete No Future Function Version with Extended Factors)...")

        args_list = []
        for symbol, (features_df, feature_cols) in self.features_data.items():
            args_list.append((
                symbol, features_df, feature_cols, 
                self.min30_data[symbol], look_ahead, min_samples,
                True  # is_training=True 明确指定为训练模式
            ))

        processed_data_list = []

        with ProcessPoolExecutor(max_workers=self.data_loader.n_processes) as executor:
            future_to_symbol = {
                executor.submit(prepare_training_data_for_symbol_no_future, args): args[0]
                for args in args_list
            }

            with tqdm(total=len(args_list), desc="Processing symbols for training") as pbar:
                for future in as_completed(future_to_symbol):
                    result = future.result()
                    if result is not None:
                        processed_data_list.append(result)
                        logger.info(f"  {result['symbol']}: {result['count']} samples")
                    pbar.update(1)

        if not processed_data_list:
            logger.warning("No valid training data generated")
            return False

        logger.info(f"Combining data from {len(processed_data_list)} symbols...")

        total_samples = sum([d['count'] for d in processed_data_list])
        logger.info(f"  Total samples: {total_samples}")

        combined_features = pd.concat([d['X'] for d in processed_data_list], ignore_index=True)
        combined_targets = pd.concat([d['y'] for d in processed_data_list], ignore_index=True)

        logger.info(f"Combined data shape: Features {combined_features.shape}, Targets {combined_targets.shape}")

        # 动态设置采样数量上限
        if FACTORS_AVAILABLE and len(combined_features) > 50000:
            sample_size = 50000
        elif len(combined_features) > 30000:
            sample_size = 30000
        else:
            sample_size = len(combined_features)

        if len(combined_features) > sample_size:
            logger.info(f"  Sampling from {len(combined_features)} to {sample_size} samples")
            sample_indices = np.random.choice(len(combined_features), sample_size, replace=False)
            combined_features = combined_features.iloc[sample_indices]
            combined_targets = combined_targets.iloc[sample_indices]

        del processed_data_list
        gc.collect()

        X_train, X_test, y_train, y_test = self.model_trainer.prepare_data(
            combined_features, combined_targets, test_size=0.25
        )

        logger.info("Training ensemble models (Complete No Future Function Version with Extended Factors)...")
        results = self.model_trainer.train_ensemble_models(X_train, X_test, y_train, y_test)

        logger.info("Model training completed!")
        for model, metric_value in results.items():
            logger.info(f"  {model}: {metric_value:.6f}")

        best_correlation = max([
            results.get('xgb_correlation', 0),
            results.get('lgb_correlation', 0),
            results.get('cb_correlation', 0)
        ])

        if best_correlation < 0.01:
            logger.warning(f"模型预测性能较差，最佳相关性: {best_correlation:.4f}")
        else:
            logger.info(f"模型预测性能良好，最佳相关性: {best_correlation:.4f}")

        return True
       
   def run_full_strategy(self, max_symbols: int = 10,
                        look_ahead: int = 3,
                        start_date: str = None, 
                        end_date: str = None,
                        sample_rate: float = 1.0):
       """运行完整策略"""
       print("="*60)
       print("Commodity Futures Quantitative Strategy - Complete No Future Function Version with Extended Factors")
       print(f"Using {self.data_loader.n_processes} processes")
       print(f"Cache directory: {self.cache_dir}")
       print(f"Extended factors available: {FACTORS_AVAILABLE}")
       if FACTORS_AVAILABLE:
           print(f"Number of factor functions: {len(factors.func_list)}")
       print("="*60)
           
       start_time = time.time()
           
       if not self.load_and_process_data(max_symbols=max_symbols, sample_rate=sample_rate):
           logger.error("Data loading failed")
           return None
           
       data_time = time.time()
       logger.info(f"Data processing time: {(data_time - start_time)/60:.1f} minutes")
           
       if not self.prepare_and_train_models(look_ahead=look_ahead):
           logger.error("Model training failed")
           return None
           
       train_time = time.time()
       logger.info(f"Model training time: {(train_time - data_time)/60:.1f} minutes")
           
       self.signal_generator = NoFutureSignalGenerator(self.model_trainer)
           
       logger.info("Generating trading signals (Complete No Future Function Version with Extended Factors)...")
       for symbol, (features_df, feature_cols) in self.features_data.items():
           signals = self.signal_generator.generate_signals_from_features(
               features_df, feature_cols, self.min30_data[symbol], symbol
           )
           if len(signals) > 0:
               self.signals[symbol] = signals
           
       logger.info("Running backtest...")
       backtest_results = self.backtest_engine.run_backtest(
           self.signals, self.min30_data, start_date, end_date
       )
           
       if backtest_results:
           report = self.result_analyzer.generate_report(backtest_results)
           print(report)
           self.result_analyzer.plot_results(backtest_results)
           
       end_time = time.time()
       logger.info(f"Total runtime: {(end_time - start_time)/60:.1f} minutes")
           
       return backtest_results
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
机器学习优化的信号到仓位映射系统
完全无未来函数，使用ML方法优化仓位决策
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# ================== ML仓位映射器 ==================
class MLPositionMapper:
    """
    使用机器学习模型学习最优的信号到仓位映射
    通过历史数据学习什么样的信号配置什么样的仓位能获得最好的收益
    """
    
    def __init__(self, lookback_window: int = 100, retrain_frequency: int = 500):
        """
        Args:
            lookback_window: 历史数据回望窗口
            retrain_frequency: 重新训练频率
        """
        self.lookback_window = lookback_window
        self.retrain_frequency = retrain_frequency
        
        # 多个映射模型
        self.position_models = {
            'xgb': None,
            'rf': None,
            'mlp': None
        }
        
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.train_counter = 0
        
        # 存储历史数据用于在线学习
        self.history_buffer = {
            'features': [],
            'targets': [],
            'rewards': []
        }
        
        # 超参数
        self.model_params = {
            'xgb': {
                'n_estimators': 100,
                'max_depth': 5,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            },
            'rf': {
                'n_estimators': 100,
                'max_depth': 5,
                'min_samples_split': 10,
                'min_samples_leaf': 5
            },
            'mlp': {
                'hidden_layer_sizes': (50, 30, 10),
                'activation': 'relu',
                'learning_rate': 'adaptive',
                'max_iter': 500
            }
        }
        
    def extract_position_features(self, signal: float, market_state: Dict) -> np.ndarray:
        """
        提取用于仓位决策的特征
        
        Args:
            signal: 原始预测信号
            market_state: 市场状态字典，包含各种市场指标
            
        Returns:
            特征向量
        """
        features = []
        
        # 信号特征
        features.append(signal)
        features.append(abs(signal))
        features.append(signal ** 2)
        features.append(np.sign(signal))
        
        # 信号强度分级
        if abs(signal) < 0.1:
            signal_strength = 0
        elif abs(signal) < 0.3:
            signal_strength = 1
        elif abs(signal) < 0.5:
            signal_strength = 2
        else:
            signal_strength = 3
        features.append(signal_strength)
        
        # 市场状态特征
        if market_state:
            # 波动率特征
            volatility = market_state.get('volatility', 0.02)
            features.append(volatility)
            features.append(1.0 / (volatility + 0.001))  # 波动率倒数
            
            # 趋势特征
            trend = market_state.get('trend', 0)
            features.append(trend)
            features.append(signal * trend)  # 信号与趋势的交互
            
            # 动量特征
            momentum = market_state.get('momentum', 0)
            features.append(momentum)
            features.append(signal * momentum)
            
            # RSI特征
            rsi = market_state.get('rsi', 50)
            features.append((rsi - 50) / 50)  # 归一化RSI
            
            # 成交量特征
            volume_ratio = market_state.get('volume_ratio', 1.0)
            features.append(np.log(volume_ratio + 1))
            
            # 市场微观结构
            spread = market_state.get('spread', 0.001)
            features.append(spread)
            
            # 历史表现
            recent_return = market_state.get('recent_return', 0)
            features.append(recent_return)
            features.append(signal * recent_return)
            
            # 持仓历史
            prev_position = market_state.get('prev_position', 0)
            features.append(prev_position)
            features.append(signal - prev_position)  # 仓位变化
            
        else:
            # 如果没有市场状态，填充默认值
            features.extend([0] * 14)
            
        return np.array(features)
    
    def calculate_position_target(self, signal: float, future_return: float, 
                                 volatility: float) -> float:
        """
        计算理想的目标仓位（用于训练）
        基于未来收益和风险计算最优仓位
        
        Args:
            signal: 预测信号
            future_return: 实际实现的未来收益
            volatility: 波动率
            
        Returns:
            目标仓位
        """
        # 基础目标：如果信号和实际收益方向一致，仓位应该大
        if signal * future_return > 0:
            # 方向正确，根据收益大小决定仓位
            base_position = np.sign(signal) * min(abs(future_return) / volatility, 1.0)
        else:
            # 方向错误，仓位应该小或者反向
            base_position = 0
            
        # 考虑风险调整
        risk_adj = min(1.0, 0.02 / volatility)
        target_position = base_position * risk_adj
        
        # 限制范围
        target_position = np.clip(target_position, -1.0, 1.0)
        
        return target_position
    
    def train_position_models(self, features: np.ndarray, targets: np.ndarray):
        """
        训练仓位映射模型
        
        Args:
            features: 特征矩阵
            targets: 目标仓位
        """
        if len(features) < 100:
            logger.warning("Insufficient data for training position models")
            return
            
        # 数据预处理
        X = self.scaler.fit_transform(features)
        y = targets
        
        # 训练XGBoost
        try:
            self.position_models['xgb'] = xgb.XGBRegressor(**self.model_params['xgb'])
            self.position_models['xgb'].fit(X, y)
            logger.info("XGBoost position model trained")
        except Exception as e:
            logger.error(f"Failed to train XGBoost: {e}")
            
        # 训练随机森林
        try:
            self.position_models['rf'] = RandomForestRegressor(**self.model_params['rf'])
            self.position_models['rf'].fit(X, y)
            logger.info("Random Forest position model trained")
        except Exception as e:
            logger.error(f"Failed to train Random Forest: {e}")
            
        # 训练神经网络
        try:
            self.position_models['mlp'] = MLPRegressor(**self.model_params['mlp'])
            self.position_models['mlp'].fit(X, y)
            logger.info("MLP position model trained")
        except Exception as e:
            logger.error(f"Failed to train MLP: {e}")
            
        self.is_fitted = True
        
    def predict_position(self, signal: float, market_state: Dict) -> float:
        """
        使用训练好的模型预测最优仓位
        
        Args:
            signal: 预测信号
            market_state: 市场状态
            
        Returns:
            预测的最优仓位
        """
        if not self.is_fitted:
            # 如果模型未训练，使用简单规则
            return self._simple_position_rule(signal, market_state)
            
        # 提取特征
        features = self.extract_position_features(signal, market_state)
        X = self.scaler.transform(features.reshape(1, -1))
        
        # 集成预测
        predictions = []
        weights = []
        
        if self.position_models['xgb'] is not None:
            try:
                pred = self.position_models['xgb'].predict(X)[0]
                predictions.append(pred)
                weights.append(1.2)
            except:
                pass
                
        if self.position_models['rf'] is not None:
            try:
                pred = self.position_models['rf'].predict(X)[0]
                predictions.append(pred)
                weights.append(1.0)
            except:
                pass
                
        if self.position_models['mlp'] is not None:
            try:
                pred = self.position_models['mlp'].predict(X)[0]
                predictions.append(pred)
                weights.append(0.8)
            except:
                pass
                
        if predictions:
            # 加权平均
            weights = np.array(weights) / sum(weights)
            position = np.average(predictions, weights=weights)
        else:
            # 回退到简单规则
            position = self._simple_position_rule(signal, market_state)
            
        # 应用约束
        position = self._apply_position_constraints(position, signal, market_state)
        
        return position
    
    def _simple_position_rule(self, signal: float, market_state: Dict) -> float:
        """简单的仓位规则（用于模型未训练时）"""
        volatility = market_state.get('volatility', 0.02)
        
        # 基于波动率调整信号
        vol_adj = min(2.0, 0.02 / volatility)
        position = signal * vol_adj * 0.5  # 保守一点
        
        return np.clip(position, -1.0, 1.0)
    
    def _apply_position_constraints(self, position: float, signal: float, 
                                   market_state: Dict) -> float:
        """
        应用仓位约束和风险管理规则
        
        Args:
            position: 预测的仓位
            signal: 原始信号
            market_state: 市场状态
            
        Returns:
            约束后的仓位
        """
        # 确保仓位方向与信号一致
        if signal * position < 0:
            position = 0
            
        # 波动率约束
        volatility = market_state.get('volatility', 0.02)
        max_position = min(1.0, 0.05 / volatility)  # 根据波动率限制最大仓位
        position = np.clip(position, -max_position, max_position)
        
        # 交易成本考虑
        prev_position = market_state.get('prev_position', 0)
        position_change = abs(position - prev_position)
        
        # 如果仓位变化太小，不值得交易
        if position_change < 0.05:
            position = prev_position
            
        # 极端市场条件下的保护
        if volatility > 0.05:  # 高波动
            position *= 0.5
        
        rsi = market_state.get('rsi', 50)
        if rsi > 80 or rsi < 20:  # 超买超卖
            position *= 0.7
            
        return position
    
    def update_with_feedback(self, features: np.ndarray, position: float, 
                            reward: float):
        """
        使用实际交易反馈更新模型（在线学习）
        
        Args:
            features: 决策时的特征
            position: 实际采取的仓位
            reward: 获得的收益
        """
        self.history_buffer['features'].append(features)
        self.history_buffer['targets'].append(position)
        self.history_buffer['rewards'].append(reward)
        
        # 限制缓冲区大小
        max_buffer = 10000
        if len(self.history_buffer['features']) > max_buffer:
            self.history_buffer['features'] = self.history_buffer['features'][-max_buffer:]
            self.history_buffer['targets'] = self.history_buffer['targets'][-max_buffer:]
            self.history_buffer['rewards'] = self.history_buffer['rewards'][-max_buffer:]
            
        self.train_counter += 1
        
        # 定期重新训练
        if self.train_counter >= self.retrain_frequency and len(self.history_buffer['features']) >= 100:
            self._retrain_models()
            self.train_counter = 0
            
    def _retrain_models(self):
        """使用累积的历史数据重新训练模型"""
        features = np.array(self.history_buffer['features'])
        rewards = np.array(self.history_buffer['rewards'])
        
        # 基于收益构建目标仓位
        # 高收益的决策应该被强化
        targets = []
        for i in range(len(rewards)):
            if i < len(rewards) - 1:
                # 使用未来几期的平均收益
                future_rewards = rewards[i:min(i+5, len(rewards))]
                avg_reward = np.mean(future_rewards)
            else:
                avg_reward = rewards[i]
                
            # 根据收益调整目标仓位
            original_position = self.history_buffer['targets'][i]
            if avg_reward > 0:
                # 收益为正，强化这个仓位
                target = original_position * (1 + min(avg_reward * 10, 0.5))
            else:
                # 收益为负，减少这个仓位
                target = original_position * (1 + max(avg_reward * 10, -0.5))
                
            targets.append(np.clip(target, -1.0, 1.0))
            
        targets = np.array(targets)
        
        # 重新训练模型
        self.train_position_models(features, targets)
        logger.info("Position models retrained with online feedback")


# ================== 市场状态分析器 ==================
class MarketStateAnalyzer:
    """
    分析市场状态，为仓位决策提供上下文信息
    完全不使用未来数据
    """
    
    def __init__(self):
        self.state_cache = {}
        
    def analyze_market_state(self, df: pd.DataFrame, current_idx: int) -> Dict:
        """
        分析当前市场状态
        
        Args:
            df: 包含市场数据的DataFrame
            current_idx: 当前时间点的索引
            
        Returns:
            市场状态字典
        """
        if current_idx < 50:
            return self._get_default_state()
            
        # 只使用current_idx之前的数据
        hist_data = df.iloc[:current_idx]
        
        state = {}
        
        # 计算波动率（使用历史数据）
        if 'returns' in hist_data.columns:
            recent_returns = hist_data['returns'].iloc[-50:]
            state['volatility'] = recent_returns.std()
        else:
            state['volatility'] = 0.02
            
        # 计算趋势（使用移动平均）
        if 'close' in hist_data.columns:
            ma_short = hist_data['close'].iloc[-10:].mean()
            ma_long = hist_data['close'].iloc[-30:].mean()
            state['trend'] = (ma_short / ma_long - 1) if ma_long > 0 else 0
        else:
            state['trend'] = 0
            
        # 计算动量
        if 'close' in hist_data.columns:
            momentum = (hist_data['close'].iloc[-1] / hist_data['close'].iloc[-10] - 1)
            state['momentum'] = momentum
        else:
            state['momentum'] = 0
            
        # RSI
        if 'rsi_14' in hist_data.columns:
            state['rsi'] = hist_data['rsi_14'].iloc[-1]
        else:
            state['rsi'] = 50
            
        # 成交量比率
        if 'volume_ratio' in hist_data.columns:
            state['volume_ratio'] = hist_data['volume_ratio'].iloc[-1]
        else:
            state['volume_ratio'] = 1.0
            
        # 价差
        if 'relative_spread' in hist_data.columns:
            state['spread'] = hist_data['relative_spread'].iloc[-1]
        else:
            state['spread'] = 0.001
            
        # 最近收益
        if 'returns' in hist_data.columns:
            state['recent_return'] = hist_data['returns'].iloc[-5:].mean()
        else:
            state['recent_return'] = 0
            
        return state
    
    def _get_default_state(self) -> Dict:
        """获取默认市场状态"""
        return {
            'volatility': 0.02,
            'trend': 0,
            'momentum': 0,
            'rsi': 50,
            'volume_ratio': 1.0,
            'spread': 0.001,
            'recent_return': 0,
            'prev_position': 0
        }


# ================== 强化学习仓位优化器 ==================
class RLPositionOptimizer:
    """
    使用强化学习思想优化仓位决策
    通过探索和利用平衡来持续改进策略
    """
    
    def __init__(self, epsilon: float = 0.1, learning_rate: float = 0.01):
        """
        Args:
            epsilon: 探索率
            learning_rate: 学习率
        """
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        
        # Q-table近似（使用函数逼近）
        self.value_estimator = None
        
        # 动作空间（离散化的仓位）
        self.action_space = np.linspace(-1.0, 1.0, 21)  # 21个离散仓位
        
        # 经验回放缓冲
        self.experience_buffer = []
        self.max_buffer_size = 5000
        
    def get_position_with_exploration(self, signal: float, market_state: Dict,
                                     ml_position: float) -> float:
        """
        获取带探索的仓位决策
        
        Args:
            signal: 预测信号
            market_state: 市场状态
            ml_position: ML模型预测的仓位
            
        Returns:
            最终仓位决策
        """
        # epsilon-贪婪策略
        if np.random.random() < self.epsilon:
            # 探索：在ML预测附近随机选择
            noise = np.random.normal(0, 0.1)
            position = ml_position + noise
        else:
            # 利用：使用ML预测
            position = ml_position
            
        # 衰减探索率
        self.epsilon *= 0.9995
        self.epsilon = max(self.epsilon, 0.01)  # 保持最小探索
        
        return np.clip(position, -1.0, 1.0)
    
    def store_experience(self, state: Dict, action: float, reward: float, 
                        next_state: Dict):
        """存储经验用于学习"""
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state
        }
        
        self.experience_buffer.append(experience)
        
        # 限制缓冲区大小
        if len(self.experience_buffer) > self.max_buffer_size:
            self.experience_buffer.pop(0)
            
    def learn_from_experience(self, batch_size: int = 32):
        """从经验中学习"""
        if len(self.experience_buffer) < batch_size:
            return
            
        # 随机采样一批经验
        batch = np.random.choice(self.experience_buffer, batch_size, replace=False)
        
        # 这里可以实现更复杂的学习算法
        # 例如DQN、A3C等
        # 简化版本：更新价值估计
        for exp in batch:
            # 计算TD误差等
            # 更新模型参数
            pass


# ================== 集成的智能仓位管理器 ==================
class IntelligentPositionManager:
    """
    集成ML、市场状态分析和RL的智能仓位管理器
    完全无未来函数
    """
    
    def __init__(self):
        self.ml_mapper = MLPositionMapper()
        self.market_analyzer = MarketStateAnalyzer()
        self.rl_optimizer = RLPositionOptimizer()
        
        # 仓位历史
        self.position_history = {}
        
        # 性能追踪
        self.performance_tracker = {
            'decisions': [],
            'rewards': [],
            'cumulative_return': 0
        }
        
    def calculate_optimal_position(self, signal: float, symbol: str, 
                                  df: pd.DataFrame, current_idx: int) -> float:
        """
        计算最优仓位
        
        Args:
            signal: 预测信号
            symbol: 品种代码
            df: 市场数据
            current_idx: 当前时间索引
            
        Returns:
            最优仓位
        """
        # 分析市场状态
        market_state = self.market_analyzer.analyze_market_state(df, current_idx)
        
        # 添加历史仓位信息
        market_state['prev_position'] = self.position_history.get(symbol, 0)
        
        # ML模型预测最优仓位
        ml_position = self.ml_mapper.predict_position(signal, market_state)
        
        # RL优化（带探索）
        final_position = self.rl_optimizer.get_position_with_exploration(
            signal, market_state, ml_position
        )
        
        # 更新历史
        self.position_history[symbol] = final_position
        
        # 记录决策
        self.performance_tracker['decisions'].append({
            'signal': signal,
            'ml_position': ml_position,
            'final_position': final_position,
            'market_state': market_state
        })
        
        return final_position
    
    def update_with_results(self, position: float, realized_return: float,
                          signal: float, market_state: Dict):
        """
        使用实际结果更新模型
        
        Args:
            position: 实际采用的仓位
            realized_return: 实现的收益
            signal: 原始信号
            market_state: 市场状态
        """
        # 更新ML模型
        features = self.ml_mapper.extract_position_features(signal, market_state)
        self.ml_mapper.update_with_feedback(features, position, realized_return)
        
        # 更新RL经验
        next_state = market_state.copy()
        self.rl_optimizer.store_experience(
            market_state, position, realized_return, next_state
        )
        
        # 更新性能追踪
        self.performance_tracker['rewards'].append(realized_return)
        self.performance_tracker['cumulative_return'] += realized_return
        
        # 定期学习
        if len(self.performance_tracker['rewards']) % 100 == 0:
            self.rl_optimizer.learn_from_experience()
            
    def train_initial_models(self, historical_data: pd.DataFrame, 
                           signals: pd.Series, actual_returns: pd.Series):
        """
        使用历史数据训练初始模型
        
        Args:
            historical_data: 历史市场数据
            signals: 历史信号
            actual_returns: 实际收益
        """
        features_list = []
        targets_list = []
        
        # 生成训练数据
        for i in range(50, len(historical_data) - 1):
            # 分析历史市场状态
            market_state = self.market_analyzer.analyze_market_state(
                historical_data, i
            )
            
            # 提取特征
            if i < len(signals):
                signal = signals.iloc[i]
                features = self.ml_mapper.extract_position_features(
                    signal, market_state
                )
                
                # 计算理想仓位（基于实际收益）
                if i < len(actual_returns) - 1:
                    future_return = actual_returns.iloc[i+1]
                    volatility = market_state.get('volatility', 0.02)
                    target_position = self.ml_mapper.calculate_position_target(
                        signal, future_return, volatility
                    )
                    
                    features_list.append(features)
                    targets_list.append(target_position)
                    
        if features_list:
            features_array = np.array(features_list)
            targets_array = np.array(targets_list)
            
            # 训练ML模型
            self.ml_mapper.train_position_models(features_array, targets_array)
            logger.info(f"Trained initial models with {len(features_array)} samples")
            
    def get_performance_summary(self) -> Dict:
        """获取性能总结"""
        if not self.performance_tracker['rewards']:
            return {}
            
        rewards = np.array(self.performance_tracker['rewards'])
        
        return {
            'total_decisions': len(self.performance_tracker['decisions']),
            'average_reward': np.mean(rewards),
            'reward_std': np.std(rewards),
            'cumulative_return': self.performance_tracker['cumulative_return'],
            'sharpe_ratio': np.mean(rewards) / (np.std(rewards) + 1e-10),
            'win_rate': np.mean(rewards > 0),
            'avg_position': np.mean([d['final_position'] 
                                    for d in self.performance_tracker['decisions']])
        }


# ================== 使用示例 ==================
def integrate_intelligent_position_manager(strategy_instance):
    """
    将智能仓位管理器集成到现有策略中
    
    Args:
        strategy_instance: 策略实例
    """
    # 创建智能仓位管理器
    position_manager = IntelligentPositionManager()
    
    # 如果有历史数据，先训练初始模型
    if hasattr(strategy_instance, 'min30_data') and strategy_instance.min30_data:
        for symbol, data in strategy_instance.min30_data.items():
            if symbol in strategy_instance.signals:
                signals = strategy_instance.signals[symbol]
                
                # 计算历史收益
                if 'close' in data.columns:
                    returns = data['close'].pct_change().fillna(0)
                    
                    # 训练初始模型
                    position_manager.train_initial_models(
                        data, signals, returns
                    )
                    break  # 只用第一个品种训练初始模型
                    
    # 替换原有的仓位计算逻辑
    strategy_instance.position_manager = position_manager
    
    logger.info("Intelligent Position Manager integrated successfully")
    
    return position_manager
# ================== 主函数 ==================
if __name__ == "__main__":
    print("Starting Complete No-Future-Function strategy with Optuna Optimization...")
        
    # 创建策略实例，启用Optuna优化
    strategy = NoFutureCommodityStrategy(
        data_path='~/autodl-tmp/data/1m/',
        info_path='~/autodl-tmp/data/info.csv',
        use_gpu=True,
        use_cache=True,
        cache_dir='~/autodl-tmp/processed_data_cache',
        use_optuna=True,  # 启用Optuna优化
        optuna_trials=50   # 设置优化试验次数
    )
        
    # 运行策略 - 注意这里的缩进要和上面的对齐
    results = strategy.run_full_strategy(
        max_symbols=15,
        look_ahead=3,
        start_date='2022-01-01',
        end_date='2024-06-30',
        sample_rate=1.0
    )
        
    print("\n" + "="*60)
    print("策略执行结果检查：")
        
    if results is None:
        print("结果为None - 策略执行失败！")
        print("请检查日志中的错误信息")
    elif not results:
        print("结果为空字典 - 策略执行失败！")
    elif isinstance(results, dict):
        print(f"结果类型正确: {type(results)}")
        print(f"结果包含 {len(results)} 个字段")
            
        # 打印关键指标
        print("\n关键指标：")
        print(f"总收益率: {results.get('total_return', 0):.2%}")
        print(f"年化收益率: {results.get('annual_return', 0):.2%}")
        print(f"夏普比率: {results.get('sharpe_ratio', 0):.2f}")
        print(f"最大回撤: {results.get('max_drawdown', 0):.2%}")
        print(f"总交易次数: {results.get('total_trades', 0)}")
        print(f"胜率: {results.get('win_rate', 0):.2%}")
            
        if results.get('total_trades', 0) > 0:
            print("\n策略执行成功！")
            print("所有未来函数已完全移除，策略100%基于历史数据运行")
            print(f"扩展因子库状态: {'已成功整合' if FACTORS_AVAILABLE else '未加载'}")
        else:
            print("\n策略执行成功但没有产生交易")
            print("可能需要调整信号生成参数")
    else:
        print(f"结果类型错误: {type(results)}")
        print("策略执行失败！")
