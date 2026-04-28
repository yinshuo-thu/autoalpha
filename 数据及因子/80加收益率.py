#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
商品期货量化策略 - 完全修复版V2
修复信号生成、收益计算、交易成本计算等核心问题
添加回测时间范围打印功能
"""

import pandas as pd
import numpy as np
import os
import logging
import warnings
warnings.filterwarnings('ignore')

from multiprocessing import Pool, cpu_count
import time
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import gc
from datetime import datetime, timedelta

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ================== 全局函数（可被pickle序列化） ==================

def load_single_symbol(args: Tuple) -> Tuple:
    """加载单个品种数据"""
    symbol, data_path, sample_rate = args
    
    try:
        file_path = os.path.join(data_path, f"{symbol}.parquet")
        
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            return symbol, None, None
        
        # 读取数据
        df = pd.read_parquet(file_path, engine='pyarrow')
        
        # 确保有datetime列
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
        elif 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
        
        # 确保列名小写
        df.columns = df.columns.str.lower()
        
        # 记录原始数据时间范围
        original_start = df.index.min()
        original_end = df.index.max()
        
        # 生成15分钟K线
        df_15m = df.resample('15T').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        # 记录15分钟K线时间范围（采样前）
        pre_sample_start = df_15m.index.min()
        pre_sample_end = df_15m.index.max()
        
        # 采样处理
        if sample_rate < 1.0 and len(df_15m) > 100:
            sample_size = int(len(df_15m) * sample_rate)
            df_15m = df_15m.iloc[-sample_size:]
        
        # 记录最终时间范围
        final_start = df_15m.index.min()
        final_end = df_15m.index.max()
        
        time_info = {
            'original_start': original_start,
            'original_end': original_end,
            'pre_sample_start': pre_sample_start,
            'pre_sample_end': pre_sample_end,
            'final_start': final_start,
            'final_end': final_end,
            'sample_rate': sample_rate,
            'bars_count': len(df_15m)
        }
        
        logger.info(f"Successfully loaded {symbol}: {len(df_15m)} bars, "
                   f"Time range: {final_start.strftime('%Y-%m-%d %H:%M')} to {final_end.strftime('%Y-%m-%d %H:%M')}")
        return symbol, df_15m, time_info
        
    except Exception as e:
        logger.error(f"Error loading {symbol}: {e}")
        return symbol, None, None


def calculate_features_for_symbol(args: Tuple) -> Tuple:
    """计算单个品种的特征 - 改进版"""
    symbol, df = args
    
    try:
        if df is None or len(df) < 100:
            return symbol, None, None
        
        df = df.copy()
        
        # 1. 价格和收益特征
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # 2. 移动平均特征
        for period in [5, 10, 20, 40, 60]:
            if len(df) >= period:
                df[f'sma_{period}'] = df['close'].rolling(window=period, center=False).mean()
                df[f'sma_{period}_ratio'] = df['close'] / df[f'sma_{period}']
        
        # 3. RSI - 修正版
        for period in [14, 21]:
            if len(df) >= period * 2:
                delta = df['close'].diff()
                gain = delta.where(delta > 0, 0).rolling(window=period, center=False).mean()
                loss = -delta.where(delta < 0, 0).rolling(window=period, center=False).mean()
                rs = gain / (loss + 1e-10)
                df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # 4. 布林带
        for period in [20]:
            if len(df) >= period:
                sma = df['close'].rolling(window=period, center=False).mean()
                std = df['close'].rolling(window=period, center=False).std()
                df[f'bb_upper_{period}'] = sma + 2 * std
                df[f'bb_lower_{period}'] = sma - 2 * std
                df[f'bb_width_{period}'] = (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']) / sma
                df[f'bb_position_{period}'] = (df['close'] - df[f'bb_lower_{period}']) / (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'] + 1e-10)
        
        # 5. 波动率特征
        for period in [10, 20, 40]:
            if len(df) >= period:
                df[f'volatility_{period}'] = df['returns'].rolling(window=period, center=False).std()
                df[f'volatility_ratio_{period}'] = df[f'volatility_{period}'] / df['returns'].rolling(window=60, center=False).std()
        
        # 6. 成交量特征
        if 'volume' in df.columns and len(df) >= 20:
            df['volume_sma_20'] = df['volume'].rolling(window=20, center=False).mean()
            df['volume_ratio'] = df['volume'] / (df['volume_sma_20'] + 1e-10)
            df['volume_trend'] = df['volume_sma_20'].pct_change(10)
        
        # 7. MACD
        if len(df) >= 35:
            ema_12 = df['close'].ewm(span=12, adjust=False).mean()
            ema_26 = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = ema_12 - ema_26
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_diff'] = df['macd'] - df['macd_signal']
            df['macd_diff_norm'] = df['macd_diff'] / (df['close'].rolling(20, center=False).std() + 1e-10)
        
        # 8. 动量指标
        for period in [5, 10, 20]:
            if len(df) >= period + 1:
                df[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1
        
        # 清理数据 - 使用前向填充，避免未来数据泄露
        df = df.fillna(method='ffill', limit=5)
        
        # 删除仍有缺失值的早期数据
        df = df.dropna()
        
        # 选择特征列
        feature_cols = [col for col in df.columns 
                       if col not in ['open', 'high', 'low', 'close', 'volume']]
        
        logger.info(f"Generated {len(feature_cols)} features for {symbol}, {len(df)} valid bars")
        return symbol, df, feature_cols
        
    except Exception as e:
        logger.error(f"Error calculating features for {symbol}: {e}")
        import traceback
        traceback.print_exc()
        return symbol, None, None


def generate_labels_for_symbol(args: Tuple) -> Tuple:
    """生成标签 - 基于未来收益的方向"""
    symbol, df, horizon, threshold = args
    
    try:
        if df is None or len(df) < 200:
            return symbol, None
        
        # 计算未来收益
        future_returns = df['close'].shift(-horizon) / df['close'] - 1
        
        # 生成标签：1=做多，-1=做空，0=不交易
        labels = pd.Series(index=df.index, dtype=float)
        labels[:] = 0
        
        # 基于阈值生成交易信号
        labels[future_returns > threshold] = 1
        labels[future_returns < -threshold] = -1
        
        # 最后horizon个数据没有未来收益，设为0
        labels.iloc[-horizon:] = 0
        
        # 统计
        long_pct = (labels == 1).mean()
        short_pct = (labels == -1).mean()
        neutral_pct = (labels == 0).mean()
        
        logger.info(f"{symbol} labels: Long={long_pct:.2%}, Short={short_pct:.2%}, Neutral={neutral_pct:.2%}")
        
        return symbol, labels
        
    except Exception as e:
        logger.error(f"Error generating labels for {symbol}: {e}")
        return symbol, None


# ================== 交易成本模型 ==================
class TradingCostModel:
    """更真实的交易成本模型"""
    
    def __init__(self):
        # 不同品种的手续费设置（双边）
        self.commission_rates = {
            'IF': 0.000046,  # 股指期货 双边万分之0.46
            'IC': 0.000046,
            'IH': 0.000046,
            'TS': 0.00002,   # 国债期货（简化为比例）双边万分之0.2
            'TF': 0.00002,
            'T': 0.00002,
            'CU': 0.0001,    # 商品期货 双边万分之1
            'AL': 0.0001,
            'ZN': 0.0001,
            'RB': 0.0001,
        }
        
        # 滑点设置（根据流动性）
        self.slippage_rates = {
            'IF': 0.0001,    # 1个基点
            'IC': 0.00015,   # 1.5个基点
            'IH': 0.0001,
            'TS': 0.00005,   # 国债流动性好
            'TF': 0.00005,
            'T': 0.00005,
            'CU': 0.0002,    # 商品滑点较大
            'AL': 0.0002,
            'ZN': 0.0002,
            'RB': 0.00025,   # 螺纹钢滑点更大
        }
        
        # 默认值（保守估计）
        self.default_commission = 0.0001  # 双边万分之1
        self.default_slippage = 0.0002    # 万分之2
    
    def get_total_cost_rate(self, symbol: str) -> float:
        """获取某品种的总交易成本率（双边手续费+滑点）"""
        commission = self.commission_rates.get(symbol, self.default_commission)
        slippage = self.slippage_rates.get(symbol, self.default_slippage)
        return commission + slippage
    
    def calculate_trading_costs(self, symbol: str, position_changes: pd.Series) -> pd.Series:
        """
        计算交易成本
        
        Parameters:
        -----------
        symbol : str
            品种代码
        position_changes : pd.Series
            仓位变化序列（绝对值表示交易量）
        
        Returns:
        --------
        pd.Series
            交易成本序列
        """
        total_cost_rate = self.get_total_cost_rate(symbol)
        return position_changes.abs() * total_cost_rate


# ================== 策略主类 ==================
class ImprovedParallelTradingStrategy:
    """改进版并行交易策略"""
    
    def __init__(self, 
                 data_path: str,
                 n_workers: int = None,
                 use_realistic_costs: bool = True):
        """
        初始化策略
        
        Parameters:
        -----------
        data_path : str
            数据路径
        n_workers : int
            并行工作进程数
        use_realistic_costs : bool
            是否使用真实的交易成本模型
        """
        self.data_path = os.path.expanduser(data_path)
        
        # 设置合理的进程数
        if n_workers is None:
            self.n_workers = min(cpu_count() - 1, 8)
        else:
            self.n_workers = min(n_workers, 8)
        self.n_workers = max(1, self.n_workers)
        
        # 交易成本模型
        self.use_realistic_costs = use_realistic_costs
        if use_realistic_costs:
            self.cost_model = TradingCostModel()
            logger.info("Using realistic trading cost model")
        else:
            # 使用简单的固定成本
            self.simple_transaction_cost = 0.0001  # 双边万分之1
            self.simple_slippage = 0.0002  # 万分之2
            logger.info(f"Using simple fixed costs: commission={self.simple_transaction_cost:.4f}, slippage={self.simple_slippage:.4f}")
        
        # 数据容器
        self.price_data = {}
        self.features_data = {}
        self.labels_data = {}
        self.predictions = {}
        self.positions = {}
        self.model = None
        self.time_info = {}
        
        logger.info(f"Strategy initialized with {self.n_workers} parallel workers")
    
    def load_data_parallel(self, symbols: List[str], sample_rate: float = 1.0):
        """并行加载数据"""
        logger.info("="*60)
        logger.info(f"Loading {len(symbols)} symbols with {self.n_workers} workers...")
        
        start_time = time.time()
        
        # 准备参数
        args_list = [(symbol, self.data_path, sample_rate) for symbol in symbols]
        
        # 使用进程池并行加载
        with Pool(processes=self.n_workers) as pool:
            results = list(tqdm(
                pool.imap_unordered(load_single_symbol, args_list),
                total=len(args_list),
                desc="Loading symbols"
            ))
        
        # 整理结果
        overall_start = None
        overall_end = None
        
        for symbol, data, time_info in results:
            if data is not None and len(data) > 100:
                self.price_data[symbol] = data
                self.time_info[symbol] = time_info
                
                # 更新整体时间范围
                if overall_start is None or time_info['final_start'] < overall_start:
                    overall_start = time_info['final_start']
                if overall_end is None or time_info['final_end'] > overall_end:
                    overall_end = time_info['final_end']
        
        load_time = time.time() - start_time
        logger.info(f"Data loading completed in {load_time:.1f} seconds")
        logger.info(f"Successfully loaded {len(self.price_data)} symbols")
        
        # 打印时间范围信息
        if overall_start and overall_end:
            logger.info("\n" + "="*60)
            logger.info("DATA TIME RANGE SUMMARY")
            logger.info("="*60)
            logger.info(f"Overall data range: {overall_start.strftime('%Y-%m-%d %H:%M')} to {overall_end.strftime('%Y-%m-%d %H:%M')}")
            logger.info(f"Total trading days: {(overall_end - overall_start).days}")
            
            # 打印每个品种的时间范围
            logger.info("\nPer-symbol time ranges:")
            for symbol in sorted(self.time_info.keys()):
                info = self.time_info[symbol]
                logger.info(f"  {symbol}: {info['final_start'].strftime('%Y-%m-%d')} to "
                           f"{info['final_end'].strftime('%Y-%m-%d')} "
                           f"({info['bars_count']} bars)")
            logger.info("="*60)
        
        return len(self.price_data) > 0
    
    def generate_features_parallel(self):
        """并行生成特征"""
        logger.info("="*60)
        logger.info(f"Generating features for {len(self.price_data)} symbols...")
        
        start_time = time.time()
        
        # 准备参数
        args_list = list(self.price_data.items())
        
        # 使用进程池并行计算
        with Pool(processes=self.n_workers) as pool:
            results = list(tqdm(
                pool.imap_unordered(calculate_features_for_symbol, args_list),
                total=len(args_list),
                desc="Computing features"
            ))
        
        # 整理结果
        for symbol, features_df, feature_cols in results:
            if features_df is not None and feature_cols is not None and len(features_df) > 100:
                self.features_data[symbol] = (features_df, feature_cols)
        
        feature_time = time.time() - start_time
        logger.info(f"Feature generation completed in {feature_time:.1f} seconds")
        logger.info(f"Successfully processed features for {len(self.features_data)} symbols")
        
        return len(self.features_data) > 0
    
    def generate_labels_parallel(self, horizon: int = 5, threshold: float = 0.002):
        """
        并行生成标签
        
        Parameters:
        -----------
        horizon : int
            预测周期（K线数）
        threshold : float
            交易阈值（最小预期收益率）
        """
        logger.info("="*60)
        logger.info(f"Generating labels (horizon={horizon}, threshold={threshold:.4f})...")
        
        start_time = time.time()
        
        # 准备参数 - 使用features_data确保数据一致性
        args_list = [(symbol, df, horizon, threshold) 
                     for symbol, (df, _) in self.features_data.items()]
        
        # 使用进程池并行计算
        with Pool(processes=self.n_workers) as pool:
            results = list(tqdm(
                pool.imap_unordered(generate_labels_for_symbol, args_list),
                total=len(args_list),
                desc="Generating labels"
            ))
        
        # 整理结果
        for symbol, labels in results:
            if labels is not None and len(labels) > 100:
                self.labels_data[symbol] = labels
        
        label_time = time.time() - start_time
        logger.info(f"Label generation completed in {label_time:.1f} seconds")
        logger.info(f"Successfully generated labels for {len(self.labels_data)} symbols")
        
        return len(self.labels_data) > 0
    
    def prepare_training_data(self, train_ratio: float = 0.7):
        """
        准备训练和测试数据
        
        Parameters:
        -----------
        train_ratio : float
            训练集比例
        """
        logger.info("="*60)
        logger.info("Preparing training data...")
        
        all_X_train = []
        all_y_train = []
        all_X_test = []
        all_y_test = []
        
        # 记录训练和测试的时间范围
        train_start = None
        train_end = None
        test_start = None
        test_end = None
        
        for symbol in self.features_data:
            if symbol not in self.labels_data:
                continue
            
            features_df, feature_cols = self.features_data[symbol]
            labels = self.labels_data[symbol]
            
            # 数据对齐
            common_index = features_df.index.intersection(labels.index)
            
            if len(common_index) < 200:
                continue
            
            # 提取特征和标签
            X = features_df.loc[common_index, feature_cols].values
            y = labels.loc[common_index].values
            
            # 划分训练集和测试集
            split_idx = int(len(X) * train_ratio)
            
            # 记录时间分割点
            split_date = common_index[split_idx]
            symbol_train_start = common_index[0]
            symbol_train_end = common_index[split_idx-1]
            symbol_test_start = common_index[split_idx]
            symbol_test_end = common_index[-1]
            
            # 更新整体时间范围
            if train_start is None or symbol_train_start < train_start:
                train_start = symbol_train_start
            if train_end is None or symbol_train_end > train_end:
                train_end = symbol_train_end
            if test_start is None or symbol_test_start < test_start:
                test_start = symbol_test_start
            if test_end is None or symbol_test_end > test_end:
                test_end = symbol_test_end
            
            X_train = X[:split_idx]
            y_train = y[:split_idx]
            X_test = X[split_idx:]
            y_test = y[split_idx:]
            
            # 过滤有效样本（去除NaN）
            valid_train = ~(np.isnan(X_train).any(axis=1) | np.isnan(y_train))
            valid_test = ~(np.isnan(X_test).any(axis=1) | np.isnan(y_test))
            
            if valid_train.sum() > 50 and valid_test.sum() > 50:
                all_X_train.append(X_train[valid_train])
                all_y_train.append(y_train[valid_train])
                all_X_test.append(X_test[valid_test])
                all_y_test.append(y_test[valid_test])
                
                logger.info(f"{symbol}: Train={valid_train.sum()}, Test={valid_test.sum()}, "
                           f"Split at {split_date.strftime('%Y-%m-%d')}")
        
        # 打印训练测试时间范围
        if train_start and test_end:
            logger.info("\n" + "="*60)
            logger.info("TRAINING/TESTING TIME SPLIT")
            logger.info("="*60)
            logger.info(f"Training period: {train_start.strftime('%Y-%m-%d %H:%M')} to {train_end.strftime('%Y-%m-%d %H:%M')}")
            logger.info(f"Training days: {(train_end - train_start).days}")
            logger.info(f"Testing period: {test_start.strftime('%Y-%m-%d %H:%M')} to {test_end.strftime('%Y-%m-%d %H:%M')}")
            logger.info(f"Testing days: {(test_end - test_start).days}")
            logger.info(f"Split ratio: {train_ratio:.1%} train / {(1-train_ratio):.1%} test")
            logger.info("="*60)
            
            # 保存时间信息
            self.train_period = (train_start, train_end)
            self.test_period = (test_start, test_end)
        
        if all_X_train:
            X_train_combined = np.vstack(all_X_train)
            y_train_combined = np.hstack(all_y_train)
            X_test_combined = np.vstack(all_X_test)
            y_test_combined = np.hstack(all_y_test)
            
            logger.info(f"Total samples - Train: {len(X_train_combined)}, Test: {len(X_test_combined)}")
            
            # 保存特征列名
            _, feature_cols = list(self.features_data.values())[0]
            self.feature_columns = feature_cols
            
            return X_train_combined, y_train_combined, X_test_combined, y_test_combined
        
        return None, None, None, None
    
    def train_model(self, model_type: str = 'rf'):
        """
        训练模型
        
        Parameters:
        -----------
        model_type : str
            模型类型 ('rf'=随机森林, 'xgb'=XGBoost)
        """
        logger.info("="*60)
        logger.info(f"Training {model_type.upper()} model...")
        
        start_time = time.time()
        
        # 准备数据
        X_train, y_train, X_test, y_test = self.prepare_training_data()
        
        if X_train is None or len(X_train) < 1000:
            logger.error("Insufficient training data")
            return False
        
        # 选择模型
        if model_type == 'rf':
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=50,
                min_samples_leaf=20,
                n_jobs=self.n_workers,
                random_state=42,
                class_weight='balanced'
            )
        elif model_type == 'xgb':
            try:
                import xgboost as xgb
                self.model = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    n_jobs=self.n_workers,
                    random_state=42,
                    scale_pos_weight=1
                )
            except ImportError:
                logger.warning("XGBoost not installed, using RandomForest instead")
                return self.train_model('rf')
        
        # 训练模型
        self.model.fit(X_train, y_train)
        
        # 评估模型
        from sklearn.metrics import accuracy_score, classification_report
        
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        
        logger.info(f"Training Accuracy: {train_acc:.4f}")
        logger.info(f"Testing Accuracy: {test_acc:.4f}")
        
        # 输出分类报告
        logger.info("\nClassification Report (Test Set):")
        logger.info(classification_report(y_test, y_pred_test, 
                                         target_names=['Short', 'Neutral', 'Long'],
                                         digits=4))
        
        train_time = time.time() - start_time
        logger.info(f"Model training completed in {train_time:.1f} seconds")
        
        return True
    
    def generate_positions(self, confidence_threshold: float = 0.4):
        """
        生成交易仓位
        
        Parameters:
        -----------
        confidence_threshold : float
            置信度阈值
        """
        logger.info("="*60)
        logger.info(f"Generating positions (confidence threshold={confidence_threshold:.2f})...")
        
        # 记录实际交易的时间范围
        position_start = None
        position_end = None
        
        for symbol in self.features_data:
            try:
                features_df, feature_cols = self.features_data[symbol]
                
                # 准备特征数据
                X = features_df[self.feature_columns].fillna(0).values
                
                # 预测
                predictions = self.model.predict(X)
                
                # 获取预测概率
                if hasattr(self.model, 'predict_proba'):
                    proba = self.model.predict_proba(X)
                    # 计算置信度（最大概率）
                    confidence = np.max(proba, axis=1)
                else:
                    confidence = np.ones(len(predictions))
                
                # 生成仓位
                positions = pd.Series(predictions, index=features_df.index)
                confidence_series = pd.Series(confidence, index=features_df.index)
                
                # 应用置信度阈值
                positions[confidence_series < confidence_threshold] = 0
                
                # 避免未来函数：延迟信号（重要！确保T时刻的信号在T+1时刻执行）
                positions = positions.shift(1).fillna(0)
                
                self.positions[symbol] = positions
                self.predictions[symbol] = pd.DataFrame({
                    'position': positions,
                    'confidence': confidence_series
                })
                
                # 更新仓位时间范围
                valid_positions = positions[positions != 0]
                if len(valid_positions) > 0:
                    symbol_pos_start = valid_positions.index[0]
                    symbol_pos_end = valid_positions.index[-1]
                    
                    if position_start is None or symbol_pos_start < position_start:
                        position_start = symbol_pos_start
                    if position_end is None or symbol_pos_end > position_end:
                        position_end = symbol_pos_end
                
                # 统计
                long_pct = (positions == 1).mean()
                short_pct = (positions == -1).mean()
                neutral_pct = (positions == 0).mean()
                
                logger.info(f"{symbol}: Long={long_pct:.2%}, Short={short_pct:.2%}, Neutral={neutral_pct:.2%}")
                
            except Exception as e:
                logger.error(f"Error generating positions for {symbol}: {e}")
                self.positions[symbol] = pd.Series(0, index=features_df.index)
        
        # 打印实际交易时间范围
        if position_start and position_end:
            logger.info("\n" + "="*60)
            logger.info("ACTUAL TRADING TIME RANGE")
            logger.info("="*60)
            logger.info(f"First trade signal: {position_start.strftime('%Y-%m-%d %H:%M')}")
            logger.info(f"Last trade signal: {position_end.strftime('%Y-%m-%d %H:%M')}")
            logger.info(f"Trading period length: {(position_end - position_start).days} days")
            logger.info("="*60)
            
            self.trading_period = (position_start, position_end)
        
        return len(self.positions) > 0
    
    def calculate_performance(self):
        """计算策略性能 - 修正版，包含真实的交易成本"""
        logger.info("="*60)
        logger.info("Calculating strategy performance...")
        
        all_returns = []
        symbol_metrics = {}
        
        # 记录回测计算的时间范围
        backtest_start = None
        backtest_end = None
        
        for symbol, positions in self.positions.items():
            if symbol not in self.features_data:
                continue
            
            features_df, _ = self.features_data[symbol]
            
            # 确保数据对齐
            common_index = positions.index.intersection(features_df.index)
            
            if len(common_index) < 100:
                continue
            
            # 更新回测时间范围
            symbol_bt_start = common_index[0]
            symbol_bt_end = common_index[-1]
            
            if backtest_start is None or symbol_bt_start < backtest_start:
                backtest_start = symbol_bt_start
            if backtest_end is None or symbol_bt_end > backtest_end:
                backtest_end = symbol_bt_end
            
            # 计算收益
            positions_aligned = positions.loc[common_index]
            prices = features_df.loc[common_index, 'close']
            
            # 计算价格收益率
            price_returns = prices.pct_change().fillna(0)
            
            # 策略收益 = 仓位 * 价格收益
            # 注意：仓位已经在generate_positions中shift(1)了，所以这里直接乘
            strategy_returns = positions_aligned * price_returns
            
            # 计算交易成本
            position_changes = positions_aligned.diff().fillna(positions_aligned.iloc[0])
            
            # 使用真实的交易成本模型
            if self.use_realistic_costs:
                trading_costs = self.cost_model.calculate_trading_costs(symbol, position_changes)
            else:
                # 使用简单的固定成本
                total_cost_rate = self.simple_transaction_cost + self.simple_slippage
                trading_costs = position_changes.abs() * total_cost_rate
            
            # 净收益 = 策略收益 - 交易成本
            net_returns = strategy_returns - trading_costs
            
            # 保存品种收益
            all_returns.append(net_returns)
            
            # 计算品种指标
            cumulative_return = (1 + net_returns).prod() - 1
            num_trades = (position_changes != 0).sum()
            
            # 计算平均每笔交易收益
            if num_trades > 0:
                trade_returns = []
                current_position = 0
                entry_price = 0
                
                for idx, (pos, price) in enumerate(zip(positions_aligned.values, prices.values)):
                    if current_position == 0 and pos != 0:
                        # 开仓
                        current_position = pos
                        entry_price = price
                    elif current_position != 0 and pos == 0:
                        # 平仓
                        exit_price = price
                        if current_position == 1:
                            trade_return = exit_price / entry_price - 1
                        else:
                            trade_return = entry_price / exit_price - 1
                        trade_returns.append(trade_return)
                        current_position = 0
                    elif current_position != 0 and pos == -current_position:
                        # 反向开仓
                        exit_price = price
                        if current_position == 1:
                            trade_return = exit_price / entry_price - 1
                        else:
                            trade_return = entry_price / exit_price - 1
                        trade_returns.append(trade_return)
                        current_position = pos
                        entry_price = price
                
                avg_trade_return = np.mean(trade_returns) if trade_returns else 0
            else:
                avg_trade_return = 0
            
            symbol_metrics[symbol] = {
                'total_return': cumulative_return,
                'num_trades': num_trades,
                'mean_return': net_returns.mean(),
                'std_return': net_returns.std(),
                'avg_trade_return': avg_trade_return,
                'cost_rate': self.cost_model.get_total_cost_rate(symbol) if self.use_realistic_costs else (self.simple_transaction_cost + self.simple_slippage)
            }
            
            logger.info(f"{symbol}: Return={cumulative_return:.2%}, Trades={num_trades}, "
                       f"Cost Rate={symbol_metrics[symbol]['cost_rate']:.4f}")
        
        # 打印回测计算时间范围
        if backtest_start and backtest_end:
            logger.info("\n" + "="*60)
            logger.info("BACKTEST CALCULATION TIME RANGE")
            logger.info("="*60)
            logger.info(f"Backtest start: {backtest_start.strftime('%Y-%m-%d %H:%M')}")
            logger.info(f"Backtest end: {backtest_end.strftime('%Y-%m-%d %H:%M')}")
            logger.info(f"Backtest period: {(backtest_end - backtest_start).days} days")
            logger.info("="*60)
            
            self.backtest_period = (backtest_start, backtest_end)
        
        if all_returns:
            # 等权重组合收益
            portfolio_returns = pd.concat(all_returns, axis=1).mean(axis=1)
            
            # 计算组合性能指标
            cumulative = (1 + portfolio_returns).cumprod()
            total_return = cumulative.iloc[-1] - 1
            
            # 年化收益（假设每天16个15分钟K线，一年250个交易日）
            periods_per_year = 16 * 250
            num_periods = len(portfolio_returns)
            years = num_periods / periods_per_year
            
            if years > 0:
                annual_return = (1 + total_return) ** (1 / years) - 1
            else:
                annual_return = 0
            
            # 年化波动率
            volatility = portfolio_returns.std() * np.sqrt(periods_per_year)
            
            # 夏普比率
            risk_free_rate = 0.03
            sharpe_ratio = (annual_return - risk_free_rate) / volatility if volatility > 0 else 0
            
            # 最大回撤
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # 胜率
            winning_periods = (portfolio_returns > 0).sum()
            total_periods = len(portfolio_returns)
            win_rate = winning_periods / total_periods if total_periods > 0 else 0
            
            # 盈亏比
            winning_returns = portfolio_returns[portfolio_returns > 0]
            losing_returns = portfolio_returns[portfolio_returns < 0]
            
            avg_win = winning_returns.mean() if len(winning_returns) > 0 else 0
            avg_loss = abs(losing_returns.mean()) if len(losing_returns) > 0 else 0
            profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0
            
            # 卡尔玛比率
            calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            return {
                'total_return': total_return,
                'annual_return': annual_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'profit_loss_ratio': profit_loss_ratio,
                'calmar_ratio': calmar_ratio,
                'total_trades': sum(m['num_trades'] for m in symbol_metrics.values()),
                'avg_trades_per_symbol': np.mean([m['num_trades'] for m in symbol_metrics.values()]),
                'symbol_metrics': symbol_metrics,
                'backtest_period': self.backtest_period if hasattr(self, 'backtest_period') else None
            }
        
        return None
    
    def print_time_summary(self):
        """打印完整的时间范围总结"""
        logger.info("\n" + "="*60)
        logger.info("COMPLETE TIME RANGE SUMMARY")
        logger.info("="*60)
        
        if hasattr(self, 'train_period'):
            logger.info(f"Training Period: {self.train_period[0].strftime('%Y-%m-%d')} to "
                       f"{self.train_period[1].strftime('%Y-%m-%d')} "
                       f"({(self.train_period[1] - self.train_period[0]).days} days)")
        
        if hasattr(self, 'test_period'):
            logger.info(f"Testing Period: {self.test_period[0].strftime('%Y-%m-%d')} to "
                       f"{self.test_period[1].strftime('%Y-%m-%d')} "
                       f"({(self.test_period[1] - self.test_period[0]).days} days)")
        
        if hasattr(self, 'trading_period'):
            logger.info(f"Trading Period: {self.trading_period[0].strftime('%Y-%m-%d')} to "
                       f"{self.trading_period[1].strftime('%Y-%m-%d')} "
                       f"({(self.trading_period[1] - self.trading_period[0]).days} days)")
        
        if hasattr(self, 'backtest_period'):
            logger.info(f"Backtest Period: {self.backtest_period[0].strftime('%Y-%m-%d')} to "
                       f"{self.backtest_period[1].strftime('%Y-%m-%d')} "
                       f"({(self.backtest_period[1] - self.backtest_period[0]).days} days)")
        
        logger.info("="*60)
    
    def run_complete_strategy(self, 
                             symbols: List[str],
                             sample_rate: float = 1.0,
                             model_type: str = 'rf',
                             confidence_threshold: float = 0.4,
                             prediction_horizon: int = 5,
                             trade_threshold: float = 0.002):
        """
        运行完整策略流程
        
        Parameters:
        -----------
        symbols : List[str]
            交易品种列表
        sample_rate : float
            数据采样率
        model_type : str
            模型类型
        confidence_threshold : float
            置信度阈值
        prediction_horizon : int
            预测周期
        trade_threshold : float
            交易阈值
        """
        logger.info("\n" + "="*60)
        logger.info("IMPROVED PARALLEL TRADING STRATEGY V2")
        logger.info(f"Using {self.n_workers} parallel workers")
        logger.info(f"Model: {model_type.upper()}")
        logger.info(f"Confidence threshold: {confidence_threshold:.2f}")
        logger.info(f"Prediction horizon: {prediction_horizon} bars")
        logger.info(f"Trade threshold: {trade_threshold:.4f}")
        logger.info(f"Cost Model: {'Realistic' if self.use_realistic_costs else 'Simple Fixed'}")
        logger.info("="*60 + "\n")
        
        total_start = time.time()
        
        # 1. 并行加载数据
        if not self.load_data_parallel(symbols, sample_rate):
            logger.error("Data loading failed")
            return None
        
        # 2. 并行生成特征
        if not self.generate_features_parallel():
            logger.error("Feature generation failed")
            return None
        
        # 3. 并行生成标签
        if not self.generate_labels_parallel(
            horizon=prediction_horizon,
            threshold=trade_threshold
        ):
            logger.error("Label generation failed")
            return None
        
        # 4. 训练模型
        if not self.train_model(model_type):
            logger.error("Model training failed")
            return None
        
        # 5. 生成仓位
        if not self.generate_positions(confidence_threshold):
            logger.error("Position generation failed")
            return None
        
        # 6. 计算性能
        results = self.calculate_performance()
        
        total_time = time.time() - total_start
        
        # 打印结果
        if results:
            logger.info("\n" + "="*60)
            logger.info("STRATEGY RESULTS")
            logger.info("="*60)
            
            # 主要指标
            logger.info(f"Total Return: {results['total_return']:.2%}")
            logger.info(f"Annual Return: {results['annual_return']:.2%}")
            logger.info(f"Volatility: {results['volatility']:.2%}")
            logger.info(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
            logger.info(f"Max Drawdown: {results['max_drawdown']:.2%}")
            logger.info(f"Win Rate: {results['win_rate']:.2%}")
            logger.info(f"Profit/Loss Ratio: {results['profit_loss_ratio']:.2f}")
            logger.info(f"Calmar Ratio: {results['calmar_ratio']:.2f}")
            logger.info(f"Total Trades: {results['total_trades']:,}")
            logger.info(f"Avg Trades per Symbol: {results['avg_trades_per_symbol']:.0f}")
            
            # 打印交易成本细节
            logger.info("\nTrading Cost Details:")
            for symbol, metrics in results['symbol_metrics'].items():
                logger.info(f"  {symbol}: Total Cost Rate = {metrics['cost_rate']:.4f} ({metrics['cost_rate']*10000:.2f} basis points)")
            
            # 打印完整的时间范围总结
            self.print_time_summary()
            
            logger.info(f"\nTotal Runtime: {total_time:.1f} seconds")
            logger.info("="*60)
        
        return results


# ================== 主函数 ==================
if __name__ == "__main__":
    # 配置
    config = {
        'data_path': '~/autodl-tmp/data/1m/',
        'symbols': ['IF', 'IC', 'IH', 'TS', 'TF', 'T', 'CU', 'AL', 'ZN', 'RB'],
        'sample_rate': 0.8,
        'n_workers': None,
        'model_type': 'rf',
        'confidence_threshold': 0.5,
        'prediction_horizon': 5,
        'trade_threshold': 0.003,
        'use_realistic_costs': True  # 使用真实的交易成本模型
    }
    
    # 打印系统信息
    logger.info(f"System CPUs: {cpu_count()}")
    logger.info(f"Data path: {config['data_path']}")
    logger.info(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 创建策略实例
    strategy = ImprovedParallelTradingStrategy(
        data_path=config['data_path'],
        n_workers=config['n_workers'],
        use_realistic_costs=config['use_realistic_costs']
    )
    
    # 运行策略
    results = strategy.run_complete_strategy(
        symbols=config['symbols'],
        sample_rate=config['sample_rate'],
        model_type=config['model_type'],
        confidence_threshold=config['confidence_threshold'],
        prediction_horizon=config['prediction_horizon'],
        trade_threshold=config['trade_threshold']
    )
    
    # 清理资源
    gc.collect()
    
    if results:
        logger.info("\nStrategy execution completed successfully!")
        
        # 如果收益为正，输出更详细的分析
        if results['total_return'] > 0:
            logger.info("\n" + "="*60)
            logger.info("PROFITABLE STRATEGY DETECTED!")
            logger.info("="*60)
            logger.info("\nTop Performing Symbols:")
            
            # 按收益排序品种
            symbol_metrics = results['symbol_metrics']
            sorted_symbols = sorted(
                symbol_metrics.items(),
                key=lambda x: x[1]['total_return'],
                reverse=True
            )
            
            for i, (symbol, metrics) in enumerate(sorted_symbols[:5], 1):
                logger.info(f"{i}. {symbol}: Return={metrics['total_return']:.2%}, "
                          f"Trades={metrics['num_trades']}, "
                          f"Avg Trade Return={metrics['avg_trade_return']:.3%}")
            
            # 打印失败的品种
            logger.info("\nWorst Performing Symbols:")
            for i, (symbol, metrics) in enumerate(sorted_symbols[-3:], 1):
                if metrics['total_return'] < 0:
                    logger.info(f"{i}. {symbol}: Return={metrics['total_return']:.2%}, "
                              f"Trades={metrics['num_trades']}")
    else:
        logger.error("\nStrategy execution failed!")