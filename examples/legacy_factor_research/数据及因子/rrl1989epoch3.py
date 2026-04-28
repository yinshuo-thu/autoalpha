#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
商品期货量化策略系统 - 完全无未来函数版本 + 扩展因子库
整合autodl-tmp/factors/factors.py中的所有因子
所有潜在的未来函数已被识别并修复
"""
# 在文件开头的导入部分添加
from torch import nn
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
import pandas as pd
import numpy as np
import os
import logging
from typing import List, Dict, Tuple, Optional
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import pyarrow.parquet as pq
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
class ImprovedRRLArchitecture(nn.Module):
    def __init__(self, feature_num):
        super().__init__()
        
        # 修复：确保embed_dim能被num_heads整除
        # 将特征数量调整为8的倍数
        num_heads = 8
        embed_dim = ((feature_num + num_heads - 1) // num_heads) * num_heads
        
        # 先将输入投影到合适的维度
        self.input_projection = nn.Linear(feature_num, embed_dim)
        
        # 添加注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.1
        )
        
        # 使用残差连接
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(embed_dim) for _ in range(3)
        ])
        
        # 输出层使用更复杂的结构
        self.output_network = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )
        
        self.feature_num = feature_num
        self.embed_dim = embed_dim
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.activation = nn.GELU()
        
    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.activation(self.fc1(x))
        x = self.norm2(x)
        x = self.fc2(x)
        return x + residual

def safe_rolling(series, window, min_periods=None, **kwargs):
    """
    确保不使用未来数据的rolling操作
    
    Args:
        series: pandas Series或DataFrame
        window: 窗口大小
        min_periods: 最小观测值数量
    """
    if min_periods is None:
        min_periods = max(1, window // 2)
    
    # 强制center=False，移除任何可能的center参数
    kwargs.pop('center', None)
    
    return series.rolling(window=window, min_periods=min_periods, center=False, **kwargs)
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
商品期货量化策略系统 - 修复版本
完全无未来函数，提高收益率
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import mutual_info_regression
import logging
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基于RRL思想的神经网络仓位生成器
整合到现有的商品期货量化策略系统中
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from collections import deque

logger = logging.getLogger(__name__)
class ImprovedRRLModel(ImprovedRRLArchitecture):
    def __init__(self, feature_num: int, hidden_sizes: List[int] = [128, 64, 32]):
        super().__init__(feature_num)
        
        # 使用继承的embed_dim
        embed_dim = self.embed_dim
        
        # Factor Interaction Branch
        layers = []
        input_size = embed_dim  # 使用embed_dim而不是feature_num
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(0.1))
            input_size = hidden_size
        self.factor_branch = nn.Sequential(*layers[:-1])
        
        # Position-Signal Interaction Branch
        self.position_branch = nn.Sequential(
            nn.Linear(hidden_sizes[-1] + 1, hidden_sizes[-1]),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_sizes[-1], 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        # Final position adjustment branch
        self.position_head = nn.Sequential(
            nn.Linear(hidden_sizes[-1], 1),
            nn.Tanh()
        )
        
        self.init_weights()
        
    def forward(self, features: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 确保输入形状正确
        features = features.reshape(-1, self.feature_num)
        pos = pos.reshape(-1, 1)
        
        # 先投影到embed_dim
        features_projected = self.input_projection(features)
        
        # Factor Interaction Branch
        factor_output = self.factor_branch(features_projected)
        
        # Position-Signal Interaction Branch
        position_input = torch.cat([factor_output, pos], dim=1)
        adjustment_ratio = self.position_branch(position_input)
        
        # Calculate target position
        target_position = self.position_head(factor_output)
        
        # Calculate trades with adjustment
        trades = adjustment_ratio * (target_position - pos)
        
        return trades



# ================== 基于Transformer的因子分支（可选升级）==================
class TransformerFactorBranch(nn.Module):
    """使用Transformer替代MLP的因子分支"""
    
    def __init__(self, feature_num: int, d_model: int = 64, nhead: int = 4, 
                 num_layers: int = 2):
        super(TransformerFactorBranch, self).__init__()
        
        # 输入投影
        self.input_projection = nn.Linear(feature_num, d_model)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出投影
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.Tanh()
        )
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # [batch_size, feature_num] -> [batch_size, 1, feature_num]
        features = features.unsqueeze(1)
        
        # 投影到d_model维度
        x = self.input_projection(features)
        
        # Transformer处理 (需要seq_len在第一维)
        x = x.transpose(0, 1)
        x = self.transformer(x)
        x = x.transpose(0, 1)
        
        # 输出投影
        x = x.squeeze(1)
        output = self.output_projection(x)
        
        return output


# ================== RRL训练器 ==================
class RRLTrainer:
    """RRL模型训练器"""
    
    def __init__(self, model: nn.Module, learning_rate: float = 1e-4,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', patience=10, factor=0.5
        )
        
        # 损失历史
        self.loss_history = []
        self.sharpe_history = []
        
    def compute_returns(self, positions: torch.Tensor, prices: torch.Tensor,
                        commission: float = 0.0002) -> torch.Tensor:
        """
        计算收益
        
        Args:
            positions: 仓位序列 [batch_size, seq_len]
            prices: 价格序列 [batch_size, seq_len]
            commission: 手续费率
        
        Returns:
            returns: 收益序列
        """
        # 计算价格变化率
        price_returns = (prices[:, 1:] - prices[:, :-1]) / prices[:, :-1]
        
        # 计算仓位变化
        position_changes = torch.abs(positions[:, 1:] - positions[:, :-1])
        
        # 计算交易成本
        transaction_costs = commission * position_changes
        
        # 计算净收益
        returns = positions[:, :-1] * price_returns - transaction_costs
        
        return returns
    
    def sharpe_ratio_loss(self, returns: torch.Tensor) -> torch.Tensor:
        """
        计算负夏普比率作为损失
        
        Args:
            returns: 收益序列
        
        Returns:
            loss: 负夏普比率
        """
        # 计算平均收益
        mean_return = returns.mean()
        
        # 计算标准差
        std_return = returns.std() + 1e-8
        
        # 计算夏普比率（年化，假设每天48个30分钟bar）
        sharpe = mean_return / std_return * np.sqrt(252 * 48)
        
        # 返回负夏普比率作为损失
        return -sharpe
    
    def train_batch(self, features: np.ndarray, prices: np.ndarray,
                   initial_positions: np.ndarray = None) -> float:
        """
        训练一个批次
        
        Args:
            features: 特征数据 [batch_size, seq_len, feature_num]
            prices: 价格数据 [batch_size, seq_len]
            initial_positions: 初始仓位
        
        Returns:
            loss: 损失值
        """
        batch_size, seq_len, feature_num = features.shape
        
        # 转换为张量
        features_tensor = torch.FloatTensor(features).to(self.device)
        prices_tensor = torch.FloatTensor(prices).to(self.device)
        
        # 初始化仓位
        if initial_positions is None:
            positions = torch.zeros(batch_size, 1).to(self.device)
        else:
            positions = torch.FloatTensor(initial_positions).to(self.device)
        
        all_positions = [positions]
        
        # 循环生成仓位
        for t in range(seq_len):
            # 获取当前特征
            current_features = features_tensor[:, t, :]
            
            # 生成交易信号
            trades = self.model(current_features, positions)
            
            # 更新仓位
            positions = positions + trades
            positions = torch.clamp(positions, -1, 1)  # 限制仓位范围
            
            all_positions.append(positions)
        
        # 合并所有仓位
        all_positions = torch.cat(all_positions[:-1], dim=1).reshape(batch_size, seq_len)
        
        # 计算收益
        returns = self.compute_returns(all_positions, prices_tensor)
        
        # 计算损失
        loss = self.sharpe_ratio_loss(returns)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return loss.item()
    
    def validate(self, features: np.ndarray, prices: np.ndarray) -> Dict:
        """验证模型性能"""
        self.model.eval()
        
        with torch.no_grad():
            # 生成仓位
            positions = self.generate_positions(features)
            
            # 计算收益
            prices_tensor = torch.FloatTensor(prices).to(self.device)
            positions_tensor = torch.FloatTensor(positions).to(self.device)
            
            returns = self.compute_returns(
                positions_tensor.unsqueeze(0), 
                prices_tensor.unsqueeze(0)
            ).squeeze(0)
            
            # 计算指标
            total_return = returns.sum().item()
            sharpe = -self.sharpe_ratio_loss(returns).item()
            
        self.model.train()
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'avg_position': positions.mean(),
            'position_changes': np.diff(positions).mean()
        }
    
    def generate_positions(self, features: np.ndarray) -> np.ndarray:
        """生成仓位序列"""
        self.model.eval()
        
        seq_len, feature_num = features.shape
        positions = []
        current_pos = torch.zeros(1, 1).to(self.device)
        
        with torch.no_grad():
            for t in range(seq_len):
                # 获取当前特征
                current_features = torch.FloatTensor(features[t:t+1]).to(self.device)
                
                # 生成交易信号
                trades = self.model(current_features, current_pos)
                
                # 更新仓位
                current_pos = current_pos + trades
                current_pos = torch.clamp(current_pos, -1, 1)
                
                positions.append(current_pos.cpu().numpy()[0, 0])
        
        self.model.train()
        
        return np.array(positions)


# ================== 集成到现有策略 ==================
class NeuralNetworkPositionGenerator:
    """神经网络仓位生成器，集成到现有策略框架"""
    def __init__(self, feature_num: int, use_transformer: bool = False):
        """
        Args:
            feature_num: 特征数量
            use_transformer: 是否使用Transformer作为因子分支
        """
        self.feature_num = feature_num
        self.use_transformer = use_transformer

        # 创建模型
        if use_transformer:
            self.model = self._create_transformer_model(feature_num)
        else:
            self.model = ImprovedRRLModel(feature_num)

        # 创建训练器
        self.trainer = RRLTrainer(self.model)

        # 特征缓存
        self.feature_scaler = None
        self.is_trained = False
        self.common_features = None  # 添加这行来存储共同特征
        
    def _create_transformer_model(self, feature_num: int) -> nn.Module:
        """创建带Transformer因子分支的模型"""
        
        class HybridRRLModel(nn.Module):
            def __init__(self, feature_num):
                super().__init__()
                self.feature_num = feature_num
                self.factor_branch = TransformerFactorBranch(feature_num)
                
                # Position-Signal Interaction Branch
                self.position_branch = nn.Sequential(
                    nn.Linear(32 + 1, 32),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(32, 1),
                    nn.Sigmoid()
                )
                
                self.position_head = nn.Sequential(
                    nn.Linear(32, 1),
                    nn.Tanh()
                )
            
            def forward(self, features, pos):
                features = features.reshape(-1, self.feature_num)
                pos = pos.reshape(-1, 1)
                
                factor_output = self.factor_branch(features)
                position_input = torch.cat([factor_output, pos], dim=1)
                adjustment_ratio = self.position_branch(position_input)
                target_position = self.position_head(factor_output)
                trades = adjustment_ratio * (target_position - pos)
                
                return trades
        
        return HybridRRLModel(feature_num)
    
    def prepare_training_data(self, features_dict: Dict, price_dict: Dict,
                            sequence_length: int = 20) -> Tuple:
        """准备训练数据"""
        all_features = []
        all_prices = []

        # 首先确定共同的特征列
        common_features = None
        for symbol in features_dict:
            features_df, feature_cols = features_dict[symbol]
            if common_features is None:
                common_features = set(feature_cols)
            else:
                common_features = common_features.intersection(set(feature_cols))

        if not common_features:
            logger.error("No common features found across symbols")
            return None, None

        # 转换为列表并排序以保持一致性
        common_features = sorted(list(common_features))
        
        # ★★★ 添加这一行 - 保存共同特征列表 ★★★
        self.common_features = common_features
        
        logger.info(f"Using {len(common_features)} common features for RRL training")

        for symbol in features_dict:
            if symbol not in price_dict:
                continue

            features_df, _ = features_dict[symbol]
            prices_df = price_dict[symbol]

            # 对齐数据
            common_index = features_df.index.intersection(prices_df.index)

            if len(common_index) < sequence_length * 2:
                continue

            # 只使用共同的特征列
            X = features_df.loc[common_index, common_features].values
            y = prices_df.loc[common_index, 'close'].values

            # 创建序列
            for i in range(len(X) - sequence_length):
                all_features.append(X[i:i+sequence_length])
                all_prices.append(y[i:i+sequence_length])

        if not all_features:
            return None, None

        # 现在所有序列应该有相同的形状
        features_array = np.array(all_features)
        prices_array = np.array(all_prices)

        # 更新特征数量
        self.feature_num = len(common_features)

        # 重新创建模型以匹配新的特征数量
        if self.use_transformer:
            self.model = self._create_transformer_model(self.feature_num)
        else:
            self.model = ImprovedRRLModel(self.feature_num)

        # 更新训练器
        self.trainer = RRLTrainer(self.model)

        return features_array, prices_array
    
    def train(self, features_dict: Dict, price_dict: Dict,
             epochs: int = 3, batch_size: int = 32):
        """训练神经网络"""
        # 准备数据
        X, y = self.prepare_training_data(features_dict, price_dict)

        if X is None:
            logger.error("No training data available")
            return False

        logger.info(f"Training RRL model with {len(X)} sequences")

        # 数据归一化
        from sklearn.preprocessing import StandardScaler
        self.feature_scaler = StandardScaler()

        # Reshape for scaling
        n_samples, seq_len, n_features = X.shape
        X_reshaped = X.reshape(-1, n_features)
        X_scaled = self.feature_scaler.fit_transform(X_reshaped)
        X = X_scaled.reshape(n_samples, seq_len, n_features)

        # 训练循环
        best_sharpe = -np.inf
        patience = 5  # 降低 patience
        patience_counter = 0

        for epoch in range(epochs):
            # 随机打乱数据
            indices = np.random.permutation(len(X))

            epoch_losses = []

            # 批次训练
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_X = X[batch_indices]
                batch_y = y[batch_indices]

                loss = self.trainer.train_batch(batch_X, batch_y)
                epoch_losses.append(loss)

            # 更频繁的验证和日志输出（每个epoch都执行）
            if len(X) > 0:  # 改变条件，每个epoch都验证
                val_idx = np.random.choice(len(X), min(10, len(X)), replace=False)
                val_metrics = self.trainer.validate(X[val_idx[0]], y[val_idx[0]])

                avg_loss = np.mean(epoch_losses)

                # 每个epoch都打印进度
                logger.info(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, "
                          f"Sharpe={val_metrics['sharpe_ratio']:.4f}, "
                          f"Return={val_metrics['total_return']:.4f}")

                # Early stopping
                if val_metrics['sharpe_ratio'] > best_sharpe:
                    best_sharpe = val_metrics['sharpe_ratio']
                    patience_counter = 0
                    # 保存最佳模型
                    self.best_model_state = self.model.state_dict()
                    logger.info(f"  New best Sharpe: {best_sharpe:.4f}")
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

                # 学习率调整
                self.trainer.scheduler.step(val_metrics['sharpe_ratio'])

        # 加载最佳模型
        if hasattr(self, 'best_model_state'):
            self.model.load_state_dict(self.best_model_state)

        self.is_trained = True
        logger.info(f"Training completed. Best Sharpe: {best_sharpe:.4f}")

        return True
    
    def generate_positions(self, features_df: pd.DataFrame, 
                          feature_cols: List[str]) -> pd.Series:
        """生成仓位信号"""
        if not self.is_trained:
            logger.warning("Model not trained, returning zero positions")
            return pd.Series(0, index=features_df.index)

        # 确保使用训练时的特征列
        if hasattr(self, 'common_features'):
            # 只使用训练时确定的共同特征
            available_cols = [col for col in self.common_features if col in features_df.columns]
            if len(available_cols) < len(self.common_features):
                # 填充缺失的特征
                for col in self.common_features:
                    if col not in features_df.columns:
                        features_df[col] = 0
            X = features_df[self.common_features].values
        else:
            X = features_df[feature_cols].values

        # 归一化
        if self.feature_scaler is not None:
            X = self.feature_scaler.transform(X)

        # 生成仓位
        positions = self.trainer.generate_positions(X)

        # 转换为Series
        positions_series = pd.Series(positions, index=features_df.index[:len(positions)])

        # 平滑处理
        positions_series = positions_series.rolling(3, min_periods=1).mean()

        return positions_series

# ================== 集成到主策略的函数 ==================
def integrate_neural_network_positions(strategy_instance):
    """
    将神经网络仓位生成器集成到现有策略
    
    Args:
        strategy_instance: 主策略实例
    """
    # 检查是否有必要的数据
    if not hasattr(strategy_instance, 'features_data') or not strategy_instance.features_data:
        logger.error("No features data available for neural network training")
        return None
    
    # 获取特征维度
    first_symbol = list(strategy_instance.features_data.keys())[0]
    _, feature_cols = strategy_instance.features_data[first_symbol]
    feature_num = len(feature_cols)
    
    logger.info(f"Creating neural network position generator with {feature_num} features")
    
    # 创建神经网络生成器
    nn_generator = NeuralNetworkPositionGenerator(
        feature_num=feature_num,
        use_transformer=False  # 可以根据需要启用Transformer
    )
    
    # 训练模型
    if hasattr(strategy_instance, 'min30_data'):
        success = nn_generator.train(
            strategy_instance.features_data,
            strategy_instance.min30_data,
            epochs=3
        )
        
        if success:
            logger.info("Neural network position generator trained successfully")
            
            # 替换原有的信号生成逻辑
            original_generate = strategy_instance.signal_generator.generate_signals_from_features
            
            def nn_generate_signals(features_df, feature_cols, df, symbol, is_realtime=False):
                # 使用神经网络生成仓位
                positions = nn_generator.generate_positions(features_df, feature_cols)
                
                # 应用延迟以避免未来函数
                positions = positions.shift(2).fillna(0)
                
                # 如果是实时模式，清零最新信号
                if is_realtime:
                    positions.iloc[-5:] = 0
                
                return positions
            
            # 替换方法
            strategy_instance.signal_generator.generate_signals_from_features = nn_generate_signals
            
            logger.info("Neural network position generator integrated successfully")
        else:
            logger.error("Failed to train neural network position generator")
    
    return nn_generator
# ================== 修复1: ImprovedTargetGenerator ==================
class ImprovedTargetGenerator:
    """修复后的目标生成器"""
    
    def __init__(self, commission_rate: float = 0.0002):
        self.commission_rate = commission_rate
    
    def generate_position_targets(self, df: pd.DataFrame, 
                                 horizon: int = 5,
                                 is_training: bool = True) -> pd.Series:
        """
        修复：添加正确的方法名
        生成更激进的目标仓位以提高收益
        """
        if not is_training:
            return pd.Series(index=df.index, data=0, dtype=float)
        
        close = df['close'].copy()
        volume = df['volume'].copy() if 'volume' in df.columns else None
        targets = pd.Series(index=df.index, dtype=float)
        targets[:] = 0
        
        # 使用更短的历史窗口和更激进的参数
        hist_window = 30  # 缩短历史窗口
        min_periods = 20
        
        for i in range(min_periods, len(df) - horizon):
            # 计算历史统计（只用过去数据）
            hist_start = max(0, i - hist_window)
            hist_returns = close.iloc[hist_start:i].pct_change().dropna()
            
            if len(hist_returns) < 10:
                continue
            
            # 计算波动率
            volatility = hist_returns.std()
            if volatility <= 0:
                volatility = 0.01
            
            # 计算未来收益（仅训练时）
            future_return = (close.iloc[i + horizon] / close.iloc[i] - 1)
            
            # 更激进的仓位计算
            if abs(future_return) > self.commission_rate * 1.5:  # 降低阈值
                # 使用更大的Kelly fraction
                position = future_return / (volatility ** 2) * 0.8  # 提高到0.8
                
                # 考虑成交量（如果有）
                if volume is not None:
                    vol_ratio = volume.iloc[i] / volume.iloc[hist_start:i].mean()
                    position *= min(vol_ratio, 1.5)
                
                # 放宽仓位限制
                position = np.clip(position, -0.8, 0.8)  # 允许80%仓位
                
                # 降低过滤阈值
                if abs(position) < 0.005:  # 从0.01降低到0.005
                    position = 0
                
                targets.iloc[i] = position
        
        # 确保有足够的非零仓位
        non_zero_ratio = (targets != 0).mean()
        logger.info(f"Generated targets with {non_zero_ratio:.2%} non-zero positions")
        
        return targets

# ================== 修复2: 增强特征工程 ==================
class EnhancedFeatureEngineer:
    """增强的特征工程"""
    
    def __init__(self):
        self.feature_importance = {}
        self.selected_features = []
    
    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建高质量特征"""
        
        # 价格特征
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # 动量特征
        for period in [3, 5, 10, 20]:
            df[f'momentum_{period}'] = df['close'].pct_change(period)
            df[f'volume_momentum_{period}'] = df['volume'].pct_change(period)
        
        # 波动率特征
        for period in [5, 10, 20]:
            returns = df['close'].pct_change()
            df[f'volatility_{period}'] = returns.rolling(period, min_periods=period//2).std()
        
        # RSI
        for period in [7, 14, 21]:
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(period, min_periods=period//2).mean()
            loss = -delta.where(delta < 0, 0).rolling(period, min_periods=period//2).mean()
            rs = gain / (loss + 1e-10)
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # 布林带
        for period in [10, 20]:
            ma = df['close'].rolling(period, min_periods=period//2).mean()
            std = df['close'].rolling(period, min_periods=period//2).std()
            df[f'bb_upper_{period}'] = ma + 2 * std
            df[f'bb_lower_{period}'] = ma - 2 * std
            df[f'bb_position_{period}'] = (df['close'] - df[f'bb_lower_{period}']) / (
                df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'] + 1e-10
            )
        
        # 成交量特征
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20, min_periods=10).mean()
        
        # 价格形态
        df['high_low_ratio'] = df['high'] / df['low'] - 1
        df['close_open_ratio'] = df['close'] / df['open'] - 1
        
        # 清理数据
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(method='ffill', limit=5).fillna(0)
        
        return df
    
    def select_top_features(self, X: pd.DataFrame, y: pd.Series, n_features: int = 50) -> list:
        """选择最重要的特征"""
        if len(X.columns) <= n_features:
            self.selected_features = X.columns.tolist()
            return self.selected_features
        
        # 计算互信息
        mi_scores = mutual_info_regression(X.values, y.values, random_state=42)
        feature_scores = pd.Series(mi_scores, index=X.columns)
        
        # 选择top特征
        self.selected_features = feature_scores.nlargest(n_features).index.tolist()
        self.feature_importance = feature_scores.to_dict()
        
        return self.selected_features

# ================== 修复3: 优化模型架构 ==================
class OptimizedModelArchitecture:
    """优化的模型架构"""
    
    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu
        self.models = {}
        self.scaler = RobustScaler()
        self.feature_cols = []
        
    def get_xgb_params(self):
        """XGBoost参数 - 更激进的设置"""
        params = {
            'objective': 'reg:squarederror',
            'max_depth': 5,  # 增加深度
            'learning_rate': 0.02,  # 提高学习率
            'n_estimators': 200,  # 减少树的数量
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,  # 降低最小权重
            'reg_alpha': 0.1,  # 降低正则化
            'reg_lambda': 1.0,
            'gamma': 0.1,
            'random_state': 42,
            'n_jobs': -1
        }
        
        if self.use_gpu:
            params.update({
                'tree_method': 'gpu_hist',
                'predictor': 'gpu_predictor'
            })
        
        return params
    
    def get_lgb_params(self):
        """LightGBM参数 - 更激进的设置"""
        return {
            'objective': 'regression',
            'metric': 'rmse',
            'num_leaves': 31,  # 增加叶子数
            'learning_rate': 0.02,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'min_data_in_leaf': 20,  # 降低最小数据量
            'lambda_l1': 0.1,
            'lambda_l2': 1.0,
            'num_iterations': 200,
            'early_stopping_rounds': 30,
            'verbose': -1,
            'random_state': 42
        }
    
    def train_ensemble(self, X_train, y_train, X_val, y_val):
        """训练集成模型"""
        
        # 训练XGBoost
        try:
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val, label=y_val)
            
            self.models['xgb'] = xgb.train(
                self.get_xgb_params(),
                dtrain,
                num_boost_round=200,
                evals=[(dval, 'eval')],
                early_stopping_rounds=30,
                verbose_eval=False
            )
            
            pred = self.models['xgb'].predict(dval)
            corr = np.corrcoef(pred, y_val)[0, 1]
            logger.info(f"XGBoost validation correlation: {corr:.4f}")
        except Exception as e:
            logger.error(f"XGBoost training failed: {e}")
        
        # 训练LightGBM
        try:
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            self.models['lgb'] = lgb.train(
                self.get_lgb_params(),
                train_data,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)]
            )
            
            pred = self.models['lgb'].predict(X_val)
            corr = np.corrcoef(pred, y_val)[0, 1]
            logger.info(f"LightGBM validation correlation: {corr:.4f}")
        except Exception as e:
            logger.error(f"LightGBM training failed: {e}")
        
        # 训练CatBoost
        try:
            cb_params = {
                'iterations': 200,
                'learning_rate': 0.02,
                'depth': 5,
                'l2_leaf_reg': 3,
                'min_data_in_leaf': 20,
                'random_strength': 1.0,
                'bagging_temperature': 0.8,
                'loss_function': 'RMSE',
                'early_stopping_rounds': 30,
                'verbose': False,
                'random_state': 42
            }
            
            self.models['cb'] = CatBoostRegressor(**cb_params)
            self.models['cb'].fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                verbose=False
            )
            
            pred = self.models['cb'].predict(X_val)
            corr = np.corrcoef(pred, y_val)[0, 1]
            logger.info(f"CatBoost validation correlation: {corr:.4f}")
        except Exception as e:
            logger.error(f"CatBoost training failed: {e}")
    
    def predict_positions(self, X):
        """预测仓位（使用集成）"""
        predictions = []
        
        # 收集各模型预测
        if 'xgb' in self.models:
            dmatrix = xgb.DMatrix(X)
            pred = self.models['xgb'].predict(dmatrix)
            predictions.append(pred)
        
        if 'lgb' in self.models:
            pred = self.models['lgb'].predict(X)
            predictions.append(pred)
        
        if 'cb' in self.models:
            pred = self.models['cb'].predict(X)
            predictions.append(pred)
        
        if not predictions:
            return np.zeros(X.shape[0])
        
        # 使用平均集成
        ensemble_pred = np.mean(predictions, axis=0)
        
        # 增强强信号
        ensemble_pred = np.where(
            np.abs(ensemble_pred) > 0.1,
            ensemble_pred * 1.5,  # 放大强信号
            ensemble_pred
        )
        
        # 限制仓位
        ensemble_pred = np.clip(ensemble_pred, -0.8, 0.8)
        
        # 过滤极小仓位
        ensemble_pred[np.abs(ensemble_pred) < 0.008] = 0
        
        return ensemble_pred

# ================== 修复4: 改进的主策略 ==================
class ImprovedEndToEndStrategy:
    """改进的端到端策略"""
    
    def __init__(self, commission_rate: float = 0.0002):
        self.commission_rate = commission_rate
        self.target_generator = ImprovedTargetGenerator(commission_rate)
        self.feature_engineer = EnhancedFeatureEngineer()
        self.model = OptimizedModelArchitecture()
        self.trained = False
        
    def prepare_training_data(self, market_data):
        """准备训练数据"""
        all_features = []
        all_targets = []
        
        for symbol, df in market_data.items():
            if len(df) < 100:
                continue
            
            # 生成特征
            features_df = self.feature_engineer.create_advanced_features(df.copy())
            
            # 生成目标
            targets = self.target_generator.generate_position_targets(
                df, horizon=5, is_training=True
            )
            
            # 对齐数据
            common_idx = features_df.index.intersection(targets.index)[50:]  # 跳过初始不稳定期
            
            if len(common_idx) > 100:
                # 选择特征列
                feature_cols = [col for col in features_df.columns 
                              if col not in ['open', 'high', 'low', 'close', 'volume']]
                
                all_features.append(features_df.loc[common_idx, feature_cols])
                all_targets.append(targets.loc[common_idx])
        
        if all_features:
            X = pd.concat(all_features, ignore_index=True)
            y = pd.concat(all_targets, ignore_index=True)
            
            # 特征选择
            selected_features = self.feature_engineer.select_top_features(X, y, n_features=30)
            X = X[selected_features]
            
            logger.info(f"Training data shape: {X.shape}")
            logger.info(f"Non-zero targets: {(y != 0).mean():.2%}")
            logger.info(f"Target mean: {y.mean():.4f}, std: {y.std():.4f}")
            
            return X, y
        
        return None, None
    
    def train(self, train_data):
        """训练模型"""
        X, y = self.prepare_training_data(train_data)
        
        if X is None or len(X) < 100:
            logger.error("Insufficient training data")
            return False
        
        # 数据分割
        split_idx = int(len(X) * 0.8)
        X_train = X.iloc[:split_idx]
        y_train = y.iloc[:split_idx]
        X_val = X.iloc[split_idx:]
        y_val = y.iloc[split_idx:]
        
        # 数据标准化
        X_train_scaled = self.model.scaler.fit_transform(X_train)
        X_val_scaled = self.model.scaler.transform(X_val)
        
        # 保存特征列
        self.model.feature_cols = X.columns.tolist()
        
        # 训练模型
        self.model.train_ensemble(X_train_scaled, y_train, X_val_scaled, y_val)
        
        self.trained = True
        return True
    
    def generate_positions(self, market_data):
        """生成交易仓位"""
        if not self.trained:
            logger.error("Model not trained yet")
            return {}
        
        positions_dict = {}
        
        for symbol, df in market_data.items():
            try:
                # 生成特征
                features_df = self.feature_engineer.create_advanced_features(df.copy())
                
                # 选择特征
                feature_cols = [col for col in features_df.columns 
                              if col not in ['open', 'high', 'low', 'close', 'volume']]
                
                # 确保使用相同的特征
                available_features = [f for f in self.model.feature_cols if f in feature_cols]
                if len(available_features) < len(self.model.feature_cols) * 0.8:
                    logger.warning(f"{symbol}: Missing too many features")
                    positions_dict[symbol] = pd.Series(index=df.index, data=0)
                    continue
                
                # 对缺失特征填充0
                X = pd.DataFrame(index=features_df.index)
                for feat in self.model.feature_cols:
                    if feat in features_df.columns:
                        X[feat] = features_df[feat]
                    else:
                        X[feat] = 0
                
                # 标准化
                X_scaled = self.model.scaler.transform(X)
                
                # 预测
                raw_positions = self.model.predict_positions(X_scaled)
                positions = pd.Series(raw_positions, index=features_df.index)
                
                # 添加延迟（避免未来函数）
                positions = positions.shift(2).fillna(0)
                
                # 平滑处理
                if len(positions) > 3:
                    positions = positions.rolling(3, min_periods=1).mean()
                
                positions_dict[symbol] = positions
                
                # 统计
                non_zero = (positions != 0).sum()
                if non_zero > 0:
                    logger.info(f"{symbol}: Generated {non_zero} positions, "
                              f"avg size: {positions[positions!=0].mean():.4f}")
                
            except Exception as e:
                logger.error(f"Error generating positions for {symbol}: {e}")
                positions_dict[symbol] = pd.Series(index=df.index, data=0)
        
        return positions_dict

# ================== 使用示例 ==================
def run_improved_strategy(market_data, train_end_date='2023-07-01'):
    """运行改进的策略"""
    
    # 分割数据
    train_data = {}
    test_data = {}
    
    for symbol, df in market_data.items():
        train_mask = df.index < pd.to_datetime(train_end_date)
        if train_mask.sum() > 100:
            train_data[symbol] = df[train_mask]
            test_data[symbol] = df[~train_mask]
    
    if not train_data:
        logger.error("No training data available")
        return {}
    
    # 创建策略
    strategy = ImprovedEndToEndStrategy(commission_rate=0.0002)
    
    # 训练
    logger.info("Training improved strategy...")
    if not strategy.train(train_data):
        logger.error("Training failed")
        return {}
    
    # 生成仓位
    logger.info("Generating positions...")
    positions = strategy.generate_positions(test_data)
    
    # 统计
    total_positions = sum([(p != 0).sum() for p in positions.values()])
    logger.info(f"Total non-zero positions generated: {total_positions}")
    
    return positions

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
    
    # 移动平均线（使用safe_rolling）
    for period in [5, 10, 20, 30, 60]:
        min_periods = max(period // 2, 3)
        df[f'sma_{period}'] = safe_rolling(df['close'], period, min_periods).mean()
        df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        df[f'volume_sma_{period}'] = safe_rolling(df['volume'], period, min_periods).mean()
    
    # 价格位置特征
    for period in [10, 20, 30]:
        min_periods = max(period // 2, 5)
        rolling_high = safe_rolling(df['high'], period, min_periods).max()
        rolling_low = safe_rolling(df['low'], period, min_periods).min()
        df[f'price_position_{period}'] = (df['close'] - rolling_low) / (rolling_high - rolling_low + 1e-10)
    
    # 波动率特征
    for period in [5, 10, 20]:
        min_periods = max(period // 2, 3)
        df[f'volatility_{period}'] = safe_rolling(df['returns'], period, min_periods).std()
        df[f'volume_volatility_{period}'] = safe_rolling(df['volume'], period, min_periods).std()
    
    # RSI指标
    for period in [6, 12, 24]:
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        min_periods = max(period // 2, 3)
        avg_gain = safe_rolling(gain, period, min_periods).mean()
        avg_loss = safe_rolling(loss, period, min_periods).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
    
    # MACD指标
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_diff'] = df['macd'] - df['macd_signal']
    
    # 布林带
    for period in [10, 20]:
        min_periods = max(period // 2, 5)
        sma = safe_rolling(df['close'], period, min_periods).mean()
        std = safe_rolling(df['close'], period, min_periods).std()
        df[f'bb_upper_{period}'] = sma + 2 * std
        df[f'bb_lower_{period}'] = sma - 2 * std
        df[f'bb_width_{period}'] = (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']) / (sma + 1e-10)
        df[f'bb_position_{period}'] = (df['close'] - df[f'bb_lower_{period}']) / (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'] + 1e-10)
    
    # ATR（平均真实范围）
    for period in [7, 14, 21]:
        min_periods = max(period // 2, 3)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift(1))
        low_close = np.abs(df['low'] - df['close'].shift(1))
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df[f'atr_{period}'] = safe_rolling(true_range, period, min_periods).mean()
    
    # OBV（能量潮指标）
    obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    df['obv'] = obv
    df['obv_sma'] = safe_rolling(obv, 20, 10).mean()
    
    # 成交量特征
    df['volume_ratio'] = df['volume'] / safe_rolling(df['volume'], 20, 10).mean()
    df['volume_trend'] = safe_rolling(df['volume'], 5, 3).mean() / safe_rolling(df['volume'], 20, 10).mean()
    
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
    
    # 填充NaN值（使用向前填充，确保不使用未来数据）
    df = df.fillna(method='ffill', limit=5)
    
    # 对于仍然存在的NaN，使用0填充
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
                # 确保历史窗口不包括当前点
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
    
    # 使用safe_rolling计算平均未来收益
    shifted_prices = close_prices.shift(-horizon)
    returns_avg = (safe_rolling(shifted_prices, horizon, 1).mean() / close_prices - 1)
    
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
        cost_threshold = 0.0004  # 双边成本
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
        self.intelligent_position_manager = None  # 初始化为None而不是实例
        self.adaptive_sizer = AdaptivePositionSizer()
        
        # 添加历史交易记录用于计算胜率
        self.trade_history = []
        self.recent_trades_window = 50
    
    def _calculate_win_rate(self):
        """计算历史胜率"""
        if not self.trade_history:
            return 0.5  # 默认50%胜率
        
        recent_trades = self.trade_history[-self.recent_trades_window:]
        if not recent_trades:
            return 0.5
            
        wins = sum(1 for trade in recent_trades if trade > 0)
        return wins / len(recent_trades)
    
    def _calculate_win_loss_ratio(self):
        """计算盈亏比"""
        if not self.trade_history:
            return 1.5  # 默认1.5盈亏比
        
        recent_trades = self.trade_history[-self.recent_trades_window:]
        wins = [trade for trade in recent_trades if trade > 0]
        losses = [trade for trade in recent_trades if trade < 0]
        
        if not wins or not losses:
            return 1.5
            
        avg_win = sum(wins) / len(wins)
        avg_loss = abs(sum(losses) / len(losses))
        
        return avg_win / avg_loss if avg_loss > 0 else 1.5
    
    def update_trade_history(self, pnl):
        """更新交易历史"""
        self.trade_history.append(pnl)
        # 限制历史长度
        if len(self.trade_history) > 1000:
            self.trade_history = self.trade_history[-1000:]
    
    def calculate_position_size_optimized(self, signal: float, symbol: str, 
                                        volatility: float = None,
                                        portfolio_value: float = None) -> float:
        # 收集市场条件
        market_conditions = {
            'volatility': volatility or 0.02,
            'historical_win_rate': self._calculate_win_rate(),
            'win_loss_ratio': self._calculate_win_loss_ratio()
        }
        
        # 使用自适应仓位调整器
        adaptive_position = self.adaptive_sizer.calculate_position(signal, market_conditions)
        
        # 如果有智能仓位管理器，结合使用
        if self.intelligent_position_manager:
            try:
                ml_position = self.intelligent_position_manager.calculate_optimal_position(
                    signal, symbol, {}, 0  # 简化参数
                )
                # 取两者的加权平均
                final_position = adaptive_position * 0.6 + ml_position * 0.4
            except:
                final_position = adaptive_position
        else:
            final_position = adaptive_position
        
        return final_position


class NoFutureSignalGenerator:
    """改进的信号生成器，严格避免未来函数"""
    
    def __init__(self, model_trainer):
        self.model_trainer = model_trainer
        self.signal_delay = 2
        self.base_threshold = 0.01
        self.signal_scale_factor = 0.05
        self.leakage_checker = DataLeakageChecker()
        
        # 添加交易频率优化器
        self.frequency_optimizer = TradingFrequencyOptimizer()

    def generate_signals_from_features(self, features_df, feature_cols, df, symbol, 
                                      is_realtime: bool = False):
        """
        生成交易信号 - 完整改进版本
        """
        try:
            if len(features_df) < 100:
                logger.warning(f"{symbol}: 数据不足，跳过信号生成")
                return pd.Series(0, index=features_df.index, dtype=float)

            logger.info(f"{symbol}: 开始生成信号 (实时模式: {is_realtime})...")

            # 准备特征
            X = self._prepare_features(features_df, feature_cols)
            if X is None:
                return pd.Series(0, index=features_df.index, dtype=float)

            # 如果是实时模式，排除最新的几个数据点
            if is_realtime:
                X = X[:-self.signal_delay] if len(X) > self.signal_delay else X

            # 获取预测（预期收益）
            try:
                if hasattr(self.model_trainer, 'predict_ensemble'):
                    predictions = self.model_trainer.predict_ensemble(X)
                elif hasattr(self.model_trainer, 'predict_positions'):
                    predictions = self.model_trainer.predict_positions(X)
                else:
                    logger.warning(f"{symbol}: No prediction method available, using zeros")
                    predictions = np.zeros(X.shape[0])
            except Exception as e:
                logger.error(f"{symbol}: Prediction failed: {e}")
                predictions = np.zeros(X.shape[0])

            # 创建信号序列
            signals = pd.Series(predictions, index=features_df.index[:len(predictions)])

            # 计算置信度分数
            confidence_scores = pd.Series(index=signals.index, dtype=float)

            # 基于多个因素计算置信度
            for i in range(len(signals)):
                if i < 20:
                    confidence_scores.iloc[i] = 0.5
                    continue

                # 获取历史窗口
                hist_start = max(0, i - 50)
                hist_signals = signals.iloc[hist_start:i]

                # 计算信号稳定性
                if len(hist_signals) > 10:
                    signal_stability = 1.0 - hist_signals.std() / (abs(hist_signals.mean()) + 1e-10)
                    signal_stability = max(0, min(1, signal_stability))
                else:
                    signal_stability = 0.5

                # 信号强度
                signal_strength = min(abs(signals.iloc[i]) * 2, 1.0)

                # 趋势一致性
                if len(hist_signals) > 5:
                    recent_trend = hist_signals.iloc[-5:].mean()
                    trend_consistency = 1.0 if np.sign(recent_trend) == np.sign(signals.iloc[i]) else 0.5
                else:
                    trend_consistency = 0.5

                # 综合置信度
                confidence = signal_stability * 0.3 + signal_strength * 0.4 + trend_consistency * 0.3
                confidence_scores.iloc[i] = confidence

            # 应用交易频率优化
            if hasattr(self, 'frequency_optimizer'):
                # 计算动态阈值
                volatility = features_df.get('volatility_20', pd.Series(0.02, index=features_df.index))
                dynamic_threshold = pd.Series(index=signals.index, dtype=float)

                for i in range(len(signals)):
                    if i < len(volatility):
                        vol = volatility.iloc[i] if pd.notna(volatility.iloc[i]) else 0.02
                        dynamic_threshold.iloc[i] = max(self.frequency_optimizer.signal_threshold, vol * 3)
                    else:
                        dynamic_threshold.iloc[i] = self.frequency_optimizer.signal_threshold

                # 过滤低置信度和弱信号
                mask = (abs(signals) > dynamic_threshold) & (confidence_scores > 0.7)
                filtered_signals = signals.where(mask, 0)
            else:
                # 使用基础阈值过滤
                threshold = self.base_threshold * (1 + features_df.get('volatility_20', 0.02).mean() * 10)
                mask = abs(signals) > threshold
                filtered_signals = signals.where(mask, 0)

            # 信号缩放
            scaled_signals = filtered_signals * self.signal_scale_factor

            # 限制信号范围
            scaled_signals = scaled_signals.clip(-1.0, 1.0)

            # 信号平滑
            if len(scaled_signals) > 3:
                smoothed_signals = scaled_signals.rolling(window=3, min_periods=1, center=False).mean()
            else:
                smoothed_signals = scaled_signals

            # 添加延迟以避免未来函数
            final_signals = smoothed_signals.shift(self.signal_delay).fillna(0)

            # 实时模式额外保护
            if is_realtime:
                final_signals.iloc[-5:] = 0

            # 统计信息
            non_zero = (final_signals != 0).sum()
            if non_zero > 0:
                avg_signal = final_signals[final_signals != 0].mean()
                max_signal = final_signals.abs().max()
                logger.info(f"{symbol}: 生成 {non_zero} 个非零信号, 平均强度: {avg_signal:.4f}, 最大强度: {max_signal:.4f}")
            else:
                logger.info(f"{symbol}: 没有生成有效信号")

            return final_signals

        except Exception as e:
            logger.error(f"生成信号时出错 {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return pd.Series(0, index=features_df.index, dtype=float)
    
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
       self.cost_aware_trading = CostAwareTrading()

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
        """执行交易 - 增强版本"""

        # 获取当前持仓
        current_position = self.positions.get(symbol)
        if current_position is None:
            current_quantity = 0
        else:
            current_quantity = current_position.quantity

        # 计算目标数量
        target_value = target_position * self.initial_capital
        target_quantity = target_value / price

        # 计算仓位变化
        quantity_diff = target_quantity - current_quantity
        position_change = abs(quantity_diff * price / self.initial_capital)

        # 检查最小交易金额
        if abs(quantity_diff * price) < 500:
            return None

        # 成本感知检查
        if hasattr(self, 'cost_aware_trading'):
            # 估算预期收益
            expected_return = abs(signal) * volatility * 2  # 简单估算

            # 检查是否值得交易
            if not self.cost_aware_trading.should_trade(expected_return, position_change):
                logger.debug(f"{symbol}: 交易不满足成本效益要求，跳过")
                return None

        # 计算滑点
        is_buy = quantity_diff > 0
        slippage_rate = 0.0001

        if is_buy:
            execution_price = price * (1 + slippage_rate)
        else:
            execution_price = price * (1 - slippage_rate)

        # 计算交易金额和手续费
        trade_amount = abs(quantity_diff * execution_price)
        commission = trade_amount * self.commission_rate

        # 检查买入资金是否充足
        if is_buy:
            required_cash = trade_amount + commission
            if required_cash > cash * 0.95:  # 保留5%现金缓冲
                # 调整买入数量
                max_quantity = (cash * 0.95 - commission) / execution_price
                if max_quantity <= 0:
                    return None
                quantity_diff = min(quantity_diff, max_quantity)
                trade_amount = abs(quantity_diff * execution_price)
                commission = trade_amount * self.commission_rate

        # 确保交易数量有意义
        if abs(quantity_diff) < 0.001:
            return None

        # 执行交易
        if is_buy:
            new_cash = cash - (trade_amount + commission)
        else:
            new_cash = cash + (trade_amount - commission)

        # 计算滑点成本
        slippage_cost = abs(execution_price - price) * abs(quantity_diff)

        # 更新或创建持仓
        if symbol not in self.positions:
            # 新建持仓
            if abs(target_quantity) > 0.001:
                # 计算止损止盈
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

                logger.debug(f"{symbol}: 开仓 - 数量: {target_quantity:.4f}, 价格: {execution_price:.4f}")
        else:
            # 更新现有持仓
            old_position = self.positions[symbol]
            new_quantity = old_position.quantity + quantity_diff

            if abs(new_quantity) < 0.001:
                # 平仓
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

                logger.debug(f"{symbol}: 平仓 - PnL: {pnl:.2f}, Net PnL: {trade.net_pnl:.2f}")
            else:
                # 部分平仓或加仓
                if quantity_diff * old_position.quantity < 0:
                    # 部分平仓
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
                    # 加仓
                    total_cost = old_position.entry_price * abs(old_position.quantity) + execution_price * abs(quantity_diff)
                    self.positions[symbol].entry_price = total_cost / abs(new_quantity)
                    self.positions[symbol].quantity = new_quantity

                    # 更新止损止盈
                    self.positions[symbol].stop_loss = risk_manager.calculate_stop_loss(
                        self.positions[symbol].entry_price, volatility, target_position, abs(signal)
                    )
                    self.positions[symbol].take_profit = risk_manager.calculate_take_profit(
                        self.positions[symbol].entry_price, volatility, target_position, abs(signal)
                    )
                    self.positions[symbol].trailing_stop = self.positions[symbol].stop_loss

                    logger.debug(f"{symbol}: 加仓 - 新数量: {new_quantity:.4f}, 平均价格: {self.positions[symbol].entry_price:.4f}")

        return new_cash
       
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
    """
    完全无未来函数的单品种数据处理函数 - 支持多时间尺度
    
    Args:
        args: (symbol, data_path, cache_dir, sample_rate, return_1m_data)
    
    Returns:
        (symbol, min30_data, min1_data, from_cache)
    """
    # 解包参数，支持旧版本兼容
    if len(args) == 4:
        symbol, data_path, cache_dir, sample_rate = args
        return_1m_data = True  # 默认返回1分钟数据
    else:
        symbol, data_path, cache_dir, sample_rate, return_1m_data = args
    
    try:
        # 导入缓存管理器
        cache = ProcessedDataCache(cache_dir)
        
        # 检查缓存
        cache_params = {
            'sample_rate': sample_rate, 
            'version': 'no_future_multiscale_v3',
            'return_1m': return_1m_data
        }
        cached_data = cache.load_data(symbol, 'min30_data_multiscale', cache_params)
        
        if cached_data is not None:
            logger.info(f"Loaded cached data for {symbol}")
            min30_data = cached_data.get('min30_data')
            min1_data = cached_data.get('min1_data', None)
            return symbol, min30_data, min1_data, True
        
        # 读取原始数据
        file_path = os.path.join(data_path, f"{symbol}.parquet")
        
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            return symbol, None, None, False
        
        logger.info(f"Processing {symbol} with multi-scale support...")
        
        # 读取1分钟数据
        df_1m = pd.read_parquet(
            file_path,
            columns=['datetime', 'open', 'high', 'low', 'close', 'volume'],
            engine='pyarrow'
        )
        
        df_1m.columns = df_1m.columns.str.lower()
        
        if 'datetime' in df_1m.columns:
            df_1m['datetime'] = pd.to_datetime(df_1m['datetime'])
            df_1m.set_index('datetime', inplace=True)
        else:
            logger.warning(f"No datetime column found for {symbol}")
            return symbol, None, None, False
        
        # 确保时间序列单调递增
        if not df_1m.index.is_monotonic_increasing:
            logger.warning(f"{symbol}: Sorting time series")
            df_1m = df_1m.sort_index()
        
        # 清理数据
        df_1m = df_1m.dropna(subset=['close', 'volume'])
        df_1m = df_1m[df_1m['volume'] > 0]
        
        # 生成30分钟K线
        min30_data = df_1m.resample('30T').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        
        min30_data = min30_data.dropna()
        
        # 采样处理
        if sample_rate < 1.0 and len(min30_data) > 100:
            original_len = len(min30_data)
            
            if return_1m_data:
                # 如果需要1分钟数据，使用尾部连续采样
                sample_size = int(len(min30_data) * sample_rate)
                min30_data = min30_data.iloc[-sample_size:]
                
                # 对应的1分钟数据也要截取
                start_time = min30_data.index[0]
                df_1m = df_1m[df_1m.index >= start_time]
            else:
                # 不需要1分钟数据时，可以随机采样
                sample_size = int(len(min30_data) * sample_rate)
                sample_indices = np.sort(np.random.choice(len(min30_data), sample_size, replace=False))
                min30_data = min30_data.iloc[sample_indices]
                df_1m = None  # 不保留1分钟数据
            
            logger.info(f"  Sampled {sample_rate*100:.0f}% of data: {original_len} -> {len(min30_data)} bars")
        
        # 清理数值数据
        min30_data = clean_numeric_data_no_future(min30_data)
        
        logger.info(f"  Final 30min bars: {len(min30_data)}, 1min data: {'Yes' if df_1m is not None else 'No'}")
        
        # 保存到缓存
        if len(min30_data) > 500:
            cache_data = {
                'min30_data': min30_data,
                'min1_data': df_1m if return_1m_data else None
            }
            cache.save_data(symbol, 'min30_data_multiscale', cache_data, cache_params)
            return symbol, min30_data, df_1m if return_1m_data else None, False
        else:
            logger.warning(f"  Insufficient data after processing: {len(min30_data)} bars")
            return symbol, None, None, False
        
    except Exception as e:
        logger.error(f"Error processing {symbol}: {e}")
        traceback.print_exc()
        return symbol, None, None, False

    
# ================== 2. 特征生成函数 ==================
def process_features_for_symbol_no_future(args):
    """
    完全无未来函数的特征生成函数 - 支持多时间尺度特征
    
    Args:
        args: (symbol, min30_data, cache_dir, min1_data)
    """
    # 解包参数，兼容旧版本
    if len(args) == 3:
        symbol, min30_data, cache_dir = args
        min1_data = None
    else:
        symbol, min30_data, cache_dir, min1_data = args
    
    try:
        cache = ProcessedDataCache(cache_dir)
        
        # 缓存参数
        cache_params = {
            'version': 'no_future_multiscale_features_v3',
            'has_1m_data': min1_data is not None,
            'factors_available': FACTORS_AVAILABLE if 'FACTORS_AVAILABLE' in globals() else False
        }
        
        cached_features = cache.load_data(symbol, 'features_multiscale', cache_params)
        
        if cached_features is not None:
            return symbol, cached_features['features_df'], cached_features['feature_cols'], True
        
        logger.info(f"Generating multi-scale features for {symbol}...")
        
        data = min30_data.copy()
        
        # 使用多时间尺度特征计算
        if min1_data is not None:
            data = calculate_multiscale_technical_indicators(data, min1_data)
        else:
            # 退回到标准特征计算
            data = calculate_safe_technical_indicators(data)
        
        # 如果有扩展因子库，添加扩展因子
        if 'FACTORS_AVAILABLE' in globals() and FACTORS_AVAILABLE:
            data = validate_and_calculate_extended_factors(data)
        
        # 选择特征列
        excluded_cols = ['open', 'high', 'low', 'close', 'volume', 'vwap', 'amount']
        feature_cols = [col for col in data.columns if col not in excluded_cols]
        
        # 针对30分钟预测优化特征选择
        if min1_data is not None:
            data, feature_cols = optimize_features_for_30min_prediction(data, feature_cols)
        
        logger.info(f"Generated {len(feature_cols)} features for {symbol}")
        
        # 去除前200行确保稳定
        if len(data) > 200:
            data = data.iloc[200:]
            logger.info(f"After removing initial rows: {len(data)} samples")
        
        # 特征质量检查
        features_to_remove = []
        for col in feature_cols:
            if col in data.columns:
                factor_data = data[col]
                
                # 缺失值检查
                null_ratio = factor_data.isnull().sum() / len(factor_data)
                if null_ratio > 0.3:
                    features_to_remove.append(col)
                    continue
                
                # 方差检查
                if factor_data.var() < 1e-12:
                    features_to_remove.append(col)
                    continue
                
                # 无穷值检查
                inf_ratio = np.isinf(factor_data).sum() / len(factor_data)
                if inf_ratio > 0.1:
                    features_to_remove.append(col)
                    continue
                
                # 清理特征
                data[col] = data[col].replace([np.inf, -np.inf], np.nan)
                data[col] = data[col].fillna(method='ffill', limit=5)
                
                # 使用历史中位数填充
                rolling_median = data[col].rolling(window=50, min_periods=20, center=False).median()
                data[col] = data[col].fillna(rolling_median)
                data[col] = data[col].fillna(0)
        
        # 移除低质量特征
        for col in features_to_remove:
            if col in feature_cols:
                feature_cols.remove(col)
            if col in data.columns:
                data = data.drop(columns=[col])
        
        logger.info(f"After quality check: {len(feature_cols)} valid features for {symbol}")
        
        if len(feature_cols) < 10:
            logger.warning(f"{symbol}: Too few valid features ({len(feature_cols)})")
            return symbol, None, None, False
        
        # 缓存结果
        features_data = {
            'features_df': data,
            'feature_cols': feature_cols
        }
        cache.save_data(symbol, 'features_multiscale', features_data, cache_params)
        
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
   def load_multiple_symbols_parallel(self, symbols: List[str], 
                                      sample_rate: float = 1.0,
                                      enable_multiscale: bool = True) -> Dict:
        """
        并行加载多个品种的数据 - 支持多时间尺度

        Args:
            symbols: 品种列表
            sample_rate: 采样率
            enable_multiscale: 是否启用多时间尺度特征

        Returns:
            包含30分钟和1分钟数据的字典
        """
        logger.info(f"Loading data for {len(symbols)} symbols using {self.n_processes} processes...")
        logger.info(f"Multi-scale features: {'Enabled' if enable_multiscale else 'Disabled'}")

        # 准备参数列表
        args_list = [
            (symbol, self.data_path, self.cache_dir, sample_rate, enable_multiscale) 
            for symbol in symbols
        ]

        min30_data = {}
        min1_data = {}
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
                        symbol, data_30m, data_1m, from_cache = future.result()

                        if data_30m is not None:
                            min30_data[symbol] = data_30m

                            if enable_multiscale and data_1m is not None:
                                min1_data[symbol] = data_1m

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
        if enable_multiscale:
            logger.info(f"  - With 1-minute data: {len(min1_data)}")

        # 返回数据字典
        return {
            'min30_data': min30_data,
            'min1_data': min1_data if enable_multiscale else {}
        }

       
   @monitor_memory
   def generate_features_parallel(self, min30_data_dict: Dict[str, pd.DataFrame],
                                min1_data_dict: Dict[str, pd.DataFrame] = None) -> Dict:
        """
        并行生成特征 - 支持多时间尺度
        
        Args:
            min30_data_dict: 30分钟数据字典
            min1_data_dict: 1分钟数据字典（可选）
        
        Returns:
            特征字典
        """
        logger.info(f"Generating features for {len(min30_data_dict)} symbols...")
        
        if min1_data_dict:
            logger.info("Using multi-scale features with 1-minute data")
        
        # 准备参数列表
        args_list = []
        for symbol, data_30m in min30_data_dict.items():
            data_1m = min1_data_dict.get(symbol) if min1_data_dict else None
            args_list.append((symbol, data_30m, self.cache_dir, data_1m))
        
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

    
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修复后的策略核心代码
解决Phase2/3冗余和信号生成问题
"""

class NoFutureCommodityStrategy:
    def __init__(self, data_path: str, info_path: str, use_gpu: bool = True, 
                 use_cache: bool = True, cache_dir: str = '~/autodl-tmp/processed_data_cache',
                 use_optuna: bool = False, optuna_trials: int = 30,
                 use_improved_components: bool = True):
        
        self.data_path = data_path
        self.info_path = info_path
        self.use_gpu = use_gpu
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        self.use_optuna = use_optuna
        self.optuna_trials = optuna_trials
        self.use_improved_components = use_improved_components
        
        self.data_loader = NoFutureDataLoader(data_path, info_path, cache_dir)
        
        if use_improved_components:
            self.improved_strategy = ImprovedEndToEndStrategy(commission_rate=0.0002)
            self.model_trainer = self.improved_strategy.model
            self.feature_engineer = self.improved_strategy.feature_engineer
            self.target_generator = self.improved_strategy.target_generator
        else:
            self.model_trainer = NoFutureModelTrainer(
                use_gpu=use_gpu, 
                use_optuna=use_optuna,
                optuna_trials=optuna_trials
            )
            self.feature_engineer = None
            self.target_generator = None
        
        self.signal_generator = None
        self.backtest_engine = NoFutureBacktestEngine(commission_rate=0.0002)
        self.result_analyzer = ResultAnalyzer()
        
        self.min30_data = {}
        self.features_data = {}
        self.signals = {}
        self.nn_generator = None  # 初始化为None
        self.trained = False
    # 在 NoFutureCommodityStrategy 中添加新方法
    def create_ensemble_strategy(self):
        """创建多策略集成"""
        
        # 如果已有多个模型/策略
        strategies = {}
        
        # ML策略
        if self.improved_strategy:
            strategies['ml_improved'] = self.improved_strategy
        
        # RRL神经网络策略  
        if self.nn_generator:
            strategies['rrl_neural'] = self.nn_generator
        
        # 可以添加其他策略
        # strategies['momentum'] = MomentumStrategy()
        
        # 动态权重分配
        weights = self._optimize_strategy_weights(strategies)
        
        return strategies, weights
    @monitor_memory
    def load_and_process_data(self, symbols: List[str] = None, 
                              max_symbols: int = 10,
                              sample_rate: float = 1.0,
                              enable_multiscale: bool = True) -> bool:
        """
        加载和处理数据 - 支持多时间尺度

        Args:
            symbols: 品种列表
            max_symbols: 最大品种数量
            sample_rate: 采样率
            enable_multiscale: 是否启用多时间尺度特征

        Returns:
            是否成功
        """
        logger.info("="*60)
        logger.info("Loading commodity data (Multi-Scale Enhanced Version)...")
        logger.info(f"Multi-scale features: {'ENABLED' if enable_multiscale else 'DISABLED'}")

        # 加载商品信息
        commodity_info = self.data_loader.load_commodity_info()
        if commodity_info is None:
            return False

        # 获取可用品种
        available_symbols = self.data_loader.get_available_symbols()

        if symbols is None:
            symbols = available_symbols[:max_symbols]
        else:
            symbols = [s for s in symbols if s in available_symbols]

        # 清理旧缓存
        if self.use_cache:
            self.data_loader.cache.clear_old_cache(days=30)

        # 加载数据（包含多时间尺度）
        data_dict = self.data_loader.load_multiple_symbols_parallel(
            symbols, sample_rate, enable_multiscale
        )

        self.min30_data = data_dict['min30_data']
        self.min1_data = data_dict.get('min1_data', {})

        if not self.min30_data:
            logger.error("No data loaded successfully")
            return False

        # 生成特征（传递1分钟数据）
        self.features_data = self.data_loader.generate_features_parallel(
            self.min30_data, 
            self.min1_data if enable_multiscale else None
        )

        if not self.features_data:
            logger.error("No features generated successfully")
            return False

        logger.info(f"Data processing completed! Processed {len(self.features_data)} symbols")

        # 如果使用多时间尺度，输出统计信息
        if enable_multiscale and self.min1_data:
            total_1m_bars = sum(len(df) for df in self.min1_data.values())
            total_30m_bars = sum(len(df) for df in self.min30_data.values())
            logger.info(f"Multi-scale data statistics:")
            logger.info(f"  - Total 1-minute bars: {total_1m_bars}")
            logger.info(f"  - Total 30-minute bars: {total_30m_bars}")
            logger.info(f"  - Data compression ratio: {total_1m_bars/total_30m_bars:.1f}x")

        return True

    def prepare_and_train_models(self, look_ahead: int = 3, min_samples: int = 300) -> bool:
        """
        准备训练数据并训练模型（移除RRL训练，避免冗余）
        """
        logger.info("="*60)
        
        # 1. 准备市场数据
        market_data_train = {}
        market_data_test = {}

        # 设置训练集截止日期
        all_dates = []
        for symbol, df in self.min30_data.items():
            all_dates.extend(df.index.tolist())
        all_dates = sorted(list(set(all_dates)))

        if len(all_dates) < 100:
            logger.error("数据不足，无法训练")
            return False

        split_idx = int(len(all_dates) * 0.8)
        train_end_date = all_dates[split_idx]

        # 分割数据
        for symbol, df in self.min30_data.items():
            if len(df) < min_samples:
                continue

            train_mask = df.index <= train_end_date
            test_mask = df.index > train_end_date

            if train_mask.sum() >= min_samples:
                market_data_train[symbol] = df[train_mask].copy()
            if test_mask.sum() > 0:
                market_data_test[symbol] = df[test_mask].copy()

        if not market_data_train:
            logger.error("没有足够的训练数据")
            return False

        logger.info(f"准备训练数据: {len(market_data_train)} 个品种")
        logger.info(f"训练集截止日期: {train_end_date}")

        # 2. 训练改进策略
        success = self.improved_strategy.train(market_data_train)

        if not success:
            logger.error("改进策略训练失败")
            return False

        logger.info("改进策略训练成功")
        
        # 3. 在测试集上生成仓位信号用于验证
        if market_data_test:
            test_positions = self.improved_strategy.generate_positions(market_data_test)
            total_positions = sum([(p != 0).sum() for p in test_positions.values()])
            logger.info(f"测试集总共生成 {total_positions} 个非零仓位")
        
        # 4. 设置模型训练器和信号生成器
        self.model_trainer = self.improved_strategy.model
        self.trained = True
        
        # 5. 初始化信号生成器（但不训练RRL）
        self.signal_generator = NoFutureSignalGenerator(self.model_trainer)
        
        return True

    def run_full_strategy(self, max_symbols: int = 10,
                          look_ahead: int = 3,
                          start_date: str = None, 
                          end_date: str = None,
                          sample_rate: float = 1.0,
                          use_neural_network: bool = True,
                          enable_multiscale: bool = True,
                          use_ensemble: bool = False):
        """
        运行完整策略 - 完整增强版本
        """
        print("="*60)
        print("Commodity Futures Quantitative Strategy")
        print("Complete No Future Function Version with Extended Factors & Neural Network")
        print(f"Using {self.data_loader.n_processes} processes")
        print(f"Cache directory: {self.cache_dir}")
        print(f"Extended factors available: {FACTORS_AVAILABLE}")
        print(f"Neural Network: {'ENABLED' if use_neural_network else 'DISABLED'}")
        print(f"Multi-scale features: {'ENABLED' if enable_multiscale else 'DISABLED'}")
        print(f"Ensemble Strategy: {'ENABLED' if use_ensemble else 'DISABLED'}")
        if FACTORS_AVAILABLE:
            print(f"Number of factor functions: {len(factors.func_list)}")
        print("="*60)

        start_time = time.time()

        # Phase 1: 数据加载
        logger.info("Phase 1: Loading and processing data...")
        if not self.load_and_process_data(
            max_symbols=max_symbols, 
            sample_rate=sample_rate,
            enable_multiscale=enable_multiscale
        ):
            logger.error("Data loading failed")
            return None

        data_time = time.time()
        logger.info(f"Data processing time: {(data_time - start_time)/60:.1f} minutes")

        # Phase 2: 模型训练
        logger.info("Phase 2: Training models...")
        if not self.prepare_and_train_models(look_ahead=look_ahead):
            logger.error("Model training failed")
            return None

        train_time = time.time()
        logger.info(f"Model training time: {(train_time - data_time)/60:.1f} minutes")

        # Phase 3: RRL集成（可选）
        if use_neural_network and self.trained:
            logger.info("Phase 3: Training and integrating RRL neural network...")
            try:
                self.nn_generator = self._train_and_integrate_rrl()
                if self.nn_generator:
                    logger.info("RRL integration successful")
                    nn_time = time.time()
                    logger.info(f"RRL training time: {(nn_time - train_time)/60:.1f} minutes")
                else:
                    logger.warning("RRL integration failed, continuing with ML only")
            except Exception as e:
                logger.error(f"Error in RRL integration: {e}")
                traceback.print_exc()

        # Phase 4: 信号生成
        logger.info("Phase 4: Generating trading signals...")

        if use_ensemble and (self.nn_generator or self.trained):
            # 使用集成策略
            logger.info("Using ensemble strategy for signal generation...")
            strategies = {}
            weights = {}

            # 收集可用策略
            if hasattr(self, 'improved_strategy') and self.improved_strategy.trained:
                strategies['ml_improved'] = self.improved_strategy
                weights['ml_improved'] = 0.4

            if self.nn_generator:
                strategies['rrl_neural'] = self.nn_generator
                weights['rrl_neural'] = 0.3

            # 基础ML策略
            if self.trained:
                strategies['ml_base'] = self
                weights['ml_base'] = 0.3

            # 归一化权重
            total_weight = sum(weights.values())
            if total_weight > 0:
                for k in weights:
                    weights[k] /= total_weight

            # 生成集成信号
            self.signals = {}
            for symbol in self.features_data:
                features_df, feature_cols = self.features_data[symbol]
                df = self.min30_data.get(symbol)

                if df is None:
                    continue

                symbol_signals = []
                symbol_weights = []

                # 获取各策略信号
                for strategy_name, strategy in strategies.items():
                    try:
                        if strategy_name == 'ml_improved':
                            positions = strategy.generate_positions({symbol: df})
                            signal = positions.get(symbol, pd.Series(0, index=features_df.index))
                        elif strategy_name == 'rrl_neural':
                            signal = strategy.generate_positions(features_df, feature_cols)
                        else:
                            signal = self.signal_generator.generate_signals_from_features(
                                features_df, feature_cols, df, symbol, is_realtime=False
                            )

                        if len(signal) > 0:
                            symbol_signals.append(signal)
                            symbol_weights.append(weights[strategy_name])
                    except Exception as e:
                        logger.error(f"Error generating signal for {symbol} with {strategy_name}: {e}")

                # 加权组合
                if symbol_signals:
                    combined_signal = pd.Series(0, index=features_df.index, dtype=float)
                    for sig, w in zip(symbol_signals, symbol_weights):
                        if len(sig) == len(combined_signal):
                            combined_signal += sig * w
                        else:
                            # 对齐索引
                            aligned = sig.reindex(combined_signal.index, fill_value=0)
                            combined_signal += aligned * w

                    self.signals[symbol] = combined_signal
                    non_zero = (combined_signal != 0).sum()
                    logger.info(f"{symbol}: Generated {non_zero} ensemble signals")
                else:
                    self.signals[symbol] = pd.Series(0, index=features_df.index)
        else:
            # 单策略信号生成
            if self.nn_generator:
                self._generate_rrl_positions()
            else:
                self._generate_ml_positions()

        signal_time = time.time()
        logger.info(f"Signal generation time: {(signal_time - train_time)/60:.1f} minutes")

        # 统计信号
        total_signals = sum([(s != 0).sum() for s in self.signals.values()])
        logger.info(f"Total non-zero signals generated: {total_signals}")

        # Phase 5: 回测
        logger.info("Phase 5: Running backtest...")

        # 添加成本感知交易
        if not hasattr(self.backtest_engine, 'cost_aware_trading'):
            self.backtest_engine.cost_aware_trading = CostAwareTrading()

        backtest_results = self.backtest_engine.run_backtest(
            self.signals, self.min30_data, start_date, end_date
        )

        backtest_time = time.time()
        logger.info(f"Backtest time: {(backtest_time - signal_time)/60:.1f} minutes")

        # Phase 6: 结果分析
        if backtest_results:
            report = self.result_analyzer.generate_report(backtest_results)

            # 添加策略特定信息
            if self.nn_generator:
                report += "\n=== Neural Network Performance ===\n"
                report += f"Model architecture: {'Transformer' if self.nn_generator.use_transformer else 'MLP'}\n"
                report += f"Feature dimension: {self.nn_generator.feature_num}\n"

            if use_ensemble:
                report += "\n=== Ensemble Strategy ===\n"
                report += f"Number of strategies: {len(strategies) if 'strategies' in locals() else 0}\n"
                if 'weights' in locals():
                    for name, weight in weights.items():
                        report += f"  {name}: {weight:.2%}\n"

            print(report)

            # 绘制结果图表
            try:
                self.result_analyzer.plot_results(backtest_results)
            except Exception as e:
                logger.error(f"Error plotting results: {e}")

            # 保存结果
            try:
                self._save_results(backtest_results, report)
            except Exception as e:
                logger.error(f"Error saving results: {e}")

        end_time = time.time()
        total_runtime = (end_time - start_time) / 60

        logger.info("="*60)
        logger.info(f"Strategy execution completed!")
        logger.info(f"Total runtime: {total_runtime:.1f} minutes")
        logger.info(f"Performance breakdown:")
        logger.info(f"  - Data loading: {(data_time - start_time)/60:.1f} min ({(data_time - start_time)/start_time*100:.0f}%)")
        logger.info(f"  - Model training: {(train_time - data_time)/60:.1f} min ({(train_time - data_time)/(end_time - start_time)*100:.0f}%)")
        logger.info(f"  - Signal generation: {(signal_time - train_time)/60:.1f} min ({(signal_time - train_time)/(end_time - start_time)*100:.0f}%)")
        logger.info(f"  - Backtesting: {(backtest_time - signal_time)/60:.1f} min ({(backtest_time - signal_time)/(end_time - start_time)*100:.0f}%)")
        logger.info("="*60)

        return backtest_results

    def _train_and_integrate_rrl(self):
        """
        训练并集成RRL（只训练一次）
        """
        if not hasattr(self, 'features_data') or not self.features_data:
            logger.error("No features data available for RRL training")
            return None

        # 获取特征维度
        first_symbol = list(self.features_data.keys())[0]
        _, feature_cols = self.features_data[first_symbol]
        feature_num = len(feature_cols)

        logger.info(f"Creating RRL position generator with {feature_num} features")

        # 创建神经网络生成器
        nn_generator = NeuralNetworkPositionGenerator(
            feature_num=feature_num,
            use_transformer=False  # MLP更稳定
        )

        # 训练模型
        if hasattr(self, 'min30_data'):
            success = nn_generator.train(
                self.features_data,
                self.min30_data,
                epochs=3
            )

            if success:
                logger.info("RRL position generator trained successfully")
                return nn_generator
            else:
                logger.error("Failed to train RRL position generator")

        return None

    def _generate_rrl_positions(self):
        """
        使用RRL直接生成端到端仓位
        """
        logger.info("Generating end-to-end positions using RRL...")
        
        for symbol, (features_df, feature_cols) in self.features_data.items():
            try:
                # RRL直接生成仓位
                positions = self.nn_generator.generate_positions(features_df, feature_cols)
                
                # 应用安全延迟（避免未来函数）
                positions = positions.shift(2).fillna(0)
                
                # 保存为信号
                self.signals[symbol] = positions
                
                non_zero = (positions != 0).sum()
                if non_zero > 0:
                    logger.info(f"{symbol}: Generated {non_zero} RRL positions, "
                              f"avg size: {positions[positions!=0].mean():.4f}")
                
            except Exception as e:
                logger.error(f"Error generating RRL positions for {symbol}: {e}")
                self.signals[symbol] = pd.Series(0, index=features_df.index)

    def _generate_ml_positions(self):
        """
        使用改进策略生成仓位
        """
        logger.info("Generating positions using improved ML strategy...")
        
        # 使用改进策略的generate_positions方法
        positions_dict = self.improved_strategy.generate_positions(self.min30_data)
        
        for symbol, positions in positions_dict.items():
            # 应用安全延迟
            positions = positions.shift(2).fillna(0)
            self.signals[symbol] = positions
            
            non_zero = (positions != 0).sum()
            if non_zero > 0:
                logger.info(f"{symbol}: Generated {non_zero} ML positions, "
                          f"avg size: {positions[positions!=0].mean():.4f}")

    def _save_results(self, backtest_results, report):
        """保存回测结果"""
        try:
            import json
            from datetime import datetime

            results_dir = os.path.expanduser('~/autodl-tmp/backtest_results/')
            os.makedirs(results_dir, exist_ok=True)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            # 保存报告
            report_file = os.path.join(results_dir, f'report_{timestamp}.txt')
            with open(report_file, 'w') as f:
                f.write(report)

            # 保存关键指标
            metrics = {
                'total_return': backtest_results['total_return'],
                'annual_return': backtest_results['annual_return'],
                'sharpe_ratio': backtest_results['sharpe_ratio'],
                'max_drawdown': backtest_results['max_drawdown'],
                'total_trades': backtest_results['total_trades'],
                'win_rate': backtest_results['win_rate']
            }

            metrics_file = os.path.join(results_dir, f'metrics_{timestamp}.json')
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)

            logger.info(f"Results saved to {results_dir}")

        except Exception as e:
            logger.error(f"Error saving results: {e}")


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
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多时间尺度特征构造优化
针对30分钟预测优化的特征工程
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
import talib

logger = logging.getLogger(__name__)

def calculate_multiscale_technical_indicators(df: pd.DataFrame, 
                                             raw_1m_data: pd.DataFrame = None) -> pd.DataFrame:
    """
    计算多时间尺度技术指标，针对30分钟预测优化
    
    Args:
        df: 30分钟K线数据
        raw_1m_data: 原始1分钟数据（可选）
    
    Returns:
        包含多尺度特征的DataFrame
    """
    logger.info(f"计算多时间尺度技术指标，数据形状: {df.shape}")
    
    # 确保必要的列存在
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in required_cols:
        if col not in df.columns:
            logger.error(f"缺少必需列: {col}")
            return df
    
    # ==================== 1. 优化的30分钟特征 ====================
    # 针对30分钟预测，使用更合适的周期参数
    
    # 短期移动平均（2-10根K线，即1-5小时）
    for period in [2, 3, 5, 8, 10]:
        df[f'sma_{period}'] = df['close'].rolling(window=period, min_periods=1, center=False).mean()
        df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
    
    # 中期移动平均（12-24根K线，即6-12小时）
    for period in [12, 16, 20, 24]:
        df[f'sma_{period}'] = df['close'].rolling(window=period, min_periods=period//2, center=False).mean()
    
    # 价格动量（使用更短的周期）
    for period in [2, 4, 8, 12]:
        df[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1
        df[f'roc_{period}'] = (df['close'] - df['close'].shift(period)) / df['close'].shift(period)
    
    # 短期RSI（更灵敏）
    for period in [4, 7, 14]:
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=period, min_periods=period//2, center=False).mean()
        avg_loss = loss.rolling(window=period, min_periods=period//2, center=False).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
    
    # 短期波动率
    for period in [3, 6, 12]:
        df[f'volatility_{period}'] = df['close'].pct_change().rolling(
            window=period, min_periods=period//2, center=False
        ).std()
    
    # VWAP偏离度（Volume Weighted Average Price）
    df['vwap'] = (df['close'] * df['volume']).rolling(window=10, center=False).sum() / \
                 df['volume'].rolling(window=10, center=False).sum()
    df['vwap_deviation'] = (df['close'] - df['vwap']) / df['vwap']
    
    # ==================== 2. 从1分钟数据提取微观结构特征 ====================
    if raw_1m_data is not None and len(raw_1m_data) > 0:
        df = add_microstructure_features(df, raw_1m_data)
    
    # ==================== 3. 多时间尺度聚合特征 ====================
    # 5分钟和15分钟的关键指标
    df = add_intermediate_timeframe_features(df)
    
    # ==================== 4. 时间衰减特征 ====================
    df = add_time_weighted_features(df)
    
    # ==================== 5. 价格行为模式特征 ====================
    df = add_price_pattern_features(df)
    
    # ==================== 6. 成交量分析特征 ====================
    df = add_volume_analysis_features(df)
    
    # 填充NaN值
    df = df.fillna(method='ffill', limit=3)
    df = df.fillna(0)
    
    logger.info(f"多时间尺度特征计算完成，生成特征数: {len(df.columns) - len(required_cols)}")
    
    return df


def add_microstructure_features(df_30m: pd.DataFrame, df_1m: pd.DataFrame) -> pd.DataFrame:
    """
    从1分钟数据中提取市场微观结构特征
    
    Args:
        df_30m: 30分钟K线数据
        df_1m: 1分钟K线数据
    
    Returns:
        添加了微观结构特征的DataFrame
    """
    features_to_add = {}
    
    # 对每个30分钟K线，计算对应的1分钟数据特征
    for idx in df_30m.index:
        # 获取当前30分钟内的1分钟数据
        end_time = idx
        start_time = idx - pd.Timedelta(minutes=30)
        
        mask = (df_1m.index > start_time) & (df_1m.index <= end_time)
        window_1m = df_1m[mask]
        
        if len(window_1m) > 0:
            # 价格波动次数（上穿下穿均价）
            mean_price = window_1m['close'].mean()
            crosses = np.sum(np.diff(np.sign(window_1m['close'] - mean_price)) != 0)
            features_to_add.setdefault('price_crosses_30m', []).append(crosses)
            
            # 最大价格偏离
            max_deviation = (window_1m['high'].max() - window_1m['low'].min()) / mean_price
            features_to_add.setdefault('max_deviation_30m', []).append(max_deviation)
            
            # 成交量分布偏度
            if 'volume' in window_1m.columns:
                volume_skew = window_1m['volume'].skew()
                features_to_add.setdefault('volume_skew_30m', []).append(
                    volume_skew if not pd.isna(volume_skew) else 0
                )
            
            # 价格加速度（二阶导数近似）
            if len(window_1m) >= 3:
                price_accel = window_1m['close'].diff().diff().mean()
                features_to_add.setdefault('price_acceleration_30m', []).append(
                    price_accel if not pd.isna(price_accel) else 0
                )
            
            # 高低点时间位置
            high_time_ratio = window_1m['high'].idxmax() 
            if pd.notna(high_time_ratio):
                time_ratio = (high_time_ratio - start_time).total_seconds() / 1800
                features_to_add.setdefault('high_time_position_30m', []).append(time_ratio)
            
        else:
            # 如果没有对应的1分钟数据，填充默认值
            for key in ['price_crosses_30m', 'max_deviation_30m', 'volume_skew_30m', 
                       'price_acceleration_30m', 'high_time_position_30m']:
                features_to_add.setdefault(key, []).append(0)
    
    # 将特征添加到DataFrame
    for key, values in features_to_add.items():
        if len(values) == len(df_30m):
            df_30m[key] = values
    
    return df_30m


def add_intermediate_timeframe_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    添加中间时间尺度（5分钟、15分钟等效）特征
    通过对30分钟数据进行不同粒度的分析来模拟
    """
    # 模拟5分钟级别的快速变化（使用更短的滚动窗口）
    df['fast_trend_1h'] = df['close'].rolling(window=2, center=False).mean() / \
                          df['close'].rolling(window=4, center=False).mean() - 1
    
    # 模拟15分钟级别的中速变化
    df['medium_trend_2h'] = df['close'].rolling(window=4, center=False).mean() / \
                           df['close'].rolling(window=8, center=False).mean() - 1
    
    # 不同时间尺度的价格位置
    for period in [4, 8, 16]:
        rolling_max = df['high'].rolling(window=period, min_periods=period//2, center=False).max()
        rolling_min = df['low'].rolling(window=period, min_periods=period//2, center=False).min()
        df[f'price_position_{period*30}m'] = (df['close'] - rolling_min) / (rolling_max - rolling_min + 1e-10)
    
    return df


def add_time_weighted_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    添加时间加权特征，近期数据权重更高
    """
    # 指数加权移动平均（更重视近期）
    for alpha in [0.1, 0.2, 0.3]:
        df[f'ewma_alpha_{int(alpha*10)}'] = df['close'].ewm(alpha=alpha, adjust=False).mean()
    
    # 加权价格动量
    weights = np.exp(-np.arange(10) * 0.1)[::-1]  # 指数衰减权重
    
    def weighted_momentum(series, weights):
        if len(series) < len(weights):
            return 0
        weighted_sum = np.sum(series[-len(weights):] * weights)
        weight_total = np.sum(weights)
        return weighted_sum / weight_total if weight_total > 0 else 0
    
    df['weighted_momentum'] = df['close'].rolling(window=10).apply(
        lambda x: weighted_momentum(x.values, weights), raw=False
    )
    
    # 近期波动率vs历史波动率
    recent_vol = df['close'].pct_change().rolling(window=6, center=False).std()
    historical_vol = df['close'].pct_change().rolling(window=24, center=False).std()
    df['vol_ratio_recent_hist'] = recent_vol / (historical_vol + 1e-10)
    
    return df


def add_price_pattern_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    添加价格形态特征
    """
    # 蜡烛图模式
    df['candle_body'] = abs(df['close'] - df['open']) / (df['high'] - df['low'] + 1e-10)
    df['upper_shadow'] = (df['high'] - df[['close', 'open']].max(axis=1)) / (df['high'] - df['low'] + 1e-10)
    df['lower_shadow'] = (df[['close', 'open']].min(axis=1) - df['low']) / (df['high'] - df['low'] + 1e-10)
    
    # 连续上涨/下跌计数
    df['consecutive_ups'] = (df['close'] > df['close'].shift(1)).astype(int)
    df['consecutive_ups'] = df['consecutive_ups'].groupby(
        (df['consecutive_ups'] != df['consecutive_ups'].shift()).cumsum()
    ).cumsum()
    
    df['consecutive_downs'] = (df['close'] < df['close'].shift(1)).astype(int)
    df['consecutive_downs'] = df['consecutive_downs'].groupby(
        (df['consecutive_downs'] != df['consecutive_downs'].shift()).cumsum()
    ).cumsum()
    
    # 支撑阻力位距离
    for period in [10, 20]:
        resistance = df['high'].rolling(window=period, center=False).max()
        support = df['low'].rolling(window=period, center=False).min()
        df[f'distance_to_resistance_{period}'] = (resistance - df['close']) / df['close']
        df[f'distance_to_support_{period}'] = (df['close'] - support) / df['close']
    
    return df


def add_volume_analysis_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    添加成交量分析特征
    """
    # 成交量移动平均
    for period in [5, 10, 20]:
        df[f'volume_ma_{period}'] = df['volume'].rolling(window=period, center=False).mean()
    
    # 成交量比率
    df['volume_ratio_5_20'] = df['volume_ma_5'] / (df['volume_ma_20'] + 1e-10)
    
    # 价量关系
    df['price_volume_corr'] = df['close'].rolling(window=10).corr(df['volume'])
    
    # 成交量异常检测
    vol_mean = df['volume'].rolling(window=20, center=False).mean()
    vol_std = df['volume'].rolling(window=20, center=False).std()
    df['volume_zscore'] = (df['volume'] - vol_mean) / (vol_std + 1e-10)
    
    # OBV的变化率
    obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    df['obv_roc'] = obv.pct_change(periods=5)
    
    return df


def optimize_features_for_30min_prediction(df: pd.DataFrame, 
                                          feature_cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """
    针对30分钟预测优化特征选择
    
    Args:
        df: 包含所有特征的DataFrame
        feature_cols: 原始特征列列表
    
    Returns:
        优化后的DataFrame和特征列
    """
    optimized_cols = []
    
    # 优先保留短中期特征
    priority_patterns = [
        'fast_', 'short_', 'quick_',  # 快速特征
        '_2', '_3', '_4', '_5', '_6',  # 短期周期
        'micro', 'intra',  # 微观结构
        'acceleration', 'velocity',  # 动量特征
        'alpha_', 'ewma_',  # 时间加权
    ]
    
    # 排除长周期特征
    exclude_patterns = [
        '_60', '_90', '_120',  # 超长周期
        'slow_', 'long_',  # 慢速特征
    ]
    
    for col in feature_cols:
        # 跳过超长周期
        if any(pattern in col for pattern in exclude_patterns):
            continue
            
        # 优先保留短期特征
        if any(pattern in col for pattern in priority_patterns):
            optimized_cols.append(col)
        # 保留中等周期特征
        elif any(str(i) in col for i in range(7, 25)):
            optimized_cols.append(col)
        # 保留没有周期标记的基础特征
        elif not any(str(i) in col for i in range(25, 100)):
            optimized_cols.append(col)
    
    # 添加专门的30分钟预测特征
    if 'close' in df.columns:
        # 最近K线的变化率序列
        for i in range(1, 5):
            col_name = f'price_change_lag_{i}'
            if col_name not in df.columns:
                df[col_name] = df['close'].pct_change(periods=i)
            if col_name not in optimized_cols:
                optimized_cols.append(col_name)
        
        # 短期动量特征
        momentum_features = ['price_velocity', 'price_acceleration']
        if 'price_velocity' not in df.columns:
            df['price_velocity'] = df['close'].diff()
        if 'price_acceleration' not in df.columns:
            df['price_acceleration'] = df['price_velocity'].diff()
        
        for feat in momentum_features:
            if feat not in optimized_cols and feat in df.columns:
                optimized_cols.append(feat)
    
    # 确保特征列存在于DataFrame中
    optimized_cols = [col for col in optimized_cols if col in df.columns]
    
    logger.info(f"Feature optimization for 30min prediction: {len(feature_cols)} -> {len(optimized_cols)}")
    
    return df, optimized_cols

# ================== 集成到主策略的修改 ==================
def process_single_symbol_data_multiscale(args):
    """
    处理单个品种的多时间尺度数据
    """
    symbol, data_path, cache_dir, sample_rate = args
    
    try:
        # 加载1分钟原始数据
        file_path = os.path.join(data_path, f"{symbol}.parquet")
        
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            return symbol, None, None, False
        
        logger.info(f"Processing {symbol} with multi-scale features...")
        
        # 读取1分钟数据
        df_1m = pd.read_parquet(
            file_path,
            columns=['datetime', 'open', 'high', 'low', 'close', 'volume'],
            engine='pyarrow'
        )
        
        df_1m.columns = df_1m.columns.str.lower()
        
        if 'datetime' in df_1m.columns:
            df_1m['datetime'] = pd.to_datetime(df_1m['datetime'])
            df_1m.set_index('datetime', inplace=True)
        
        # 生成30分钟K线
        df_30m = df_1m.resample('30T').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        # 采样处理
        if sample_rate < 1.0 and len(df_30m) > 100:
            # 对于多时间尺度，需要保持时间连续性
            # 使用尾部采样而不是随机采样
            sample_size = int(len(df_30m) * sample_rate)
            df_30m = df_30m.iloc[-sample_size:]
            
            # 对应的1分钟数据也要截取
            start_time = df_30m.index[0]
            df_1m = df_1m[df_1m.index >= start_time]
        
        logger.info(f"  30min bars: {len(df_30m)}, 1min bars: {len(df_1m)}")
        
        if len(df_30m) > 500:
            return symbol, df_30m, df_1m, False
        else:
            logger.warning(f"  Insufficient data: {len(df_30m)} bars")
            return symbol, None, None, False
            
    except Exception as e:
        logger.error(f"Error processing {symbol}: {e}")
        return symbol, None, None, False


# ================== 使用示例 ==================
def integrate_multiscale_features(original_calculate_function):
    """
    装饰器：将原始的特征计算函数替换为多时间尺度版本
    
    使用方法:
    在主策略文件中，替换 calculate_safe_technical_indicators 函数：
    
    # 原始函数
    # data = calculate_safe_technical_indicators(data)
    
    # 替换为
    data = calculate_multiscale_technical_indicators(data, raw_1m_data)
    """
    def wrapper(df: pd.DataFrame, **kwargs):
        # 尝试获取1分钟数据
        raw_1m_data = kwargs.get('raw_1m_data', None)
        
        if raw_1m_data is not None:
            # 使用多时间尺度版本
            return calculate_multiscale_technical_indicators(df, raw_1m_data)
        else:
            # 回退到原始版本
            return original_calculate_function(df)
    
    return wrapper

class StrategyOptimizationSuite:
    """策略优化套件 - 解决核心问题"""
    
    def __init__(self):
        self.optimizations = {
            'signal_quality': True,
            'cost_reduction': True,
            'risk_management': True,
            'feature_engineering': True,
            'position_sizing': True
        }
    
    # ================== 1. 信号质量提升 ==================
    def enhance_signal_quality(self, signal_generator):
        """
        提升信号质量，减少噪音交易
        """
        # 动态阈值调整
        signal_generator.base_threshold = 0.005  # 提高基础阈值，减少弱信号
        signal_generator.signal_scale_factor = 0.2  # 增加缩放因子
        
        # 添加信号确认机制
        signal_generator.min_confirmation_bars = 2  # 需要2个连续信号确认
        signal_generator.signal_smoothing = 5  # 增加平滑窗口
        
        return signal_generator
    
    # ================== 2. 交易成本优化 ==================
    def optimize_trading_costs(self, risk_manager):
        """
        降低交易频率，减少成本
        """
        # 提高最小交易阈值
        risk_manager.min_position_change = 0.1  # 10%的仓位变化才交易
        risk_manager.min_holding_period = 10  # 最少持有10个周期(5小时)
        risk_manager.max_daily_trades = 5  # 每日最大交易次数降到5次
        
        # 改进仓位优化器
        if hasattr(risk_manager, 'position_optimizer'):
            risk_manager.position_optimizer.min_trade_threshold = 0.05  # 5%阈值
            risk_manager.position_optimizer.commission_rate = 0.0003  # 考虑更高成本
        
        return risk_manager
    
    # ================== 3. 目标变量优化 ==================
    def create_improved_target_variable(self, df: pd.DataFrame, horizon: int = 5, 
                                       is_training: bool = True) -> pd.Series:
        """
        改进的目标变量生成，使用更稳定的标签
        """
        close_prices = df['close'].copy()
        target = pd.Series(index=df.index, dtype=float)
        target[:] = 0
        
        if not is_training:
            return target
        
        for i in range(len(df)):
            if i < 100 or i >= len(df) - horizon * 2:
                target.iloc[i] = 0
                continue
            
            # 计算历史波动率（只用过去数据）
            hist_returns = close_prices.iloc[max(0, i-100):i].pct_change().dropna()
            if len(hist_returns) < 50:
                continue
                
            volatility = hist_returns.std()
            
            # 动态阈值：基于波动率和成本
            cost_threshold = 0.0006  # 双边成本
            signal_threshold = max(volatility * 1.5, cost_threshold * 3)  # 提高阈值
            
            # 使用更稳定的未来收益计算
            if i < len(df) - horizon * 2:
                # 多时间尺度收益
                returns_short = (close_prices.iloc[i+horizon] / close_prices.iloc[i] - 1)
                returns_medium = (close_prices.iloc[i+horizon*2] / close_prices.iloc[i] - 1)
                
                # 平滑收益（减少噪音）
                future_window = close_prices.iloc[i+1:i+horizon+1]
                avg_future = future_window.mean() if len(future_window) > 0 else close_prices.iloc[i]
                returns_smooth = (avg_future / close_prices.iloc[i] - 1)
                
                # 综合收益（更保守）
                combined_return = returns_short * 0.5 + returns_smooth * 0.3 + returns_medium * 0.2
                
                # 只在强信号时生成标签
                if abs(combined_return) > signal_threshold:
                    # 使用分级标签
                    if abs(combined_return) > signal_threshold * 2:
                        target.iloc[i] = np.sign(combined_return) * 1.0
                    else:
                        target.iloc[i] = np.sign(combined_return) * 0.5
                else:
                    target.iloc[i] = 0  # 中性区域
        
        return target
    
    # ================== 4. 特征工程优化 ==================
    def add_market_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        添加市场状态识别特征
        """
        # 市场趋势强度
        df['trend_strength'] = df['close'].rolling(20).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
        )
        
        # 市场波动状态
        df['volatility_regime'] = df['returns'].rolling(50).std()
        df['volatility_percentile'] = df['volatility_regime'].rolling(200).rank(pct=True)
        
        # 成交量异常检测
        df['volume_spike'] = df['volume'] / df['volume'].rolling(20).mean()
        
        # 市场效率指标
        df['efficiency_ratio'] = abs(df['close'].diff(10)) / (
            df['close'].diff().abs().rolling(10).sum() + 1e-10
        )
        
        return df
    
    # ================== 5. 智能止损止盈 ==================
    def create_adaptive_exit_rules(self, position, market_conditions: Dict) -> Dict:
        """
        基于市场条件的自适应退出规则
        """
        volatility = market_conditions.get('volatility', 0.02)
        trend_strength = market_conditions.get('trend_strength', 0)
        
        # 动态止损
        if volatility > 0.03:  # 高波动市场
            stop_loss_multiplier = 3.0
            take_profit_multiplier = 4.0
        else:  # 低波动市场
            stop_loss_multiplier = 2.0
            take_profit_multiplier = 3.0
        
        # 趋势调整
        if abs(trend_strength) > 0.5:  # 强趋势
            # 顺势持仓更久
            take_profit_multiplier *= 1.5
        
        return {
            'stop_loss_mult': stop_loss_multiplier,
            'take_profit_mult': take_profit_multiplier,
            'trailing_stop': volatility * 2  # 移动止损
        }
    
    # ================== 6. 模型集成优化 ==================
    def optimize_ensemble_weights(self, model_performances: Dict) -> Dict:
        """
        基于性能动态调整模型权重
        """
        # 基于验证集性能调整权重
        weights = {}
        total_score = 0
        
        for model, perf in model_performances.items():
            # 使用方向准确率和相关性的组合
            score = perf.get('direction_acc', 0.5) * perf.get('correlation', 0)
            score = max(0, score)  # 确保非负
            weights[model] = score
            total_score += score
        
        # 归一化
        if total_score > 0:
            for model in weights:
                weights[model] /= total_score
        else:
            # 均等权重
            n_models = len(model_performances)
            for model in weights:
                weights[model] = 1.0 / n_models
        
        return weights
    
    # ================== 7. 仓位优化改进 ==================
    def calculate_kelly_position(self, signal: float, win_rate: float, 
                                win_loss_ratio: float, max_position: float = 0.25) -> float:
        """
        使用Kelly公式计算最优仓位
        """
        if win_loss_ratio <= 0 or win_rate <= 0:
            return 0
        
        # Kelly公式: f = (p*b - q) / b
        # p: 胜率, q: 败率, b: 盈亏比
        p = win_rate
        q = 1 - win_rate
        b = win_loss_ratio
        
        kelly_fraction = (p * b - q) / b if b > 0 else 0
        
        # 使用1/4 Kelly（更保守）
        conservative_kelly = kelly_fraction * 0.25
        
        # 根据信号强度调整
        position = conservative_kelly * abs(signal) * 2
        position = np.sign(signal) * min(abs(position), max_position)
        
        return position
    
    # ================== 8. 实时性能监控 ==================
    def create_performance_monitor(self) -> Dict:
        """
        创建性能监控指标
        """
        return {
            'rolling_sharpe': [],  # 滚动夏普
            'rolling_drawdown': [],  # 滚动回撤
            'win_rate_30d': [],  # 30天胜率
            'avg_trade_pnl': [],  # 平均交易盈亏
            'regime_performance': {}  # 不同市场状态下的表现
        }

# ================== 集成到主策略的函数 ==================
def apply_optimizations_to_strategy(strategy_instance):
    """
    将优化应用到策略实例
    """
    optimizer = StrategyOptimizationSuite()
    
    # 1. 优化信号生成器
    if hasattr(strategy_instance, 'signal_generator'):
        strategy_instance.signal_generator = optimizer.enhance_signal_quality(
            strategy_instance.signal_generator
        )
    
    # 2. 优化风险管理器（如果存在）
    if hasattr(strategy_instance, 'backtest_engine'):
        risk_manager = NoFutureRiskManager()
        risk_manager = optimizer.optimize_trading_costs(risk_manager)
        # 可以将优化后的风险管理器应用到回测引擎
    
    # 3. 替换目标变量生成函数
    import types
    strategy_instance.create_improved_target = types.MethodType(
        optimizer.create_improved_target_variable, strategy_instance
    )
    
    logger.info("Strategy optimizations applied successfully")
    return strategy_instance


# ================== 优化后的参数配置 ==================
class OptimizedStrategyConfig:
    """优化后的策略参数配置"""
    
    # 模型参数（更保守）
    MODEL_PARAMS = {
        'xgboost': {
            'max_depth': 4,  # 减少过拟合
            'learning_rate': 0.03,  # 更小的学习率
            'n_estimators': 200,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'min_child_weight': 5,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'gamma': 0.2
        },
        'lightgbm': {
            'num_leaves': 20,  # 减少复杂度
            'learning_rate': 0.03,
            'feature_fraction': 0.7,
            'bagging_fraction': 0.7,
            'min_data_in_leaf': 30,
            'lambda_l1': 0.1,
            'lambda_l2': 1.0
        },
        'catboost': {
            'iterations': 200,
            'learning_rate': 0.03,
            'depth': 4,
            'l2_leaf_reg': 5,
            'min_data_in_leaf': 30
        }
    }
    
    # 交易参数
    TRADING_PARAMS = {
        'min_signal_strength': 0.1,  # 最小信号强度
        'max_position_size': 0.15,  # 最大仓位
        'max_correlation_between_positions': 0.7,  # 持仓相关性限制
        'min_holding_periods': 10,  # 最小持有期
        'max_daily_trades': 5,  # 每日最大交易
        'commission_rate': 0.0003,  # 考虑滑点的实际成本
    }
    
    # 风险参数
    RISK_PARAMS = {
        'max_drawdown_limit': 0.15,  # 最大回撤限制
        'position_reduce_on_drawdown': 0.5,  # 回撤时减仓比例
        'volatility_scaling': True,  # 波动率缩放
        'correlation_limit': 0.8,  # 相关性限制
    }
def run_optimized_strategy(data_path: str, info_path: str):
    """
    运行优化后的策略
    """
    
    # 创建策略实例
    strategy = NoFutureCommodityStrategy(
        data_path=data_path,
        info_path=info_path,
        use_gpu=True,
        use_optuna=True,
        optuna_trials=100  # 增加优化次数
    )
    
    # 应用优化
    strategy = apply_optimizations_to_strategy(strategy)
    
    # 应用优化后的参数
    config = OptimizedStrategyConfig()
    if hasattr(strategy, 'model_trainer'):
        strategy.model_trainer.xgb_params.update(config.MODEL_PARAMS['xgboost'])
        strategy.model_trainer.lgb_params.update(config.MODEL_PARAMS['lightgbm'])
        strategy.model_trainer.cb_params.update(config.MODEL_PARAMS['catboost'])
    
    # 运行策略
    results = strategy.run_full_strategy(
        max_symbols=10,  # 减少品种数量，提高质量
        look_ahead=3,
        start_date='2015-01-01',
        end_date='2024-06-30',
        sample_rate=1.0
    )
    
    return results
def integrate_rrl_to_strategy(strategy_instance, train_epochs=3, use_transformer=False):
    """
    完整集成RRL神经网络到策略
    
    Args:
        strategy_instance: 主策略实例
        train_epochs: 训练轮数
        use_transformer: 是否使用Transformer架构
    """
    # 1. 检查数据可用性
    if not hasattr(strategy_instance, 'features_data') or not strategy_instance.features_data:
        logger.error("No features data available for RRL training")
        return None
    
    # 2. 获取特征维度
    first_symbol = list(strategy_instance.features_data.keys())[0]
    _, feature_cols = strategy_instance.features_data[first_symbol]
    feature_num = len(feature_cols)
    
    # 3. 创建神经网络生成器
    nn_generator = NeuralNetworkPositionGenerator(
        feature_num=feature_num,
        use_transformer=use_transformer
    )
    
    # 4. 训练神经网络
    if hasattr(strategy_instance, 'min30_data'):
        logger.info(f"Training RRL with {train_epochs} epochs...")
        success = nn_generator.train(
            strategy_instance.features_data,
            strategy_instance.min30_data,
            epochs=train_epochs
        )
        
        if success:
            logger.info("RRL training successful!")
            
            # 5. 创建混合信号生成器
            original_generate = strategy_instance.signal_generator.generate_signals_from_features
            
            def hybrid_signal_generator(features_df, feature_cols, df, symbol, is_realtime=False):
                # 获取原始ML信号
                ml_signals = original_generate(features_df, feature_cols, df, symbol, is_realtime)
                
                # 获取神经网络仓位
                nn_positions = nn_generator.generate_positions(features_df, feature_cols)
                
                # 混合策略：结合ML和RRL
                if len(ml_signals) == len(nn_positions):
                    # 动态权重：根据市场状态调整
                    volatility = features_df['volatility_20'].iloc[-1] if 'volatility_20' in features_df.columns else 0.02
                    
                    if volatility > 0.03:  # 高波动时更依赖RRL
                        final_signals = nn_positions * 0.7 + ml_signals * 0.3
                    else:  # 低波动时平衡两者
                        final_signals = nn_positions * 0.5 + ml_signals * 0.5
                else:
                    final_signals = ml_signals
                
                # 应用安全延迟
                final_signals = final_signals.shift(2).fillna(0)
                
                if is_realtime:
                    final_signals.iloc[-5:] = 0
                
                return final_signals
            
            # 替换信号生成方法
            strategy_instance.signal_generator.generate_signals_from_features = hybrid_signal_generator
            
            return nn_generator
    
    return None
# 修复 OptimizedModelArchitecture 的属性问题
def patch_optimized_architecture():
    OptimizedModelArchitecture.selected_features = []
    OptimizedModelArchitecture.scaler_fitted = False
    
    original_init = OptimizedModelArchitecture.__init__
    def patched_init(self, use_gpu=False):
        original_init(self, use_gpu)
        self.selected_features = []
        self.scaler_fitted = False
    OptimizedModelArchitecture.__init__ = patched_init
    
    # 同时修复 train 方法
    original_train = ImprovedEndToEndStrategy.train
    def patched_train(self, train_data):
        result = original_train(self, train_data)
        if result and hasattr(self, 'model'):
            self.model.selected_features = self.model.feature_cols
            self.model.scaler_fitted = True
        return result
    ImprovedEndToEndStrategy.train = patched_train

# 在运行策略前调用
patch_optimized_architecture()
def apply_critical_fixes(strategy_instance):
    """
    应用关键修复补丁
    """
    # 修复1: 为 OptimizedModelArchitecture 添加预测方法
    def predict_ensemble(self, X):
        """为 OptimizedModelArchitecture 添加预测方法"""
        if hasattr(self, 'improved_strategy') and hasattr(self.improved_strategy, 'model'):
            return self.improved_strategy.model.predict_positions(X)
        else:
            logger.warning("Improved strategy not available, returning zeros")
            return np.zeros(X.shape[0])
    
    # 将方法添加到实例
    if hasattr(strategy_instance, 'model_trainer'):
        strategy_instance.model_trainer.predict_ensemble = predict_ensemble.__get__(
            strategy_instance.model_trainer, type(strategy_instance.model_trainer)
        )
    
    logger.info("Critical fixes applied successfully")
    return strategy_instance
class TradingFrequencyOptimizer:
    def __init__(self):
        self.signal_threshold = 0.15  # 提高到15%
        self.min_holding_period = 20  # 增加到10小时
        self.max_daily_trades = 3  # 降到每天3次
        
    def filter_signals(self, signals, confidence_scores):
        """只保留高置信度信号"""
        # 动态阈值：根据市场状态调整
        volatility = self.calculate_market_volatility()
        dynamic_threshold = max(0.15, volatility * 3)
        
        # 只交易top信号
        filtered = signals.where(
            (abs(signals) > dynamic_threshold) & 
            (confidence_scores > 0.7)
        )
        return filtered
class AdaptivePositionSizer:
    def __init__(self):
        self.base_position = 0.05
        self.max_position = 0.15
        
    def calculate_position(self, signal, market_conditions):
        """计算自适应仓位"""
        # 获取市场条件
        win_rate = market_conditions.get('historical_win_rate', 0.5)
        win_loss_ratio = market_conditions.get('win_loss_ratio', 1.5)
        volatility = market_conditions.get('volatility', 0.02)
        
        # Kelly公式计算
        if win_loss_ratio > 0:
            kelly = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
            kelly_fraction = max(0, min(kelly * 0.25, 0.15))  # 1/4 Kelly，更保守
        else:
            kelly_fraction = 0.05
        
        # 波动率调整
        vol_scalar = min(1, 0.02 / volatility) if volatility > 0 else 1
        
        # 信号强度调整
        signal_scalar = min(abs(signal) * 2, 1)
        
        # 计算最终仓位
        position = kelly_fraction * vol_scalar * signal_scalar * np.sign(signal)
        
        return np.clip(position, -self.max_position, self.max_position)
class CostAwareTrading:
    def __init__(self):
        self.commission = 0.0003
        self.slippage = 0.0002
        self.min_profit_threshold = 0.002  # 0.2%最小利润
        
    def should_trade(self, expected_return, position_change):
        """基于成本决定是否交易"""
        total_cost = (self.commission + self.slippage) * 2
        
        # 预期收益必须覆盖成本的3倍
        if expected_return < total_cost * 3:
            return False
            
        # 仓位变化必须足够大
        if abs(position_change) < 0.1:
            return False
            
        return True
def create_ensemble_strategy():
    """创建多策略集成"""
    
    strategies = {
        'momentum': MomentumStrategy(),
        'mean_reversion': MeanReversionStrategy(), 
        'ml_based': MLStrategy(),
        'rrl_neural': RRLStrategy()
    }
    
    # 动态权重分配
    weights = optimize_strategy_weights(strategies, validation_data)
    
    # 投票机制
    final_signal = weighted_voting(strategies, weights)
    
    return final_signal
# ================== 修复补丁 ==================
def apply_critical_fixes_complete():
    """应用所有必要的修复"""
    
    # 修复1: 为 ImprovedRRLModel 添加 init_weights
    def init_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    ImprovedRRLModel.init_weights = init_weights
    
    # 修复2: 为 NoFutureRiskManager 添加 calculate_position_size
    def calculate_position_size(self, signal, volatility, portfolio_value, 
                               recent_returns, positions):
        """兼容性方法"""
        return self.calculate_position_size_optimized(
            signal=signal,
            symbol="",
            volatility=volatility,
            portfolio_value=portfolio_value
        )
    
    NoFutureRiskManager.calculate_position_size = calculate_position_size
    
    # 修复3: 确保 calculate_stop_loss 和 calculate_take_profit 存在
    def calculate_stop_loss(self, entry_price, volatility, position, signal_strength):
        """计算止损价格"""
        stop_distance = volatility * self.stop_loss_multiplier
        if position > 0:
            return entry_price * (1 - stop_distance)
        else:
            return entry_price * (1 + stop_distance)
    
    def calculate_take_profit(self, entry_price, volatility, position, signal_strength):
        """计算止盈价格"""
        profit_distance = volatility * self.take_profit_multiplier
        if position > 0:
            return entry_price * (1 + profit_distance)
        else:
            return entry_price * (1 - profit_distance)
    
    def should_exit_position(self, position, current_price, holding_periods, 
                            exit_signal, volatility):
        """判断是否应该退出仓位"""
        # 持有时间太长
        if holding_periods > 100:
            return True, "timeout"
        
        # 止损
        if position.quantity > 0:
            if current_price < position.stop_loss:
                return True, "stop_loss"
        else:
            if current_price > position.stop_loss:
                return True, "stop_loss"
        
        # 止盈
        if position.quantity > 0:
            if current_price > position.take_profit:
                return True, "take_profit"
        else:
            if current_price < position.take_profit:
                return True, "take_profit"
        
        # 信号反转
        if exit_signal * position.quantity < -0.1:
            return True, "signal_reverse"
        
        return False, ""
    
    NoFutureRiskManager.calculate_stop_loss = calculate_stop_loss
    NoFutureRiskManager.calculate_take_profit = calculate_take_profit
    NoFutureRiskManager.should_exit_position = should_exit_position
    
    print("All critical fixes applied successfully!")
# ================== 交易执行修复补丁 ==================
def apply_complete_trading_fix():
    """
    完整的交易执行修复补丁
    解决0交易问题
    """
    
    print("\n" + "="*60)
    print("应用交易执行修复补丁...")
    print("="*60 + "\n")
    
    # ===== 1. 修复回测引擎的交易执行 =====
    def patch_backtest_engine():
        """修复回测引擎"""
        
        # 保存原始方法
        original_execute = NoFutureBacktestEngine._execute_trade
        
        def fixed_execute_trade(self, symbol: str, target_position: float, 
                               price: float, date: pd.Timestamp, cash: float,
                               risk_manager: NoFutureRiskManager, signal: float,
                               volatility: float) -> Optional[float]:
            """修复后的交易执行"""
            
            # 获取当前持仓
            current_position = self.positions.get(symbol)
            if current_position is None:
                current_quantity = 0
            else:
                current_quantity = current_position.quantity
            
            # === 修改1: 使用更激进的仓位计算 ===
            # 原来: target_value = target_position * self.initial_capital
            # 现在: 增加仓位乘数
            position_multiplier = 5.0  # 放大5倍
            target_value = target_position * self.initial_capital * position_multiplier
            target_quantity = target_value / price
            
            # 计算仓位变化
            quantity_diff = target_quantity - current_quantity
            
            # === 修改2: 降低最小交易金额 ===
            # 原来: if abs(quantity_diff * price) < 500
            MIN_TRADE_AMOUNT = 50  # 从500降到50
            if abs(quantity_diff * price) < MIN_TRADE_AMOUNT:
                return None
            
            # === 修改3: 移除成本感知检查 ===
            # 注释掉过于严格的成本检查
            # if hasattr(self, 'cost_aware_trading'):
            #     if not self.cost_aware_trading.should_trade(expected_return, position_change):
            #         return None
            
            # === 修改4: 简化仓位变化检查 ===
            position_change = abs(quantity_diff * price / self.initial_capital)
            if position_change < 0.0001:  # 从0.001降到0.0001
                return None
            
            # 计算滑点
            is_buy = quantity_diff > 0
            slippage_rate = 0.0001  # 保持原有滑点
            
            if is_buy:
                execution_price = price * (1 + slippage_rate)
            else:
                execution_price = price * (1 - slippage_rate)
            
            # 计算交易金额和手续费
            trade_amount = abs(quantity_diff * execution_price)
            commission = trade_amount * self.commission_rate
            
            # 检查买入资金是否充足
            if is_buy:
                required_cash = trade_amount + commission
                if required_cash > cash * 0.98:  # 从0.95提高到0.98
                    max_quantity = (cash * 0.98 - commission) / execution_price
                    if max_quantity <= 0:
                        return None
                    quantity_diff = min(quantity_diff, max_quantity)
                    trade_amount = abs(quantity_diff * execution_price)
                    commission = trade_amount * self.commission_rate
            
            # === 修改5: 降低最小数量要求 ===
            if abs(quantity_diff) < 0.00001:  # 从0.001降到0.00001
                return None
            
            # 执行交易
            if is_buy:
                new_cash = cash - (trade_amount + commission)
            else:
                new_cash = cash + (trade_amount - commission)
            
            # 计算滑点成本
            slippage_cost = abs(execution_price - price) * abs(quantity_diff)
            
            # 更新或创建持仓（简化逻辑）
            if symbol not in self.positions:
                if abs(target_quantity) > 0.00001:
                    self.positions[symbol] = Position(
                        symbol=symbol,
                        quantity=target_quantity,
                        entry_price=execution_price,
                        entry_time=date,
                        current_price=execution_price,
                        unrealized_pnl=0,
                        stop_loss=execution_price * 0.95,  # 简化止损
                        take_profit=execution_price * 1.05,  # 简化止盈
                        trailing_stop=execution_price * 0.95,
                        highest_price=execution_price,
                        lowest_price=execution_price
                    )
                    logger.debug(f"{symbol}: 开仓 - 数量: {target_quantity:.4f}, 价格: {execution_price:.4f}")
            else:
                # 更新现有持仓
                old_position = self.positions[symbol]
                new_quantity = old_position.quantity + quantity_diff
                
                if abs(new_quantity) < 0.00001:
                    # 平仓
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
                    logger.debug(f"{symbol}: 平仓 - PnL: {pnl:.2f}")
                else:
                    # 调整仓位
                    self.positions[symbol].quantity = new_quantity
                    if quantity_diff * old_position.quantity > 0:
                        # 同向加仓
                        total_cost = old_position.entry_price * abs(old_position.quantity) + \
                                   execution_price * abs(quantity_diff)
                        self.positions[symbol].entry_price = total_cost / abs(new_quantity)
            
            return new_cash
        
        # 应用补丁
        NoFutureBacktestEngine._execute_trade = fixed_execute_trade
        print("✓ 回测引擎交易执行已修复")
    
    # ===== 2. 修复风险管理器 =====
    def patch_risk_manager():
        """修复风险管理器"""
        
        # 修改风险管理参数
        NoFutureRiskManager.max_position = 0.5  # 从0.2提高到0.5
        NoFutureRiskManager.max_leverage = 3.0  # 从1.5提高到3.0
        NoFutureRiskManager.min_holding_period = 1  # 从3降到1
        NoFutureRiskManager.max_daily_trades = 50  # 从15提高到50
        
        # 简化仓位计算
        def simple_position_size(self, signal, volatility, portfolio_value, 
                                recent_returns, positions):
            """简化的仓位计算"""
            # 直接使用信号强度
            position = signal * 0.5  # 信号的50%
            return np.clip(position, -0.5, 0.5)
        
        NoFutureRiskManager.calculate_position_size = simple_position_size
        
        # 简化退出条件
        def simple_should_exit(self, position, current_price, holding_periods, 
                              exit_signal, volatility):
            """简化的退出条件"""
            # 只在信号强烈反转时退出
            if exit_signal * position.quantity < -0.5:
                return True, "signal_reverse"
            # 或持有时间过长
            if holding_periods > 200:
                return True, "timeout"
            return False, ""
        
        NoFutureRiskManager.should_exit_position = simple_should_exit
        
        print("✓ 风险管理器已简化")
    
    # ===== 3. 修复信号强度 =====
    def amplify_signals(signals_dict, multiplier=10.0):
        """放大信号强度"""
        amplified = {}
        
        for symbol, signal in signals_dict.items():
            # 统计原始信号
            original_nonzero = (signal != 0).sum()
            original_mean = signal[signal != 0].mean() if original_nonzero > 0 else 0
            
            # 放大信号
            amplified_signal = signal * multiplier
            
            # 限制范围
            amplified_signal = amplified_signal.clip(-1.0, 1.0)
            
            # 过滤极小信号
            amplified_signal = amplified_signal.where(abs(amplified_signal) > 0.01, 0)
            
            amplified[symbol] = amplified_signal
            
            # 统计放大后
            final_nonzero = (amplified_signal != 0).sum()
            final_mean = amplified_signal[amplified_signal != 0].mean() if final_nonzero > 0 else 0
            
            print(f"  {symbol}: {original_nonzero} signals (avg={original_mean:.4f}) -> "
                  f"{final_nonzero} signals (avg={final_mean:.4f})")
        
        print(f"✓ 信号已放大 {multiplier}x")
        return amplified
    
    # ===== 4. 应用所有补丁 =====
    patch_backtest_engine()
    patch_risk_manager()
    
    # 返回信号放大函数
    return amplify_signals


# ================== 使用补丁的主函数修改 ==================
def run_strategy_with_fix():
    """运行策略时应用修复"""
    
    # 创建策略实例
    strategy = NoFutureCommodityStrategy(
        data_path='~/autodl-tmp/data/1m/',
        info_path='~/autodl-tmp/data/info.csv',
        use_gpu=True,
        use_improved_components=True
    )
    
    # 应用所有基础修复
    apply_critical_fixes_complete()
    patch_optimized_architecture()
    strategy = apply_critical_fixes(strategy)
    
    # ===== 应用交易执行修复 =====
    amplify_signals = apply_complete_trading_fix()
    
    # 修改策略的run_full_strategy方法
    original_run = strategy.run_full_strategy
    
    def run_with_signal_amplification(*args, **kwargs):
        # 先运行原始策略
        strategy._original_run = original_run
        
        # 在内部修改信号生成后的处理
        original_backtest = strategy.backtest_engine.run_backtest
        
        def run_backtest_with_amplification(signals_dict, price_data_dict, 
                                           start_date=None, end_date=None):
            # 放大信号
            print("\n" + "="*60)
            print("放大交易信号...")
            amplified_signals = amplify_signals(signals_dict, multiplier=10.0)
            print("="*60 + "\n")
            
            # 运行原始回测
            return original_backtest(amplified_signals, price_data_dict, 
                                    start_date, end_date)
        
        strategy.backtest_engine.run_backtest = run_backtest_with_amplification
        
        # 运行策略
        return original_run(*args, **kwargs)
    
    strategy.run_full_strategy = run_with_signal_amplification
    
    # 运行策略
    results = strategy.run_full_strategy(
        max_symbols=8,
        use_neural_network=True,
        enable_multiscale=True
    )
    
    return results

# ================== 简化的 __main__ 代码 ==================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("COMMODITY FUTURES QUANTITATIVE STRATEGY")
    print("="*60)

    # 最优配置参数
    config = {
        'data_path': '~/autodl-tmp/data/1m/',
        'info_path': '~/autodl-tmp/data/info.csv',
        'max_symbols': 8,  # 8个品种平衡计算效率和分散化
        'start_date': '2020-01-01',  # 近期数据更有代表性
        'end_date': '2024-06-30',
        'use_gpu': True,  # 使用GPU加速
        'use_optuna': False,  # 关闭Optuna以加快速度
        'optuna_trials': 30,
        'use_neural_network': True,  # 启用神经网络
        'use_improved': True,  # 使用改进组件
        'enable_multiscale': True,  # 启用多时间尺度
        'sample_rate': 0.8  # 80%采样率，平衡数据量和计算速度
    }

    print(f"\nOptimized Configuration:")
    print(f"  Data Path: {config['data_path']}")
    print(f"  Max Symbols: {config['max_symbols']}")
    print(f"  Date Range: {config['start_date']} to {config['end_date']}")
    print(f"  GPU: {'Enabled' if config['use_gpu'] else 'Disabled'}")
    print(f"  Optuna: {'Enabled' if config['use_optuna'] else 'Disabled'}")
    print(f"  Neural Network: {'Enabled' if config['use_neural_network'] else 'Disabled'}")
    print(f"  Improved Components: {'Enabled' if config['use_improved'] else 'Disabled'}")
    print(f"  Multi-scale Features: {'Enabled' if config['enable_multiscale'] else 'Disabled'}")
    print(f"  Sample Rate: {config['sample_rate']*100:.0f}%")
    print("\n" + "="*60 + "\n")

    
    try:
        # 应用补丁修复

        apply_critical_fixes_complete()
        patch_optimized_architecture()
            
        # 创建策略实例
        
        results = run_strategy_with_fix()
        # 打印最终结果摘要
        if results:
            print("\n" + "="*60)
            print("STRATEGY EXECUTION COMPLETED SUCCESSFULLY!")
            print("="*60)
            print(f"\nKey Performance Metrics:")
            print(f"  Total Return: {results.get('total_return', 0):.2%}")
            print(f"  Annual Return: {results.get('annual_return', 0):.2%}")
            print(f"  Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
            print(f"  Max Drawdown: {results.get('max_drawdown', 0):.2%}")
            print(f"  Win Rate: {results.get('win_rate', 0):.2%}")
            print(f"  Total Trades: {results.get('total_trades', 0)}")

            if results.get('total_trades', 0) > 0:
                print(f"\nTrade Analysis:")
                print(f"  Profit Factor: {results.get('profit_factor', 0):.2f}")
                print(f"  Avg Position: {results.get('avg_positions', 0):.2f}")

                if 'exit_reasons' in results:
                    print(f"\nExit Reasons:")
                    for reason, count in results['exit_reasons'].items():
                        print(f"    {reason}: {count}")

            print("\n" + "="*60)
        else:
            print("\n" + "="*60)
            print("STRATEGY EXECUTION FAILED")
            print("Please check the logs for details")
            print("="*60)

    except KeyboardInterrupt:
        print("\n\nStrategy execution interrupted by user")
    except Exception as e:
        print(f"\n\nError running strategy: {e}")
        traceback.print_exc()
