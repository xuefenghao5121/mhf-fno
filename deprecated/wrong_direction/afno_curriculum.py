"""
AFNO Frequency Sparsity Curriculum (AFNO-FS)
===========================================

AFNO 频率稀疏度课程学习实现。

核心思想:
    通过渐进式调整频率稀疏度，实现从低频主导到高频精细的
    自适应训练过程。

训练阶段:
    1. 早期阶段: 高稀疏度，保留低频模式，快速捕捉大尺度特征
    2. 中期阶段: 中等稀疏度，引入中频模式
    3. 后期阶段: 低稀疏度，引入高频模式，精细调整

参考文献:
    - AFNO: Adaptive Fourier Neural Operators (arXiv:2111.13587)
    - Curriculum Learning: A Survey (arXiv:2106.03957)

版本: 1.0.0
作者: Tianyuan Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Callable, Dict, List
import math


class FrequencySparsityScheduler:
    """
    频率稀疏度调度器
    
    根据训练进度动态调整稀疏度阈值/比例
    
    支持的调度策略:
        - linear: 线性递减
        - root: 根号递减（前期快，后期慢）
        - exp: 指数递减（前期慢，后期快）
        - cosine: 余弦调度
        - step: 阶梯式递减
        - adaptive: 基于损失的自适应调整
    """
    
    def __init__(
        self,
        scheduler_type: str = 'linear',
        max_sparsity_ratio: float = 0.8,  # 早期：保留 20% 频率
        min_sparsity_ratio: float = 0.1,  # 后期：保留 90% 频率
        total_epochs: int = 100,
        warmup_epochs: int = 10,
        step_schedule: Optional[List[int]] = None,  # 阶梯式调度
        adaptive_threshold: float = 0.01,  # 自适应调整的损失变化阈值
        patience: int = 5  # 自适应调整的耐心值
    ):
        """
        Args:
            scheduler_type: 调度器类型 ('linear', 'root', 'exp', 'cosine', 'step', 'adaptive')
            max_sparsity_ratio: 早期最大稀疏度比例（0-1）
            min_sparsity_ratio: 后期最小稀疏度比例（0-1）
            total_epochs: 总训练 epoch 数
            warmup_epochs: 预热 epoch 数（保持最大稀疏度）
            step_schedule: 阶梯式调度边界（仅用于 step 模式）
            adaptive_threshold: 自适应调整的损失变化阈值
            patience: 自适应调整的耐心值
        """
        self.scheduler_type = scheduler_type
        self.max_sparsity_ratio = max_sparsity_ratio
        self.min_sparsity_ratio = min_sparsity_ratio
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.step_schedule = step_schedule or [30, 60, 80]
        self.adaptive_threshold = adaptive_threshold
        self.patience = patience
        
        # 自适应调度状态
        self.best_loss = float('inf')
        self.wait_count = 0
        self.current_sparsity_ratio = max_sparsity_ratio
        
        # 历史记录
        self.history = {
            'epochs': [],
            'sparsity_ratios': [],
            'losses': []
        }
    
    def get_sparsity_ratio(self, epoch: int, loss: Optional[float] = None) -> float:
        """
        获取当前 epoch 的稀疏度比例
        
        Args:
            epoch: 当前 epoch
            loss: 当前损失（仅用于 adaptive 模式）
        
        Returns:
            float: 稀疏度比例 (0-1)
        """
        # 预热阶段
        if epoch < self.warmup_epochs:
            sparsity_ratio = self.max_sparsity_ratio
        else:
            # 调整进度 (0-1)
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            progress = min(progress, 1.0)
            
            # 根据调度器类型计算稀疏度
            if self.scheduler_type == 'linear':
                sparsity_ratio = self._linear_schedule(progress)
            elif self.scheduler_type == 'root':
                sparsity_ratio = self._root_schedule(progress)
            elif self.scheduler_type == 'exp':
                sparsity_ratio = self._exp_schedule(progress)
            elif self.scheduler_type == 'cosine':
                sparsity_ratio = self._cosine_schedule(progress)
            elif self.scheduler_type == 'step':
                sparsity_ratio = self._step_schedule(epoch)
            elif self.scheduler_type == 'adaptive':
                sparsity_ratio = self._adaptive_schedule(epoch, loss)
            else:
                raise ValueError(f"不支持的调度器类型: {self.scheduler_type}")
        
        # 记录历史
        self.history['epochs'].append(epoch)
        self.history['sparsity_ratios'].append(sparsity_ratio)
        if loss is not None:
            self.history['losses'].append(loss)
        
        self.current_sparsity_ratio = sparsity_ratio
        return sparsity_ratio
    
    def _linear_schedule(self, progress: float) -> float:
        """线性调度: max -> min"""
        return self.max_sparsity_ratio + (self.min_sparsity_ratio - self.max_sparsity_ratio) * progress
    
    def _root_schedule(self, progress: float) -> float:
        """根号调度: 前期快，后期慢"""
        return self.max_sparsity_ratio + (self.min_sparsity_ratio - self.max_sparsity_ratio) * math.sqrt(progress)
    
    def _exp_schedule(self, progress: float) -> float:
        """指数调度: 前期慢，后期快"""
        return self.max_sparsity_ratio + (self.min_sparsity_ratio - self.max_sparsity_ratio) * (1 - math.exp(-3 * progress))
    
    def _cosine_schedule(self, progress: float) -> float:
        """余弦调度: 平滑过渡"""
        return self.max_sparsity_ratio + (self.min_sparsity_ratio - self.max_sparsity_ratio) * (1 - math.cos(progress * math.pi)) / 2
    
    def _step_schedule(self, epoch: int) -> float:
        """阶梯式调度"""
        if epoch < self.step_schedule[0]:
            return self.max_sparsity_ratio
        elif epoch < self.step_schedule[1]:
            return 0.5
        elif epoch < self.step_schedule[2]:
            return 0.3
        else:
            return self.min_sparsity_ratio
    
    def _adaptive_schedule(self, epoch: int, loss: Optional[float] = None) -> float:
        """自适应调度: 基于损失变化调整"""
        if loss is None:
            return self.current_sparsity_ratio
        
        # 记录损失
        if loss < self.best_loss - self.adaptive_threshold:
            self.best_loss = loss
            self.wait_count = 0
        else:
            self.wait_count += 1
        
        # 如果损失停滞，降低稀疏度（保留更多频率）
        if self.wait_count >= self.patience:
            self.wait_count = 0
            self.current_sparsity_ratio = max(
                self.current_sparsity_ratio * 0.8,
                self.min_sparsity_ratio
            )
        
        return self.current_sparsity_ratio
    
    def reset(self):
        """重置调度器状态"""
        self.best_loss = float('inf')
        self.wait_count = 0
        self.current_sparsity_ratio = self.max_sparsity_ratio
        self.history = {
            'epochs': [],
            'sparsity_ratios': [],
            'losses': []
        }


class AFNOWithCurriculum(nn.Module):
    """
    带频率稀疏度课程的 AFNO 模型
    
    在训练过程中动态调整频率稀疏度，实现渐进式频谱学习。
    
    训练流程:
        1. 初始化: 创建 AFNO 模型和频率稀疏度调度器
        2. 每个 epoch: 根据调度器获取当前稀疏度比例
        3. 前向传播: 使用当前稀疏度比例进行前向传播
        4. 记录稀疏度变化: 用于分析和可视化
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        n_layers: int,
        n_modes: Tuple[int, ...],
        num_blocks: int = 8,
        max_sparsity_ratio: float = 0.8,
        min_sparsity_ratio: float = 0.1,
        total_epochs: int = 100,
        scheduler_type: str = 'linear',
        warmup_epochs: int = 10,
        init_sparsity_threshold: float = 0.01,
        activation: str = 'gelu'
    ):
        """
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            hidden_channels: 隐藏层通道数
            n_layers: FNO 层数
            n_modes: 频率模式数
            num_blocks: block-diagonal 块数
            max_sparsity_ratio: 早期最大稀疏度比例
            min_sparsity_ratio: 后期最小稀疏度比例
            total_epochs: 总训练 epoch 数
            scheduler_type: 调度器类型
            warmup_epochs: 预热 epoch 数
            init_sparsity_threshold: 初始稀疏度阈值
            activation: 激活函数
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.n_modes = n_modes
        self.num_blocks = num_blocks
        
        # 输入投影
        self.fc0 = nn.Linear(in_channels, hidden_channels)
        
        # 导入 AFNO 层
        from .afno import AFNOBlockDiagonalConv
        
        # AFNO 层
        self.afno_layers = nn.ModuleList([
            AFNOBlockDiagonalConv(
                hidden_channels,
                hidden_channels,
                n_modes,
                num_blocks=num_blocks,
                init_sparsity_threshold=init_sparsity_threshold
            )
            for _ in range(n_layers)
        ])
        
        # 激活函数
        if activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'relu':
            self.activation = F.relu
        else:
            raise ValueError(f"不支持的激活函数: {activation}")
        
        # 输出投影
        self.fc1 = nn.Linear(hidden_channels, out_channels)
        
        # 频率稀疏度调度器
        self.scheduler = FrequencySparsityScheduler(
            scheduler_type=scheduler_type,
            max_sparsity_ratio=max_sparsity_ratio,
            min_sparsity_ratio=min_sparsity_ratio,
            total_epochs=total_epochs,
            warmup_epochs=warmup_epochs
        )
        
        # 训练模式标志
        self.is_training_mode = False
        self.current_epoch = 0
        self.current_sparsity_ratio = max_sparsity_ratio
    
    def set_training_mode(self, epoch: int, training: bool = True):
        """
        设置训练模式并更新当前稀疏度
        
        Args:
            epoch: 当前 epoch
            training: 是否为训练模式
        """
        self.is_training_mode = training
        self.current_epoch = epoch
        
        if training:
            self.current_sparsity_ratio = self.scheduler.get_sparsity_ratio(epoch)
        else:
            # 推理模式使用最小稀疏度（保留最多频率）
            self.current_sparsity_ratio = self.scheduler.min_sparsity_ratio
    
    def forward(
        self,
        x: torch.Tensor,
        sparsity_ratio: Optional[float] = None,
        loss: Optional[float] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量
            sparsity_ratio: 稀疏度比例（覆盖当前值）
            loss: 当前损失（用于自适应调度）
        
        Returns:
            torch.Tensor: 输出张量
        """
        # 确定稀疏度比例
        if sparsity_ratio is None:
            if self.is_training_mode and self.scheduler.scheduler_type == 'adaptive':
                # 自适应调度: 需要损失信息
                sparsity_ratio = self.scheduler.get_sparsity_ratio(self.current_epoch, loss)
            else:
                sparsity_ratio = self.current_sparsity_ratio
        
        # 输入投影
        if x.dim() == 3:
            x = self.fc0(x.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            x = self.fc0(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        # AFNO 层
        for afno_layer in self.afno_layers:
            x = afno_layer(x, sparsity_ratio=sparsity_ratio)
            x = self.activation(x)
        
        # 输出投影
        if x.dim() == 3:
            x = self.fc1(x.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            x = self.fc1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        return x
    
    def get_sparsity_history(self) -> Dict:
        """
        获取稀疏度历史记录
        
        Returns:
            dict: 稀疏度历史
        """
        return self.scheduler.history
    
    def get_current_sparsity_ratio(self) -> float:
        """
        获取当前稀疏度比例
        
        Returns:
            float: 当前稀疏度比例
        """
        return self.current_sparsity_ratio
    
    def get_sparsity_info(self, x: torch.Tensor) -> List[Dict]:
        """
        获取所有 AFNO 层的稀疏化信息
        
        Args:
            x: 输入张量
        
        Returns:
            List[dict]: 每层的稀疏化信息
        """
        sparsity_info = []
        for i, afno_layer in enumerate(self.afno_layers):
            info = afno_layer.get_sparsity_info(x)
            info['layer'] = i
            sparsity_info.append(info)
        return sparsity_info


def create_afno_with_curriculum(
    in_channels: int = 1,
    out_channels: int = 1,
    hidden_channels: int = 64,
    n_layers: int = 4,
    n_modes: Tuple[int, ...] = (12, 12),
    num_blocks: int = 8,
    max_sparsity_ratio: float = 0.8,
    min_sparsity_ratio: float = 0.1,
    total_epochs: int = 100,
    scheduler_type: str = 'linear',
    warmup_epochs: int = 10
) -> AFNOWithCurriculum:
    """
    创建带频率稀疏度课程的 AFNO 模型（工厂函数）
    
    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        hidden_channels: 隐藏层通道数
        n_layers: FNO 层数
        n_modes: 频率模式数
        num_blocks: block-diagonal 块数
        max_sparsity_ratio: 早期最大稀疏度比例
        min_sparsity_ratio: 后期最小稀疏度比例
        total_epochs: 总训练 epoch 数
        scheduler_type: 调度器类型
        warmup_epochs: 预热 epoch 数
    
    Returns:
        AFNOWithCurriculum: AFNO 模型
    """
    return AFNOWithCurriculum(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=hidden_channels,
        n_layers=n_layers,
        n_modes=n_modes,
        num_blocks=num_blocks,
        max_sparsity_ratio=max_sparsity_ratio,
        min_sparsity_ratio=min_sparsity_ratio,
        total_epochs=total_epochs,
        scheduler_type=scheduler_type,
        warmup_epochs=warmup_epochs
    )


__all__ = [
    'FrequencySparsityScheduler',
    'AFNOWithCurriculum',
    'create_afno_with_curriculum'
]
