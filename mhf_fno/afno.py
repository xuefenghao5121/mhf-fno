"""
AFNO: Adaptive Fourier Neural Operator
=======================================

Adaptive Fourier Neural Operators with block-diagonal spectral convolution
and frequency sparsification.

核心特性:
    1. Block-Diagonal Spectral Convolution: 参数高效的多头频谱处理
    2. Frequency Sparsification: 自适应高频噪声过滤
    3. Adaptive Weight Sharing: 动态权重共享机制

参考文献:
    Adaptive Fourier Neural Operators: Efficient Token Mixers for Transformers
    arXiv:2111.13587

版本: 1.0.0
作者: Tianyuan Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Union, List
import warnings
import math


class AFNOBlockDiagonalConv(nn.Module):
    """
    AFNO Block-Diagonal Spectral Convolution
    
    将频谱空间的权重矩阵分解为 block-diagonal 结构，显著减少参数量。
    
    数学表达:
        传统 FNO: R ∈ C^{c×c×k}
        AFNO: R = diag(R_1, R_2, ..., R_b), R_i ∈ C^{c/b × c/b ×k}
    
    参数减少:
        从 O(c²k) 降低到 O(c²k/b)
    
    特性:
        - Block-diagonal 权重分解
        - 频率稀疏化（软阈值/硬阈值）
        - 自适应权重共享（门控机制）
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: Tuple[int, ...],
        num_blocks: int = 8,
        bias: bool = True,
        init_sparsity_threshold: float = 0.01,
        enable: bool = True
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes = n_modes
        self.num_blocks = num_blocks
        self.enable = enable
        
        # 检查通道数是否能被块数整除
        if in_channels % num_blocks != 0 or out_channels % num_blocks != 0:
            warnings.warn(
                f"通道数 ({in_channels}, {out_channels}) 不能被 num_blocks ({num_blocks}) 整除",
                UserWarning
            )
            self.enable = False
        
        if self.enable:
            self.block_size = in_channels // num_blocks
            assert self.block_size == out_channels // num_blocks
            
            # 根据 n_modes 的长度确定权重形状
            if len(n_modes) == 1:
                # 1D 情况
                weight_shape = (num_blocks, self.block_size, self.block_size, n_modes[0])
            else:
                # 2D 情况 (rfft2 后最后一维是 W//2+1)
                modes_y = n_modes[1]  # 不使用 +1，rfft2 已经处理了
                weight_shape = (num_blocks, self.block_size, self.block_size, n_modes[0], modes_y)
            
            # Xavier 风格初始化
            init_std = (2 / (in_channels + out_channels)) ** 0.5
            self.weight = nn.Parameter(
                torch.randn(*weight_shape, dtype=torch.cfloat) * init_std
            )
            
            # 可学习的稀疏化阈值
            self.sparsity_threshold = nn.Parameter(
                torch.tensor(init_sparsity_threshold)
            )
            
            # 自适应权重共享门控（每个 block 一个门控）
            self.gates = nn.Parameter(torch.zeros(num_blocks))
        else:
            # 回退到标准权重形状
            modes_y = n_modes[-1] // 2 + 1 if len(n_modes) > 1 else n_modes[-1]
            weight_shape = (in_channels, out_channels, n_modes[0], modes_y) if len(n_modes) > 1 else (in_channels, out_channels, n_modes[0])
            self.weight = nn.Parameter(
                torch.randn(*weight_shape, dtype=torch.cfloat) * 0.01
            )
        
        # 偏置项
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels, *((1,) * len(n_modes))))
        else:
            self.bias = None
    
    def forward(self, x: torch.Tensor, sparsity_ratio: Optional[float] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (B, C, L) 用于 1D 或 (B, C, H, W) 用于 2D
            sparsity_ratio: 稀疏度比例 (0-1)，如果为 None 则使用可学习的阈值
        
        Returns:
            torch.Tensor: 输出张量，形状与输入相同
        """
        if not self.enable:
            return self._forward_standard(x)
        
        if x.dim() == 3:
            return self._forward_1d(x, sparsity_ratio)
        elif x.dim() == 4:
            return self._forward_2d(x, sparsity_ratio)
        else:
            raise ValueError(f"不支持的输入维度: {x.dim()}")
    
    def _forward_standard(self, x: torch.Tensor) -> torch.Tensor:
        """标准前向传播（回退）"""
        if x.dim() == 3:
            x_freq = torch.fft.rfft(x, dim=-1)
            out_freq = torch.einsum('bif,iwf->bof', x_freq[..., :self.n_modes[0]], self.weight)
            x_out = torch.fft.irfft(out_freq, n=x.shape[-1], dim=-1)
        else:
            x_freq = torch.fft.rfft2(x)
            out_freq = torch.einsum('bijy,iwjy->bowy', 
                                   x_freq[..., :self.n_modes[0], :self.n_modes[1]//2+1], 
                                   self.weight)
            x_out = torch.fft.irfft2(out_freq, s=x.shape[-2:])
        
        if self.bias is not None:
            x_out = x_out + self.bias
        
        return x_out
    
    def _forward_1d(self, x: torch.Tensor, sparsity_ratio: Optional[float] = None) -> torch.Tensor:
        """1D 前向传播（优化版）"""
        B, C, L = x.shape
        
        # FFT
        x_freq = torch.fft.rfft(x, dim=-1)
        
        # 计算实际可用的模式数
        n_modes = min(self.n_modes[0], x_freq.shape[-1])
        
        # 重塑为多头格式 (B, n_blocks, block_in, freq_len)
        x_freq = x_freq.view(B, self.num_blocks, self.block_size, -1)
        
        # 预分配输出张量
        out_freq = torch.zeros(
            B, self.num_blocks, self.block_size, x_freq.shape[-1],
            dtype=x_freq.dtype, device=x.device
        )
        
        # 多头频域卷积（仅处理低频部分）
        # einsum: 'bif,iof->bof' 
        # b: batch, i: input channel per block, o: output channel per block, f: frequency
        out_freq[..., :n_modes] = torch.einsum(
            'bif,iof->bof',
            x_freq[..., :n_modes], 
            self.weight[..., :n_modes]
        )
        
        # 合并多头
        out_freq = out_freq.reshape(B, self.out_channels, -1)
        
        # 应用频率稀疏化
        out_freq = self._apply_sparsification(out_freq, sparsity_ratio)
        
        # IFFT
        x_out = torch.fft.irfft(out_freq, n=L, dim=-1)
        
        # 添加偏置
        if self.bias is not None:
            x_out = x_out + self.bias
            
        return x_out
    
    def _forward_2d(self, x: torch.Tensor, sparsity_ratio: Optional[float] = None) -> torch.Tensor:
        """2D 前向传播（优化版）"""
        B, C, H, W = x.shape
        
        # 2D FFT
        x_freq = torch.fft.rfft2(x)
        
        out_freq = torch.zeros_like(x_freq)
        
        n_modes_x = min(self.n_modes[0], x_freq.shape[-2])
        n_modes_y = min(self.n_modes[1], x_freq.shape[-1])
        
        # 重塑为多头格式
        x_blocks = x_freq.chunk(self.num_blocks, dim=1)
        
        # 计算门控权重
        gates = torch.sigmoid(self.gates)
        
        for i, (w, x_b) in enumerate(zip(self.weight, x_blocks)):
            # 截断到实际模式数
            x_b_trunc = x_b[..., :n_modes_x, :n_modes_y]
            # Block-diagonal 乘法
            out_b = torch.einsum('bijy,iojy->bojy', x_b_trunc, w)
            # 应用门控
            out_b = out_b * gates[i]
            # 放回原位置
            out_freq[:, i*self.block_size:(i+1)*self.block_size, :n_modes_x, :n_modes_y] = out_b
        
        # 应用频率稀疏化
        out_freq = self._apply_sparsification(out_freq, sparsity_ratio)
        
        # IFFT
        x_out = torch.fft.irfft2(out_freq, s=(H, W))
        
        # 添加偏置
        if self.bias is not None:
            x_out = x_out + self.bias
        
        return x_out
    
    def _apply_sparsification(self, x_freq: torch.Tensor, sparsity_ratio: Optional[float] = None) -> torch.Tensor:
        """
        应用频率稀疏化
        
        Args:
            x_freq: 频谱张量
            sparsity_ratio: 稀疏化比例 (0-1)
        
        Returns:
            稀疏化后的频谱张量
        """
        if sparsity_ratio is not None:
            # 基于比例的稀疏化：保留 top-p% 的显著频率
            threshold = torch.quantile(x_freq.abs(), 1.0 - sparsity_ratio)
            sparsity_mask = x_freq.abs() > threshold
            return x_freq * sparsity_mask
        else:
            # 使用可学习阈值进行软阈值化
            threshold = F.softplus(self.sparsity_threshold)
            x_mag = x_freq.abs()
            x_sign = torch.sign(x_freq)
            x_sparse = x_sign * F.relu(x_mag - threshold)
            return x_sparse
    
    def get_sparsity_info(self, x: torch.Tensor) -> dict:
        """
        获取稀疏化信息
        
        Args:
            x: 输入张量
        
        Returns:
            dict: 稀疏化信息
        """
        if x.dim() == 3:
            x_freq = torch.fft.rfft(x, dim=-1)
        else:
            x_freq = torch.fft.rfft2(x)
        
        if hasattr(self, 'sparsity_threshold'):
            threshold = F.softplus(self.sparsity_threshold).item()
        else:
            threshold = 0.0
        
        # 计算稀疏度
        sparse_mask = x_freq.abs() > threshold
        sparsity_ratio = 1.0 - sparse_mask.float().mean().item()
        
        return {
            'threshold': threshold,
            'sparsity_ratio': sparsity_ratio,
            'num_blocks': self.num_blocks,
            'block_size': self.block_size
        }


class AFNO(nn.Module):
    """
    AFNO 模型
    
    整合 Block-diagonal Spectral Convolution 和频率稀疏化机制
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        n_layers: int,
        n_modes: Tuple[int, ...],
        num_blocks: int = 8,
        sparsity_ratio: float = 0.5,
        init_sparsity_threshold: float = 0.01,
        activation: str = 'gelu'
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.n_modes = n_modes
        self.num_blocks = num_blocks
        self.sparsity_ratio = sparsity_ratio
        
        # 输入投影
        self.fc0 = nn.Linear(in_channels, hidden_channels)
        
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
    
    def forward(self, x: torch.Tensor, sparsity_ratio: Optional[float] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量
            sparsity_ratio: 稀疏度比例（覆盖当前值）
        
        Returns:
            torch.Tensor: 输出张量
        """
        if sparsity_ratio is None:
            sparsity_ratio = self.sparsity_ratio
        
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
    
    def get_sparsity_info(self, x: torch.Tensor) -> List[dict]:
        """
        获取所有 AFNO 层的稀疏化信息
        
        Args:
            x: 输入张量
        
        Returns:
            List[dict]: 每层的稀疏化化信息
        """
        sparsity_info = []
        for i, afno_layer in enumerate(self.afno_layers):
            info = afno_layer.get_sparsity_info(x)
            info['layer'] = i
            sparsity_info.append(info)
        return sparsity_info


def create_afno(
    in_channels: int = 1,
    out_channels: int = 1,
    hidden_channels: int = 64,
    n_layers: int = 4,
    n_modes: Tuple[int, ...] = (12, 12),
    num_blocks: int = 8,
    sparsity_ratio: float = 0.5
) -> AFNO:
    """
    创建 AFNO 模型（工厂函数）
    
    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        hidden_channels: 隐藏层通道数
        n_layers: FNO 层数
        n_modes: 频率模式数
        num_blocks: block-diagonal 块数
        sparsity_ratio: 稀疏度比例
    
    Returns:
        AFNO: AFNO 模型
    """
    return AFNO(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=hidden_channels,
        n_layers=n_layers,
        n_modes=n_modes,
        num_blocks=num_blocks,
        sparsity_ratio=sparsity_ratio
    )


__all__ = [
    'AFNOBlockDiagonalConv',
    'AFNO',
    'create_afno'
]
