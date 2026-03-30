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
        
        if in_channels % num_blocks != 0 or out_channels % num_blocks != 0:
            warnings.warn(
                f"通道数 ({in_channels}, {out_channels}) 不能被 num_blocks ({num_blocks}) 整除",
                UserWarning
            )
            self.enable = False
        
        if self.enable:
            self.block_size = in_channels // num_blocks
            assert self.block_size == out_channels // num_blocks
            
            if len(n_modes) == 1:
                weight_shape = (num_blocks, self.block_size, self.block_size, n_modes[0])
            else:
                modes_y = n_modes[1] // 2 + 1 if len(n_modes) > 1 else n_modes[0]
                weight_shape = (num_blocks, self.block_size, self.block_size, n_modes[0], modes_y)
            
            init_std = (2 / (in_channels + out_channels)) ** 0.5
            self.weight = nn.Parameter(
                torch.randn(*weight_shape, dtype=torch.cfloat) * init_std
            )
            
            self.sparsity_threshold = nn.Parameter(
                torch.tensor(init_sparsity_threshold)
            )
            
            self.gates = nn.Parameter(torch.zeros(num_blocks))
        else:
            modes_y = n_modes[-1] // 2 + 1 if len(n_modes) > 1 else n_modes[-1]
            weight_shape = (in_channels, out_channels, n_modes[0], modes_y) if len(n_modes) > 1 else (in_channels, out_channels, n_modes[0])
            self.weight = nn.Parameter(
                torch.randn(*weight_shape, dtype=torch.cfloat) * 0.01
            )
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels, *((1,) * len(n_modes))))
        else:
            self.bias = None
    
    def forward(self, x: torch.Tensor, sparsity_ratio: Optional[float] = None) -> torch.Tensor:
        if not self.enable:
            return self._forward_standard(x)
        
        if x.dim() == 3:
            return self._forward_1d(x, sparsity_ratio)
        elif x.dim() == 4:
            return self._forward_2d(x, sparsity_ratio)
        else:
            raise ValueError(f"不支持的输入维度: {x.dim()}")
    
    def _forward_standard(self, x: torch.Tensor) -> torch.Tensor:
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
        B, C, L = x.shape
        
        x_freq = torch.fft.rfft(x, dim=-1)
        
        batch_size = B
        out_freq = torch.zeros_like(x_freq)
        
        n_modes = min(self.n_modes[0], x_freq.shape[-1])
        
        x_blocks = x_freq.chunk(self.num_blocks, dim=1)
        
        gates = torch.sigmoid(self.gates)
        
        for i, (w, x_b) in enumerate(zip(self.weight, x_blocks)):
            x_b_trunc = x_b[..., :n_modes]
            out_b = torch.einsum('bif,iof->bof', x_b_trunc, w)
            out_b = out_b * gates[i]
            out_freq[:, i*self.block_size:(i+1)*self.block_size, :n_modes] = out_b
        
        out_freq = self._apply_sparsification(out_freq, sparsity_ratio)
        
        x_out = torch.fft.irfft(out_freq, n=L, dim=-1)
        
        if self.bias is not None:
            x_out = x_out + self.bias
        
        return x_out
    
    def _forward_2d(self, x: torch.Tensor, sparsity_ratio: Optional[float] = None) -> torch.Tensor:
        B, C, H, W = x.shape
        
        x_freq = torch.fft.rfft2(x)
        
        out_freq = torch.zeros_like(x_freq)
        
        n_modes_x = min(self.n_modes[0], x_freq.shape[-2])
        n_modes_y = min(self.n_modes[1], x_freq.shape[-1])
        
        x_blocks = x_freq.chunk(self.num_blocks, dim=1)
        
        gates = torch.sigmoid(self.gates)
        
        for i, (w, x_b) in enumerate(zip(self.weight, x_blocks)):
            x_b_trunc = x_b[..., :n_modes_x, :n_modes_y]
            out_b = torch.einsum('bijy,iojy->bojy', x_b_trunc, w)
            out_b = out_b * gates[i]
            out_freq[:, i*self.block_size:(i+1)*self.block_size, :n_modes_x, :n_modes_y] = out_b
        
        out_freq = self._apply_sparsification(out_freq, sparsity_ratio)
        
        x_out = torch.fft.irfft2(out_freq, s=(H, W))
        
        if self.bias is not None:
            x_out = x_out + self.bias
        
        return x_out
    
    def _apply_sparsification(self, x_freq: torch.Tensor, sparsity_ratio: Optional[float] = None) -> torch.Tensor:
        if sparsity_ratio is not None:
            threshold = torch.quantile(x_freq.abs(), 1.0 - sparsity_ratio)
            sparsity_mask = x_freq.abs() > threshold
            return x_freq * sparsity_mask
        else:
            threshold = F.softplus(self.sparsity_threshold)
            x_mag = x_freq.abs()
            x_sign = torch.sign(x_freq)
            x_sparse = x_sign * F.relu(x_mag - threshold)
            return x_sparse
    
    def get_sparsity_info(self, x: torch.Tensor) -> dict:
        if x.dim() == 3:
            x_freq = torch.fft.rfft(x, dim=-1)
        else:
            x_freq = torch.fft.rfft2(x)
        
        threshold = F.softplus(self.sparsity_threshold).item()
        
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
        
        self.fc0 = nn.Linear(in_channels, hidden_channels)
        
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
        
        if activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'relu':
            self.activation = F.relu
        else:
            raise ValueError(f"不支持的激活函数: {activation}")
        
        self.fc1 = nn.Linear(hidden_channels, out_channels)
    
    def forward(self, x: torch.Tensor, sparsity_ratio: Optional[float] = None) -> torch.Tensor:
        if sparsity_ratio is None:
            sparsity_ratio = self.sparsity_ratio
        
        if x.dim() == 3:
            x = self.fc0(x.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            x = self.fc0(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        for afno_layer in self.afno_layers:
            x = afno_layer(x, sparsity_ratio=sparsity_ratio)
            x = self.activation(x)
        
        if x.dim() == 3:
            x = self.fc1(x.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            x = self.fc1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        return x


def create_afno(
    in_channels: int = 1,
    out_channels: int = 1,
    hidden_channels: int = 64,
    n_layers: int = 4,
    n_modes: Tuple[int, ...] = (12, 12),
    num_blocks: int = 8,
    sparsity_ratio: float = 0.5
) -> AFNO:
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
