"""
MHF-FNO 简化版 - 专注于正确性

用于 NeuralOperator 官方 Benchmark 测试
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union

Number = Union[int, float]


class MHFSpectralConv2D(nn.Module):
    """
    2D Multi-Head Fourier Spectral Convolution
    
    简化版实现，专注于正确性
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: Tuple[int, int],
        n_heads: int = 4,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes = n_modes
        self.n_heads = n_heads
        
        assert in_channels % n_heads == 0
        assert out_channels % n_heads == 0
        
        self.head_in_channels = in_channels // n_heads
        self.head_out_channels = out_channels // n_heads
        
        init_std = (2 / (in_channels + out_channels)) ** 0.5
        
        # 频域权重: (n_heads, head_in, head_out, n_modes_x, n_modes_y//2+1)
        # 对于 rfft2，最后一维是 n // 2 + 1
        self.weight = nn.Parameter(
            torch.randn(n_heads, self.head_in_channels, self.head_out_channels,
                       n_modes[0], n_modes[1] // 2 + 1, dtype=torch.cfloat) * init_std
        )
        
        self.bias = nn.Parameter(init_std * torch.randn(out_channels, 1, 1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, in_channels, H, W)
        """
        B, C, H, W = x.shape
        
        # 2D FFT
        x_freq = torch.fft.rfft2(x, dim=(-2, -1), norm='forward')
        
        # 重塑为多头
        # (B, C, H, W//2+1) -> (B, n_heads, head_in, H, W//2+1)
        x_freq = x_freq.view(B, self.n_heads, self.head_in_channels, H, -1)
        
        # 初始化输出
        out_H, out_W = x_freq.shape[-2], x_freq.shape[-1]
        out_freq = torch.zeros(B, self.n_heads, self.head_out_channels, out_H, out_W,
                              dtype=x_freq.dtype, device=x.device)
        
        # 频域混合
        m_x, m_y = min(self.n_modes[0], out_H), min(self.n_modes[1] // 2 + 1, out_W)
        
        # einsum: (B, D, I, H, W) x (D, I, O, H, W) -> (B, D, O, H, W)
        # D = n_heads, I = head_in_channels, O = head_out_channels
        out_freq[:, :, :, :m_x, :m_y] = torch.einsum(
            'bdiXY,dioXY->bdoXY',
            x_freq[:, :, :, :m_x, :m_y],
            self.weight[:, :, :, :m_x, :m_y]
        )
        
        # 合并多头
        out_freq = out_freq.reshape(B, self.out_channels, out_H, out_W)
        
        # 2D IFFT
        x_out = torch.fft.irfft2(out_freq, s=(H, W), dim=(-2, -1), norm='forward')
        
        # 添加 bias
        x_out = x_out + self.bias
        
        return x_out


class MHFBlock2D(nn.Module):
    """MHF Block for 2D"""
    
    def __init__(
        self,
        channels: int,
        n_modes: Tuple[int, int],
        n_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.spectral_conv = MHFSpectralConv2D(
            in_channels=channels,
            out_channels=channels,
            n_modes=n_modes,
            n_heads=n_heads,
        )
        
        # Channel MLP (like FNO)
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(channels * 2, channels, 1),
        )
        
        self.norm1 = nn.GroupNorm(1, channels)
        self.norm2 = nn.GroupNorm(1, channels)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Spectral conv + residual
        x = x + self.dropout(self.spectral_conv(self.norm1(x)))
        # Channel MLP + residual
        x = x + self.dropout(self.channel_mlp(self.norm2(x)))
        return x


class MHFFNO2D(nn.Module):
    """
    MHF-based Fourier Neural Operator for 2D
    
    兼容 NeuralOperator FNO 接口
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 64,
        n_modes: Tuple[int, int] = (16, 16),
        n_layers: int = 4,
        n_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        # Lifting
        self.lifting = nn.Conv2d(in_channels, hidden_channels, 1)
        
        # MHF Blocks
        self.blocks = nn.ModuleList([
            MHFBlock2D(
                channels=hidden_channels,
                n_modes=n_modes,
                n_heads=n_heads,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])
        
        # Projection
        self.projection = nn.Conv2d(hidden_channels, out_channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, in_channels, H, W)
        """
        # Lifting
        x = self.lifting(x)
        
        # MHF Blocks
        for block in self.blocks:
            x = block(x)
        
        # Projection
        x = self.projection(x)
        
        return x


# 测试代码
if __name__ == "__main__":
    print("测试 MHFFNO2D...")
    
    model = MHFFNO2D(
        in_channels=1,
        out_channels=1,
        hidden_channels=32,
        n_modes=(8, 8),
        n_layers=3,
        n_heads=4,
    )
    
    x = torch.randn(4, 1, 16, 16)
    y = model(x)
    
    print(f"输入: {x.shape}")
    print(f"输出: {y.shape}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
    print("✅ 测试通过")