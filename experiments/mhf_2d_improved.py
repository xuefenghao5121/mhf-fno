"""
MHF-FNO 改进版 - 修复频域处理问题

关键改进：
1. 正确的残差连接
2. 更好的权重初始化
3. 简化多头机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ImprovedMHFSpectralConv2D(nn.Module):
    """
    改进的 2D MHF Spectral Convolution
    
    关键改进：
    1. 使用正确的残差连接
    2. 简化多头机制
    3. 更好的权重初始化
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
        
        # 确保通道数可被头数整除
        assert in_channels % n_heads == 0
        assert out_channels % n_heads == 0
        
        self.head_in_channels = in_channels // n_heads
        self.head_out_channels = out_channels // n_heads
        
        # 频域权重
        # 对于 rfft2，最后一维是 W // 2 + 1
        self.weights_real = nn.Parameter(
            torch.randn(n_heads, self.head_in_channels, self.head_out_channels,
                       n_modes[0], n_modes[1] // 2 + 1) * 0.01
        )
        self.weights_imag = nn.Parameter(
            torch.randn(n_heads, self.head_in_channels, self.head_out_channels,
                       n_modes[0], n_modes[1] // 2 + 1) * 0.01
        )
        
        # 偏置
        self.bias = nn.Parameter(torch.zeros(out_channels, 1, 1))
        
        # 残差连接（如果输入输出通道不同）
        if in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, in_channels, H, W)
        """
        B, C, H, W = x.shape
        
        # 保存残差
        residual = self.residual(x)
        
        # 2D FFT
        x_freq = torch.fft.rfft2(x, dim=(-2, -1), norm='ortho')
        
        # 重塑为多头
        # (B, C, H, W//2+1) -> (B, n_heads, head_in, H, W//2+1)
        x_freq = x_freq.view(B, self.n_heads, self.head_in_channels, H, -1)
        
        # 初始化输出
        freq_H, freq_W = x_freq.shape[-2], x_freq.shape[-1]
        out_freq = torch.zeros(B, self.n_heads, self.head_out_channels, freq_H, freq_W,
                              dtype=torch.cfloat, device=x.device)
        
        # 频域混合
        m_x = min(self.n_modes[0], freq_H)
        m_y = min(self.n_modes[1] // 2 + 1, freq_W)
        
        # 获取输入频率
        x_freq_cut = x_freq[:, :, :, :m_x, :m_y]
        
        # 构建复数权重
        weights = torch.complex(self.weights_real[:, :, :, :m_x, :m_y], 
                               self.weights_imag[:, :, :, :m_x, :m_y])
        
        # 频域混合 (B, D, I, X, Y) x (D, I, O, X, Y) -> (B, D, O, X, Y)
        out_freq[:, :, :, :m_x, :m_y] = torch.einsum(
            'bdiXY,dioXY->bdoXY',
            x_freq_cut,
            weights
        )
        
        # 合并多头
        out_freq = out_freq.reshape(B, self.out_channels, freq_H, freq_W)
        
        # 2D IFFT
        x_out = torch.fft.irfft2(out_freq, s=(H, W), dim=(-2, -1), norm='ortho')
        
        # 添加残差和偏置
        x_out = x_out + residual + self.bias
        
        return x_out


class ImprovedMHFBlock(nn.Module):
    """改进的 MHF Block"""
    
    def __init__(
        self,
        channels: int,
        n_modes: Tuple[int, int],
        n_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        # Spectral Conv (with residual inside)
        self.spectral_conv = ImprovedMHFSpectralConv2D(
            in_channels=channels,
            out_channels=channels,
            n_modes=n_modes,
            n_heads=n_heads,
        )
        
        # Channel MLP (FFN)
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(channels * 2, channels, 1),
        )
        
        self.norm1 = nn.GroupNorm(1, channels)
        self.norm2 = nn.GroupNorm(1, channels)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Spectral conv (残差已在内部)
        x = self.spectral_conv(self.norm1(x))
        
        # FFN + residual
        x = x + self.dropout(self.ffn(self.norm2(x)))
        
        return x


class ImprovedMHFFNO(nn.Module):
    """
    改进的 MHF-FNO
    
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
            ImprovedMHFBlock(
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


if __name__ == "__main__":
    print("测试 ImprovedMHFFNO...")
    
    model = ImprovedMHFFNO(
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