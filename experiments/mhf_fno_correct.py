"""
MHF-FNO 正确实现 - 基于对 FNO 的深入分析

关键发现：
1. FNO 不使用多头机制
2. FNO 有独立的跳跃连接
3. 参数量主要来自 SpectralConv 的权重 (in_ch, out_ch, modes_x, modes_y)

MHF 策略：通过低秩分解减少参数量
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class LowRankSpectralConv2D(nn.Module):
    """
    低秩频谱卷积
    
    通过低秩分解减少参数量：
    原始: (in_ch, out_ch, modes_x, modes_y) 
    低秩: (in_ch, rank) + (rank, out_ch, modes_x, modes_y)
    
    参数量比例: rank / in_ch
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: Tuple[int, int],
        rank: Optional[int] = None,  # 低秩，None 表示不使用低秩
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes = n_modes
        self.rank = rank
        
        # 低秩分解
        if rank is not None and rank < min(in_channels, out_channels):
            # 低秩模式
            self.use_low_rank = True
            # 降维: (in_ch, rank)
            self.down_real = nn.Parameter(torch.randn(in_channels, rank) * 0.01)
            self.down_imag = nn.Parameter(torch.randn(in_channels, rank) * 0.01)
            # 频域权重: (rank, out_ch, modes_x, modes_y//2+1)
            self.weight_real = nn.Parameter(
                torch.randn(rank, out_channels, n_modes[0], n_modes[1] // 2 + 1) * 0.01
            )
            self.weight_imag = nn.Parameter(
                torch.randn(rank, out_channels, n_modes[0], n_modes[1] // 2 + 1) * 0.01
            )
        else:
            # 标准模式 (同 FNO)
            self.use_low_rank = False
            self.weight_real = nn.Parameter(
                torch.randn(in_channels, out_channels, n_modes[0], n_modes[1] // 2 + 1) * 0.01
            )
            self.weight_imag = nn.Parameter(
                torch.randn(in_channels, out_channels, n_modes[0], n_modes[1] // 2 + 1) * 0.01
            )
        
        self.bias = nn.Parameter(torch.zeros(out_channels, 1, 1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, in_channels, H, W)
        """
        B, C, H, W = x.shape
        
        # 2D FFT
        x_freq = torch.fft.rfft2(x, dim=(-2, -1), norm='ortho')
        
        # 获取输出形状
        freq_H, freq_W = x_freq.shape[-2], x_freq.shape[-1]
        m_x, m_y = min(self.n_modes[0], freq_H), min(self.n_modes[1] // 2 + 1, freq_W)
        
        # 初始化输出
        out_freq = torch.zeros(B, self.out_channels, freq_H, freq_W,
                              dtype=torch.cfloat, device=x.device)
        
        # 获取输入频率
        x_freq_cut = x_freq[:, :, :m_x, :m_y]  # (B, in_ch, m_x, m_y)
        
        if self.use_low_rank:
            # 低秩分解: (B, in_ch, m_x, m_y) -> (B, rank, m_x, m_y) -> (B, out_ch, m_x, m_y)
            # 第一步: 降维 (复数矩阵乘法)
            down = torch.complex(self.down_real, self.down_imag)  # (in_ch, rank)
            x_down = torch.einsum('biXY,ir->brXY', x_freq_cut, down)  # (B, rank, m_x, m_y)
            
            # 第二步: 频域权重
            weight = torch.complex(
                self.weight_real[:, :, :m_x, :m_y],
                self.weight_imag[:, :, :m_x, :m_y]
            )  # (rank, out_ch, m_x, m_y)
            out_freq[:, :, :m_x, :m_y] = torch.einsum('brXY,roXY->boXY', x_down, weight)
        else:
            # 标准模式 (同 FNO)
            weight = torch.complex(
                self.weight_real[:, :, :m_x, :m_y],
                self.weight_imag[:, :, :m_x, :m_y]
            )  # (in_ch, out_ch, m_x, m_y)
            out_freq[:, :, :m_x, :m_y] = torch.einsum('biXY,ioXY->boXY', x_freq_cut, weight)
        
        # 2D IFFT
        x_out = torch.fft.irfft2(out_freq, s=(H, W), dim=(-2, -1), norm='ortho')
        
        # 添加 bias
        x_out = x_out + self.bias
        
        return x_out


class MHFFNOBlock(nn.Module):
    """
    MHF-FNO Block - 模仿 FNO 的结构
    """
    
    def __init__(
        self,
        channels: int,
        n_modes: Tuple[int, int],
        rank: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        # Spectral Conv
        self.spectral_conv = LowRankSpectralConv2D(
            in_channels=channels,
            out_channels=channels,
            n_modes=n_modes,
            rank=rank,
        )
        
        # Skip connection (模仿 FNO 的 Flattened1dConv)
        self.skip = nn.Conv2d(channels, channels, 1)
        
        # Channel MLP
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(channels * 2, channels, 1),
        )
        
        # Channel MLP skip (模仿 SoftGating)
        self.channel_mlp_skip = nn.Parameter(torch.ones(channels))
        
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Skip connection
        x_skip = self.skip(x)
        
        # Spectral Conv
        x_spectral = self.spectral_conv(x)
        
        # Add skip
        x = x_spectral + x_skip
        
        # Activation
        x = self.activation(x)
        
        # Channel MLP
        x_mlp = self.channel_mlp(x)
        
        # Add skip with gating
        x = x_mlp + x * self.channel_mlp_skip.view(1, -1, 1, 1)
        
        return x


class MHFFNO(nn.Module):
    """
    MHF-FNO - 通过低秩分解减少参数量
    
    兼容 NeuralOperator FNO 接口
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 64,
        n_modes: Tuple[int, int] = (16, 16),
        n_layers: int = 4,
        rank: Optional[int] = None,  # 低秩参数，None 表示标准模式
        dropout: float = 0.0,
    ):
        super().__init__()
        
        # Lifting (模仿 FNO 的 ChannelMLP)
        self.lifting = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels * 2, 1),
            nn.GELU(),
            nn.Conv2d(hidden_channels * 2, hidden_channels, 1),
        )
        
        # MHF Blocks
        self.blocks = nn.ModuleList([
            MHFFNOBlock(
                channels=hidden_channels,
                n_modes=n_modes,
                rank=rank,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])
        
        # Projection (模仿 FNO 的 ChannelMLP)
        self.projection = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels * 2, 1),
            nn.GELU(),
            nn.Conv2d(hidden_channels * 2, out_channels, 1),
        )
    
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


def count_parameters(model):
    """计算参数量"""
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    print("=" * 60)
    print(" MHF-FNO 参数量对比")
    print("=" * 60)
    
    # 标准模式
    model_standard = MHFFNO(
        in_channels=1,
        out_channels=1,
        hidden_channels=32,
        n_modes=(8, 8),
        n_layers=3,
        rank=None,  # 标准模式
    )
    
    # 低秩模式
    model_lowrank = MHFFNO(
        in_channels=1,
        out_channels=1,
        hidden_channels=32,
        n_modes=(8, 8),
        n_layers=3,
        rank=8,  # 低秩模式
    )
    
    print(f"\n标准模式: {count_parameters(model_standard):,} 参数")
    print(f"低秩模式 (rank=8): {count_parameters(model_lowrank):,} 参数")
    
    # 测试前向传播
    x = torch.randn(4, 1, 16, 16)
    y_standard = model_standard(x)
    y_lowrank = model_lowrank(x)
    
    print(f"\n输入: {x.shape}")
    print(f"标准输出: {y_standard.shape}")
    print(f"低秩输出: {y_lowrank.shape}")
    
    print("\n✅ 测试通过")