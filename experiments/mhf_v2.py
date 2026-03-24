"""
MHF-FNO 简化正确实现

核心设计：
1. 多头学习不同的频域变换（不是分割频率）
2. 每个头有完整的频域权重，但通过低秩分解减少参数
3. 保持完整的通道交互

参数效率策略：
- 标准 FNO: (in_ch, out_ch, modes_x, modes_y)
- MHF-FNO: n_heads * (in_ch, out_ch, modes_x, modes_y//n_heads)
- 或: n_heads * (in_ch//n_heads, out_ch//n_heads, modes_x, modes_y) 低秩分解
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class MultiHeadSpectralConv2D(nn.Module):
    """
    多头频谱卷积
    
    设计思路：
    - n_heads 个独立的频域变换
    - 每个头保持完整通道交互
    - 通过共享部分权重减少参数
    
    参数量：n_heads * (in_ch, out_ch, modes_x, modes_y//2+1)
    为了减少参数，使用通道低秩分解
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: Tuple[int, int],
        n_heads: int = 4,
        rank_ratio: float = 0.5,  # 低秩比例
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes = n_modes
        self.n_heads = n_heads
        
        # 低秩分解
        self.rank = max(1, int(min(in_channels, out_channels) * rank_ratio))
        
        # 频域模式数（rfft2 的最后一维是 W//2+1）
        self.modes_x = n_modes[0]
        self.modes_y = n_modes[1] // 2 + 1
        
        # 多头权重：使用低秩分解减少参数
        # 降维: (n_heads, in_ch, rank)
        self.down_weights = nn.Parameter(
            torch.randn(n_heads, in_channels, self.rank) * 0.01
        )
        # 频域权重: (n_heads, rank, out_ch, modes_x, modes_y)
        self.freq_weights_real = nn.Parameter(
            torch.randn(n_heads, self.rank, out_channels, self.modes_x, self.modes_y) * 0.01
        )
        self.freq_weights_imag = nn.Parameter(
            torch.randn(n_heads, self.rank, out_channels, self.modes_x, self.modes_y) * 0.01
        )
        
        self.bias = nn.Parameter(torch.zeros(out_channels, 1, 1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, in_channels, H, W)
        """
        B, C, H, W = x.shape
        
        # 2D FFT
        x_freq = torch.fft.rfft2(x, dim=(-2, -1), norm='ortho')
        
        freq_H, freq_W = x_freq.shape[-2], x_freq.shape[-1]
        
        # 获取实际使用的模式数
        m_x = min(self.modes_x, freq_H)
        m_y = min(self.modes_y, freq_W)
        
        # 初始化输出
        out_freq = torch.zeros(B, self.out_channels, freq_H, freq_W,
                              dtype=torch.cfloat, device=x.device)
        
        # 获取输入频率（只取低频部分）
        x_freq_cut = x_freq[:, :, :m_x, :m_y]  # (B, in_ch, m_x, m_y)
        
        # 多头处理
        for h in range(self.n_heads):
            # 第一步：通道降维（复数运算）
            down_weight = torch.complex(
                self.down_weights[h], 
                torch.zeros_like(self.down_weights[h])
            )  # (in_ch, rank) -> 复数
            
            # (B, in_ch, m_x, m_y) x (in_ch, rank) -> (B, rank, m_x, m_y)
            x_down = torch.einsum('biXY,ir->brXY', x_freq_cut, down_weight)
            
            # 第二步：频域变换
            weight = torch.complex(
                self.freq_weights_real[h, :, :, :m_x, :m_y],
                self.freq_weights_imag[h, :, :, :m_x, :m_y]
            )  # (rank, out_ch, m_x, m_y)
            
            # (B, rank, m_x, m_y) x (rank, out_ch, m_x, m_y) -> (B, out_ch, m_x, m_y)
            out_h = torch.einsum('brXY,roXY->boXY', x_down, weight)
            
            # 累加（多头输出取平均）
            out_freq[:, :, :m_x, :m_y] += out_h
        
        # 平均多头输出
        out_freq[:, :, :m_x, :m_y] /= self.n_heads
        
        # 2D IFFT
        x_out = torch.fft.irfft2(out_freq, s=(H, W), dim=(-2, -1), norm='ortho')
        
        # 添加 bias
        x_out = x_out + self.bias
        
        return x_out


class MHFFNOBlock(nn.Module):
    """MHF-FNO Block"""
    
    def __init__(
        self,
        channels: int,
        n_modes: Tuple[int, int],
        n_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        # MHF Spectral Conv
        self.spectral_conv = MultiHeadSpectralConv2D(
            in_channels=channels,
            out_channels=channels,
            n_modes=n_modes,
            n_heads=n_heads,
            rank_ratio=0.5,
        )
        
        # Skip connection
        self.skip = nn.Conv2d(channels, channels, 1)
        
        # Channel MLP
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
        # MHF Spectral Conv + Skip
        x_skip = self.skip(x)
        x_spectral = self.spectral_conv(self.norm1(x))
        x = x_spectral + x_skip
        
        # Channel MLP + Skip
        x = x + self.dropout(self.channel_mlp(self.norm2(x)))
        
        return x


class MHFFNO(nn.Module):
    """
    MHF-FNO - 多头傅里叶神经算子
    
    关键设计：
    1. 多头学习不同的频域变换
    2. 通过低秩分解减少参数
    3. 保持完整通道交互
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
            MHFFNOBlock(
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
        # Lifting
        x = self.lifting(x)
        
        # MHF Blocks
        for block in self.blocks:
            x = F.gelu(block(x))
        
        # Projection
        x = self.projection(x)
        
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    print("=" * 60)
    print(" MHF-FNO 参数量对比")
    print("=" * 60)
    
    from neuralop.models import FNO
    
    hidden_channels = 32
    n_modes = (8, 8)
    n_layers = 3
    
    # 标准 FNO
    model_fno = FNO(
        n_modes=n_modes,
        hidden_channels=hidden_channels,
        in_channels=1,
        out_channels=1,
        n_layers=n_layers,
    )
    
    # MHF-FNO
    model_mhf = MHFFNO(
        in_channels=1,
        out_channels=1,
        hidden_channels=hidden_channels,
        n_modes=n_modes,
        n_layers=n_layers,
        n_heads=4,
    )
    
    print(f"\n参数量对比:")
    print(f"  FNO:     {count_parameters(model_fno):,}")
    print(f"  MHF-FNO: {count_parameters(model_mhf):,}")
    
    # 测试前向传播
    x = torch.randn(4, 1, 16, 16)
    y_fno = model_fno(x)
    y_mhf = model_mhf(x)
    
    print(f"\n输入: {x.shape}")
    print(f"FNO 输出: {y_fno.shape}")
    print(f"MHF-FNO 输出: {y_mhf.shape}")
    
    print("\n✅ 测试通过")