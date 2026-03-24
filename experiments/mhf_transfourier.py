"""
MHF-FNO 正确实现 - 完全遵循 TransFourier 设计

TransFourier MHF 核心设计：
1. 每个头有独立的频域权重（不分割通道，不分割频率）
2. 每个头独立进行 FFT -> 频域混合 -> IFFT
3. 多头输出通过 Concat 融合（每个头负责部分输出通道）

参数量分析：
- 标准 FNO: (in_ch, out_ch, modes) * n_layers
- MHF-FNO: (in_ch, out_ch/n_heads, modes) * n_heads * n_layers
- 参数量相近！
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class TransFourierMHF2D(nn.Module):
    """
    TransFourier 风格的多头频谱卷积
    
    完全遵循 TransFourier 的 MHF 设计：
    1. 每个头独立进行 FFT -> 频域混合 -> IFFT
    2. 频域权重: (n_heads, in_ch, out_ch_per_head, modes_x, modes_y)
    3. 输出通过 Concat 融合
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
        
        # 每个头的输出通道数
        self.out_channels_per_head = out_channels // n_heads
        assert out_channels % n_heads == 0, "out_channels 必须能被 n_heads 整除"
        
        # 频域模式数 (rfft2 的最后一维是 W//2+1)
        self.modes_x = n_modes[0]
        self.modes_y = n_modes[1] // 2 + 1
        
        # 每个头的频域权重
        # (n_heads, in_ch, out_ch_per_head, modes_x, modes_y)
        self.weights_real = nn.Parameter(
            torch.randn(n_heads, in_channels, self.out_channels_per_head, 
                       self.modes_x, self.modes_y) * 0.01
        )
        self.weights_imag = nn.Parameter(
            torch.randn(n_heads, in_channels, self.out_channels_per_head,
                       self.modes_x, self.modes_y) * 0.01
        )
        
        # Bias
        self.bias = nn.Parameter(torch.zeros(out_channels, 1, 1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, in_channels, H, W)
        """
        B, C, H, W = x.shape
        
        # 2D FFT
        x_freq = torch.fft.rfft2(x, dim=(-2, -1), norm='ortho')
        
        freq_H, freq_W = x_freq.shape[-2], x_freq.shape[-1]
        m_x = min(self.modes_x, freq_H)
        m_y = min(self.modes_y, freq_W)
        
        # 存储每个头的输出
        head_outputs = []
        
        for h in range(self.n_heads):
            # 获取该头的权重
            weight = torch.complex(
                self.weights_real[h, :, :, :m_x, :m_y],
                self.weights_imag[h, :, :, :m_x, :m_y]
            )  # (in_ch, out_ch_per_head, m_x, m_y)
            
            # 频域混合
            # x_freq: (B, in_ch, freq_H, freq_W)
            # weight: (in_ch, out_ch_per_head, m_x, m_y)
            # -> (B, out_ch_per_head, m_x, m_y)
            out_freq = torch.zeros(B, self.out_channels_per_head, freq_H, freq_W,
                                  dtype=torch.cfloat, device=x.device)
            out_freq[:, :, :m_x, :m_y] = torch.einsum(
                'biXY,ioXY->boXY',
                x_freq[:, :, :m_x, :m_y],
                weight
            )
            
            # IFFT
            out = torch.fft.irfft2(out_freq, s=(H, W), dim=(-2, -1), norm='ortho')
            head_outputs.append(out)
        
        # Concat 所有头的输出
        # (B, out_ch_per_head * n_heads, H, W) = (B, out_ch, H, W)
        x_out = torch.cat(head_outputs, dim=1)
        
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
        self.spectral_conv = TransFourierMHF2D(
            in_channels=channels,
            out_channels=channels,
            n_modes=n_modes,
            n_heads=n_heads,
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
    MHF-FNO - TransFourier 风格多头傅里叶神经算子
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
    print(" MHF-FNO (TransFourier 风格) 参数量对比")
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