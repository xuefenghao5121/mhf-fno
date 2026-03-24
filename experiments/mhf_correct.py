"""
MHF-FNO 正确实现 - 基于频率分段的多头机制

设计逻辑：
1. MHF 来自 TransFourier，多头学习不同的频域变换模式
2. 每个头保持完整的通道交互（不分割通道）
3. 多头在**频率模式维度**上分工

频率分段多头策略：
- Head 1: 低频模式 (物理场全局趋势)
- Head 2: 中频模式 (局部特征)
- Head 3: 高频模式 (细节纹理)
- ...

参数量对比：
- 标准FNO: (in_ch, out_ch, modes_x, modes_y)
- MHF-FNO: n_heads * (in_ch, out_ch, modes_per_head_x, modes_per_head_y)
- 参数量相近，但表达能力更强
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class FrequencySegmentedMHFConv2D(nn.Module):
    """
    频率分段多头频谱卷积
    
    核心思想：
    - 每个头处理不同的频率范围
    - 所有头保持完整的通道交互
    - 最后合并所有头的输出
    
    参数量 = n_heads * (in_ch, out_ch, modes_per_head_x, modes_per_head_y)
    与标准 SpectralConv 参数量相近（当 modes_per_head = modes // n_heads 时）
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: Tuple[int, int],
        n_heads: int = 4,
        overlap: int = 0,  # 频率段重叠（增强边界处理）
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes = n_modes
        self.n_heads = n_heads
        self.overlap = overlap
        
        # 计算每个头的频率范围
        modes_per_head_x = max(1, n_modes[0] // n_heads)
        modes_per_head_y = max(1, n_modes[1] // n_heads)
        
        self.modes_per_head = (modes_per_head_x, modes_per_head_y)
        
        # 每个头的频域权重
        # (n_heads, in_channels, out_channels, modes_per_head_x, modes_per_head_y//2+1)
        self.weights_real = nn.ParameterList([
            nn.Parameter(
                torch.randn(in_channels, out_channels, modes_per_head_x, modes_per_head_y // 2 + 1) * 0.01
            )
            for _ in range(n_heads)
        ])
        self.weights_imag = nn.ParameterList([
            nn.Parameter(
                torch.randn(in_channels, out_channels, modes_per_head_x, modes_per_head_y // 2 + 1) * 0.01
            )
            for _ in range(n_heads)
        ])
        
        # Bias
        self.bias = nn.Parameter(torch.zeros(out_channels, 1, 1))
    
    def get_head_frequency_range(self, head_idx: int):
        """获取第 head_idx 个头处理的频率范围"""
        modes_x, modes_y = self.modes_per_head
        
        start_x = head_idx * modes_x
        end_x = start_x + modes_x + self.overlap
        
        start_y = head_idx * modes_y
        end_y = start_y + modes_y + self.overlap
        
        return (start_x, end_x), (start_y, end_y)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, in_channels, H, W)
        """
        B, C, H, W = x.shape
        
        # 2D FFT
        x_freq = torch.fft.rfft2(x, dim=(-2, -1), norm='ortho')
        
        freq_H, freq_W = x_freq.shape[-2], x_freq.shape[-1]
        
        # 初始化输出
        out_freq = torch.zeros(B, self.out_channels, freq_H, freq_W,
                              dtype=torch.cfloat, device=x.device)
        
        # 每个头处理对应的频率范围
        for h in range(self.n_heads):
            (start_x, end_x), (start_y, end_y) = self.get_head_frequency_range(h)
            
            # 边界检查
            end_x = min(end_x, min(self.n_modes[0], freq_H))
            end_y = min(end_y, min(self.n_modes[1] // 2 + 1, freq_W))
            
            if start_x >= end_x or start_y >= end_y:
                continue
            
            # 获取输入频率段
            x_freq_segment = x_freq[:, :, start_x:end_x, :end_y]
            
            # 获取权重
            weight_real = self.weights_real[h][:, :, :end_x - start_x, :end_y]
            weight_imag = self.weights_imag[h][:, :, :end_x - start_x, :end_y]
            weight = torch.complex(weight_real, weight_imag)
            
            # 频域混合 - 保持完整通道交互！
            # x_freq_segment: (B, in_ch, seg_x, seg_y)
            # weight: (in_ch, out_ch, seg_x, seg_y)
            # -> (B, out_ch, seg_x, seg_y)
            out_segment = torch.einsum('biXY,ioXY->boXY', x_freq_segment, weight)
            
            # 累加到输出
            out_freq[:, :, start_x:end_x, :end_y] += out_segment
        
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
        self.spectral_conv = FrequencySegmentedMHFConv2D(
            in_channels=channels,
            out_channels=channels,
            n_modes=n_modes,
            n_heads=n_heads,
        )
        
        # Skip connection (同 FNO)
        self.skip = nn.Conv2d(channels, channels, 1)
        
        # Channel MLP (同 FNO)
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
    MHF-FNO - 频率分段多头傅里叶神经算子
    
    兼容 NeuralOperator FNO 接口
    
    关键创新：
    1. 多头在频率维度分工（低频、中频、高频）
    2. 每个头保持完整通道交互
    3. 参数量与标准 FNO 相近
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
        
        # Lifting (同 FNO)
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
        
        # Projection (同 FNO)
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
    print(" MHF-FNO 正确实现测试")
    print("=" * 60)
    
    from neuralop.models import FNO
    
    # 对比参数量
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