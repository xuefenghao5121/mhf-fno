"""
Engram-MHF-FNO 完整实现

架构：
输入 → ┬→ 空间域 Engram (局部模式记忆) ─→ α ×
       ├→ MHF 模块 (频域混合) ─────────→ β ×  → 门控融合 → 输出
       └→ 频域 Engram (频率模式记忆) ──→ γ ×

核心创新：
1. 空间域 Engram：记忆局部物理模式（如 3×3 窗口）
2. MHF：TransFourier 风格的多头频域混合
3. 频域 Engram：记忆重复频率成分
4. 门控融合：动态权重融合三条路径
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


# ============================================
# 1. 空间域 Engram
# ============================================

class SpatialEngram(nn.Module):
    """
    空间域 Engram - 记忆局部物理模式
    
    原理：
    1. 将输入场划分为局部窗口（如 3×3）
    2. 使用确定性哈希将窗口模式映射到嵌入表索引
    3. 查表获取嵌入，重建输出
    
    优势：
    - 记忆重复出现的局部模式
    - O(1) 查表，计算高效
    - 可卸载到 CPU 内存
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        window_size: int = 3,
        num_patterns: int = 10000,
        embed_dim: int = 64,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.window_size = window_size
        self.num_patterns = num_patterns
        self.embed_dim = embed_dim
        
        # 嵌入表（可卸载到 CPU）
        self.embedding_table = nn.Embedding(num_patterns, embed_dim)
        
        # 输出投影（直接投影到目标通道数）
        self.output_proj = nn.Linear(embed_dim, out_channels)
        
        # 窗口提取的 padding
        self.padding = window_size // 2
    
    def deterministic_hash(self, windows: torch.Tensor) -> torch.Tensor:
        """
        确定性哈希：将窗口模式映射到索引
        
        使用简单的量化 + 哈希方法
        windows: (B, C, H, W, window_size, window_size)
        """
        # 量化到固定精度
        quantized = (windows * 100).long()
        
        # 使用多项式滚动哈希
        # hash = sum(windows[i] * p^i) mod num_patterns
        flat = quantized.flatten(start_dim=-2)  # (B, C, H, W, window_size^2)
        
        # 简化：使用 PyTorch 的 hash 函数
        # 将每个窗口映射为一个整数
        hash_values = flat.sum(dim=-1)  # (B, C, H, W)
        
        # 取模得到索引
        indices = torch.abs(hash_values) % self.num_patterns
        
        # 对通道维度取平均（需要转为 float）
        indices = indices.float().mean(dim=1).long()  # (B, H, W)
        
        return indices
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # 提取局部窗口
        # 使用 unfold 提取滑动窗口
        x_pad = F.pad(x, [self.padding] * 4, mode='replicate')
        
        windows = x_pad.unfold(2, self.window_size, 1).unfold(3, self.window_size, 1)
        # windows: (B, C, H, W, window_size, window_size)
        
        # 哈希索引
        indices = self.deterministic_hash(windows)  # (B, H, W)
        
        # 查表
        embeddings = self.embedding_table(indices)  # (B, H, W, embed_dim)
        
        # 投影到输出
        output = self.output_proj(embeddings)  # (B, H, W, out_channels)
        output = output.permute(0, 3, 1, 2)  # (B, out_channels, H, W)
        
        return output


# ============================================
# 2. MHF 模块 (TransFourier 风格)
# ============================================

class MHFSpectralConv2D(nn.Module):
    """
    Multi-Head Fourier Spectral Convolution
    
    基于 TransFourier 的设计：
    - 每个头独立进行 FFT -> 频域混合 -> IFFT
    - 保持完整的通道交互
    - 输出通过 Concat 融合
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
        
        # 每个头负责部分输出通道
        self.out_channels_per_head = out_channels // n_heads
        assert out_channels % n_heads == 0
        
        # 频域模式数
        self.modes_x = n_modes[0]
        self.modes_y = n_modes[1] // 2 + 1
        
        # 每个头的频域权重
        self.weights_real = nn.Parameter(
            torch.randn(n_heads, in_channels, self.out_channels_per_head,
                       self.modes_x, self.modes_y) * 0.01
        )
        self.weights_imag = nn.Parameter(
            torch.randn(n_heads, in_channels, self.out_channels_per_head,
                       self.modes_x, self.modes_y) * 0.01
        )
        
        self.bias = nn.Parameter(torch.zeros(out_channels, 1, 1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # 2D FFT
        x_freq = torch.fft.rfft2(x, dim=(-2, -1), norm='ortho')
        
        freq_H, freq_W = x_freq.shape[-2], x_freq.shape[-1]
        m_x = min(self.modes_x, freq_H)
        m_y = min(self.modes_y, freq_W)
        
        head_outputs = []
        
        for h in range(self.n_heads):
            weight = torch.complex(
                self.weights_real[h, :, :, :m_x, :m_y],
                self.weights_imag[h, :, :, :m_x, :m_y]
            )
            
            out_freq = torch.zeros(B, self.out_channels_per_head, freq_H, freq_W,
                                  dtype=torch.cfloat, device=x.device)
            out_freq[:, :, :m_x, :m_y] = torch.einsum(
                'biXY,ioXY->boXY',
                x_freq[:, :, :m_x, :m_y],
                weight
            )
            
            out = torch.fft.irfft2(out_freq, s=(H, W), dim=(-2, -1), norm='ortho')
            head_outputs.append(out)
        
        x_out = torch.cat(head_outputs, dim=1)
        x_out = x_out + self.bias
        
        return x_out


# ============================================
# 3. 频域 Engram
# ============================================

class SpectralEngram(nn.Module):
    """
    频域 Engram - 记忆重复频率成分
    
    原理：
    1. 对输入进行 FFT
    2. 对频域系数进行哈希索引
    3. 查表获取频域嵌入
    4. IFFT 恢复时域
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: Tuple[int, int],
        num_patterns: int = 5000,
        embed_dim: int = 32,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes = n_modes
        self.num_patterns = num_patterns
        self.embed_dim = embed_dim
        
        self.modes_x = n_modes[0]
        self.modes_y = n_modes[1] // 2 + 1
        
        # 频域嵌入表
        self.freq_embedding = nn.Embedding(num_patterns, embed_dim)
        
        # 频域投影
        self.freq_proj_real = nn.Linear(embed_dim, out_channels * self.modes_x * self.modes_y)
        self.freq_proj_imag = nn.Linear(embed_dim, out_channels * self.modes_x * self.modes_y)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # FFT
        x_freq = torch.fft.rfft2(x, dim=(-2, -1), norm='ortho')
        
        freq_H, freq_W = x_freq.shape[-2], x_freq.shape[-1]
        m_x = min(self.modes_x, freq_H)
        m_y = min(self.modes_y, freq_W)
        
        # 提取低频成分
        low_freq = x_freq[:, :, :m_x, :m_y]  # (B, C, m_x, m_y)
        
        # 哈希索引（基于频域幅度）
        magnitude = torch.abs(low_freq).mean(dim=(1, 2, 3))  # (B,)
        indices = (magnitude * 1000).long() % self.num_patterns
        
        # 查表
        embeddings = self.freq_embedding(indices)  # (B, embed_dim)
        
        # 投影到频域
        real = self.freq_proj_real(embeddings)  # (B, out_ch * m_x * m_y)
        imag = self.freq_proj_imag(embeddings)
        
        # 重塑为频域形状
        real = real.view(B, self.out_channels, m_x, m_y)
        imag = imag.view(B, self.out_channels, m_x, m_y)
        
        # 构建频域输出
        out_freq = torch.zeros(B, self.out_channels, freq_H, freq_W,
                              dtype=torch.cfloat, device=x.device)
        out_freq[:, :, :m_x, :m_y] = torch.complex(real, imag)
        
        # IFFT
        x_out = torch.fft.irfft2(out_freq, s=(H, W), dim=(-2, -1), norm='ortho')
        
        return x_out


# ============================================
# 4. 门控融合层
# ============================================

class GatedFusion(nn.Module):
    """
    门控融合层 - 动态权重融合三条路径
    
    output = α * spatial + β * mhf + γ * spectral
    
    α, β, γ 由小 MLP 动态计算
    """
    
    def __init__(self, channels: int, hidden_dim: int = 64):
        super().__init__()
        
        # 门控网络
        self.gate_net = nn.Sequential(
            nn.Linear(channels * 3, hidden_dim),  # 输入：三条路径拼接
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
            nn.Softmax(dim=-1),
        )
        
        # 可学习的基线权重
        self.base_alpha = nn.Parameter(torch.tensor(0.33))
        self.base_beta = nn.Parameter(torch.tensor(0.34))
        self.base_gamma = nn.Parameter(torch.tensor(0.33))
    
    def forward(
        self,
        spatial_out: torch.Tensor,
        mhf_out: torch.Tensor,
        spectral_out: torch.Tensor,
    ) -> torch.Tensor:
        """
        三条路径的输出融合
        """
        # 全局平均池化作为门控输入
        spatial_pool = spatial_out.mean(dim=(2, 3))  # (B, C)
        mhf_pool = mhf_out.mean(dim=(2, 3))
        spectral_pool = spectral_out.mean(dim=(2, 3))
        
        # 拼接
        gate_input = torch.cat([spatial_pool, mhf_pool, spectral_pool], dim=-1)  # (B, 3C)
        
        # 计算门控权重
        weights = self.gate_net(gate_input)  # (B, 3)
        alpha = weights[:, 0:1].unsqueeze(-1)  # (B, 1, 1)
        beta = weights[:, 1:2].unsqueeze(-1)
        gamma = weights[:, 2:3].unsqueeze(-1)
        
        # 融合
        output = alpha * spatial_out + beta * mhf_out + gamma * spectral_out
        
        return output


# ============================================
# 5. 完整的 Engram-MHF-FNO
# ============================================

class EngramMHFFNOBlock(nn.Module):
    """
    Engram-MHF-FNO Block
    
    包含：
    - 空间域 Engram
    - MHF 模块
    - 频域 Engram
    - 门控融合
    - 残差连接
    """
    
    def __init__(
        self,
        channels: int,
        n_modes: Tuple[int, int],
        n_heads: int = 4,
        window_size: int = 3,
        num_patterns: int = 10000,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        # 三条路径
        self.spatial_engram = SpatialEngram(
            in_channels=channels,
            out_channels=channels,
            window_size=window_size,
            num_patterns=num_patterns,
        )
        
        self.mhf = MHFSpectralConv2D(
            in_channels=channels,
            out_channels=channels,
            n_modes=n_modes,
            n_heads=n_heads,
        )
        
        self.spectral_engram = SpectralEngram(
            in_channels=channels,
            out_channels=channels,
            n_modes=n_modes,
            num_patterns=num_patterns // 2,
        )
        
        # 门控融合
        self.gate = GatedFusion(channels)
        
        # 残差连接
        self.skip = nn.Conv2d(channels, channels, 1)
        
        # Channel MLP
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(channels * 2, channels, 1),
        )
        
        self.norm = nn.GroupNorm(1, channels)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 三条路径
        spatial_out = self.spatial_engram(x)
        mhf_out = self.mhf(x)
        spectral_out = self.spectral_engram(x)
        
        # 门控融合
        fused = self.gate(spatial_out, mhf_out, spectral_out)
        
        # 残差
        x_skip = self.skip(x)
        x = fused + x_skip
        
        # Channel MLP
        x = x + self.dropout(self.channel_mlp(self.norm(x)))
        
        return x


class EngramMHFFNO(nn.Module):
    """
    Engram-MHF-FNO 完整模型
    
    创新点：
    1. 双域 Engram 记忆（空间域 + 频域）
    2. MHF 模块（TransFourier 风格）
    3. 门控融合
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 64,
        n_modes: Tuple[int, int] = (16, 16),
        n_layers: int = 4,
        n_heads: int = 4,
        window_size: int = 3,
        num_patterns: int = 10000,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        # Lifting
        self.lifting = nn.Conv2d(in_channels, hidden_channels, 1)
        
        # Engram-MHF Blocks
        self.blocks = nn.ModuleList([
            EngramMHFFNOBlock(
                channels=hidden_channels,
                n_modes=n_modes,
                n_heads=n_heads,
                window_size=window_size,
                num_patterns=num_patterns,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])
        
        # Projection
        self.projection = nn.Conv2d(hidden_channels, out_channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Lifting
        x = self.lifting(x)
        
        # Blocks
        for block in self.blocks:
            x = F.gelu(block(x))
        
        # Projection
        x = self.projection(x)
        
        return x


# ============================================
# 测试
# ============================================

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    print("=" * 60)
    print(" Engram-MHF-FNO 完整模型测试")
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
    
    # Engram-MHF-FNO
    model_engram = EngramMHFFNO(
        in_channels=1,
        out_channels=1,
        hidden_channels=hidden_channels,
        n_modes=n_modes,
        n_layers=n_layers,
        n_heads=4,
        window_size=3,
        num_patterns=10000,
    )
    
    print(f"\n参数量对比:")
    print(f"  FNO:             {count_parameters(model_fno):,}")
    print(f"  Engram-MHF-FNO:  {count_parameters(model_engram):,}")
    
    # 测试前向传播
    x = torch.randn(4, 1, 16, 16)
    y_fno = model_fno(x)
    y_engram = model_engram(x)
    
    print(f"\n输入: {x.shape}")
    print(f"FNO 输出: {y_fno.shape}")
    print(f"Engram-MHF-FNO 输出: {y_engram.shape}")
    
    print("\n✅ 测试通过")