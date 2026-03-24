"""
Engram-MHF-FNO 完整实现（修复版）

架构：
输入 → ┬→ 空间域 Engram (局部模式记忆) ─→ α ×
       ├→ MHF 模块 (频域混合) ─────────→ β ×  → 门控融合 → 输出
       └→ 频域 Engram (频率模式记忆) ──→ γ ×
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class SpatialEngram(nn.Module):
    """空间域 Engram - 修复版"""
    
    def __init__(
        self,
        channels: int,
        window_size: int = 3,
        num_patterns: int = 1000,
    ):
        super().__init__()
        
        self.channels = channels
        self.window_size = window_size
        self.num_patterns = num_patterns
        
        # 嵌入表：每个 pattern 对应一个 channels 维度的向量
        self.embedding_table = nn.Embedding(num_patterns, channels)
        
        # 输出投影，确保输出维度正确
        self.proj = nn.Linear(channels, channels)
        
        self.padding = window_size // 2
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # 使用平均池化提取局部特征
        pooled = F.avg_pool2d(x, self.window_size, stride=1, padding=self.padding)
        # pooled: (B, C, H, W)
        
        # 计算哈希索引：对通道维度求和，得到每个空间位置的标量
        hash_values = pooled.abs().sum(dim=1)  # (B, H, W)
        
        # 量化并取模
        indices = (hash_values * 100).long() % self.num_patterns
        indices = indices.clamp(0, self.num_patterns - 1)
        
        # 查表：indices (B, H, W) -> embeddings (B, H, W, C)
        embeddings = self.embedding_table(indices)
        
        # 投影
        output = self.proj(embeddings)  # (B, H, W, C)
        
        # 转置为 (B, C, H, W)
        output = output.permute(0, 3, 1, 2)
        
        return output


class MHFSpectralConv2D(nn.Module):
    """MHF 频谱卷积"""
    
    def __init__(
        self,
        channels: int,
        n_modes: Tuple[int, int],
        n_heads: int = 4,
    ):
        super().__init__()
        
        self.channels = channels
        self.n_modes = n_modes
        self.n_heads = n_heads
        
        assert channels % n_heads == 0, f"channels ({channels}) 必须能被 n_heads ({n_heads}) 整除"
        
        self.out_channels_per_head = channels // n_heads
        
        self.modes_x = n_modes[0]
        self.modes_y = n_modes[1] // 2 + 1
        
        # 每个头的频域权重
        self.weights_real = nn.Parameter(
            torch.randn(n_heads, channels, self.out_channels_per_head,
                       self.modes_x, self.modes_y) * 0.01
        )
        self.weights_imag = nn.Parameter(
            torch.randn(n_heads, channels, self.out_channels_per_head,
                       self.modes_x, self.modes_y) * 0.01
        )
        
        self.bias = nn.Parameter(torch.zeros(channels, 1, 1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # FFT
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


class SpectralEngram(nn.Module):
    """频域 Engram"""
    
    def __init__(
        self,
        channels: int,
        n_modes: Tuple[int, int],
        num_patterns: int = 500,
    ):
        super().__init__()
        
        self.channels = channels
        self.n_modes = n_modes
        self.num_patterns = num_patterns
        
        self.modes_x = n_modes[0]
        self.modes_y = n_modes[1] // 2 + 1
        
        # 频域嵌入表
        self.freq_embedding = nn.Embedding(num_patterns, channels)
        
        # 频域权重
        self.freq_weight_real = nn.Parameter(
            torch.randn(channels, self.modes_x, self.modes_y) * 0.01
        )
        self.freq_weight_imag = nn.Parameter(
            torch.randn(channels, self.modes_x, self.modes_y) * 0.01
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # FFT
        x_freq = torch.fft.rfft2(x, dim=(-2, -1), norm='ortho')
        
        freq_H, freq_W = x_freq.shape[-2], x_freq.shape[-1]
        m_x = min(self.modes_x, freq_H)
        m_y = min(self.modes_y, freq_W)
        
        # 哈希索引（基于频域幅度）
        magnitude = torch.abs(x_freq[:, :, :m_x, :m_y]).mean(dim=(1, 2, 3))
        indices = (magnitude * 1000).long() % self.num_patterns
        indices = indices.clamp(0, self.num_patterns - 1)
        
        # 查表获取调制因子
        mod = self.freq_embedding(indices)  # (B, channels)
        
        # 构建频域权重
        freq_weight = torch.complex(
            self.freq_weight_real[:, :m_x, :m_y],
            self.freq_weight_imag[:, :m_x, :m_y]
        )  # (C, m_x, m_y)
        
        # 应用频域权重和调制
        out_freq = torch.zeros(B, C, freq_H, freq_W, dtype=torch.cfloat, device=x.device)
        weighted = x_freq[:, :, :m_x, :m_y] * freq_weight.unsqueeze(0)  # (B, C, m_x, m_y)
        out_freq[:, :, :m_x, :m_y] = weighted * mod.view(B, C, 1, 1)
        
        # IFFT
        x_out = torch.fft.irfft2(out_freq, s=(H, W), dim=(-2, -1), norm='ortho')
        
        return x_out


class GatedFusion(nn.Module):
    """门控融合层"""
    
    def __init__(self, channels: int, hidden_dim: int = 32):
        super().__init__()
        
        self.gate_net = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
            nn.Softmax(dim=-1),
        )
    
    def forward(
        self,
        spatial_out: torch.Tensor,
        mhf_out: torch.Tensor,
        spectral_out: torch.Tensor,
    ) -> torch.Tensor:
        B = spatial_out.shape[0]
        
        # 使用 MHF 输出计算门控权重
        gate_input = mhf_out.mean(dim=(2, 3))  # (B, C)
        weights = self.gate_net(gate_input)  # (B, 3)
        
        alpha = weights[:, 0].view(B, 1, 1, 1)  # (B, 1, 1, 1)
        beta = weights[:, 1].view(B, 1, 1, 1)
        gamma = weights[:, 2].view(B, 1, 1, 1)
        
        output = alpha * spatial_out + beta * mhf_out + gamma * spectral_out
        
        return output


class EngramMHFFNOBlock(nn.Module):
    """Engram-MHF-FNO Block"""
    
    def __init__(
        self,
        channels: int,
        n_modes: Tuple[int, int],
        n_heads: int = 4,
        window_size: int = 3,
        num_patterns: int = 1000,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        # 三条路径
        self.spatial_engram = SpatialEngram(channels, window_size, num_patterns)
        self.mhf = MHFSpectralConv2D(channels, n_modes, n_heads)
        self.spectral_engram = SpectralEngram(channels, n_modes, num_patterns // 2)
        
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
        B, C, H, W = x.shape
        
        # 三条路径
        spatial_out = self.spatial_engram(x)      # (B, C, H, W)
        mhf_out = self.mhf(x)                      # (B, C, H, W)
        spectral_out = self.spectral_engram(x)     # (B, C, H, W)
        
        # 验证维度
        assert spatial_out.shape == (B, C, H, W), f"spatial_out: {spatial_out.shape}"
        assert mhf_out.shape == (B, C, H, W), f"mhf_out: {mhf_out.shape}"
        assert spectral_out.shape == (B, C, H, W), f"spectral_out: {spectral_out.shape}"
        
        # 门控融合
        fused = self.gate(spatial_out, mhf_out, spectral_out)
        
        # 残差
        x_skip = self.skip(x)
        x = fused + x_skip
        
        # Channel MLP
        x = x + self.dropout(self.channel_mlp(self.norm(x)))
        
        return x


class EngramMHFFNO(nn.Module):
    """Engram-MHF-FNO 完整模型"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 64,
        n_modes: Tuple[int, int] = (16, 16),
        n_layers: int = 4,
        n_heads: int = 4,
        window_size: int = 3,
        num_patterns: int = 1000,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        # Lifting
        self.lifting = nn.Conv2d(in_channels, hidden_channels, 1)
        
        # Blocks
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
        x = self.lifting(x)
        
        for block in self.blocks:
            x = F.gelu(block(x))
        
        x = self.projection(x)
        
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    print("=" * 60)
    print(" Engram-MHF-FNO 测试")
    print("=" * 60)
    
    from neuralop.models import FNO
    
    hidden_channels = 32
    n_modes = (8, 8)
    n_layers = 3
    
    model_fno = FNO(
        n_modes=n_modes,
        hidden_channels=hidden_channels,
        in_channels=1,
        out_channels=1,
        n_layers=n_layers,
    )
    
    model_engram = EngramMHFFNO(
        in_channels=1,
        out_channels=1,
        hidden_channels=hidden_channels,
        n_modes=n_modes,
        n_layers=n_layers,
        n_heads=4,
        window_size=3,
        num_patterns=1000,
    )
    
    print(f"\n参数量对比:")
    print(f"  FNO:             {count_parameters(model_fno):,}")
    print(f"  Engram-MHF-FNO:  {count_parameters(model_engram):,}")
    
    x = torch.randn(4, 1, 16, 16)
    y_fno = model_fno(x)
    y_engram = model_engram(x)
    
    print(f"\n输入: {x.shape}")
    print(f"FNO 输出: {y_fno.shape}")
    print(f"Engram-MHF-FNO 输出: {y_engram.shape}")
    
    print("\n✅ 测试通过")