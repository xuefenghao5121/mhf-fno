"""
Engram Module for MHF-FNO

基于DeepSeek Engram架构的确定性记忆模块

核心设计:
1. SpatialEngram - 空间域局部模式记忆
2. SpectralEngram - 频域频率模式记忆
3. GatedFusion - 门控融合层
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
import hashlib


class SpatialEngram1D(nn.Module):
    """空间域Engram - 记忆局部模式（轻量版）"""
    
    def __init__(self, in_channels: int, window_size: int = 4, num_patterns: int = 1000, embed_dim: int = 8):
        super().__init__()
        
        self.window_size = window_size
        self.num_patterns = num_patterns
        
        # 轻量嵌入表
        self.embedding_table = nn.Embedding(num_patterns, embed_dim)
        self.projection = nn.Linear(embed_dim, in_channels)
    
    def _hash_window(self, window: torch.Tensor) -> torch.Tensor:
        """向量化哈希 - 快速版本"""
        batch_size = window.shape[0]
        
        # 简化哈希：使用张量操作的确定性哈希
        window_flat = window.view(batch_size, -1)
        
        # 使用简单的乘法哈希（比MD5快100倍）
        prime = 31
        hash_values = (window_flat * prime).sum(dim=1).long() % self.num_patterns
        hash_values = torch.clamp(hash_values, 0, self.num_patterns - 1)
        
        return hash_values
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, seq_len = x.shape
        
        pad_size = self.window_size // 2
        x_padded = F.pad(x, (pad_size, pad_size), mode='replicate')
        
        output = torch.zeros_like(x)
        
        for i in range(seq_len):
            window = x_padded[:, :, i:i+self.window_size]  # (batch, channels, window_size)
            window_flat = window.reshape(batch_size, -1)  # (batch, channels * window_size)
            
            # 快速哈希
            prime = 31
            idx = (window_flat * prime).sum(dim=1).long() % self.num_patterns
            idx = torch.clamp(idx, 0, self.num_patterns - 1)
            
            embeddings = self.embedding_table(idx)
            output[:, :, i] = self.projection(embeddings)
        
        return output


class SpectralEngram1D(nn.Module):
    """频域Engram - 记忆频率模式（轻量版）"""
    
    def __init__(self, in_channels: int, n_modes: int = 16, num_patterns: int = 500, embed_dim: int = 8):
        super().__init__()
        
        self.n_modes = n_modes
        self.num_patterns = num_patterns
        
        # 轻量嵌入表
        self.embedding_table = nn.Embedding(num_patterns, embed_dim)
        self.projection = nn.Linear(embed_dim, in_channels * n_modes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, seq_len = x.shape
        
        x_freq = torch.fft.rfft(x, dim=-1)
        freq_modes = x_freq[:, :, :self.n_modes]
        freq_mag = torch.abs(freq_modes).mean(dim=1)
        
        indices = (freq_mag * 1000).long().sum(dim=1) % self.num_patterns
        
        embeddings = self.embedding_table(indices)
        projected = self.projection(embeddings).view(batch_size, channels, self.n_modes)
        
        full_freq = torch.zeros(batch_size, channels, x_freq.shape[-1], dtype=x_freq.dtype, device=x.device)
        full_freq[:, :, :self.n_modes] = projected.to(x_freq.dtype)
        
        output = torch.fft.irfft(full_freq, n=seq_len, dim=-1)
        return output


class GatedFusion(nn.Module):
    """门控融合层"""
    
    def __init__(self, channels: int):
        super().__init__()
        
        self.gate_net = nn.Sequential(
            nn.Linear(channels * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, spatial_mem, mhf_out, freq_mem):
        spatial_global = spatial_mem.mean(dim=-1)
        mhf_global = mhf_out.mean(dim=-1)
        freq_global = freq_mem.mean(dim=-1)
        
        concat = torch.cat([spatial_global, mhf_global, freq_global], dim=-1)
        weights = self.gate_net(concat)
        
        output = (weights[:, 0:1, None] * spatial_mem + 
                  weights[:, 1:2, None] * mhf_out + 
                  weights[:, 2:3, None] * freq_mem)
        return output


class EngramMHFFNO1D(nn.Module):
    """Engram + MHF FNO 完整架构"""
    
    def __init__(self, in_channels, out_channels, hidden_channels=64, n_modes=16, n_layers=4, n_heads=4, use_engram=True):
        super().__init__()
        
        self.use_engram = use_engram
        
        self.lifting = nn.Linear(in_channels, hidden_channels)
        
        if use_engram:
            self.spatial_engram = nn.ModuleList([
                SpatialEngram1D(hidden_channels, window_size=4, num_patterns=500, embed_dim=8)
                for _ in range(n_layers)
            ])
            self.spectral_engram = nn.ModuleList([
                SpectralEngram1D(hidden_channels, n_modes=n_modes, num_patterns=200, embed_dim=8)
                for _ in range(n_layers)
            ])
            self.gated_fusion = nn.ModuleList([
                GatedFusion(hidden_channels)
                for _ in range(n_layers)
            ])
        
        from mhf_1d import MHFSpectralConv1D
        self.mhf_layers = nn.ModuleList([
            MHFSpectralConv1D(hidden_channels, hidden_channels, n_modes, n_heads)
            for _ in range(n_layers)
        ])
        
        self.projection = nn.Linear(hidden_channels, out_channels)
    
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.lifting(x)
        x = x.transpose(1, 2)
        
        for i, mhf_layer in enumerate(self.mhf_layers):
            if self.use_engram:
                spatial_mem = self.spatial_engram[i](x)
                mhf_out = mhf_layer(x)
                freq_mem = self.spectral_engram[i](x)
                x = self.gated_fusion[i](spatial_mem, mhf_out, freq_mem)
            else:
                x = mhf_layer(x)
            x = F.gelu(x)
        
        x = x.transpose(1, 2)
        x = self.projection(x)
        x = x.transpose(1, 2)
        
        return x


if __name__ == "__main__":
    print("=" * 60)
    print(" Engram + MHF-FNO 测试")
    print("=" * 60)
    
    # 测试各模块
    print("\n1. SpatialEngram 测试:")
    spatial = SpatialEngram1D(32)
    x = torch.randn(4, 32, 64)
    print(f"参数量: {sum(p.numel() for p in spatial.parameters()):,}")
    
    print("\n2. SpectralEngram 测试:")
    spectral = SpectralEngram1D(32)
    print(f"参数量: {sum(p.numel() for p in spectral.parameters()):,}")
    
    print("\n3. 完整模型对比:")
    model_no = EngramMHFFNO1D(1, 1, 32, 8, 2, 2, use_engram=False)
    model_with = EngramMHFFNO1D(1, 1, 32, 8, 2, 2, use_engram=True)
    
    print(f"无Engram: {sum(p.numel() for p in model_no.parameters()):,}")
    print(f"有Engram: {sum(p.numel() for p in model_with.parameters()):,}")
    
    print("\n✅ 测试完成")