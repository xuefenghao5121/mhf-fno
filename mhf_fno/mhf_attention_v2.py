"""
MHF-FNO with Multiple Attention Variants
=========================================

支持多种注意力变体的 MHF-FNO 实现。

变体：
- 'senet': 原始 SENet 风格 (基线)
- 'mha': 真正的多头注意力
- 'coda': CoDA-NO 风格
- 'hybrid': 混合空间-频域注意力

作者: Tianyuan Team - 天渠
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, List
import warnings

from neuralop.models import FNO

from .mhf_fno import MHFSpectralConv
from .attention_variants import (
    CrossHeadMultiHeadAttention,
    FrequencyDomainAttention,
    HybridSpatialFrequencyAttention,
    CoDAStyleAttention,
)


ATTENTION_VARIANTS = {
    'senet': 'SENet风格的轻量级注意力 (原实现)',
    'mha': '真正的跨头多头注意力',
    'coda': 'CoDA-NO风格的瓶颈注意力',
    'hybrid': '混合空间-频域注意力',
}


class MHFSpectralConvV2(MHFSpectralConv):
    """
    支持多种注意力变体的多头频谱卷积
    
    Args:
        attention_type: 注意力类型，可选 'senet', 'mha', 'coda', 'hybrid'
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: Tuple[int, ...],
        n_heads: int = 4,
        attention_type: str = 'mha',
        attention_dropout: float = 0.0,
        **kwargs
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            n_modes=n_modes,
            n_heads=n_heads,
            **kwargs
        )
        
        self.attention_type = attention_type
        self.use_attention = attention_type != 'none' and self.use_mhf
        
        if self.use_attention:
            head_dim = self.head_out
            
            if attention_type == 'senet':
                # 原始 SENet 风格 (用于对比)
                from .mhf_attention import CrossHeadAttention
                self.attention = CrossHeadAttention(
                    n_heads=n_heads,
                    channels_per_head=head_dim,
                    reduction=4,
                    dropout=attention_dropout
                )
            elif attention_type == 'mha':
                # 真正的多头注意力
                self.attention = CrossHeadMultiHeadAttention(
                    n_heads=n_heads,
                    head_dim=head_dim,
                    num_attn_heads=min(4, head_dim),  # 注意力头数
                    dropout=attention_dropout
                )
            elif attention_type == 'coda':
                # CoDA 风格
                self.attention = CoDAStyleAttention(
                    n_heads=n_heads,
                    head_dim=head_dim,
                    bottleneck=max(4, head_dim // 2),
                    dropout=attention_dropout
                )
            elif attention_type == 'hybrid':
                # 混合空间-频域
                self.attention = HybridSpatialFrequencyAttention(
                    n_heads=n_heads,
                    head_dim=head_dim,
                    num_modes=n_modes[0],
                    dropout=attention_dropout
                )
            else:
                raise ValueError(f"未知的注意力类型: {attention_type}")
    
    def _forward_1d(self, x: torch.Tensor) -> torch.Tensor:
        """带注意力的 1D 前向传播"""
        B, C, L = x.shape
        
        # FFT
        x_freq = torch.fft.rfft(x, dim=-1)
        n_modes = min(self.modes_list[0], x_freq.shape[-1])
        
        # 重塑为多头格式
        x_freq = x_freq.view(B, self.n_heads, self.head_in, -1)
        
        # 预分配输出
        out_freq = torch.zeros(
            B, self.n_heads, self.head_out, x_freq.shape[-1],
            dtype=x_freq.dtype, device=x.device
        )
        
        # 多头频域卷积
        out_freq[..., :n_modes] = torch.einsum(
            'bhif,hiof->bhof',
            x_freq[..., :n_modes],
            self.weight[..., :n_modes]
        )
        
        # IFFT
        out_freq_merged = out_freq.reshape(B, self.out_channels, -1)
        x_out_spatial = torch.fft.irfft(out_freq_merged, n=L, dim=-1)
        
        # 跨头注意力
        if self.use_attention:
            x_heads = x_out_spatial.view(B, self.n_heads, self.head_out, L)
            x_heads = self.attention(x_heads)
            x_out_spatial = x_heads.reshape(B, self.out_channels, L)
        
        if self.bias is not None:
            x_out_spatial = x_out_spatial + self.bias
        
        return x_out_spatial
    
    def _forward_2d(self, x: torch.Tensor) -> torch.Tensor:
        """带注意力的 2D 前向传播"""
        B, C, H, W = x.shape
        
        # 2D FFT
        x_freq = torch.fft.rfft2(x, dim=(-2, -1))
        freq_H, freq_W = x_freq.shape[-2], x_freq.shape[-1]
        
        m_x = min(self.modes_list[0], freq_H)
        m_y = min(self.weight.shape[-1], freq_W)
        
        # 重塑为多头格式
        x_freq = x_freq.view(B, self.n_heads, self.head_in, freq_H, freq_W)
        
        # 预分配输出
        out_freq = torch.zeros(
            B, self.n_heads, self.head_out, freq_H, freq_W,
            dtype=x_freq.dtype, device=x.device
        )
        
        # 多头频域卷积
        out_freq[:, :, :, :m_x, :m_y] = torch.einsum(
            'bhiXY,hioXY->bhoXY',
            x_freq[:, :, :, :m_x, :m_y],
            self.weight[:, :, :, :m_x, :m_y]
        )
        
        # IFFT
        out_freq_merged = out_freq.reshape(B, self.out_channels, freq_H, freq_W)
        x_out_spatial = torch.fft.irfft2(out_freq_merged, s=(H, W), dim=(-2, -1))
        
        # 跨头注意力
        if self.use_attention:
            x_heads = x_out_spatial.view(B, self.n_heads, self.head_out, H, W)
            x_heads = self.attention(x_heads)
            x_out_spatial = x_heads.reshape(B, self.out_channels, H, W)
        
        if self.bias is not None:
            x_out_spatial = x_out_spatial + self.bias
        
        return x_out_spatial
    
    def extra_repr(self) -> str:
        return (
            f'in_channels={self.in_channels}, out_channels={self.out_channels}, '
            f'n_modes={self.modes_list}, n_heads={self.n_heads}, '
            f'attention={self.attention_type}'
        )


def create_mhf_fno_v2(
    n_modes: Tuple[int, ...],
    hidden_channels: int,
    in_channels: int = 1,
    out_channels: int = 1,
    n_layers: int = 3,
    n_heads: int = 4,
    attention_type: str = 'mha',
    mhf_layers: Optional[List[int]] = None,
    attention_dropout: float = 0.0
) -> FNO:
    """
    创建指定注意力类型的 MHF-FNO 模型
    
    Args:
        attention_type: 注意力类型
            - 'none': 不使用注意力
            - 'senet': SENet 风格 (原实现)
            - 'mha': 多头注意力
            - 'coda': CoDA 风格
            - 'hybrid': 混合注意力
    """
    if mhf_layers is None:
        mhf_layers = [0, n_layers - 1]
    
    if hidden_channels % n_heads != 0:
        warnings.warn(
            f"hidden_channels ({hidden_channels}) 不能被 n_heads ({n_heads}) 整除",
            UserWarning
        )
    
    # 创建基础 FNO
    model = FNO(
        n_modes=n_modes,
        hidden_channels=hidden_channels,
        in_channels=in_channels,
        out_channels=out_channels,
        n_layers=n_layers
    )
    
    # 替换为带注意力的 MHF
    for layer_idx in mhf_layers:
        if layer_idx < n_layers:
            mhf_conv = MHFSpectralConvV2(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                n_modes=n_modes,
                n_heads=n_heads,
                attention_type=attention_type,
                attention_dropout=attention_dropout
            )
            model.fno_blocks.convs[layer_idx] = mhf_conv
    
    return model


__all__ = [
    'MHFSpectralConvV2',
    'create_mhf_fno_v2',
    'ATTENTION_VARIANTS',
]