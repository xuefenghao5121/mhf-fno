"""
MHF-FNO with Cross-Head Attention
===================================

解决 MHF-FNO 的独立性假设问题：
- 当前 MHF: 头之间完全独立，无法跨频率交互
- 问题: NS 等复杂 PDE 需要全频率耦合
- 解决方案: 添加跨头注意力机制，允许不同频率头之间的信息交换

设计原则:
- 参数高效：注意力模块参数量应远小于 MHF 卷积
- 计算高效：避免 O(N^2) 的全局注意力
- 保持 MHF 的优势：频率分离 + 头间交互

参考文献:
- TransFourier: Multi-Head Attention in Spectral Domain
- CoDA-NO: Continuous-Discrete Augmented Neural Operator
- Squeeze-and-Excitation Networks (SENet)

版本: 1.0.1
作者: Tianyuan Team - 天渠
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Union
import warnings

from neuralop.models import FNO
from neuralop.layers.spectral_convolution import SpectralConv

from .mhf_fno import MHFSpectralConv, get_device, check_cuda_memory


class CrossHeadAttention(nn.Module):
    """
    轻量级跨头注意力模块
    
    使用 Squeeze-and-Excitation 风格的通道注意力，
    在保持参数效率的同时实现头间信息交互。
    
    设计思路:
        1. 全局平均池化: [B, n_heads, C, H, W] -> [B, n_heads, C]
        2. 头间注意力: [B, n_heads, C] -> [B, n_heads, C]
        3. 门控融合: 注意力权重与原始特征相乘
    
    参数量:
        - 每个头约 2 * C^2 + 2 * C (Q,K,V投影 + FFN)
        - 相比展开空间维度的方案，参数减少 99%+
    
    Attributes:
        n_heads (int): MHF 头数量
        channels_per_head (int): 每个头的通道数
        reduction (int): 中间层缩减比例，默认 4
    """
    
    def __init__(
        self,
        n_heads: int,
        channels_per_head: int,
        reduction: int = 4,
        dropout: float = 0.0
    ):
        """
        初始化轻量级跨头注意力模块。
        
        Args:
            n_heads: MHF 的头数量
            channels_per_head: 每个头的通道数
            reduction: 中间层缩减比例，默认 4
            dropout: Dropout 概率，默认 0.0
        """
        super().__init__()
        
        self.n_heads = n_heads
        self.channels_per_head = channels_per_head
        
        # 嵌入维度 = 每个头的通道数
        self.embed_dim = channels_per_head
        hidden_dim = max(channels_per_head // reduction, 4)
        
        # 头间注意力层
        # Query/Key/Value 投影 (共享参数)
        self.query = nn.Linear(channels_per_head, channels_per_head, bias=False)
        self.key = nn.Linear(channels_per_head, channels_per_head, bias=False)
        self.value = nn.Linear(channels_per_head, channels_per_head, bias=False)
        
        # 缩放因子
        self.scale = channels_per_head ** -0.5
        
        # 输出投影 + FFN
        self.out_proj = nn.Linear(channels_per_head, channels_per_head)
        self.ffn = nn.Sequential(
            nn.Linear(channels_per_head, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, channels_per_head),
            nn.Dropout(dropout)
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(channels_per_head)
        self.norm2 = nn.LayerNorm(channels_per_head)
        
        # 门控参数
        self.gate = nn.Parameter(torch.ones(1) * 0.5)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。
        
        Args:
            x: 输入张量，形状为 (B, n_heads, C_per_head, ...) 
            
        Returns:
            torch.Tensor: 输出张量，形状与输入相同
        """
        B = x.shape[0]
        original_shape = x.shape
        spatial_dims = original_shape[3:]  # (H, W) or (L,)
        
        # Step 1: 全局平均池化 [B, n_heads, C, H, W] -> [B, n_heads, C]
        if x.dim() == 4:  # 1D: [B, n_heads, C, L]
            x_pooled = x.mean(dim=-1)  # [B, n_heads, C]
        else:  # 2D: [B, n_heads, C, H, W]
            x_pooled = x.mean(dim=(-2, -1))  # [B, n_heads, C]
        
        # Step 2: 头间注意力
        residual = x_pooled
        x_norm = self.norm1(x_pooled)
        
        # Q, K, V 投影
        Q = self.query(x_norm)  # [B, n_heads, C]
        K = self.key(x_norm)
        V = self.value(x_norm)
        
        # 计算注意力权重 [B, n_heads, n_heads]
        attn_weights = torch.matmul(Q, K.transpose(-1, -2)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # 注意力输出
        attn_out = torch.matmul(attn_weights, V)  # [B, n_heads, C]
        attn_out = self.out_proj(attn_out)
        
        # 残差连接
        x_attn = residual + attn_out
        
        # Step 3: FFN
        residual = x_attn
        x_norm = self.norm2(x_attn)
        ffn_out = self.ffn(x_norm)
        x_out = residual + ffn_out
        
        # Step 4: 门控融合
        # 将注意力增强的特征广播回原始空间
        # 使用门控机制混合原始特征和注意力增强特征
        gate = torch.sigmoid(self.gate)
        
        # x_pooled (原始池化) vs x_out (注意力增强)
        # 通过门控混合：out = gate * attention_enhanced + (1-gate) * original
        attention_weights = x_out - x_pooled  # 增量
        attention_weights = gate * attention_weights + x_pooled
        
        # 广播回空间维度并应用
        if x.dim() == 4:  # 1D
            attention_weights = attention_weights.unsqueeze(-1)  # [B, n_heads, C, 1]
        else:  # 2D
            attention_weights = attention_weights.unsqueeze(-1).unsqueeze(-1)  # [B, n_heads, C, 1, 1]
        
        # 逐元素调制
        out = x * (1 + attention_weights)
        
        return out


class MHFSpectralConvWithAttention(MHFSpectralConv):
    """
    带跨头注意力的多头频谱卷积层
    
    继承 MHFSpectralConv，添加轻量级跨头注意力机制。
    
    工作流程:
        1. 标准多头频域卷积 (继承自 MHFSpectralConv)
        2. 轻量级跨头注意力 (SENet 风格)
        3. 合并多头输出
    
    参数效率:
        - MHF 卷积: ~n_heads * (C/n_heads)^2 * modes
        - 注意力: ~3 * (C/n_heads)^2 + FFN
        - 注意力参数占 MHF 参数的 <1%
    
    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数  
        n_modes: 频率模式数元组
        n_heads: 多头数量，默认 4
        use_attention: 是否启用跨头注意力，默认 True
        attn_reduction: 注意力中间层缩减比例，默认 4
        attn_dropout: 注意力 dropout 概率，默认 0.0
        **kwargs: 传递给父类的其他参数
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: Tuple[int, ...],
        n_heads: int = 4,
        use_attention: bool = True,
        attn_reduction: int = 4,
        attn_dropout: float = 0.0,
        **kwargs
    ):
        # 初始化父类
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            n_modes=n_modes,
            n_heads=n_heads,
            **kwargs
        )
        
        self.use_attention = use_attention and self.use_mhf
        
        if self.use_attention:
            # 轻量级跨头注意力模块
            # 参数量: ~3 * head_out^2 + 2 * head_out * (head_out//reduction)
            # 对于 head_out=8: ~200 参数 (vs MHF 权重的 ~16K 参数)
            self.cross_head_attn = CrossHeadAttention(
                n_heads=self.n_heads,
                channels_per_head=self.head_out,
                reduction=attn_reduction,
                dropout=attn_dropout
            )
    
    def _forward_1d(self, x: torch.Tensor) -> torch.Tensor:
        """带注意力的 1D 前向传播"""
        B, C, L = x.shape
        
        # FFT
        x_freq = torch.fft.rfft(x, dim=-1)
        
        # 计算实际可用的模式数
        n_modes = min(self.modes_list[0], x_freq.shape[-1])
        
        # 重塑为多头格式
        x_freq = x_freq.view(B, self.n_heads, self.head_in, -1)
        
        # 预分配输出张量
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
        
        # IFFT 得到空域表示
        out_freq_merged = out_freq.reshape(B, self.out_channels, -1)
        x_out_spatial = torch.fft.irfft(out_freq_merged, n=L, dim=-1)
        
        # 跨头注意力（如果启用）
        if self.use_attention:
            x_heads = x_out_spatial.view(B, self.n_heads, self.head_out, L)
            x_heads = self.cross_head_attn(x_heads)
            x_out_spatial = x_heads.reshape(B, self.out_channels, L)
        
        # 添加偏置
        if self.bias is not None:
            x_out_spatial = x_out_spatial + self.bias
            
        return x_out_spatial
    
    def _forward_2d(self, x: torch.Tensor) -> torch.Tensor:
        """带注意力的 2D 前向传播"""
        B, C, H, W = x.shape
        
        # 2D FFT
        x_freq = torch.fft.rfft2(x, dim=(-2, -1))
        
        freq_H, freq_W = x_freq.shape[-2], x_freq.shape[-1]
        
        # 计算实际可用的模式数
        m_x = min(self.modes_list[0], freq_H)
        m_y = min(self.weight.shape[-1], freq_W)
        
        # 重塑为多头格式
        x_freq = x_freq.view(B, self.n_heads, self.head_in, freq_H, freq_W)
        
        # 预分配输出张量
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
        
        # IFFT 得到空域表示
        out_freq_merged = out_freq.reshape(B, self.out_channels, freq_H, freq_W)
        x_out_spatial = torch.fft.irfft2(out_freq_merged, s=(H, W), dim=(-2, -1))
        
        # 跨头注意力（如果启用）
        if self.use_attention:
            x_heads = x_out_spatial.view(B, self.n_heads, self.head_out, H, W)
            x_heads = self.cross_head_attn(x_heads)
            x_out_spatial = x_heads.reshape(B, self.out_channels, H, W)
        
        # 添加偏置
        if self.bias is not None:
            x_out_spatial = x_out_spatial + self.bias
            
        return x_out_spatial
    
    def extra_repr(self) -> str:
        """返回额外的模块表示信息。"""
        return (
            f'in_channels={self.in_channels}, out_channels={self.out_channels}, '
            f'n_modes={self.modes_list}, n_heads={self.n_heads}, '
            f'use_mhf={self.use_mhf}, use_attention={self.use_attention}'
        )


def create_mhf_fno_with_attention(
    n_modes: Tuple[int, ...],
    hidden_channels: int,
    in_channels: int = 1,
    out_channels: int = 1,
    n_layers: int = 3,
    mhf_layers: Optional[List[int]] = None,
    n_heads: int = 4,
    attention_layers: Optional[List[int]] = None,
    attn_dropout: float = 0.0,
    positional_embedding: Optional[str] = 'grid'
) -> FNO:
    """
    创建带跨头注意力的 MHF-FNO 模型。
    
    Args:
        n_modes: 频率模式数元组
        hidden_channels: 隐藏层通道数
        in_channels: 输入通道数，默认 1
        out_channels: 输出通道数，默认 1
        n_layers: FNO 层数，默认 3
        mhf_layers: 使用 MHF 的层索引列表，默认 [0, 2]（首尾层）
        n_heads: 多头数量，默认 4
        attention_layers: 使用注意力的层索引列表，默认与 mhf_layers 相同
        attn_dropout: 注意力 dropout 概率，默认 0.0
        positional_embedding: 位置嵌入类型，默认 'grid'，1D 数据建议设为 None
        
    Returns:
        FNO: 配置好的带注意力的 MHF-FNO 模型
    """
    if mhf_layers is None:
        mhf_layers = [0, n_layers - 1]
    
    if attention_layers is None:
        attention_layers = mhf_layers
    
    # 验证配置
    if hidden_channels % n_heads != 0:
        warnings.warn(
            f"hidden_channels ({hidden_channels}) 不能被 n_heads ({n_heads}) 整除，"
            f"MHF 将回退到标准卷积。",
            UserWarning
        )
    
    # 创建基础 FNO 模型
    model = FNO(
        n_modes=n_modes,
        hidden_channels=hidden_channels,
        in_channels=in_channels,
        out_channels=out_channels,
        n_layers=n_layers,
        positional_embedding=positional_embedding
    )
    
    # 替换指定层的卷积为带注意力的 MHF
    for layer_idx in mhf_layers:
        if layer_idx < n_layers:
            use_attn = layer_idx in attention_layers
            mhf_conv = MHFSpectralConvWithAttention(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                n_modes=n_modes,
                n_heads=n_heads,
                use_attention=use_attn,
                attn_dropout=attn_dropout
            )
            model.fno_blocks.convs[layer_idx] = mhf_conv
    
    return model


class MHFFNOWithAttention:
    """
    带跨头注意力的 MHF-FNO 预设配置工厂类。
    
    提供带注意力的 MHF-FNO 配置预设。
    
    Example:
        >>> # 推荐配置（带注意力）
        >>> model = MHFFNOWithAttention.best_config(n_modes=(8, 8), hidden_channels=32)
        >>> 
        >>> # 轻量配置（所有层使用 MHF + 注意力）
        >>> model = MHFFNOWithAttention.full_attention_config(n_modes=(8, 8))
    """
    
    @staticmethod
    def best_config(
        n_modes: Tuple[int, ...] = (8, 8),
        hidden_channels: int = 32,
        in_channels: int = 1,
        out_channels: int = 1,
        n_heads: int = 4,
        attn_dropout: float = 0.0,
        positional_embedding: Optional[str] = 'grid'
    ) -> FNO:
        """
        最佳配置预设: 首尾层使用带注意力的 MHF。
        
        Args:
            n_modes: 频率模式数，默认 (8, 8)
            hidden_channels: 隐藏通道数，默认 32
            in_channels: 输入通道数，默认 1
            out_channels: 输出通道数，默认 1
            n_heads: 多头数量，默认 4
            attn_dropout: 注意力 dropout 概率，默认 0.0
            positional_embedding: 位置嵌入类型，默认 'grid'，1D 数据建议设为 None
            
        Returns:
            FNO: 配置好的带注意力的 MHF-FNO 模型
        """
        return create_mhf_fno_with_attention(
            n_modes=n_modes,
            hidden_channels=hidden_channels,
            in_channels=in_channels,
            out_channels=out_channels,
            mhf_layers=[0, 2],
            n_heads=n_heads,
            attention_layers=[0, 2],
            attn_dropout=attn_dropout,
            positional_embedding=positional_embedding
        )
    
    @staticmethod
    def full_attention_config(
        n_modes: Tuple[int, ...] = (8, 8),
        hidden_channels: int = 32,
        in_channels: int = 1,
        out_channels: int = 1,
        n_layers: int = 3,
        n_heads: int = 4,
        attn_dropout: float = 0.0
    ) -> FNO:
        """
        全注意力配置: 所有层都使用带注意力的 MHF。
        
        适用于需要最大频率耦合的复杂 PDE 问题。
        
        Args:
            n_modes: 频率模式数
            hidden_channels: 隐藏通道数
            in_channels: 输入通道数
            out_channels: 输出通道数
            n_layers: 层数
            n_heads: 多头数量
            attn_dropout: 注意力 dropout 概率
            
        Returns:
            FNO: 配置好的全注意力 MHF-FNO 模型
        """
        return create_mhf_fno_with_attention(
            n_modes=n_modes,
            hidden_channels=hidden_channels,
            in_channels=in_channels,
            out_channels=out_channels,
            n_layers=n_layers,
            mhf_layers=list(range(n_layers)),
            n_heads=n_heads,
            attention_layers=list(range(n_layers)),
            attn_dropout=attn_dropout
        )
    
    @staticmethod
    def light_config(
        n_modes: Tuple[int, ...] = (8, 8),
        hidden_channels: int = 16,
        in_channels: int = 1,
        out_channels: int = 1,
        n_heads: int = 4,
        attn_dropout: float = 0.0
    ) -> FNO:
        """
        轻量配置: 所有层使用带注意力的 MHF，较小的隐藏通道数。
        
        参数量最少，适合资源受限场景。
        
        Args:
            n_modes: 频率模式数
            hidden_channels: 隐藏通道数
            in_channels: 输入通道数
            out_channels: 输出通道数
            n_heads: 多头数量
            attn_dropout: 注意力 dropout 概率
            
        Returns:
            FNO: 配置好的轻量 MHF-FNO 模型
        """
        return create_mhf_fno_with_attention(
            n_modes=n_modes,
            hidden_channels=hidden_channels,
            in_channels=in_channels,
            out_channels=out_channels,
            mhf_layers=[0, 1, 2],
            n_heads=n_heads,
            attention_layers=[0, 1, 2],
            attn_dropout=attn_dropout
        )


# 导出公共 API
__all__ = [
    'CrossHeadAttention',
    'MHFSpectralConvWithAttention',
    'create_mhf_fno_with_attention',
    'MHFFNOWithAttention',
    'get_device',
    'check_cuda_memory'
]