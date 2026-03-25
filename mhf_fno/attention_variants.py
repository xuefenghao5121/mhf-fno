"""
跨头注意力变体实现
==================

包含多种注意力机制实现，用于对比实验：
1. CrossHeadMultiHeadAttention - 真正的跨头多头注意力
2. FrequencyDomainAttention - 频域注意力
3. HybridSpatialFrequencyAttention - 混合空间-频域注意力

作者: Tianyuan Team - 天渠
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class CrossHeadMultiHeadAttention(nn.Module):
    """
    真正的跨头多头注意力机制
    
    与 SENet 风格的区别：
    - SENet: 全局平均池化 -> 标量门控 (注意力权重只是一个标量)
    - 本实现: 完整的 Q/K/V 注意力 -> 向量加权 (每个位置有不同的注意力权重)
    
    工作流程：
    1. 对每个空间位置，将 n_heads 维度视为序列长度
    2. 做 n_heads -> n_heads 的注意力
    3. 每个头可以关注其他头的特征
    
    参数量分析：
    - Q/K/V 投影: 3 * head_dim * head_dim = 3 * 8 * 8 = 192 (per head)
    - 输出投影: head_dim * head_dim = 64 (per head)
    - 总计: 约 256 参数 per head
    
    Attributes:
        n_heads: 头数量
        head_dim: 每个头的维度
        num_attn_heads: 注意力头数
    """
    
    def __init__(
        self,
        n_heads: int,
        head_dim: int,
        num_attn_heads: int = 4,
        dropout: float = 0.0,
        use_bias: bool = True
    ):
        super().__init__()
        
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.num_attn_heads = num_attn_heads
        self.scale = (head_dim // num_attn_heads) ** -0.5
        
        # 多头注意力投影
        # 将每个头的特征进一步拆分为 num_attn_heads 个子头
        assert head_dim % num_attn_heads == 0, "head_dim 必须能被 num_attn_heads 整除"
        self.sub_head_dim = head_dim // num_attn_heads
        
        # Q, K, V 投影 (对每个 MHF head)
        self.q_proj = nn.Linear(head_dim, head_dim, bias=use_bias)
        self.k_proj = nn.Linear(head_dim, head_dim, bias=use_bias)
        self.v_proj = nn.Linear(head_dim, head_dim, bias=use_bias)
        self.out_proj = nn.Linear(head_dim, head_dim, bias=use_bias)
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        
        # 层归一化
        self.norm = nn.LayerNorm(head_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [B, n_heads, head_dim, H, W] (2D) 或 [B, n_heads, head_dim, L] (1D)
        
        Returns:
            输出张量，形状与输入相同
        """
        original_shape = x.shape
        B = x.shape[0]
        spatial_dims = x.shape[3:]  # (H, W) or (L,)
        
        # 重塑为 [B, H*W, n_heads, head_dim] (2D) 或 [B, L, n_heads, head_dim] (1D)
        if x.dim() == 4:  # 1D: [B, n_heads, C, L]
            L = x.shape[3]
            x = x.permute(0, 3, 1, 2)  # [B, L, n_heads, head_dim]
        else:  # 2D: [B, n_heads, C, H, W]
            H, W = x.shape[3], x.shape[4]
            x = x.permute(0, 3, 4, 1, 2).reshape(B, H * W, self.n_heads, self.head_dim)
        
        # 残差
        residual = x
        
        # 层归一化
        x = self.norm(x)
        
        # Q, K, V 投影
        # x: [B, spatial, n_heads, head_dim]
        Q = self.q_proj(x)  # [B, spatial, n_heads, head_dim]
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # 重塑为多头注意力格式
        # [B, spatial, n_heads, num_attn_heads, sub_head_dim]
        spatial_size = x.shape[1]
        Q = Q.view(B, spatial_size, self.n_heads, self.num_attn_heads, self.sub_head_dim)
        K = K.view(B, spatial_size, self.n_heads, self.num_attn_heads, self.sub_head_dim)
        V = V.view(B, spatial_size, self.n_heads, self.num_attn_heads, self.sub_head_dim)
        
        # 转置: [B, spatial, num_attn_heads, n_heads, sub_head_dim]
        Q = Q.transpose(2, 3)
        K = K.transpose(2, 3)
        V = V.transpose(2, 3)
        
        # 计算注意力: [B, spatial, num_attn_heads, n_heads, n_heads]
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        
        # 应用注意力: [B, spatial, num_attn_heads, n_heads, sub_head_dim]
        out = torch.matmul(attn, V)
        
        # 重塑回原始维度: [B, spatial, n_heads, head_dim]
        out = out.transpose(2, 3).reshape(B, spatial_size, self.n_heads, self.head_dim)
        
        # 输出投影
        out = self.out_proj(out)
        out = self.proj_dropout(out)
        
        # 残差连接
        out = residual + out
        
        # 恢复原始形状
        if len(original_shape) == 4:  # 1D
            out = out.permute(0, 2, 3, 1)  # [B, n_heads, head_dim, L]
        else:  # 2D
            out = out.reshape(B, H, W, self.n_heads, self.head_dim)
            out = out.permute(0, 3, 4, 1, 2)  # [B, n_heads, head_dim, H, W]
        
        return out


class FrequencyDomainAttention(nn.Module):
    """
    频域注意力机制
    
    在频域直接做注意力，而非空间域。
    优势：
    - 频域表示更紧凑，计算效率更高
    - 自然地捕捉频率间的交互
    - 更符合 FNO 的设计理念
    
    实现：
    1. 输入频域特征 (复数)
    2. 分离实部/虚部
    3. 在频率模式上做注意力
    4. 重新组合为复数输出
    
    注意：此模块应在频域卷积之后、IFFT 之前应用
    """
    
    def __init__(
        self,
        n_heads: int,
        head_dim: int,
        num_modes: int,
        dropout: float = 0.0
    ):
        """
        Args:
            n_heads: MHF 头数量
            head_dim: 每个头的维度
            num_modes: 频率模式数量
            dropout: Dropout 概率
        """
        super().__init__()
        
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.num_modes = num_modes
        
        # 复数到实数投影 (实部 + 虚部 = 2 * head_dim)
        self.real_proj = nn.Linear(head_dim * 2, head_dim)
        self.imag_proj = nn.Linear(head_dim * 2, head_dim)
        
        # 频率注意力 (在模式维度)
        self.freq_attn = nn.MultiheadAttention(
            embed_dim=head_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # 层归一化
        self.norm = nn.LayerNorm(head_dim)
        
        # 门控参数
        self.gate = nn.Parameter(torch.ones(1) * 0.1)
        
    def forward(self, x_freq: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x_freq: 频域特征 [B, n_heads, head_dim, freq_H, freq_W] (2D 复数)
                   或 [B, n_heads, head_dim, freq_L] (1D 复数)
        
        Returns:
            增强后的频域特征，形状相同
        """
        if not x_freq.is_complex():
            raise ValueError("FrequencyDomainAttention 期望复数输入")
        
        original_shape = x_freq.shape
        B = x_freq.shape[0]
        
        # 分离实部和虚部
        x_real = x_freq.real
        x_imag = x_freq.imag
        
        # 根据维度处理
        if x_freq.dim() == 4:  # 1D: [B, n_heads, head_dim, freq_L]
            freq_L = x_freq.shape[3]
            # 合并头维度: [B, freq_L, n_heads * head_dim]
            x_real_flat = x_real.permute(0, 3, 1, 2).reshape(B, freq_L, -1)
            x_imag_flat = x_imag.permute(0, 3, 1, 2).reshape(B, freq_L, -1)
            
            # 拼接实部虚部: [B, freq_L, 2 * n_heads * head_dim]
            x_combined = torch.cat([x_real_flat, x_imag_flat], dim=-1)
            
            # 频率注意力
            x_norm = self.norm(x_combined)
            attn_out, _ = self.freq_attn(x_norm, x_norm, x_norm)
            
            # 门控融合
            out_combined = x_combined + self.gate * attn_out
            
            # 分离实部虚部
            out_real = out_combined[..., :x_real_flat.shape[-1]]
            out_imag = out_combined[..., x_real_flat.shape[-1]:]
            
            # 恢复形状
            out_real = out_real.reshape(B, freq_L, self.n_heads, self.head_dim).permute(0, 2, 3, 1)
            out_imag = out_imag.reshape(B, freq_L, self.n_heads, self.head_dim).permute(0, 2, 3, 1)
            
        else:  # 2D: [B, n_heads, head_dim, freq_H, freq_W]
            freq_H, freq_W = x_freq.shape[3], x_freq.shape[4]
            
            # 展平空间频率: [B, n_heads, head_dim, freq_H * freq_W]
            x_real_flat = x_real.reshape(B, self.n_heads, self.head_dim, -1)
            x_imag_flat = x_imag.reshape(B, self.n_heads, self.head_dim, -1)
            
            # 转置: [B, freq_H * freq_W, n_heads * head_dim]
            x_real_flat = x_real_flat.permute(0, 3, 1, 2).reshape(B, -1, self.n_heads * self.head_dim)
            x_imag_flat = x_imag_flat.permute(0, 3, 1, 2).reshape(B, -1, self.n_heads * self.head_dim)
            
            # 拼接实部虚部
            x_combined = torch.cat([x_real_flat, x_imag_flat], dim=-1)
            
            # 频率注意力
            x_norm = self.norm(x_combined)
            attn_out, _ = self.freq_attn(x_norm, x_norm, x_norm)
            
            # 门控融合
            out_combined = x_combined + self.gate * attn_out
            
            # 分离实部虚部
            out_real = out_combined[..., :x_real_flat.shape[-1]]
            out_imag = out_combined[..., x_real_flat.shape[-1]:]
            
            # 恢复形状
            out_real = out_real.reshape(B, freq_H * freq_W, self.n_heads, self.head_dim)
            out_real = out_real.permute(0, 2, 3, 1).reshape(B, self.n_heads, self.head_dim, freq_H, freq_W)
            out_imag = out_imag.reshape(B, freq_H * freq_W, self.n_heads, self.head_dim)
            out_imag = out_imag.permute(0, 2, 3, 1).reshape(B, self.n_heads, self.head_dim, freq_H, freq_W)
        
        # 重新组合为复数
        out = torch.complex(out_real, out_imag)
        
        return out


class HybridSpatialFrequencyAttention(nn.Module):
    """
    混合空间-频域注意力
    
    同时在空间域和频域做注意力，然后融合。
    
    理论依据：
    - 空间域注意力：捕捉局部模式和全局依赖
    - 频域注意力：捕捉频率间耦合（对于 PDE 求解很重要）
    - 融合：结合两者优势
    
    参数量：
    - 空间注意力: ~256 * n_heads
    - 频域注意力: ~512 * n_heads
    - 融合层: ~128
    总计约 1K 参数 (相比 MHF 卷积的 ~16K 参数，增加 <10%)
    """
    
    def __init__(
        self,
        n_heads: int,
        head_dim: int,
        num_modes: int,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.n_heads = n_heads
        self.head_dim = head_dim
        
        # 空间注意力 (简化版跨头注意力)
        self.spatial_attn = nn.Sequential(
            nn.LayerNorm(head_dim),
            nn.Linear(head_dim, head_dim),
            nn.GELU(),
            nn.Linear(head_dim, head_dim),
        )
        
        # 频域注意力 (在 IFFT 后的特征上)
        self.freq_attn = nn.Sequential(
            nn.LayerNorm(head_dim),
            nn.Linear(head_dim, head_dim),
            nn.GELU(),
            nn.Linear(head_dim, head_dim),
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(head_dim * 2, head_dim),
            nn.GELU(),
            nn.Linear(head_dim, head_dim),
        )
        
        # 门控参数
        self.spatial_gate = nn.Parameter(torch.ones(1) * 0.3)
        self.freq_gate = nn.Parameter(torch.ones(1) * 0.3)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x_spatial: torch.Tensor,
        x_freq_flat: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x_spatial: 空间域特征 [B, n_heads, head_dim, H, W] 或 [B, n_heads, head_dim, L]
            x_freq_flat: 频域特征的展平版本 (可选)
        
        Returns:
            增强后的空间域特征
        """
        original_shape = x_spatial.shape
        B = x_spatial.shape[0]
        
        # 空间注意力
        if x_spatial.dim() == 4:  # 1D
            x_spatial_perm = x_spatial.permute(0, 3, 1, 2)  # [B, L, n_heads, head_dim]
        else:  # 2D
            H, W = x_spatial.shape[3], x_spatial.shape[4]
            x_spatial_perm = x_spatial.permute(0, 3, 4, 1, 2).reshape(B, -1, self.n_heads, self.head_dim)
        
        # 全局平均池化 (在空间维度)
        spatial_pooled = x_spatial_perm.mean(dim=1)  # [B, n_heads, head_dim]
        
        # 空间注意力增强
        spatial_enhanced = self.spatial_attn(spatial_pooled)  # [B, n_heads, head_dim]
        spatial_enhanced = spatial_pooled + torch.sigmoid(self.spatial_gate) * spatial_enhanced
        
        # 频域注意力 (使用 FFT 获取频域特征)
        if x_freq_flat is None:
            # 从空间域计算频域特征
            if x_spatial.dim() == 4:  # 1D
                x_freq = torch.fft.rfft(x_spatial, dim=-1)
                freq_pooled = torch.abs(x_freq).mean(dim=-1)  # [B, n_heads, head_dim]
            else:  # 2D
                x_freq = torch.fft.rfft2(x_spatial, dim=(-2, -1))
                freq_pooled = torch.abs(x_freq).mean(dim=(-2, -1))  # [B, n_heads, head_dim]
        else:
            freq_pooled = x_freq_flat
        
        # 频域注意力增强
        freq_enhanced = self.freq_attn(freq_pooled)  # [B, n_heads, head_dim]
        freq_enhanced = freq_pooled + torch.sigmoid(self.freq_gate) * freq_enhanced
        
        # 融合
        combined = torch.cat([spatial_enhanced, freq_enhanced], dim=-1)  # [B, n_heads, head_dim * 2]
        fused = self.fusion(combined)  # [B, n_heads, head_dim]
        
        # 广播回原始空间维度
        if x_spatial.dim() == 4:  # 1D
            fused = fused.unsqueeze(2)  # [B, n_heads, head_dim, 1]
        else:  # 2D
            fused = fused.unsqueeze(-1).unsqueeze(-1)  # [B, n_heads, head_dim, 1, 1]
        
        # 残差连接 + 调制
        out = x_spatial * (1 + 0.1 * fused)
        
        return out


class CoDAStyleAttention(nn.Module):
    """
    CoDA-NO 风格的跨头注意力
    
    参考: CoDA-NO: Continuous-Discrete Augmented Neural Operator
    
    核心思想：
    - 使用连续-离散混合表示
    - 跨头注意力学习频率模式之间的连续交互
    - 参数高效的瓶颈结构
    
    架构：
    1. 头特征压缩: [n_heads, head_dim] -> [n_heads, bottleneck]
    2. 跨头注意力: 在压缩空间做注意力
    3. 特征重建: [n_heads, bottleneck] -> [n_heads, head_dim]
    """
    
    def __init__(
        self,
        n_heads: int,
        head_dim: int,
        bottleneck: int = 4,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.bottleneck = bottleneck
        
        # 瓶颈压缩
        self.compress = nn.Linear(head_dim, bottleneck)
        
        # 跨头注意力 (在瓶颈空间)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=bottleneck,
            num_heads=2,  # 小瓶颈用少量头
            dropout=dropout,
            batch_first=True
        )
        
        # 重建
        self.expand = nn.Linear(bottleneck, head_dim)
        
        # 层归一化
        self.norm = nn.LayerNorm(head_dim)
        
        # 门控
        self.gate = nn.Parameter(torch.ones(1) * 0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: [B, n_heads, head_dim, H, W] 或 [B, n_heads, head_dim, L]
        
        Returns:
            增强后的特征
        """
        original_shape = x.shape
        B = x.shape[0]
        
        # 空间池化
        if x.dim() == 4:  # 1D
            x_pooled = x.mean(dim=-1)  # [B, n_heads, head_dim]
        else:  # 2D
            x_pooled = x.mean(dim=(-2, -1))
        
        residual = x_pooled
        
        # 压缩
        x_compressed = self.compress(x_pooled)  # [B, n_heads, bottleneck]
        
        # 跨头注意力
        x_attn, _ = self.cross_attn(x_compressed, x_compressed, x_compressed)
        
        # 重建
        x_expanded = self.expand(x_attn)  # [B, n_heads, head_dim]
        
        # 残差 + 门控
        out_pooled = residual + self.gate * x_expanded
        out_pooled = self.norm(out_pooled)
        
        # 计算增量
        delta = out_pooled - residual
        
        # 广播回空间维度
        if x.dim() == 4:  # 1D
            delta = delta.unsqueeze(-1)
        else:  # 2D
            delta = delta.unsqueeze(-1).unsqueeze(-1)
        
        # 应用增量
        out = x + delta
        
        return out


# 导出
__all__ = [
    'CrossHeadMultiHeadAttention',
    'FrequencyDomainAttention',
    'HybridSpatialFrequencyAttention',
    'CoDAStyleAttention',
]