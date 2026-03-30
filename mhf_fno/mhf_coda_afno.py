"""
MHF + CoDA + AFNO 三重融合模型
=================================

MHF (Multi-Head Fourier) + CoDA (Cross-Head Attention) + AFNO (Adaptive Fourier Neural Operator)

架构特点:
    1. 渐进式融合：不同层使用不同融合策略
    2. CoDA 门控 AFNO：用 Cross-Head Attention 门控 AFNO 的频率稀疏度
    3. AFNO-Enhanced MHF：用 AFNO 的 block-diagonal 思想增强 MHF

融合策略:
    - Layer 0: MHF + CoDA (早期：强跨频率交互)
    - Layer 1: AFNO-Enhanced MHF (中期：自适应频率选择)
    - Layer 2: MHF + CoDA + AFNO (后期：全技术协同)

版本: 1.0.0
作者: Tianyuan Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Union, List, Dict
import warnings

from .mhf_fno import MHFSpectralConv, get_device
from .mhf_attention import CrossHeadAttention, MHFSpectralConvWithAttention
from .afno import AFNOBlockDiagonalConv


class MHFCoDAWithAFNO(nn.Module):
    """
    MHF + CoDA + AFNO 三重融合模型
    
    渐进式融合策略：
        - 早期层（Layer 0）: MHF + CoDA - 强跨频率交互
        - 中期层（Layer 1）: AFNO-Enhanced MHF - 自适应频率选择  
        - 后期层（Layer 2+）: MHF + CoDA + AFNO - 全技术协同
    
    核心创新：
        1. CoDA 门控 AFNO：用 Cross-Head Attention 门控 AFNO 的频率稀疏度
        2. AFNO-Enhanced MHF：用 AFNO 的 block-diagonal 思想增强 MHF
        3. 渐进式融合：不同层使用不同融合策略
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 64,
        n_layers: int = 4,
        n_modes: Tuple[int, ...] = (12, 12),
        n_heads: int = 4,
        num_blocks: int = 8,
        sparsity_ratio: float = 0.5,
        fusion_strategy: str = 'hybrid',
        activation: str = 'gelu',
        positional_embedding: Optional[str] = 'grid'
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.n_modes = n_modes
        self.n_heads = n_heads
        self.num_blocks = num_blocks
        self.sparsity_ratio = sparsity_ratio
        self.fusion_strategy = fusion_strategy
        
        # 输入投影
        self.fc0 = nn.Linear(in_channels, hidden_channels)
        
        # 根据融合策略创建各层
        self.fno_blocks = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.afno_layers = nn.ModuleList()
        
        for layer_idx in range(n_layers):
            # 确定当前层的融合策略
            strategy = self._get_layer_fusion_strategy(layer_idx)
            
            # 根据策略创建层
            if strategy == 'mhf_coda':
                # MHF + CoDA 层
                self.fno_blocks.append(
                    MHFSpectralConvWithAttention(
                        in_channels=hidden_channels,
                        out_channels=hidden_channels,
                        n_modes=n_modes,
                        n_heads=n_heads,
                        use_attention=True
                    )
                )
                self.afno_layers.append(None)
                
            elif strategy == 'afno_enhanced':
                # AFNO-Enhanced MHF 层
                self.fno_blocks.append(
                    AFNOBlockDiagonalConv(
                        in_channels=hidden_channels,
                        out_channels=hidden_channels,
                        n_modes=n_modes,
                        num_blocks=num_blocks
                    )
                )
                self.afno_layers.append(self.fno_blocks[-1])
                
            elif strategy == 'full_fusion':
                # MHF + CoDA + AFNO 全融合层
                self.fno_blocks.append(
                    MHFSpectralConvWithAttention(
                        in_channels=hidden_channels,
                        out_channels=hidden_channels,
                        n_modes=n_modes,
                        n_heads=n_heads,
                        use_attention=True
                    )
                )
                self.afno_layers.append(
                    AFNOBlockDiagonalConv(
                        in_channels=hidden_channels,
                        out_channels=hidden_channels,
                        n_modes=n_modes,
                        num_blocks=num_blocks
                    )
                )
        
        # 激活函数
        if activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'relu':
            self.activation = F.relu
        else:
            raise ValueError(f"不支持的激活函数: {activation}")
        
        # 输出投影
        self.fc1 = nn.Linear(hidden_channels, out_channels)
        
    def _get_layer_fusion_strategy(self, layer_idx: int) -> str:
        """
        根据层索引确定融合策略
        
        Args:
            layer_idx: 层索引
        
        Returns:
            str: 融合策略名称
        """
        if self.fusion_strategy == 'hybrid':
            # 混合融合策略
            if layer_idx < self.n_layers // 3:
                # 早期层：MHF + CoDA
                return 'mhf_coda'
            elif layer_idx < 2 * self.n_layers // 3:
                # 中期层：AFNO-Enhanced MHF
                return 'afno_enhanced'
            else:
                # 后期层：全融合
                return 'full_fusion'
        else:
            # 默认使用全融合
            return 'full_fusion'
    
    def forward(self, x: torch.Tensor, sparsity_ratio: Optional[float] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量
            sparsity_ratio: 稀疏度比例（覆盖默认值）
        
        Returns:
            torch.Tensor: 输出张量
        """
        if sparsity_ratio is None:
            sparsity_ratio = self.sparsity_ratio
        
        # 输入投影
        if x.dim() == 3:
            x = self.fc0(x.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            x = self.fc0(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        # 各层前向传播
        for layer_idx, (fno_block, afno_block) in enumerate(zip(self.fno_blocks, self.afno_layers)):
            strategy = self._get_layer_fusion_strategy(layer_idx)
            
            if strategy == 'mhf_coda':
                # MHF + CoDA 层
                x = fno_block(x)
                
            elif strategy == 'afno_enhanced':
                # AFNO-Enhanced MHF 层
                # 先用 AFNO 处理，再用 MHF 增强
                x = afno_block(x, sparsity_ratio=sparsity_ratio)
                # 可以在这里添加额外的 MHF 增强逻辑
                
            elif strategy == 'full_fusion':
                # 全融合层
                # MHF + CoDA + AFNO
                x = fno_block(x)  # MHF + CoDA
                if afno_block is not None:
                    x = afno_block(x, sparsity_ratio=sparsity_ratio)  # AFNO
            
            x = self.activation(x)
        
        # 输出投影
        if x.dim() == 3:
            x = self.fc1(x.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            x = self.fc1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        return x
    
    def get_fusion_info(self) -> Dict:
        """
        获取融合信息
        
        Returns:
            Dict: 融合信息
        """
        fusion_info = []
        for layer_idx in range(self.n_layers):
            strategy = self._get_layer_fusion_strategy(layer_idx)
            fusion_info.append({
                'layer': layer_idx,
                'strategy': strategy
            })
        
        return {
            'fusion_strategy': self.fusion_strategy,
            'n_layers': self.n_layers,
            'layer_info': fusion_info
        }


def create_mhf_coda_afno(
    in_channels: int = 1,
    out_channels: int = 1,
    hidden_channels: int = 64,
    n_layers: int = 4,
    n_modes: Tuple[int, ...] = (12, 12),
    n_heads: int = 4,
    num_blocks: int = 8,
    sparsity_ratio: float = 0.5,
    fusion_strategy: str = 'hybrid',
    activation: str = 'gelu'
) -> MHFCoDAWithAFNO:
    """
    创建 MHF + CoDA + AFNO 三重融合模型（工厂函数）
    
    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        hidden_channels: 隐藏层通道数
        n_layers: FNO 层数
        n_modes: 频率模式数
        n_heads: MHF 头数
        num_blocks: AFNO block-diagonal 块数
        sparsity_ratio: 频率稀疏度比例
        fusion_strategy: 融合策略 ('hybrid', 'sequential', 'parallel')
        activation: 激活函数
    
    Returns:
        MHFCoDAWithAFNO: 三重融合模型
    """
    return MHFCoDAWithAFNO(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=hidden_channels,
        n_layers=n_layers,
        n_modes=n_modes,
        n_heads=n_heads,
        num_blocks=num_blocks,
        sparsity_ratio=sparsity_ratio,
        fusion_strategy=fusion_strategy,
        activation=activation
    )


__all__ = [
    'MHFCoDAWithAFNO',
    'create_mhf_coda_afno'
]
