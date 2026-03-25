"""
MHF-FNO Plugin

Multi-Head Fourier Neural Operator Plugin for NeuralOperator 2.0.0

使用示例:
    >>> from mhf_fno import MHFFNO, get_device
    >>> 
    >>> # 创建模型
    >>> model = MHFFNO.best_config(n_modes=(8, 8), hidden_channels=32)
    >>> 
    >>> # 移动到 GPU (如果可用)
    >>> device = get_device()
    >>> model = model.to(device)
    >>> 
    >>> # 前向传播
    >>> x = torch.randn(4, 1, 16, 16).to(device)
    >>> y = model(x)
    
带跨头注意力的 MHF-FNO (v1.2.0 新增):
    
    >>> from mhf_fno import MHFFNOWithAttention
    >>> 
    >>> # 推荐配置：带注意力
    >>> model = MHFFNOWithAttention.best_config(n_modes=(8, 8), hidden_channels=32)
    >>> 
    >>> # 全注意力配置：所有层都有注意力
    >>> model = MHFFNOWithAttention.full_attention_config(n_modes=(8, 8))
"""

from .mhf_fno import (
    MHFSpectralConv, 
    create_hybrid_fno, 
    MHFFNO,
    PINOLoss,
    get_device,
    check_cuda_memory
)

from .mhf_attention import (
    CrossHeadAttention,
    MHFSpectralConvWithAttention,
    create_mhf_fno_with_attention,
    MHFFNOWithAttention
)

__version__ = "1.2.0"
__author__ = "Tianyuan Team"

__all__ = [
    # MHF-FNO 基础功能
    "MHFSpectralConv", 
    "create_hybrid_fno", 
    "MHFFNO",
    "PINOLoss",
    "get_device",
    "check_cuda_memory",
    # MHF-FNO 跨头注意力 (v1.2.0 新增)
    "CrossHeadAttention",
    "MHFSpectralConvWithAttention",
    "create_mhf_fno_with_attention",
    "MHFFNOWithAttention"
]