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
"""

from .mhf_fno import (
    MHFSpectralConv, 
    create_hybrid_fno, 
    MHFFNO,
    get_device,
    check_cuda_memory
)

__version__ = "1.1.0"
__author__ = "Tianyuan Team"

__all__ = [
    "MHFSpectralConv", 
    "create_hybrid_fno", 
    "MHFFNO",
    "get_device",
    "check_cuda_memory"
]