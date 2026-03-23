"""
MHF-FNO Plugin

Multi-Head Fourier Neural Operator Plugin for NeuralOperator 2.0.0
"""

from .mhf_fno import MHFSpectralConv, create_hybrid_fno, MHFFNO

__version__ = "1.0.0"
__all__ = ["MHFSpectralConv", "create_hybrid_fno", "MHFFNO"]