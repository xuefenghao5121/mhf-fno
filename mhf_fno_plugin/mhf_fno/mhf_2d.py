"""
MHF-FNO 2D Implementation

Multi-Head Fourier Neural Operator for 2D problems
Based on TransFourier paper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class MHFSpectralConv2D(nn.Module):
    """
    2D Multi-Head Fourier Spectral Convolution
    
    Key design:
    - Each head processes part of channels: head_in = in_ch // n_heads
    - Weight: (n_heads, head_in, head_out, modes_x, modes_y)
    - Parameters reduced by n_heads times!
    """
    
    def __init__(self, in_channels, out_channels, n_modes: Tuple[int, int], n_heads=4):
        super().__init__()
        
        assert in_channels % n_heads == 0
        assert out_channels % n_heads == 0
        
        self.n_heads = n_heads
        self.head_in = in_channels // n_heads
        self.head_out = out_channels // n_heads
        
        self.modes_x = n_modes[0]
        self.modes_y = n_modes[1] // 2 + 1  # rfft2 last dimension
        
        init_std = (2 / (in_channels + out_channels)) ** 0.5
        
        # Frequency domain weights for each head
        self.weight = nn.Parameter(
            torch.randn(n_heads, self.head_in, self.head_out, 
                       self.modes_x, self.modes_y, dtype=torch.cfloat) * init_std
        )
        self.bias = nn.Parameter(init_std * torch.randn(out_channels, 1, 1))
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # 2D FFT
        x_freq = torch.fft.rfft2(x, dim=(-2, -1))
        
        freq_H, freq_W = x_freq.shape[-2], x_freq.shape[-1]
        m_x = min(self.modes_x, freq_H)
        m_y = min(self.modes_y, freq_W)
        
        # Reshape to multi-head
        x_freq = x_freq.view(B, self.n_heads, self.head_in, freq_H, freq_W)
        
        # Frequency mixing (multi-head independent)
        out_freq = torch.zeros(B, self.n_heads, self.head_out, freq_H, freq_W,
                              dtype=x_freq.dtype, device=x.device)
        out_freq[:, :, :, :m_x, :m_y] = torch.einsum(
            'bhiXY,hioXY->bhoXY',
            x_freq[:, :, :, :m_x, :m_y],
            self.weight[:, :, :, :m_x, :m_y]
        )
        
        # Merge heads
        out_freq = out_freq.reshape(B, self.head_out * self.n_heads, freq_H, freq_W)
        
        # IFFT
        x_out = torch.fft.irfft2(out_freq, s=(H, W), dim=(-2, -1))
        x_out = x_out + self.bias
        
        return x_out


class MHFFNO2D(nn.Module):
    """MHF-FNO 2D Model"""
    
    def __init__(self, in_channels=1, out_channels=1, hidden_channels=32,
                 n_modes: Tuple[int, int] = (8, 8), n_layers=3, n_heads=4):
        super().__init__()
        
        self.lifting = nn.Conv2d(in_channels, hidden_channels, 1)
        
        self.convs = nn.ModuleList([
            MHFSpectralConv2D(hidden_channels, hidden_channels, n_modes, n_heads)
            for _ in range(n_layers)
        ])
        
        self.projection = nn.Conv2d(hidden_channels, out_channels, 1)
    
    def forward(self, x):
        x = self.lifting(x)
        
        for conv in self.convs:
            x = F.gelu(conv(x))
        
        x = self.projection(x)
        
        return x


if __name__ == "__main__":
    # Test
    model = MHFFNO2D(1, 1, 32, (8, 8), 3, 4)
    x = torch.randn(4, 1, 16, 16)
    y = model(x)
    
    params = sum(p.numel() for p in model.parameters())
    print(f"Input: {x.shape}")
    print(f"Output: {y.shape}")
    print(f"Parameters: {params:,}")