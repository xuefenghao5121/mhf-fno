"""
MHF-FNO 1D Implementation

Multi-Head Fourier Neural Operator for 1D problems
Based on TransFourier paper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MHFSpectralConv1D(nn.Module):
    """
    1D Multi-Head Fourier Spectral Convolution
    
    Key design:
    - Each head processes part of channels: head_in = in_ch // n_heads
    - Weight: (n_heads, head_in, head_out, n_modes)
    - Parameters reduced by n_heads times!
    """
    
    def __init__(self, in_channels, out_channels, n_modes, n_heads=4):
        super().__init__()
        
        assert in_channels % n_heads == 0
        assert out_channels % n_heads == 0
        
        self.n_heads = n_heads
        self.head_in = in_channels // n_heads
        self.head_out = out_channels // n_heads
        
        init_std = (2 / (in_channels + out_channels)) ** 0.5
        
        # Frequency domain weights for each head
        self.weight = nn.Parameter(
            torch.randn(n_heads, self.head_in, self.head_out, n_modes, 
                       dtype=torch.cfloat) * init_std
        )
        self.bias = nn.Parameter(init_std * torch.randn(out_channels, 1))
    
    def forward(self, x):
        B, C, L = x.shape
        
        # FFT
        x_freq = torch.fft.rfft(x, dim=-1)
        
        # Reshape to multi-head
        x_freq = x_freq.view(B, self.n_heads, self.head_in, -1)
        
        # Frequency mixing (multi-head independent)
        n_modes = min(self.weight.shape[-1], x_freq.shape[-1])
        out_freq = torch.einsum(
            'bhif,hiof->bhof', 
            x_freq[..., :n_modes], 
            self.weight[..., :n_modes]
        )
        
        # Merge heads
        out_freq = out_freq.reshape(B, self.head_out * self.n_heads, -1)
        
        # IFFT
        x_out = torch.fft.irfft(out_freq, n=L, dim=-1)
        x_out = x_out + self.bias
        
        return x_out


class MHFFNO1D(nn.Module):
    """MHF-FNO 1D Model"""
    
    def __init__(self, in_channels, out_channels, hidden_channels=64, 
                 n_modes=16, n_layers=4, n_heads=4):
        super().__init__()
        
        self.lifting = nn.Linear(in_channels, hidden_channels)
        
        self.convs = nn.ModuleList([
            MHFSpectralConv1D(hidden_channels, hidden_channels, n_modes, n_heads)
            for _ in range(n_layers)
        ])
        
        self.projection = nn.Linear(hidden_channels, out_channels)
    
    def forward(self, x):
        # x: (B, C, L)
        x = x.transpose(1, 2)  # (B, L, C)
        x = self.lifting(x)
        x = x.transpose(1, 2)  # (B, C, L)
        
        for conv in self.convs:
            x = F.gelu(conv(x))
        
        x = x.transpose(1, 2)  # (B, L, C)
        x = self.projection(x)
        x = x.transpose(1, 2)  # (B, C, L)
        
        return x


if __name__ == "__main__":
    # Test
    model = MHFFNO1D(2, 1, 32, 8, 2, 2)
    x = torch.randn(4, 2, 64)
    y = model(x)
    
    params = sum(p.numel() for p in model.parameters())
    print(f"Input: {x.shape}")
    print(f"Output: {y.shape}")
    print(f"Parameters: {params:,}")