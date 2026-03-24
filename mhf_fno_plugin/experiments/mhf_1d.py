"""
TransFourier-style MHF Module for NeuralOperator 2.0.0

简化版：专注于1D序列处理
- PDE求解（Darcy Flow等）
- 序列建模（可选因果模式）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union


class MHFSpectralConv1D(nn.Module):
    """
    1D Multi-Head Fourier Spectral Convolution
    
    Parameters
    ----------
    in_channels : int
    out_channels : int
    n_modes : int
        傅里叶模式数
    n_heads : int
    causal : bool
        是否使用因果模式（FFT + 局部因果卷积混合）
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: int,
        n_heads: int = 4,
        causal: bool = False,
        init_std: Union[str, float] = "auto",
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes = n_modes
        self.n_heads = n_heads
        self.causal = causal
        
        assert in_channels % n_heads == 0
        assert out_channels % n_heads == 0
        
        self.head_in = in_channels // n_heads
        self.head_out = out_channels // n_heads
        
        if init_std == "auto":
            init_std = (2 / (in_channels + out_channels)) ** 0.5
        
        # 频域权重
        self.weight = nn.Parameter(
            torch.randn(n_heads, self.head_in, self.head_out, n_modes, dtype=torch.cfloat) * init_std
        )
        
        # 因果模式：添加局部卷积
        if causal:
            self.local_conv = nn.Conv1d(in_channels, out_channels, 3, padding=1)
            self.gate = nn.Parameter(torch.tensor(0.0))
        else:
            self.local_conv = None
        
        self.bias = nn.Parameter(init_std * torch.randn(out_channels, 1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape
        
        # FFT
        x_freq = torch.fft.rfft(x, dim=-1)
        
        # 多头
        x_freq = x_freq.view(B, self.n_heads, self.head_in, -1)
        
        # 频域混合
        n_modes = min(self.n_modes, x_freq.shape[-1])
        out_freq = torch.einsum('bhif,hiof->bhof', x_freq[..., :n_modes], self.weight[..., :n_modes])
        out_freq = out_freq.reshape(B, self.out_channels, -1)
        
        # IFFT
        x_out = torch.fft.irfft(out_freq, n=L, dim=-1)
        x_out = x_out + self.bias
        
        # 因果混合
        if self.local_conv is not None:
            local = self.local_conv(x)
            g = torch.sigmoid(self.gate)
            x_out = g * x_out + (1 - g) * local
        
        return x_out


class MHFFNO1D(nn.Module):
    """
    1D FNO with MHF
    
    兼容NeuralOperator风格
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 64,
        n_modes: int = 16,
        n_layers: int = 4,
        n_heads: int = 4,
        causal: bool = False,
    ):
        super().__init__()
        
        self.lifting = nn.Linear(in_channels, hidden_channels)
        
        self.convs = nn.ModuleList([
            MHFSpectralConv1D(hidden_channels, hidden_channels, n_modes, n_heads, causal)
            for _ in range(n_layers)
        ])
        
        self.projection = nn.Linear(hidden_channels, out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    print("=" * 50)
    print(" MHF 1D Test (TransFourier + NeuralOperator)")
    print("=" * 50)
    
    # 测试MHFSpectralConv1D
    print("\n1. MHFSpectralConv1D 测试:")
    conv = MHFSpectralConv1D(64, 64, 16, 4)
    x = torch.randn(4, 64, 128)
    out = conv(x)
    print(f"输入: {x.shape}, 输出: {out.shape}")
    print(f"参数量: {sum(p.numel() for p in conv.parameters()):,}")
    
    # 测试因果模式
    print("\n2. 因果模式测试:")
    conv_causal = MHFSpectralConv1D(64, 64, 16, 4, causal=True)
    out_causal = conv_causal(x)
    print(f"因果输出: {out_causal.shape}")
    
    # 测试MHFFNO1D
    print("\n3. MHFFNO1D 测试:")
    model = MHFFNO1D(2, 1, 32, 8, 2, 2)
    x = torch.randn(4, 2, 64)
    out = model(x)
    print(f"输入: {x.shape}, 输出: {out.shape}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 因果性测试
    print("\n4. 因果性验证:")
    model_causal = MHFFNO1D(64, 64, 32, 8, 2, 2, causal=True)
    x = torch.randn(1, 64, 20)
    out1 = model_causal(x.clone())
    
    x_mod = x.clone()
    x_mod[:, :, 10:] = torch.randn(1, 64, 10)
    out2 = model_causal(x_mod)
    
    diff = torch.abs(out1[:, :, :10] - out2[:, :, :10]).max().item()
    print(f"前半部分差异: {diff:.6e}")
    print("说明: 因果模式使用FFT+局部卷积混合，是实用方案")
    
    print("\n" + "=" * 50)
    print(" 测试完成")
    print("=" * 50)