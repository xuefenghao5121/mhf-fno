"""
MHF-PyramidUNO: 针对 UNO 的 MHF 新设计 (修正版)

核心创新:
1. 自适应 MHF 适配不规则 n_modes
2. 编码器/解码器使用不同的 n_heads
"""

import torch
import torch.nn as nn
from neuralop.layers.spectral_convolution import SpectralConv
from neuralop.losses.data_losses import LpLoss


class AdaptiveMHFConv(nn.Module):
    """
    自适应 MHF 卷积 (修正版)
    """
    
    def __init__(self, in_channels, out_channels, n_modes, n_heads=None):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 处理 n_modes
        if isinstance(n_modes, list):
            n_modes = tuple(n_modes)
        self.n_modes = n_modes
        
        # n_modes[1] 是 rfft 后的维度
        modes_x = n_modes[0]
        modes_y = n_modes[1] if len(n_modes) > 1 else n_modes[0]
        self.modes_x = modes_x
        self.modes_y = modes_y
        
        # 自动确定 n_heads
        if n_heads is None:
            import math
            gcd = math.gcd(in_channels, out_channels)
            self.n_heads = min(gcd, 4)  # 最多 4 头
        else:
            self.n_heads = n_heads
        
        self.use_mhf = (in_channels % self.n_heads == 0 and out_channels % self.n_heads == 0)
        
        if self.use_mhf:
            self.head_in = in_channels // self.n_heads
            self.head_out = out_channels // self.n_heads
            
            # MHF 权重: (n_heads, head_in, head_out, modes_x, modes_y)
            weight_shape = (self.n_heads, self.head_in, self.head_out, modes_x, modes_y)
            init_std = (2 / (in_channels + out_channels)) ** 0.5
            self.weight = nn.Parameter(torch.randn(*weight_shape, dtype=torch.cfloat) * init_std)
        else:
            # 标准 SpectralConv 权重
            self.weight = nn.Parameter(
                torch.randn(in_channels, out_channels, modes_x, modes_y, dtype=torch.cfloat) * 0.01
            )
        
        self.bias = nn.Parameter(torch.zeros(out_channels, 1, 1))
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # FFT (rfft2 后 W 维度变成 W//2+1)
        x_freq = torch.fft.rfft2(x, dim=(-2, -1))
        freq_H, freq_W = x_freq.shape[-2], x_freq.shape[-1]
        
        # 获取实际可用的 modes
        m_x = min(self.modes_x, freq_H)
        m_y = min(self.modes_y, freq_W)
        
        if self.use_mhf:
            # MHF 路径
            # x_freq: (B, C, freq_H, freq_W)
            # 重塑为 (B, n_heads, head_in, freq_H, freq_W)
            x_heads = x_freq.view(B, self.n_heads, self.head_in, freq_H, freq_W)
            
            # 输出
            out_heads = torch.zeros(B, self.n_heads, self.head_out, freq_H, freq_W,
                                    dtype=x_freq.dtype, device=x_freq.device)
            
            # 对每个头进行频域变换
            # weight: (n_heads, head_in, head_out, modes_x, modes_y)
            for h in range(self.n_heads):
                # x_heads[b, h, :, :m_x, :m_y] @ weight[h, :, :, :m_x, :m_y]
                # (head_in, m_x, m_y) @ (head_in, head_out, m_x, m_y) -> (head_out, m_x, m_y)
                for mx in range(m_x):
                    for my in range(m_y):
                        out_heads[:, h, :, mx, my] = torch.einsum(
                            'bi,io->bo',
                            x_heads[:, h, :, mx, my],
                            self.weight[h, :, :, mx, my]
                        )
            
            # 合并头
            out_freq = out_heads.reshape(B, self.out_channels, freq_H, freq_W)
        else:
            # 标准路径
            out_freq = torch.zeros(B, self.out_channels, freq_H, freq_W,
                                   dtype=x_freq.dtype, device=x_freq.device)
            for mx in range(m_x):
                for my in range(m_y):
                    out_freq[:, :, mx, my] = torch.einsum(
                        'bi,io->bo',
                        x_freq[:, :, mx, my],
                        self.weight[:, :, mx, my]
                    )
        
        # IFFT
        x_out = torch.fft.irfft2(out_freq, s=(H, W), dim=(-2, -1))
        x_out = x_out + self.bias
        
        return x_out


# ============================================
# 简化测试
# ============================================

def main():
    print("\n" + "=" * 70)
    print(" MHF 自适应卷积测试")
    print("=" * 70)
    
    # 测试不同配置
    configs = [
        {"in_ch": 32, "out_ch": 32, "n_modes": (8, 5), "n_heads": 4},
        {"in_ch": 32, "out_ch": 64, "n_modes": (8, 5), "n_heads": 4},
        {"in_ch": 64, "out_ch": 64, "n_modes": (8, 5), "n_heads": 4},
        {"in_ch": 128, "out_ch": 64, "n_modes": (8, 5), "n_heads": 4},
        {"in_ch": 96, "out_ch": 32, "n_modes": (8, 5), "n_heads": 4},
    ]
    
    for cfg in configs:
        print(f"\n测试: in={cfg['in_ch']}, out={cfg['out_ch']}, modes={cfg['n_modes']}")
        
        conv = AdaptiveMHFConv(cfg['in_ch'], cfg['out_ch'], cfg['n_modes'], cfg['n_heads'])
        
        # 前向传播
        x = torch.randn(2, cfg['in_ch'], 16, 16)
        try:
            y = conv(x)
            params = sum(p.numel() for p in conv.parameters())
            print(f"  ✅ 成功! 输出: {y.shape}, 参数: {params:,}")
        except Exception as e:
            print(f"  ❌ 失败: {e}")
    
    print("\n" + "=" * 70)
    print(" 测试完成")
    print("=" * 70)


if __name__ == "__main__":
    main()