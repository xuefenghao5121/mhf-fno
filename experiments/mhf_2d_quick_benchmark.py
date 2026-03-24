"""
MHF-FNO 2D 简化版高分辨率测试

只测试 32×32 和 64×64
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
from typing import Tuple


class MHFSpectralConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, n_modes, n_heads=4):
        super().__init__()
        
        assert in_channels % n_heads == 0
        assert out_channels % n_heads == 0
        
        self.n_heads = n_heads
        self.head_in = in_channels // n_heads
        self.head_out = out_channels // n_heads
        
        self.modes_x = n_modes[0]
        self.modes_y = n_modes[1] // 2 + 1
        
        init_std = (2 / (in_channels + out_channels)) ** 0.5
        
        self.weight = nn.Parameter(
            torch.randn(n_heads, self.head_in, self.head_out, self.modes_x, self.modes_y, 
                       dtype=torch.cfloat) * init_std
        )
        self.bias = nn.Parameter(init_std * torch.randn(out_channels, 1, 1))
    
    def forward(self, x):
        B, C, H, W = x.shape
        x_freq = torch.fft.rfft2(x, dim=(-2, -1))
        freq_H, freq_W = x_freq.shape[-2], x_freq.shape[-1]
        m_x, m_y = min(self.modes_x, freq_H), min(self.modes_y, freq_W)
        
        x_freq = x_freq.view(B, self.n_heads, self.head_in, freq_H, freq_W)
        out_freq = torch.zeros(B, self.n_heads, self.head_out, freq_H, freq_W,
                              dtype=x_freq.dtype, device=x.device)
        out_freq[:, :, :, :m_x, :m_y] = torch.einsum(
            'bhiXY,hioXY->bhoXY', x_freq[:, :, :, :m_x, :m_y], self.weight[:, :, :, :m_x, :m_y])
        out_freq = out_freq.reshape(B, self.head_out * self.n_heads, freq_H, freq_W)
        
        x_out = torch.fft.irfft2(out_freq, s=(H, W), dim=(-2, -1))
        return x_out + self.bias


class MHFFNO2D(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=32, n_modes=(8, 8), n_layers=3, n_heads=4):
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
        return self.projection(x)


class SpectralConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, n_modes):
        super().__init__()
        self.modes_x = n_modes[0]
        self.modes_y = n_modes[1] // 2 + 1
        self.weight = nn.Parameter(
            torch.randn(in_channels, out_channels, self.modes_x, self.modes_y, dtype=torch.cfloat) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_channels, 1, 1))
    
    def forward(self, x):
        B, C, H, W = x.shape
        x_freq = torch.fft.rfft2(x, dim=(-2, -1))
        freq_H, freq_W = x_freq.shape[-2], x_freq.shape[-1]
        m_x, m_y = min(self.modes_x, freq_H), min(self.modes_y, freq_W)
        
        out_freq = torch.zeros(B, self.weight.shape[1], freq_H, freq_W, dtype=x_freq.dtype, device=x.device)
        out_freq[:, :, :m_x, :m_y] = torch.einsum('biXY,ioXY->boXY', 
            x_freq[:, :, :m_x, :m_y], self.weight[:, :, :m_x, :m_y])
        
        x_out = torch.fft.irfft2(out_freq, s=(H, W), dim=(-2, -1))
        return x_out + self.bias


class StandardFNO2D(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=32, n_modes=(8, 8), n_layers=3):
        super().__init__()
        self.lifting = nn.Conv2d(in_channels, hidden_channels, 1)
        self.convs = nn.ModuleList([
            SpectralConv2D(hidden_channels, hidden_channels, n_modes) for _ in range(n_layers)
        ])
        self.projection = nn.Conv2d(hidden_channels, out_channels, 1)
    
    def forward(self, x):
        x = self.lifting(x)
        for conv in self.convs:
            x = F.gelu(conv(x))
        return self.projection(x)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def test_resolution(res):
    print(f"\n{'='*50}")
    print(f"测试分辨率: {res}×{res}")
    print(f"{'='*50}")
    
    # 生成数据
    torch.manual_seed(42)
    n_train, n_test = 200, 50
    
    a_list = [torch.sigmoid(torch.randn(res, res)) * 2 + 0.5 for _ in range(n_train + n_test)]
    a_field = torch.stack(a_list).unsqueeze(1)
    
    u_list = []
    for i in range(n_train + n_test):
        a = a_field[i, 0]
        u = torch.fft.ifft2(torch.fft.fft2(a) * 0.1).real
        u = (u - u.min()) / (u.max() - u.min() + 1e-6)
        u_list.append(u)
    u_field = torch.stack(u_list).unsqueeze(1)
    
    train_x, train_y = a_field[:n_train], u_field[:n_train]
    test_x, test_y = a_field[n_train:], u_field[n_train:]
    
    hidden = 32
    n_modes = (min(8, res//4), min(8, res//4))
    n_layers = 3
    
    # FNO
    print("测试 FNO...")
    model_fno = StandardFNO2D(1, 1, hidden, n_modes, n_layers)
    params_fno = count_params(model_fno)
    
    optimizer = torch.optim.Adam(model_fno.parameters(), lr=1e-3)
    start = time.time()
    for _ in range(30):
        perm = torch.randperm(n_train)
        for i in range(0, n_train, 16):
            bx, by = train_x[perm[i:i+16]], train_y[perm[i:i+16]]
            optimizer.zero_grad()
            F.mse_loss(model_fno(bx), by).backward()
            optimizer.step()
    time_fno = time.time() - start
    
    with torch.no_grad():
        l2_fno = (torch.norm(model_fno(test_x) - test_y) / torch.norm(test_y)).item()
    
    # MHF-FNO
    print("测试 MHF-FNO...")
    model_mhf = MHFFNO2D(1, 1, hidden, n_modes, n_layers)
    params_mhf = count_params(model_mhf)
    
    optimizer = torch.optim.Adam(model_mhf.parameters(), lr=1e-3)
    start = time.time()
    for _ in range(30):
        perm = torch.randperm(n_train)
        for i in range(0, n_train, 16):
            bx, by = train_x[perm[i:i+16]], train_y[perm[i:i+16]]
            optimizer.zero_grad()
            F.mse_loss(model_mhf(bx), by).backward()
            optimizer.step()
    time_mhf = time.time() - start
    
    with torch.no_grad():
        l2_mhf = (torch.norm(model_mhf(test_x) - test_y) / torch.norm(test_y)).item()
    
    print(f"\n{'模型':<12} {'参数量':<12} {'L2误差':<10} {'时间':<8}")
    print("-" * 45)
    print(f"{'FNO':<12} {params_fno:<12,} {l2_fno:<10.4f} {time_fno:<8.1f}s")
    print(f"{'MHF-FNO':<12} {params_mhf:<12,} {l2_mhf:<10.4f} {time_mhf:<8.1f}s")
    
    params_change = (params_mhf - params_fno) / params_fno * 100
    l2_change = (l2_mhf - l2_fno) / l2_fno * 100
    print(f"\n改进: 参数量 {params_change:+.1f}%, L2误差 {l2_change:+.1f}%")
    
    return {'FNO': params_fno, 'MHF': params_mhf, 'l2_fno': l2_fno, 'l2_mhf': l2_mhf}


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print(" MHF-FNO 2D 高分辨率测试")
    print("=" * 50)
    
    results = {}
    for res in [32, 64]:
        results[res] = test_resolution(res)
    
    print("\n" + "=" * 50)
    print(" 汇总")
    print("=" * 50)
    print(f"{'分辨率':<10} {'FNO参数':<12} {'MHF参数':<12} {'参数减少':<12}")
    print("-" * 50)
    for res in [32, 64]:
        r = results[res]
        change = (r['MHF'] - r['FNO']) / r['FNO'] * 100
        print(f"{res}×{res:<6} {r['FNO']:<12,} {r['MHF']:<12,} {change:+.1f}%")