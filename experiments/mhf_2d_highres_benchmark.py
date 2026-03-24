"""
MHF-FNO 2D 高分辨率测试

测试分辨率：32×32, 64×64, 128×128

对比：
- 参数量随分辨率变化
- L2 误差随分辨率变化
- 训练时间随分辨率变化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
from typing import Tuple


# ============================================
# MHF SpectralConv 2D
# ============================================

class MHFSpectralConv2D(nn.Module):
    """2D Multi-Head Fourier Spectral Convolution"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: Tuple[int, int],
        n_heads: int = 4,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes = n_modes
        self.n_heads = n_heads
        
        assert in_channels % n_heads == 0
        assert out_channels % n_heads == 0
        
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        x_freq = torch.fft.rfft2(x, dim=(-2, -1))
        
        freq_H, freq_W = x_freq.shape[-2], x_freq.shape[-1]
        m_x = min(self.modes_x, freq_H)
        m_y = min(self.modes_y, freq_W)
        
        x_freq = x_freq.view(B, self.n_heads, self.head_in, freq_H, freq_W)
        
        out_freq = torch.zeros(B, self.n_heads, self.head_out, freq_H, freq_W,
                              dtype=x_freq.dtype, device=x.device)
        out_freq[:, :, :, :m_x, :m_y] = torch.einsum(
            'bhiXY,hioXY->bhoXY',
            x_freq[:, :, :, :m_x, :m_y],
            self.weight[:, :, :, :m_x, :m_y]
        )
        
        out_freq = out_freq.reshape(B, self.out_channels, freq_H, freq_W)
        
        x_out = torch.fft.irfft2(out_freq, s=(H, W), dim=(-2, -1))
        x_out = x_out + self.bias
        
        return x_out


class MHFFNO2D(nn.Module):
    """MHF-FNO 2D"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 64,
        n_modes: Tuple[int, int] = (16, 16),
        n_layers: int = 4,
        n_heads: int = 4,
    ):
        super().__init__()
        
        self.lifting = nn.Conv2d(in_channels, hidden_channels, 1)
        
        self.convs = nn.ModuleList([
            MHFSpectralConv2D(hidden_channels, hidden_channels, n_modes, n_heads)
            for _ in range(n_layers)
        ])
        
        self.projection = nn.Conv2d(hidden_channels, out_channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lifting(x)
        
        for conv in self.convs:
            x = F.gelu(conv(x))
        
        x = self.projection(x)
        
        return x


# ============================================
# 标准 SpectralConv 2D
# ============================================

class SpectralConv2D(nn.Module):
    """标准 SpectralConv 2D"""
    
    def __init__(self, in_channels: int, out_channels: int, n_modes: Tuple[int, int]):
        super().__init__()
        
        self.n_modes = n_modes
        self.modes_x = n_modes[0]
        self.modes_y = n_modes[1] // 2 + 1
        
        self.weight = nn.Parameter(
            torch.randn(in_channels, out_channels, self.modes_x, self.modes_y, dtype=torch.cfloat) * 0.01
        )
        self.bias = nn.Parameter(torch.zeros(out_channels, 1, 1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        x_freq = torch.fft.rfft2(x, dim=(-2, -1))
        
        freq_H, freq_W = x_freq.shape[-2], x_freq.shape[-1]
        m_x = min(self.modes_x, freq_H)
        m_y = min(self.modes_y, freq_W)
        
        out_freq = torch.zeros(B, self.weight.shape[1], freq_H, freq_W,
                              dtype=x_freq.dtype, device=x.device)
        out_freq[:, :, :m_x, :m_y] = torch.einsum('biXY,ioXY->boXY', 
                                                   x_freq[:, :, :m_x, :m_y], 
                                                   self.weight[:, :, :m_x, :m_y])
        
        x_out = torch.fft.irfft2(out_freq, s=(H, W), dim=(-2, -1))
        x_out = x_out + self.bias
        
        return x_out


class StandardFNO2D(nn.Module):
    """标准 FNO 2D"""
    
    def __init__(self, in_channels, out_channels, hidden_channels=64, n_modes=(16, 16), n_layers=4):
        super().__init__()
        
        self.lifting = nn.Conv2d(in_channels, hidden_channels, 1)
        
        self.convs = nn.ModuleList([
            SpectralConv2D(hidden_channels, hidden_channels, n_modes)
            for _ in range(n_layers)
        ])
        
        self.projection = nn.Conv2d(hidden_channels, out_channels, 1)
    
    def forward(self, x):
        x = self.lifting(x)
        
        for conv in self.convs:
            x = F.gelu(conv(x))
        
        x = self.projection(x)
        
        return x


# ============================================
# 数据生成
# ============================================

def generate_darcy_2d_data(
    n_samples: int = 500,
    resolution: int = 64,
    seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor]:
    """生成 2D Darcy Flow 数据"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    a_list = []
    for _ in range(n_samples):
        a = torch.randn(resolution, resolution)
        a = torch.sigmoid(a) * 2 + 0.5
        a_list.append(a)
    
    a_field = torch.stack(a_list).unsqueeze(1)
    
    u_list = []
    for i in range(n_samples):
        a = a_field[i, 0]
        u = torch.fft.ifft2(torch.fft.fft2(a) * 0.1).real
        u = (u - u.min()) / (u.max() - u.min() + 1e-6)
        u_list.append(u)
    
    u_field = torch.stack(u_list).unsqueeze(1)
    
    return a_field, u_field


# ============================================
# 测试函数
# ============================================

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def train_and_evaluate(model, train_x, train_y, test_x, test_y, epochs=50, batch_size=16):
    """训练并评估"""
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    n_train = train_x.shape[0]
    n_batches = n_train // batch_size
    
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        
        perm = torch.randperm(n_train)
        train_x = train_x[perm]
        train_y = train_y[perm]
        
        for i in range(n_batches):
            bx = train_x[i*batch_size:(i+1)*batch_size]
            by = train_y[i*batch_size:(i+1)*batch_size]
            
            optimizer.zero_grad()
            pred = model(bx)
            loss = F.mse_loss(pred, by)
            loss.backward()
            optimizer.step()
    
    train_time = time.time() - start_time
    
    # 评估
    model.eval()
    with torch.no_grad():
        pred = model(test_x)
        l2_error = (torch.norm(pred - test_y) / torch.norm(test_y)).item()
    
    return l2_error, train_time


def test_resolution(resolution: int, n_train: int = 500, n_test: int = 100):
    """测试指定分辨率"""
    
    print(f"\n{'='*60}")
    print(f" 分辨率: {resolution}×{resolution}")
    print(f"{'='*60}")
    
    # 生成数据
    print(f"生成数据...")
    a_field, u_field = generate_darcy_2d_data(n_train + n_test, resolution)
    
    train_x = a_field[:n_train]
    train_y = u_field[:n_train]
    test_x = a_field[-n_test:]
    test_y = u_field[-n_test:]
    
    # 配置
    hidden_channels = 32
    n_modes = (min(16, resolution//2), min(16, resolution//2))
    n_layers = 3
    n_heads = 4
    epochs = 50
    
    results = {}
    
    # FNO
    print(f"测试 FNO...")
    model_fno = StandardFNO2D(1, 1, hidden_channels, n_modes, n_layers)
    params_fno = count_parameters(model_fno)
    l2_fno, time_fno = train_and_evaluate(model_fno, train_x, train_y, test_x, test_y, epochs)
    
    results['FNO'] = {'params': params_fno, 'l2': l2_fno, 'time': time_fno}
    
    # MHF-FNO
    print(f"测试 MHF-FNO...")
    model_mhf = MHFFNO2D(1, 1, hidden_channels, n_modes, n_layers, n_heads)
    params_mhf = count_parameters(model_mhf)
    l2_mhf, time_mhf = train_and_evaluate(model_mhf, train_x, train_y, test_x, test_y, epochs)
    
    results['MHF-FNO'] = {'params': params_mhf, 'l2': l2_mhf, 'time': time_mhf}
    
    # 打印结果
    print(f"\n{'模型':<15} {'参数量':<12} {'L2误差':<12} {'训练时间':<12}")
    print("-" * 55)
    for name, res in results.items():
        print(f"{name:<15} {res['params']:<12,} {res['l2']:<12.4f} {res['time']:<12.2f}s")
    
    # 改进
    params_change = (results['MHF-FNO']['params'] - results['FNO']['params']) / results['FNO']['params'] * 100
    l2_change = (results['MHF-FNO']['l2'] - results['FNO']['l2']) / results['FNO']['l2'] * 100
    time_change = (results['MHF-FNO']['time'] - results['FNO']['time']) / results['FNO']['time'] * 100
    
    print(f"\n改进: 参数量 {params_change:+.1f}%, L2误差 {l2_change:+.2f}%, 时间 {time_change:+.1f}%")
    
    return results


def main():
    print("\n" + "=" * 70)
    print(" MHF-FNO 2D 高分辨率测试")
    print("=" * 70)
    
    resolutions = [32, 64, 128]
    all_results = {}
    
    for res in resolutions:
        all_results[res] = test_resolution(res)
    
    # 汇总
    print("\n" + "=" * 70)
    print(" 📊 高分辨率测试汇总")
    print("=" * 70)
    
    print(f"\n{'分辨率':<10} {'FNO参数':<12} {'MHF参数':<12} {'参数减少':<12} {'FNO L2':<10} {'MHF L2':<10}")
    print("-" * 70)
    
    for res in resolutions:
        r = all_results[res]
        params_change = (r['MHF-FNO']['params'] - r['FNO']['params']) / r['FNO']['params'] * 100
        print(f"{res}×{res:<6} {r['FNO']['params']:<12,} {r['MHF-FNO']['params']:<12,} "
              f"{params_change:+.1f}%      {r['FNO']['l2']:<10.4f} {r['MHF-FNO']['l2']:<10.4f}")
    
    print("\n✅ 测试完成")


if __name__ == "__main__":
    main()