"""
MHF-FNO 2D 完美基准测试

基于 1D 成功设计扩展到 2D：
- 每个头处理部分通道
- 使用 einsum 高效计算
- 简洁架构，无额外复杂模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
from typing import Tuple


# ============================================
# 标准 SpectralConv 2D
# ============================================

class SpectralConv2D(nn.Module):
    """标准 SpectralConv 2D"""
    
    def __init__(self, in_channels: int, out_channels: int, n_modes: Tuple[int, int]):
        super().__init__()
        
        self.n_modes = n_modes
        self.modes_x = n_modes[0]
        self.modes_y = n_modes[1] // 2 + 1  # rfft2 的最后一维
        
        self.weight = nn.Parameter(
            torch.randn(in_channels, out_channels, self.modes_x, self.modes_y, dtype=torch.cfloat) * 0.01
        )
        self.bias = nn.Parameter(torch.zeros(out_channels, 1, 1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # 2D FFT
        x_freq = torch.fft.rfft2(x, dim=(-2, -1))
        
        freq_H, freq_W = x_freq.shape[-2], x_freq.shape[-1]
        m_x = min(self.modes_x, freq_H)
        m_y = min(self.modes_y, freq_W)
        
        # 频域混合
        out_freq = torch.zeros(B, self.weight.shape[1], freq_H, freq_W,
                              dtype=x_freq.dtype, device=x.device)
        out_freq[:, :, :m_x, :m_y] = torch.einsum('biXY,ioXY->boXY', 
                                                   x_freq[:, :, :m_x, :m_y], 
                                                   self.weight[:, :, :m_x, :m_y])
        
        # IFFT
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
# MHF SpectralConv 2D (TransFourier 风格)
# ============================================

class MHFSpectralConv2D(nn.Module):
    """
    2D Multi-Head Fourier Spectral Convolution
    
    关键设计（与 1D 成功版本一致）：
    - 每个头处理部分通道：head_in = in_ch // n_heads, head_out = out_ch // n_heads
    - 频域权重：(n_heads, head_in, head_out, modes_x, modes_y)
    - 参数量减少 n_heads 倍！
    """
    
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
        
        assert in_channels % n_heads == 0, f"in_channels ({in_channels}) 必须能被 n_heads ({n_heads}) 整除"
        assert out_channels % n_heads == 0, f"out_channels ({out_channels}) 必须能被 n_heads ({n_heads}) 整除"
        
        self.head_in = in_channels // n_heads
        self.head_out = out_channels // n_heads
        
        self.modes_x = n_modes[0]
        self.modes_y = n_modes[1] // 2 + 1  # rfft2 的最后一维
        
        init_std = (2 / (in_channels + out_channels)) ** 0.5
        
        # 频域权重：每个头独立
        # 参数量 = n_heads * head_in * head_out * modes_x * modes_y
        #       = (in_ch * out_ch * modes_x * modes_y) / n_heads
        # 相比标准 SpectralConv2D 减少了 n_heads 倍！
        self.weight = nn.Parameter(
            torch.randn(n_heads, self.head_in, self.head_out, self.modes_x, self.modes_y, 
                       dtype=torch.cfloat) * init_std
        )
        
        self.bias = nn.Parameter(init_std * torch.randn(out_channels, 1, 1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # 2D FFT
        x_freq = torch.fft.rfft2(x, dim=(-2, -1))
        
        freq_H, freq_W = x_freq.shape[-2], x_freq.shape[-1]
        m_x = min(self.modes_x, freq_H)
        m_y = min(self.modes_y, freq_W)
        
        # 重塑为多头：(B, C, H, W_freq) -> (B, n_heads, head_in, H, W_freq)
        x_freq = x_freq.view(B, self.n_heads, self.head_in, freq_H, freq_W)
        
        # 频域混合（多头独立）
        # einsum: (B, heads, head_in, X, Y) x (heads, head_in, head_out, X, Y) 
        #         -> (B, heads, head_out, X, Y)
        out_freq = torch.zeros(B, self.n_heads, self.head_out, freq_H, freq_W,
                              dtype=x_freq.dtype, device=x.device)
        out_freq[:, :, :, :m_x, :m_y] = torch.einsum(
            'bhiXY,hioXY->bhoXY',
            x_freq[:, :, :, :m_x, :m_y],
            self.weight[:, :, :, :m_x, :m_y]
        )
        
        # 合并多头：(B, n_heads, head_out, H, W_freq) -> (B, out_ch, H, W_freq)
        out_freq = out_freq.reshape(B, self.out_channels, freq_H, freq_W)
        
        # IFFT
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
# 数据生成
# ============================================

def generate_darcy_2d_data(
    n_samples: int = 1000,
    resolution: int = 32,
    seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor]:
    """生成 2D Darcy Flow 数据（简化版）"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 输入场：渗透率 a(x, y)
    a_list = []
    for _ in range(n_samples):
        # 随机生成 2D 渗透率场
        a = torch.randn(resolution, resolution)
        a = torch.sigmoid(a) * 2 + 0.5  # 归一化到 [0.5, 2.5]
        a_list.append(a)
    
    a_field = torch.stack(a_list).unsqueeze(1)  # (N, 1, H, W)
    
    # 输出场：压力 u(x, y) - 简化求解
    u_list = []
    for i in range(n_samples):
        a = a_field[i, 0]
        # 简化的 Poisson 解：u 受 a 的影响
        u = torch.fft.ifft2(torch.fft.fft2(a) * 0.1).real
        u = (u - u.min()) / (u.max() - u.min() + 1e-6)  # 归一化
        u_list.append(u)
    
    u_field = torch.stack(u_list).unsqueeze(1)  # (N, 1, H, W)
    
    return a_field, u_field


# ============================================
# 训练和评估
# ============================================

def train_model(model, train_x, train_y, val_x, val_y, epochs=100, lr=1e-3, batch_size=32):
    """训练模型"""
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    
    n_train = train_x.shape[0]
    n_batches = n_train // batch_size
    
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
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
            
            epoch_loss += loss.item()
        
        scheduler.step()
        
        # 验证
        if (epoch + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(val_x)
                val_l2 = (torch.norm(val_pred - val_y) / torch.norm(val_y)).item()
            print(f"Epoch {epoch+1}: Loss={epoch_loss/n_batches:.6f}, Val L2={val_l2:.4f}")
    
    train_time = time.time() - start_time
    return train_time


def evaluate_model(model, test_x, test_y):
    """评估模型"""
    
    model.eval()
    with torch.no_grad():
        pred = model(test_x)
        l2_error = (torch.norm(pred - test_y) / torch.norm(test_y)).item()
    
    return l2_error


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


# ============================================
# 主测试
# ============================================

def run_benchmark():
    print("\n" + "=" * 70)
    print(" MHF-FNO 2D 完美基准测试")
    print("=" * 70)
    
    # 配置
    n_train, n_val, n_test = 500, 100, 100
    resolution = 32
    hidden_channels = 32
    n_modes = (8, 8)
    n_layers = 3
    n_heads = 4
    epochs = 100
    
    # 生成数据
    print("\n📊 生成 2D Darcy Flow 数据...")
    a_field, u_field = generate_darcy_2d_data(n_train + n_val + n_test, resolution)
    
    train_x = a_field[:n_train]
    train_y = u_field[:n_train]
    val_x = a_field[n_train:n_train+n_val]
    val_y = u_field[n_train:n_train+n_val]
    test_x = a_field[-n_test:]
    test_y = u_field[-n_test:]
    
    print(f"   训练: {train_x.shape}, 验证: {val_x.shape}, 测试: {test_x.shape}")
    
    results = {}
    
    # ===== 测试标准 FNO =====
    print("\n" + "-" * 50)
    print("🔬 测试标准 FNO 2D")
    print("-" * 50)
    
    model_fno = StandardFNO2D(1, 1, hidden_channels, n_modes, n_layers)
    n_params_fno = count_parameters(model_fno)
    print(f"   参数量: {n_params_fno:,}")
    
    time_fno = train_model(model_fno, train_x, train_y, val_x, val_y, epochs=epochs)
    l2_fno = evaluate_model(model_fno, test_x, test_y)
    
    print(f"\n   ✅ L2误差: {l2_fno:.4f}, 时间: {time_fno:.2f}s")
    results['FNO'] = {'params': n_params_fno, 'l2': l2_fno, 'time': time_fno}
    
    # ===== 测试 MHF-FNO =====
    print("\n" + "-" * 50)
    print("🔬 测试 MHF-FNO 2D (TransFourier)")
    print("-" * 50)
    
    model_mhf = MHFFNO2D(1, 1, hidden_channels, n_modes, n_layers, n_heads)
    n_params_mhf = count_parameters(model_mhf)
    print(f"   参数量: {n_params_mhf:,}")
    
    time_mhf = train_model(model_mhf, train_x, train_y, val_x, val_y, epochs=epochs)
    l2_mhf = evaluate_model(model_mhf, test_x, test_y)
    
    print(f"\n   ✅ L2误差: {l2_mhf:.4f}, 时间: {time_mhf:.2f}s")
    results['MHF-FNO'] = {'params': n_params_mhf, 'l2': l2_mhf, 'time': time_mhf}
    
    # ===== 汇总 =====
    print("\n" + "=" * 70)
    print(" 📈 汇总对比")
    print("=" * 70)
    
    print(f"\n{'模型':<15} {'参数量':<15} {'L2误差':<12} {'训练时间':<12}")
    print("-" * 60)
    
    for name, res in results.items():
        print(f"{name:<15} {res['params']:<15,} {res['l2']:<12.4f} {res['time']:<12.2f}s")
    
    # 计算改进
    params_change = (results['MHF-FNO']['params'] - results['FNO']['params']) / results['FNO']['params'] * 100
    l2_change = (results['MHF-FNO']['l2'] - results['FNO']['l2']) / results['FNO']['l2'] * 100
    time_change = (results['MHF-FNO']['time'] - results['FNO']['time']) / results['FNO']['time'] * 100
    
    print(f"\n📊 MHF-FNO vs FNO:")
    print(f"   参数量变化: {params_change:+.1f}%")
    print(f"   L2误差变化: {l2_change:+.2f}%")
    print(f"   训练时间变化: {time_change:+.2f}%")
    
    # 判断
    print("\n" + "=" * 70)
    passed = 0
    if params_change < 0:
        print(f" ✅ 参数效率提升 ({-params_change:.1f}% 减少)")
        passed += 1
    if l2_change <= 5:
        print(f" ✅ 精度达标 (L2误差变化 {l2_change:+.1f}%)")
        passed += 1
    if time_change < 0:
        print(f" ✅ 训练加速 ({-time_change:.1f}% 减少)")
        passed += 1
    
    print(f"\n 通过 {passed}/3 项指标")
    
    return results


if __name__ == "__main__":
    run_benchmark()