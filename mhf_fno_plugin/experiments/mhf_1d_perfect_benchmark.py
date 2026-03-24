"""
MHF-FNO 1D 完美基准测试

对比：
1. 标准 FNO (1D)
2. MHF-FNO (TransFourier 风格)

目标：
- 参数量减少
- L2 误差相近或更好
- 训练时间更短
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
from typing import Tuple


# ============================================
# 标准 SpectralConv 1D
# ============================================

class SpectralConv1D(nn.Module):
    """标准 SpectralConv 1D"""
    
    def __init__(self, in_channels: int, out_channels: int, n_modes: int):
        super().__init__()
        
        self.n_modes = n_modes
        
        self.weight = nn.Parameter(
            torch.randn(in_channels, out_channels, n_modes, dtype=torch.cfloat) * 0.01
        )
        self.bias = nn.Parameter(torch.zeros(out_channels, 1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape
        
        x_freq = torch.fft.rfft(x, dim=-1)
        
        n_modes = min(self.n_modes, x_freq.shape[-1])
        out_freq = torch.zeros(B, self.weight.shape[1], x_freq.shape[-1], 
                              dtype=x_freq.dtype, device=x.device)
        out_freq[:, :, :n_modes] = torch.einsum('bif,iOf->bOf', x_freq[:, :, :n_modes], self.weight)
        
        x_out = torch.fft.irfft(out_freq, n=L, dim=-1)
        x_out = x_out + self.bias
        
        return x_out


class StandardFNO1D(nn.Module):
    """标准 FNO 1D"""
    
    def __init__(self, in_channels, out_channels, hidden_channels=64, n_modes=16, n_layers=4):
        super().__init__()
        
        self.lifting = nn.Linear(in_channels, hidden_channels)
        
        self.convs = nn.ModuleList([
            SpectralConv1D(hidden_channels, hidden_channels, n_modes)
            for _ in range(n_layers)
        ])
        
        self.projection = nn.Linear(hidden_channels, out_channels)
    
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.lifting(x)
        x = x.transpose(1, 2)
        
        for conv in self.convs:
            x = F.gelu(conv(x))
        
        x = x.transpose(1, 2)
        x = self.projection(x)
        x = x.transpose(1, 2)
        
        return x


# ============================================
# MHF SpectralConv 1D (TransFourier 风格)
# ============================================

class MHFSpectralConv1D(nn.Module):
    """
    1D Multi-Head Fourier Spectral Convolution
    
    关键设计：
    - 每个头处理部分通道：head_in = in_ch // n_heads, head_out = out_ch // n_heads
    - 频域权重：(n_heads, head_in, head_out, n_modes)
    - 参数量：n_heads * head_in * head_out * n_modes = (in_ch * out_ch * n_modes) / n_heads
    - 相比标准 SpectralConv 参数量减少 n_heads 倍！
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: int,
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
        
        init_std = (2 / (in_channels + out_channels)) ** 0.5
        
        # 频域权重：每个头独立
        # 参数量 = n_heads * head_in * head_out * n_modes
        #       = (in_ch * out_ch * n_modes) / n_heads
        # 相比标准 SpectralConv 减少了 n_heads 倍！
        self.weight = nn.Parameter(
            torch.randn(n_heads, self.head_in, self.head_out, n_modes, dtype=torch.cfloat) * init_std
        )
        
        self.bias = nn.Parameter(init_std * torch.randn(out_channels, 1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape
        
        # FFT
        x_freq = torch.fft.rfft(x, dim=-1)
        
        # 重塑为多头：(B, C, L_freq) -> (B, n_heads, head_in, L_freq)
        x_freq = x_freq.view(B, self.n_heads, self.head_in, -1)
        
        # 频域混合（多头独立）
        n_modes = min(self.n_modes, x_freq.shape[-1])
        # einsum: (B, heads, head_in, freq) x (heads, head_in, head_out, modes) -> (B, heads, head_out, freq)
        out_freq = torch.einsum('bhif,hiof->bhof', x_freq[..., :n_modes], self.weight[..., :n_modes])
        
        # 合并多头：(B, n_heads, head_out, L_freq) -> (B, out_ch, L_freq)
        out_freq = out_freq.reshape(B, self.out_channels, -1)
        
        # IFFT
        x_out = torch.fft.irfft(out_freq, n=L, dim=-1)
        x_out = x_out + self.bias
        
        return x_out


class MHFFNO1D(nn.Module):
    """MHF-FNO 1D"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 64,
        n_modes: int = 16,
        n_layers: int = 4,
        n_heads: int = 4,
    ):
        super().__init__()
        
        self.lifting = nn.Linear(in_channels, hidden_channels)
        
        self.convs = nn.ModuleList([
            MHFSpectralConv1D(hidden_channels, hidden_channels, n_modes, n_heads)
            for _ in range(n_layers)
        ])
        
        self.projection = nn.Linear(hidden_channels, out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # (B, L, C)
        x = self.lifting(x)
        x = x.transpose(1, 2)  # (B, C, L)
        
        for conv in self.convs:
            x = F.gelu(conv(x))
        
        x = x.transpose(1, 2)  # (B, L, C)
        x = self.projection(x)
        x = x.transpose(1, 2)  # (B, C, L)
        
        return x


# ============================================
# 数据生成
# ============================================

def generate_darcy_1d_data(
    n_samples: int = 1000,
    resolution: int = 64,
    seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor]:
    """生成 1D Darcy Flow 数据"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    x = torch.linspace(0, 1, resolution)
    
    a_list = []
    for _ in range(n_samples):
        n_modes = np.random.randint(2, 6)
        a = torch.ones(resolution)
        for _ in range(n_modes):
            freq = np.random.rand() * 10 + 1
            phase = np.random.rand() * 2 * np.pi
            amp = np.random.rand() * 0.5 + 0.5
            a = a * (1 + amp * torch.sin(2 * np.pi * freq * x + phase))
        a = torch.clamp(a, 0.1, 10)
        a_list.append(a)
    
    a_field = torch.stack(a_list)
    
    # 简化的 Darcy 解
    u_list = []
    for i in range(n_samples):
        a = a_field[i]
        # 解析解近似
        u = x * (1 - x) / (a.mean() + 0.1)
        u = u * (1 + 0.2 * (a - a.mean()) / (a.std() + 1e-6))
        u_list.append(u)
    
    u_field = torch.stack(u_list)
    
    return a_field.unsqueeze(1), u_field.unsqueeze(1)


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
        sample_errors = torch.norm(pred - test_y, dim=(1, 2)) / torch.norm(test_y, dim=(1, 2))
    
    return {
        'relative_l2': l2_error,
        'mean_error': sample_errors.mean().item(),
        'std_error': sample_errors.std().item(),
    }


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


# ============================================
# 主测试
# ============================================

def run_benchmark():
    print("\n" + "=" * 70)
    print(" MHF-FNO 1D 完美基准测试")
    print("=" * 70)
    
    # 配置
    n_train, n_val, n_test = 800, 100, 100
    resolution = 64
    hidden_channels = 32
    n_modes = 16
    n_layers = 4
    n_heads = 4  # MHF 头数
    epochs = 100
    
    # 生成数据
    print("\n📊 生成 1D Darcy Flow 数据...")
    a_field, u_field = generate_darcy_1d_data(n_train + n_val + n_test, resolution)
    
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
    print("🔬 测试标准 FNO 1D")
    print("-" * 50)
    
    model_fno = StandardFNO1D(1, 1, hidden_channels, n_modes, n_layers)
    n_params_fno = count_parameters(model_fno)
    print(f"   参数量: {n_params_fno:,}")
    
    time_fno = train_model(model_fno, train_x, train_y, val_x, val_y, epochs=epochs)
    eval_fno = evaluate_model(model_fno, test_x, test_y)
    
    print(f"\n   ✅ L2误差: {eval_fno['relative_l2']:.4f}, 时间: {time_fno:.2f}s")
    results['FNO'] = {'params': n_params_fno, 'l2': eval_fno['relative_l2'], 'time': time_fno}
    
    # ===== 测试 MHF-FNO =====
    print("\n" + "-" * 50)
    print("🔬 测试 MHF-FNO 1D (TransFourier)")
    print("-" * 50)
    
    model_mhf = MHFFNO1D(1, 1, hidden_channels, n_modes, n_layers, n_heads)
    n_params_mhf = count_parameters(model_mhf)
    print(f"   参数量: {n_params_mhf:,}")
    
    time_mhf = train_model(model_mhf, train_x, train_y, val_x, val_y, epochs=epochs)
    eval_mhf = evaluate_model(model_mhf, test_x, test_y)
    
    print(f"\n   ✅ L2误差: {eval_mhf['relative_l2']:.4f}, 时间: {time_mhf:.2f}s")
    results['MHF-FNO'] = {'params': n_params_mhf, 'l2': eval_mhf['relative_l2'], 'time': time_mhf}
    
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