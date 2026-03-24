"""
Darcy Flow 真实场景测试

对比：
1. 标准FNO
2. MHF-FNO (TransFourier风格)

评估指标：
- 相对L2误差
- 训练时间
- 参数量
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
from typing import Dict, Tuple

# 导入MHF模块
import sys
sys.path.insert(0, '/root/.openclaw/workspace/memory/projects/tianyuan-fft/experiments')
from mhf_1d import MHFSpectralConv1D, MHFFNO1D


# ============================================
# 数据生成：Darcy Flow 1D简化版
# ============================================

def generate_darcy_data(
    n_samples: int = 1000,
    resolution: int = 64,
    seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor]:
    """生成1D Darcy Flow数据（简化版）"""
    torch.manual_seed(seed)
    
    x = torch.linspace(0, 1, resolution)
    
    a_list = []
    for _ in range(n_samples):
        n_modes = torch.randint(2, 6, (1,)).item()
        a = torch.ones(resolution)
        for _ in range(n_modes):
            freq = torch.rand(1).item() * 10 + 1
            phase = torch.rand(1).item() * 2 * np.pi
            amp = torch.rand(1).item() * 0.5 + 0.5
            a = a * (1 + amp * torch.sin(2 * np.pi * freq * x + phase))
        a = torch.clamp(a, 0.1, 10)
        a_list.append(a)
    
    a_field = torch.stack(a_list)
    
    u_list = []
    for i in range(n_samples):
        a = a_field[i]
        u = x * (1 - x) / (a.mean() + 0.1)
        u = u * (1 + 0.2 * (a - a.mean()) / a.std())
        u_list.append(u)
    
    u_field = torch.stack(u_list)
    
    return a_field.unsqueeze(1), u_field.unsqueeze(1)


# ============================================
# 标准FNO实现
# ============================================

class SpectralConv1D(nn.Module):
    """标准SpectralConv 1D"""
    
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
        out_freq = torch.zeros(B, self.weight.shape[1], x_freq.shape[-1], dtype=x_freq.dtype, device=x.device)
        out_freq[:, :, :n_modes] = torch.einsum('bif,iOf->bOf', x_freq[:, :, :n_modes], self.weight)
        
        x_out = torch.fft.irfft(out_freq, n=L, dim=-1)
        x_out = x_out + self.bias
        
        return x_out


class StandardFNO1D(nn.Module):
    """标准FNO 1D"""
    
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
        model.eval()
        with torch.no_grad():
            val_loss = F.mse_loss(model(val_x), val_y).item()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train: {epoch_loss/n_batches:.6f}, Val: {val_loss:.6f}")
    
    train_time = time.time() - start_time
    return train_time


def evaluate_model(model, test_x, test_y):
    """评估模型"""
    
    model.eval()
    with torch.no_grad():
        pred = model(test_x)
        l2_error = torch.norm(pred - test_y) / torch.norm(test_y)
        sample_errors = torch.norm(pred - test_y, dim=(1, 2)) / torch.norm(test_y, dim=(1, 2))
    
    return {
        'relative_l2': l2_error.item(),
        'mean_error': sample_errors.mean().item(),
        'std_error': sample_errors.std().item(),
    }


# ============================================
# 主测试流程
# ============================================

def run_benchmark():
    print("\n" + "=" * 60)
    print(" Darcy Flow 真实场景测试")
    print(" 对比: 标准FNO vs MHF-FNO")
    print("=" * 60)
    
    # 配置
    n_train, n_val, n_test = 800, 100, 100
    resolution = 64
    hidden_channels = 32
    n_modes = 16
    n_layers = 3
    epochs = 100
    
    # 生成数据
    print("\n生成Darcy Flow数据...")
    a_field, u_field = generate_darcy_data(n_train + n_val + n_test, resolution)
    
    train_x = a_field[:n_train]
    train_y = u_field[:n_train]
    val_x = a_field[n_train:n_train+n_val]
    val_y = u_field[n_train:n_train+n_val]
    test_x = a_field[-n_test:]
    test_y = u_field[-n_test:]
    
    print(f"训练: {train_x.shape}, 验证: {val_x.shape}, 测试: {test_x.shape}")
    
    results = {}
    
    # ===== 测试标准FNO =====
    print("\n" + "-" * 40)
    print("测试标准FNO...")
    print("-" * 40)
    
    model_fno = StandardFNO1D(1, 1, hidden_channels, n_modes, n_layers)
    n_params_fno = sum(p.numel() for p in model_fno.parameters())
    print(f"参数量: {n_params_fno:,}")
    
    time_fno = train_model(model_fno, train_x, train_y, val_x, val_y, epochs=epochs)
    eval_fno = evaluate_model(model_fno, test_x, test_y)
    
    print(f"\n测试结果: L2误差={eval_fno['relative_l2']:.4f}, 时间={time_fno:.2f}s")
    results['FNO'] = {'params': n_params_fno, 'l2': eval_fno['relative_l2'], 'time': time_fno}
    
    # ===== 测试MHF-FNO =====
    print("\n" + "-" * 40)
    print("测试MHF-FNO (TransFourier风格)...")
    print("-" * 40)
    
    model_mhf = MHFFNO1D(1, 1, hidden_channels, n_modes, n_layers, n_heads=4)
    n_params_mhf = sum(p.numel() for p in model_mhf.parameters())
    print(f"参数量: {n_params_mhf:,}")
    
    time_mhf = train_model(model_mhf, train_x, train_y, val_x, val_y, epochs=epochs)
    eval_mhf = evaluate_model(model_mhf, test_x, test_y)
    
    print(f"\n测试结果: L2误差={eval_mhf['relative_l2']:.4f}, 时间={time_mhf:.2f}s")
    results['MHF-FNO'] = {'params': n_params_mhf, 'l2': eval_mhf['relative_l2'], 'time': time_mhf}
    
    # ===== 汇总 =====
    print("\n" + "=" * 60)
    print(" 汇总对比")
    print("=" * 60)
    
    print(f"\n{'模型':<15} {'参数量':<12} {'L2误差':<12} {'训练时间':<12}")
    print("-" * 55)
    
    for name, res in results.items():
        print(f"{name:<15} {res['params']:<12,} {res['l2']:<12.4f} {res['time']:<12.2f}s")
    
    l2_change = (results['MHF-FNO']['l2'] - results['FNO']['l2']) / results['FNO']['l2'] * 100
    time_change = (results['MHF-FNO']['time'] - results['FNO']['time']) / results['FNO']['time'] * 100
    
    print(f"\nMHF-FNO vs FNO:")
    print(f"  L2误差: {l2_change:+.2f}%")
    print(f"  训练时间: {time_change:+.2f}%")
    
    return results


if __name__ == "__main__":
    run_benchmark()