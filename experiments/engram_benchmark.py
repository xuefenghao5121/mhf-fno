"""
三架构Benchmark对比测试

对比:
1. FNO (NeuralOperator官方)
2. MHF-FNO (TransFourier风格)
3. Engram + MHF-FNO (双域记忆)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys

sys.path.insert(0, '/root/.openclaw/workspace/memory/projects/tianyuan-fft/experiments')
from mhf_1d import MHFFNO1D
from engram_mhf import EngramMHFFNO1D
from neuralop.models import FNO


def generate_darcy_2d(n_samples, resolution):
    torch.manual_seed(42)
    a_field = torch.randn(n_samples, 1, resolution, resolution)
    a_field = torch.sigmoid(a_field) * 2 + 0.5
    u_field = torch.zeros(n_samples, 1, resolution, resolution)
    for i in range(n_samples):
        noise = torch.randn(resolution, resolution)
        u = torch.fft.ifft2(torch.fft.fft2(noise) * 0.1).real
        u = (u - u.min()) / (u.max() - u.min())
        u_field[i, 0] = u
    return a_field, u_field


def run_benchmark():
    print("\n" + "=" * 70)
    print(" 三架构 Benchmark 对比")
    print("=" * 70)
    
    # 配置
    resolution = 32
    n_train = 300
    n_test = 50
    epochs = 30
    
    print(f"\n配置: {resolution}x{resolution}, 训练{n_train}, 测试{n_test}, {epochs}轮")
    
    # 生成数据
    print("\n生成Darcy Flow 2D数据...")
    train_a, train_u = generate_darcy_2d(n_train, resolution)
    test_a, test_u = generate_darcy_2d(n_test, resolution)
    
    # 展平为1D
    train_a_1d = train_a.view(n_train, 1, -1)
    train_u_1d = train_u.view(n_train, 1, -1)
    test_a_1d = test_a.view(n_test, 1, -1)
    test_u_1d = test_u.view(n_test, 1, -1)
    
    print(f"展平后: {train_a_1d.shape}")
    
    results = {}
    
    # ===== 1. FNO (官方) =====
    print("\n" + "-" * 50)
    print("1. FNO (NeuralOperator官方)")
    print("-" * 50)
    
    model_fno = FNO(n_modes=(8, 8), hidden_channels=32, in_channels=1, out_channels=1, n_layers=3)
    params_fno = sum(p.numel() for p in model_fno.parameters())
    print(f"参数量: {params_fno:,}")
    
    optimizer = torch.optim.Adam(model_fno.parameters(), lr=1e-3)
    start = time.time()
    for epoch in range(epochs):
        model_fno.train()
        pred = model_fno(train_a)
        loss = F.mse_loss(pred, train_u)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    time_fno = time.time() - start
    
    model_fno.eval()
    with torch.no_grad():
        pred = model_fno(test_a)
        l2_fno = (torch.norm(pred - test_u) / torch.norm(test_u)).item()
    
    print(f"L2误差: {l2_fno:.4f}, 时间: {time_fno:.2f}s")
    results['FNO'] = {'params': params_fno, 'l2': l2_fno, 'time': time_fno}
    
    # ===== 2. MHF-FNO =====
    print("\n" + "-" * 50)
    print("2. MHF-FNO (TransFourier风格)")
    print("-" * 50)
    
    model_mhf = MHFFNO1D(1, 1, 32, 32, 3, 4)
    params_mhf = sum(p.numel() for p in model_mhf.parameters())
    print(f"参数量: {params_mhf:,}")
    
    optimizer = torch.optim.Adam(model_mhf.parameters(), lr=1e-3)
    start = time.time()
    for epoch in range(epochs):
        model_mhf.train()
        pred = model_mhf(train_a_1d)
        loss = F.mse_loss(pred, train_u_1d)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    time_mhf = time.time() - start
    
    model_mhf.eval()
    with torch.no_grad():
        pred = model_mhf(test_a_1d)
        l2_mhf = (torch.norm(pred - test_u_1d) / torch.norm(test_u_1d)).item()
    
    print(f"L2误差: {l2_mhf:.4f}, 时间: {time_mhf:.2f}s")
    results['MHF-FNO'] = {'params': params_mhf, 'l2': l2_mhf, 'time': time_mhf}
    
    # ===== 3. Engram + MHF-FNO =====
    print("\n" + "-" * 50)
    print("3. Engram + MHF-FNO (双域记忆)")
    print("-" * 50)
    
    model_engram = EngramMHFFNO1D(1, 1, 32, 32, 3, 4, use_engram=True)
    params_engram = sum(p.numel() for p in model_engram.parameters())
    print(f"参数量: {params_engram:,}")
    
    optimizer = torch.optim.Adam(model_engram.parameters(), lr=1e-3)
    start = time.time()
    for epoch in range(epochs):
        model_engram.train()
        pred = model_engram(train_a_1d)
        loss = F.mse_loss(pred, train_u_1d)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    time_engram = time.time() - start
    
    model_engram.eval()
    with torch.no_grad():
        pred = model_engram(test_a_1d)
        l2_engram = (torch.norm(pred - test_u_1d) / torch.norm(test_u_1d)).item()
    
    print(f"L2误差: {l2_engram:.4f}, 时间: {time_engram:.2f}s")
    results['Engram-MHF'] = {'params': params_engram, 'l2': l2_engram, 'time': time_engram}
    
    # ===== 汇总 =====
    print("\n" + "=" * 70)
    print(" 汇总对比")
    print("=" * 70)
    
    print(f"\n{'模型':<20} {'参数量':<12} {'L2误差':<12} {'训练时间':<12}")
    print("-" * 60)
    
    for name, res in results.items():
        print(f"{name:<20} {res['params']:<12,} {res['l2']:<12.4f} {res['time']:<12.2f}s")
    
    # 计算改进
    print(f"\n相对于FNO:")
    for name in ['MHF-FNO', 'Engram-MHF']:
        p_change = (results[name]['params'] - results['FNO']['params']) / results['FNO']['params'] * 100
        l2_change = (results[name]['l2'] - results['FNO']['l2']) / results['FNO']['l2'] * 100
        t_change = (results[name]['time'] - results['FNO']['time']) / results['FNO']['time'] * 100
        print(f"  {name}: 参数{p_change:+.1f}%, L2{l2_change:+.1f}%, 时间{t_change:+.1f}%")
    
    return results


if __name__ == "__main__":
    run_benchmark()