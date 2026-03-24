"""
NeuralOperator官方Benchmark测试

使用neuraloperator库自带的数据集和训练流程对比:
1. 标准FNO (NeuralOperator实现)
2. MHF-FNO (TransFourier风格)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
import numpy as np
from typing import Dict

# 导入MHF模块
sys.path.insert(0, '/root/.openclaw/workspace/memory/projects/tianyuan-fft/experiments')
from mhf_1d import MHFFNO1D

# 导入NeuralOperator
from neuralop.models import FNO
print("✅ NeuralOperator导入成功")


def generate_darcy_2d(n_samples, resolution):
    """生成2D Darcy Flow数据"""
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
    """运行benchmark对比"""
    
    print("\n" + "=" * 70)
    print(" NeuralOperator 2D Darcy Flow Benchmark")
    print("=" * 70)
    
    # 配置
    resolution = 32
    n_train = 500
    n_test = 100
    epochs = 50
    
    print(f"\n配置:")
    print(f"  分辨率: {resolution}x{resolution}")
    print(f"  训练样本: {n_train}")
    print(f"  测试样本: {n_test}")
    print(f"  训练轮数: {epochs}")
    
    # 生成数据
    print("\n生成Darcy Flow 2D数据...")
    train_a, train_u = generate_darcy_2d(n_train, resolution)
    test_a, test_u = generate_darcy_2d(n_test, resolution)
    print(f"训练集: {train_a.shape}, 测试集: {test_a.shape}")
    
    results = {}
    
    # ===== FNO (NeuralOperator) =====
    print("\n" + "-" * 50)
    print("测试 FNO (NeuralOperator官方实现)")
    print("-" * 50)
    
    model_fno = FNO(
        n_modes=(8, 8),
        hidden_channels=32,
        in_channels=1,
        out_channels=1,
        n_layers=3,
    )
    
    n_params_fno = sum(p.numel() for p in model_fno.parameters())
    print(f"参数量: {n_params_fno:,}")
    
    optimizer = torch.optim.Adam(model_fno.parameters(), lr=1e-3)
    
    start_time = time.time()
    for epoch in range(epochs):
        model_fno.train()
        pred = model_fno(train_a)
        loss = F.mse_loss(pred, train_u)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            model_fno.eval()
            with torch.no_grad():
                test_pred = model_fno(test_a)
                test_loss = F.mse_loss(test_pred, test_u).item()
            print(f"Epoch {epoch+1}: Loss = {loss.item():.6f}, Test = {test_loss:.6f}")
    
    train_time_fno = time.time() - start_time
    
    model_fno.eval()
    with torch.no_grad():
        test_pred = model_fno(test_a)
        l2_fno = (torch.norm(test_pred - test_u) / torch.norm(test_u)).item()
    
    print(f"\n结果: L2误差={l2_fno:.4f}, 时间={train_time_fno:.2f}s")
    results['FNO'] = {'params': n_params_fno, 'l2': l2_fno, 'time': train_time_fno}
    
    # ===== MHF-FNO (1D展平) =====
    print("\n" + "-" * 50)
    print("测试 MHF-FNO (TransFourier风格)")
    print("-" * 50)
    
    # 展平为1D
    train_a_1d = train_a.view(n_train, 1, -1)
    train_u_1d = train_u.view(n_train, 1, -1)
    test_a_1d = test_a.view(n_test, 1, -1)
    test_u_1d = test_u.view(n_test, 1, -1)
    
    print(f"展平后: {train_a_1d.shape}")
    
    model_mhf = MHFFNO1D(
        in_channels=1,
        out_channels=1,
        hidden_channels=32,
        n_modes=32,
        n_layers=3,
        n_heads=4,
    )
    
    n_params_mhf = sum(p.numel() for p in model_mhf.parameters())
    print(f"参数量: {n_params_mhf:,}")
    
    optimizer = torch.optim.Adam(model_mhf.parameters(), lr=1e-3)
    
    start_time = time.time()
    for epoch in range(epochs):
        model_mhf.train()
        pred = model_mhf(train_a_1d)
        loss = F.mse_loss(pred, train_u_1d)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            model_mhf.eval()
            with torch.no_grad():
                test_pred = model_mhf(test_a_1d)
                test_loss = F.mse_loss(test_pred, test_u_1d).item()
            print(f"Epoch {epoch+1}: Loss = {loss.item():.6f}, Test = {test_loss:.6f}")
    
    train_time_mhf = time.time() - start_time
    
    model_mhf.eval()
    with torch.no_grad():
        test_pred = model_mhf(test_a_1d)
        l2_mhf = (torch.norm(test_pred - test_u_1d) / torch.norm(test_u_1d)).item()
    
    print(f"\n结果: L2误差={l2_mhf:.4f}, 时间={train_time_mhf:.2f}s")
    results['MHF-FNO'] = {'params': n_params_mhf, 'l2': l2_mhf, 'time': train_time_mhf}
    
    # ===== 汇总 =====
    print("\n" + "=" * 70)
    print(" 汇总对比")
    print("=" * 70)
    
    print(f"\n{'模型':<20} {'参数量':<15} {'L2误差':<12} {'训练时间':<12}")
    print("-" * 65)
    
    for name, res in results.items():
        print(f"{name:<20} {res['params']:<15,} {res['l2']:<12.4f} {res['time']:<12.2f}s")
    
    l2_change = (results['MHF-FNO']['l2'] - results['FNO']['l2']) / results['FNO']['l2'] * 100
    time_change = (results['MHF-FNO']['time'] - results['FNO']['time']) / results['FNO']['time'] * 100
    
    print(f"\nMHF-FNO vs FNO:")
    print(f"  参数量变化: {(results['MHF-FNO']['params'] - results['FNO']['params']) / results['FNO']['params'] * 100:+.1f}%")
    print(f"  L2误差变化: {l2_change:+.2f}%")
    print(f"  训练时间变化: {time_change:+.2f}%")
    
    return results


if __name__ == "__main__":
    run_benchmark()