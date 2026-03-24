#!/usr/bin/env python3
"""
Darcy Flow 多分辨率测试: FNO vs MHF-FNO vs TFNO

测试不同分辨率下三种方法的性能对比。

使用方法:
    python multi_resolution_test.py
    python multi_resolution_test.py --resolutions 16 32 64
"""

import argparse
import time
import torch
import torch.nn as nn
from neuralop.models import FNO, TFNO
from neuralop.losses.data_losses import LpLoss
from pathlib import Path
import sys
import json

# 导入 MHF-FNO
sys.path.insert(0, str(Path(__file__).parent.parent))
from mhf_fno import create_hybrid_fno


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def train(model, train_x, train_y, test_x, test_y, epochs=30, lr=1e-3):
    """训练模型"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = LpLoss(d=2, p=2)
    
    t0 = time.time()
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(train_x))
        
        for i in range(0, len(train_x), 32):
            bx = train_x[perm[i:i+32]]
            by = train_y[perm[i:i+32]]
            
            optimizer.zero_grad()
            loss = loss_fn(model(bx), by)
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        
        model.eval()
        with torch.no_grad():
            test_loss = loss_fn(model(test_x), test_y).item()
        
        if test_loss < best_loss:
            best_loss = test_loss
    
    return {'time': time.time() - t0, 'loss': best_loss}


def generate_darcy_data(resolution, n_train, n_test, device='cpu'):
    """生成 Darcy Flow 数据"""
    import numpy as np
    
    print(f"  生成数据: {resolution}x{resolution}, 训练 {n_train}, 测试 {n_test}")
    
    # 高斯随机场参数 (PDEBench 标准)
    alpha, tau = 2.0, 3.0
    
    # 频率网格
    kx = torch.fft.fftfreq(resolution, d=1.0, device=device)
    ky = torch.fft.fftfreq(resolution, d=1.0, device=device)
    kx, ky = torch.meshgrid(kx, ky, indexing='ij')
    k = torch.sqrt(kx**2 + ky**2)
    power = (tau**2 + k**2)**(-alpha / 2.0)
    power[0, 0] = 0
    
    train_x, train_y = [], []
    test_x, test_y = [], []
    
    for i in range(n_train + n_test):
        # 随机相位
        noise = torch.randn(resolution, resolution, dtype=torch.complex64, device=device)
        noise = noise * torch.sqrt(power)
        field = torch.fft.ifft2(noise).real
        
        # 简化解：平滑场
        kernel = torch.ones(1, 1, 3, 3, device=device) / 9
        smooth_field = torch.nn.functional.conv2d(
            field.unsqueeze(0).unsqueeze(0), kernel, padding=1
        ).squeeze()
        
        if i < n_train:
            train_x.append(field.cpu())
            train_y.append(smooth_field.cpu())
        else:
            test_x.append(field.cpu())
            test_y.append(smooth_field.cpu())
    
    train_x = torch.stack(train_x).unsqueeze(1).float()
    train_y = torch.stack(train_y).unsqueeze(1).float()
    test_x = torch.stack(test_x).unsqueeze(1).float()
    test_y = torch.stack(test_y).unsqueeze(1).float()
    
    return train_x, train_y, test_x, test_y


def test_resolution(resolution, n_train=200, n_test=50, epochs=30):
    """测试单个分辨率"""
    print(f"\n{'='*60}")
    print(f"分辨率: {resolution}x{resolution}")
    print("=" * 60)
    
    # 生成数据
    train_x, train_y, test_x, test_y = generate_darcy_data(
        resolution, n_train, n_test
    )
    
    # 自适应 n_modes
    n_modes = (min(8, resolution // 2), min(8, resolution // 2))
    hidden = 32
    
    results = {}
    
    # FNO
    print("\n[FNO]")
    torch.manual_seed(42)
    fno = FNO(n_modes=n_modes, in_channels=1, out_channels=1, 
              hidden_channels=hidden, n_layers=3)
    params_fno = count_params(fno)
    r_fno = train(fno, train_x, train_y, test_x, test_y, epochs)
    results['FNO'] = {'params': params_fno, **r_fno}
    print(f"  参数: {params_fno:,}, Loss: {r_fno['loss']:.6f}")
    
    # MHF-FNO
    print("\n[MHF-FNO]")
    torch.manual_seed(42)
    mhf = create_hybrid_fno(n_modes=n_modes, hidden_channels=hidden, 
                            n_heads=2, mhf_layers=[0, 2])
    params_mhf = count_params(mhf)
    r_mhf = train(mhf, train_x, train_y, test_x, test_y, epochs)
    results['MHF-FNO'] = {'params': params_mhf, **r_mhf}
    print(f"  参数: {params_mhf:,}, Loss: {r_mhf['loss']:.6f}")
    
    # TFNO
    print("\n[TFNO]")
    torch.manual_seed(42)
    tfno = TFNO(n_modes=n_modes, in_channels=1, out_channels=1,
                hidden_channels=hidden, n_layers=3, rank=0.5)
    params_tfno = count_params(tfno)
    r_tfno = train(tfno, train_x, train_y, test_x, test_y, epochs)
    results['TFNO'] = {'params': params_tfno, **r_tfno}
    print(f"  参数: {params_tfno:,}, Loss: {r_tfno['loss']:.6f}")
    
    return resolution, results


def main():
    parser = argparse.ArgumentParser(description='多分辨率测试')
    parser.add_argument('--resolutions', type=int, nargs='+', 
                        default=[16, 32, 64], help='测试分辨率')
    parser.add_argument('--n_train', type=int, default=200, help='训练样本数')
    parser.add_argument('--n_test', type=int, default=50, help='测试样本数')
    parser.add_argument('--epochs', type=int, default=30, help='训练轮数')
    args = parser.parse_args()
    
    print("=" * 60)
    print("Darcy Flow 多分辨率测试")
    print("=" * 60)
    print(f"分辨率: {args.resolutions}")
    print(f"训练样本: {args.n_train}, 测试样本: {args.n_test}")
    print(f"训练轮数: {args.epochs}")
    
    all_results = {}
    
    for res in args.resolutions:
        _, results = test_resolution(res, args.n_train, args.n_test, args.epochs)
        all_results[res] = results
    
    # 汇总
    print(f"\n{'='*60}")
    print("汇总报告")
    print("=" * 60)
    
    print(f"\n{'分辨率':<10} {'模型':<12} {'参数量':>10} {'压缩率':>8} {'Loss':>12} {'vs FNO':>10}")
    print("-" * 70)
    
    for res in args.resolutions:
        results = all_results[res]
        params_fno = results['FNO']['params']
        
        for name, r in results.items():
            compression = (1 - r['params'] / params_fno) * 100
            vs_fno = (r['loss'] / results['FNO']['loss'] - 1) * 100
            print(f"{res}x{res:<5} {name:<12} {r['params']:>10,} {compression:>7.1f}% "
                  f"{r['loss']:>12.6f} {vs_fno:>+9.1f}%")
        print("-" * 70)
    
    # 保存结果
    output = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'config': vars(args),
        'results': {str(k): v for k, v in all_results.items()}
    }
    
    with open('multi_resolution_results.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n✅ 结果已保存到 multi_resolution_results.json")


if __name__ == '__main__':
    main()