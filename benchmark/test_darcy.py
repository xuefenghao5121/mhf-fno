#!/usr/bin/env python3
"""
Darcy Flow 测试 (方案B修复版)
"""

import sys
import time
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from neuralop.models import FNO

sys.path.insert(0, str(Path(__file__).parent.parent))

from mhf_fno.mhf_attention_v2 import create_mhf_fno_v2


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def train_and_eval(model, train_x, train_y, test_x, test_y, epochs, batch_size, lr, device, verbose=True):
    """训练并评估模型"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.MSELoss()
    
    best_test_loss = float('inf')
    best_epoch = 0
    
    for epoch in range(epochs):
        # Train
        model.train()
        perm = torch.randperm(train_x.shape[0], device=device)
        
        for i in range(0, train_x.shape[0], batch_size):
            idx = perm[i:i+batch_size]
            bx, by = train_x[idx].to(device), train_y[idx].to(device)
            
            optimizer.zero_grad()
            output = model(bx)
            loss = loss_fn(output, by)
            
            if not (torch.isnan(loss) or torch.isinf(loss)):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        
        # Eval
        model.eval()
        test_loss = 0
        n_test = 0
        with torch.no_grad():
            for i in range(0, test_x.shape[0], batch_size):
                bx, by = test_x[i:i+batch_size].to(device), test_y[i:i+batch_size].to(device)
                loss = loss_fn(model(bx), by)
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    test_loss += loss.item()
                    n_test += 1
        
        test_loss /= max(n_test, 1)
        
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_epoch = epoch + 1
        
        scheduler.step()
        
        if verbose and (epoch + 1) % 25 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: Test {test_loss:.6f} (best: {best_test_loss:.6f} @ {best_epoch})")
    
    return best_test_loss, best_epoch


def main():
    device = torch.device('cpu')
    base_path = Path(__file__).parent.parent
    data_path = base_path / 'data'
    
    print("=" * 70)
    print("方案 B: Darcy Flow 测试 (椭圆型PDE)")
    print("=" * 70)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 加载数据
    darcy_train_path = data_path / 'darcy_train_32.pt'
    darcy_test_path = data_path / 'darcy_test_32.pt'
    
    if not (darcy_train_path.exists() and darcy_test_path.exists()):
        print("错误: Darcy数据文件不存在")
        return
    
    train_data = torch.load(darcy_train_path, weights_only=False, map_location='cpu')
    test_data = torch.load(darcy_test_path, weights_only=False, map_location='cpu')
    
    train_x = train_data['x'].unsqueeze(1).float()
    train_y = train_data['y'].unsqueeze(1).float()
    test_x = test_data['x'].unsqueeze(1).float()
    test_y = test_data['y'].unsqueeze(1).float()
    
    print(f"训练集: {train_x.shape}")
    print(f"测试集: {test_x.shape}")
    
    # 归一化
    train_x = (train_x - train_x.mean(dim=(-2, -1), keepdim=True)) / (train_x.std(dim=(-2, -1), keepdim=True) + 1e-8)
    train_y = (train_y - train_y.mean(dim=(-2, -1), keepdim=True)) / (train_y.std(dim=(-2, -1), keepdim=True) + 1e-8)
    test_x = (test_x - test_x.mean(dim=(-2, -1), keepdim=True)) / (test_x.std(dim=(-2, -1), keepdim=True) + 1e-8)
    test_y = (test_y - test_y.mean(dim=(-2, -1), keepdim=True)) / (test_y.std(dim=(-2, -1), keepdim=True) + 1e-8)
    
    n_modes = (12, 12)
    hidden_channels = 32
    batch_size = 16
    lr = 1e-3
    epochs = 100
    
    results = {}
    
    # FNO 基准
    print("\n--- FNO 基准 (Darcy Flow) ---")
    torch.manual_seed(42)
    model_fno = FNO(n_modes=n_modes, hidden_channels=hidden_channels, in_channels=1, out_channels=1, n_layers=3).to(device)
    params_fno = count_parameters(model_fno)
    print(f"参数量: {params_fno:,}")
    
    t0 = time.time()
    fno_loss, fno_epoch = train_and_eval(model_fno, train_x, train_y, test_x, test_y, epochs, batch_size, lr, device)
    fno_time = time.time() - t0
    
    print(f"最佳 Loss: {fno_loss:.6f} @ Epoch {fno_epoch}")
    print(f"训练时间: {fno_time:.1f}s")
    
    results['FNO'] = {
        'params': params_fno,
        'best_loss': fno_loss,
        'best_epoch': fno_epoch,
        'time': fno_time,
        'vs_fno': 0.0
    }
    
    # MHF 变体测试
    configs = [
        ('CoDA-[0,2]', 4, 'coda', [0, 2]),
        ('CoDA-full', 4, 'coda', [0, 1, 2]),
        ('MHSA-[0,2]', 4, 'mhsa', [0, 2]),
    ]
    
    for name, n_heads, attn_type, mhf_layers in configs:
        print(f"\n--- {name} (Darcy Flow) ---")
        torch.manual_seed(42)
        model = create_mhf_fno_v2(
            n_modes, hidden_channels,
            n_heads=n_heads,
            attention_type=attn_type,
            mhf_layers=mhf_layers
        ).to(device)
        params = count_parameters(model)
        print(f"参数量: {params:,}")
        
        t0 = time.time()
        loss, epoch = train_and_eval(model, train_x, train_y, test_x, test_y, epochs, batch_size, lr, device)
        elapsed = time.time() - t0
        
        diff = (loss - fno_loss) / fno_loss * 100
        print(f"最佳 Loss: {loss:.6f} @ Epoch {epoch}")
        print(f"vs FNO: {diff:+.2f}%")
        print(f"训练时间: {elapsed:.1f}s")
        
        results[name] = {
            'params': params,
            'best_loss': loss,
            'best_epoch': epoch,
            'time': elapsed,
            'vs_fno': diff
        }
    
    # 汇总报告
    print("\n" + "=" * 70)
    print("Darcy Flow 测试结果")
    print("=" * 70)
    
    print(f"\n{'配置':<20} {'参数量':<12} {'最佳Loss':<12} {'vs FNO':<10}")
    print("-" * 60)
    for name, r in results.items():
        print(f"{name:<20} {r['params']:,} {r['best_loss']:<12.6f} {r['vs_fno']:+.2f}%")
    
    # MHF适用边界分析
    coda_results = [(k, v) for k, v in results.items() if 'CoDA' in k or 'MHSA' in k]
    if coda_results:
        best = min(coda_results, key=lambda x: x[1]['best_loss'])
        print(f"\nDarcy最佳配置: {best[0]} ({best[1]['vs_fno']:+.2f}% vs FNO)")
    
    # 保存结果
    output_path = base_path / 'results' / 'darcy_results.json'
    with open(output_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': results,
            'fno_baseline': fno_loss
        }, f, indent=2)
    
    print(f"\n结果已保存: {output_path}")
    
    return results


if __name__ == '__main__':
    main()