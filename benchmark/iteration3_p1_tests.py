#!/usr/bin/env python3
"""
迭代 3 - P1 改进方向测试
========================

测试方案:
- 方案A: 数据增强 (200 → 1000 样本)
- 方案B: Darcy Flow 测试 (椭圆型PDE)

目标: 
- NS: ≤ -10% vs FNO
- Darcy: 验证MHF适用边界

作者: Tianyuan Team
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

from mhf_fno import create_hybrid_fno
from mhf_fno.mhf_attention_v2 import create_mhf_fno_v2
from benchmark.generate_data import (
    generate_navier_stokes_2d,
    generate_darcy_flow
)


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


def test_data_augmentation(device, base_path):
    """
    方案A: 数据增强测试
    
    - 生成 1000 训练样本 (vs 200)
    - 使用 CoDA-[0,2] 配置
    """
    print("\n" + "=" * 70)
    print("方案 A: 数据增强测试 (200 → 1000 样本)")
    print("=" * 70)
    
    data_path = base_path / 'data'
    
    # 检查是否已有大数据集
    large_train_path = data_path / 'ns_train_32_large.pt'
    large_test_path = data_path / 'ns_test_32_large.pt'
    
    if not (large_train_path.exists() and large_test_path.exists()):
        print("生成 1000 训练样本...")
        torch.manual_seed(123)
        generate_navier_stokes_2d(
            n_train=1000,
            n_test=200,
            resolution=32,
            viscosity=1e-3,
            n_steps=50,
            output_dir=str(data_path),
            device='cpu',
            verbose=True
        )
        # 重命名为 large
        import shutil
        shutil.move(str(data_path / 'ns_train_32.pt'), str(large_train_path))
        shutil.move(str(data_path / 'ns_test_32.pt'), str(large_test_path))
    else:
        print(f"使用已有数据集: {large_train_path}")
    
    # 加载大数据集
    train_data = torch.load(large_train_path, weights_only=False, map_location='cpu')
    test_data = torch.load(large_test_path, weights_only=False, map_location='cpu')
    
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
    batch_size = 32  # 更大批次适合更多数据
    lr = 1e-3
    epochs = 100
    
    results = {}
    
    # FNO 基准
    print("\n--- FNO 基准 (1000 样本) ---")
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
    
    # CoDA-[0,2]
    print("\n--- CoDA-[0,2] (1000 样本) ---")
    torch.manual_seed(42)
    model_coda = create_mhf_fno_v2(
        n_modes, hidden_channels, 
        n_heads=4, 
        attention_type='coda', 
        mhf_layers=[0, 2]
    ).to(device)
    params_coda = count_parameters(model_coda)
    print(f"参数量: {params_coda:,}")
    
    t0 = time.time()
    coda_loss, coda_epoch = train_and_eval(model_coda, train_x, train_y, test_x, test_y, epochs, batch_size, lr, device)
    coda_time = time.time() - t0
    
    coda_diff = (coda_loss - fno_loss) / fno_loss * 100
    print(f"最佳 Loss: {coda_loss:.6f} @ Epoch {coda_epoch}")
    print(f"vs FNO: {coda_diff:+.2f}%")
    print(f"训练时间: {coda_time:.1f}s")
    
    results['CoDA-[0,2]'] = {
        'params': params_coda,
        'best_loss': coda_loss,
        'best_epoch': coda_epoch,
        'time': coda_time,
        'vs_fno': coda_diff
    }
    
    return results, fno_loss


def test_darcy_flow(device, base_path):
    """
    方案B: Darcy Flow 测试
    
    - 椭圆型PDE
    - 验证MHF适用边界
    """
    print("\n" + "=" * 70)
    print("方案 B: Darcy Flow 测试 (椭圆型PDE)")
    print("=" * 70)
    
    data_path = base_path / 'data'
    
    # 检查是否已有Darcy数据
    darcy_train_path = data_path / 'darcy_train_32.pt'
    darcy_test_path = data_path / 'darcy_test_32.pt'
    
    if not (darcy_train_path.exists() and darcy_test_path.exists()):
        print("生成 Darcy Flow 数据集 (32x32)...")
        torch.manual_seed(456)
        generate_darcy_flow(
            n_train=500,
            n_test=100,
            resolution=32,
            output_dir=str(data_path),
            device='cpu',
            verbose=True
        )
        # 重命名为 32
        import shutil
        shutil.move(str(data_path / 'darcy_train_16.pt'), str(darcy_train_path))
        shutil.move(str(data_path / 'darcy_test_16.pt'), str(darcy_test_path))
    else:
        print(f"使用已有数据集: {darcy_train_path}")
    
    # 加载数据
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
    
    return results, fno_loss


def main():
    device = torch.device('cpu')
    base_path = Path(__file__).parent.parent
    
    print("=" * 70)
    print("迭代 3: P1 改进方向测试")
    print("=" * 70)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"设备: {device}")
    
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'iteration': 3,
        'target': {
            'ns': '≤ -10% vs FNO',
            'darcy': '验证MHF适用边界'
        },
        'tests': {}
    }
    
    # 方案A: 数据增强
    print("\n" + "=" * 70)
    print("开始方案 A: 数据增强测试")
    print("=" * 70)
    
    try:
        aug_results, aug_fno_loss = test_data_augmentation(device, base_path)
        all_results['tests']['data_augmentation'] = {
            'status': 'success',
            'results': aug_results,
            'fno_baseline': aug_fno_loss
        }
    except Exception as e:
        print(f"❌ 方案A失败: {e}")
        import traceback
        traceback.print_exc()
        all_results['tests']['data_augmentation'] = {
            'status': 'failed',
            'error': str(e)
        }
    
    # 方案B: Darcy Flow
    print("\n" + "=" * 70)
    print("开始方案 B: Darcy Flow 测试")
    print("=" * 70)
    
    try:
        darcy_results, darcy_fno_loss = test_darcy_flow(device, base_path)
        all_results['tests']['darcy_flow'] = {
            'status': 'success',
            'results': darcy_results,
            'fno_baseline': darcy_fno_loss
        }
    except Exception as e:
        print(f"❌ 方案B失败: {e}")
        import traceback
        traceback.print_exc()
        all_results['tests']['darcy_flow'] = {
            'status': 'failed',
            'error': str(e)
        }
    
    # 汇总报告
    print("\n" + "=" * 70)
    print("迭代 3 汇总报告")
    print("=" * 70)
    
    # 方案A结果
    if 'data_augmentation' in all_results['tests'] and all_results['tests']['data_augmentation']['status'] == 'success':
        aug = all_results['tests']['data_augmentation']['results']
        print("\n【方案A: 数据增强】")
        print(f"{'配置':<20} {'参数量':<12} {'最佳Loss':<12} {'vs FNO':<10}")
        print("-" * 60)
        for name, r in aug.items():
            print(f"{name:<20} {r['params']:,} {r['best_loss']:<12.6f} {r['vs_fno']:+.2f}%")
        
        # 检查是否达标
        if 'CoDA-[0,2]' in aug:
            coda_vs_fno = aug['CoDA-[0,2]']['vs_fno']
            if coda_vs_fno <= -10:
                print(f"\n✅ NS目标达成! ({coda_vs_fno:+.2f}% ≤ -10%)")
            else:
                print(f"\n❌ NS目标未达成 ({coda_vs_fno:+.2f}% > -10%)")
    
    # 方案B结果
    if 'darcy_flow' in all_results['tests'] and all_results['tests']['darcy_flow']['status'] == 'success':
        darcy = all_results['tests']['darcy_flow']['results']
        print("\n【方案B: Darcy Flow】")
        print(f"{'配置':<20} {'参数量':<12} {'最佳Loss':<12} {'vs FNO':<10}")
        print("-" * 60)
        for name, r in darcy.items():
            print(f"{name:<20} {r['params']:,} {r['best_loss']:<12.6f} {r['vs_fno']:+.2f}%")
        
        # 分析MHF适用边界
        coda_results = [(k, v) for k, v in darcy.items() if 'CoDA' in k or 'MHSA' in k]
        if coda_results:
            best = min(coda_results, key=lambda x: x[1]['best_loss'])
            print(f"\nDarcy最佳配置: {best[0]} ({best[1]['vs_fno']:+.2f}% vs FNO)")
            
            # 比较NS和Darcy
            if 'data_augmentation' in all_results['tests'] and all_results['tests']['data_augmentation']['status'] == 'success':
                ns_coda = aug.get('CoDA-[0,2]', {}).get('vs_fno', 0)
                darcy_coda = best[1]['vs_fno']
                print(f"\nMHF适用边界分析:")
                print(f"  NS (抛物型): {ns_coda:+.2f}% vs FNO")
                print(f"  Darcy (椭圆型): {darcy_coda:+.2f}% vs FNO")
                if darcy_coda < ns_coda:
                    print(f"  结论: MHF在椭圆型PDE上更有效 ✅")
                else:
                    print(f"  结论: MHF在抛物型PDE上更有效")
    
    # 保存结果
    output_path = base_path / 'results' / 'iteration3_results.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n结果已保存: {output_path}")
    
    # 更新迭代日志
    log_path = base_path / 'ITERATION_LOG.md'
    with open(log_path, 'a') as f:
        f.write(f"\n| 3 | {'✅ 完成' if all_results['tests'].get('data_augmentation', {}).get('status') == 'success' else '❌'} | 待定 |\n")
    
    return all_results


if __name__ == '__main__':
    main()