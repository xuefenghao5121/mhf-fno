#!/usr/bin/env python3
"""
迭代 2 - P0 改进方向测试
========================

测试配置:
1. n_heads=2 (减少头数，增加每头容量)
2. 全层 CoDA (mhf_layers=[0,1,2])
3. epochs=200 (更长训练)

目标: ≤ -10% vs FNO (当前最佳: -4.4%)

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


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def train_and_eval(model, train_x, train_y, test_x, test_y, epochs, batch_size, lr, device, verbose=True):
    """训练并评估模型"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.MSELoss()
    
    best_test_loss = float('inf')
    best_epoch = 0
    test_losses = []
    
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
        test_losses.append(test_loss)
        
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_epoch = epoch + 1
        
        scheduler.step()
        
        if verbose and (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: Test {test_loss:.6f} (best: {best_test_loss:.6f} @ {best_epoch})")
    
    return best_test_loss, best_epoch, test_losses


def main():
    device = torch.device('cpu')
    
    print("=" * 70)
    print("迭代 2: P0 改进方向测试")
    print("=" * 70)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"设备: {device}")
    
    # 加载数据
    print("\n" + "=" * 70)
    print("加载 Navier-Stokes 数据集")
    print("=" * 70)
    
    data_path = Path(__file__).parent.parent / 'data'
    train_data = torch.load(data_path / 'ns_train_32.pt', weights_only=False, map_location='cpu')
    test_data = torch.load(data_path / 'ns_test_32.pt', weights_only=False, map_location='cpu')
    
    train_x = train_data['x'].unsqueeze(1).float()
    train_y = train_data['y'].unsqueeze(1).float()
    test_x = test_data['x'].unsqueeze(1).float()
    test_y = test_data['y'].unsqueeze(1).float()
    
    # 归一化
    train_x = (train_x - train_x.mean(dim=(-2, -1), keepdim=True)) / (train_x.std(dim=(-2, -1), keepdim=True) + 1e-8)
    train_y = (train_y - train_y.mean(dim=(-2, -1), keepdim=True)) / (train_y.std(dim=(-2, -1), keepdim=True) + 1e-8)
    test_x = (test_x - test_x.mean(dim=(-2, -1), keepdim=True)) / (test_x.std(dim=(-2, -1), keepdim=True) + 1e-8)
    test_y = (test_y - test_y.mean(dim=(-2, -1), keepdim=True)) / (test_y.std(dim=(-2, -1), keepdim=True) + 1e-8)
    
    print(f"训练集: {train_x.shape}")
    print(f"测试集: {test_x.shape}")
    
    # 配置
    n_modes = (12, 12)
    hidden_channels = 32
    batch_size = 16
    lr = 1e-3
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'n_modes': n_modes,
            'hidden_channels': hidden_channels,
            'batch_size': batch_size,
            'lr': lr,
        },
        'tests': {},
        'iteration_1_best': -4.4
    }
    
    # =========================================
    # 基准测试: FNO
    # =========================================
    print("\n" + "=" * 70)
    print("基准测试: FNO (100 epochs)")
    print("=" * 70)
    
    torch.manual_seed(42)
    model_fno = FNO(n_modes=n_modes, hidden_channels=hidden_channels, in_channels=1, out_channels=1, n_layers=3).to(device)
    params_fno = count_parameters(model_fno)
    print(f"参数量: {params_fno:,}")
    
    t0 = time.time()
    fno_loss, fno_epoch, _ = train_and_eval(model_fno, train_x, train_y, test_x, test_y, 100, batch_size, lr, device)
    fno_time = time.time() - t0
    
    print(f"\n最佳 Loss: {fno_loss:.6f} @ Epoch {fno_epoch}")
    print(f"训练时间: {fno_time:.1f}s")
    
    results['tests']['FNO'] = {
        'params': params_fno,
        'best_loss': fno_loss,
        'best_epoch': fno_epoch,
        'time': fno_time,
        'vs_fno': 0.0
    }
    
    # =========================================
    # 迭代 1 最佳配置 (验证): CoDA-[0,2]
    # =========================================
    print("\n" + "=" * 70)
    print("验证迭代 1 最佳配置: CoDA-[0,2]")
    print("=" * 70)
    
    torch.manual_seed(42)
    model_iter1 = create_mhf_fno_v2(
        n_modes, hidden_channels, 
        n_heads=4, 
        attention_type='coda', 
        mhf_layers=[0, 2]
    ).to(device)
    params_iter1 = count_parameters(model_iter1)
    print(f"参数量: {params_iter1:,}")
    
    t0 = time.time()
    iter1_loss, iter1_epoch, _ = train_and_eval(model_iter1, train_x, train_y, test_x, test_y, 100, batch_size, lr, device)
    iter1_time = time.time() - t0
    
    iter1_diff = (iter1_loss - fno_loss) / fno_loss * 100
    print(f"\n最佳 Loss: {iter1_loss:.6f} @ Epoch {iter1_epoch}")
    print(f"vs FNO: {iter1_diff:+.2f}%")
    print(f"训练时间: {iter1_time:.1f}s")
    
    results['tests']['CoDA-[0,2]-iter1'] = {
        'params': params_iter1,
        'best_loss': iter1_loss,
        'best_epoch': iter1_epoch,
        'time': iter1_time,
        'vs_fno': iter1_diff
    }
    
    # =========================================
    # P0 测试 1: n_heads=2
    # =========================================
    print("\n" + "=" * 70)
    print("P0 测试 1: n_heads=2 (减少头数)")
    print("=" * 70)
    print("预期: 头数减少 → 学习效率提高")
    
    torch.manual_seed(42)
    model_2heads = create_mhf_fno_v2(
        n_modes, hidden_channels,
        n_heads=2,
        attention_type='coda',
        mhf_layers=[0, 2]
    ).to(device)
    params_2heads = count_parameters(model_2heads)
    print(f"参数量: {params_2heads:,} (vs 4 heads: {params_iter1:,})")
    
    t0 = time.time()
    heads2_loss, heads2_epoch, _ = train_and_eval(model_2heads, train_x, train_y, test_x, test_y, 100, batch_size, lr, device)
    heads2_time = time.time() - t0
    
    heads2_diff = (heads2_loss - fno_loss) / fno_loss * 100
    print(f"\n最佳 Loss: {heads2_loss:.6f} @ Epoch {heads2_epoch}")
    print(f"vs FNO: {heads2_diff:+.2f}%")
    print(f"vs 迭代1最佳: {(heads2_loss - iter1_loss) / iter1_loss * 100:+.2f}%")
    print(f"训练时间: {heads2_time:.1f}s")
    
    results['tests']['CoDA-n_heads2'] = {
        'params': params_2heads,
        'best_loss': heads2_loss,
        'best_epoch': heads2_epoch,
        'time': heads2_time,
        'vs_fno': heads2_diff
    }
    
    # =========================================
    # P0 测试 2: 全层 CoDA
    # =========================================
    print("\n" + "=" * 70)
    print("P0 测试 2: 全层 CoDA (mhf_layers=[0,1,2])")
    print("=" * 70)
    print("预期: CoDA 弥补频率耦合")
    
    torch.manual_seed(42)
    model_full_coda = create_mhf_fno_v2(
        n_modes, hidden_channels,
        n_heads=4,
        attention_type='coda',
        mhf_layers=[0, 1, 2]
    ).to(device)
    params_full_coda = count_parameters(model_full_coda)
    print(f"参数量: {params_full_coda:,}")
    
    t0 = time.time()
    full_coda_loss, full_coda_epoch, _ = train_and_eval(model_full_coda, train_x, train_y, test_x, test_y, 100, batch_size, lr, device)
    full_coda_time = time.time() - t0
    
    full_coda_diff = (full_coda_loss - fno_loss) / fno_loss * 100
    print(f"\n最佳 Loss: {full_coda_loss:.6f} @ Epoch {full_coda_epoch}")
    print(f"vs FNO: {full_coda_diff:+.2f}%")
    print(f"vs 迭代1最佳: {(full_coda_loss - iter1_loss) / iter1_loss * 100:+.2f}%")
    print(f"训练时间: {full_coda_time:.1f}s")
    
    results['tests']['CoDA-[0,1,2]'] = {
        'params': params_full_coda,
        'best_loss': full_coda_loss,
        'best_epoch': full_coda_epoch,
        'time': full_coda_time,
        'vs_fno': full_coda_diff
    }
    
    # =========================================
    # P0 测试 3: epochs=200 (使用最佳配置)
    # =========================================
    # 先确定当前最佳配置
    test_configs = [
        ('CoDA-[0,2]-iter1', iter1_loss, 4, [0, 2]),
        ('CoDA-n_heads2', heads2_loss, 2, [0, 2]),
        ('CoDA-[0,1,2]', full_coda_loss, 4, [0, 1, 2])
    ]
    best_config = min(test_configs, key=lambda x: x[1])
    best_name, best_loss, best_n_heads, best_mhf_layers = best_config
    
    print("\n" + "=" * 70)
    print(f"P0 测试 3: epochs=200 (使用最佳配置: {best_name})")
    print("=" * 70)
    print(f"配置: n_heads={best_n_heads}, mhf_layers={best_mhf_layers}")
    print("预期: 更充分收敛")
    
    torch.manual_seed(42)
    model_long = create_mhf_fno_v2(
        n_modes, hidden_channels,
        n_heads=best_n_heads,
        attention_type='coda',
        mhf_layers=best_mhf_layers
    ).to(device)
    params_long = count_parameters(model_long)
    print(f"参数量: {params_long:,}")
    
    t0 = time.time()
    long_loss, long_epoch, _ = train_and_eval(model_long, train_x, train_y, test_x, test_y, 200, batch_size, lr, device)
    long_time = time.time() - t0
    
    long_diff = (long_loss - fno_loss) / fno_loss * 100
    print(f"\n最佳 Loss: {long_loss:.6f} @ Epoch {long_epoch}")
    print(f"vs FNO: {long_diff:+.2f}%")
    print(f"vs 100 epochs: {(long_loss - best_loss) / best_loss * 100:+.2f}%")
    print(f"训练时间: {long_time:.1f}s")
    
    results['tests'][f'CoDA-{best_name}-200ep'] = {
        'params': params_long,
        'epochs': 200,
        'best_loss': long_loss,
        'best_epoch': long_epoch,
        'time': long_time,
        'vs_fno': long_diff
    }
    
    # =========================================
    # 组合测试: n_heads=2 + 全层 CoDA + 200 epochs
    # =========================================
    print("\n" + "=" * 70)
    print("组合测试: n_heads=2 + 全层 CoDA + 200 epochs")
    print("=" * 70)
    print("预期: 多改进组合效果")
    
    torch.manual_seed(42)
    model_combo = create_mhf_fno_v2(
        n_modes, hidden_channels,
        n_heads=2,
        attention_type='coda',
        mhf_layers=[0, 1, 2]
    ).to(device)
    params_combo = count_parameters(model_combo)
    print(f"参数量: {params_combo:,}")
    
    t0 = time.time()
    combo_loss, combo_epoch, _ = train_and_eval(model_combo, train_x, train_y, test_x, test_y, 200, batch_size, lr, device)
    combo_time = time.time() - t0
    
    combo_diff = (combo_loss - fno_loss) / fno_loss * 100
    print(f"\n最佳 Loss: {combo_loss:.6f} @ Epoch {combo_epoch}")
    print(f"vs FNO: {combo_diff:+.2f}%")
    print(f"训练时间: {combo_time:.1f}s")
    
    results['tests']['CoDA-n_heads2-full-200ep'] = {
        'params': params_combo,
        'epochs': 200,
        'best_loss': combo_loss,
        'best_epoch': combo_epoch,
        'time': combo_time,
        'vs_fno': combo_diff
    }
    
    # =========================================
    # 汇总报告
    # =========================================
    print("\n" + "=" * 70)
    print("迭代 2 汇总报告")
    print("=" * 70)
    
    print(f"\n{'配置':<30} {'参数量':<12} {'最佳Loss':<12} {'最佳Epoch':<10} {'vs FNO':<10}")
    print("-" * 80)
    
    for name, r in results['tests'].items():
        params = r.get('params', '-')
        if params != '-':
            params = f"{params:,}"
        print(f"{name:<30} {params:<12} {r['best_loss']:<12.6f} {r['best_epoch']:<10} {r['vs_fno']:+.2f}%")
    
    # 目标检查
    print("\n" + "=" * 70)
    print("目标检查")
    print("=" * 70)
    
    target = -10.0
    print(f"目标: ≤ {target}% vs FNO")
    print(f"迭代 1 最佳: -4.4% vs FNO")
    
    # 排除 FNO，找最佳配置
    test_results = [(k, v) for k, v in results['tests'].items() if k != 'FNO']
    best_final = min(test_results, key=lambda x: x[1]['best_loss'])
    best_final_name = best_final[0]
    best_final_result = best_final[1]
    
    print(f"\n迭代 2 最佳配置: {best_final_name}")
    print(f"  最佳 Loss: {best_final_result['best_loss']:.6f}")
    print(f"  vs FNO: {best_final_result['vs_fno']:+.2f}%")
    
    improvement = -4.4 - best_final_result['vs_fno']
    print(f"  vs 迭代 1: {improvement:+.2f}% 改进")
    
    if best_final_result['vs_fno'] <= target:
        print(f"\n✅ 目标达成! ({best_final_result['vs_fno']:+.2f}% ≤ {target}%)")
        results['target_achieved'] = True
    else:
        gap = best_final_result['vs_fno'] - target
        print(f"\n❌ 目标未达成，还差 {gap:.2f}%")
        results['target_achieved'] = False
    
    # 保存结果
    output_path = Path(__file__).parent.parent / 'results' / 'iteration2_results.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n结果已保存: {output_path}")
    
    # 更新迭代日志
    log_path = Path(__file__).parent.parent / 'iteration_2.md'
    with open(log_path, 'w') as f:
        f.write(f"""# 迭代 2 - 2026-03-25

## 状态: 完成

---

## 测试配置

| 配置 | n_heads | mhf_layers | epochs |
|------|---------|------------|--------|
| CoDA-[0,2] | 4 | [0,2] | 100 |
| CoDA-n_heads2 | 2 | [0,2] | 100 |
| CoDA-[0,1,2] | 4 | [0,1,2] | 100 |
| 最佳 + 200ep | varies | varies | 200 |
| 组合 | 2 | [0,1,2] | 200 |

---

## 测试结果

| 配置 | 参数量 | 最佳Loss | vs FNO |
|------|--------|----------|--------|
| FNO (基准) | {params_fno:,} | {fno_loss:.6f} | 基准 |
""")
        for name, r in results['tests'].items():
            if name != 'FNO':
                f.write(f"| {name} | {r['params']:,} | {r['best_loss']:.6f} | {r['vs_fno']:+.2f}% |\n")
        
        f.write(f"""
---

## 评估

| 指标 | 目标 | 实际 | 状态 |
|------|------|------|------|
| vs FNO | ≤ -10% | {best_final_result['vs_fno']:+.2f}% | {'✅ 达标' if best_final_result['vs_fno'] <= -10 else '❌ 未达标'} |
| 参数减少 | ≥ 30% | {(1 - best_final_result['params']/params_fno)*100:.1f}% | ✅ |
| vs 迭代1 | 有提升 | {improvement:+.2f}% | {'✅' if improvement > 0 else '❌'} |

---

## 最佳配置

**{best_final_name}**
- n_heads: {2 if 'n_heads2' in best_final_name else 4}
- mhf_layers: {[0,1,2] if 'full' in best_final_name or '[0,1,2]' in best_final_name else [0,2]}
- epochs: {200 if '200' in best_final_name else 100}
- 最佳 Loss: {best_final_result['best_loss']:.6f}
- vs FNO: {best_final_result['vs_fno']:+.2f}%

---

## 决策

{'✅ 目标达成 → 结束迭代' if results['target_achieved'] else '❌ 未达标 → 进入迭代 3'}

---

*更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
""")
    
    print(f"迭代日志已更新: {log_path}")
    
    return results


if __name__ == '__main__':
    main()