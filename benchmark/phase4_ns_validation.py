#!/usr/bin/env python3
"""
Phase 4: NS 数据集验证架构师建议

测试方案:
1. 基准: MHF-FNO mhf_layers=[0,2]
2. 配置1: 全层 MHF+CoDA [0,1,2]
3. 配置2: CoDA bottleneck 调优 [2, 4, 6]
4. 配置3: 150 epochs

目标: ≤ -10% vs FNO (当前最佳: -4.5%)

作者: 天渠 @ Tianyuan Team
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
from mhf_fno.attention_variants import CoDAStyleAttention


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
        
        if verbose and (epoch + 1) % 25 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: Test {test_loss:.6f} (best: {best_test_loss:.6f} @ {best_epoch})")
    
    return best_test_loss, best_epoch, test_losses


def create_coda_with_params(n_modes, hidden_channels, n_heads, bottleneck, mhf_layers):
    """创建带自定义 bottleneck 的 CoDA 模型"""
    from mhf_fno.mhf_attention_v2 import MHFSpectralConvV2
    
    model = FNO(
        n_modes=n_modes,
        hidden_channels=hidden_channels,
        in_channels=1,
        out_channels=1,
        n_layers=3
    )
    
    for layer_idx in mhf_layers:
        if layer_idx < 3:
            # 创建自定义 CoDA
            class CustomCoDAConv(MHFSpectralConvV2):
                def __init__(self, *args, bottleneck=4, **kwargs):
                    super().__init__(*args, attention_type='coda', **kwargs)
                    # 替换注意力模块
                    if self.use_attention:
                        self.attention = CoDAStyleAttention(
                            n_heads=self.n_heads,
                            head_dim=self.head_out,
                            bottleneck=bottleneck
                        )
            
            mhf_conv = CustomCoDAConv(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                n_modes=n_modes,
                n_heads=n_heads,
                bottleneck=bottleneck
            )
            model.fno_blocks.convs[layer_idx] = mhf_conv
    
    return model


def main():
    device = torch.device('cpu')
    
    print("=" * 70)
    print("Phase 4: NS 数据集验证 - 架构师建议测试")
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
    n_heads = 4
    batch_size = 16
    lr = 1e-3
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'n_modes': n_modes,
            'hidden_channels': hidden_channels,
            'n_heads': n_heads,
            'batch_size': batch_size,
            'lr': lr,
        },
        'tests': {}
    }
    
    # 基准 FNO (用于对比)
    print("\n" + "=" * 70)
    print("基准测试: FNO")
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
        'time': fno_time
    }
    
    # 测试 1: 基准 MHF-FNO [0,2]
    print("\n" + "=" * 70)
    print("测试 1: 基准 MHF-FNO (mhf_layers=[0,2])")
    print("=" * 70)
    
    torch.manual_seed(42)
    model_baseline = create_hybrid_fno(n_modes, hidden_channels, n_heads=n_heads, mhf_layers=[0, 2]).to(device)
    params_baseline = count_parameters(model_baseline)
    print(f"参数量: {params_baseline:,}")
    
    t0 = time.time()
    baseline_loss, baseline_epoch, _ = train_and_eval(model_baseline, train_x, train_y, test_x, test_y, 100, batch_size, lr, device)
    baseline_time = time.time() - t0
    
    baseline_diff = (baseline_loss - fno_loss) / fno_loss * 100
    print(f"\n最佳 Loss: {baseline_loss:.6f} @ Epoch {baseline_epoch}")
    print(f"vs FNO: {baseline_diff:+.2f}%")
    print(f"训练时间: {baseline_time:.1f}s")
    
    results['tests']['MHF-FNO-Baseline'] = {
        'params': params_baseline,
        'best_loss': baseline_loss,
        'best_epoch': baseline_epoch,
        'time': baseline_time,
        'vs_fno': baseline_diff
    }
    
    # 测试 2: 全层 MHF+CoDA [0,1,2]
    print("\n" + "=" * 70)
    print("测试 2: 全层 MHF+CoDA (mhf_layers=[0,1,2])")
    print("=" * 70)
    
    torch.manual_seed(42)
    model_full = create_mhf_fno_v2(n_modes, hidden_channels, n_heads=n_heads, attention_type='coda', mhf_layers=[0, 1, 2]).to(device)
    params_full = count_parameters(model_full)
    print(f"参数量: {params_full:,}")
    
    t0 = time.time()
    full_loss, full_epoch, _ = train_and_eval(model_full, train_x, train_y, test_x, test_y, 100, batch_size, lr, device)
    full_time = time.time() - t0
    
    full_diff = (full_loss - fno_loss) / fno_loss * 100
    print(f"\n最佳 Loss: {full_loss:.6f} @ Epoch {full_epoch}")
    print(f"vs FNO: {full_diff:+.2f}%")
    print(f"vs 基准: {(full_loss - baseline_loss) / baseline_loss * 100:+.2f}%")
    print(f"训练时间: {full_time:.1f}s")
    
    results['tests']['Full-CoDA'] = {
        'params': params_full,
        'best_loss': full_loss,
        'best_epoch': full_epoch,
        'time': full_time,
        'vs_fno': full_diff
    }
    
    # 测试 3: CoDA bottleneck 调优
    print("\n" + "=" * 70)
    print("测试 3: CoDA bottleneck 调优")
    print("=" * 70)
    
    bottleneck_tests = [2, 4, 6]
    
    for bn in bottleneck_tests:
        print(f"\n  Bottleneck = {bn}")
        print("  " + "-" * 50)
        
        torch.manual_seed(42)
        model_bn = create_coda_with_params(n_modes, hidden_channels, n_heads, bottleneck=bn, mhf_layers=[0, 1, 2])
        model_bn = model_bn.to(device)
        
        t0 = time.time()
        bn_loss, bn_epoch, _ = train_and_eval(model_bn, train_x, train_y, test_x, test_y, 100, batch_size, lr, device, verbose=False)
        bn_time = time.time() - t0
        
        bn_diff = (bn_loss - fno_loss) / fno_loss * 100
        print(f"  最佳 Loss: {bn_loss:.6f} @ Epoch {bn_epoch}")
        print(f"  vs FNO: {bn_diff:+.2f}%")
        
        results['tests'][f'CoDA-bn{bn}'] = {
            'bottleneck': bn,
            'best_loss': bn_loss,
            'best_epoch': bn_epoch,
            'time': bn_time,
            'vs_fno': bn_diff
        }
    
    # 测试 4: 150 epochs (使用最佳配置)
    # 先确定最佳 bottleneck
    best_bn_result = min(
        [(k, v) for k, v in results['tests'].items() if k.startswith('CoDA-bn')],
        key=lambda x: x[1]['best_loss']
    )
    best_bn = best_bn_result[1]['bottleneck']
    
    print("\n" + "=" * 70)
    print(f"测试 4: 150 epochs (全层 CoDA, bottleneck={best_bn})")
    print("=" * 70)
    
    torch.manual_seed(42)
    model_long = create_coda_with_params(n_modes, hidden_channels, n_heads, bottleneck=best_bn, mhf_layers=[0, 1, 2])
    model_long = model_long.to(device)
    
    t0 = time.time()
    long_loss, long_epoch, _ = train_and_eval(model_long, train_x, train_y, test_x, test_y, 150, batch_size, lr, device)
    long_time = time.time() - t0
    
    long_diff = (long_loss - fno_loss) / fno_loss * 100
    print(f"\n最佳 Loss: {long_loss:.6f} @ Epoch {long_epoch}")
    print(f"vs FNO: {long_diff:+.2f}%")
    print(f"训练时间: {long_time:.1f}s")
    
    results['tests']['CoDA-150epochs'] = {
        'bottleneck': best_bn,
        'epochs': 150,
        'best_loss': long_loss,
        'best_epoch': long_epoch,
        'time': long_time,
        'vs_fno': long_diff
    }
    
    # 汇总报告
    print("\n" + "=" * 70)
    print("汇总报告")
    print("=" * 70)
    
    print(f"\n{'配置':<20} {'参数量':<12} {'最佳Loss':<12} {'最佳Epoch':<10} {'vs FNO':<10}")
    print("-" * 70)
    
    for name, r in results['tests'].items():
        params = r.get('params', '-')
        if params != '-':
            params = f"{params:,}"
        print(f"{name:<20} {params:<12} {r['best_loss']:<12.6f} {r['best_epoch']:<10} {r['vs_fno']:+.2f}%")
    
    # 目标检查
    print("\n" + "=" * 70)
    print("目标检查")
    print("=" * 70)
    
    target = -10.0
    print(f"目标: ≤ {target}% vs FNO")
    print(f"当前最佳: {baseline_diff:.2f}% vs FNO")
    
    best_config = min(results['tests'].items(), key=lambda x: x[1]['best_loss'])
    best_name = best_config[0]
    best_result = best_config[1]
    
    print(f"\n最佳配置: {best_name}")
    print(f"  最佳 Loss: {best_result['best_loss']:.6f}")
    print(f"  vs FNO: {best_result['vs_fno']:+.2f}%")
    
    if best_result['vs_fno'] <= target:
        print(f"\n✅ 目标达成! ({best_result['vs_fno']:+.2f}% ≤ {target}%)")
    else:
        gap = best_result['vs_fno'] - target
        print(f"\n❌ 目标未达成，还差 {gap:.2f}%")
        print(f"\n建议:")
        print(f"  1. 进一步增加 epochs (150 → 200)")
        print(f"  2. 调整学习率策略")
        print(f"  3. 增加模型容量 (hidden_channels)")
    
    # 保存结果
    output_path = Path('./results/phase4_validation.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n结果已保存: {output_path}")
    
    return results


if __name__ == '__main__':
    main()