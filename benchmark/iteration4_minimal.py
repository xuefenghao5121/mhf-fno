#!/usr/bin/env python3
"""
迭代 4 极简版测试 - 方案A
==========================

极简配置快速验证：
- epochs = 50 (从 300 降低)
- 仅测试 3 个配置
- 预计时间: ~20分钟

作者: Tianyuan Team - 天渠
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

from mhf_fno.mhf_attention_v2 import MHFSpectralConvV2
from mhf_fno.attention_variants import CoDAStyleAttention


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def train_and_eval(
    model, train_x, train_y, test_x, test_y,
    epochs, batch_size, lr, device, verbose=True
):
    """训练并评估模型"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.MSELoss()
    
    best_test_loss = float('inf')
    best_epoch = 0
    history = []
    
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
        
        scheduler.step()
        
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
        history.append(test_loss)
        
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_epoch = epoch + 1
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: Test {test_loss:.6f} (best: {best_test_loss:.6f} @ {best_epoch})")
    
    return best_test_loss, best_epoch, history


def create_coda_model(
    n_modes, hidden_channels,
    n_heads=4,
    bottleneck=4,
    gate_init=0.1,
    mhf_layers=[0, 2]
):
    """创建带自定义参数的 CoDA 模型"""
    
    model = FNO(
        n_modes=n_modes,
        hidden_channels=hidden_channels,
        in_channels=1,
        out_channels=1,
        n_layers=3
    )
    
    head_dim = hidden_channels // n_heads
    
    for layer_idx in mhf_layers:
        # 创建自定义 CoDA 注意力
        class CustomCoDA(CoDAStyleAttention):
            def __init__(self, n_heads, head_dim, bottleneck, gate_init):
                super().__init__(n_heads, head_dim, bottleneck)
                nn.init.constant_(self.gate, gate_init)
        
        # 创建 MHF 卷积
        mhf_conv = MHFSpectralConvV2(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            n_modes=n_modes,
            n_heads=n_heads,
            attention_type='none'  # 先不用默认注意力
        )
        
        # 手动设置注意力
        mhf_conv.attention = CustomCoDA(n_heads, head_dim, bottleneck, gate_init)
        mhf_conv.use_attention = True
        
        model.fno_blocks.convs[layer_idx] = mhf_conv
    
    return model


def main():
    device = torch.device('cpu')
    base_path = Path(__file__).parent.parent
    data_path = base_path / 'data'
    results_path = base_path / 'results'
    results_path.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("迭代 4 极简版测试 - 方案A")
    print("=" * 70)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"设备: {device}")
    print(f"Epochs: 50 (极简配置)")
    
    # 加载数据
    train_data = torch.load(data_path / 'ns_train_32_large.pt', weights_only=False, map_location='cpu')
    test_data = torch.load(data_path / 'ns_test_32_large.pt', weights_only=False, map_location='cpu')
    
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
    batch_size = 32
    lr = 1e-3
    epochs = 50  # 极简配置
    
    results = {}
    
    # =========================================
    # 测试配置
    # =========================================
    test_configs = [
        {'name': 'FNO', 'type': 'fno'},
        {'name': 'CoDA-default', 'type': 'coda', 'bottleneck': 4, 'gate_init': 0.1},
        {'name': 'CoDA-optimized', 'type': 'coda', 'bottleneck': 2, 'gate_init': 0.05},
    ]
    
    for config in test_configs:
        name = config['name']
        print(f"\n{'='*70}")
        print(f"测试: {name}")
        print(f"{'='*70}")
        
        torch.manual_seed(42)
        
        if config['type'] == 'fno':
            model = FNO(
                n_modes=n_modes,
                hidden_channels=hidden_channels,
                in_channels=1,
                out_channels=1,
                n_layers=3
            ).to(device)
        else:
            model = create_coda_model(
                n_modes, hidden_channels,
                n_heads=4,
                bottleneck=config['bottleneck'],
                gate_init=config['gate_init'],
                mhf_layers=[0, 2]
            ).to(device)
        
        params = count_parameters(model)
        print(f"参数量: {params:,}")
        
        if config['type'] == 'coda':
            print(f"配置: bottleneck={config['bottleneck']}, gate_init={config['gate_init']}")
        
        t0 = time.time()
        loss, epoch, history = train_and_eval(
            model, train_x, train_y, test_x, test_y,
            epochs, batch_size, lr, device
        )
        elapsed = time.time() - t0
        
        print(f"\n最佳 Loss: {loss:.6f} @ Epoch {epoch}")
        print(f"训练时间: {elapsed:.1f}s")
        
        results[name] = {
            'params': params,
            'best_loss': loss,
            'best_epoch': epoch,
            'time': elapsed,
            'history': history[-10:]  # 保存最后10个epoch
        }
        
        if config['type'] == 'coda':
            results[name]['bottleneck'] = config['bottleneck']
            results[name]['gate_init'] = config['gate_init']
    
    # =========================================
    # 计算相对 FNO 的改进
    # =========================================
    fno_loss = results['FNO']['best_loss']
    for name in results:
        if name != 'FNO':
            diff = (results[name]['best_loss'] - fno_loss) / fno_loss * 100
            results[name]['vs_fno'] = diff
        else:
            results[name]['vs_fno'] = 0.0
    
    # =========================================
    # 汇总报告
    # =========================================
    print("\n" + "=" * 70)
    print("汇总报告")
    print("=" * 70)
    
    print(f"\n{'配置':<20} {'参数量':<12} {'最佳Loss':<12} {'vs FNO':<10}")
    print("-" * 70)
    
    for name, r in sorted(results.items(), key=lambda x: x[1]['best_loss']):
        print(f"{name:<20} {r['params']:,} {r['best_loss']:<12.6f} {r['vs_fno']:+.2f}%")
    
    # 最佳配置推荐
    best_name = min(results.items(), key=lambda x: x[1]['best_loss'])[0]
    best = results[best_name]
    
    print(f"\n{'='*70}")
    print("最佳配置推荐")
    print(f"{'='*70}")
    print(f"名称: {best_name}")
    print(f"最佳 Loss: {best['best_loss']:.6f}")
    print(f"vs FNO: {best['vs_fno']:+.2f}%")
    
    if best_name != 'FNO':
        print(f"bottleneck: {best.get('bottleneck', 'N/A')}")
        print(f"gate_init: {best.get('gate_init', 'N/A')}")
    
    # 目标检查
    print(f"\n{'='*70}")
    print("目标检查")
    print(f"{'='*70}")
    print(f"目标: ≤ -10% vs FNO")
    
    if best['vs_fno'] <= -10:
        print("✅ 目标达成!")
    else:
        gap = -10 - best['vs_fno']
        print(f"⚠️ 距离目标还差 {gap:.2f}%")
    
    # 保存结果
    output = {
        'timestamp': datetime.now().isoformat(),
        'iteration': '4-minimal-A',
        'epochs': epochs,
        'results': {k: {kk: vv for kk, vv in v.items() if kk != 'history'} for k, v in results.items()},
        'best_config': {
            'name': best_name,
            'loss': best['best_loss'],
            'vs_fno': best['vs_fno'],
            'bottleneck': best.get('bottleneck'),
            'gate_init': best.get('gate_init')
        },
        'fno_baseline': fno_loss,
        'target': '-10% vs FNO'
    }
    
    output_path = results_path / 'iteration4_minimal.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n结果已保存: {output_path}")
    
    return output


if __name__ == '__main__':
    main()