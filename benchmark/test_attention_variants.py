#!/usr/bin/env python3
"""
跨头注意力变体对比测试
====================

对比四种注意力变体在 NS 数据集上的效果：
1. none: 纯 MHF-FNO (基线)
2. senet: SENet 风格 (原实现，已证明效果不佳)
3. mha: 真正的多头注意力 (新实现)
4. coda: CoDA-NO 风格 (新实现)
5. hybrid: 混合空间-频域注意力 (新实现)

目标：找到最佳注意力方案

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

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from mhf_fno import create_hybrid_fno
from mhf_fno.mhf_attention_v2 import create_mhf_fno_v2, ATTENTION_VARIANTS


def count_parameters(model):
    """计算参数量"""
    return sum(p.numel() for p in model.parameters())


def train_epoch(model, train_x, train_y, optimizer, loss_fn, batch_size, device):
    """训练一个 epoch"""
    model.train()
    n_train = train_x.shape[0]
    perm = torch.randperm(n_train, device=device)
    total_loss = 0
    n_batches = 0
    
    for i in range(0, n_train, batch_size):
        idx = perm[i:i+batch_size]
        bx = train_x[idx].to(device)
        by = train_y[idx].to(device)
        
        optimizer.zero_grad()
        output = model(bx)
        loss = loss_fn(output, by)
        
        if torch.isnan(loss) or torch.isinf(loss):
            continue
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / max(n_batches, 1)


def evaluate(model, test_x, test_y, loss_fn, batch_size, device):
    """评估模型"""
    model.eval()
    n_test = test_x.shape[0]
    total_loss = 0
    n_batches = 0
    
    with torch.no_grad():
        for i in range(0, n_test, batch_size):
            bx = test_x[i:i+batch_size].to(device)
            by = test_y[i:i+batch_size].to(device)
            loss = loss_fn(model(bx), by)
            if not (torch.isnan(loss) or torch.isinf(loss)):
                total_loss += loss.item()
                n_batches += 1
    
    return total_loss / max(n_batches, 1)


def run_single_experiment(model, name, train_x, train_y, test_x, test_y, config, device):
    """运行单个模型实验"""
    print(f"\n{'='*60}")
    print(f"测试: {name}")
    print(f"{'='*60}")
    
    params = count_parameters(model)
    print(f"参数量: {params:,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    loss_fn = nn.MSELoss()
    
    best_test_loss = float('inf')
    train_losses = []
    test_losses = []
    
    t0 = time.time()
    for epoch in range(config['epochs']):
        train_loss = train_epoch(model, train_x, train_y, optimizer, loss_fn, config['batch_size'], device)
        test_loss = evaluate(model, test_x, test_y, loss_fn, config['batch_size'], device)
        scheduler.step()
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        best_test_loss = min(best_test_loss, test_loss)
        
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{config['epochs']}: Train {train_loss:.6f}, Test {test_loss:.6f}")
    
    train_time = time.time() - t0
    print(f"\n最佳测试 Loss: {best_test_loss:.6f}")
    print(f"训练时间: {train_time:.1f}s")
    
    return {
        'parameters': params,
        'best_test_loss': best_test_loss,
        'final_train_loss': train_losses[-1],
        'train_time_s': train_time,
    }


def run_comparison(config):
    """运行对比实验"""
    
    device = torch.device(config.get('device', 'cpu'))
    print(f"使用设备: {device}")
    
    # 加载数据
    print(f"\n{'='*60}")
    print("加载 Navier-Stokes 数据集")
    print(f"{'='*60}")
    
    data_path = Path(config['data_dir'])
    train_data = torch.load(data_path / 'ns_train_32.pt', weights_only=False, map_location='cpu')
    test_data = torch.load(data_path / 'ns_test_32.pt', weights_only=False, map_location='cpu')
    
    train_x = train_data['x'].unsqueeze(1).float()
    train_y = train_data['y'].unsqueeze(1).float()
    test_x = test_data['x'].unsqueeze(1).float()
    test_y = test_data['y'].unsqueeze(1).float()
    
    # 数据归一化
    train_x = (train_x - train_x.mean(dim=(-2, -1), keepdim=True)) / (train_x.std(dim=(-2, -1), keepdim=True) + 1e-8)
    train_y = (train_y - train_y.mean(dim=(-2, -1), keepdim=True)) / (train_y.std(dim=(-2, -1), keepdim=True) + 1e-8)
    test_x = (test_x - test_x.mean(dim=(-2, -1), keepdim=True)) / (test_x.std(dim=(-2, -1), keepdim=True) + 1e-8)
    test_y = (test_y - test_y.mean(dim=(-2, -1), keepdim=True)) / (test_y.std(dim=(-2, -1), keepdim=True) + 1e-8)
    
    print(f"训练集: {train_x.shape}")
    print(f"测试集: {test_x.shape}")
    
    # 模型配置
    n_modes = (12, 12)
    hidden_channels = 32
    n_heads = 4
    
    results = {
        'config': config,
        'timestamp': datetime.now().isoformat(),
        'dataset': {
            'train_samples': train_x.shape[0],
            'test_samples': test_x.shape[0],
            'resolution': f"{train_x.shape[-1]}x{train_x.shape[-2]}",
        },
        'models': {}
    }
    
    # 测试所有注意力变体
    attention_types = ['none', 'senet', 'mha', 'coda', 'hybrid']
    
    for attn_type in attention_types:
        torch.manual_seed(config['seed'])
        
        if attn_type == 'none':
            # 纯 MHF-FNO
            model = create_hybrid_fno(
                n_modes=n_modes,
                hidden_channels=hidden_channels,
                n_heads=n_heads,
            ).to(device)
            name = "MHF-FNO (无注意力)"
        else:
            model = create_mhf_fno_v2(
                n_modes=n_modes,
                hidden_channels=hidden_channels,
                n_heads=n_heads,
                attention_type=attn_type,
            ).to(device)
            name = f"MHF+{attn_type.upper()}"
        
        result = run_single_experiment(model, name, train_x, train_y, test_x, test_y, config, device)
        results['models'][attn_type] = result
    
    # 汇总报告
    print(f"\n{'='*80}")
    print("汇总报告")
    print(f"{'='*80}")
    
    print(f"\n{'模型':<25} {'参数量':<12} {'最佳Loss':<12} {'vs 无注意力':<15}")
    print("-" * 70)
    
    baseline_loss = results['models']['none']['best_test_loss']
    
    for attn_type in attention_types:
        r = results['models'][attn_type]
        name = attn_type if attn_type == 'none' else f"MHF+{attn_type.upper()}"
        
        if attn_type == 'none':
            diff_str = "基线"
        else:
            diff = (r['best_test_loss'] - baseline_loss) / baseline_loss * 100
            diff_str = f"{diff:+.2f}%"
        
        print(f"{name:<25} {r['parameters']:<12,} {r['best_test_loss']:<12.6f} {diff_str:<15}")
    
    # 找出最佳方案
    best_type = min(attention_types, key=lambda x: results['models'][x]['best_test_loss'])
    best_loss = results['models'][best_type]['best_test_loss']
    
    print(f"\n{'='*80}")
    print("结论")
    print(f"{'='*80}")
    
    if best_type == 'none':
        print("⚠️ 最佳方案是纯 MHF-FNO (无注意力)，注意力机制未能带来改进")
    else:
        improvement = (baseline_loss - best_loss) / baseline_loss * 100
        print(f"✅ 最佳方案: {best_type.upper()}")
        print(f"   相比纯 MHF-FNO 改善: {improvement:.2f}%")
    
    # 保存结果
    output_path = Path(config['output_dir']) / 'attention_variants_comparison.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n结果已保存: {output_path}")
    
    return results


if __name__ == '__main__':
    config = {
        'data_dir': './data',
        'output_dir': './results',
        'epochs': 80,  # 适中训练轮数
        'batch_size': 16,
        'lr': 1e-3,
        'seed': 42,
        'device': 'cpu',
    }
    
    run_comparison(config)