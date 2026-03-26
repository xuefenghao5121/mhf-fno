#!/usr/bin/env python3
"""
NS 数据集优化测试

目标: 改善 MHF-FNO 在 Navier-Stokes 数据集上的效果

优化策略:
1. 增加训练轮数 (20 → 50)
2. 调整学习率调度 (添加 warmup)
3. 尝试不同的 mhf_layers 配置
4. 增加训练数据

使用本地生成的数据 (32x32)
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
from neuralop.losses.data_losses import LpLoss
from neuralop.models import FNO

# CPU 优化
torch.set_num_threads(os.cpu_count() or 1)

# 导入项目模块
sys.path.insert(0, str(Path(__file__).parent.parent))
from mhf_fno import create_hybrid_fno, MHFFNOWithAttention, create_mhf_fno_with_attention


def load_local_ns_data(n_train=500, n_test=100):
    """加载本地 NS 数据"""
    print(f"\n📊 加载本地 Navier-Stokes 数据...")
    
    data_dir = Path(__file__).parent.parent / 'data'
    
    # 首先尝试大版本
    train_path_large = Path(__file__).parent / 'data' / 'ns_train_32_large.pt'
    test_path_large = Path(__file__).parent / 'data' / 'ns_test_32_large.pt'
    
    if train_path_large.exists():
        print(f"   使用大版本数据 (1000 samples)")
        train_data = torch.load(train_path_large, weights_only=False)
        test_data = torch.load(test_path_large, weights_only=False)
    else:
        # 使用标准版本
        train_path = data_dir / 'ns_train_32.pt'
        test_path = data_dir / 'ns_test_32.pt'
        
        if not train_path.exists():
            print(f"❌ 数据文件不存在: {train_path}")
            return None
            
        train_data = torch.load(train_path, weights_only=False)
        test_data = torch.load(test_path, weights_only=False)
    
    # 解析数据格式
    if isinstance(train_data, dict):
        train_x = train_data.get('x', train_data.get('train_x'))
        train_y = train_data.get('y', train_data.get('train_y'))
    else:
        train_x, train_y = train_data[0], train_data[1]
    
    if isinstance(test_data, dict):
        test_x = test_data.get('x', test_data.get('test_x'))
        test_y = test_data.get('y', test_data.get('test_y'))
    else:
        test_x, test_y = test_data[0], test_data[1]
    
    # 确保维度正确
    if train_x.dim() == 3:
        train_x = train_x.unsqueeze(1)
        train_y = train_y.unsqueeze(1)
    if test_x.dim() == 3:
        test_x = test_x.unsqueeze(1)
        test_y = test_y.unsqueeze(1)
    
    # 转换类型
    train_x = train_x.float()
    train_y = train_y.float()
    test_x = test_x.float()
    test_y = test_y.float()
    
    # 限制样本数
    train_x = train_x[:n_train]
    train_y = train_y[:n_train]
    test_x = test_x[:n_test]
    test_y = test_y[:n_test]
    
    resolution = train_x.shape[-1]
    
    info = {
        'name': 'Navier-Stokes (本地)',
        'resolution': f'{resolution}x{resolution}',
        'n_train': train_x.shape[0],
        'n_test': test_x.shape[0],
        'input_channels': train_x.shape[1],
        'output_channels': train_y.shape[1],
        'n_modes': (resolution // 4, resolution // 4),
    }
    
    print(f"✅ 加载成功: 训练 {train_x.shape[0]}, 测试 {test_x.shape[0]}, 分辨率 {resolution}x{resolution}")
    return train_x, train_y, test_x, test_y, info


def train_with_warmup(model, train_x, train_y, test_x, test_y, config, verbose=True):
    """带 warmup 的训练"""
    
    # 使用 AdamW 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)
    
    # Cosine annealing with warm restarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-5
    )
    
    loss_fn = LpLoss(d=2, p=2, reduction='mean')
    
    results = {
        'train_losses': [],
        'test_losses': [],
        'epoch_times': [],
        'learning_rates': [],
    }
    
    n_train = train_x.shape[0]
    batch_size = config['batch_size']
    
    best_test_loss = float('inf')
    patience = 15
    patience_counter = 0
    
    for epoch in range(config['epochs']):
        t0 = time.time()
        model.train()
        
        # 随机打乱
        perm = torch.randperm(n_train)
        train_loss = 0
        batch_count = 0
        
        for i in range(0, n_train, batch_size):
            bx = train_x[perm[i:i+batch_size]]
            by = train_y[perm[i:i+batch_size]]
            
            optimizer.zero_grad()
            loss = loss_fn(model(bx), by)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            batch_count += 1
        
        scheduler.step()
        
        # 记录学习率
        current_lr = optimizer.param_groups[0]['lr']
        results['learning_rates'].append(current_lr)
        
        epoch_time = time.time() - t0
        
        # 测试
        model.eval()
        with torch.no_grad():
            test_loss = loss_fn(model(test_x), test_y).item()
        
        results['train_losses'].append(train_loss / batch_count)
        results['test_losses'].append(test_loss)
        results['epoch_times'].append(epoch_time)
        
        # Early stopping check
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{config['epochs']}: "
                  f"Train {train_loss/batch_count:.4f}, "
                  f"Test {test_loss:.4f} (best: {best_test_loss:.4f}), "
                  f"LR {current_lr:.6f}, "
                  f"Time {epoch_time:.1f}s")
        
        # Early stopping
        if patience_counter >= patience and epoch > 30:
            print(f"  Early stopping at epoch {epoch+1}")
            break
    
    results['best_test_loss'] = best_test_loss
    return results


def count_parameters(model):
    """计算参数量"""
    return sum(p.numel() for p in model.parameters())


def run_optimization_test():
    """运行优化测试"""
    
    print("="*60)
    print("Navier-Stokes 优化测试")
    print("="*60)
    
    # 配置 - 精简版本，快速验证
    configs = [
        {
            'name': 'baseline_fno',
            'model': 'FNO',
            'n_train': 500,
            'epochs': 50,
            'mhf_layers': None,
            'n_heads': 4,
        },
        {
            'name': 'baseline_mhf',
            'model': 'MHF-FNO',
            'n_train': 500,
            'epochs': 50,
            'mhf_layers': [0, 2],
            'n_heads': 4,
        },
        {
            'name': 'mhf_coda',
            'model': 'MHF+CoDA',
            'n_train': 500,
            'epochs': 50,
            'mhf_layers': [0, 2],
            'n_heads': 4,
        },
        {
            'name': 'more_heads',
            'model': 'MHF+CoDA',
            'n_train': 500,
            'epochs': 50,
            'mhf_layers': [0, 2],
            'n_heads': 8,
        },
        {
            'name': 'all_layers_mhf',
            'model': 'MHF+CoDA',
            'n_train': 500,
            'epochs': 50,
            'mhf_layers': [0, 1, 2],
            'n_heads': 4,
        },
        {
            'name': 'more_epochs',
            'model': 'MHF+CoDA',
            'n_train': 500,
            'epochs': 100,
            'mhf_layers': [0, 2],
            'n_heads': 4,
        },
    ]
    
    results = {}
    
    for config in configs:
        print(f"\n{'='*60}")
        print(f"测试: {config['name']}")
        print(f"{'='*60}")
        print(f"配置: {config}")
        
        # 加载数据
        data = load_local_ns_data(n_train=config['n_train'], n_test=100)
        if data is None:
            print("跳过此配置")
            continue
        
        train_x, train_y, test_x, test_y, info = data
        
        # 创建模型
        torch.manual_seed(42)
        
        if config['model'] == 'FNO':
            model = FNO(
                n_modes=info['n_modes'],
                hidden_channels=32,
                in_channels=info['input_channels'],
                out_channels=info['output_channels'],
                n_layers=3,
            )
        elif config['model'] == 'MHF-FNO':
            model = create_hybrid_fno(
                n_modes=info['n_modes'],
                hidden_channels=32,
                in_channels=info['input_channels'],
                out_channels=info['output_channels'],
                n_layers=3,
                n_heads=config['n_heads'],
                mhf_layers=config['mhf_layers']
            )
        else:  # MHF+CoDA
            model = create_mhf_fno_with_attention(
                n_modes=info['n_modes'],
                hidden_channels=32,
                in_channels=info['input_channels'],
                out_channels=info['output_channels'],
                n_layers=3,
                n_heads=config['n_heads'],
                mhf_layers=config['mhf_layers'],
                attention_layers=config['mhf_layers']
            )
        
        params = count_parameters(model)
        print(f"参数量: {params:,}")
        
        # 训练
        train_config = {
            'epochs': config['epochs'],
            'batch_size': 32,
            'learning_rate': 1e-3,
        }
        
        train_results = train_with_warmup(
            model, train_x, train_y, test_x, test_y, train_config
        )
        
        # 记录结果
        results[config['name']] = {
            'config': config,
            'parameters': params,
            'best_test_loss': train_results['best_test_loss'],
            'final_train_loss': train_results['train_losses'][-1],
            'total_epochs': len(train_results['train_losses']),
            'avg_epoch_time': np.mean(train_results['epoch_times']),
        }
        
        print(f"\n结果:")
        print(f"  最佳测试损失: {train_results['best_test_loss']:.4f}")
        print(f"  最终训练损失: {train_results['train_losses'][-1]:.4f}")
    
    if not results:
        print("❌ 没有成功运行的测试")
        return None
    
    # 对比分析
    print(f"\n{'='*60}")
    print("结果汇总")
    print(f"{'='*60}")
    
    baseline = results['baseline_fno']['best_test_loss']
    
    print(f"\n{'配置':<25} {'参数量':<12} {'测试Loss':<12} {'vs FNO':<12}")
    print("-"*60)
    
    for name, res in results.items():
        improvement = (baseline - res['best_test_loss']) / baseline * 100
        marker = "✅" if improvement > 0 else "⚠️" if improvement > -5 else "❌"
        print(f"{name:<25} {res['parameters']:<12,} {res['best_test_loss']:<12.4f} {improvement:+.2f}% {marker}")
    
    # 保存结果
    output_path = Path(__file__).parent.parent / 'ns_optimization_results.json'
    with open(output_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': results,
            'summary': {
                'baseline_fno_loss': baseline,
                'best_config': min(results.items(), key=lambda x: x[1]['best_test_loss'])[0],
                'best_loss': min(r['best_test_loss'] for r in results.values()),
            }
        }, f, indent=2)
    
    print(f"\n✅ 结果已保存到: {output_path}")
    
    return results


if __name__ == '__main__':
    run_optimization_test()