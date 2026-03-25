#!/usr/bin/env python3
"""
MHF-FNO 多数据集综合测试

支持本地生成的数据文件，包括:
- Darcy Flow 2D
- Burgers 1D
- Navier-Stokes 2D

使用方法:
    python run_multi_dataset.py --datasets darcy burgers navier_stokes
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

# ============================================================================
# CPU 优化: 使用所有可用核心
# ============================================================================
torch.set_num_threads(os.cpu_count() or 1)
print(f"🔧 PyTorch 使用 {torch.get_num_threads()} 个 CPU 线程")

# 导入 MHF-FNO 核心库
sys.path.insert(0, str(Path(__file__).parent.parent))
from mhf_fno import MHFFNO, MHFFNOWithAttention, create_hybrid_fno


# ============================================================================
# 数据加载
# ============================================================================

def load_local_data(data_path, dataset_type='darcy'):
    """加载本地生成的数据"""
    print(f"\n📊 加载 {dataset_type} 数据: {data_path}")
    
    data = torch.load(data_path, weights_only=False)
    
    if isinstance(data, dict):
        x = data.get('x', data.get('train_x'))
        y = data.get('y', data.get('train_y'))
    else:
        x, y = data[0], data[1]
    
    # 确保维度正确
    if dataset_type == 'burgers':
        # 1D 数据: [N, L] -> [N, 1, L]
        if x.dim() == 2:
            x = x.unsqueeze(1)
            y = y.unsqueeze(1)
    else:
        # 2D 数据: [N, H, W] -> [N, 1, H, W]
        if x.dim() == 3:
            x = x.unsqueeze(1)
            y = y.unsqueeze(1)
    
    x = x.float()
    y = y.float()
    
    print(f"  数据形状: x={x.shape}, y={y.shape}")
    return x, y


def get_test_path(train_path):
    """从训练文件路径推导测试文件路径"""
    return train_path.replace('train', 'test')


# ============================================================================
# 模型定义 - 使用核心库
# ============================================================================

# 注意: 不再使用自定义 MHFFNO1D，改用核心库的 create_hybrid_fno
# 原因: 自定义实现缺少激活函数，导致性能严重下降


# ============================================================================
# 训练和测试
# ============================================================================

def train_model(model, train_x, train_y, test_x, test_y, config, verbose=True):
    """训练模型"""
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['epochs']
    )
    
    # 根据数据维度选择损失函数
    d = 1 if train_x.dim() == 3 else 2
    loss_fn = LpLoss(d=d, p=2, reduction='mean')
    
    results = {
        'train_losses': [],
        'test_losses': [],
        'epoch_times': [],
    }
    
    n_train = train_x.shape[0]
    batch_size = config['batch_size']
    
    for epoch in range(config['epochs']):
        t0 = time.time()
        model.train()
        
        perm = torch.randperm(n_train)
        train_loss = 0
        batch_count = 0
        
        for i in range(0, n_train, batch_size):
            bx = train_x[perm[i:i+batch_size]]
            by = train_y[perm[i:i+batch_size]]
            
            optimizer.zero_grad()
            loss = loss_fn(model(bx), by)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            batch_count += 1
        
        scheduler.step()
        epoch_time = time.time() - t0
        
        model.eval()
        with torch.no_grad():
            test_loss = loss_fn(model(test_x), test_y).item()
        
        results['train_losses'].append(train_loss / batch_count)
        results['test_losses'].append(test_loss)
        results['epoch_times'].append(epoch_time)
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{config['epochs']}: "
                  f"Train {train_loss/batch_count:.4f}, "
                  f"Test {test_loss:.4f}, "
                  f"Time {epoch_time:.1f}s")
    
    return results


def count_parameters(model):
    """计算参数量"""
    return sum(p.numel() for p in model.parameters())


def run_single_dataset(dataset_name, train_path, test_path, config):
    """运行单个数据集的测试"""
    
    # 加载数据
    train_x, train_y = load_local_data(train_path, dataset_name)
    test_x, test_y = load_local_data(test_path, dataset_name)
    
    # 限制样本数
    n_train = min(config['n_train'], train_x.shape[0])
    n_test = min(config['n_test'], test_x.shape[0])
    train_x, train_y = train_x[:n_train], train_y[:n_train]
    test_x, test_y = test_x[:n_test], test_y[:n_test]
    
    # 确定模型配置
    is_1d = train_x.dim() == 3
    if is_1d:
        resolution = train_x.shape[-1]
        n_modes = resolution // 2
        n_modes_tuple = (n_modes,)
    else:
        resolution = train_x.shape[-1]
        n_modes_tuple = (resolution // 2, resolution // 2)
    
    info = {
        'name': dataset_name,
        'resolution': resolution,
        'n_train': n_train,
        'n_test': n_test,
        'input_channels': train_x.shape[1],
        'output_channels': train_y.shape[1],
        'is_1d': is_1d,
    }
    
    print(f"\n{'='*60}")
    print(f"数据集: {dataset_name}")
    print(f"{'='*60}")
    print(f"  分辨率: {resolution}")
    print(f"  训练集: {n_train}, 测试集: {n_test}")
    print(f"  1D: {is_1d}")
    
    results = {'dataset': info, 'models': {}}
    hidden_channels = 32
    
    # 测试 FNO
    print(f"\n--- 测试 FNO (基准) ---")
    torch.manual_seed(config['seed'])
    
    if is_1d:
        model_fno = FNO(
            n_modes=n_modes_tuple,
            hidden_channels=hidden_channels,
            in_channels=info['input_channels'],
            out_channels=info['output_channels'],
            n_layers=3,
        )
    else:
        model_fno = FNO(
            n_modes=n_modes_tuple,
            hidden_channels=hidden_channels,
            in_channels=info['input_channels'],
            out_channels=info['output_channels'],
            n_layers=3,
        )
    
    params_fno = count_parameters(model_fno)
    print(f"  参数量: {params_fno:,}")
    
    train_results_fno = train_model(
        model_fno, train_x, train_y, test_x, test_y, config
    )
    
    results['models']['FNO'] = {
        'parameters': params_fno,
        'train_losses': train_results_fno['train_losses'],
        'test_losses': train_results_fno['test_losses'],
        'best_test_loss': min(train_results_fno['test_losses']),
        'avg_epoch_time': float(np.mean(train_results_fno['epoch_times'])),
    }
    
    # 测试 MHF-FNO
    print(f"\n--- 测试 MHF-FNO ---")
    torch.manual_seed(config['seed'])
    
    # 使用核心库的 create_hybrid_fno 创建模型
    # 修复: 不再使用自定义 MHFFNO1D (缺少激活函数)
    model_mhf = create_hybrid_fno(
        n_modes=n_modes_tuple,
        hidden_channels=hidden_channels,
        in_channels=info['input_channels'],
        out_channels=info['output_channels'],
        n_layers=3,
        n_heads=4,
        mhf_layers=[0, 2],  # 首尾层使用 MHF
    )
    
    params_mhf = count_parameters(model_mhf)
    param_reduction = (1 - params_mhf / params_fno) * 100
    print(f"  参数量: {params_mhf:,} ({param_reduction:.1f}% reduction)")
    
    train_results_mhf = train_model(
        model_mhf, train_x, train_y, test_x, test_y, config
    )
    
    results['models']['MHF-FNO'] = {
        'parameters': params_mhf,
        'train_losses': train_results_mhf['train_losses'],
        'test_losses': train_results_mhf['test_losses'],
        'best_test_loss': min(train_results_mhf['test_losses']),
        'avg_epoch_time': float(np.mean(train_results_mhf['epoch_times'])),
    }
    
    # 计算 MHF-FNO vs FNO 对比
    improvement = (results['models']['FNO']['best_test_loss'] - 
                   results['models']['MHF-FNO']['best_test_loss']) / \
                  results['models']['FNO']['best_test_loss'] * 100
    
    print(f"\n--- 结果对比 ---")
    print(f"  {'指标':<20} {'FNO':<15} {'MHF-FNO':<15} {'变化':<15}")
    print(f"  {'-'*60}")
    print(f"  {'参数量':<20} {params_fno:<15,} {params_mhf:<15,} {param_reduction:+.1f}%")
    print(f"  {'最佳测试Loss':<20} {results['models']['FNO']['best_test_loss']:<15.4f} "
          f"{results['models']['MHF-FNO']['best_test_loss']:<15.4f} {improvement:+.1f}%")
    
    results['comparison'] = {
        'param_reduction_pct': param_reduction,
        'loss_improvement_pct': improvement,
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='MHF-FNO 多数据集测试')
    parser.add_argument('--datasets', nargs='+', default=['darcy', 'burgers', 'navier_stokes'],
                       help='要测试的数据集')
    parser.add_argument('--data_dir', type=str, default='../data',
                       help='数据目录')
    parser.add_argument('--n_train', type=int, default=500, help='训练集大小')
    parser.add_argument('--n_test', type=int, default=100, help='测试集大小')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--output', type=str, default='multi_dataset_results.json',
                       help='输出文件')
    
    args = parser.parse_args()
    
    config = {
        'n_train': args.n_train,
        'n_test': args.n_test,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'seed': args.seed,
    }
    
    print("="*60)
    print("MHF-FNO 多数据集基准测试")
    print("="*60)
    print(f"配置: {config}")
    print(f"数据目录: {args.data_dir}")
    
    # 数据文件映射
    data_files = {
        'darcy': ('darcy_train_32.pt', 'darcy_test_32.pt'),
        'burgers': ('burgers_train_256.pt', 'burgers_test_256.pt'),
        'navier_stokes': ('ns_train_32_large.pt', 'ns_test_32_large.pt'),
    }
    
    all_results = {
        'config': config,
        'results': {},
        'timestamp': datetime.now().isoformat(),
    }
    
    data_dir = Path(args.data_dir)
    
    for dataset in args.datasets:
        if dataset not in data_files:
            print(f"⚠️  未知数据集: {dataset}")
            continue
        
        train_file, test_file = data_files[dataset]
        train_path = data_dir / train_file
        test_path = data_dir / test_file
        
        if not train_path.exists():
            print(f"⚠️  数据文件不存在: {train_path}")
            continue
        
        result = run_single_dataset(
            dataset, str(train_path), str(test_path), config
        )
        all_results['results'][dataset] = result
    
    # 保存结果
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✅ 结果已保存到: {output_path}")
    
    # 打印汇总表格
    print(f"\n{'='*80}")
    print("多数据集测试汇总")
    print(f"{'='*80}")
    print(f"{'数据集':<15} {'分辨率':<10} {'FNO参数':<12} {'MHF参数':<12} {'参数减少':<10} {'FNO Loss':<12} {'MHF Loss':<12} {'提升':<10}")
    print(f"{'-'*80}")
    
    for dataset, result in all_results['results'].items():
        info = result['dataset']
        fno = result['models']['FNO']
        mhf = result['models']['MHF-FNO']
        comp = result['comparison']
        
        print(f"{dataset:<15} {info['resolution']:<10} {fno['parameters']:<12,} "
              f"{mhf['parameters']:<12,} {comp['param_reduction_pct']:<10.1f}% "
              f"{fno['best_test_loss']:<12.4f} {mhf['best_test_loss']:<12.4f} "
              f"{comp['loss_improvement_pct']:+.2f}%")


if __name__ == '__main__':
    main()