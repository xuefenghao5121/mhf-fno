#!/usr/bin/env python3
"""
方案A快速测试: 三数据集 × 三模型

配置:
- epochs: 20
- n_train: 200
- n_test: 50
- batch_size: 32
- learning_rate: 1e-3
- seed: 42

测试矩阵:
| 数据集   | 分辨率  | FNO | MHF-FNO | MHF+CoDA |
|----------|---------|-----|---------|----------|
| Darcy    | 32×32   | ✅  | ✅      | ✅       |
| Burgers  | 256     | ✅  | ✅      | ✅       |
| NS       | 32×32   | ✅  | ✅      | ✅       |
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from neuralop.losses.data_losses import LpLoss
from neuralop.models import FNO

# CPU 优化
torch.set_num_threads(os.cpu_count() or 1)

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from mhf_fno import MHFSpectralConv, create_hybrid_fno
from mhf_fno.mhf_attention import MHFFNOWithAttention


# ============================================================================
# 数据加载
# ============================================================================

def load_local_data(dataset_name, data_dir='./data'):
    """加载本地数据"""
    data_dir = Path(data_dir)
    
    if dataset_name == 'darcy':
        train_path = data_dir / 'darcy_train_32.pt'
        test_path = data_dir / 'darcy_test_32.pt'
        resolution = 32
        is_1d = False
    elif dataset_name == 'burgers':
        train_path = data_dir / 'burgers_train_256.pt'
        test_path = data_dir / 'burgers_test_256.pt'
        resolution = 256
        is_1d = True
    elif dataset_name == 'navier_stokes' or dataset_name == 'ns':
        train_path = data_dir / 'ns_train_32.pt'
        test_path = data_dir / 'ns_test_32.pt'
        resolution = 32
        is_1d = False
    else:
        raise ValueError(f"未知数据集: {dataset_name}")
    
    print(f"\n📊 加载 {dataset_name} 数据 ({resolution}{'×' + str(resolution) if not is_1d else ''})...")
    
    train_data = torch.load(train_path, weights_only=False)
    test_data = torch.load(test_path, weights_only=False)
    
    # 解析数据格式
    if isinstance(train_data, dict):
        train_x = train_data.get('x', train_data.get('train_x'))
        train_y = train_data.get('y', train_data.get('train_y'))
        test_x = test_data.get('x', test_data.get('test_x'))
        test_y = test_data.get('y', test_data.get('test_y'))
    else:
        train_x, train_y = train_data[0], train_data[1]
        test_x, test_y = test_data[0], test_data[1]
    
    # 确保维度正确
    if train_x.dim() == 2:  # [N, L] - 1D without channel
        train_x = train_x.unsqueeze(1)  # [N, 1, L]
        train_y = train_y.unsqueeze(1)
        test_x = test_x.unsqueeze(1)
        test_y = test_y.unsqueeze(1)
    elif train_x.dim() == 3:  # [N, H, W] - 2D without channel
        train_x = train_x.unsqueeze(1)  # [N, 1, H, W]
        train_y = train_y.unsqueeze(1)
        test_x = test_x.unsqueeze(1)
        test_y = test_y.unsqueeze(1)
    
    # 转换为 float
    train_x = train_x.float()
    train_y = train_y.float()
    test_x = test_x.float()
    test_y = test_y.float()
    
    info = {
        'name': dataset_name,
        'resolution': resolution,
        'is_1d': is_1d,
        'train_size': train_x.shape[0],
        'test_size': test_x.shape[0],
        'input_channels': train_x.shape[1],
        'output_channels': train_y.shape[1],
    }
    
    print(f"✅ 加载成功: 训练 {train_x.shape[0]}, 测试 {test_x.shape[0]}")
    return train_x, train_y, test_x, test_y, info


# ============================================================================
# 模型创建
# ============================================================================

def create_fno(info, hidden_channels=32):
    """创建标准 FNO 模型"""
    if info['is_1d']:
        n_modes = (info['resolution'] // 2,)
        # 1D 数据需要禁用 positional_embedding
        return FNO(
            n_modes=n_modes,
            hidden_channels=hidden_channels,
            in_channels=info['input_channels'],
            out_channels=info['output_channels'],
            n_layers=3,
            positional_embedding=None,  # 禁用 1D 的位置嵌入
        )
    else:
        n_modes = (info['resolution'] // 2, info['resolution'] // 2)
        return FNO(
            n_modes=n_modes,
            hidden_channels=hidden_channels,
            in_channels=info['input_channels'],
            out_channels=info['output_channels'],
            n_layers=3,
        )


def create_mhf_fno(info, hidden_channels=32, n_heads=4):
    """创建 MHF-FNO 模型"""
    if info['is_1d']:
        n_modes = (info['resolution'] // 2,)
    else:
        n_modes = (info['resolution'] // 2, info['resolution'] // 2)
    
    return create_hybrid_fno(
        n_modes=n_modes,
        hidden_channels=hidden_channels,
        in_channels=info['input_channels'],
        out_channels=info['output_channels'],
        n_layers=3,
        n_heads=n_heads,
        mhf_layers=[0, 2],
        positional_embedding=None if info['is_1d'] else 'grid'
    )


def create_mhf_coda(info, hidden_channels=32, n_heads=4):
    """创建 MHF+CoDA 模型"""
    if info['is_1d']:
        n_modes = (info['resolution'] // 2,)
    else:
        n_modes = (info['resolution'] // 2, info['resolution'] // 2)
    
    return MHFFNOWithAttention.best_config(
        n_modes=n_modes,
        hidden_channels=hidden_channels,
        in_channels=info['input_channels'],
        out_channels=info['output_channels'],
        n_heads=n_heads,
        positional_embedding=None if info['is_1d'] else 'grid'
    )


# ============================================================================
# 训练
# ============================================================================

def train_model(model, train_x, train_y, test_x, test_y, config, verbose=True):
    """训练模型"""
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['epochs']
    )
    
    # 根据维度选择损失函数
    if train_x.dim() == 4:  # 2D
        loss_fn = LpLoss(d=2, p=2, reduction='mean')
    else:  # 1D
        loss_fn = LpLoss(d=1, p=2, reduction='mean')
    
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
            optimizer.step()
            
            train_loss += loss.item()
            batch_count += 1
        
        scheduler.step()
        epoch_time = time.time() - t0
        
        # 测试
        model.eval()
        with torch.no_grad():
            test_loss = loss_fn(model(test_x), test_y).item()
        
        results['train_losses'].append(train_loss / batch_count)
        results['test_losses'].append(test_loss)
        results['epoch_times'].append(epoch_time)
        
        if verbose and (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{config['epochs']}: "
                  f"Train {train_loss/batch_count:.4f}, "
                  f"Test {test_loss:.4f}, "
                  f"Time {epoch_time:.1f}s")
    
    return results


def count_parameters(model):
    """计算参数量"""
    return sum(p.numel() for p in model.parameters())


# ============================================================================
# 主测试函数
# ============================================================================

def run_plan_a(data_dir='./data', output_dir='.'):
    """运行方案A测试"""
    
    config = {
        'epochs': 20,
        'n_train': 200,
        'n_test': 50,
        'batch_size': 32,
        'learning_rate': 1e-3,
        'seed': 42
    }
    
    print("="*60)
    print("方案A快速测试")
    print("="*60)
    print(f"配置: {config}")
    
    datasets = ['darcy', 'burgers', 'navier_stokes']
    models = ['FNO', 'MHF-FNO', 'MHF+CoDA']
    
    all_results = {
        'config': config,
        'results': {},
        'timestamp': datetime.now().isoformat()
    }
    
    total_start = time.time()
    
    for dataset_name in datasets:
        print(f"\n{'#'*60}")
        print(f"# 数据集: {dataset_name}")
        print(f"{'#'*60}")
        
        # 加载数据
        train_x, train_y, test_x, test_y, info = load_local_data(
            dataset_name, data_dir
        )
        
        # 限制样本数
        train_x = train_x[:config['n_train']]
        train_y = train_y[:config['n_train']]
        test_x = test_x[:config['n_test']]
        test_y = test_y[:config['n_test']]
        
        dataset_results = {'dataset': info, 'models': {}}
        
        for model_name in models:
            print(f"\n{'='*60}")
            print(f"测试 {model_name}")
            print(f"{'='*60}")
            
            # 创建模型
            torch.manual_seed(config['seed'])
            if model_name == 'FNO':
                model = create_fno(info)
            elif model_name == 'MHF-FNO':
                model = create_mhf_fno(info)
            else:  # MHF+CoDA
                model = create_mhf_coda(info)
            
            params = count_parameters(model)
            print(f"参数量: {params:,}")
            
            # 训练
            t0 = time.time()
            train_results = train_model(
                model, train_x, train_y, test_x, test_y, config
            )
            train_time = time.time() - t0
            
            best_test_loss = min(train_results['test_losses'])
            avg_epoch_time = np.mean(train_results['epoch_times'])
            
            dataset_results['models'][model_name] = {
                'parameters': params,
                'best_test_loss': best_test_loss,
                'avg_epoch_time': avg_epoch_time,
                'total_time': train_time,
                'train_losses': train_results['train_losses'],
                'test_losses': train_results['test_losses'],
            }
            
            print(f"\n结果: 最佳测试 Loss = {best_test_loss:.4f}, "
                  f"平均 epoch 时间 = {avg_epoch_time:.2f}s")
        
        # 计算相对 FNO 的改进
        fno_loss = dataset_results['models']['FNO']['best_test_loss']
        for model_name in ['MHF-FNO', 'MHF+CoDA']:
            model_loss = dataset_results['models'][model_name]['best_test_loss']
            improvement = (fno_loss - model_loss) / fno_loss * 100
            dataset_results['models'][model_name]['improvement_vs_fno'] = improvement
        
        all_results['results'][dataset_name] = dataset_results
    
    total_time = time.time() - total_start
    
    # 保存结果
    output_path = Path(output_dir) / 'plan_a_results.json'
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✅ 结果已保存到: {output_path}")
    
    # 打印汇总表格
    print(f"\n{'='*80}")
    print("方案A测试结果汇总")
    print(f"{'='*80}")
    print(f"总耗时: {total_time:.1f}s ({total_time/60:.1f}分钟)")
    print()
    print(f"{'数据集':<15} {'模型':<12} {'参数量':<12} {'Test Loss':<12} {'vs FNO':<10}")
    print("-"*80)
    
    for dataset_name in datasets:
        dataset_results = all_results['results'][dataset_name]
        for model_name in models:
            model_data = dataset_results['models'][model_name]
            params = model_data['parameters']
            loss = model_data['best_test_loss']
            
            if model_name == 'FNO':
                vs_fno = "-"
            else:
                improvement = model_data.get('improvement_vs_fno', 0)
                vs_fno = f"{improvement:+.2f}%"
            
            print(f"{dataset_name:<15} {model_name:<12} {params:<12,} {loss:<12.4f} {vs_fno:<10}")
    
    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='方案A快速测试')
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='数据目录')
    parser.add_argument('--output_dir', type=str, default='.',
                       help='输出目录')
    
    args = parser.parse_args()
    
    run_plan_a(args.data_dir, args.output_dir)