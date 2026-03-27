#!/usr/bin/env python3
"""
MHF-FNO 多数据集基准测试

支持多种数据格式:
- PT 格式 (NeuralOperator 内置 或 本地生成)
- H5 格式 (PDEBench 单文件 / Zenodo 双文件)

使用方法:
    # 使用内置数据
    python run_benchmarks.py --dataset darcy
    
    # 使用本地生成的数据
    python run_benchmarks.py --dataset darcy --data_path ./data/darcy_train_16.pt
    
    # 指定分辨率
    python run_benchmarks.py --dataset darcy --resolution 32
    
    # H5 单文件格式 (PDEBench)
    python run_benchmarks.py --dataset darcy --format h5 --data_path ./data/2D_DarcyFlow_Train.h5
    
    # H5 双文件格式 (Zenodo 下载，训练集测试集分开) ✨ 新增
    python run_benchmarks.py --dataset navier_stokes --format h5 \
        --train_path ./data/2D_NS_Re100_Train.h5 \
        --test_path ./data/2D_NS_Re100_Test.h5

依赖:
    pip install neuralop torch numpy h5py
"""

import sys
import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from neuralop.losses.data_losses import LpLoss
from neuralop.models import FNO

# 导入数据加载器
sys.path.insert(0, str(Path(__file__).parent))
from data_loader import load_dataset

# ============================================================================
# CPU 优化: 使用所有可用核心
# ============================================================================
torch.set_num_threads(os.cpu_count() or 1)


# ============================================================================
# MHF-FNO 核心实现 (使用核心库)
# ============================================================================

# 导入核心实现
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mhf_fno import MHFSpectralConv, create_hybrid_fno


class MHFFNO(nn.Module):
    """
    MHF-FNO 模型 (包装器)
    
    使用核心库的 create_hybrid_fno 创建模型，确保行为一致性。
    """
    
    def __init__(self, n_modes, hidden_channels, in_channels, out_channels, 
                 n_layers=3, n_heads=4, mhf_layers=None):
        super().__init__()
        
        # 使用核心库创建模型
        self.model = create_hybrid_fno(
            n_modes=n_modes,
            hidden_channels=hidden_channels,
            in_channels=in_channels,
            out_channels=out_channels,
            n_layers=n_layers,
            n_heads=n_heads,
            mhf_layers=mhf_layers
        )
    
    def forward(self, x):
        return self.model(x)


# ============================================================================
# 训练和测试
# ============================================================================

def train_model(model, train_x, train_y, test_x, test_y, config, verbose=True):
    """训练模型"""
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['epochs']
    )
    loss_fn = LpLoss(d=2, p=2, reduction='mean')
    
    results = {
        'train_losses': [],
        'test_losses': [],
        'epoch_times': [],
    }
    
    n_train = train_x.shape[0]
    batch_size = config['batch_size']
    eval_every = config.get('eval_every', 10)  # 每10个epoch评估一次测试集
    
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
        results['train_losses'].append(train_loss / batch_count)
        results['epoch_times'].append(epoch_time)
        
        # 定期评估测试集（不是每个epoch都评估）
        if (epoch + 1) % eval_every == 0 or epoch == config['epochs'] - 1:
            model.eval()
            with torch.no_grad():
                test_loss = loss_fn(model(test_x), test_y).item()
            results['test_losses'].append(test_loss)
            
            if verbose:
                print(f"  Epoch {epoch+1}/{config['epochs']}: "
                      f"Train {train_loss/batch_count:.4f}, "
                      f"Test {test_loss:.4f}, "
                      f"Time {epoch_time:.1f}s")
    
    # 确保test_losses长度正确（最后一个epoch可能没有评估）
    while len(results['test_losses']) < len(results['train_losses']):
        results['test_losses'].append(results['test_losses'][-1] if results['test_losses'] else float('nan'))
    
    return results


def measure_inference(model, x, n_runs=100):
    """测量推理延迟"""
    model.eval()
    
    # 预热
    with torch.no_grad():
        _ = model(x[:1])
    
    # 计时
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.time()
            _ = model(x[:1])
            times.append(time.time() - t0)
    
    return {
        'mean_ms': np.mean(times) * 1000,
        'std_ms': np.std(times) * 1000,
    }


def count_parameters(model):
    """计算参数量"""
    return sum(p.numel() for p in model.parameters())


# ============================================================================
# 主测试函数
# ============================================================================

def get_dataset_config(dataset_name):
    """
    根据数据集返回最佳MHF-FNO配置
    
    Args:
        dataset_name: 'darcy', 'burgers', 'navier_stokes'
    
    Returns:
        dict: MHF-FNO 配置参数
    """
    # 默认配置
    config = {
        'use_pino': False,
        'use_coda': False,
        'mhf_layers': [],
        'n_heads': 4,
        'pino_weight': 0.1,
    }
    
    if dataset_name == 'navier_stokes':
        # NS数据集：启用PINO + CoDA + MHF（最佳性能组合）
        # PINO对NS方程至关重要（∂u/∂t + (u·∇)u = -∇p + ν∇²u）
        config.update({
            'use_pino': True,      # 启用物理约束
            'use_coda': True,       # 启用Cross-Head Attention
            'mhf_layers': [0, 2],   # 在第1和第3层使用MHF
            'pino_weight': 0.1,
            'n_heads': 4,
        })
    elif dataset_name == 'darcy':
        # Darcy数据集：MHF + CoDA
        # 椭圆PDE不需要时间演化，PINO收益较小
        config.update({
            'use_pino': False,
            'use_coda': True,
            'mhf_layers': [0, 2],
            'n_heads': 4,
        })
    elif dataset_name == 'burgers':
        # Burgers数据集：MHF
        # 简单对流扩散，CoDA和PINO收益有限
        config.update({
            'use_pino': False,
            'use_coda': False,
            'mhf_layers': [0],
            'n_heads': 2,
        })
    
    return config


def run_benchmark(dataset_name, config):
    """运行单个数据集的基准测试"""
    
    # 加载数据 - 使用新的通用数据加载器
    data = load_dataset(
        dataset_name=dataset_name,
        data_format=config.get('data_format', 'pt'),
        train_path=config.get('train_path'),
        test_path=config.get('test_path'),
        data_path=config.get('data_path'),
        n_train=config['n_train'],
        n_test=config['n_test'],
        resolution=config.get('resolution'),
    )
    
    if data is None:
        return None
    
    train_x, train_y, test_x, test_y, info = data
    
    print(f"\n数据集信息:")
    print(f"  名称: {info['name']}")
    print(f"  分辨率: {info['resolution']}")
    print(f"  训练集: {info['n_train']}")
    print(f"  测试集: {info['n_test']}")
    
    # 模型配置
    n_modes = info['n_modes']
    hidden_channels = 32
    
    results = {'dataset': info, 'models': {}}
    
    # 测试 FNO (基准)
    print(f"\n{'='*60}")
    print(f"测试 FNO (基准)")
    print(f"{'='*60}")
    
    torch.manual_seed(config['seed'])
    # 处理不同维度的 n_modes
    if isinstance(n_modes, tuple):
        model_fno = FNO(
            n_modes=n_modes,
            hidden_channels=hidden_channels,
            in_channels=info['input_channels'],
            out_channels=info['output_channels'],
            n_layers=3,
        )
    else:
        # 1D 情况
        model_fno = FNO(
            n_modes=n_modes,
            hidden_channels=hidden_channels,
            in_channels=info['input_channels'],
            out_channels=info['output_channels'],
            n_layers=3,
        )
    
    params_fno = count_parameters(model_fno)
    print(f"参数量: {params_fno:,}")
    
    train_results_fno = train_model(
        model_fno, train_x, train_y, test_x, test_y, config
    )
    
    inference_fno = measure_inference(model_fno, test_x)
    
    results['models']['FNO'] = {
        'parameters': params_fno,
        'train_losses': train_results_fno['train_losses'],
        'test_losses': train_results_fno['test_losses'],
        'best_test_loss': min(train_results_fno['test_losses']),
        'avg_epoch_time': np.mean(train_results_fno['epoch_times']),
        'inference_ms': inference_fno['mean_ms'],
    }
    
    # 测试 MHF-FNO
    print(f"\n{'='*60}")
    print(f"测试 MHF-FNO")
    print(f"{'='*60}")
    
    # 🔥 关键修复：根据数据集自动配置模型
    mhf_config = get_dataset_config(dataset_name)
    print(f"自动配置:")
    print(f"  use_pino: {mhf_config['use_pino']}")
    print(f"  use_coda: {mhf_config['use_coda']}")
    print(f"  mhf_layers: {mhf_config['mhf_layers']}")
    print(f"  n_heads: {mhf_config['n_heads']}")
    
    torch.manual_seed(config['seed'])
    if isinstance(n_modes, tuple) and len(n_modes) == 2:
        # 2D
        model_mhf = MHFFNO(
            n_modes=n_modes,
            hidden_channels=hidden_channels,
            in_channels=info['input_channels'],
            out_channels=info['output_channels'],
            n_layers=3,
            n_heads=mhf_config['n_heads'],
            mhf_layers=mhf_config['mhf_layers'],
            use_coda=mhf_config['use_coda'],
            use_pino=mhf_config['use_pino'],
        )
    else:
        # 1D
        model_mhf = MHFFNO(
            n_modes=(n_modes[0],),
            hidden_channels=hidden_channels,
            in_channels=info['input_channels'],
            out_channels=info['output_channels'],
            n_layers=3,
            n_heads=mhf_config['n_heads'],
            mhf_layers=mhf_config['mhf_layers'],
            use_coda=mhf_config['use_coda'],
            use_pino=mhf_config['use_pino'],
        )
    
    params_mhf = count_parameters(model_mhf)
    print(f"参数量: {params_mhf:,} ({(1-params_mhf/params_fno)*100:.1f}% reduction)")
    
    train_results_mhf = train_model(
        model_mhf, train_x, train_y, test_x, test_y, config
    )
    
    inference_mhf = measure_inference(model_mhf, test_x)
    
    results['models']['MHF-FNO'] = {
        'parameters': params_mhf,
        'train_losses': train_results_mhf['train_losses'],
        'test_losses': train_results_mhf['test_losses'],
        'best_test_loss': min(train_results_mhf['test_losses']),
        'avg_epoch_time': np.mean(train_results_mhf['epoch_times']),
        'inference_ms': inference_mhf['mean_ms'],
    }
    
    # 对比
    print(f"\n{'='*60}")
    print(f"结果对比")
    print(f"{'='*60}")
    
    improvement = (results['models']['FNO']['best_test_loss'] - 
                   results['models']['MHF-FNO']['best_test_loss']) / \
                  results['models']['FNO']['best_test_loss'] * 100
    
    print(f"{'指标':<20} {'FNO':<15} {'MHF-FNO':<15} {'变化':<15}")
    print(f"{'-'*60}")
    print(f"{'参数量':<20} {params_fno:<15,} {params_mhf:<15,} {(1-params_mhf/params_fno)*100:+.1f}%")
    print(f"{'最佳测试Loss':<20} {results['models']['FNO']['best_test_loss']:<15.4f} "
          f"{results['models']['MHF-FNO']['best_test_loss']:<15.4f} {improvement:+.1f}%")
    print(f"{'推理延迟(ms)':<20} {inference_fno['mean_ms']:<15.2f} "
          f"{inference_mhf['mean_ms']:<15.2f} {(1-inference_mhf['mean_ms']/inference_fno['mean_ms'])*100:+.1f}%")
    
    return results


# ============================================================================
# 主程序
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='MHF-FNO 基准测试')
    parser.add_argument('--dataset', type=str, default='darcy',
                       choices=['darcy', 'burgers', 'navier_stokes', 'all'],
                       help='数据集选择')
    parser.add_argument('--format', type=str, default='pt',
                       choices=['pt', 'h5'],
                       help='数据格式 (pt=NeuralOperator内置或本地文件, h5=PDEBench/Zenodo)')
    parser.add_argument('--data_path', type=str, default=None,
                       help='数据文件路径 (单文件模式: PT 和 H5)')
    parser.add_argument('--train_path', type=str, default=None,
                       help='训练集文件路径 (双文件模式: Zenodo H5 或 PT)')
    parser.add_argument('--test_path', type=str, default=None,
                       help='测试集文件路径 (双文件模式: Zenodo H5 或 PT)')
    # 新增参数: 分辨率、粘度、时间步数
    parser.add_argument('--resolution', type=int, default=None,
                       help='空间分辨率 (Darcy/Navier-Stokes), 默认从文件推断')
    parser.add_argument('--viscosity', type=float, default=1e-3,
                       help='粘性系数 (Navier-Stokes)')
    parser.add_argument('--n_steps', type=int, default=100,
                       help='时间步数 (Navier-Stokes)')
    # 其他参数
    parser.add_argument('--n_train', type=int, default=1000, help='训练集大小')
    parser.add_argument('--n_test', type=int, default=200, help='测试集大小')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--output', type=str, default='benchmark_results.json',
                       help='输出文件')
    
    args = parser.parse_args()
    
    config = {
        'n_train': args.n_train,
        'n_test': args.n_test,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'seed': args.seed,
        'data_format': args.format,
        'data_path': args.data_path,
        'train_path': args.train_path,
        'test_path': args.test_path,
        'resolution': args.resolution,
        'viscosity': args.viscosity,
        'n_steps': args.n_steps,
    }
    
    print("="*60)
    print("MHF-FNO 基准测试")
    print("="*60)
    print(f"配置: {config}")
    
    # 检查双文件参数完整性
    if (args.train_path is not None and args.test_path is None) or \
       (args.train_path is None and args.test_path is not None):
        print("\n⚠️  警告: 双文件模式需要同时提供 --train_path 和 --test_path")
        print("   如果只下载了一个文件，请使用 --data_path")
        return
    
    all_results = {'config': config, 'results': {}, 'timestamp': datetime.now().isoformat()}
    
    if args.dataset == 'all':
        datasets = ['darcy', 'burgers', 'navier_stokes']
    else:
        datasets = [args.dataset]
    
    for dataset in datasets:
        print(f"\n{'#'*60}")
        print(f"# 数据集: {dataset}")
        print(f"{'#'*60}")
        
        result = run_benchmark(dataset, config)
        if result:
            all_results['results'][dataset] = result
    
    # 保存结果
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✅ 结果已保存到: {output_path}")
    
    # 生成汇总
    print(f"\n{'='*60}")
    print("汇总报告")
    print(f"{'='*60}")
    
    for dataset, result in all_results['results'].items():
        print(f"\n{result['dataset']['name']}:")
        print(f"  FNO: 参数 {result['models']['FNO']['parameters']:,}, "
              f"Loss {result['models']['FNO']['best_test_loss']:.4f}")
        print(f"  MHF-FNO: 参数 {result['models']['MHF-FNO']['parameters']:,}, "
              f"Loss {result['models']['MHF-FNO']['best_test_loss']:.4f}")


if __name__ == '__main__':
    main()
