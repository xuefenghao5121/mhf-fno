#!/usr/bin/env python3
"""
完整三模型基准测试: FNO, MHF-FNO, MHF+CoDA

测试矩阵:
- 数据集: Darcy 32×32, Burgers 256, Navier-Stokes 32×32
- 模型: FNO (基准), MHF-FNO, MHF+CoDA
- 配置: n_train=500, n_test=100, epochs=50, batch_size=32, lr=1e-3, seed=42

使用方法:
    python run_full_benchmarks.py --generate-data
    python run_full_benchmarks.py --dataset darcy --resolution 32
    python run_full_benchmarks.py --dataset all
"""

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

# CPU 优化
torch.set_num_threads(os.cpu_count() or 1)

# 项目模块
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from mhf_fno import MHFFNO, MHFFNOWithAttention, create_hybrid_fno


# ============================================================================
# 数据加载函数 (复用 generate_data.py 的生成器)
# ============================================================================

def load_local_darcy(data_path, n_train, n_test):
    """加载本地 Darcy 数据"""
    data = torch.load(data_path, weights_only=False)
    if isinstance(data, dict):
        train_x = data.get('x', data.get('train_x'))
        train_y = data.get('y', data.get('train_y'))
    else:
        train_x, train_y = data[0], data[1]
    
    if train_x.dim() == 3:
        train_x = train_x.unsqueeze(1).float()
        train_y = train_y.unsqueeze(1).float()
    else:
        train_x = train_x.float()
        train_y = train_y.float()
    
    # 分割
    train_x = train_x[:n_train]
    train_y = train_y[:n_train]
    test_x = train_x[n_train:n_train+n_test] if len(train_x) > n_train else train_x[:n_test]
    test_y = train_y[n_train:n_train+n_test] if len(train_y) > n_train else train_y[:n_test]
    
    resolution = train_x.shape[-1]
    return train_x, train_y, test_x, test_y, resolution


def load_local_burgers(data_path, n_train, n_test):
    """加载本地 Burgers 数据"""
    data = torch.load(data_path, weights_only=False)
    if isinstance(data, dict):
        train_x = data.get('x', data.get('train_x'))
        train_y = data.get('y', data.get('train_y'))
    else:
        train_x, train_y = data[0], data[1]
    
    if train_x.dim() == 2:
        train_x = train_x.unsqueeze(1).float()
        train_y = train_y.unsqueeze(1).float()
    else:
        train_x = train_x.float()
        train_y = train_y.float()
    
    train_x = train_x[:n_train]
    train_y = train_y[:n_train]
    test_x = train_x[n_train:n_train+n_test] if len(train_x) > n_train else train_x[:n_test]
    test_y = train_y[n_train:n_train+n_test] if len(train_y) > n_train else train_y[:n_test]
    
    resolution = train_x.shape[-1]
    return train_x, train_y, test_x, test_y, resolution


def load_local_ns(data_path, n_train, n_test):
    """加载本地 Navier-Stokes 数据"""
    return load_local_darcy(data_path, n_train, n_test)  # 格式相同


# ============================================================================
# 数据生成 (调用 generate_data.py)
# ============================================================================

def generate_all_data(config):
    """生成所有数据集"""
    from generate_data import (
        generate_darcy_flow,
        generate_burgers_1d,
        generate_navier_stokes_2d
    )
    
    output_dir = config.get('output_dir', './data')
    device = config.get('device', 'cpu')
    
    results = {}
    
    # Darcy 32×32
    print("\n" + "="*60)
    print("生成 Darcy Flow 32×32")
    print("="*60)
    results['darcy'] = generate_darcy_flow(
        n_train=config['n_train'],
        n_test=config['n_test'],
        resolution=32,
        output_dir=output_dir,
        device=device
    )
    
    # Burgers 256
    print("\n" + "="*60)
    print("生成 Burgers 1D (256 points)")
    print("="*60)
    results['burgers'] = generate_burgers_1d(
        n_train=config['n_train'],
        n_test=config['n_test'],
        n_points=256,
        viscosity=0.1,
        output_dir=output_dir,
        device=device
    )
    
    # Navier-Stokes 32×32
    print("\n" + "="*60)
    print("生成 Navier-Stokes 32×32")
    print("="*60)
    results['navier_stokes'] = generate_navier_stokes_2d(
        n_train=config['n_train'],
        n_test=config['n_test'],
        resolution=32,
        viscosity=1e-3,
        n_steps=100,
        output_dir=output_dir,
        device=device
    )
    
    return results


# ============================================================================
# 训练和测试函数
# ============================================================================

def train_model(model, train_x, train_y, test_x, test_y, config, model_name="Model"):
    """训练模型并记录完整的 loss 曲线"""
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['epochs']
    )
    
    # 根据维度选择 loss
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
    
    print(f"\n训练 {model_name}...")
    
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
            pred = model(bx)
            loss = loss_fn(pred, by)
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
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{config['epochs']}: "
                  f"Train {train_loss/batch_count:.4f}, "
                  f"Test {test_loss:.4f}, "
                  f"Time {epoch_time:.1f}s")
    
    return results


def count_parameters(model):
    """计算参数量"""
    return sum(p.numel() for p in model.parameters())


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


# ============================================================================
# 单数据集测试
# ============================================================================

def run_dataset_benchmark(dataset_name, config, data_dir='./data'):
    """运行单个数据集的三模型测试"""
    
    print(f"\n{'#'*70}")
    print(f"# 数据集: {dataset_name}")
    print(f"{'#'*70}")
    
    # 加载数据
    if dataset_name == 'darcy':
        data_path = f"{data_dir}/darcy_train_32.pt"
        train_x, train_y, test_x, test_y, resolution = load_local_darcy(
            data_path, config['n_train'], config['n_test']
        )
        n_modes = (resolution // 2, resolution // 2)
        is_1d = False
    elif dataset_name == 'burgers':
        data_path = f"{data_dir}/burgers_train_256.pt"
        train_x, train_y, test_x, test_y, resolution = load_local_burgers(
            data_path, config['n_train'], config['n_test']
        )
        n_modes = (resolution // 2,)
        is_1d = True
    elif dataset_name == 'navier_stokes':
        data_path = f"{data_dir}/ns_train_32.pt"
        train_x, train_y, test_x, test_y, resolution = load_local_ns(
            data_path, config['n_train'], config['n_test']
        )
        n_modes = (resolution // 2, resolution // 2)
        is_1d = False
    else:
        print(f"❌ 未知数据集: {dataset_name}")
        return None
    
    print(f"\n数据集信息:")
    print(f"  分辨率: {resolution}{'×'+str(resolution) if not is_1d else ''}")
    print(f"  训练集: {train_x.shape[0]}")
    print(f"  测试集: {test_x.shape[0]}")
    print(f"  n_modes: {n_modes}")
    
    in_channels = train_x.shape[1]
    out_channels = train_y.shape[1]
    hidden_channels = 32
    
    results = {
        'dataset': dataset_name,
        'resolution': resolution,
        'n_modes': n_modes,
        'models': {}
    }
    
    # =========================================================================
    # 模型 1: FNO (基准)
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"测试 FNO (基准)")
    print(f"{'='*60}")
    
    torch.manual_seed(config['seed'])
    model_fno = FNO(
        n_modes=n_modes,
        hidden_channels=hidden_channels,
        in_channels=in_channels,
        out_channels=out_channels,
        n_layers=3,
    )
    
    params_fno = count_parameters(model_fno)
    print(f"参数量: {params_fno:,}")
    
    train_fno = train_model(model_fno, train_x, train_y, test_x, test_y, config, "FNO")
    inference_fno = measure_inference(model_fno, test_x)
    
    results['models']['FNO'] = {
        'parameters': params_fno,
        'train_losses': train_fno['train_losses'],
        'test_losses': train_fno['test_losses'],
        'best_test_loss': min(train_fno['test_losses']),
        'final_test_loss': train_fno['test_losses'][-1],
        'avg_epoch_time': np.mean(train_fno['epoch_times']),
        'inference_ms': inference_fno['mean_ms'],
    }
    
    # =========================================================================
    # 模型 2: MHF-FNO
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"测试 MHF-FNO")
    print(f"{'='*60}")
    
    torch.manual_seed(config['seed'])
    model_mhf = MHFFNO.best_config(
        n_modes=n_modes,
        hidden_channels=hidden_channels,
        in_channels=in_channels,
        out_channels=out_channels,
    )
    
    params_mhf = count_parameters(model_mhf)
    print(f"参数量: {params_mhf:,} ({(1-params_mhf/params_fno)*100:.1f}% reduction)")
    
    train_mhf = train_model(model_mhf, train_x, train_y, test_x, test_y, config, "MHF-FNO")
    inference_mhf = measure_inference(model_mhf, test_x)
    
    results['models']['MHF-FNO'] = {
        'parameters': params_mhf,
        'train_losses': train_mhf['train_losses'],
        'test_losses': train_mhf['test_losses'],
        'best_test_loss': min(train_mhf['test_losses']),
        'final_test_loss': train_mhf['test_losses'][-1],
        'avg_epoch_time': np.mean(train_mhf['epoch_times']),
        'inference_ms': inference_mhf['mean_ms'],
        'param_reduction': (1 - params_mhf/params_fno) * 100,
    }
    
    # =========================================================================
    # 模型 3: MHF+CoDA
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"测试 MHF+CoDA")
    print(f"{'='*60}")
    
    torch.manual_seed(config['seed'])
    model_coda = MHFFNOWithAttention.best_config(
        n_modes=n_modes,
        hidden_channels=hidden_channels,
        in_channels=in_channels,
        out_channels=out_channels,
        n_heads=4,
    )
    
    params_coda = count_parameters(model_coda)
    print(f"参数量: {params_coda:,} ({(1-params_coda/params_fno)*100:.1f}% reduction)")
    
    train_coda = train_model(model_coda, train_x, train_y, test_x, test_y, config, "MHF+CoDA")
    inference_coda = measure_inference(model_coda, test_x)
    
    results['models']['MHF+CoDA'] = {
        'parameters': params_coda,
        'train_losses': train_coda['train_losses'],
        'test_losses': train_coda['test_losses'],
        'best_test_loss': min(train_coda['test_losses']),
        'final_test_loss': train_coda['test_losses'][-1],
        'avg_epoch_time': np.mean(train_coda['epoch_times']),
        'inference_ms': inference_coda['mean_ms'],
        'param_reduction': (1 - params_coda/params_fno) * 100,
    }
    
    # =========================================================================
    # 结果对比
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"结果对比: {dataset_name}")
    print(f"{'='*60}")
    
    fno_loss = results['models']['FNO']['best_test_loss']
    mhf_loss = results['models']['MHF-FNO']['best_test_loss']
    coda_loss = results['models']['MHF+CoDA']['best_test_loss']
    
    print(f"\n{'模型':<15} {'参数量':>12} {'参数减少':>10} {'最佳Loss':>12} {'vs FNO':>10}")
    print(f"{'-'*60}")
    print(f"{'FNO':<15} {params_fno:>12,} {'-':>10} {fno_loss:>12.4f} {'基准':>10}")
    print(f"{'MHF-FNO':<15} {params_mhf:>12,} {(1-params_mhf/params_fno)*100:>9.1f}% {mhf_loss:>12.4f} {(mhf_loss-fno_loss)/fno_loss*100:>9.1f}%")
    print(f"{'MHF+CoDA':<15} {params_coda:>12,} {(1-params_coda/params_fno)*100:>9.1f}% {coda_loss:>12.4f} {(coda_loss-fno_loss)/fno_loss*100:>9.1f}%")
    
    return results


# ============================================================================
# 主程序
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='MHF-FNO 完整三模型基准测试')
    parser.add_argument('--dataset', type=str, default='all',
                       choices=['darcy', 'burgers', 'navier_stokes', 'all'],
                       help='数据集选择')
    parser.add_argument('--generate-data', action='store_true',
                       help='生成数据后再测试')
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='数据目录')
    parser.add_argument('--output', type=str, default='full_benchmark_results.json',
                       help='输出文件')
    parser.add_argument('--n_train', type=int, default=500, help='训练集大小')
    parser.add_argument('--n_test', type=int, default=100, help='测试集大小')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='计算设备')
    
    args = parser.parse_args()
    
    config = {
        'n_train': args.n_train,
        'n_test': args.n_test,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'seed': args.seed,
        'device': args.device,
        'output_dir': args.data_dir,
    }
    
    print("="*70)
    print("MHF-FNO 完整三模型基准测试")
    print("="*70)
    print(f"配置: {json.dumps(config, indent=2)}")
    
    # 设置随机种子
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # 生成数据
    if args.generate_data:
        print("\n" + "="*70)
        print("生成数据集...")
        print("="*70)
        generate_all_data(config)
    
    # 运行测试
    all_results = {
        'config': config,
        'results': {},
        'timestamp': datetime.now().isoformat(),
    }
    
    datasets = ['darcy', 'burgers', 'navier_stokes'] if args.dataset == 'all' else [args.dataset]
    
    t0_total = time.time()
    
    for dataset in datasets:
        result = run_dataset_benchmark(dataset, config, args.data_dir)
        if result:
            all_results['results'][dataset] = result
    
    all_results['total_time'] = time.time() - t0_total
    
    # 保存结果
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else x)
    
    print(f"\n✅ 结果已保存到: {output_path}")
    
    # 生成汇总报告
    print(f"\n{'='*70}")
    print("汇总报告")
    print(f"{'='*70}")
    
    for dataset, result in all_results['results'].items():
        print(f"\n【{dataset.upper()}】分辨率: {result['resolution']}")
        fno = result['models']['FNO']
        mhf = result['models']['MHF-FNO']
        coda = result['models']['MHF+CoDA']
        
        print(f"  FNO:      参数 {fno['parameters']:>8,}, Loss {fno['best_test_loss']:.4f}")
        print(f"  MHF-FNO:  参数 {mhf['parameters']:>8,}, Loss {mhf['best_test_loss']:.4f}, 提升 {(fno['best_test_loss']-mhf['best_test_loss'])/fno['best_test_loss']*100:+.1f}%")
        print(f"  MHF+CoDA: 参数 {coda['parameters']:>8,}, Loss {coda['best_test_loss']:.4f}, 提升 {(fno['best_test_loss']-coda['best_test_loss'])/fno['best_test_loss']*100:+.1f}%")
    
    print(f"\n总用时: {all_results['total_time']:.1f}s")


if __name__ == '__main__':
    main()