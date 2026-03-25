#!/usr/bin/env python3
"""
MHF-FNO 多数据集基准测试

支持多种数据格式:
- PT 格式 (NeuralOperator 内置 或 本地生成)
- H5 格式 (PDEBench 数据集)

使用方法:
    # 使用内置数据
    python run_benchmarks.py --dataset darcy
    
    # 使用本地生成的数据 (自动解析分辨率)
    python run_benchmarks.py --dataset darcy --data_path ./data/darcy_train_16.pt
    
    # 指定分辨率
    python run_benchmarks.py --dataset darcy --resolution 32
    
    # H5 格式
    python run_benchmarks.py --dataset darcy --format h5 --data_path ./data/2D_DarcyFlow_Train.h5
    
    # Navier-Stokes with viscosity and steps
    python run_benchmarks.py --dataset navier_stokes --resolution 64 --viscosity 1e-3 --n_steps 100

依赖:
    pip install neuralop torch numpy h5py
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
import re

import numpy as np
import torch
import torch.nn as nn
from neuralop.losses.data_losses import LpLoss
from neuralop.models import FNO

# ============================================================================
# CPU 优化: 使用所有可用核心
# ============================================================================
torch.set_num_threads(os.cpu_count() or 1)

# H5 支持
try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


# ============================================================================
# 辅助函数: 从数据文件解析分辨率
# ============================================================================

def parse_resolution_from_filename(filename: str) -> int:
    """
    从文件名解析分辨率。
    
    支持的命名格式:
    - darcy_train_16.pt -> 16
    - ns_train_64.pt -> 64
    - burgers_train_1024.pt -> 1024
    - darcy_test_32_large.pt -> 32
    
    Args:
        filename: 文件名 (不含路径)
    
    Returns:
        int: 解析出的分辨率，如果无法解析返回默认值 16
    """
    # 尝试匹配常见模式: _数字.pt 或 _数字_
    match = re.search(r'_(\d+)(?:_|\.pt$)', filename)
    if match:
        return int(match.group(1))
    
    # 默认返回 16
    print(f"⚠️  无法从文件名 '{filename}' 解析分辨率，使用默认值 16")
    return 16


def parse_resolution_from_tensor(tensor: torch.Tensor) -> tuple:
    """
    从张量形状解析分辨率。
    
    Args:
        tensor: 数据张量 [N, C, H, W] 或 [N, C, L]
    
    Returns:
        tuple: (分辨率, 维度)
            - 2D: (H, '2d') 其中 H=W
            - 1D: (L, '1d')
    """
    if tensor.dim() == 4:  # [N, C, H, W]
        H, W = tensor.shape[-2], tensor.shape[-1]
        if H == W:
            return (H, '2d')
        return ((H, W), '2d')
    elif tensor.dim() == 3:  # [N, C, L]
        return (tensor.shape[-1], '1d')
    else:
        raise ValueError(f"不支持的张量维度: {tensor.dim()}")


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
# 数据集加载
# ============================================================================

# PDEBench 数据集下载地址
DATASET_URLS = {
    "darcy_2d": {
        "train": "https://darus.uni-stuttgart.de/api/access/datafile/152002",
        "test": "https://darus.uni-stuttgart.de/api/access/datafile/152003",
    },
    "navier_stokes_2d": {
        "train": "https://darus.uni-stuttgart.de/api/access/datafile/151996",
        "test": "https://darus.uni-stuttgart.de/api/access/datafile/151997",
    },
    "burgers_1d": {
        "data": "https://darus.uni-stuttgart.de/api/access/datafile/151990",
    },
}


def load_h5_darcy(h5_path, n_train=1000, n_test=200, resolution=None):
    """从 H5 文件加载 Darcy Flow 数据"""
    if not HAS_H5PY:
        raise ImportError("需要安装 h5py: pip install h5py")
    
    print(f"\n📊 从 H5 加载 Darcy Flow: {h5_path}")
    
    with h5py.File(h5_path, 'r') as f:
        # PDEBench 格式
        if 'tensor' in f:
            data = f['tensor'][:]
        elif 'data' in f:
            data = f['data'][:]
        else:
            # 尝试其他常见键
            keys = list(f.keys())
            data = f[keys[0]][:]
        
        # Darcy Flow 通常是 [N, H, W] 或 [N, 1, H, W]
        if data.ndim == 3:
            data = data[:, np.newaxis, :, :]  # 添加 channel 维度
        
        # 分割输入输出 (假设前一半是输入，后一半是输出)
        n_samples = data.shape[0]
        split = n_samples // 2
        
        train_x = torch.from_numpy(data[:n_train]).float()
        train_y = torch.from_numpy(data[split:split+n_train]).float()
        test_x = torch.from_numpy(data[n_train:n_train+n_test]).float()
        test_y = torch.from_numpy(data[split+n_train:split+n_train+n_test]).float()
        
        # 下采样
        if resolution and train_x.shape[-1] != resolution:
            train_x = torch.nn.functional.interpolate(train_x, size=(resolution, resolution), mode='bilinear')
            train_y = torch.nn.functional.interpolate(train_y, size=(resolution, resolution), mode='bilinear')
            test_x = torch.nn.functional.interpolate(test_x, size=(resolution, resolution), mode='bilinear')
            test_y = torch.nn.functional.interpolate(test_y, size=(resolution, resolution), mode='bilinear')
    
    info = {
        'name': 'Darcy Flow (H5)',
        'resolution': f'{train_x.shape[-1]}x{train_x.shape[-2]}',
        'n_train': train_x.shape[0],
        'n_test': test_x.shape[0],
        'input_channels': train_x.shape[1],
        'output_channels': train_y.shape[1],
        'n_modes': (train_x.shape[-1] // 2, train_x.shape[-2] // 2),
    }
    
    print(f"✅ 加载成功: 训练 {train_x.shape[0]}, 测试 {test_x.shape[0]}")
    return train_x, train_y, test_x, test_y, info


def load_h5_navier_stokes(h5_path, n_train=1000, n_test=200, resolution=None):
    """从 H5 文件加载 Navier-Stokes 数据"""
    if not HAS_H5PY:
        raise ImportError("需要安装 h5py: pip install h5py")
    
    print(f"\n📊 从 H5 加载 Navier-Stokes: {h5_path}")
    
    with h5py.File(h5_path, 'r') as f:
        # Navier-Stokes 通常是 [N, T, H, W] 或 [N, H, W]
        if 'tensor' in f:
            data = f['tensor'][:]
        elif 'data' in f:
            data = f['data'][:]
        else:
            keys = list(f.keys())
            data = f[keys[0]][:]
        
        # 如果有时间维度，取最后时刻
        if data.ndim == 4:
            data = data[:, -1, :, :]  # 取最后时刻
        
        if data.ndim == 3:
            data = data[:, np.newaxis, :, :]
        
        # 分割输入输出
        n_samples = data.shape[0]
        split = n_samples // 2
        
        train_x = torch.from_numpy(data[:n_train]).float()
        train_y = torch.from_numpy(data[split:split+n_train]).float()
        test_x = torch.from_numpy(data[n_train:n_train+n_test]).float()
        test_y = torch.from_numpy(data[split+n_train:split+n_train+n_test]).float()
        
        if resolution and train_x.shape[-1] != resolution:
            train_x = torch.nn.functional.interpolate(train_x, size=(resolution, resolution), mode='bilinear')
            train_y = torch.nn.functional.interpolate(train_y, size=(resolution, resolution), mode='bilinear')
            test_x = torch.nn.functional.interpolate(test_x, size=(resolution, resolution), mode='bilinear')
            test_y = torch.nn.functional.interpolate(test_y, size=(resolution, resolution), mode='bilinear')
    
    info = {
        'name': 'Navier-Stokes (H5)',
        'resolution': f'{train_x.shape[-1]}x{train_x.shape[-2]}',
        'n_train': train_x.shape[0],
        'n_test': test_x.shape[0],
        'input_channels': train_x.shape[1],
        'output_channels': train_y.shape[1],
        'n_modes': (train_x.shape[-1] // 4, train_x.shape[-2] // 4),
    }
    
    print(f"✅ 加载成功: 训练 {train_x.shape[0]}, 测试 {test_x.shape[0]}")
    return train_x, train_y, test_x, test_y, info


def load_darcy_flow(n_train=1000, n_test=200, resolution=None, data_format='pt', data_path=None):
    """
    加载 Darcy Flow 数据集
    
    支持三种模式:
    1. 本地 PT 文件: data_path 指向具体的 .pt 文件
    2. 内置数据: data_path 为 None，使用 NeuralOperator 内置数据
    3. H5 文件: data_format='h5'
    
    Args:
        n_train: 训练样本数
        n_test: 测试样本数
        resolution: 空间分辨率 (如果为 None，从文件名或张量形状推断)
        data_format: 数据格式 ('pt' 或 'h5')
        data_path: 数据文件路径
    """
    # H5 格式
    if data_format == 'h5' and data_path:
        return load_h5_darcy(data_path, n_train, n_test, resolution)
    
    # PT 格式 (本地文件优先)
    try:
        from pathlib import Path
        
        # 情况 1: 指定了本地 PT 文件
        if data_path and Path(data_path).exists():
            print(f"\n📊 从本地文件加载 Darcy Flow: {data_path}")
            
            # 解析分辨率
            if resolution is None:
                resolution = parse_resolution_from_filename(Path(data_path).name)
                print(f"   从文件名解析分辨率: {resolution}")
            
            # 加载数据
            data = torch.load(data_path, weights_only=False)
            
            # 处理数据格式
            if isinstance(data, dict):
                train_x = data.get('x', data.get('train_x'))
                train_y = data.get('y', data.get('train_y'))
            else:
                # 假设是元组或列表
                train_x, train_y = data[0], data[1]
            
            # 确保维度正确
            if train_x.dim() == 3:  # [N, H, W]
                train_x = train_x.unsqueeze(1)  # [N, 1, H, W]
                train_y = train_y.unsqueeze(1)
            
            # 转换为 float
            train_x = train_x.float()
            train_y = train_y.float()
            
            # 确定测试文件路径
            test_path = data_path.replace('train', 'test')
            if Path(test_path).exists():
                test_data = torch.load(test_path, weights_only=False)
                if isinstance(test_data, dict):
                    test_x = test_data.get('x', test_data.get('test_x'))
                    test_y = test_data.get('y', test_data.get('test_y'))
                else:
                    test_x, test_y = test_data[0], test_data[1]
                
                if test_x.dim() == 3:
                    test_x = test_x.unsqueeze(1)
                    test_y = test_y.unsqueeze(1)
                test_x = test_x.float()
                test_y = test_y.float()
            else:
                # 从训练数据分割
                print(f"   测试文件不存在，从训练数据分割")
                split_idx = int(len(train_x) * 0.8)
                test_x = train_x[split_idx:split_idx+n_test]
                test_y = train_y[split_idx:split_idx+n_test]
                train_x = train_x[:n_train]
                train_y = train_y[:n_train]
            
            # 限制样本数
            train_x = train_x[:n_train]
            train_y = train_y[:n_train]
            test_x = test_x[:n_test]
            test_y = test_y[:n_test]
            
            # 更新分辨率
            actual_resolution = train_x.shape[-1]
            if resolution != actual_resolution:
                print(f"   实际分辨率: {actual_resolution}x{actual_resolution}")
                resolution = actual_resolution
            
            info = {
                'name': 'Darcy Flow (本地)',
                'resolution': f'{resolution}x{resolution}',
                'n_train': train_x.shape[0],
                'n_test': test_x.shape[0],
                'input_channels': train_x.shape[1],
                'output_channels': train_y.shape[1],
                'n_modes': (resolution // 2, resolution // 2),
            }
            
            print(f"✅ 加载成功: 训练 {train_x.shape[0]}, 测试 {test_x.shape[0]}")
            return train_x, train_y, test_x, test_y, info
        
        # 情况 2: 使用 NeuralOperator 内置数据
        if resolution is None:
            resolution = 16  # 默认分辨率
        
        print(f"\n📊 加载 Darcy Flow ({resolution}x{resolution}) [NeuralOperator 内置]...")
        
        # 尝试加载内置数据
        builtin_path = Path('/usr/local/lib/python3.11/site-packages/neuralop/data/datasets/data/')
        
        if (builtin_path / f'darcy_train_{resolution}.pt').exists():
            # 使用内置数据
            train_data = torch.load(builtin_path / f'darcy_train_{resolution}.pt', weights_only=False)
            test_data = torch.load(builtin_path / f'darcy_test_{resolution}.pt', weights_only=False)
            
            train_x = train_data['x'].unsqueeze(1).float()
            train_y = train_data['y'].unsqueeze(1).float()
            test_x = test_data['x'].unsqueeze(1).float()
            test_y = test_data['y'].unsqueeze(1).float()
        else:
            # 使用 API 下载
            from neuralop.data.datasets import load_darcy_flow_small
            train_loader, test_loader, _ = load_darcy_flow_small(
                n_train=n_train,
                n_tests=[n_test],
                batch_size=32,
                test_batch_sizes=[32],
            )
            train_batch = next(iter(train_loader))
            test_batch = next(iter(test_loader))
            train_x = train_batch['x']
            train_y = train_batch['y']
            test_x = test_batch['x']
            test_y = test_batch['y']
        
        # 限制样本数
        train_x = train_x[:n_train]
        train_y = train_y[:n_train]
        test_x = test_x[:n_test]
        test_y = test_y[:n_test]
        
        info = {
            'name': 'Darcy Flow',
            'resolution': f'{resolution}x{resolution}',
            'n_train': train_x.shape[0],
            'n_test': test_x.shape[0],
            'input_channels': train_x.shape[1],
            'output_channels': train_y.shape[1],
            'n_modes': (resolution // 2, resolution // 2),
        }
        
        print(f"✅ 加载成功: 训练 {train_x.shape[0]}, 测试 {test_x.shape[0]}")
        return train_x, train_y, test_x, test_y, info
        
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        print("请确保已安装 neuralop: pip install neuralop")
        return None


def load_burgers(n_train=1000, n_test=200):
    """加载 Burgers 方程数据集"""
    print(f"\n📊 加载 Burgers 方程...")
    
    try:
        from neuralop.data.datasets import load_mini_burgers_1dtime
        from pathlib import Path
        
        # 数据下载路径
        data_path = Path.home() / '.neuralop' / 'data'
        data_path.mkdir(parents=True, exist_ok=True)
        
        train_loader, test_loader, _ = load_mini_burgers_1dtime(
            data_path=data_path,
            n_train=n_train,
            n_test=n_test,
            batch_size=32,
            test_batch_size=32,
        )
        
        train_batch = next(iter(train_loader))
        test_batch = next(iter(test_loader))
        
        train_x = train_batch['x']
        train_y = train_batch['y']
        test_x = test_batch['x']
        test_y = test_batch['y']
        
        # Burgers 是 1D+time，需要调整
        resolution = train_x.shape[-1]
        
        info = {
            'name': 'Burgers Equation',
            'resolution': f'{resolution}',
            'n_train': train_x.shape[0],
            'n_test': test_x.shape[0],
            'input_channels': train_x.shape[1],
            'output_channels': train_y.shape[1],
            'n_modes': (resolution // 2,),
        }
        
        print(f"✅ 加载成功: 训练 {train_x.shape[0]}, 测试 {test_x.shape[0]}")
        return train_x, train_y, test_x, test_y, info
        
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return None


def load_navier_stokes(n_train=1000, n_test=200, resolution=64):
    """加载 Navier-Stokes 数据集"""
    print(f"\n📊 加载 Navier-Stokes ({resolution}x{resolution})...")
    print("⚠️  注意: 数据下载可能需要较长时间 (~1.5GB)")
    
    try:
        from neuralop.data.datasets import load_navier_stokes_pt
        
        train_loader, test_loaders, _ = load_navier_stokes_pt(
            n_train=n_train,
            n_tests=[n_test],
            batch_size=32,
            test_batch_sizes=[32],
            train_resolution=resolution,
            test_resolutions=[resolution],
        )
        
        test_loader = test_loaders[resolution]
        
        train_batch = next(iter(train_loader))
        test_batch = next(iter(test_loader))
        
        train_x = train_batch['x']
        train_y = train_batch['y']
        test_x = test_batch['x']
        test_y = test_batch['y']
        
        info = {
            'name': 'Navier-Stokes',
            'resolution': f'{resolution}x{resolution}',
            'n_train': train_x.shape[0],
            'n_test': test_x.shape[0],
            'input_channels': train_x.shape[1],
            'output_channels': train_y.shape[1],
            'n_modes': (resolution // 4, resolution // 4),
        }
        
        print(f"✅ 加载成功: 训练 {train_x.shape[0]}, 测试 {test_x.shape[0]}")
        return train_x, train_y, test_x, test_y, info
        
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return None


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
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{config['epochs']}: "
                  f"Train {train_loss/batch_count:.4f}, "
                  f"Test {test_loss:.4f}, "
                  f"Time {epoch_time:.1f}s")
    
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

def run_benchmark(dataset_name, config):
    """运行单个数据集的基准测试"""
    
    # 加载数据
    if dataset_name == 'darcy':
        data = load_darcy_flow(
            n_train=config['n_train'],
            n_test=config['n_test'],
            resolution=config.get('resolution'),
            data_format=config.get('data_format', 'pt'),
            data_path=config.get('data_path'),
        )
    elif dataset_name == 'burgers':
        data = load_burgers(
            n_train=config['n_train'],
            n_test=config['n_test'],
        )
    elif dataset_name == 'navier_stokes':
        data = load_navier_stokes(
            n_train=config['n_train'],
            n_test=config['n_test'],
            resolution=config.get('resolution', 64),
        )
    else:
        print(f"❌ 未知数据集: {dataset_name}")
        return None
    
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
    
    torch.manual_seed(config['seed'])
    model_mhf = MHFFNO(
        n_modes=n_modes,
        hidden_channels=hidden_channels,
        in_channels=info['input_channels'],
        out_channels=info['output_channels'],
        n_layers=3,
        n_heads=4,
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
                       help='数据格式 (pt=NeuralOperator内置或本地文件, h5=PDEBench)')
    parser.add_argument('--data_path', type=str, default=None,
                       help='数据文件路径 (支持 PT 和 H5 格式)')
    # 新增参数: 分辨率、粘度、时间步数
    parser.add_argument('--resolution', type=int, default=None,
                       help='空间分辨率 (Darcy/Navier-Stokes), 默认从文件推断或使用 16')
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
        'resolution': args.resolution,
        'viscosity': args.viscosity,
        'n_steps': args.n_steps,
    }
    
    print("="*60)
    print("MHF-FNO 基准测试")
    print("="*60)
    print(f"配置: {config}")
    
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