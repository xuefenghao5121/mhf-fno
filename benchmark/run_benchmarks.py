#!/usr/bin/env python3
"""
MHF-FNO 多数据集基准测试

测试 MHF-FNO 在多个商用 benchmark 上的表现

使用方法:
    python run_benchmarks.py --dataset darcy
    python run_benchmarks.py --dataset burgers
    python run_benchmarks.py --dataset navier_stokes
    python run_benchmarks.py --dataset all

依赖:
    pip install neuralop torch numpy
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from neuralop.losses.data_losses import LpLoss
from neuralop.models import FNO


# ============================================================================
# MHF-FNO 核心实现
# ============================================================================

class MHFSpectralConv(nn.Module):
    """
    Multi-Head Fourier Spectral Convolution
    
    将标准的频域卷积分解为多个头，每个头处理不同的频率子空间
    """
    
    def __init__(self, in_channels, out_channels, n_modes, n_heads=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes = n_modes
        self.n_heads = n_heads
        
        # 每个头的通道数
        self.head_channels = out_channels // n_heads
        
        # 可学习的频域权重 (每个头独立)
        self.weight = nn.Parameter(
            torch.randn(n_heads, self.head_channels, self.head_channels, *n_modes)
        )
        
        # 线性变换
        self.fc = nn.Linear(in_channels, out_channels)
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """尺度多样性初始化"""
        with torch.no_grad():
            for h in range(self.n_heads):
                scale = 0.01 * (2 ** h)  # 不同头使用不同尺度
                nn.init.normal_(self.weight[h], mean=0, std=scale)
            nn.init.xavier_normal_(self.fc.weight)
            nn.init.zeros_(self.fc.bias)
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # FFT
        x_ft = torch.fft.rfftn(x, dim=(-2, -1))
        
        # 多头处理
        out_ft = torch.zeros(
            batch_size, self.out_channels, *x_ft.shape[-2:],
            dtype=x_ft.dtype, device=x.device
        )
        
        for h in range(self.n_heads):
            # 每个头处理一部分输出通道
            start = h * self.head_channels
            end = start + self.head_channels
            
            # 频域卷积 (只处理低频部分)
            for i in range(min(self.n_modes[0], x_ft.shape[-2])):
                for j in range(min(self.n_modes[1], x_ft.shape[-1])):
                    out_ft[:, start:end, i, j] = torch.einsum(
                        'bi,bio->bo',
                        x_ft[:, :, i, j],
                        self.weight[h, :, :, i, j]
                    )
        
        # IFFT
        x = torch.fft.irfftn(out_ft, dim=(-2, -1))
        
        # 线性变换
        x = self.fc(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        return x


class MHFFNO(nn.Module):
    """MHF-FNO 模型"""
    
    def __init__(self, n_modes, hidden_channels, in_channels, out_channels, 
                 n_layers=3, n_heads=4, mhf_layers=None):
        super().__init__()
        
        self.n_layers = n_layers
        self.hidden_channels = hidden_channels
        
        # 默认：第1层和最后一层使用 MHF
        if mhf_layers is None:
            mhf_layers = [0, n_layers - 1]
        self.mhf_layers = set(mhf_layers)
        
        # 输入投影
        self.fc_in = nn.Linear(in_channels, hidden_channels)
        
        # FNO 层
        self.fno_blocks = nn.ModuleList()
        for i in range(n_layers):
            use_mhf = i in self.mhf_layers
            
            block = nn.ModuleDict({
                'conv': MHFSpectralConv(
                    hidden_channels, hidden_channels, n_modes, n_heads
                ) if use_mhf else nn.Identity(),
                'w': nn.Conv2d(hidden_channels, hidden_channels, 1),
                'use_mhf': use_mhf,
            })
            self.fno_blocks.append(block)
        
        # 输出投影
        self.fc_out = nn.Linear(hidden_channels, out_channels)
    
    def forward(self, x):
        # 输入投影
        x = self.fc_in(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        # FNO 层
        for block in self.fno_blocks:
            if block['use_mhf']:
                x = x + block['conv'](x)
            x = x + block['w'](x)
        
        # 输出投影
        x = self.fc_out(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        return x


# ============================================================================
# 数据集加载
# ============================================================================

def load_darcy_flow(n_train=1000, n_test=200, resolution=16):
    """加载 Darcy Flow 数据集"""
    print(f"\n📊 加载 Darcy Flow ({resolution}x{resolution})...")
    
    try:
        import torch
        from pathlib import Path
        
        # 直接加载内置数据
        data_path = Path('/usr/local/lib/python3.11/site-packages/neuralop/data/datasets/data/')
        
        if not (data_path / 'darcy_train_16.pt').exists():
            # 如果没有内置数据，使用 API 下载
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
        else:
            # 使用内置数据
            train_data = torch.load(data_path / 'darcy_train_16.pt', weights_only=False)
            test_data = torch.load(data_path / 'darcy_test_16.pt', weights_only=False)
            
            train_x = train_data['x'].unsqueeze(1).float()  # 添加 channel 维度
            train_y = train_data['y'].unsqueeze(1).float()
            test_x = test_data['x'].unsqueeze(1).float()
            test_y = test_data['y'].unsqueeze(1).float()
            
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