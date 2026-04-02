#!/usr/bin/env python3
"""
MHF-FNO 预训练脚本 (v1.6.4 - CPU性能优化版)

基于本地数据集训练预训练模型，导出推理就绪的 .pth 文件。

支持的数据集:
- Darcy Flow 32x32 (默认)
- Navier-Stokes 128x128
- Burgers 1D
- 自定义 PT/H5 数据集

性能优化:
- 优化数据加载并行策略
- CPU线程绑定提升缓存利用率
- 可配置OpenMP/MKL线程数
- 预加载到内存减少IO阻塞

使用方法:
    # Darcy Flow (默认)
    python train_pretrained.py

    # Navier-Stokes
    python train_pretrained.py --dataset navier_stokes

    # 自定义数据集
    python train_pretrained.py --dataset custom \
        --train_path ./data/train.pt --test_path ./data/test.pt

    # 快速验证 (少量 epoch)
    python train_pretrained.py --epochs 5 --batch_size 32

    # 多核CPU优化（推荐128核系统）
    python train_pretrained.py --num_workers 8 --omp_num_threads 16 --bind_cpu --pin_memory
"""

import argparse
import json
import time
import sys
import os
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 性能优化：设置CPU线程绑定
def bind_cpu_to_core(core_id: int):
    """绑定当前进程到指定CPU核心"""
    try:
        import psutil
        p = psutil.Process()
        p.cpu_affinity([core_id])
        return True
    except ImportError:
        return False
    except Exception:
        return False

def set_cpu_affinity(start_core: int, num_threads: int):
    """设置CPU亲和性，绑定进程到连续的核心范围"""
    try:
        import psutil
        p = psutil.Process()
        cores = list(range(start_core, start_core + num_threads))
        p.cpu_affinity(cores)
        return cores
    except ImportError:
        return None
    except Exception as e:
        print(f"⚠️  CPU绑定失败: {e}")
        return None

# 环境变量配置：提前设置OpenMP/MKL线程数
def configure_openmp_threads(num_threads: int):
    """配置OpenMP/MKL线程数，需要在导入torch之前调用"""
    if num_threads <= 0:
        return
    
    os.environ['OMP_NUM_THREADS'] = str(num_threads)
    os.environ['MKL_NUM_THREADS'] = str(num_threads)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(num_threads)
    os.environ['NUMEXPR_NUM_THREADS'] = str(num_threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(num_threads)
    
    # 设置FFT规划缓存，提升重复FFT性能
    os.environ['FFT_CACHE_SIZE'] = '1048576'

# 添加项目根目录
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from mhf_fno import MHFFNOWithAttention, MHFFNO, get_device, PINOLoss

# 本地数据路径
DATA_PATH = Path("/home/huawei/Desktop/home/xuefenghao/workspace/mhf-data")


# ============================================================================
# 数据加载
# ============================================================================

def load_darcy_data(n_train=None, n_test=None):
    """加载 Darcy Flow 32x32 数据"""
    train_path = DATA_PATH / "darcy_train_32.pt"
    test_path = DATA_PATH / "darcy_test_32.pt"
    
    if not train_path.exists():
        raise FileNotFoundError(f"Darcy 数据不存在: {train_path}")
    
    train_data = torch.load(train_path, map_location='cpu', weights_only=False)
    test_data = torch.load(test_path, map_location='cpu', weights_only=False)
    
    def extract(data):
        if isinstance(data, dict):
            x = data.get('x', data.get('input'))
            y = data.get('y', data.get('output', x))
        elif isinstance(data, (tuple, list)):
            x, y = data[0], data[1]
        else:
            x, y = data, data
        return x.float(), y.float()
    
    train_x, train_y = extract(train_data)
    test_x, test_y = extract(test_data)
    
    # 确保维度 [N, C, H, W]
    if train_x.dim() == 3:
        train_x, train_y = train_x.unsqueeze(1), train_y.unsqueeze(1)
    if test_x.dim() == 3:
        test_x, test_y = test_x.unsqueeze(1), test_y.unsqueeze(1)
    
    if n_train:
        train_x, train_y = train_x[:n_train], train_y[:n_train]
    if n_test:
        test_x, test_y = test_x[:n_test], test_y[:n_test]
    
    return train_x, train_y, test_x, test_y


def load_ns_data(n_train=None, n_test=None):
    """加载 Navier-Stokes 128x128 数据"""
    train_path = DATA_PATH / "nsforcing_train_128.pt"
    test_path = DATA_PATH / "nsforcing_test_128.pt"
    
    if not train_path.exists():
        raise FileNotFoundError(f"NS 数据不存在: {train_path}")
    
    train_data = torch.load(train_path, map_location='cpu', weights_only=False)
    test_data = torch.load(test_path, map_location='cpu', weights_only=False)
    
    def extract(data):
        if isinstance(data, dict):
            x = data.get('x', data.get('input'))
            y = data.get('y', data.get('output', x))
        elif isinstance(data, (tuple, list)):
            x, y = data[0], data[1]
        else:
            x, y = data, data
        return x.float(), y.float()
    
    train_x, train_y = extract(train_data)
    test_x, test_y = extract(test_data)
    
    # NS 数据可能是 [N, T, H, W] 或 [N, C, H, W]
    if train_x.dim() == 4 and train_x.shape[1] > 4:
        # 时间序列: 取第一个和最后一个时间步
        train_x, train_y = train_x[:, 0:1], train_x[:, -1:]
        test_x, test_y = test_x[:, 0:1], test_x[:, -1:]
    elif train_x.dim() == 3:
        train_x, train_y = train_x.unsqueeze(1), train_y.unsqueeze(1)
        test_x, test_y = test_x.unsqueeze(1), test_y.unsqueeze(1)
    
    if n_train:
        train_x, train_y = train_x[:n_train], train_y[:n_train]
    if n_test:
        test_x, test_y = test_x[:n_test], test_y[:n_test]
    
    return train_x, train_y, test_x, test_y


def load_burgers_data(n_train=None, n_test=None):
    """加载 Burgers 数据"""
    burgers_path = DATA_PATH / "burgers"
    if not burgers_path.exists():
        # 尝试根目录下的 burgers 文件
        for name in ['rand_burgers_data_R10.pt', 'uniform_burgers_data_R10.pt']:
            p = DATA_PATH / name
            if p.exists():
                data = torch.load(p, map_location='cpu', weights_only=False)
                if isinstance(data, dict):
                    x, y = data['x'].float(), data['y'].float()
                elif isinstance(data, (tuple, list)):
                    x, y = data[0].float(), data[1].float()
                else:
                    x, y = data.float(), data.float()
                if x.dim() == 2:
                    x, y = x.unsqueeze(1), y.unsqueeze(1)
                n = len(x)
                split = int(0.8 * n)
                if n_train:
                    split = n_train
                if n_test:
                    end = split + n_test
                else:
                    end = n
                return x[:split], y[:split], x[split:end], y[split:end]
        raise FileNotFoundError(f"Burgers 数据不存在: {burgers_path}")
    raise FileNotFoundError(f"Burgers 数据不存在: {burgers_path}")


def load_custom_data(train_path, test_path, n_train=None, n_test=None):
    """加载自定义 PT 数据"""
    train_data = torch.load(train_path, map_location='cpu', weights_only=False)
    test_data = torch.load(test_path, map_location='cpu', weights_only=False)
    
    def extract(data):
        if isinstance(data, dict):
            x = data.get('x', data.get('input'))
            y = data.get('y', data.get('output', x))
        elif isinstance(data, (tuple, list)):
            x, y = data[0], data[1]
        else:
            x, y = data, data
        return x.float(), y.float()
    
    train_x, train_y = extract(train_data)
    test_x, test_y = extract(test_data)
    
    # 自动添加通道维度
    for t in [train_x, train_y, test_x, test_y]:
        if t.dim() == 2:
            t.unsqueeze_(1)
    
    if n_train:
        train_x, train_y = train_x[:n_train], train_y[:n_train]
    if n_test:
        test_x, test_y = test_x[:n_test], test_y[:n_test]
    
    return train_x, train_y, test_x, test_y


# ============================================================================
# 损失函数
# ============================================================================

class LpLoss(nn.Module):
    """相对 Lp 损失"""
    def __init__(self, d=2, p=2):
        super().__init__()
        self.d = d
        self.p = p
    
    def forward(self, pred, target):
        diff = pred - target
        loss = torch.norm(diff.reshape(diff.shape[0], -1), p=self.p, dim=1) / \
               torch.norm(target.reshape(target.shape[0], -1), p=self.p, dim=1)
        return loss.mean()


# ============================================================================
# 训练
# ============================================================================

def train(args):
    device = get_device()
    print(f"🔧 设备: {device}")
    
    # 加载数据
    print(f"\n📊 加载数据集: {args.dataset}")
    if args.dataset == 'darcy':
        train_x, train_y, test_x, test_y = load_darcy_data(args.n_train, args.n_test)
        model_kwargs = {
            'n_modes': (16, 16),
            'hidden_channels': 32,
            'in_channels': 1, 'out_channels': 1,
            'n_heads': 2,
        }
        config = {**model_kwargs, 'mhf_layers': [0, 2], 'attention_layers': [0, 2]}
    elif args.dataset == 'navier_stokes':
        train_x, train_y, test_x, test_y = load_ns_data(args.n_train, args.n_test)
        model_kwargs = {
            'n_modes': (32, 32),
            'hidden_channels': 64,
            'in_channels': train_x.shape[1], 'out_channels': train_y.shape[1],
            'n_heads': 4,
        }
        config = {**model_kwargs, 'mhf_layers': [0, 2], 'attention_layers': [0, 2]}
    elif args.dataset == 'burgers':
        train_x, train_y, test_x, test_y = load_burgers_data(args.n_train, args.n_test)
        model_kwargs = {
            'n_modes': (16,),
            'hidden_channels': 32,
            'in_channels': 1, 'out_channels': 1,
            'n_heads': 2,
            'positional_embedding': None,
        }
        config = {**model_kwargs, 'mhf_layers': [0, 2], 'attention_layers': [0, 2]}
    elif args.dataset == 'custom':
        train_x, train_y, test_x, test_y = load_custom_data(
            args.train_path, args.test_path, args.n_train, args.n_test)
        spatial_dims = train_x.shape[-2:]
        modes = tuple(min(s // 2, 32) for s in spatial_dims) if len(spatial_dims) == 2 else (min(spatial_dims[0] // 2, 32),)
        model_kwargs = {
            'n_modes': modes,
            'hidden_channels': args.hidden_channels,
            'in_channels': train_x.shape[1], 'out_channels': train_y.shape[1],
            'n_heads': args.n_heads,
        }
        config = {**model_kwargs, 'mhf_layers': [0, 2], 'attention_layers': [0, 2]}
    else:
        raise ValueError(f"不支持的数据集: {args.dataset}")
    
    print(f"   训练集: {train_x.shape} -> {train_y.shape}")
    print(f"   测试集: {test_x.shape} -> {test_y.shape}")
    
    # 创建模型
    print(f"\n🤖 创建 MHF-FNO+CODA 模型...")
    model = MHFFNOWithAttention.best_config(**model_kwargs)
    model = model.to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   参数量: {n_params:,}")
    
    # DataLoader
    train_dataset = TensorDataset(train_x, train_y)
    
    # 优化的数据加载配置
    loader_kwargs = {
        'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': args.num_workers,
    }
    
    # 可选：固定内存和预取优化
    if hasattr(args, 'pin_memory'):
        loader_kwargs['pin_memory'] = args.pin_memory
    if hasattr(args, 'prefetch_factor') and args.num_workers > 0:
        loader_kwargs['prefetch_factor'] = args.prefetch_factor
        loader_kwargs['persistent_workers'] = True
    
    train_loader = DataLoader(train_dataset, **loader_kwargs)
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = LpLoss(d=2 if train_x.dim() == 4 else 1, p=2)
    
    # 训练循环
    print(f"\n🚀 开始训练 ({args.epochs} epochs)...")
    best_loss = float('inf')
    history = {'train_loss': [], 'test_loss': [], 'lr': []}
    
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        n_batches = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        
        # 测试
        model.eval()
        with torch.no_grad():
            test_x_dev, test_y_dev = test_x.to(device), test_y.to(device)
            # 分批测试避免 OOM
            test_preds = []
            for i in range(0, len(test_x_dev), args.batch_size):
                bx = test_x_dev[i:i+args.batch_size]
                test_preds.append(model(bx).cpu())
            test_pred = torch.cat(test_preds, dim=0)
            test_loss = loss_fn(test_pred, test_y).item()
        
        history['train_loss'].append(avg_loss)
        history['test_loss'].append(test_loss)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        if test_loss < best_loss:
            best_loss = test_loss
        
        if (epoch + 1) % args.log_interval == 0 or epoch == 0 or epoch == args.epochs - 1:
            elapsed = time.time() - t0
            print(f"   Epoch {epoch+1:3d}/{args.epochs} | "
                  f"Train: {avg_loss:.6f} | Test: {test_loss:.6f} | "
                  f"Best: {best_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e} | "
                  f"Time: {elapsed:.1f}s")
    
    elapsed = time.time() - t0
    print(f"\n✅ 训练完成! 总用时: {elapsed:.1f}s")
    print(f"   最佳测试损失: {best_loss:.6f}")
    
    # 导出模型
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = output_dir / f"mhf_fno_{args.dataset}_pretrained.pth"
    
    # 保存完整预训练包 (config 用于重建模型，model_kwargs 用于 best_config API)
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,  # 完整配置 (含 mhf_layers 等)
        'model_kwargs': model_kwargs,  # best_config 兼容参数
        'dataset': args.dataset,
        'version': '1.6.4',
        'train_history': history,
        'best_test_loss': best_loss,
        'total_epochs': args.epochs,
        'timestamp': datetime.now().isoformat(),
        'input_shape': list(train_x.shape[1:]),
        'output_shape': list(train_y.shape[1:]),
    }, model_path)
    
    print(f"💾 预训练模型已保存: {model_path}")
    print(f"   文件大小: {model_path.stat().st_size / 1024:.1f} KB")
    
    # 保存训练历史
    history_path = output_dir / f"mhf_fno_{args.dataset}_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    return model_path


def main():
    parser = argparse.ArgumentParser(description='MHF-FNO 预训练')
    parser.add_argument('--dataset', type=str, default='darcy',
                        choices=['darcy', 'navier_stokes', 'burgers', 'custom'])
    parser.add_argument('--train_path', type=str, default=None, help='自定义数据集训练路径')
    parser.add_argument('--test_path', type=str, default=None, help='自定义数据集测试路径')
    parser.add_argument('--n_train', type=int, default=None, help='训练样本数')
    parser.add_argument('--n_test', type=int, default=None, help='测试样本数')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=64, help='批大小')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--hidden_channels', type=int, default=32, help='隐藏通道数')
    parser.add_argument('--n_heads', type=int, default=4, help='MHF 头数')
    parser.add_argument('--output_dir', type=str, default=str(ROOT / 'pretrained' / 'models'))
    parser.add_argument('--log_interval', type=int, default=5, help='日志间隔')
    
    # === 性能优化参数 ===
    parser.add_argument('--num_workers', type=int, default=0, 
                        help='数据加载worker数量 (推荐: CPU核数/4 到 CPU核数/2)')
    parser.add_argument('--omp_num_threads', type=int, default=0,
                        help='OpenMP/MKL线程数 (设置为物理核数，避免超线程竞争)')
    parser.add_argument('--bind_cpu', action='store_true',
                        help='启用CPU线程绑定，提升缓存利用率')
    parser.add_argument('--pin_memory', action='store_true',
                        help='在DataLoader中固定内存（仅对CPU训练影响不大）')
    parser.add_argument('--prefetch_factor', type=int, default=2,
                        help='数据预取因子 (仅当num_workers > 0时生效)')
    parser.add_argument('--start_core', type=int, default=0,
                        help='CPU绑定起始核心ID，用于多进程训练')
    
    args = parser.parse_args()
    
    # === 应用性能优化配置 ===
    if args.omp_num_threads > 0:
        configure_openmp_threads(args.omp_num_threads)
        torch.set_num_threads(args.omp_num_threads)
        torch.set_num_interop_threads(args.omp_num_threads)
        print(f"⚙️  配置OpenMP/MKL线程数: {args.omp_num_threads}")
    
    if args.bind_cpu and args.omp_num_threads > 0:
        cores = set_cpu_affinity(args.start_core, args.omp_num_threads)
        if cores:
            print(f"⚙️  CPU线程绑定完成，核心范围: {cores[0]} - {cores[-1]}")
    
    train(args)


if __name__ == '__main__':
    main()
