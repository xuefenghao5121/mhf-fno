#!/usr/bin/env python3
"""
MHF-FNO 性能对比测试

对比三种模型:
1. FNO (基线)
2. MHF-FNO (使用 MHF 替换 SpectralConv)
3. MHF-FNO + CoDA (添加注意力机制)
"""

import torch
import torch.nn as nn
import time
from pathlib import Path
import argparse
import json

from neuralop.models import FNO
from neuralop.losses import LpLoss


def load_pt_dataset(train_path: str, test_path: str, device: str = 'cpu'):
    """
    加载 PT 格式数据集
    """
    train_data = torch.load(train_path)
    test_data = torch.load(test_path)
    
    return {
        'train_x': train_data['x'].to(device),
        'train_y': train_data['y'].to(device),
        'test_x': test_data['x'].to(device),
        'test_y': test_data['y'].to(device),
    }


def train_model(
    model: nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    epochs: int,
    batch_size: int,
    lr: float,
    loss_fn
):
    """
    训练模型
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    
    n_samples = train_x.shape[0]
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        
        # 随机打乱
        indices = torch.randperm(n_samples)
        
        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_x = train_x[batch_indices]
            batch_y = train_y[batch_indices]
            
            optimizer.zero_grad()
            
            out = model(batch_x)
            loss = loss_fn(out, batch_y)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        
        print(f"    Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.6f}")
    
    return losses


def test_model(
    model: nn.Module,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    loss_fn
):
    """
    测试模型
    """
    model.eval()
    with torch.no_grad():
        out = model(test_x)
        test_loss = loss_fn(out, test_y)
    
    return test_loss.item()


def create_fno(
    in_channels: int,
    out_channels: int,
    n_modes: tuple,
    hidden_channels: int,
    n_layers: int,
    device: str
) -> nn.Module:
    """
    创建 FNO 模型
    """
    return FNO(
        n_modes=n_modes,
        hidden_channels=hidden_channels,
        in_channels=in_channels,
        out_channels=out_channels,
        n_layers=n_layers
    ).to(device)


def run_comparison(
    dataset_name: str,
    train_path: str,
    test_path: str,
    n_modes: tuple,
    hidden_channels: int = 32,
    n_layers: int = 3,
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 0.001,
    device: str = 'cpu'
) -> dict:
    """
    运行对比测试
    """
    print("=" * 80)
    print(f"数据集: {dataset_name}")
    print("=" * 80)
    
    # 加载数据
    print(f"\n加载数据集...")
    data = load_pt_dataset(train_path, test_path, device)
    train_x, train_y, test_x, test_y = data['train_x'], data['train_y'], data['test_x'], data['test_y']
    
    print(f"  训练集: {train_x.shape}")
    print(f"  测试集: {test_x.shape}")
    
    # 损失函数
    loss_fn = LpLoss(d=2 if train_x.ndim == 4 else 1)
    
    results = {}
    
    # ========== 测试 1: FNO (基线) ==========
    print("\n" + "=" * 80)
    print("测试 1: FNO (基线)")
    print("=" * 80)
    
    fno_model = create_fno(
        in_channels=train_x.shape[1],
        out_channels=train_y.shape[1],
        n_modes=n_modes,
        hidden_channels=hidden_channels,
        n_layers=n_layers,
        device=device
    )
    
    n_params_fno = sum(p.numel() for p in fno_model.parameters())
    print(f"  参数量: {n_params_fno:,}")
    
    start_time = time.time()
    train_losses_fno = train_model(
        fno_model, train_x, train_y,
        epochs=epochs, batch_size=batch_size, lr=lr, loss_fn=loss_fn
    )
    train_time_fno = time.time() - start_time
    
    test_loss_fno = test_model(fno_model, test_x, test_y, loss_fn)
    
    results['fno'] = {
        'model': 'FNO',
        'parameters': n_params_fno,
        'train_losses': train_losses_fno,
        'best_train_loss': min(train_losses_fno),
        'final_train_loss': train_losses_fno[-1],
        'test_loss': test_loss_fno,
        'train_time': train_time_fno,
    }
    
    print(f"\n  最终训练 Loss: {train_losses_fno[-1]:.6f}")
    print(f"  测试 Loss: {test_loss_fno:.6f}")
    print(f"  训练时间: {train_time_fno:.2f}s")
    
    # ========== 结果汇总 ==========
    print("\n" + "=" * 80)
    print("结果汇总")
    print("=" * 80)
    
    for model_name, result in results.items():
        print(f"\n{result['model']}:")
        print(f"  参数量: {result['parameters']:,}")
        print(f"  最终训练 Loss: {result['final_train_loss']:.6f}")
        print(f"  测试 Loss: {result['test_loss']:.6f}")
        print(f"  训练时间: {result['train_time']:.2f}s")
    
    return {
        'dataset': dataset_name,
        'results': results,
        'config': {
            'n_modes': n_modes,
            'hidden_channels': hidden_channels,
            'n_layers': n_layers,
            'epochs': epochs,
            'batch_size': batch_size,
            'lr': lr,
            'device': device,
        }
    }


def main():
    parser = argparse.ArgumentParser(description='MHF-FNO 性能对比测试')
    parser.add_argument('--dataset', type=str, required=True, help='数据集名称')
    parser.add_argument('--train_path', type=str, required=True, help='训练数据路径')
    parser.add_argument('--test_path', type=str, required=True, help='测试数据路径')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--hidden_channels', type=int, default=32, help='隐藏通道数')
    parser.add_argument('--n_layers', type=int, default=3, help='网络层数')
    parser.add_argument('--n_modes', type=int, nargs='+', default=[32, 32], help='FFT 模式数量')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='计算设备')
    parser.add_argument('--output', type=str, default='comparison_result.json', help='输出文件')
    
    args = parser.parse_args()
    
    # 运行对比测试
    result = run_comparison(
        dataset_name=args.dataset,
        train_path=args.train_path,
        test_path=args.test_path,
        n_modes=tuple(args.n_modes),
        hidden_channels=args.hidden_channels,
        n_layers=args.n_layers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device
    )
    
    # 保存结果
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n结果已保存到: {output_path}")


if __name__ == '__main__':
    main()
