#!/usr/bin/env python3
"""
Darcy Flow 基准测试（简化版）

直接使用生成的 PT 文件进行测试
"""

import torch
import torch.nn as nn
import time
from pathlib import Path
import argparse

from neuralop.models import FNO
from neuralop.losses import LpLoss


def test_darcy_flow(
    train_path: str,
    test_path: str,
    n_modes: tuple = (32, 32),
    hidden_channels: int = 32,
    n_layers: int = 3,
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 0.001,
    device: str = 'cpu'
) -> dict:
    """
    测试 Darcy Flow 数据集
    """
    print("=" * 70)
    print("Darcy Flow 基准测试")
    print("=" * 70)
    
    # 加载数据
    print(f"\n加载数据...")
    train_data = torch.load(train_path)
    test_data = torch.load(test_path)
    
    train_x = train_data['x'].to(device)
    train_y = train_data['y'].to(device)
    test_x = test_data['x'].to(device)
    test_y = test_data['y'].to(device)
    
    print(f"  训练集: {train_x.shape}")
    print(f"  测试集: {test_x.shape}")
    
    # 创建模型
    print(f"\n创建 FNO 模型...")
    model = FNO(
        n_modes=n_modes,
        hidden_channels=hidden_channels,
        in_channels=train_x.shape[1],
        out_channels=train_y.shape[1],
        n_layers=n_layers
    ).to(device)
    
    # 计算参数量
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {n_params:,}")
    
    # 损失函数和优化器
    loss_fn = LpLoss(d=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # 训练
    print(f"\n开始训练 ({epochs} epochs)...")
    train_losses = []
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # 随机打乱
        indices = torch.randperm(train_x.shape[0])
        epoch_loss = 0.0
        n_batches = 0
        
        for i in range(0, train_x.shape[0], batch_size):
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
        train_losses.append(avg_loss)
        
        epoch_time = time.time() - epoch_start
        
        print(f"  Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.6f}, Time = {epoch_time:.2f}s")
    
    total_train_time = time.time() - start_time
    
    # 测试
    print(f"\n测试...")
    model.eval()
    with torch.no_grad():
        test_out = model(test_x)
        test_loss = loss_fn(test_out, test_y)
    
    # 推理速度测试
    print(f"\n推理速度测试...")
    model.eval()
    with torch.no_grad():
        inference_times = []
        for i in range(0, min(100, test_x.shape[0]), 1):
            start = time.time()
            _ = model(test_x[i:i+1])
            end = time.time()
            inference_times.append((end - start) * 1000)  # 转换为毫秒
        
        avg_inference_time = sum(inference_times) / len(inference_times)
    
    # 结果
    result = {
        'model': 'FNO',
        'n_modes': n_modes,
        'hidden_channels': hidden_channels,
        'n_layers': n_layers,
        'parameters': n_params,
        'train_samples': train_x.shape[0],
        'test_samples': test_x.shape[0],
        'resolution': train_x.shape[2],
        'epochs': epochs,
        'batch_size': batch_size,
        'lr': lr,
        'device': device,
        'train_losses': train_losses,
        'best_train_loss': min(train_losses),
        'final_train_loss': train_losses[-1],
        'test_loss': test_loss.item(),
        'total_train_time': total_train_time,
        'avg_inference_time_ms': avg_inference_time,
    }
    
    print("\n" + "=" * 70)
    print("测试结果")
    print("=" * 70)
    print(f"  模型: FNO")
    print(f"  参数量: {n_params:,}")
    print(f"  最终训练 Loss: {result['final_train_loss']:.6f}")
    print(f"  测试 Loss: {result['test_loss']:.6f}")
    print(f"  总训练时间: {total_train_time:.2f}s")
    print(f"  平均推理时间: {avg_inference_time:.3f}ms")
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Darcy Flow 基准测试')
    parser.add_argument('--train_path', type=str, default='./data/darcy_train_64_fixed.pt',
                       help='训练数据路径')
    parser.add_argument('--test_path', type=str, default='./data/darcy_test_64_fixed.pt',
                       help='测试数据路径')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--n_modes', type=int, nargs=2, default=[32, 32],
                       help='FFT 模式数量')
    parser.add_argument('--hidden_channels', type=int, default=32,
                       help='隐藏通道数')
    parser.add_argument('--n_layers', type=int, default=3,
                       help='网络层数')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='学习率')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'], help='计算设备')
    
    args = parser.parse_args()
    
    result = test_darcy_flow(
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
    import json
    output_path = Path('darcy_benchmark_result.json')
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n结果已保存到: {output_path}")


if __name__ == '__main__':
    main()
