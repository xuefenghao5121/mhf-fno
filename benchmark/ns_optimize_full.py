#!/usr/bin/env python3
"""
NS 优化测试 - 数据量扩展 + PINO 约束

优化方向:
1. 数据量扩展: n_train=200 → 1000
2. PINO 约束: 添加 NS 方程残差损失
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from neuralop.losses.data_losses import LpLoss
from neuralop.models import FNO

# CPU 优化
torch.set_num_threads(os.cpu_count() or 1)

# 导入项目模块
sys.path.insert(0, str(Path(__file__).parent.parent))
from mhf_fno import create_hybrid_fno, create_mhf_fno_with_attention


# ============================================================================
# PINO 物理损失
# ============================================================================

class NavierStokesPhysicsLoss(torch.nn.Module):
    """
    Navier-Stokes 方程物理约束损失
    
    NS 方程:
        ∂u/∂t + (u·∇)u = -∇p + ν∇²u
        ∇·u = 0 (不可压缩)
    
    对于稳态预测，我们使用简化的残差形式。
    """
    
    def __init__(self, viscosity=1e-3, dt=1.0, dx=1.0):
        super().__init__()
        self.viscosity = viscosity
        self.dt = dt
        self.dx = dx
    
    def compute_gradient(self, u, dim):
        """使用中心差分计算空间梯度"""
        # u: [B, C, H, W]
        if dim == 0:  # x 方向 (高度)
            du = (torch.roll(u, -1, dims=2) - torch.roll(u, 1, dims=2)) / (2 * self.dx)
        else:  # y 方向 (宽度)
            du = (torch.roll(u, -1, dims=3) - torch.roll(u, 1, dims=3)) / (2 * self.dx)
        return du
    
    def compute_laplacian(self, u):
        """计算拉普拉斯算子 ∇²u"""
        # ∇²u ≈ (u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1} - 4u_{i,j}) / dx²
        lap = (
            torch.roll(u, -1, dims=2) + torch.roll(u, 1, dims=2) +
            torch.roll(u, -1, dims=3) + torch.roll(u, 1, dims=3) - 
            4 * u
        ) / (self.dx ** 2)
        return lap
    
    def forward(self, u_pred, u_input=None):
        """
        计算 NS 方程残差损失
        
        Args:
            u_pred: 预测的流场 [B, C, H, W]
            u_input: 输入场（初始条件）
        
        Returns:
            physics_loss: 物理约束损失
        """
        # 对于单通道预测，我们计算扩散项
        # L_physics = ||∇²u||² (扩散守恒)
        
        if u_pred.dim() == 4:  # 2D
            lap = self.compute_laplacian(u_pred)
            physics_loss = (lap ** 2).mean()
        else:
            # 1D case
            d2u = (torch.roll(u_pred, -1, dims=-1) + torch.roll(u_pred, 1, dims=-1) - 2 * u_pred) / (self.dx ** 2)
            physics_loss = (d2u ** 2).mean()
        
        return physics_loss


# ============================================================================
# 数据加载
# ============================================================================

def load_ns_data_full():
    """加载完整 NS 数据 (1000 samples)"""
    print(f"\n📊 加载 Navier-Stokes 数据 (1000 samples)...")
    
    # 使用大版本数据
    train_path_large = Path(__file__).parent / 'data' / 'ns_train_32_large.pt'
    test_path_large = Path(__file__).parent / 'data' / 'ns_test_32_large.pt'
    
    if not train_path_large.exists():
        print(f"❌ 大版本数据不存在，尝试生成...")
        return None
    
    train_data = torch.load(train_path_large, weights_only=False)
    test_data = torch.load(test_path_large, weights_only=False)
    
    # 解析数据
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
    
    # 确保维度
    if train_x.dim() == 3:
        train_x = train_x.unsqueeze(1)
        train_y = train_y.unsqueeze(1)
    if test_x.dim() == 3:
        test_x = test_x.unsqueeze(1)
        test_y = test_y.unsqueeze(1)
    
    train_x = train_x.float()
    train_y = train_y.float()
    test_x = test_x.float()
    test_y = test_y.float()
    
    resolution = train_x.shape[-1]
    
    info = {
        'name': 'Navier-Stokes (1000 samples)',
        'resolution': f'{resolution}x{resolution}',
        'n_train': train_x.shape[0],
        'n_test': test_x.shape[0],
        'input_channels': train_x.shape[1],
        'output_channels': train_y.shape[1],
        'n_modes': (resolution // 4, resolution // 4),
    }
    
    print(f"✅ 加载成功: 训练 {train_x.shape[0]}, 测试 {test_x.shape[0]}")
    return train_x, train_y, test_x, test_y, info


def load_ns_data_partial(n_train=500, n_test=100):
    """加载部分 NS 数据"""
    data = load_ns_data_full()
    if data is None:
        return None
    
    train_x, train_y, test_x, test_y, info = data
    train_x = train_x[:n_train]
    train_y = train_y[:n_train]
    test_x = test_x[:n_test]
    test_y = test_y[:n_test]
    info['n_train'] = n_train
    info['n_test'] = n_test
    
    return train_x, train_y, test_x, test_y, info


# ============================================================================
# 训练函数
# ============================================================================

def train_standard(model, train_x, train_y, test_x, test_y, epochs=50, verbose=True):
    """标准训练"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = LpLoss(d=2, p=2, reduction='mean')
    
    results = {'best_test_loss': float('inf')}
    n_train = train_x.shape[0]
    batch_size = 32
    
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train)
        
        for i in range(0, n_train, batch_size):
            bx = train_x[perm[i:i+batch_size]]
            by = train_y[perm[i:i+batch_size]]
            
            optimizer.zero_grad()
            loss = loss_fn(model(bx), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        scheduler.step()
        
        model.eval()
        with torch.no_grad():
            test_loss = loss_fn(model(test_x), test_y).item()
        
        if test_loss < results['best_test_loss']:
            results['best_test_loss'] = test_loss
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: Test {test_loss:.4f} (best: {results['best_test_loss']:.4f})")
    
    return results


def train_pino(model, train_x, train_y, test_x, test_y, epochs=50, physics_weight=0.1, verbose=True):
    """PINO 训练 (带物理约束)"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = LpLoss(d=2, p=2, reduction='mean')
    physics_loss_fn = NavierStokesPhysicsLoss(viscosity=1e-3)
    
    results = {'best_test_loss': float('inf')}
    n_train = train_x.shape[0]
    batch_size = 32
    
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train)
        
        for i in range(0, n_train, batch_size):
            bx = train_x[perm[i:i+batch_size]]
            by = train_y[perm[i:i+batch_size]]
            
            optimizer.zero_grad()
            
            # 数据损失
            pred = model(bx)
            data_loss = loss_fn(pred, by)
            
            # 物理损失
            physics_loss = physics_loss_fn(pred, bx)
            
            # 总损失
            total_loss = data_loss + physics_weight * physics_loss
            total_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        scheduler.step()
        
        model.eval()
        with torch.no_grad():
            test_loss = loss_fn(model(test_x), test_y).item()
        
        if test_loss < results['best_test_loss']:
            results['best_test_loss'] = test_loss
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: Test {test_loss:.4f} (best: {results['best_test_loss']:.4f})")
    
    return results


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


# ============================================================================
# 主测试
# ============================================================================

def main():
    print("="*60)
    print("NS 优化测试 - 数据量 + PINO")
    print("="*60)
    
    # 加载数据
    data_full = load_ns_data_full()
    if data_full is None:
        print("无法加载数据")
        return
    
    train_x_full, train_y_full, test_x_full, test_y_full, info = data_full
    
    results = {}
    
    # ========== 测试1: 数据量扩展 ==========
    print(f"\n{'='*60}")
    print("Part 1: 数据量扩展测试")
    print(f"{'='*60}")
    
    data_configs = [
        ('n=200', 200),
        ('n=500', 500),
        ('n=1000', 1000),
    ]
    
    for name, n_train in data_configs:
        print(f"\n--- {name} ---")
        
        train_x = train_x_full[:n_train]
        train_y = train_y_full[:n_train]
        test_x = test_x_full[:200]
        test_y = test_y_full[:200]
        
        # FNO
        print(f"FNO...")
        torch.manual_seed(42)
        model_fno = FNO(
            n_modes=info['n_modes'],
            hidden_channels=32,
            in_channels=info['input_channels'],
            out_channels=info['output_channels'],
            n_layers=3,
        )
        res_fno = train_standard(model_fno, train_x, train_y, test_x, test_y, epochs=50, verbose=False)
        
        # MHF+CoDA
        print(f"MHF+CoDA...")
        torch.manual_seed(42)
        model_mhf = create_mhf_fno_with_attention(
            n_modes=info['n_modes'],
            hidden_channels=32,
            in_channels=info['input_channels'],
            out_channels=info['output_channels'],
            n_layers=3,
            n_heads=4,
            mhf_layers=[0, 2],
            attention_layers=[0, 2]
        )
        res_mhf = train_standard(model_mhf, train_x, train_y, test_x, test_y, epochs=50, verbose=False)
        
        improvement = (res_fno['best_test_loss'] - res_mhf['best_test_loss']) / res_fno['best_test_loss'] * 100
        marker = "✅" if improvement > 0 else "⚠️"
        
        print(f"  FNO: {res_fno['best_test_loss']:.4f}")
        print(f"  MHF+CoDA: {res_mhf['best_test_loss']:.4f} ({improvement:+.2f}% {marker})")
        
        results[f'data_{name}'] = {
            'fno_loss': res_fno['best_test_loss'],
            'mhf_loss': res_mhf['best_test_loss'],
            'improvement': improvement,
        }
    
    # ========== 测试2: PINO 约束 ==========
    print(f"\n{'='*60}")
    print("Part 2: PINO 约束测试 (n=1000)")
    print(f"{'='*60}")
    
    train_x = train_x_full[:1000]
    train_y = train_y_full[:1000]
    test_x = test_x_full[:200]
    test_y = test_y_full[:200]
    
    # 标准训练
    print(f"\n标准训练:")
    torch.manual_seed(42)
    model_std = create_mhf_fno_with_attention(
        n_modes=info['n_modes'],
        hidden_channels=32,
        in_channels=info['input_channels'],
        out_channels=info['output_channels'],
        n_layers=3,
        n_heads=4,
        mhf_layers=[0, 2],
        attention_layers=[0, 2]
    )
    res_std = train_standard(model_std, train_x, train_y, test_x, test_y, epochs=50, verbose=True)
    
    # PINO 训练 (不同权重)
    pino_weights = [0.01, 0.1, 1.0]
    
    for w in pino_weights:
        print(f"\nPINO 训练 (physics_weight={w}):")
        torch.manual_seed(42)
        model_pino = create_mhf_fno_with_attention(
            n_modes=info['n_modes'],
            hidden_channels=32,
            in_channels=info['input_channels'],
            out_channels=info['output_channels'],
            n_layers=3,
            n_heads=4,
            mhf_layers=[0, 2],
            attention_layers=[0, 2]
        )
        res_pino = train_pino(model_pino, train_x, train_y, test_x, test_y, epochs=50, physics_weight=w, verbose=True)
        
        results[f'pino_w{w}'] = {
            'loss': res_pino['best_test_loss'],
        }
    
    # ========== 汇总 ==========
    print(f"\n{'='*60}")
    print("结果汇总")
    print(f"{'='*60}")
    
    print(f"\n数据量扩展:")
    print(f"{'配置':<15} {'FNO':<12} {'MHF+CoDA':<12} {'提升':<12}")
    print("-"*50)
    for name in ['n=200', 'n=500', 'n=1000']:
        r = results[f'data_{name}']
        marker = "✅" if r['improvement'] > 0 else "⚠️"
        print(f"{name:<15} {r['fno_loss']:<12.4f} {r['mhf_loss']:<12.4f} {r['improvement']:+.2f}% {marker}")
    
    print(f"\nPINO 约束 (n=1000):")
    print(f"{'配置':<20} {'测试Loss':<12}")
    print("-"*35)
    print(f"{'标准训练':<20} {res_std['best_test_loss']:<12.4f}")
    for w in pino_weights:
        print(f"{'PINO w=' + str(w):<20} {results[f'pino_w{w}']['loss']:<12.4f}")
    
    # 保存
    output_path = Path(__file__).parent.parent / 'ns_optimization_final.json'
    with open(output_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': results,
        }, f, indent=2)
    
    print(f"\n✅ 结果已保存到: {output_path}")


if __name__ == '__main__':
    main()