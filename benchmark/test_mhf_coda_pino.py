#!/usr/bin/env python3
"""
MHF+CoDA+PINO 测试：真实 NS 速度场数据

使用：
- MHF+CoDA（跨头注意力）
- 真实 NS 速度场数据（时间序列）
- PINO 物理约束（NS 方程残差）

目标：
验证 MHF+CoDA+PINO 在真实 NS 数据上的效果

作者: 天渊团队
日期: 2026-03-26
"""

import sys
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from mhf_fno import MHFFNOWithAttention
from mhf_fno.pino_high_freq import HighFreqPINOLoss


def load_ns_velocity_data(data_path, n_train, n_test, time_steps=10):
    """加载 NS 速度场数据"""
    data = torch.load(data_path, weights_only=False)
    
    velocity = data['velocity']  # [N, T, 2, H, W]
    pressure = data['pressure']  # [N, T, 1, H, W]
    
    # 只使用前 time_steps 个时间步
    velocity = velocity[:, :time_steps]
    
    # 分割训练集和测试集
    train_velocity = velocity[:n_train]
    test_velocity = velocity[n_train:n_train+n_test]
    
    # 准备输入输出对：(u^t, u^{t+1})
    train_x = train_velocity[:, :-1]  # [N, T-1, 2, H, W]
    train_y = train_velocity[:, 1:]   # [N, T-1, 2, H, W]
    test_x = test_velocity[:, :-1]
    test_y = test_velocity[:, 1:]
    
    # 展平时间维度：[N, T-1, 2, H, W] -> [N*(T-1), 2, H, W]
    train_x = train_x.reshape(-1, 2, 64, 64)
    train_y = train_y.reshape(-1, 2, 64, 64)
    test_x = test_x.reshape(-1, 2, 64, 64)
    test_y = test_y.reshape(-1, 2, 64, 64)
    
    return train_x, train_y, test_x, test_y


class NSPhysicsLoss(nn.Module):
    """
    Navier-Stokes 物理损失
    
    NS 方程: ∂u/∂t + (u·∇)u = -∇p + ν∇²u
    """
    
    def __init__(self, viscosity=1e-3, dt=0.01, lambda_physics=0.001):
        super().__init__()
        self.nu = viscosity
        self.dt = dt
        self.lambda_phy = lambda_physics
    
    def compute_ns_residual(self, u_pred, u_prev):
        """计算 NS 方程残差"""
        # u_pred, u_prev: [B, 2, H, W]
        
        # 时间导数: ∂u/∂t ≈ (u^{t+1} - u^t) / dt
        u_t = (u_pred - u_prev) / self.dt
        
        # 空间导数
        u_x = torch.gradient(u_pred[:, 0], dim=-1)[0]  # ∂u/∂x
        u_y = torch.gradient(u_pred[:, 0], dim=-2)[0]  # ∂u/∂y
        v_x = torch.gradient(u_pred[:, 1], dim=-1)[0]  # ∂v/∂x
        v_y = torch.gradient(u_pred[:, 1], dim=-2)[0]  # ∂v/∂y
        
        # 平流项: (u·∇)u
        advection_u = u_pred[:, 0] * u_x + u_pred[:, 1] * u_y
        advection_v = u_pred[:, 0] * v_x + u_pred[:, 1] * v_y
        
        # 拉普拉斯算子: ∇²u
        u_xx = torch.gradient(u_x, dim=-1)[0]
        u_yy = torch.gradient(u_y, dim=-2)[0]
        v_xx = torch.gradient(v_x, dim=-1)[0]
        v_yy = torch.gradient(v_y, dim=-2)[0]
        
        laplacian_u = u_xx + u_yy
        laplacian_v = v_xx + v_yy
        
        # NS 方程残差: ∂u/∂t + (u·∇)u - ν∇²u
        residual_u = u_t[:, 0] + advection_u - self.nu * laplacian_u
        residual_v = u_t[:, 1] + advection_v - self.nu * laplacian_v
        
        # 总残差
        residual = residual_u**2 + residual_v**2
        
        return residual.mean()
    
    def forward(self, u_pred, u_true, u_prev):
        """总损失 = 数据损失 + λ × 物理损失"""
        L_data = F.mse_loss(u_pred, u_true)
        L_physics = self.compute_ns_residual(u_pred, u_prev)
        
        return L_data + self.lambda_phy * L_physics


def train_model(model, train_x, train_y, test_x, test_y, config, model_name, use_pino=False):
    """训练模型"""
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    if use_pino:
        loss_fn = NSPhysicsLoss(
            viscosity=1e-3,
            dt=0.01,
            lambda_physics=0.001
        )
    else:
        loss_fn = nn.MSELoss()
    
    best_test_loss = float('inf')
    
    print(f"\n训练 {model_name}...")
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    n_train = train_x.shape[0]
    batch_size = config['batch_size']
    
    for epoch in range(config['epochs']):
        model.train()
        perm = torch.randperm(n_train)
        train_loss = 0
        batch_count = 0
        
        for i in range(0, n_train, batch_size):
            bx = train_x[perm[i:i+batch_size]]
            by = train_y[perm[i:i+batch_size]]
            
            optimizer.zero_grad()
            pred = model(bx)
            
            if use_pino:
                # PINO 需要上一时刻的数据
                # 这里简化处理：假设 bx 就是 u^t, by 是 u^{t+1}
                loss = loss_fn(pred, by, bx)
            else:
                loss = loss_fn(pred, by)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            batch_count += 1
        
        # 测试
        model.eval()
        with torch.no_grad():
            test_pred = model(test_x)
            test_loss = F.mse_loss(test_pred, test_y).item()
            
            if test_loss < best_test_loss:
                best_test_loss = test_loss
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{config['epochs']}: Train {train_loss/batch_count:.4f}, Test {test_loss:.4f}")
    
    return best_test_loss


def main():
    print("=" * 70)
    print("MHF+CoDA+PINO 测试：真实 NS 速度场数据")
    print("=" * 70)
    
    config = {
        'n_train': 150,
        'n_test': 50,
        'time_steps': 10,
        'epochs': 20,
        'batch_size': 16,
        'learning_rate': 0.0001,
    }
    
    # 加载数据
    data_path = Path(__file__).parent / 'data' / 'ns_real_velocity.pt'
    print(f"\n加载数据: {data_path}")
    train_x, train_y, test_x, test_y = load_ns_velocity_data(
        data_path,
        config['n_train'],
        config['n_test'],
        config['time_steps']
    )
    print(f"训练集: {train_x.shape}")
    print(f"测试集: {test_x.shape}")
    
    # 测试 1: MHF+CoDA（无 PINO）
    print("\n" + "=" * 70)
    print("测试 1: MHF+CoDA（无 PINO）")
    print("=" * 70)
    model1 = MHFFNOWithAttention.best_config(
        n_modes=(16, 16),
        hidden_channels=32,
        in_channels=2,
        out_channels=2
    )
    loss1 = train_model(model1, train_x, train_y, test_x, test_y, config, "MHF+CoDA", use_pino=False)
    print(f"\n✓ MHF+CoDA 最佳测试损失: {loss1:.4f}")
    
    # 测试 2: MHF+CoDA+PINO
    print("\n" + "=" * 70)
    print("测试 2: MHF+CoDA+PINO")
    print("=" * 70)
    model2 = MHFFNOWithAttention.best_config(
        n_modes=(16, 16),
        hidden_channels=32,
        in_channels=2,
        out_channels=2
    )
    loss2 = train_model(model2, train_x, train_y, test_x, test_y, config, "MHF+CoDA+PINO", use_pino=True)
    print(f"\n✓ MHF+CoDA+PINO 最佳测试损失: {loss2:.4f}")
    
    # 结果
    print("\n" + "=" * 70)
    print("结果对比")
    print("=" * 70)
    print(f"{'模型':<30} {'Test Loss':>12} {'vs Baseline':>12}")
    print("-" * 54)
    print(f"{'MHF+CoDA (baseline)':<30} {loss1:>12.4f} {'baseline':>12}")
    
    diff_pct = (loss2 - loss1) / loss1 * 100
    status = "✅ 成功" if diff_pct < 0 else "❌ 失败"
    print(f"{'MHF+CoDA+PINO':<30} {loss2:>12.4f} {diff_pct:>+11.2f}% {status}")
    
    print("\n" + "=" * 70)
    print("测试结论")
    print("=" * 70)
    if diff_pct < 0:
        print(f"✅ 成功! PINO 在真实 NS 数据上有效")
        print(f"   提升: {-diff_pct:.2f}%")
    else:
        print(f"❌ 失败! PINO 在真实 NS 数据上无效")
        print(f"   恶化: {diff_pct:.2f}%")
        print("\n可能原因:")
        print("  1. 真实 NS 数据的物理约束更复杂")
        print("  2. lambda_physics 需要调优")
        print("  3. 需要更长的训练时间")
    
    # 保存结果
    results = {
        'test': 'MHF+CoDA+PINO on real NS velocity data',
        'config': config,
        'mhf_coda_loss': loss1,
        'mhf_coda_pino_loss': loss2,
        'improvement_pct': -diff_pct,
        'success': diff_pct < 0
    }
    
    output_path = Path(__file__).parent.parent / 'mhf_coda_pino_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ 结果已保存: {output_path}")


if __name__ == "__main__":
    main()
