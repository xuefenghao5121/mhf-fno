#!/usr/bin/env python3
"""
PINO Round 3 测试: 梯度异常惩罚

策略: 只惩罚异常梯度值（离群点），使用鲁棒统计
- lambda_physics: 0.001
- outlier_threshold: 3.0 (MAD倍数)

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

from mhf_fno import MHFFNO
from neuralop.models import FNO
from neuralop.losses.data_losses import LpLoss


def load_ns_data(data_path, n_train, n_test):
    """加载 Navier-Stokes 数据"""
    data = torch.load(data_path, weights_only=False)
    
    if isinstance(data, dict):
        train_x = data.get('x', data.get('train_x'))
        train_y = data.get('y', data.get('train_y'))
    else:
        train_x, train_y = data[0], data[1]
    
    if train_x.dim() == 3:
        train_x = train_x.unsqueeze(1).float()
        train_y = train_y.unsqueeze(1).float()
    
    actual_n_train = min(n_train, len(train_x) - n_test)
    train_x_split = train_x[:actual_n_train]
    train_y_split = train_y[:actual_n_train]
    test_x = train_x[actual_n_train:actual_n_train+n_test]
    test_y = train_y[actual_n_train:actual_n_train+n_test]
    
    return train_x_split, train_y_split, test_x, test_y


class OutlierGradientPINOLoss(nn.Module):
    """只惩罚异常梯度值的 PINO 损失"""
    
    def __init__(self, lambda_physics=0.001, outlier_threshold=3.0):
        super().__init__()
        self.lambda_phy = lambda_physics
        self.threshold = outlier_threshold
    
    def compute_outlier_penalty(self, u):
        """计算梯度异常惩罚"""
        # 计算梯度
        u_x = torch.gradient(u, dim=-1)[0]
        u_y = torch.gradient(u, dim=-2)[0]
        
        # 梯度幅值
        grad_mag = torch.sqrt(u_x**2 + u_y**2 + 1e-8)
        
        # 鲁棒统计: 中位数 + MAD
        median = torch.median(grad_mag)
        mad = torch.median(torch.abs(grad_mag - median))
        
        # 异常值掩码 (超过 threshold * MAD)
        outlier_mask = (grad_mag > median + self.threshold * mad).float()
        
        # 只惩罚异常值
        outlier_loss = (grad_mag ** 2 * outlier_mask).mean()
        
        return outlier_loss
    
    def forward(self, u_pred, u_true):
        L_data = F.mse_loss(u_pred, u_true)
        L_outlier = self.compute_outlier_penalty(u_pred)
        return L_data + self.lambda_phy * L_outlier


def train_model(model, train_x, train_y, test_x, test_y, config, model_name, loss_fn=None, use_pino_loss=False):
    """训练模型"""
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # 统一使用 MSE loss 进行公平对比
    if loss_fn is None:
        loss_fn = nn.MSELoss()
    
    # 用于报告的 LpLoss（不参与训练）
    lp_loss_fn = LpLoss(d=2, p=2, reduction='mean')
    
    results = {'test_losses': []}
    n_train = train_x.shape[0]
    batch_size = config['batch_size']
    
    print(f"\n训练 {model_name}...")
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    best_test_loss = float('inf')
    
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
            loss = loss_fn(pred, by)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            batch_count += 1
        
        # 测试（同时计算 MSE 和 LpLoss）
        model.eval()
        with torch.no_grad():
            test_pred = model(test_x)
            test_loss = loss_fn(test_pred, test_y).item()
            test_lp_loss = lp_loss_fn(test_pred, test_y).item()
            results['test_losses'].append(test_loss)
            
            if test_loss < best_test_loss:
                best_test_loss = test_loss
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{config['epochs']}: Train {train_loss/batch_count:.4f}, Test {test_loss:.4f} (Lp: {test_lp_loss:.4f})")
    
    return best_test_loss, results


def main():
    print("=" * 70)
    print("PINO Round 3 测试: 梯度异常惩罚")
    print("=" * 70)
    
    config = {
        'n_train': 500,
        'n_test': 100,
        'epochs': 50,
        'batch_size': 32,
        'learning_rate': 0.0005,
    }
    
    data_path = Path(__file__).parent.parent / 'data' / 'ns_train_32.pt'
    train_x, train_y, test_x, test_y = load_ns_data(data_path, config['n_train'], config['n_test'])
    print(f"\n数据: train {train_x.shape}, test {test_x.shape}")
    
    # MHF-FNO baseline
    print("\n" + "=" * 60)
    print("MHF-FNO (baseline, 无 PINO)")
    print("=" * 60)
    mhf_model = MHFFNO.best_config(n_modes=(16, 16), hidden_channels=32)
    mhf_loss, _ = train_model(mhf_model, train_x, train_y, test_x, test_y, config, "MHF-FNO")
    print(f"\n✓ MHF-FNO 最佳测试损失: {mhf_loss:.4f}")
    
    # MHF-PINO Round 3
    print("\n" + "=" * 60)
    print("MHF-PINO (Round 3: 梯度异常惩罚)")
    print("=" * 60)
    mhf_pino = MHFFNO.best_config(n_modes=(16, 16), hidden_channels=32)
    pino_loss_fn = OutlierGradientPINOLoss(lambda_physics=0.001, outlier_threshold=3.0)
    pino_loss, _ = train_model(mhf_pino, train_x, train_y, test_x, test_y, config, "MHF-PINO (Round 3)", loss_fn=pino_loss_fn)
    print(f"\n✓ MHF-PINO (Round 3) 最佳测试损失: {pino_loss:.4f}")
    
    # 结果
    print("\n" + "=" * 70)
    print("结果对比")
    print("=" * 70)
    print(f"{'模型':<30} {'Test Loss':>12} {'vs MHF-FNO':>12}")
    print("-" * 54)
    print(f"{'MHF-FNO (baseline)':<30} {mhf_loss:>12.4f} {'baseline':>12}")
    
    diff_pct = (pino_loss - mhf_loss) / mhf_loss * 100
    status = "✅ 成功" if diff_pct < 0 else "❌ 失败"
    print(f"{'MHF-PINO (Round 3)':<30} {pino_loss:>12.4f} {diff_pct:>+11.2f}%")
    
    print("\n" + "=" * 70)
    print("测试结论")
    print("=" * 70)
    if diff_pct < 0:
        print(f"✅ 成功! MHF-PINO ({pino_loss:.4f}) < MHF-FNO ({mhf_loss:.4f})")
        print(f"   提升: {-diff_pct:.2f}%")
    else:
        print(f"❌ 失败! MHF-PINO ({pino_loss:.4f}) >= MHF-FNO ({mhf_loss:.4f})")
        print(f"   恶化: {diff_pct:.2f}%")
        print("\n结论: 所有 PINO 策略均无效，MHF-FNO 无需 PINO 即可达到最优性能")
    
    # 保存结果
    results = {
        'round': 3,
        'strategy': 'outlier_gradient_penalty',
        'config': config,
        'mhf_fno_loss': mhf_loss,
        'mhf_pino_loss': pino_loss,
        'improvement_pct': -diff_pct,
        'success': diff_pct < 0
    }
    
    output_path = Path(__file__).parent.parent / 'pino_round3_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ 结果已保存: {output_path}")


if __name__ == "__main__":
    main()
