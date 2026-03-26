#!/usr/bin/env python3
"""
PINO 诊断测试 - 找出性能问题根源
"""

import sys
from pathlib import Path

import torch
from neuralop.losses.data_losses import LpLoss

sys.path.insert(0, str(Path(__file__).parent.parent))
from mhf_fno import create_mhf_fno_with_attention, PINOLoss

# 强制刷新输出
import functools
print = functools.partial(print, flush=True)


def diagnose_pino_loss():
    """诊断 PINO 损失函数"""
    print("=" * 80)
    print("PINO 损失函数诊断")
    print("=" * 80)
    
    # 创建测试数据
    torch.manual_seed(42)
    batch_size = 4
    H, W = 32, 32
    
    u_pred = torch.randn(batch_size, 1, H, W)
    u_true = torch.randn(batch_size, 1, H, W)
    
    # 测试不同损失函数
    print("\n1️⃣ 损失函数对比:")
    print("-" * 40)
    
    # MSE
    mse_loss = torch.nn.functional.mse_loss(u_pred, u_true)
    print(f"MSE Loss: {mse_loss.item():.6f}")
    
    # LpLoss
    lp_loss_fn = LpLoss(d=2, p=2, reduction='mean')
    lp_loss = lp_loss_fn(u_pred, u_true)
    print(f"LpLoss (p=2, mean): {lp_loss.item():.6f}")
    
    # PINOLoss
    pino_loss_fn = PINOLoss(lambda_physics=0.01, smoothness_weight=0.5)
    pino_loss = pino_loss_fn(u_pred, u_true)
    print(f"PINO Loss (λ=0.01): {pino_loss.item():.6f}")
    
    # 分解 PINO 损失
    print("\n2️⃣ PINO 损失分解:")
    print("-" * 40)
    
    L_data = torch.nn.functional.mse_loss(u_pred, u_true)
    L_smooth = pino_loss_fn.compute_smoothness_loss(u_pred)
    L_laplacian = pino_loss_fn.compute_laplacian_loss(u_pred)
    
    print(f"L_data (MSE): {L_data.item():.6f}")
    print(f"L_smooth: {L_smooth.item():.6f}")
    print(f"L_laplacian: {L_laplacian.item():.6f}")
    print(f"L_physics: {0.5 * L_smooth.item() + 0.5 * L_laplacian.item():.6f}")
    print(f"Total PINO: {L_data.item() + 0.01 * (0.5 * L_smooth.item() + 0.5 * L_laplacian.item()):.6f}")
    
    # 测试 lambda_physics 的影响
    print("\n3️⃣ Lambda Physics 影响:")
    print("-" * 40)
    
    for lambda_val in [0.0, 0.001, 0.01, 0.1, 1.0]:
        pino_fn = PINOLoss(lambda_physics=lambda_val)
        loss = pino_fn(u_pred, u_true)
        print(f"λ={lambda_val:5.3f}: loss={loss.item():.6f}")
    
    print("\n" + "=" * 80)


def quick_training_test():
    """快速训练对比测试"""
    print("\n" + "=" * 80)
    print("快速训练对比 (确保损失函数一致)")
    print("=" * 80)
    
    # 加载数据
    train_path = Path(__file__).parent / 'data' / 'ns_train_32_large.pt'
    test_path = Path(__file__).parent / 'data' / 'ns_test_32_large.pt'
    
    train_data = torch.load(train_path, weights_only=False)
    test_data = torch.load(test_path, weights_only=False)
    
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
    
    # 只用前 100 个样本快速测试
    train_x = train_x[:100]
    train_y = train_y[:100]
    test_x = test_x[:20]
    test_y = test_y[:20]
    
    device = torch.device('cpu')
    
    # 统一使用 LpLoss 作为测试指标
    test_loss_fn = LpLoss(d=2, p=2)
    
    # ========================================
    # 测试 1: 标准 FNO (MSE loss)
    # ========================================
    print("\n1️⃣ FNO + MSE Loss:")
    print("-" * 40)
    
    from neuralop.models import FNO
    
    model_fno = FNO(
        n_modes=(16, 16),
        hidden_channels=32,
        in_channels=1,
        out_channels=1,
        n_layers=4
    ).to(device)
    
    optimizer = torch.optim.Adam(model_fno.parameters(), lr=1e-3)
    mse_loss_fn = torch.nn.MSELoss()
    
    for epoch in range(5):
        model_fno.train()
        optimizer.zero_grad()
        y_pred = model_fno(train_x)
        loss = mse_loss_fn(y_pred, train_y)
        loss.backward()
        optimizer.step()
        
        model_fno.eval()
        with torch.no_grad():
            test_pred = model_fno(test_x)
            test_loss = test_loss_fn(test_pred, test_y).item()
        
        print(f"Epoch {epoch+1}: train_loss={loss.item():.6f}, test_lploss={test_loss:.6f}")
    
    fno_final_loss = test_loss
    
    # ========================================
    # 测试 2: MHF-FNO + PINO (但测试用 LpLoss)
    # ========================================
    print("\n2️⃣ MHF-FNO + PINO Loss:")
    print("-" * 40)
    
    model_pino = create_mhf_fno_with_attention(
        n_modes=(16, 16),
        hidden_channels=32,
        in_channels=1,
        out_channels=1,
        n_layers=4,
        n_heads=2,
        mhf_layers=[0],
        attention_layers=[0]
    ).to(device)
    
    optimizer = torch.optim.Adam(model_pino.parameters(), lr=1e-3)
    pino_loss_fn = PINOLoss(lambda_physics=0.01)
    
    for epoch in range(5):
        model_pino.train()
        optimizer.zero_grad()
        y_pred = model_pino(train_x)
        loss = pino_loss_fn(y_pred, train_y)
        loss.backward()
        optimizer.step()
        
        model_pino.eval()
        with torch.no_grad():
            test_pred = model_pino(test_x)
            test_loss = test_loss_fn(test_pred, test_y).item()
        
        print(f"Epoch {epoch+1}: train_loss={loss.item():.6f}, test_lploss={test_loss:.6f}")
    
    pino_final_loss = test_loss
    
    # ========================================
    # 结果对比
    # ========================================
    print("\n" + "=" * 80)
    print("📊 结果对比")
    print("=" * 80)
    print(f"FNO (MSE):  {fno_final_loss:.6f}")
    print(f"PINO:       {pino_final_loss:.6f}")
    
    improvement = ((fno_final_loss - pino_final_loss) / fno_final_loss) * 100
    print(f"\n改进: {improvement:+.2f}%")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    diagnose_pino_loss()
    quick_training_test()
