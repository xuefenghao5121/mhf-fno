#!/usr/bin/env python3
"""
最简化测试 - 只测试 loss 函数
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from mhf_fno.pino_high_freq import HighFreqPINOLoss

print("测试高频噪声惩罚 PINO loss...")

# 创建测试数据
u_pred = torch.randn(4, 1, 32, 32)
u_true = torch.randn(4, 1, 32, 32)

# 测试 loss
loss_fn = HighFreqPINOLoss(lambda_physics=0.0001, freq_threshold=0.5)
print(f"✓ Loss函数创建成功")

loss = loss_fn(u_pred, u_true)
print(f"✓ Loss计算成功: {loss.item():.6f}")

# 分解查看
L_data = torch.nn.functional.mse_loss(u_pred, u_true)
L_high_freq = loss_fn.compute_high_freq_penalty(u_pred)
print(f"  L_data: {L_data.item():.6f}")
print(f"  L_high_freq: {L_high_freq.item():.6f}")
print(f"  Total: {L_data.item() + 0.0001 * L_high_freq.item():.6f}")

print("\n✅ Loss函数测试通过")
