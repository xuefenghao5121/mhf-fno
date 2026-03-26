#!/usr/bin/env python3
"""
简化版 PINO 测试 - 用于调试

只测试 MHF-PINO，减少 epochs
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from mhf_fno import MHFFNO
from mhf_fno.pino_high_freq import HighFreqPINOLoss
from neuralop.models import FNO
from neuralop.losses.data_losses import LpLoss

print("✓ 导入成功")

# 加载数据
data_path = Path(__file__).parent.parent / 'data' / 'ns_train_32.pt'
print(f"加载数据: {data_path}")
data = torch.load(data_path, weights_only=False)
train_x = data.get('x', data.get('train_x'))[:500].unsqueeze(1).float()
train_y = data.get('y', data.get('train_y'))[:500].unsqueeze(1).float()
test_x = train_x[400:500]
test_y = train_y[400:500]
train_x = train_x[:400]

print(f"✓ 数据加载成功: train {train_x.shape}, test {test_x.shape}")

# 测试 FNO
print("\n" + "="*60)
print("测试 FNO (10 epochs)")
print("="*60)

torch.manual_seed(42)
model_fno = FNO(n_modes=(16,16), hidden_channels=32, in_channels=1, out_channels=1, n_layers=3)
print(f"参数量: {sum(p.numel() for p in model_fno.parameters()):,}")

optimizer = torch.optim.Adam(model_fno.parameters(), lr=0.0005)
loss_fn = LpLoss(d=2, p=2, reduction='mean')

for epoch in range(10):
    model_fno.train()
    perm = torch.randperm(400)
    for i in range(0, 400, 32):
        bx = train_x[perm[i:i+32]]
        by = train_y[perm[i:i+32]]
        optimizer.zero_grad()
        pred = model_fno(bx)
        loss = loss_fn(pred, by)
        loss.backward()
        optimizer.step()
    
    model_fno.eval()
    with torch.no_grad():
        test_loss = loss_fn(model_fno(test_x), test_y).item()
    
    print(f"Epoch {epoch+1}/10: Test Loss = {test_loss:.4f}")

# 测试 MHF-PINO
print("\n" + "="*60)
print("测试 MHF-PINO (10 epochs)")
print("="*60)

torch.manual_seed(42)
model_pino = MHFFNO.best_config(n_modes=(16,16), hidden_channels=32, in_channels=1, out_channels=1)
print(f"参数量: {sum(p.numel() for p in model_pino.parameters()):,}")

pino_loss_fn = HighFreqPINOLoss(lambda_physics=0.0001, freq_threshold=0.5)
optimizer = torch.optim.Adam(model_pino.parameters(), lr=0.0005)

for epoch in range(10):
    model_pino.train()
    perm = torch.randperm(400)
    for i in range(0, 400, 32):
        bx = train_x[perm[i:i+32]]
        by = train_y[perm[i:i+32]]
        optimizer.zero_grad()
        pred = model_pino(bx)
        loss = pino_loss_fn(pred, by)
        loss.backward()
        optimizer.step()
    
    model_pino.eval()
    with torch.no_grad():
        test_loss = loss_fn(model_pino(test_x), test_y).item()
    
    print(f"Epoch {epoch+1}/10: Test Loss = {test_loss:.4f}")

print("\n✅ 简化测试完成")
