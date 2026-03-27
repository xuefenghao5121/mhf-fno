#!/usr/bin/env python3
"""
MHF-FNO 快速测试最佳配置

快速测试 MHF+CoDA+PINO 组合的最佳性能配置
用于验证核心功能和快速基准测试
"""

import torch
import sys
from pathlib import Path

# 添加路径
sys.path.insert(0, str(Path(__file__).parent / 'mhf_fno'))

from mhf_fno import MHFFNO
from data_loader import load_dataset

print("=" * 70)
print("MHF-FNO 快速测试")
print("=" * 70)

# ============================================================================
# 配置
# ============================================================================

DATASET = 'darcy'
DATA_FORMAT = 'pt'
RESOLUTION = 64

# MHF+CoDA+PINO 最佳配置
CONFIG = {
    'hidden_channels': 32,
    'n_layers': 4,
    'n_modes': (16, 16),
    'mhf_layers': [0, 2],  # 第1层和第3层使用MHF
    'n_heads': 2,
    'use_coda': True,
    'use_pino': True,
    'pino_weight': 0.1,
    'epochs': 10,
    'batch_size': 16,
    'learning_rate': 1e-3,
}

# ============================================================================
# 加载数据
# ============================================================================

print(f"\n加载数据集: {DATASET}")
print(f"分辨率: {RESOLUTION}x{RESOLUTION}")

try:
    train_x, train_y, test_x, test_y, info = load_dataset(
        dataset_name=DATASET,
        data_format=DATA_FORMAT,
        n_train=100,
        n_test=20,
        resolution=RESOLUTION
    )
    print(f"✅ 加载成功: {info}")
except Exception as e:
    print(f"❌ 数据加载失败: {e}")
    sys.exit(1)

# ============================================================================
# 创建模型
# ============================================================================

print(f"\n创建 MHF-FNO 模型:")
print(f"  隐藏通道: {CONFIG['hidden_channels']}")
print(f"  层数: {CONFIG['n_layers']}")
print(f"  MHF 层: {CONFIG['mhf_layers']}")
print(f"  CoDA: {CONFIG['use_coda']}")
print(f"  PINO: {CONFIG['use_pino']}")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"  设备: {device}")

try:
    model = MHFFNO(
        in_channels=train(info['input_channels']),
        out_channels=train_y.shape[1],
        hidden_channels=CONFIG['hidden_channels'],
        n_modes=CONFIG['n_modes'],
        n_layers=CONFIG['n_layers'],
        mhf_layers=CONFIG['mhf_layers'],
        n_heads=CONFIG['n_heads'],
        use_coda=CONFIG['use_coda'],
    ).to(device)
    print(f"✅ 模型创建成功")
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")
except Exception as e:
    print(f"❌ 模型创建失败: {e}")
    sys.exit(1)

# ============================================================================
# 快速训练
# ============================================================================

print(f"\n快速训练 ({CONFIG['epochs']} epochs)")
print("-" * 70)

optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])

for epoch in range(CONFIG['epochs']):
    model.train()
    train_loss = 0.0
    
    for i in range(0, len(train_x), CONFIG['batch_size']):
        batch_x = train_x[i:i+CONFIG['batch_size']].to(device)
        batch_y = train_y[i:i+CONFIG['batch_size']].to(device)
        
        optimizer.zero_grad()
        pred = model(batch_x)
        
        # 基础损失
        loss = torch.nn.functional.mse_loss(pred, batch_y)
        
        # PINO 约束
        if CONFIG['use_pino']:
            pino_loss = model.pino_loss(batch_x, pred)
            loss = loss + CONFIG['pino_weight'] * pino_loss
        
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    train_loss /= (len(train_x) // CONFIG['batch_size'])
    
    # 测试
    model.eval()
    with torch.no_grad():
        test_pred = model(test_x.to(device))
        test_loss = torch.nn.functional.mse_loss(test_pred, test_y.to(device)).item()
    
    print(f"Epoch {epoch+1:2d}/{CONFIG['epochs']}: Train={train_loss:.6f}, Test={test_loss:.6f}")

# ============================================================================
# 结果汇总
# ============================================================================

print("\n" + "=" * 70)
print("测试完成!")
print("=" * 70)
print(f"\n最终测试损失: {test_loss:.6f}")
print(f"模型配置: MHF+CoDA+PINO")
print(f"M层数: {CONFIG['mhf_layers']}")
print(f"头数: {CONFIG['n_heads']}")
print(f"设备: {device}")
