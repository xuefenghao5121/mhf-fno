#!/usr/bin/env python3
"""
MHF-FNO 基础使用示例

展示如何在自定义数据上使用 MHF-FNO

作者: 天渠 (Tianqu) - 天渊团队
"""

import torch
import torch.nn as nn
import numpy as np

# 导入 MHF-FNO
from mhf_fno import create_mhf_fno_with_attention

print("=" * 70)
print("MHF-FNO 基础使用示例")
print("=" * 70)

# ============================================================
# 1. 创建模型
# ============================================================
print("\n[1] 创建 MHF-FNO 模型")
print("-" * 70)

# 推荐配置（Darcy/Burgers 数据集）
model = create_mhf_fno_with_attention(
    n_modes=(16, 16),      # 频率模式数量，通常为 resolution // 2
    hidden_channels=32,    # 隐藏层通道数
    in_channels=1,         # 输入通道数（如：1个物理量）
    out_channels=1,        # 输出通道数
    n_layers=3,            # FNO 层数
    mhf_layers=[0, 2],     # 哪些层使用 MHF（0=第一层，2=最后一层）
    n_heads=4,             # MHF 头的数量
    attention_layers=[0, -1]  # 哪些层使用跨头注意力
)

# 统计参数量
total_params = sum(p.numel() for p in model.parameters())
print(f"✅ 模型创建成功")
print(f"   参数量: {total_params:,}")
print(f"   配置: MHF layers=[0,2], Heads=4, Attention=[0,-1]")

# ============================================================
# 2. 准备数据
# ============================================================
print("\n[2] 准备示例数据")
print("-" * 70)

# 生成随机示例数据
batch_size = 8
resolution = 32
x = torch.randn(batch_size, 1, resolution, resolution)
y = torch.randn(batch_size, 1, resolution, resolution)

print(f"✅ 数据准备完成")
print(f"   输入形状: {x.shape}  # [batch, channels, height, width]")
print(f"   输出形状: {y.shape}")

# ============================================================
# 3. 前向传播
# ============================================================
print("\n[3] 前向传播测试")
print("-" * 70)

model.eval()
with torch.no_grad():
    y_pred = model(x)

print(f"✅ 前向传播成功")
print(f"   预测形状: {y_pred.shape}")
print(f"   预测范围: [{y_pred.min():.4f}, {y_pred.max():.4f}]")

# ============================================================
# 4. 训练示例
# ============================================================
print("\n[4] 训练循环示例")
print("-" * 70)

# 损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)

# 简单训练循环
model.train()
for epoch in range(3):  # 仅演示3个epoch
    optimizer.zero_grad()
    y_pred = model(x)
    loss = criterion(y_pred, y)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 1 == 0:
        print(f"   Epoch {epoch+1}/3, Loss: {loss.item():.6f}")

print(f"✅ 训练循环完成")

# ============================================================
# 5. 不同配置示例
# ============================================================
print("\n[5] 不同场景的推荐配置")
print("-" * 70)

# 5.1 Darcy/Burgers 激进配置
print("\n[Darcy/Burgers 激进配置]")
model_aggressive = create_mhf_fno_with_attention(
    n_modes=(16, 16),
    hidden_channels=32,
    mhf_layers=[0, 2],      # 首尾层使用 MHF
    n_heads=4,
    attention_layers=[0, 2]  # 首尾层使用注意力
)
params_agg = sum(p.numel() for p in model_aggressive.parameters())
print(f"   参数量: {params_agg:,}")
print(f"   特点: 参数减少 30-50%，性能提升 7-32%")

# 5.2 Navier-Stokes 保守配置
print("\n[Navier-Stokes 保守配置]")
model_conservative = create_mhf_fno_with_attention(
    n_modes=(16, 16),
    hidden_channels=32,
    mhf_layers=[0],         # 仅第一层使用 MHF
    n_heads=2,              # 较少的头
    attention_layers=[0]     # 仅第一层使用注意力
)
params_cons = sum(p.numel() for p in model_conservative.parameters())
print(f"   参数量: {params_cons:,}")
print(f"   特点: 参数减少 24%，性能持平")

# 5.3 基础 FNO（无 MHF）
print("\n[基础 FNO 配置（对比用）]")
model_fno = create_mhf_fno_with_attention(
    n_modes=(16, 16),
    hidden_channels=32,
    mhf_layers=[],          # 不使用 MHF
    attention_layers=[]      # 不使用注意力
)
params_fno = sum(p.numel() for p in model_fno.parameters())
print(f"   参数量: {params_fno:,}")
print(f"   特点: 标准 FNO，作为基准")

print(f"\n✅ 所有配置创建成功")

# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 70)
print("📋 使用总结")
print("=" * 70)
print("""
1. 导入模型:
   from mhf_fno import create_mhf_fno_with_attention

2. 创建模型:
   model = create_mhf_fno_with_attention(
       n_modes=(16, 16),
       hidden_channels=32,
       mhf_layers=[0, 2],  # 根据数据集调整
       n_heads=4
   )

3. 训练:
   optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
   loss = criterion(model(x), y)
   loss.backward()
   optimizer.step()

4. 配置选择:
   - Darcy/Burgers: mhf_layers=[0,2], n_heads=4 (激进)
   - Navier-Stokes: mhf_layers=[0], n_heads=2 (保守)

5. 完整文档:
   - README.md: 完整使用说明
   - QUICK_START.md: 快速开始指南
   - TIANYUAN_CONFIG.md: 详细配置文档
""")
print("=" * 70)
print("✅ 示例运行完成！")
print("=" * 70)
