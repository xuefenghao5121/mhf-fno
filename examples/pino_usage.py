#!/usr/bin/env python3
"""
PINO 物理约束使用示例（实验性）

注意: 当前 PINO 实现需要时间序列数据才能发挥最佳效果

作者: 天渠 (Tianqu) - 天渊团队
"""

import torch
import torch.nn as nn
import numpy as np

# 导入 MHF-FNO 和 PINO
from mhf_fno import create_mhf_fno_with_attention
from mhf_fno.pino_high_freq import HighFreqPINOLoss

print("=" * 70)
print("PINO 物理约束使用示例（实验性）")
print("=" * 70)

# ============================================================
# 重要说明
# ============================================================
print("\n⚠️  PINO 使用注意事项")
print("-" * 70)
print("""
1. 当前实现基于高频约束（High-Frequency PINO）
2. 适用于静态场数据（如 Darcy Flow）
3. 对于时间序列数据（如 NS），建议实现完整的残差约束
4. lambda_physics 参数需要调优（建议从 1e-4 开始）
5. PINO 效果依赖于数据质量和物理方程的准确性
""")

# ============================================================
# 1. 创建模型
# ============================================================
print("\n[1] 创建带 PINO 的 MHF-FNO 模型")
print("-" * 70)

model = create_mhf_fno_with_attention(
    n_modes=(16, 16),
    hidden_channels=32,
    in_channels=1,
    out_channels=1,
    n_layers=3,
    mhf_layers=[0, 2],
    n_heads=4,
    attention_layers=[0, -1]
)

print(f"✅ 模型创建成功")
print(f"   参数量: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================
# 2. 创建 PINO 损失
# ============================================================
print("\n[2] 创建 PINO 物理约束损失")
print("-" * 70)

# PINO 损失配置
pino_loss = HighFreqPINOLoss(
    lambda_physics=0.0001,    # 物理损失权重（需要调优）
    threshold_ratio=0.5       # 高频阈值比例
)

print(f"✅ PINO 损失创建成功")
print(f"   lambda_physics: {pino_loss.lambda_physics}")
print(f"   threshold_ratio: {pino_loss.threshold_ratio}")

# ============================================================
# 3. 准备数据
# ============================================================
print("\n[3] 准备示例数据")
print("-" * 70)

# 生成示例数据（Darcy Flow 风格）
batch_size = 16
resolution = 32

# 输入：扩散系数场 a(x)
x = torch.randn(batch_size, 1, resolution, resolution)
# 简单平滑
x = torch.nn.functional.avg_pool2d(x, 3, stride=1, padding=1)

# 输出：压力场 u(x)（简化）
y = torch.nn.functional.avg_pool2d(x, 3, stride=1, padding=1)

print(f"✅ 数据准备完成")
print(f"   输入形状: {x.shape}")
print(f"   输出形状: {y.shape}")

# ============================================================
# 4. 训练循环（带 PINO）
# ============================================================
print("\n[4] 训练循环（数据损失 + PINO 物理损失）")
print("-" * 70)

# 损失函数和优化器
data_criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)

# 训练参数
epochs = 10

model.train()
for epoch in range(epochs):
    optimizer.zero_grad()
    
    # 前向传播
    y_pred = model(x)
    
    # 计算损失
    data_loss = data_criterion(y_pred, y)
    physics_loss = pino_loss(y_pred)
    total_loss = data_loss + physics_loss
    
    # 反向传播
    total_loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 2 == 0:
        print(f"   Epoch {epoch+1}/{epochs}")
        print(f"     数据损失: {data_loss.item():.6f}")
        print(f"     物理损失: {physics_loss.item():.6f}")
        print(f"     总损失:   {total_loss.item():.6f}")

print(f"\n✅ 训练完成")

# ============================================================
# 5. 测试不同 lambda_physics 值
# ============================================================
print("\n[5] 测试不同 lambda_physics 值")
print("-" * 70)

lambda_values = [0.0, 1e-5, 1e-4, 1e-3]
results = []

for lambda_phys in lambda_values:
    pino = HighFreqPINOLoss(lambda_physics=lambda_phys)
    
    model.eval()
    with torch.no_grad():
        y_pred = model(x)
        data_loss = data_criterion(y_pred, y).item()
        phys_loss = pino(y_pred).item()
        total = data_loss + phys_loss
    
    results.append({
        'lambda': lambda_phys,
        'data_loss': data_loss,
        'physics_loss': phys_loss,
        'total': total
    })
    
    print(f"   λ={lambda_phys:.1e}: Data={data_loss:.6f}, Physics={phys_loss:.6f}, Total={total:.6f}")

print(f"\n✅ 测试完成")

# ============================================================
# 6. PINO 效果分析
# ============================================================
print("\n[6] PINO 效果分析")
print("-" * 70)

# 找到最佳 lambda
best_result = min(results, key=lambda r: r['total'])
print(f"最佳 lambda_physics: {best_result['lambda']:.1e}")
print(f"对应总损失: {best_result['total']:.6f}")

# 与纯数据驱动对比
pure_data_result = results[0]  # lambda=0
improvement = (pure_data_result['total'] - best_result['total']) / pure_data_result['total'] * 100

print(f"\n相比纯数据驱动:")
print(f"   损失改善: {improvement:+.2f}%")

# ============================================================
# 7. 高级用法：自适应 lambda
# ============================================================
print("\n[7] 高级用法：自适应 lambda 调度")
print("-" * 70)

class AdaptivePINOTrainer:
    """自适应 PINO 训练器"""
    
    def __init__(self, model, initial_lambda=1e-4):
        self.model = model
        self.lambda_physics = initial_lambda
        self.pino_loss = HighFreqPINOLoss(lambda_physics=self.lambda_physics)
    
    def update_lambda(self, epoch, max_epochs):
        """根据训练进度调整 lambda"""
        # 前期小权重，后期逐渐增大
        progress = epoch / max_epochs
        self.lambda_physics = 1e-5 + progress * 1e-3
        self.pino_loss.lambda_physics = self.lambda_physics
        return self.lambda_physics

# 示例
trainer = AdaptivePINOTrainer(model)
print("自适应 lambda 调度示例:")
for epoch in [0, 10, 20, 30, 40, 49]:
    lambda_val = trainer.update_lambda(epoch, 50)
    print(f"   Epoch {epoch:2d}: λ = {lambda_val:.6f}")

# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 70)
print("📋 PINO 使用总结")
print("=" * 70)
print("""
1. 创建 PINO 损失:
   from mhf_fno.pino_high_freq import HighFreqPINOLoss
   pino_loss = HighFreqPINOLoss(lambda_physics=1e-4)

2. 训练循环:
   y_pred = model(x)
   data_loss = criterion(y_pred, y)
   physics_loss = pino_loss(y_pred)
   total_loss = data_loss + physics_loss

3. 参数调优:
   - lambda_physics: 从 1e-4 开始，根据效果调整
   - threshold_ratio: 通常 0.5 即可
   - 建议使用验证集选择最佳 lambda

4. 适用场景:
   ✅ 静态场问题（Darcy, 热传导）
   ✅ 有物理约束需求的问题
   ⚠️  时间序列问题（需要更复杂的残差计算）
   ⚠️  数据质量差的问题（PINO 可能引入偏差）

5. 高级技巧:
   - 自适应 lambda 调度
   - 多阶段训练（先数据驱动，后加 PINO）
   - 结合其他正则化方法

6. 完整示例:
   - examples/ns_real_data.py: NS 方程完整示例
   - docs/paper-notes/pino-literature.md: PINO 理论
""")
print("=" * 70)
print("✅ PINO 示例运行完成！")
print("=" * 70)
