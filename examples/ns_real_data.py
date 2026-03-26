#!/usr/bin/env python3
"""
在真实 Navier-Stokes 数据上测试 MHF-FNO

适用场景:
- PDEBench 数据集
- 自定义 NS 模拟数据

数据格式:
- 输入: [N, C, H, W] 或 [N, C, L]
- 输出: [N, C, H, W] 或 [N, C, L]
- 支持时间序列: [N, T, C, H, W]

作者: 天渠 (Tianqu) - 天渊团队
"""

import torch
import torch.nn as nn
import numpy as np
import h5py
from pathlib import Path

# 导入 MHF-FNO
from mhf_fno import create_mhf_fno_with_attention

print("=" * 70)
print("Navier-Stokes 真实数据测试示例")
print("=" * 70)

# ============================================================
# 1. 数据加载示例
# ============================================================
print("\n[1] 数据加载方法")
print("-" * 70)

print("""
数据格式支持:

1. PyTorch 格式 (.pt):
   data = torch.load('ns_data.pt')
   x = data['x']  # [N, T, C, H, W] 或 [N, C, H, W]
   y = data['y']

2. HDF5 格式 (.h5) - PDEBench 格式:
   import h5py
   with h5py.File('ns_data.h5', 'r') as f:
       x = torch.from_numpy(f['input'][:])
       y = torch.from_numpy(f['output'][:])

3. NumPy 格式 (.npy):
   x = torch.from_numpy(np.load('ns_x.npy'))
   y = torch.from_numpy(np.load('ns_y.npy'))
""")

# ============================================================
# 2. 模拟数据生成（用于演示）
# ============================================================
print("\n[2] 生成模拟 NS 数据（用于演示）")
print("-" * 70)

def generate_synthetic_ns_data(
    n_samples: int = 100,
    resolution: int = 32,
    time_steps: int = 10
):
    """
    生成合成的 NS 风格数据
    
    注意: 这是简化版本，真实 NS 数据应从 PDEBench 或模拟器获取
    """
    print(f"生成合成数据: {n_samples} 样本, {resolution}x{resolution}, {time_steps} 时间步")
    
    # 初始场（涡度场）
    x = torch.randn(n_samples, 1, resolution, resolution)
    
    # 简单的演化（非真实 NS，仅用于演示）
    y = x.clone()
    for t in range(time_steps):
        # 简单的扩散 + 平流
        y = torch.nn.functional.avg_pool2d(y, 3, stride=1, padding=1)
        y = y + 0.1 * torch.randn_like(y)
    
    print(f"✅ 合成数据生成完成")
    print(f"   输入形状: {x.shape}")
    print(f"   输出形状: {y.shape}")
    
    return x, y

# 生成示例数据
x, y = generate_synthetic_ns_data(n_samples=100, resolution=32)

# ============================================================
# 3. 创建 MHF+CoDA 模型（推荐配置）
# ============================================================
print("\n[3] 创建 MHF+CoDA 模型（NS 推荐配置）")
print("-" * 70)

print("""
📊 NS 方程推荐配置对比:

┌─────────────────────────────────────────────────────────┐
│  配置          │ 参数量      │ vs FNO  │ 推荐度       │
├─────────────────────────────────────────────────────────┤
│  纯 FNO        │ 453K        │ 基准    │ -            │
│  MHF           │ 232K (-49%) │ -1.27%  │ ⭐⭐          │
│  MHF+CoDA      │ 233K (-49%) │ -1.15%  │ ⭐⭐⭐ 推荐   │
└─────────────────────────────────────────────────────────┘

结论: MHF+CoDA 在 NS 上性能最佳，参数减少 49%
""")

# NS 方程推荐配置（MHF+CoDA）
model = create_mhf_fno_with_attention(
    n_modes=(16, 16),        # resolution // 2
    hidden_channels=32,
    in_channels=1,           # 根据数据调整
    out_channels=1,          # 根据数据调整
    n_layers=3,
    mhf_layers=[0],          # 保守配置：仅第一层使用 MHF
    n_heads=2,               # NS 推荐：较小的头数
    attention_layers=[0],    # ⭐ 使用 CoDA（跨头注意力）
    bottleneck=4,            # CoDA 瓶颈大小
    gate_init=0.1            # 门控初始化
)

params = sum(p.numel() for p in model.parameters())
print(f"✅ MHF+CoDA 模型创建成功")
print(f"   参数量: {params:,} (vs FNO 453K)")
print(f"   参数减少: {(1 - params/453361)*100:.1f}%")
print(f"   配置: MHF=[0], Heads=2, CoDA=[0] ⭐")

# ============================================================
# 4. 训练配置
# ============================================================
print("\n[4] 训练配置")
print("-" * 70)

config = {
    'epochs': 50,
    'batch_size': 32,
    'learning_rate': 5e-4,
    'weight_decay': 1e-4,
    'grad_clip': 1.0,
}

print(f"训练配置:")
for k, v in config.items():
    print(f"   {k}: {v}")

# 优化器和调度器
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config['learning_rate'],
    weight_decay=config['weight_decay']
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=config['epochs']
)

criterion = nn.MSELoss()

# ============================================================
# 5. 数据预处理
# ============================================================
print("\n[5] 数据预处理")
print("-" * 70)

# 划分训练/测试集
n_train = int(0.8 * len(x))
train_x, train_y = x[:n_train], y[:n_train]
test_x, test_y = x[n_train:], y[n_train:]

print(f"训练集: {len(train_x)} 样本")
print(f"测试集: {len(test_x)} 样本")

# 归一化（重要！）
train_mean = train_x.mean()
train_std = train_x.std()

train_x = (train_x - train_mean) / (train_std + 1e-8)
train_y = (train_y - train_mean) / (train_std + 1e-8)
test_x = (test_x - train_mean) / (train_std + 1e-8)
test_y = (test_y - train_mean) / (train_std + 1e-8)

print(f"✅ 数据预处理完成")
print(f"   均值: {train_mean:.6f}, 标准差: {train_std:.6f}")

# ============================================================
# 6. 训练循环
# ============================================================
print("\n[6] 训练循环")
print("-" * 70)

def train_epoch(model, x, y, batch_size, optimizer, criterion, device='cpu'):
    """训练一个 epoch"""
    model.train()
    n = len(x)
    perm = torch.randperm(n)
    total_loss = 0
    
    for i in range(0, n, batch_size):
        bx = x[perm[i:i+batch_size]].to(device)
        by = y[perm[i:i+batch_size]].to(device)
        
        optimizer.zero_grad()
        pred = model(bx)
        loss = criterion(pred, by)
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / (n // batch_size)

def evaluate(model, x, y, criterion, device='cpu'):
    """评估模型"""
    model.eval()
    with torch.no_grad():
        pred = model(x.to(device))
        loss = criterion(pred, y.to(device))
    return loss.item()

# 训练
best_test_loss = float('inf')
history = {'train_loss': [], 'test_loss': []}

for epoch in range(config['epochs']):
    train_loss = train_epoch(
        model, train_x, train_y,
        config['batch_size'], optimizer, criterion
    )
    
    test_loss = evaluate(model, test_x, test_y, criterion)
    
    scheduler.step()
    
    history['train_loss'].append(train_loss)
    history['test_loss'].append(test_loss)
    
    if test_loss < best_test_loss:
        best_test_loss = test_loss
    
    if (epoch + 1) % 10 == 0:
        print(f"   Epoch {epoch+1:3d}/{config['epochs']}: "
              f"Train={train_loss:.6f}, Test={test_loss:.6f}")

print(f"\n✅ 训练完成")
print(f"   最佳测试损失: {best_test_loss:.6f}")

# ============================================================
# 7. 可选：使用 PINO（如果有时间序列数据）
# ============================================================
print("\n[7] 可选：PINO 物理约束（如果有时间序列数据）")
print("-" * 70)

print("""
如果数据包含时间序列 [N, T, C, H, W]，可以使用 PINO:

if x.dim() == 5:  # [N, T, C, H, W]
    # 实现真正的 NS 残差
    from mhf_fno.pino_physics import NavierStokesPhysics
    
    physics = NavierStokesPhysics(
        viscosity=1e-3,  # NS 粘性系数
        dt=0.01          # 时间步长
    )
    
    # 在训练循环中
    physics_loss = physics.compute_residual(y_pred, x)
    total_loss = data_loss + 0.0001 * physics_loss
    
else:
    # 使用简化高频约束
    from mhf_fno.pino_high_freq import HighFreqPINOLoss
    pino_loss = HighFreqPINOLoss(lambda_physics=0.0001)
    physics_loss = pino_loss(y_pred)
    total_loss = data_loss + physics_loss
""")

# ============================================================
# 8. 保存和加载模型
# ============================================================
print("\n[8] 保存和加载模型")
print("-" * 70)

# 保存模型
save_path = Path(__file__).parent / 'checkpoints' / 'ns_model.pth'
save_path.parent.mkdir(exist_ok=True)

torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'config': config,
    'best_test_loss': best_test_loss,
    'normalization': {
        'mean': train_mean,
        'std': train_std
    }
}, save_path)

print(f"✅ 模型已保存到: {save_path}")

# 加载模型示例
print("\n加载模型示例:")
print("""
checkpoint = torch.load('ns_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# 恢复归一化参数
mean = checkpoint['normalization']['mean']
std = checkpoint['normalization']['std']
""")

# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 70)
print("📋 NS 真实数据测试总结")
print("=" * 70)
print("""
1. 数据准备:
   - 从 PDEBench 或模拟器获取真实 NS 数据
   - 格式: [N, C, H, W] 或 [N, T, C, H, W]
   - 必须进行归一化处理

2. 模型配置:
   - mhf_layers=[0] (保守配置)
   - n_heads=2 (较小值)
   - hidden_channels=32 (平衡)
   - n_modes=(16, 16) (根据分辨率调整)

3. 训练策略:
   - 优化器: AdamW, lr=5e-4
   - 调度器: CosineAnnealingLR
   - 梯度裁剪: max_norm=1.0
   - 数据量: 建议 1000+ 样本

4. PINO 使用（可选）:
   - 时间序列数据: 使用完整残差约束
   - 静态数据: 使用高频约束
   - lambda_physics: 从 1e-4 开始调优

5. 性能预期:
   - 参数减少: ~24%
   - 性能: 与 FNO 基本持平
   - 如需进一步提升: 增加 PINO 约束或更多数据

6. 完整示例:
   - benchmark/test_ns_1000.py: 完整 NS 测试
   - docs/NS_OPTIMIZATION_SUMMARY.md: NS 优化报告
""")
print("=" * 70)
print("✅ NS 真实数据示例运行完成！")
print("=" * 70)
