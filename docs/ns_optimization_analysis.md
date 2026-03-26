# NS 方程优化方案分析

> **分析时间**: 2026-03-26
> **分析者**: 天渊（架构师）
> **目标**: 在不增加数据量的前提下，改善 MHF-FNO 在 Navier-Stokes 方程上的性能

---

## 一、问题根源分析

### 1.1 当前表现

| 数据集 | PDE 类型 | vs FNO | 参数减少 |
|--------|----------|--------|----------|
| Darcy Flow | 椭圆型 | **+8.17%** ✅ | 48.6% |
| Burgers | 抛物型 | **+32.12%** ✅✅ | 31.7% |
| **Navier-Stokes** | **双曲型** | **-1.27%** ⚠️ | 48.8% |

### 1.2 矛盾点

**MHF 核心假设**: 频率解耦，各头独立处理频率子空间

**NS 方程特性**: 
- 强频率耦合（Kolmogorov 能量级联）
- 非线性对流项 `(u·∇)u` 产生跨频率交互
- 湍流的多尺度特性

**结论**: MHF 的频率解耦假设与 NS 的强耦合特性存在根本冲突

---

## 二、优化方案全景

### 2.1 综合评估表

| 方向 | 预期提升 | 实现难度 | 代码修改量 | 推荐度 | 效果/成本比 |
|------|----------|----------|------------|--------|-------------|
| **1. PINO 物理约束** | +3-5% | 中 | ~100 行 | ⭐⭐⭐⭐⭐ | **最高** |
| **2. 模型配置优化** | +1-3% | 低 | ~20 行 | ⭐⭐⭐⭐⭐ | **极高** |
| **3. 训练策略优化** | +1-2% | 低 | ~30 行 | ⭐⭐⭐⭐ | 高 |
| **4. 架构改进** | +2-4% | 高 | ~200 行 | ⭐⭐⭐ | 中 |
| **5. 数据增强** | +1-2% | 中 | ~50 行 | ⭐⭐⭐ | 中 |

### 2.2 优先级排序（按效果/成本比）

1. **🥇 模型配置优化**（立即可行）
   - 效果: +1-3%
   - 成本: 仅需修改配置参数
   - 时间: 5 分钟

2. **🥈 PINO 物理约束**（强烈推荐）
   - 效果: +3-5%
   - 成本: 中等（需实现 NS 残差计算）
   - 时间: 1-2 小时

3. **🥉 训练策略优化**（低风险）
   - 效果: +1-2%
   - 成本: 低（修改训练脚本）
   - 时间: 10 分钟

4. **架构改进**（长期优化）
   - 效果: +2-4%
   - 成本: 高（需要设计新模块）
   - 时间: 1-2 天

5. **数据增强**（辅助手段）
   - 效果: +1-2%
   - 成本: 中等
   - 时间: 30 分钟

---

## 三、详细方案

### 方向 1: PINO 物理约束 ⭐⭐⭐⭐⭐

#### 3.1.1 原理

将 NS 方程的物理约束作为正则化项加入损失函数：

```
L_total = L_data + λ × L_physics

其中：
- L_data: 数据损失（预测 vs 真实）
- L_physics: NS 方程残差
  ∂u/∂t + (u·∇)u + ∇p - ν∇²u = 0
```

#### 3.1.2 为什么有效？

| 优势 | 说明 |
|------|------|
| **弥补 MHF 不足** | 物理约束强制模型遵守频率耦合规律 |
| **无需更多数据** | 从方程本身获得监督信号 |
| **泛化性强** | 物理规律在任何数据上都适用 |
| **可解释** | 损失函数直接反映物理准确性 |

#### 3.1.3 实现方案

```python
class PINOLoss(nn.Module):
    """PINO 物理约束损失"""
    
    def __init__(self, viscosity=1e-3, lambda_physics=0.1):
        super().__init__()
        self.nu = viscosity
        self.lambda_p = lambda_physics
    
    def compute_ns_residual(self, u, pred_u):
        """
        计算 NS 方程残差
        
        NS: ∂u/∂t + (u·∇)u = -∇p + ν∇²u
        
        简化版（不可压缩）：
        ∂u/∂t + (u·∇)u - ν∇²u = 0
        """
        # 时间导数（假设输入 u 是 t 时刻，pred_u 是 t+dt 时刻）
        # dt 需要从数据中获取
        du_dt = (pred_u - u) / self.dt
        
        # 空间导数（使用有限差分）
        # ∇u
        grad_u = self.compute_gradient(pred_u)
        
        # 对流项 (u·∇)u
        convection = torch.sum(u * grad_u, dim=1, keepdim=True)
        
        # 拉普拉斯项 ∇²u
        laplacian = self.compute_laplacian(pred_u)
        diffusion = self.nu * laplacian
        
        # NS 残差
        residual = du_dt + convection - diffusion
        
        return residual
    
    def compute_gradient(self, u):
        """计算空间梯度（中心差分）"""
        # u: [B, C, H, W]
        dx = (torch.roll(u, -1, dims=-1) - torch.roll(u, 1, dims=-1)) / 2
        dy = (torch.roll(u, -1, dims=-2) - torch.roll(u, 1, dims=-2)) / 2
        return torch.stack([dx, dy], dim=-1)
    
    def compute_laplacian(self, u):
        """计算拉普拉斯算子"""
        lap = (
            torch.roll(u, 1, dims=-1) + 
            torch.roll(u, -1, dims=-1) +
            torch.roll(u, 1, dims=-2) + 
            torch.roll(u, -1, dims=-2) -
            4 * u
        )
        return lap
    
    def forward(self, pred, target, input_u):
        # 数据损失
        loss_data = F.mse_loss(pred, target)
        
        # 物理损失
        residual = self.compute_ns_residual(input_u, pred)
        loss_physics = torch.mean(residual ** 2)
        
        # 总损失
        return loss_data + self.lambda_p * loss_physics
```

#### 3.1.4 修改点

| 文件 | 修改内容 | 行数 |
|------|----------|------|
| `losses.py` | 新增 PINOLoss 类 | ~80 行 |
| `run_benchmarks.py` | 替换损失函数 | ~10 行 |
| `config.py` | 添加 λ 参数 | ~5 行 |

**总代码量**: ~100 行

#### 3.1.5 预期效果

- **短期**（λ=0.1, 10 epochs 测试）: +1-2%
- **中期**（λ 调优, 50 epochs）: +3-5%
- **长期**（自适应 λ, 梯度平衡）: +5-8%

---

### 方向 2: 模型配置优化 ⭐⭐⭐⭐⭐（立即可行）

#### 3.2.1 减少MHF 层数

**原理**: NS 方程需要中间层的标准卷积来处理频率耦合

```python
# 当前配置（效果 -1.27%）
mhf_layers = [0, 2]  # 首尾层

# 方案 A: 只用第一层
mhf_layers = [0]     # 保留更多标准卷积

# 方案 B: 全标准卷积（作为对照组）
mhf_layers = []      # 无 MHF，纯 FNO + CoDA
```

**预期效果**: +0.5-1.5%

#### 3.2.2 减少头数

**原理**: 更粗粒度的频率分解，减少解耦程度

```python
# 当前配置
n_heads = 4  # 细粒度分解

# 优化配置
n_heads = 2  # 粗粒度分解，保留更多频率交互
```

**预期效果**: +0.5-1%

#### 3.2.3 增强 CoDA 瓶颈

**原理**: 更强的跨头信息交换能力

```python
# 当前配置
bottleneck = 4
gate_init = 0.1

# 优化配置
bottleneck = 8       # 增加瓶颈容量
gate_init = 0.05     # 降低初始门控，让注意力逐渐增强
```

**预期效果**: +0.5-1%

#### 3.2.4 完整配置对比

```python
# 当前配置
model = MHFFNOWithAttention.best_config(
    n_modes=(12, 12),
    hidden_channels=32,
    n_heads=4,
    mhf_layers=[0, 2],
    bottleneck=4,
    gate_init=0.1
)

# 优化配置（NS 专用）
model = MHFFNOWithAttention.best_config(
    n_modes=(12, 12),
    hidden_channels=32,
    n_heads=2,              # 减少头数
    mhf_layers=[0],         # 只用第一层
    bottleneck=8,           # 增强瓶颈
    gate_init=0.05          # 降低初始门控
)
```

#### 3.2.5 修改点

| 文件 | 修改内容 | 行数 |
|------|----------|------|
| `run_benchmarks.py` | 修改模型配置 | ~5 行 |
| `mhf_attention.py` | 添加 bottleneck 参数 | ~10 行 |

**总代码量**: ~20 行

---

### 方向 3: 训练策略优化 ⭐⭐⭐⭐

#### 3.3.1 学习率调度

```python
# 当前
optimizer = Adam(lr=1e-3)
scheduler = CosineAnnealingLR(T_max=epochs)

# 优化方案 A: OneCycleLR（推荐）
optimizer = Adam(lr=5e-4)
scheduler = OneCycleLR(
    optimizer,
    max_lr=1e-3,
    epochs=epochs,
    steps_per_epoch=len(train_loader),
    pct_start=0.3  # 30% warmup
)

# 优化方案 B: WarmUp + CosineDecay
scheduler = SequentialScheduler([
    LinearWarmUp(optimizer, start_lr=1e-6, end_lr=1e-3, epochs=5),
    CosineAnnealingLR(optimizer, T_max=epochs-5)
])
```

**预期效果**: +0.5-1%

#### 3.3.2 正则化

```python
# 早停
early_stopping = True
patience = 10

# 权重衰减
optimizer = Adam(lr=1e-3, weight_decay=1e-4)

# Dropout（在 CoDA 中已支持）
attn_dropout = 0.1
```

**预期效果**: +0.3-0.5%

#### 3.3.3 梯度裁剪

```python
# 防止梯度爆炸（NS 训练中常见）
grad_clip = 1.0

# 在训练循环中
optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
optimizer.step()
```

**预期效果**: +0.2-0.5%

#### 3.3.4 修改点

| 文件 | 修改内容 | 行数 |
|------|----------|------|
| `run_benchmarks.py` | 修改训练配置 | ~20 行 |
| `run_benchmarks.py` | 添加早停逻辑 | ~10 行 |

**总代码量**: ~30 行

---

### 方向 4: 架构改进 ⭐⭐⭐

#### 3.4.1 残差连接增强

```python
class MHFSpectralConvEnhanced(MHFSpectralConvWithAttention):
    """增强残差连接的 MHF 卷积"""
    
    def forward(self, x):
        # 保存原始输入
        residual = x
        
        # MHF 卷积
        mhf_out = super().forward(x)
        
        # 增强残差：混合原始输入和MHF 输出
        # 对于 NS，给予原始输入更高权重
        out = 0.6 * residual + 0.4 * mhf_out
        
        return out
```

**预期效果**: +0.5-1%

#### 3.4.2 频域注意力

```python
class FrequencyDomainAttention(nn.Module):
    """频域自注意力机制"""
    
    def __init__(self, n_modes, hidden_channels):
        super().__init__()
        self.freq_attn = nn.MultiheadAttention(
            embed_dim=hidden_channels,
            num_heads=4,
            batch_first=True
        )
    
    def forward(self, x_freq):
        # x_freq: [B, C, H, W] (频域)
        B, C, H, W = x_freq.shape
        
        # 展平空间维度
        x_flat = x_freq.view(B, C, -1).transpose(1, 2)  # [B, H*W, C]
        
        # 自注意力
        attn_out, _ = self.freq_attn(x_flat, x_flat, x_flat)
        
        # 恢复形状
        out = attn_out.transpose(1, 2).view(B, C, H, W)
        
        return out
```

**预期效果**: +1-2%

#### 3.4.3 多尺度融合

```python
class MultiScaleFNO(nn.Module):
    """多尺度 FNO"""
    
    def __init__(self, scales=[16, 32, 64]):
        super().__init__()
        self.scales = scales
        self.fnos = nn.ModuleList([
            FNO(n_modes=(s//4, s//4), hidden_channels=32)
            for s in scales
        ])
        self.fusion = nn.Conv2d(32 * len(scales), 32, 1)
    
    def forward(self, x):
        multi_scale_features = []
        
        for scale, fno in zip(self.scales, self.fnos):
            # 下采样
            x_scaled = F.interpolate(x, size=(scale, scale), mode='bilinear')
            
            # FNO 处理
            feat = fno(x_scaled)
            
            # 上采样回原始分辨率
            feat = F.interpolate(feat, size=x.shape[-2:], mode='bilinear')
            multi_scale_features.append(feat)
        
        # 融合
        out = torch.cat(multi_scale_features, dim=1)
        out = self.fusion(out)
        
        return out
```

**预期效果**: +1-2%

#### 3.4.4 修改点

| 方案 | 文件 | 修改内容 | 行数 |
|------|------|----------|------|
| 残差增强 | `mhf_attention.py` | 修改 forward | ~20 行 |
| 频域注意力 | `mhf_attention.py` | 新增模块 | ~80 行 |
| 多尺度 | `mhf_fno.py` | 新增类 | ~100 行 |

**总代码量**: ~200 行

---

### 方向 5: 数据增强 ⭐⭐⭐

#### 3.5.1 物理保持的数据增强

```python
class PhysicsAugmentation:
    """保持物理不变性的数据增强"""
    
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, x, y):
        # 1. 空间翻转（x 方向）
        if torch.rand(1) < self.p:
            x = torch.flip(x, dims=[-1])
            y = torch.flip(y, dims=[-1])
        
        # 2. 旋转 90°（2D NS 保持不变性）
        if torch.rand(1) < self.p:
            k = torch.randint(1, 4, (1,)).item()
            x = torch.rot90(x, k, dims=[-2, -1])
            y = torch.rot90(y, k, dims=[-2, -1])
        
        # 3. 时间平移（如果数据包含时间序列）
        # x: [B, T, C, H, W] -> x[:, t_shift:]
        
        return x, y
```

**预期效果**: +0.5-1%

#### 3.5.2 频域增强

```python
def frequency_augmentation(u, noise_level=0.01):
    """频域数据增强"""
    # FFT
    u_freq = torch.fft.rfft2(u)
    
    # 添加高频噪声
    H, W = u_freq.shape[-2:]
    noise = torch.randn_like(u_freq) * noise_level
    
    # 只添加到高频部分
    mask = torch.zeros_like(u_freq)
    mask[..., H//4:, W//4:] = 1
    u_freq = u_freq + noise * mask
    
    # IFFT
    u_aug = torch.fft.irfft2(u_freq, s=u.shape[-2:])
    
    return u_aug
```

**预期效果**: +0.3-0.5%

#### 3.5.3 修改点

| 文件 | 修改内容 | 行数 |
|------|----------|------|
| `augmentation.py` | 新增增强类 | ~40 行 |
| `run_benchmarks.py` | 集成增强 | ~10 行 |

**总代码量**: ~50 行

---

## 四、立即可行方案

### 方案 A: 配置优化（5 分钟）

**修改文件**: `benchmark/run_benchmarks.py`

```python
# 找到MHF-FNO 模型创建部分（约第 320 行）
# 替换为：

# NS 专用配置
if dataset_name == 'navier_stokes':
    model_mhf = MHFFNO(
        n_modes=n_modes,
        hidden_channels=hidden_channels,
        in_channels=info['input_channels'],
        out_channels=info['output_channels'],
        n_layers=3,
        n_heads=2,          # 减少头数
        mhf_layers=[0],     # 只用第一层
    )
else:
    # 其他数据集保持原配置
    model_mhf = MHFFNO(
        n_modes=n_modes,
        hidden_channels=hidden_channels,
        in_channels=info['input_channels'],
        out_channels=info['output_channels'],
        n_layers=3,
        n_heads=4,
    )
```

**预期效果**: +1-2%

---

### 方案 B: 训练优化（10 分钟）

**修改文件**: `benchmark/run_benchmarks.py`

```python
# 找到训练函数（约第 220 行）
# 修改优化器和训练循环：

def train_model(model, train_x, train_y, test_x, test_y, config, verbose=True):
    # 使用 OneCycleLR
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=5e-4,
        weight_decay=1e-4
    )
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-3,
        epochs=config['epochs'],
        steps_per_epoch=(train_x.shape[0] + config['batch_size'] - 1) // config['batch_size'],
        pct_start=0.3
    )
    
    # ... 其他代码 ...
    
    for epoch in range(config['epochs']):
        # ... 训练代码 ...
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        scheduler.step()  # 每个 batch 更新
        
        # 早停
        if len(results['test_losses']) > 10:
            if min(results['test_losses'][-10:]) > min(results['test_losses'][:-10]):
                print(f"Early stopping at epoch {epoch}")
                break
```

**预期效果**: +0.5-1%

---

## 五、推荐实施路线

### 阶段 1: 快速验证（1 小时）

1. **方案 A**: 配置优化
   - 修改 `n_heads=2`, `mhf_layers=[0]`
   - 运行 20 epochs 测试

2. **方案 B**: 训练优化
   - 添加 OneCycleLR + 梯度裁剪
   - 运行 20 epochs 测试

**预期效果**: +1.5-3%
**风险**: 极低

---

### 阶段 2: PINO 约束（2-3 小时）

1. 实现 `PINOLoss` 类
2. 集成到训练流程
3. 调优 λ 参数

**预期效果**: +3-5%
**风险**: 低

---

### 阶段 3: 架构改进（1-2 天）

1. 实现频域注意力
2. 实现多尺度融合
3. 消融实验

**预期效果**: +2-4%
**风险**: 中等

---

## 六、总结

### 最佳方案组合

| 优先级 | 方案 | 效果 | 成本 | ROI |
|--------|------|------|------|-----|
| 🥇 | **配置优化** | +1-2% | 5分钟 | 极高 |
| 🥈 | **PINO 约束** | +3-5% | 2 小时 | 很高 |
| 🥉 | **训练优化** | +0.5-1% | 10 分钟 | 高 |

### 预期总体提升

- **保守估计**: +2-3%（配置 + 训练优化）
- **乐观估计**: +5-8%（所有方案组合）

### 下一步行动

1. **立即执行**: 方案 A（配置优化）
2. **本周完成**: 方案 B（训练优化）
3. **下周计划**: PINO 约束实现

---

**报告完成时间**: 2026-03-26 01:45
**预计验证时间**: 1 小时（方案 A + B）
