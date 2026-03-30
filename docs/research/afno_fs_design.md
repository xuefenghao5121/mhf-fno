# AFNO 频率稀疏度课程 (AFNO-FS) 技术设计文档

> **研究方向**: 方向B: AFNO频率稀疏度课程 (AFNO-FS)
> **创建时间**: 2026-03-30
> **版本**: 1.0.0

---

## 1. 概述

### 1.1 研究动机

AFNO (Adaptive Fourier Neural Operator) 的核心创新之一是频率稀疏化 (Frequency Sparsification)，通过过滤高频噪声来提升模型鲁棒性。然而，固定稀疏度策略存在以下问题：

1. **早期高频信息丢失**：训练初期高频信号可能包含重要信息
2. **后期噪声保留**：训练后期高频噪声可能影响精度
3. **收敛速度慢**：固定策略无法根据训练进度自适应调整

**AFNO-FS (AFNO Frequency Sparsity Curriculum)** 通过渐进式调整频率稀疏度，实现从低频主导到高频精细的自适应训练过程。

### 1.2 核心思想

**课程式频率学习**：
```
阶段 0 (预热期):   高稀疏度 (80%)  → 保留低频模式，快速收敛
阶段 1 (早期):      中高稀疏度 (60%) → 引入中低频模式
阶段 2 (中期):      中等稀疏度 (40%) → 引入全频模式
阶段 3 (后期):      低稀疏度 (20%)  → 精细调整高频特征
阶段 4 (微调期):    最小稀疏度 (10%) → 充分利用全频谱
```

### 1.3 预期优势

| 指标 | 固定稀疏度 AFNO | AFNO-FS | 改进 |
|------|----------------|---------|------|
| 训练收敛速度 | 100 epochs | 60 epochs | **-40%** |
| 最终精度 (MSE) | 0.7180 | 0.7100 | **+1.1%** |
| 高频特征学习 | 受限 | 渐进式 | **显著** |
| 泛化能力 | 中等 | 强 | **提升** |

---

## 2. 技术架构

### 2.1 系统架构

```
AFNO-FS 模型架构:

输入 x (B, C, H, W)
    ↓
输入投影 Linear(C → hidden_channels)
    ↓
┌─────────────────────────────────────┐
│  AFNO Layer 1                      │
│  ├─ Block-Diagonal Spectral Conv  │
│  ├─ Frequency Sparsification        │
│  └─ Sparsity Scheduler (动态)      │
├─────────────────────────────────────┤
│  AFNO Layer 2                      │
│  ├─ Block-Diagonal Spectral Conv  │
│  ├─ Frequency Sparsification        │
│  └─ Sparsity Scheduler (动态)      │
├─────────────────────────────────────┤
│  ...                               │
└─────────────────────────────────────┘
    ↓
输出投影 Linear(hidden_channels → C)
    ↓
输出 y (B, C, H, W)
```

### 2.2 核心组件

#### 2.2.1 AFNOBlockDiagonalConv

**功能**：实现 Block-Diagonal 频谱卷积 + 频率稀疏化

**关键特性**：
- Block-Diagonal 权重分解：`R = diag(R_1, ..., R_b)`
- 可学习的稀疏化阈值：`λ ∈ R`
- 自适应权重共享：`gates ∈ R^b`

#### 2.2.2 FrequencySparsityScheduler

**功能**：根据训练进度动态调整稀疏度

**支持的调度策略**：

1. **Linear (线性调度)**
   ```python
   sparsity_ratio = max_ratio + (min_ratio - max_ratio) * progress
   ```

2. **Root (根号调度)**
   ```python
   sparsity_ratio = max_ratio + (min_ratio - max_ratio) * sqrt(progress)
   ```

3. **Exp (指数调度)**
   ```python
   sparsity_ratio = max_ratio + (min_ratio - max_ratio) * (1 - exp(-3*progress))
   ```

4. **Cosine (余弦调度)**
   ```python
   sparsity_ratio = max_ratio + (min_ratio - max_ratio) * (1 - cos(progress*π)) / 2
   ```

5. **Step (阶梯调度)**
   ```python
   if epoch < 30:      return 0.8
   elif epoch < 60:    return 0.5
   elif epoch < 80:    return 0.3
   else:               return 0.1
   ```

6. **Adaptive (自适应调度)**
   ```python
   if loss_improvement > threshold:
       decrease_sparsity_ratio()  # 损失下降，降低稀疏度
   ```

#### 2.2.3 AFNOWithCurriculum

**功能**：集成频率稀疏度课程学习

**关键方法**：
- `set_training_mode(epoch, training)`: 设置训练模式并更新稀疏度
- `get_current_sparsity_ratio()`: 获取当前稀疏度
- `get_sparsity_history()`: 获取稀疏度历史

---

## 3. 实现细节

### 3.1 文件结构

```
mhf_fno/
├── afno.py                    # AFNO 基础实现
│   ├── AFNOBlockDiagonalConv   # Block-Diagonal 频谱卷积
│   ├── AFNO                    # AFNO 模型
│   └── create_afno             # 工厂函数
│
├── afno_curriculum.py         # AFNO-FS 实现
│   ├── FrequencySparsityScheduler      # 稀疏度调度器
│   ├── AFNOWithCurriculum      # 带课程的 AFNO
│   └── create_afno_with_curriculum    # 工厂函数
│
└── __init__.py                 # 导出接口

examples/
├── afno_curriculum_usage.py    # 使用示例
└── afno_fs_experiment.py      # 实验验证脚本
```

### 3.2 使用示例

```python
from mhf_fno.afno_curriculum import create_afno_with_curriculum

# 创建模型
model = create_afno_with_curriculum(
    in_channels=1,
    out_channels=1,
    hidden_channels=64,
    n_layers=4,
    n_modes=(12, 12),
    num_blocks=8,
    max_sparsity_ratio=0.8,  # 早期保留 20% 频率
    min_sparsity_ratio=0.1,  # 后期保留 90% 频率
    total_epochs=100,
    scheduler_type='linear',
    warmup_epochs=10
)

# 训练循环
for epoch in range(100):
    model.train()
    model.set_training_mode(epoch, training=True)
    
    for batch in dataloader:
        optimizer.zero_grad()
        y_pred = model(x_batch)  # 自动使用当前稀疏度
        loss = F.mse_loss(y_pred, y_batch)
        loss.backward()
        optimizer.step()
    
    # 获取当前稀疏度
    sparsity_ratio = model.get_current_sparsity_ratio()
    print(f"Epoch {epoch}, Sparsity: {sparsity_ratio:.3f}")
```

---

## 4. 实验设计

### 4.1 实验目标

验证 AFNO-FS 相比固定稀疏度 AFNO 的性能优势。

### 4.2 对比方法

| 方法 | 描述 | 稀疏度策略 |
|------|------|-----------|
| Fixed-0.1 | 固定稀疏度 | 10% (保留 90% 频率) |
| Fixed-0.5 | 固定稀疏度 | 50% (保留 50% 频率) |
|**Fixed-0.8**| **固定稀疏度** | **80% (保留 20% 频率)** |
| Curriculum-Linear | 线性调度 | 80% → 10% |
| Curriculum-Root | 根号调度 | 80% → 10% |
| Curriculum-Exp | 指数调度 | 80% → 10% |

### 4.3 预期结果

| 方法 | Final Val Loss | 训练 Time | 稀疏度变化 |
|------|----------------|-----------|-----------|
| Fixed-0.1 | 0.7150 | 基线 | 固定 0.1 |
| Fixed-0.5 | 0.7180 | 基线 | 固定 0.5 |
| Fixed-0.8 | 0.7220 | 基线 | 固定 0.8 |
| Curriculum-Linear | 0.7100 | **快速** | 0.8→0.1 (线性) |
| Curriculum-Root | 0.7080 | **快速** | 0.8→0.1 (根号) |
| Curriculum-Exp | 0.7120 | **快速** | 0.8→0.1 (指数) |

---

## 5. 运行实验

### 5.1 单个调度器训练

```bash
python examples/afno_curriculum_usage.py \
    --scheduler linear \
    --epochs 100 \
    --save results/linear_scheduler.png
```

### 5.2 对比不同调度器

```bash
python examples/afno_curriculum_usage.py \
    --compare \
    --epochs 100 \
    --save results/scheduler_comparison.png
```

### 5.3 运行完整实验

```bash
python examples/afno_fs_experiment.py
```

---

## 6. 下一步优化建议

### 6.1 短期优化

1. **自适应调度器增强**
   - 基于验证损失的自适应调整
   - 基于梯度范数的动态调整
   - 多目标优化（精度 + 速度）

2. **稀疏度调度策略扩展**
   - 指数衰减调度
   - Piecewise 线性调度
   - 学习率协同调度

3. **性能优化**
   - CUDA kernel 优化（稀疏化操作）
   - 混合精度训练
   - 梯度检查点

### 6.2 中期优化

1. **多分辨率支持**
   - 分辨率自适应稀疏度
   - 跨分辨率课程学习

2. **多层差异化**
   - 不同层使用不同调度器
   - 层级化稀疏度策略

3. **物理约束集成**
   - 保持边界条件的稀疏化
   - 物理一致性感知的稀疏度

### 6.3 长期优化

1. **自动化调优**
   - AutoML 榆索最佳调度器
   - 强化学习优化调度策略

2. **理论分析**
   - 收敛性证明
   - 泛化界分析

3. **跨应用扩展**
   - 3D PDE 扩展
   - 时序 PDE 扩展

---

## 7. 参考文献

1. **AFNO**: Adaptive Fourier Neural Operators: Efficient Token Mixers for Transformers
   - arXiv:2111.13587

2. **Curriculum Learning**: Curriculum Learning: A Survey
   - arXiv:2106.03957

3. **FNO**: Fourier Neural Operator for Parametric Partial Differential Equations
   - arXiv:2010.08895

---

*文档版本: 1.0.0*
*最后更新: 2026-03-30*
*状态: 实现完成，待实验验证*
