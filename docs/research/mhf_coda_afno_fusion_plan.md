# MHF+CoDA+AFNO 三重融合研究计划

> **研究方向**: MHF + CoDA + AFNO 三重融合
> **研究日期**: 2026-03-30
> **版本**: 2.0.0（已修正 CODA 理解）

---

## 📋 核心发现：CODA 的正确含义

### ✅ CODA = Cross-Head Attention（跨头注意力机制）

**关键修正**：
- ❌ **错误理解**：CODA = Curriculum and Data Augmentation
- ✅ **正确理解**：CODA = Cross-Head Attention（跨头注意力机制）

**实现位置**：
- `mhf_fno/mhf_attention.py`（165行完整实现）

**核心类**：
- `CrossHeadAttention` - 跨头注意力模块
- `MHFSpectralConvWithAttention` - 带注意力的 MHF 频谱卷积
- `MHFFNOWithAttention` - 带注意力的 MHF-FNO 模型

**设计目的**：
- 解决 MHF 的头之间**完全独立**的问题
- 实现**跨频率交互**（不同频率头之间的信息交换）
- 保持参数高效（参数量 < MHF 卷积的 1%）

---

## 🎯 研究目标

### 主要目标

探索 MHF（Multi-Head Fourier）、CoDA（Cross-Head Attention）和 AFNO（Adaptive Fourier Neural Operator）的**三重融合**方案，验证其有效性。

### 研究问题

1. **MHF+CoDA 的协同效应如何？**
   - Cross-Head Attention 如何增强 MHF 的跨频率交互？
   - 在哪些 PDE 类型上效果最显著？

2. **MHF+CoDA+AFNO 三重融合是否有效？**
   - 如何有机整合三种技术？
   - 能否超越任何两两组合？

3. **与现有方案对比效果如何？**
   - 对比纯 MHF、纯 AFNO、MHF+AFNO、MHF+CoDA
   - 对比 neuraloperator 库普通 FNO 以及 TFNO

---

## 📊 方案对比设计

### 对比方案（用户指定）

| 方案名称 | 技术 | 描述 |
|----------|------|------|
| **FNO-Baseline** | Standard FNO | neuraloperator 库普通 FNO |
| **TFNO-Baseline** | TFNO | 张量分解 FNO |
| **MHF-Only** | MHF | 纯 MHF 方案 |
| **AFNO-Only** | AFNO | 纯 AFNO 方案 |
| **MHF+CoDA** | MHF + CoDA | 跨头注意力增强 |
| **MHF+AFNO** | MHF + AFNO | 参数效率融合 |
| **MHF+CoDA+AFNO** | MHF + CoDA + AFNO | **三重融合（核心）** |

### 对比实验设计

| 数据集 | PDE 类型 | 推荐方案 | 测试配置 |
|--------|----------|----------|----------|
| Darcy Flow 2D | 椭圆型 | MHF+CoDA | 全部方案对比 |
| Burgers 1D | 抛物型 | MHF+CoDA+AFNO | 全部方案对比 |
| NS 2D (标量) | 双曲型 | MHF | 全部方案对比 |
| NS 2D (真实数据) | 双曲型 | **MHF+CoDA+PINO** | 全部方案对比 + PINO |

---

## 🏗️ 三重融合架构设计

### 方案 1: 顺序融合（Sequential Fusion）

**架构设计**：
```
Input → Embedding
  ↓
Layer 0: MHF + CoDA (首尾层使用）
  ↓
Layer 1: MHF + AFNO (中间层使用 AFNO 块对角 + 频率稀疏化)
  ↓
Layer 2: MHF + CoDA
  ↓
Layer 3: Standard FNO (可选）
  ↓
Projection → Output
```

**优势**：
- 层级明确，易于调试
- 可以独立控制每种技术的作用范围

**劣势**：
- 可能存在信息瓶颈
- 融合方式较简单

### 方案 2: 并行融合（Parallel Fusion）

**架构设计**：
```
Input → Embedding
  ↓
  ├─ MHF Path: [MHF Layers]
  ├─ CoDA Path: [MHF + CoDA Layers]
  └─ AFNO Path: [AFNO Layers]
  ↓
Feature Fusion (Concat/Attention Weighted)
  ↓
Projection → Output
```

**优势**：
- 并行计算，可以充分利用硬件
- 易于分析每种技术的独立贡献

**劣势**：
- 参数量增加
- 融合权重学习困难

### 方案 3: 混合融合（Hybrid Fusion）⭐ 推荐

**架构设计**：
```
Input → Embedding
  ↓
Layer 0: MHF + CoDA (早期：强跨频率交互)
  ↓
Layer 1: AFNO-Enhanced MHF (中期：自适应频率选择)
  ↓
Layer 2: MHF + CoDA + AFNO (后期：全技术协同)
  ↓
Layer 3: CoDA-Gated AFNO (可选：CoDA 门控 AFNO）
  ↓
Projection → Output
```

**核心创新**：
- **渐进式融合**：不同层使用不同融合策略
- **CoDA 门控 AFNO**：用 Cross-Head Attention 门控 AFNO 的频率稀疏度
- **AFNO-Enhanced MHF**：用 AFNO 的 block-diagonal 思想增强 MHF

---

## 🔬 实验设计

### 实验数据集

| 数据集 | 样本数 | 分辨率 | 输入 | 输出 | 复杂度 |
|--------|--------|--------|------|------|--------|
| Darcy Flow | 1000 | 64×64, 128×128 | 渗透率场 | 压力场 | 低 |
| Burgers | 500 | 256, 512 | 初值 | 速度场 | 中 |
| NS 2D (合成) | 1000 | 64×64, 128×128 | 初始条件 | 速度场 | 高 |
| NS 2D (真实) | 1000 | 128×128 | 历史速度场 | 未来速度场 | 极高 |

### 评估指标

| 指标类型 | 指标 | 说明 |
|----------|------|------|
| **精度指标** | MSE/RMSE | 均方/根均方误差 |
| | L2 Relative Error | 相对 L2 误差 |
| | L∞ Error | 最大绝对误差 |
| **效率指标** | 参数量 | 模型参数数量 |
| | 推理延迟 | 单次前向传播时间 |
| | 训练时间 | 达到目标精度的时间 |
| **泛化指标** | 泛化差距 | Train Error - Test Error |
| | 分辨率泛化 | 跨分辨率性能 |
| | 数据分布泛化 | 跨数据集性能 |

### 实验配置

**训练配置**：
```python
training_config = {
    'epochs': 50,
    'batch_size': 32,
    'learning_rate': 5e-4,
    'weight_decay': 1e-4,
    'scheduler': 'cosine',
    'warmup_epochs': 5,
    'grad_clip': 1.0
}
```

**模型配置**：
```python
model_config = {
    'hidden_channels': 32,
    'n_layers': 3,
    'n_modes': (16, 16),
    'n_heads': 4,
    'num_blocks': 8,  # AFNO 特有
    'sparsity_ratio': 0.5  # AFNO 特有
}
```

---

## 🚀 实施计划

### 阶段 1: 基础实现（第1-2天）

**任务清单**：
- [x] 理解现有 MHF+CoDA 实现
- [ ] 实现 AFNO 基础模块（Block-Diagonal Conv + 频率稀疏化）
- [ ] 设计三重融合架构（方案 3：混合融合）
- [ ] 实现 `MHFCoDAWithAFNO` 模型

**交付物**：
- AFNO 基础模块代码
- 三重融合模型代码
- 单元测试

### 阶段 2: 实验验证（第3-5天）

**任务清单**：
- [ ] 实现对比测试框架
- [ ] 在 Darcy Flow 上运行对比实验
- [ ] 在 Burgers 上运行对比实验
- [ ] 在 NS 2D 上运行对比实验

**交付物**：
- 完整实验结果
- 初步性能对比表
- 训练曲线图

### 阶段 3: 深入分析（第6-7天）

**任务清单**：
- [ ] 真实 NS 数据测试（需要 Zenodo 数据）
- [ ] NS 测试加 PINO（物理约束）
- [ ] 消融实验（MHF、CoDA、AFNO 各自贡献）
- [ ] 跨分辨率泛化测试

**交付物**：
- 真实数据实验结果
- 消融实验分析
- 泛化能力评估

### 阶段 4: 报告撰写（第8天）

**任务清单**：
- [ ] 生成完整实验报告
- [ ] 可视化结果对比
- [ ] 优化建议和下一步计划

**交付物**：
- 完整实验报告
- 结果可视化图表
- 优化建议

---

## 📈 预期结果

### 预期性能排序

| 预期排名 | 方案 | 预期精度 | 预期效率 |
|----------|------|----------|----------|
| 🥇 | **MHF+CoDA+AFNO** | **最佳** | **高** |
| 🥈 | MHF+CoDA | 很好 | 中高 |
| 🥉 | MHF+AFNO | 好 | 很高 |
| 4 | AFNO-Only | 较好 | 很高 |
| 5 | MHF-Only | 中等 | 高 |
| 6 | TFNO-Baseline | 中等 | 中等 |
| 7 | FNO-Baseline | 基线 | 基线 |

### 预期改进幅度

| 指标 | Baseline FNO | MHF+CoDA+AFNO | 相对改进 |
|------|--------------|----------------|----------|
| 参数量 | 1.68M | 0.48M | **-71%** |
| 推理延迟 | 21.5ms | 12.0ms | **-44%** |
| 训练时间 | 75min | 40min | **-47%** |
| MSE (Darcy) | 0.7245 | 0.7080 | **+2.3%** |
| 泛化差距 | 0.0450 | 0.0250 | **-44%** |

---

## 🔍 消融实验设计

### 实验 1: MHF 贡献分析

**对比**：
- FNO-Baseline vs MHF-Only vs MHF+CoDA vs MHF+AFNO

**目的**：验证 MHF 独立贡献

### 实验 2: CoDA 贡献分析

**对比**：
- MHF-Only vs MHF+CoDA vs MHF+CoDA+AFNO

**目的**：验证 Cross-Head Attention 的跨频率交互效果

### 实验 3: AFNO 贡献分析

**对比**：
- MHF+CoDA vs MHF+AFNO vs MHF+CoDA+AFNO

**目的**：验证 AFNO 的 block-diagonal 和频率稀疏化效果

### 实验 4: 融合策略分析

**对比**：
- 顺序融合 vs 并行融合 vs 混合融合

**目的**：选择最佳融合策略

---

## 📚 参考文献

1. **MHF**:
   - TransFourier: Multi-Head Attention in Spectral Domain

2. **CoDA**:
   - CoDA-NO: Continuous-Discrete Augmented Neural Operator
   - Squeeze-and-Excitation Networks (SENet)

3. **AFNO**:
   - Adaptive Fourier Neural Operators: Efficient Token Mixers for Transformers
   - arXiv:2111.13587

4. **FNO**:
   - Fourier Neural Operator for Parametric Partial Differential Equations
   - arXiv:2010.08895

5. **PINO**:
   - Physics-Informed Neural Operator
   - arXiv:2111.03794

---

## ✅ 下一步行动

### 立即执行（今天）

1. **✅ 理解现有 MHF+CoDA 实现**（已完成）
2. [ ] 开始实现 AFNO 基础模块
3. [ ] 设计三重融合架构

### 短期执行（明天）

4. [ ] 完成三重融合模型实现
5. [ ] 单元测试
6. [ ] 准备实验数据

### 中期执行（后天）

7. [ ] 运行对比实验
8. [ ] 分析实验结果
9. [ ] 撰写实验报告

---

*研究计划版本: 2.0.0（已修正 CODA 理解）*
*创建时间: 2026-03-30*
*状态: 准备实施*
