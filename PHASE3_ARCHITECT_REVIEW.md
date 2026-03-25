# Phase 3: 架构师审查报告

> **审查人**: 天渊（架构师）  
> **日期**: 2026-03-25  
> **团队**: 天渊团队

---

## 一、迭代 1 结果总结

### 1.1 Phase 2 测试结果

| 变体 | 参数量 | Best Loss | vs MHF-FNO | vs FNO |
|------|--------|-----------|------------|--------|
| MHF-FNO (无注意力) | 140,017 | 0.0547 | 基准 | **-3.4%** ✅ |
| SENet 风格 | 140,763 | 0.0550 | -0.5% ❌ | -2.9% |
| **CoDA 风格** | 140,363 | **0.0541** | **+1.1%** ✅ | **-4.5%** ✅ |
| MHA 风格 | 140,625 | 0.0581 | -6.2% ❌ | +2.5% ❌ |

**当前最佳方案**: CoDA 风格注意力  
**整体效果**: vs FNO -4.5%（目标 -10%，差距 5.5%）

---

## 二、CoDA 实现审查

### 2.1 代码实现分析

**文件**: `mhf_fno/attention_variants.py`

```python
class CoDAStyleAttention(nn.Module):
    def __init__(self, n_heads, head_dim, bottleneck=4, dropout=0.0):
        # 1. 瓶颈压缩: head_dim → bottleneck
        self.compress = nn.Linear(head_dim, bottleneck)
        
        # 2. 跨头注意力: 在瓶颈空间做注意力
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=bottleneck,  # 压缩后的维度
            num_heads=2,           # 少量注意力头
            dropout=dropout
        )
        
        # 3. 特征重建: bottleneck → head_dim
        self.expand = nn.Linear(bottleneck, head_dim)
        
        # 4. 门控参数
        self.gate = nn.Parameter(torch.ones(1) * 0.1)  # 初始门控值 0.1
```

### 2.2 实现正确性评估

| 方面 | 评估 | 说明 |
|------|------|------|
| 瓶颈结构 | ✅ 正确 | head_dim=8 → bottleneck=4，压缩率 50% |
| 跨头注意力 | ✅ 正确 | n_heads 作为序列长度，正确实现 |
| 门控机制 | ⚠️ 可优化 | 固定初始值 0.1，可能不是最优 |
| 残差连接 | ✅ 正确 | `out = x + gate * delta` |
| 层归一化 | ⚠️ 位置 | 在门控后做 LayerNorm，可能不是最佳 |

### 2.3 潜在问题

#### 问题 1: 瓶颈大小固定

```python
bottleneck = max(4, head_dim // 2)  # 对于 head_dim=8，bottleneck=4
```

**影响**: 
- 压缩率 50% 可能过高，丢失信息
- 不同 PDE 可能需要不同的压缩率

**建议**: 将 `bottleneck` 作为超参数调优

#### 问题 2: 注意力位置

```python
# 当前: 空间域注意力 (IFFT 之后)
x_out_spatial = torch.fft.irfft2(out_freq_merged, ...)
x_heads = x_out_spatial.view(B, self.n_heads, self.head_out, H, W)
x_heads = self.cross_head_attn(x_heads)  # ← 在这里做注意力
```

**潜在改进**: 在频域做注意力可能更有效

---

## 三、为什么 CoDA 比 MHF-FNO 原版好

### 3.1 理论分析

| 特性 | MHF-FNO | CoDA |
|------|---------|------|
| 头间交互 | ❌ 完全独立 | ✅ 通过注意力学习 |
| 频率耦合 | ❌ 无法耦合 | ✅ 跨频率信息交换 |
| 参数量 | 140,017 | 140,363 (+0.25%) |
| 计算开销 | 基准 | +3.4% |

### 3.2 工作原理

```
MHF-FNO 头间独立:
  Head_1 (低频) ──── 输出_1
  Head_2 (中频) ──── 输出_2  ← 无法交互
  Head_3 (中频) ──── 输出_3
  Head_4 (高频) ──── 输出_4

CoDA 跨头注意力:
  Head_1 ─┐
  Head_2 ─┼──▶ 瓶颈压缩 ──▶ 跨头注意力 ──▶ 特征重建 ─┬──▶ 增强输出
  Head_3 ─┤                                          │
  Head_4 ─┘                                          │
                                                     │
                         原始特征 ◀─────────────────┘
                              ↓
                         门控融合 ──▶ 最终输出
```

### 3.3 关键优势

1. **信息整合**: 低频和高频头可以交换信息
2. **自适应耦合**: 门控学习最优的融合强度
3. **参数高效**: 瓶颈结构避免参数爆炸

---

## 四、为什么 CoDA 比 MHA 好

### 4.1 对比分析

| 特性 | MHA (CrossHeadMultiHeadAttention) | CoDA (CoDAStyleAttention) |
|------|-----------------------------------|---------------------------|
| 计算复杂度 | O(H*W*n_heads^2) | O(n_heads^2) |
| 参数量 | ~140,625 | ~140,363 |
| Best Loss | 0.0581 | **0.0541** |
| vs MHF-FNO | -6.2% ❌ | **+1.1%** ✅ |

### 4.2 MHA 为什么失败

**问题 1: 空间位置独立注意力**

```python
# MHA: 对每个空间位置单独做跨头注意力
# 输入: [B, n_heads, head_dim, H, W]
# 重塑: [B, H*W, n_heads, head_dim]  ← 展平空间维度
# 注意力: [B, H*W, num_attn_heads, n_heads, n_heads]  ← 每个位置独立
```

**后果**:
- 计算量爆炸: H*W=1024 个位置，每个位置做一次注意力
- 小数据集过拟合: 200 个训练样本无法支撑大量参数
- 信息冗余: 相邻位置的注意力模式相似

**问题 2: 无全局信息**

- MHA 对每个空间位置独立处理，缺乏全局上下文
- CoDA 先全局池化，获取全局频率特征

### 4.3 CoDA 为什么成功

**优势 1: 全局池化**

```python
# CoDA: 先池化再注意力
if x.dim() == 4:  # 1D
    x_pooled = x.mean(dim=-1)  # [B, n_heads, head_dim]
else:  # 2D
    x_pooled = x.mean(dim=(-2, -1))  # 空间维度池化
```

**优势 2: 瓶颈压缩**

- head_dim=8 → bottleneck=4
- 参数减少 75%，正则化效果

**优势 3: 门控学习**

```python
self.gate = nn.Parameter(torch.ones(1) * 0.1)  # 可学习的门控
out = x + self.gate * delta  # 残差 + 门控增强
```

---

## 五、如何达到 -10% 目标

### 5.1 当前差距

- 当前最佳: vs FNO -4.5%
- 目标: vs FNO -10%
- **差距**: 5.5%

### 5.2 改进方案

#### 方案 1: CoDA 超参数优化 ⭐ 推荐

| 参数 | 当前值 | 建议范围 | 预期收益 |
|------|--------|----------|----------|
| bottleneck | 4 | [2, 4, 6, 8] | 1-2% |
| num_attn_heads | 2 | [1, 2, 4] | 0.5-1% |
| gate_init | 0.1 | [0.05, 0.1, 0.2, 0.3] | 0.5-1% |

**实现**: 网格搜索或贝叶斯优化

#### 方案 2: 全层注意力 ⭐ 推荐

```python
# 当前: 只在首尾层使用 MHF+CoDA
mhf_layers = [0, 2]  # 2/3 层

# 建议: 所有层使用 MHF+CoDA
mhf_layers = [0, 1, 2]  # 3/3 层
```

**预期收益**: 1-2%

#### 方案 3: 增强训练策略

| 策略 | 当前 | 建议 | 预期收益 |
|------|------|------|----------|
| epochs | 80 | 150-200 | 1-2% |
| 数据量 | 200 | 500-1000 | 2-3% |
| 学习率调度 | CosineAnnealing | Warmup+Cosine | 0.5% |
| 数据增强 | 无 | 随机翻转/旋转 | 0.5-1% |

#### 方案 4: 架构增强 (风险较高)

**4.1 双域注意力**

```python
# 频域注意力 + 空间域注意力
class DualDomainCoDA(nn.Module):
    def forward(self, x):
        # 频域分支
        x_freq = torch.fft.rfft2(x)
        freq_attn = self.freq_attention(x_freq)
        
        # 空间域分支
        x_spatial = torch.fft.irfft2(x_freq)
        spatial_attn = self.spatial_attention(x_spatial)
        
        # 融合
        return self.fusion(freq_attn, spatial_attn)
```

**预期收益**: 1-3% (但增加复杂度)

**4.2 残差门控**

```python
# 当前
out = x + gate * delta

# 建议: 残差门控
out = (1 - gate) * x + gate * (x + delta)
```

---

## 六、改进建议优先级

### P0 - 立即执行 (预期收益 2-4%)

| 改进 | 预期收益 | 工作量 | 风险 |
|------|----------|--------|------|
| 全层 MHF+CoDA | 1-2% | 低 | 低 |
| CoDA 超参数调优 | 1-2% | 中 | 低 |
| 增加训练 epochs | 0.5-1% | 低 | 无 |

### P1 - 短期改进 (预期收益 1-3%)

| 改进 | 预期收益 | 工作量 | 风险 |
|------|----------|--------|------|
| 增加数据量 | 2-3% | 中 | 低 |
| 数据增强 | 0.5-1% | 低 | 低 |
| 残差门控优化 | 0.5-1% | 低 | 低 |

### P2 - 长期探索 (预期收益 1-3%)

| 改进 | 预期收益 | 工作量 | 风险 |
|------|----------|--------|------|
| 双域注意力 | 1-3% | 高 | 中 |
| 自适应瓶颈 | 0.5-1% | 中 | 低 |

---

## 七、下一步行动计划

### 迭代 2 建议

**Phase 1**: 快速验证

```python
# 配置 1: 全层 CoDA
model = create_mhf_fno_v2(
    n_modes=(12, 12),
    hidden_channels=32,
    n_heads=4,
    attention_type='coda',
    mhf_layers=[0, 1, 2]  # 全层
)

# 配置 2: 增大瓶颈
model = create_mhf_fno_v2(
    ...,
    attention_type='coda',
    bottleneck=6  # 从 4 增加到 6
)
```

**训练配置**:
- epochs: 150 (从 80 增加)
- 数据: 500 样本 (如果有)

### 预期结果

| 配置 | 预期 vs FNO |
|------|-------------|
| 全层 CoDA | -6% ~ -7% |
| 全层 + 超参调优 | -7% ~ -8% |
| 全层 + 更多数据 | -8% ~ -10% |

---

## 八、总结

### 审查结论

1. **CoDA 实现正确** ✅ - 瓶颈结构和门控机制设计合理
2. **CoDA 比 MHF-FNO 好** ✅ - 跨头注意力实现频率耦合
3. **CoDA 比 MHA 好** ✅ - 全局池化 + 瓶颈压缩避免过拟合

### 达到目标可行性

- **当前**: -4.5%
- **P0 改进后**: -6% ~ -8% (差距 2-4%)
- **P1 改进后**: -8% ~ -10% (有望达标)

### 建议

**进入迭代 2**，重点执行:
1. 全层 MHF+CoDA 配置
2. CoDA 超参数调优 (bottleneck, gate_init)
3. 增加训练 epochs 到 150

---

*审查完成 - 天渊*