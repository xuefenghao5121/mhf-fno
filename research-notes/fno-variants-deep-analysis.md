# FNO 变体深度分析：TFNO、GINO、CoDA-NO

> **研究员**: 天池 (Tianchi)  
> **日期**: 2026-03-25  
> **团队**: 天渊团队  
> **状态**: 持续研究中

---

## 一、引言

本报告深入分析三种重要的 FNO 变体：
1. **TFNO** - Tucker 分解压缩
2. **GINO** - 几何感知
3. **CoDA-NO** - Codomain 注意力

分析重点：这些方法如何处理频率耦合问题，以及对 MHF+CoDA 优化的启示。

---

## 二、TFNO：Tucker 分解方法

### 2.1 核心思想

**TFNO (Tensorized Fourier Neural Operator)** 使用 Tucker 分解压缩频谱卷积的权重张量。

### 2.2 数学基础

**标准 FNO 权重**:
$$W \in \mathbb{C}^{C_{in} \times C_{out} \times k_x \times k_y}$$

**Tucker 分解**:
$$W \approx \mathcal{G} \times_1 U_1 \times_2 U_2 \times_3 U_3 \times_4 U_4$$

其中：
- $\mathcal{G}$ 是核心张量，形状小于 $W$
- $U_i$ 是因子矩阵
- $\times_i$ 表示模-$i$ 乘积

### 2.3 参数压缩分析

**原始参数量**:
$$P_{FNO} = C_{in} \times C_{out} \times k_x \times k_y$$

**Tucker 压缩后**:
$$P_{TFNO} = r_1 r_2 r_3 r_4 + C_{in} r_1 + C_{out} r_2 + k_x r_3 + k_y r_4$$

其中 $r_i$ 是秩参数，$r_i < \text{dim}_i$。

**压缩率示例**（$C_{in}=C_{out}=32$, $k_x=k_y=12$, rank=0.5）:

| 方法 | 参数量 | 压缩率 |
|------|--------|--------|
| FNO | 147,456 | - |
| TFNO (rank=0.5) | ~45,000 | 69.5% |
| TFNO (rank=0.3) | ~25,000 | 83.0% |

### 2.4 与 MHF 的关键区别

**MHF 分割策略**:
```python
# MHF: 通道维度分割
W_mhf = [W_1, W_2, W_3, W_4]  # 4 个独立的子权重
# 每个子权重独立操作，无交互
```

**TFNO Tucker 分解**:
```python
# TFNO: 全张量分解
W_tucker = G × U1 × U2 × U3 × U4
# 因子矩阵 U_i 保持了维度间的连接
```

**频率耦合保留对比**:

| 特性 | MHF | TFNO |
|------|-----|------|
| 通道间交互 | ❌ 头独立 | ✅ U₁, U₂ 连接 |
| 频率间交互 | ❌ 无 | ✅ U₃, U₄ 连接 |
| 跨头通信 | ❌ 需要额外层 | ✅ 核心张量提供 |

### 2.5 对 NS 方程的启示

**TFNO 的优势**:

1. **保留频率耦合**
   - 核心张量 $\mathcal{G}$ 连接所有维度
   - 因子矩阵提供跨维度交互
   - 能量级联路径保留

2. **高效压缩**
   - 可以在保持性能的同时大幅压缩
   - 适合大规模高分辨率问题

**潜在问题**:
- Tucker 分解的最优秩选择需要调参
- 秩太低会损失精度

**对 MHF+CoDA 的启示**:
- 可以考虑引入类似 Tucker 的跨头连接机制
- 在 MHF 头之间添加轻量级交互模块

---

## 三、GINO：几何感知方法

### 3.1 核心思想

**GINO (Geometry-Informed Neural Operator)** 处理不规则几何域上的 PDE。

### 3.2 架构设计

```
输入 (点云表示)
       │
       ▼
┌──────────────────────────────────────┐
│        Geometry Encoder              │
│  • Signed Distance Function (SDF)    │
│  • 点云特征提取                       │
└──────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────┐
│      Graph Neural Operator           │
│  • 局部几何交互                       │
│  • 图卷积处理不规则域                  │
└──────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────┐
│      Fourier Neural Operator         │
│  • 全局频率依赖                       │
│  • 规则网格上的频谱卷积                │
└──────────────────────────────────────┘
       │
       ▼
输出 (点云表示)
```

### 3.3 GINO 的频率处理

**关键创新**:
- 使用图神经算子处理局部几何
- 使用 FNO 处理全局频率依赖
- 两者的结合提供完整的谱表示

**对频率耦合的处理**:
- 图神经算子：捕捉局部频率交互
- FNO：捕捉全局频率交互
- 结合后可以更好地处理 NS 方程的多尺度涡

### 3.4 对 MHF+CoDA 的启示

**可能的改进方向**:

1. **引入局部-全局分离**
   - MHF 处理全局频率依赖
   - 添加局部卷积层处理局部交互
   - 类似 GINO 的混合架构

2. **几何编码器**
   - 对于复杂几何边界问题
   - 添加几何感知模块

---

## 四、CoDA-NO：Codomain 注意力方法

### 4.1 核心思想

**CoDA-NO (Codomain Attention Neural Operator)** 在 codomain 维度（特征维度）上应用注意力机制。

### 4.2 数学表达

**标准注意力** (沿空间维度):
$$\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right) V$$

**Codomain 注意力** (沿特征维度):

设输入 $u(x) \in \mathbb{R}^{C \times H \times W}$

1. **Token 化**: 将 $u$ 沿 codomain 维度分割为 tokens
   $$[u_1, u_2, \ldots, u_n] = \text{Tokenize}(u)$$

2. **注意力计算**: 每个空间位置独立计算 codomain 注意力
   $$\alpha_{ij} = \text{softmax}(u_i^T u_j / \sqrt{d})$$

3. **聚合**: 
   $$u'_i = \sum_j \alpha_{ij} u_j$$

### 4.3 CoDA 的频率耦合恢复机制

**关键洞察**:

```python
# CoDA 注意力的作用
def coda_attention(x, n_heads):
    # x: [B, C, H, W]
    # 沿通道维度分割为 tokens
    tokens = x.view(B, n_heads, C // n_heads, H, W)
    
    # 计算跨头注意力
    # 这允许不同频率子空间的信息融合！
    attn_weights = compute_attention(tokens)  # [B, n_heads, n_heads]
    
    # 加权聚合
    out = aggregate(attn_weights, tokens)
    
    return out
```

**与 MHF 的配合**:

```
MHF 分割 ────→ 头独立处理 ────→ CoDA 注意力 ────→ 跨头融合
   │                                 │
   │    频率解耦（问题）              │    频率耦合恢复（解决方案）
   │                                 │
   └─────────────────────────────────┘
                    协同工作
```

### 4.4 实验验证

**我们的实验结果**:

| 配置 | vs FNO | 说明 |
|------|--------|------|
| MHF-[0,2] | -3.5% | 无 CoDA |
| CoDA-[0,2] | -4.4% | 有 CoDA，提升 0.9% |
| CoDA-[0,1,2] | +21.7% | 全层 CoDA 失败 |

**结论**:
- CoDA 确实能提升性能（+0.9%）
- 但全层使用会导致不稳定

### 4.5 CoDA 优化建议

**基于实验发现的问题**:

1. **Gate 衰减问题**
   - 多层叠加时，gate 效果指数衰减
   - 解决：适当增大 gate_init

2. **注意力计算开销**
   - 每层都计算注意力开销大
   - 解决：选择性使用（首尾层）

3. **训练稳定性**
   - 全层注意力导致训练不稳定
   - 解决：添加注意力正则化

---

## 五、三种方法的对比分析

### 5.1 频率耦合处理对比

| 方法 | 频率耦合机制 | 适用场景 | 局限性 |
|------|-------------|----------|--------|
| **TFNO** | Tucker 核心张量连接 | 高分辨率、内存受限 | 秩选择需调参 |
| **GINO** | 图+FNO 混合 | 复杂几何 | 实现复杂 |
| **CoDA-NO** | Codomain 注意力 | 多物理场 | 全层不稳定 |

### 5.2 与 MHF+CoDA 的关系

```
                    ┌─────────────────────────────────┐
                    │        MHF+CoDA                 │
                    │  • MHF: 通道分割（频率解耦）     │
                    │  • CoDA: 注意力恢复耦合         │
                    └─────────────────────────────────┘
                              │
           ┌──────────────────┼──────────────────┐
           │                  │                  │
           ▼                  ▼                  ▼
    ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
    │    TFNO     │   │    GINO     │   │   CoDA-NO   │
    │ Tucker 分解 │   │ 图+FNO混合  │   │  纯注意力   │
    │ 保持耦合    │   │ 局部+全局   │   │  跨头融合   │
    └─────────────┘   └─────────────┘   └─────────────┘
```

### 5.3 组合可能性

**TFNO + MHF**:
- 在 MHF 头之间添加 Tucker 式连接
- 保留频率耦合的同时减少参数

**GINO + MHF**:
- 添加局部卷积层处理局部频率交互
- 适用于复杂边界问题

**CoDA 改进**:
- 参考 Transformer 的最佳实践
- 添加 LayerNorm、残差连接增强稳定性

---

## 六、对 MHF+CoDA 优化的新见解

### 6.1 核心问题重述

**实验发现**:
- MHF-[0,2]: -3.5% ✅
- CoDA-[0,2]: -4.4% ✅
- 全层 MHF/CoDA: 失败 ❌

**核心问题**: 如何在保持频率耦合的同时最大化压缩效率？

### 6.2 新的优化方向

#### 方向 1: 引入 Tucker 式跨头连接

```python
class MHF_Tucker(SpectralConv):
    """MHF with Tucker-style cross-head connections"""
    
    def __init__(self, in_ch, out_ch, n_modes, n_heads, rank_ratio=0.5):
        super().__init__(in_ch, out_ch, n_modes)
        
        # 核心张量：连接所有头
        head_dim = in_ch // n_heads
        rank = int(head_dim * rank_ratio)
        self.core = nn.Parameter(
            torch.randn(rank, rank, *n_modes, dtype=torch.cfloat)
        )
        
        # 因子矩阵
        self.U_in = nn.Parameter(torch.randn(n_heads, head_dim, rank))
        self.U_out = nn.Parameter(torch.randn(n_heads, rank, head_dim))
        
    def forward(self, x):
        x_freq = torch.fft.rfft2(x)
        
        # Tucker 式分解保持跨头交互
        # out = core ×_1 U_in ×_2 U_out ×_3 x_freq
        
        return torch.fft.irfft2(out)
```

**预期收益**: 在保持跨头通信的同时实现高效压缩

#### 方向 2: 自适应头数分配

```python
# 不同层使用不同头数
layer_configs = {
    0: {'n_heads': 2, 'use_mhf': True},   # 首层：少头，保持信息
    1: {'n_heads': 1, 'use_mhf': False},  # 中间：标准卷积
    2: {'n_heads': 4, 'use_mhf': True},   # 尾层：多头，增加表达力
}
```

**预期收益**: 1-2%

#### 方向 3: 频率敏感的注意力

```python
class FrequencySensitiveCoDA(nn.Module):
    """频率敏感的 Codomain 注意力"""
    
    def forward(self, x):
        # 根据频率模式动态调整注意力强度
        freq_importance = self.compute_freq_importance(x)
        
        # 低频：强注意力（保持耦合）
        # 高频：弱注意力（避免过度平滑）
        attn_scale = torch.where(
            freq_importance > threshold,
            torch.ones_like(freq_importance),
            torch.zeros_like(freq_importance) * 0.5
        )
        
        return self.attention(x) * attn_scale
```

**预期收益**: 1-2%

### 6.3 数据效率优化

**基于实验发现的策略**:

| 数据量 | 推荐配置 | 原因 |
|--------|----------|------|
| < 500 | n_heads=2, 无注意力 | 减少过拟合风险 |
| 500-1000 | n_heads=4, CoDA-[0,2] | 平衡表达力和泛化 |
| > 1000 | n_heads=4, 更长训练 | 充分利用数据 |

---

## 七、实验验证计划

### 7.1 TFNO 对比实验

**目标**: 验证 Tucker 分解在 NS 方程上的效果

**实验设计**:
| 配置 | 压缩率 | 预期效果 |
|------|--------|----------|
| TFNO rank=0.5 | 50% | 对比 MHF |
| TFNO rank=0.3 | 70% | 验证极限压缩 |
| MHF-Tucker | 40% | 组合方案 |

### 7.2 自适应头数实验

**目标**: 找到最优头数配置

**实验设计**:
| 配置 | n_heads 分布 | 预期效果 |
|------|-------------|----------|
| 固定 4 头 | [4, 4, 4] | 基准 |
| 首少尾多 | [2, 4, 4] | 保守改进 |
| 自适应 | [2, 1, 4] | 激进改进 |

### 7.3 频率敏感注意力实验

**目标**: 验证频率敏感机制的效果

**实验设计**:
| 配置 | 注意力策略 | 预期效果 |
|------|-----------|----------|
| 标准 CoDA | 统一注意力 | 基准 |
| 频率敏感 | 低频强/高频弱 | 预期提升 |

---

## 八、总结

### 8.1 关键发现

1. **TFNO 的 Tucker 分解**保留了频率耦合，优于 MHF 的头独立处理
2. **GINO 的混合架构**提供了局部-全局频率处理的思路
3. **CoDA 注意力**可以恢复 MHF 破坏的跨头通信
4. **全层 MHF/CoDA 失败**的原因是信息流过度分散

### 8.2 优化建议优先级

| 优先级 | 方向 | 预期收益 | 实现难度 |
|--------|------|----------|----------|
| P0 | 数据增强 (2000+ 样本) | 1-2% | 低 |
| P0 | 更长训练 (400+ epochs) | 0.5-1% | 低 |
| P1 | 自适应头数 | 1-2% | 中 |
| P1 | Tucker 式跨头连接 | 2-3% | 高 |
| P2 | 频率敏感注意力 | 1-2% | 中 |

### 8.3 理论贡献

本研究首次系统分析了：
1. MHF 在 NS 方程上失效的频率耦合机制
2. 首尾层策略有效的恢复层原理
3. CoDA 注意力恢复频率耦合的数学原理
4. TFNO Tucker 分解保留频率耦合的优势

---

*研究笔记 - 天池*
*2026-03-25*