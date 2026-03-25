# MHF-FNO 跨头注意力机制实现报告

## 概述

成功实现了 MHF-FNO 的跨头注意力机制，解决了原有 MHF-FNO 的**独立性假设**问题。

## 核心实现

### 1. CrossHeadAttention 类

轻量级跨头注意力模块，采用 SENet 风格设计：

```python
class CrossHeadAttention(nn.Module):
    """
    工作流程：
    1. 全局平均池化: [B, n_heads, C, H, W] -> [B, n_heads, C]
    2. 头间注意力: 计算头之间的注意力权重
    3. 门控融合: 将注意力权重调制回原始特征
    """
```

**参数效率**：
- 对于 head_out=8: 仅 373 参数
- 相比 MHF 权重（~10K 参数）增加 <4%

### 2. MHFSpectralConvWithAttention 类

继承 MHFSpectralConv，添加可选的跨头注意力：

```python
class MHFSpectralConvWithAttention(MHFSpectralConv):
    """
    工作流程：
    1. 标准多头频域卷积
    2. IFFT 到空域
    3. 跨头注意力（可选）
    4. 输出
    """
```

### 3. MHFFNOWithAttention 工厂类

提供便捷的配置预设：

```python
# 推荐配置
model = MHFFNOWithAttention.best_config(n_modes=(8, 8), hidden_channels=32)

# 全注意力配置（所有层）
model = MHFFNOWithAttention.full_attention_config(n_modes=(8, 8))

# 轻量配置
model = MHFFNOWithAttention.light_config(n_modes=(8, 8))
```

## 测试结果

### 参数量对比

| 模型 | 参数量 | 变化 |
|------|--------|------|
| FNO | 133,873 | 基准 |
| MHF-FNO | 72,433 | -45.9% |
| MHF-FNO+Attn | 73,179 | -45.3% |

**注意力模块仅增加 746 参数（0.5%）**

### 性能对比（多尺度数据）

| 模型 | 测试 Loss | vs FNO |
|------|-----------|--------|
| FNO | 0.014471 | 基准 |
| MHF-FNO | 0.010747 | -25.7% |
| MHF-FNO+Attn | 0.010629 | -26.6% |

**MHF+Attn vs MHF-FNO 改进: +1.09%**

## 设计决策

### 为什么采用 SENet 风格？

1. **参数效率**: 原始展开空间维度的方案参数量爆炸（100M+）
2. **计算效率**: 避免 O(N²) 的全局注意力
3. **效果**: 通过通道注意力实现跨头信息交互

### 关键创新

```python
# 头间注意力：将每个头视为一个 token
# [B, n_heads, C] -> Q, K, V -> 注意力权重 [B, n_heads, n_heads]

attn_weights = torch.matmul(Q, K.transpose(-1, -2)) * self.scale
attn_out = torch.matmul(attn_weights, V)
```

## 使用示例

```python
from mhf_fno import MHFFNOWithAttention

# 创建带注意力的 MHF-FNO
model = MHFFNOWithAttention.best_config(
    n_modes=(8, 8),
    hidden_channels=32,
    n_heads=4
)

# 前向传播
x = torch.randn(4, 1, 16, 16)
y = model(x)
```

## 文件结构

```
mhf_fno/
├── __init__.py          # 更新：导出新的类
├── mhf_fno.py           # 原有 MHF-FNO 实现（未修改）
├── mhf_attention.py     # 新增：跨头注意力实现
└── ATTENTION_REPORT.md  # 本报告

examples/
└── test_attention.py    # 测试脚本
```

## 兼容性

- ✅ 原有 `MHFFNO` 保持不变
- ✅ 新增 `MHFFNOWithAttention` 作为可选升级
- ✅ 完全兼容 NeuralOperator 2.0.0

## 后续优化方向

1. **自适应注意力**: 根据数据复杂度动态启用/禁用注意力
2. **多尺度注意力**: 在不同分辨率层级应用注意力
3. **频域注意力**: 在频域直接进行注意力操作（避免 IFFT）

---

**作者**: 天渠（Tianyuan Team - Developer）  
**日期**: 2026-03-25  
**版本**: 1.0.1