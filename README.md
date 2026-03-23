# MHF-FNO: Multi-Head Fourier Neural Operator

> 专注 MHF-FNO 的深入研究与优化

## 🎯 项目定位

**MHF (Multi-Head Fourier) 的核心价值**：

- **多头多样性**：不同头学习不同的频域模式
- **隐式正则化**：块对角结构防止过拟合
- **边缘层优势**：在输入层效果最佳

## 📊 实验结果

### 边缘层 MHF 有效性

| 配置 | 参数变化 | 精度变化 |
|------|----------|----------|
| **仅输入层 MHF** | **-22.9%** | **+1.44%** ✅ |
| 仅输出层 MHF | -22.9% | -0.83% |
| 边缘层 MHF | -45.9% | -1.00% |
| 中间层 MHF | -22.9% | -0.82% |
| 全部 MHF | -68.8% | -37.86% ❌ |

### 多头多样性分析

| 指标 | 输入层 | 输出层 |
|------|--------|--------|
| 平均头间相似度 | 0.96 | 0.96 |
| 多样性得分 | 0.04 | 0.04 |

## 💡 关键发现

1. **输入层 MHF 效果最好**
2. **多头多样性不足**（相似度 96%）
3. **问题复杂度不是主要原因**
4. **需要显式多样性约束**

## 📁 项目结构

```
mhf_fno/
├── __init__.py
├── mhf_fno.py      # 核心 MHF SpectralConv
├── mhf_1d.py       # 1D 版本
└── mhf_2d.py       # 2D 版本
```

## 🔧 使用方法

```python
from mhf_fno import MHFSpectralConv
from neuralop.models import FNO

# 创建 FNO 模型
model = FNO(n_modes=(8, 8), hidden_channels=32, in_channels=1, out_channels=1, n_layers=3)

# 替换输入层为 MHF（推荐）
model.fno_blocks.convs[0] = MHFSpectralConv(32, 32, (8, 8), n_heads=4)
```

## 📚 研究问题

1. 如何提高多头多样性？
2. 输入层 MHF 为什么有效？
3. MHF 与 Multi-Head Attention 的关系？
4. MHF 的最佳应用场景？

## 📄 License

MIT License

---

**天渊团队** | 频域之渊，无穷探索