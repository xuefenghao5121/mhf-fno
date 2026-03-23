# MHF-FNO: Multi-Head Fourier Neural Operator

> 专注 MHF-FNO 的深入研究与优化

## 🎯 项目定位

**MHF (Multi-Head Fourier) 的核心价值**：

- **隐式正则化**：防止过拟合，提高泛化能力
- **泛化能力提升**：泛化差距减少 34.5%
- **测试精度提升**：+2.87%

## 📊 核心发现

### 泛化能力对比

| 模型 | 训练Loss | 测试Loss | 泛化差距 | 差距% |
|------|----------|----------|----------|-------|
| FNO (基准) | 0.0683 | 0.0957 | 0.0274 | 40.1% |
| MHF-FNO (随机) | 0.0735 | 0.0943 | 0.0208 | 28.3% |
| **MHF-FNO (尺度)** | **0.0736** | **0.0930** | **0.0193** | **26.2%** |

### 关键洞察

1. **隐式正则化**
   - MHF 训练 Loss 更高 → 阻止过度拟合
   - 泛化差距从 40.1% 降到 26.2%

2. **最佳配置**
   - 仅输入层 MHF 效果最好
   - 尺度多样性初始化最优

3. **多头多样性**
   - 初始化策略可提升多样性 140%
   - 多样性对泛化有正面影响

## 🔧 使用方法

```python
from mhf_fno import MHFSpectralConv
from neuralop.models import FNO

# 创建 FNO 模型
model = FNO(n_modes=(8, 8), hidden_channels=32, in_channels=1, out_channels=1, n_layers=3)

# 替换输入层为 MHF（推荐）
model.fno_blocks.convs[0] = MHFSpectralConv(32, 32, (8, 8), n_heads=4)
```

## 📁 项目结构

```
mhf_fno/
├── __init__.py
├── mhf_fno.py      # 核心 MHF SpectralConv
├── mhf_1d.py       # 1D 版本
└── mhf_2d.py       # 2D 版本

研究脚本:
├── mhf_research.py            # 深度研究
├── mhf_diversity_analysis.py  # 多样性分析
├── test_init_strategies.py    # 初始化策略测试
└── test_generalization.py     # 泛化能力测试
```

## 📚 研究成果

### 1. 泛化能力

- ✅ MHF 提供隐式正则化
- ✅ 泛化差距减少 34.5%
- ✅ 测试精度提升 2.87%

### 2. 初始化策略

| 策略 | 多样性提升 | 精度影响 |
|------|------------|----------|
| 频率带分离 | +49% | +0.1% |
| **尺度多样性** | **+140%** | **+2.87%** |

### 3. 边缘层配置

| 配置 | 参数变化 | 精度变化 |
|------|----------|----------|
| **仅输入层 MHF** | **-22.9%** | **+1.44%** |
| 边缘层 MHF | -45.9% | -1.00% |
| 全部 MHF | -68.8% | -37.86% ❌ |

## 📄 License

MIT License

---

**天渊团队** | 频域之渊，无穷探索