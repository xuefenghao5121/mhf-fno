# MHF-FNO 项目基线报告

> 基线日期: 2026-03-22 23:56  
> 版本: v1.0.0  
> Git Commit: 45add5d

---

## 项目概述

**名称**: MHF-FNO (Multi-Head Fourier Neural Operator)  
**仓库**: https://github.com/xuefenghao5121/mhf-fno  
**兼容性**: NeuralOperator 2.0.0

---

## 核心成果

| 指标 | FNO-32 (基准) | MHF-FNO (最佳) | 改进 |
|------|--------------|----------------|------|
| **参数量** | 133,873 | 72,433 | **-45.9%** |
| **L2误差** | 0.0961 ± 0.0028 | 0.0919 ± 0.0011 | **-4.4%** |
| **稳定性** | 标准差 0.0028 | 标准差 0.0011 | **+60%** |

---

## 最佳配置

```
模型结构: FNO(n_modes=(8,8), hidden=32, n_layers=3)
MHF 层:   第1层 + 第3层 (mhf_layers=[0, 2])
头数:     4
```

---

## 关键发现

1. **混合配置 > 全 MHF**
   - 全 MHF: 参数-69%, 误差+31% ❌
   - 混合 (1+3层): 参数-46%, 误差-4.4% ✅

2. **边缘层最重要**
   - 第1层 (输入): MHF ✅
   - 中间层: 标准 SpectralConv ✅
   - 第3层 (输出): MHF ✅

3. **隐式正则化**
   - MHF 参数少 → 防止过拟合
   - FNO-32 过拟合趋势明显
   - MHF-FNO 收敛更稳定

---

## 测试矩阵

### 混合层配置测试

| 配置 | 参数量 | L2误差 | 结论 |
|------|--------|--------|------|
| 第1层 MHF | 103,153 | 0.0939 | ✅ 可用 |
| 第3层 MHF | 103,153 | 0.0951 | ✅ 可用 |
| **第1+3层 MHF** | **72,433** | **0.0919** | **✅ 最佳** |
| 第2+3层 MHF | 72,433 | 0.0966 | ⚠️ 一般 |
| 全部 MHF | 41,713 | 0.1266 | ❌ 不推荐 |

### Benchmark 环境

- 数据集: Darcy Flow 16×16
- 训练样本: 1,000
- 测试样本: 50
- Epochs: 150
- 运行次数: 3次取平均

---

## 文件清单

```
mhf_fno_plugin/
├── README.md              # 6.3KB - 完整使用指南
├── mhf_fno.py             # 6.5KB - 核心实现
├── __init__.py            # 239B - 包初始化
├── setup.py               # 348B - pip 安装
├── requirements.txt       # 40B - 依赖
├── benchmark_results.json # 502B - 测试结果
├── examples/
│   └── basic_usage.py     # 1.6KB - 使用示例
└── .gitignore             # 100B
```

---

## 安装与使用

### 安装

```bash
pip install git+https://github.com/xuefenghao5121/mhf-fno.git
```

### 基础使用

```python
from mhf_fno import MHFFNO

# 最佳配置 (参数-46%, 精度+4.4%)
model = MHFFNO.best_config()
```

### NeuralOperator 集成

```python
from neuralop.models import FNO
from mhf_fno import create_hybrid_fno

# 混合配置
model = create_hybrid_fno(
    n_modes=(8, 8),
    hidden_channels=32,
    mhf_layers=[0, 2]  # 第1+3层
)
```

---

## 后续计划

| 任务 | 优先级 | 状态 |
|------|--------|------|
| 32×32 分辨率测试 | 高 | ⏳ 数据下载问题 |
| Navier-Stokes 数据集 | 中 | 待开始 |
| PyPI 正式发布 | 低 | 待开始 |

---

## 基线确认

- [x] 核心功能完成
- [x] Benchmark 测试通过
- [x] 文档完整
- [x] GitHub 发布
- [x] 商用验证通过

**基线确认人**: 鸡你太美  
**确认日期**: 2026-03-22