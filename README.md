# MHF-FNO: Multi-Head Fourier Neural Operator

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**MHF-FNO** 是一种基于多头注意力机制的 Fourier Neural Operator 变体，通过将频域卷积分解为多个头，实现参数效率的提升。

## 核心特性

- ✅ **参数减少 23-35%** - 通过多头结构减少参数量
- ✅ **精度相当** - 在 Darcy Flow 上 L2 误差仅 +4.9%
- ✅ **推理更快** - CPU 推理延迟降低 9%
- ✅ **即插即用** - 兼容 NeuralOperator 2.0.0

## 快速开始

### 安装

```bash
pip install -r requirements.txt
```

### 基本使用

```python
from mhf_fno import MHFFNO
import torch

# 创建模型
model = MHFFNO(
    n_modes=(16, 16),        # 频率模式数
    hidden_channels=32,      # 隐藏通道数
    in_channels=1,           # 输入通道数
    out_channels=1,          # 输出通道数
    n_layers=3,              # FNO 层数
    n_heads=4,               # MHF 头数
)

# 训练
x = torch.randn(32, 1, 16, 16)  # (batch, channel, height, width)
y = model(x)
```

### 运行基准测试

```bash
# 测试 Darcy Flow (数据内置)
python run_benchmarks.py --dataset darcy

# 测试 Burgers 方程 (需下载数据)
python run_benchmarks.py --dataset burgers

# 自定义参数
python run_benchmarks.py \
    --dataset darcy \
    --n_train 1000 \
    --n_test 200 \
    --epochs 50 \
    --batch_size 32
```

## 测试结果

### Darcy Flow (16×16)

| 模型 | 参数量 | L2 误差 | 推理延迟 |
|------|--------|---------|----------|
| FNO | 133,873 | 0.1022 | 3.59ms |
| **MHF-FNO** | **103,153** | 0.1072 | **3.26ms** |
| 变化 | **-22.9%** | +4.9% | **-9.2%** |

### 多头数量敏感性

| n_heads | 参数量 | L2 误差 | 推荐场景 |
|---------|--------|---------|----------|
| 2 | 113,393 | 0.1035 | 精度优先 |
| **4** | **103,153** | **0.1072** | **平衡（推荐）** |
| 8 | 98,033 | 0.1083 | 参数效率优先 |

## 配置建议

### 根据模型规模选择 n_heads

| 模型规模 | 建议 n_heads |
|----------|--------------|
| 大 (>10万参数) | 4-8 |
| 中 (5-10万参数) | 2-4 |
| 小 (<5万参数) | 1-2 |

### 根据分辨率选择 n_modes

| 分辨率 | 建议 n_modes |
|--------|--------------|
| 16×16 | (8, 8) |
| 32×32 | (16, 16) |
| 64×64 | (32, 32) |

**重要**: n_modes 应该覆盖足够的频率信息。对于低分辨率数据，建议 `n_modes ≥ resolution // 2`。

## 项目结构

```
mhf-fno/
├── mhf_fno/              # 核心模块
│   ├── __init__.py
│   ├── mhf_1d.py         # 1D MHF 实现
│   ├── mhf_2d.py         # 2D MHF 实现
│   └── mhf_fno.py        # MHF-FNO 模型
├── examples/             # 使用示例
│   └── basic_usage.py
├── run_benchmarks.py     # 基准测试脚本
├── BENCHMARK_GUIDE.md    # 详细测试指南
├── requirements.txt      # 依赖
├── setup.py              # 安装配置
└── README.md             # 本文件
```

## API 参考

### MHFFNO

```python
MHFFNO(
    n_modes: tuple,           # 频率模式数，如 (16, 16)
    hidden_channels: int,     # 隐藏通道数
    in_channels: int,         # 输入通道数
    out_channels: int,        # 输出通道数
    n_layers: int = 3,        # FNO 层数
    n_heads: int = 4,         # MHF 头数
    mhf_layers: list = None,  # 使用 MHF 的层索引，默认 [0, -1]
)
```

### MHFSpectralConv

```python
MHFSpectralConv(
    in_channels: int,
    out_channels: int,
    n_modes: tuple,
    n_heads: int = 4,
)
```

## 原理简介

MHF-FNO 的核心思想是将标准的频域卷积分解为多个头：

1. **多头分解**: 每个头独立学习频域权重
2. **尺度多样性初始化**: 不同头使用不同的初始化尺度
3. **混合配置**: 只在部分层使用 MHF，保持稳定性

### 与标准 FNO 的对比

| 特性 | FNO | MHF-FNO |
|------|-----|---------|
| 频域卷积 | 单一权重矩阵 | 多头分解 |
| 参数量 | O(C² × M²) | O((C/H)² × M² × H) |
| 参数减少 | - | 约 20-30% |

## 引用

如果您在研究中使用 MHF-FNO，请引用：

```bibtex
@misc{mhf-fno,
  author = {Tianyuan Team},
  title = {MHF-FNO: Multi-Head Fourier Neural Operator},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/xuefenghao5121/mhf-fno}
}
```

## 参考

- [Fourier Neural Operator](https://arxiv.org/abs/2010.08895)
- [NeuralOperator Library](https://github.com/neuraloperator/neuraloperator)
- [TransFourier: FFT Is All You Need](https://openreview.net/forum?id=TSHMAEItPc)

## License

MIT License - 详见 [LICENSE](LICENSE)

---

**团队**: 天渊团队 (Tianyuan Team)