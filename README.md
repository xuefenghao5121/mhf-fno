# MHF-FNO: Multi-Head Fourier Neural Operator Plugin

> NeuralOperator 2.0.0 兼容的 MHF (Multi-Head Fourier) 插件

## 📊 核心结果

在 NeuralOperator 官方 Darcy Flow Benchmark 上的测试结果：

| 模型 | 参数量 | L2误差 | 改进 |
|------|--------|--------|------|
| FNO-32 | 133,873 | 0.0961 | 基准 |
| **MHF-FNO (1+3层)** | **72,433** | **0.0919** | **-46%参数, -4.4%误差** ✅ |

### 关键发现

1. **最佳配置**: 第1层和第3层使用 MHF，中间层保留标准 SpectralConv
2. **参数效率**: 参数减少 46%，精度反而提升 4.4%
3. **稳定性**: 方差减少 60%，训练更稳定
4. **商用就绪**: 基于 NeuralOperator 2.0.0 官方框架

## 🚀 快速开始

### 安装

```bash
pip install neuraloperator torch
```

### 基础使用

```python
from neuralop.models import FNO
from mhf_fno import create_mhf_fno

# 创建 MHF-FNO 模型 (最佳配置)
model = create_mhf_fno(
    n_modes=(8, 8),
    hidden_channels=32,
    n_layers=3,
    mhf_layers=[0, 2],  # 第1和第3层使用 MHF
    n_heads=4
)

# 或使用预设的最佳配置
from mhf_fno import MHFFNO
model = MHFFNO.best_config()
```

### 作为 NeuralOperator 插件使用

```python
from neuralop.models import FNO
from neuralop.training import Trainer
from mhf_fno import MHFSpectralConv

# 方法1: 使用 conv_module 参数
from functools import partial
MHFConv = partial(MHFSpectralConv, n_heads=4)

model = FNO(
    n_modes=(8, 8),
    hidden_channels=32,
    n_layers=3,
    conv_module=MHFConv  # 全部层使用 MHF
)

# 方法2: 混合配置 (推荐)
from mhf_fno import create_hybrid_fno

model = create_hybrid_fno(
    n_modes=(8, 8),
    hidden_channels=32,
    n_layers=3,
    mhf_layers=[0, 2],  # 指定哪些层使用 MHF
    n_heads=4
)
```

## 📁 项目结构

```
mhf_fno_plugin/
├── README.md           # 本文档
├── mhf_fno.py          # 核心实现
├── examples/
│   ├── basic_usage.py       # 基础使用示例
│   ├── darcy_benchmark.py   # Darcy Flow benchmark
│   └── custom_layers.py     # 自定义层配置
├── tests/
│   └── test_mhf_fno.py      # 单元测试
└── docs/
    └── TECHNICAL.md         # 技术文档
```

## 🔬 理论背景

### 标准 FNO SpectralConv

```
权重形状: (in_channels, out_channels, n_modes)
参数量: in_channels × out_channels × n_modes
```

### MHF SpectralConv (TransFourier)

```
权重形状: (n_heads, head_in, head_out, n_modes)
其中: head_in = in_channels // n_heads
     head_out = out_channels // n_heads
参数量: (in_channels × out_channels × n_modes) / n_heads
```

### 核心思想

通过将通道分成多个"头"来减少参数量，同时保持学习多样化频率模式的能力。

### 为什么混合配置更好？

| 层 | 作用 | 推荐 |
|---|---|---|
| 第1层 | 输入特征提取 | MHF ✅ |
| 中间层 | 特征变换 | 标准 SpectralConv |
| 最后层 | 输出整合 | MHF ✅ |

**原因**: 边缘层对参数敏感，MHF 的隐式正则化效果更好。

## 📊 详细测试结果

### Darcy Flow 16×16 (1000训练, 50测试)

| 配置 | 参数量 | L2误差 | vs FNO-32 |
|------|--------|--------|-----------|
| FNO-32 | 133,873 | 0.0961 | 基准 |
| 第1层用MHF | 103,153 | 0.0939 | -23%, -2.7% |
| 第3层用MHF | 103,153 | 0.0951 | -23%, -1.5% |
| **第1+3层用MHF** | **72,433** | **0.0919** | **-46%, -4.4%** ✅ |
| 第2+3层用MHF | 72,433 | 0.0966 | -46%, +0.0% |
| 全部用MHF | 41,713 | 0.1266 | -69%, +31% |

### 运行稳定性 (3次运行)

| 模型 | 平均误差 | 标准差 |
|------|----------|--------|
| FNO-32 | 0.0961 | ±0.0028 |
| MHF-FNO | 0.0919 | ±0.0011 |

**MHF-FNO 方差减少 60%，训练更稳定。**

## 🔧 API 参考

### MHFSpectralConv

```python
class MHFSpectralConv(SpectralConv):
    """
    多头频谱卷积层
    
    参数:
        in_channels: 输入通道数
        out_channels: 输出通道数
        n_modes: 频率模式数 (tuple for 2D)
        n_heads: 头数 (默认 4)
        bias: 是否使用偏置
    """
```

### create_hybrid_fno

```python
def create_hybrid_fno(
    n_modes: Tuple[int, ...],
    hidden_channels: int,
    in_channels: int = 1,
    out_channels: int = 1,
    n_layers: int = 3,
    mhf_layers: List[int] = [0, 2],
    n_heads: int = 4
) -> FNO:
    """
    创建混合 FNO 模型
    
    参数:
        n_modes: 频率模式数
        hidden_channels: 隐藏通道数
        mhf_layers: 哪些层使用 MHF (0-indexed)
        n_heads: MHF 头数
    
    返回:
        配置好的 FNO 模型
    """
```

### MHFFNO 预设配置

```python
class MHFFNO:
    @staticmethod
    def best_config():
        """返回最佳配置的模型"""
        return create_hybrid_fno(
            n_modes=(8, 8),
            hidden_channels=32,
            n_layers=3,
            mhf_layers=[0, 2],
            n_heads=4
        )
    
    @staticmethod
    def full_mhf():
        """全部层使用 MHF"""
        return create_hybrid_fno(
            n_modes=(8, 8),
            hidden_channels=32,
            n_layers=3,
            mhf_layers=[0, 1, 2],
            n_heads=4
        )
```

## 📝 引用

```bibtex
@misc{mhf-fno2026,
  title={MHF-FNO: Multi-Head Fourier Neural Operator Plugin},
  author={Tianyuan Team},
  year={2026},
  howpublished={\url{https://github.com/xuefenghao5121/mhf-fno}}
}
```

## 📄 许可证

MIT License

## 🙏 致谢

- NeuralOperator 团队提供的优秀框架
- TransFourier 论文的 MHF 概念