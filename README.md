# MHF-FNO

Multi-Head Fourier Neural Operator Plugin for NeuralOperator 2.0.0

## 📊 测试结果

### Darcy Flow 16×16

| 模型 | 参数量 | L2误差 | 改进 |
|------|--------|--------|------|
| FNO | 133,873 | 0.0957 | 基准 |
| **MHF-FNO** | **72,433** | **0.0966** | **-46%参数, +1%误差** |

## 🚀 安装

```bash
pip install git+https://github.com/xuefenghao5121/mhf-fno.git
```

## 📖 使用

```python
from mhf_fno import MHFSpectralConv, create_hybrid_fno, MHFFNO

# 方式 1: 使用预设最佳配置
model = MHFFNO.best_config()  # 边缘层 MHF, 参数-46%

# 方式 2: 自定义配置
model = create_hybrid_fno(
    n_modes=(8, 8),
    hidden_channels=32,
    mhf_layers=[0, 2],  # 第1层和第3层使用 MHF
    n_heads=4
)

# 方式 3: 手动替换
from neuralop.models import FNO
model = FNO(n_modes=(8, 8), hidden_channels=32, in_channels=1, out_channels=1, n_layers=3)
model.fno_blocks.convs[0] = MHFSpectralConv(32, 32, (8, 8), n_heads=4)
model.fno_blocks.convs[2] = MHFSpectralConv(32, 32, (8, 8), n_heads=4)
```

## 🔬 设计理念

MHF 的核心价值是**隐式正则化**，而不是参数压缩：

- **适用场景**: 小数据集 (<1000 样本)
- **最佳配置**: 边缘层 (第1层和最后1层) 使用 MHF
- **不推荐**: 全部层使用 MHF (会过度压缩)

详见 [OPERATOR_REVIEW.md](mhf_fno/OPERATOR_REVIEW.md)

## 📁 项目结构

```
mhf-fno/
├── mhf_fno/
│   ├── __init__.py
│   ├── mhf_fno.py          # 核心实现
│   ├── mhf_1d.py           # 1D 版本
│   └── mhf_2d.py           # 2D 版本
├── examples/
│   └── basic_usage.py
├── README.md
└── setup.py
```

## 📚 参考

1. TransFourier: FFT Is All You Need (OpenReview)
2. Fourier Neural Operators Explained (arXiv 2512.01421)
3. NeuralOperator: https://github.com/neuraloperator/neuraloperator

## 📄 License

MIT License