# MHF-FNO: Multi-Head Fourier Neural Operator

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**拷贝即用** 的 MHF-FNO 实现，在标准 FNO 基础上减少 **31% 参数**，在 Darcy Flow 2D 上精度提升 **30%**。

---

## 🚀 快速开始

```bash
# 安装依赖
pip install neuralop torch numpy

# 克隆仓库
git clone https://github.com/xuefenghao5121/mhf-fno.git
cd mhf-fno

# 运行示例
python examples/example.py
```

---

## 📊 测试结果

### 多数据集对比

| 数据集 | FNO参数 | MHF参数 | 参数变化 | FNO Loss | MHF Loss | Loss变化 |
|--------|---------|---------|----------|----------|----------|----------|
| **Darcy 2D** | 133,873 | 92,913 | **-30.6%** | 0.000074 | 0.000051 | **-30.2%** ✅ |
| Burgers 1D | 26,289 | 24,241 | -7.8% | 0.002335 | 0.002586 | +10.7% |
| Navier-Stokes 2D | 133,873 | 92,913 | -30.6% | 0.000298 | 0.000657 | +120% |

### Darcy Flow 2D (推荐场景)

| 指标 | FNO | MHF-FNO | 变化 |
|------|-----|---------|------|
| 参数量 | 133,873 | 92,913 | **-30.6%** ✅ |
| L2 Loss | 0.000074 | 0.000051 | **-30.2%** ✅ |
| 训练时间 | 52s | 50s | -4% |

---

## ⭐ 推荐配置

```python
from mhf_fno import MHFFNO

# 最佳配置 (经多数据集验证)
model = MHFFNO(
    n_modes=(8, 8),       # 频率模式数
    hidden_channels=32,   # 隐藏通道
    in_channels=1,        # 输入通道
    out_channels=1,       # 输出通道
    n_layers=3,           # FNO 层数
    n_heads=2,            # ⭐ 推荐 2
    mhf_layers=[0, 2],    # ⭐ 首尾层使用 MHF
)
```

---

## 📁 项目结构

```
mhf-fno/
├── mhf_fno/              # 核心模块
│   ├── __init__.py
│   ├── mhf_1d.py
│   ├── mhf_2d.py
│   └── mhf_fno.py
├── examples/             # 示例代码
│   ├── README.md
│   └── example.py
├── benchmark/            # 基准测试
│   ├── README.md
│   ├── run_benchmarks.py
│   ├── generate_data.py
│   └── BENCHMARK_GUIDE.md
├── data/                 # 数据集目录
│   └── README.md
├── README.md
├── requirements.txt
└── LICENSE
```

---

## 📖 论文参数参考

基于 FNO 论文 (Li et al., 2020) 和 PDEBench (Takamoto et al., 2022) 的标准参数：

### Darcy Flow 2D

| 参数 | FNO 论文 | PDEBench |
|------|----------|----------|
| 分辨率 | 16×16 | 421×421 |
| 训练样本 | 1,000 | 5,000 |
| 测试样本 | 100 | 500 |

### Burgers 1D

| 参数 | FNO 论文 | PDEBench |
|------|----------|----------|
| 空间分辨率 | 1024 | 1024 |
| 粘性系数 ν | 0.1 | 0.01 |

### Navier-Stokes 2D

| 参数 | FNO 论文 | PDEBench |
|------|----------|----------|
| 分辨率 | 64×64 | 128×128 |
| 粘度 ν | 1e-3 | 1e-3 ~ 1e-5 |

**参考文献**:
- FNO: Li et al., 2020. [arXiv:2010.08895](https://arxiv.org/abs/2010.08895)
- PDEBench: Takamoto et al., 2022. [arXiv:2210.07182](https://arxiv.org/abs/2210.07182)

---

## 🔧 运行基准测试

```bash
cd benchmark

# 生成数据
python generate_data.py --dataset darcy --n_train 500 --n_test 100

# 运行测试
python run_benchmarks.py --dataset darcy
```

详见 `benchmark/BENCHMARK_GUIDE.md`

---

## 引用

```bibtex
@misc{mhf-fno,
  author = {Tianyuan Team},
  title = {MHF-FNO: Multi-Head Fourier Neural Operator},
  year = {2026},
  url = {https://github.com/xuefenghao5121/mhf-fno}
}
```

---

## License

MIT License