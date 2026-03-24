# MHF-FNO: Multi-Head Fourier Neural Operator

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**专注 Darcy Flow 2D** 的 MHF-FNO 实现，在标准 FNO 基础上减少 **31% 参数**，精度提升 **30%**。

---

## 🎯 适用场景

MHF-FNO 特别适合以下 PDE 类型：

| PDE 类型 | 特征 | MHF 效果 |
|----------|------|----------|
| **Darcy Flow 2D** ✅ | 椭圆型、扩散主导、平滑解 | **最佳** |
| Burgers 1D | 双曲型、对流主导 | 一般 |
| Navier-Stokes 2D | 混合型、湍流多尺度 | 不推荐 |

**推荐**: Darcy Flow、热传导、扩散方程等椭圆型 PDE

---

## 📊 Darcy Flow 2D 测试结果

### FNO vs MHF-FNO vs TFNO

| 模型 | 参数量 | 压缩率 | L2 Loss | vs FNO |
|------|--------|--------|---------|--------|
| **FNO** (基准) | 133,873 | - | 0.000074 | - |
| **MHF-FNO** | 92,913 | **-30.6%** | 0.000051 | **-30.2%** ✅ |
| **TFNO** (rank=0.5) | ~67,000 | -50% | ~0.00007 | ~-5% |

### 方法对比

| 特性 | FNO | MHF-FNO | TFNO |
|------|-----|---------|------|
| **压缩原理** | - | 通道分割 | Tucker分解 |
| **参数减少** | - | 30% | 50-90% |
| **Darcy效果** | 基准 | ✅ 最佳 | ⚠️ 需调参 |
| **易用性** | 简单 | 简单 | 需调rank |
| **推荐场景** | 通用 | Darcy/椭圆PDE | 高压缩需求 |

**结论**: Darcy Flow 场景推荐 MHF-FNO（参数-30%, 精度+30%）

---

## 🚀 快速开始

```bash
# 安装依赖
pip install neuralop>=2.0.0 torch numpy

# 克隆仓库
git clone https://github.com/xuefenghao5121/mhf-fno.git
cd mhf-fno

# 生成 Darcy 数据并测试
cd benchmark
python generate_data.py --dataset darcy --n_train 500 --n_test 100
python run_benchmarks.py --dataset darcy
```

---

## ⭐ 推荐配置

### MHF-FNO (Darcy Flow 推荐)

```python
from mhf_fno import create_hybrid_fno, MHFFNO

# 方法1: 使用预设配置 (推荐)
model = MHFFNO.best_config(n_modes=(8, 8), hidden_channels=32)

# 方法2: 自定义配置
model = create_hybrid_fno(
    n_modes=(8, 8),
    hidden_channels=32,
    n_heads=2,           # ⭐ 推荐 2
    mhf_layers=[0, 2],   # ⭐ 首尾层使用 MHF
)
```

### TFNO (高压缩需求)

```python
from neuralop.models import TFNO

# TFNO 使用 Tucker 分解压缩
# rank=0.5 约减少 50% 参数
model = TFNO(
    n_modes=(8, 8),
    in_channels=1,
    out_channels=1,
    hidden_channels=32,
    n_layers=3,
    rank=0.5,  # 压缩率：0.1=90%, 0.5=50%, 1.0=无压缩
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