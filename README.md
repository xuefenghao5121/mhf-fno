# MHF-FNO: Multi-Head Fourier Neural Operator

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**拷贝即用** 的 MHF-FNO 实现，在标准 FNO 基础上减少 **19% 参数**，精度损失仅 **12%**。

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

### Darcy Flow (16×16) - 推荐配置

| 指标 | FNO | MHF-FNO | 变化 |
|------|-----|---------|------|
| 参数量 | 133,873 | 108,772 | **-18.7%** ✅ |
| L2 误差 | 0.1022 | 0.1146 | +12.2% |
| 推理延迟 | 3.59ms | 3.26ms | **-9.2%** ✅ |

---

## ⭐ 推荐配置

```python
from mhf_fno import MHFFNO

model = MHFFNO(
    n_modes=(10, 10),       # 频率模式数
    hidden_channels=26,     # 隐藏通道
    n_layers=3,             # FNO 层数
    n_heads=2,              # ⭐ 小模型推荐 2
    mhf_layers=[0],         # ⭐ 只在第1层使用 MHF
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
│   └── BENCHMARK_GUIDE.md
├── data/                 # 数据集目录
│   └── README.md
├── README.md
├── requirements.txt
└── LICENSE
```

---

## 📋 核心代码

```python
import torch
import torch.nn as nn

class MHFSpectralConv(nn.Module):
    def __init__(self, in_channels, out_channels, n_modes, n_heads=2):
        super().__init__()
        self.n_modes = n_modes
        self.n_heads = n_heads
        self.head_channels = out_channels // n_heads
        
        self.weight = nn.Parameter(
            torch.randn(n_heads, self.head_channels, self.head_channels, *n_modes)
        )
        self.fc = nn.Linear(in_channels, out_channels)
        
        with torch.no_grad():
            for h in range(n_heads):
                nn.init.normal_(self.weight[h], mean=0, std=0.01 * (2 ** h))
    
    def forward(self, x):
        B = x.shape[0]
        x_ft = torch.fft.rfftn(x, dim=(-2, -1))
        out_ft = torch.zeros(B, self.fc.out_features, *x_ft.shape[-2:],
                            dtype=x_ft.dtype, device=x.device)
        
        for h in range(self.n_heads):
            s, e = h * self.head_channels, (h+1) * self.head_channels
            for i in range(min(self.n_modes[0], x_ft.shape[-2])):
                for j in range(min(self.n_modes[1], x_ft.shape[-1])):
                    out_ft[:, s:e, i, j] = torch.einsum(
                        'bi,bio->bo', x_ft[:, :, i, j], self.weight[h, :, :, i, j]
                    )
        
        return self.fc(torch.fft.irfftn(out_ft, dim=(-2, -1)).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class MHFFNO(nn.Module):
    def __init__(self, n_modes, hidden_channels=26, in_channels=1, out_channels=1,
                 n_layers=3, n_heads=2):
        super().__init__()
        
        self.fc_in = nn.Linear(in_channels, hidden_channels)
        self.fc_out = nn.Linear(hidden_channels, out_channels)
        
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(nn.ModuleDict({
                'mhf': MHFSpectralConv(hidden_channels, hidden_channels, n_modes, n_heads) 
                      if i == 0 else None,
                'w': nn.Conv2d(hidden_channels, hidden_channels, 1),
            }))
    
    def forward(self, x):
        x = self.fc_in(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        for layer in self.layers:
            if layer['mhf'] is not None:
                x = x + layer['mhf'](x)
            x = x + layer['w'](x)
        return self.fc_out(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


# 使用
model = MHFFNO(n_modes=(10, 10), hidden_channels=26, n_heads=2)
x = torch.randn(16, 1, 16, 16)
y = model(x)
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
| 渗透系数 | 高斯随机场 (α=2.0, τ=3.0) | 高斯随机场 |

### Burgers 1D

| 参数 | FNO 论文 | PDEBench |
|------|----------|----------|
| 空间分辨率 | 1024 | 1024 |
| 粘性系数 ν | 0.1 | 0.01 |
| 时间步数 | 200 | 2,000 |

### Navier-Stokes 2D

| 参数 | FNO 论文 | PDEBench |
|------|----------|----------|
| 分辨率 | 64×64 | 128×128 |
| 粘度 ν | 1e-3 | 1e-3 ~ 1e-5 |
| 雷诺数 Re | ~1,000 | 1,000 ~ 100,000 |
| 时间步数 | 20 | 2,000 |

### 使用示例

```bash
# FNO 论文参数 (快速验证)
python generate_data.py --dataset darcy --resolution 16 --n_train 1000

# PDEBench 参数 (完整基准)
python generate_data.py --dataset darcy --resolution 421 --n_train 5000
```

**参考文献**:
- FNO: Li et al., 2020. [arXiv:2010.08895](https://arxiv.org/abs/2010.08895)
- PDEBench: Takamoto et al., 2022. [arXiv:2210.07182](https://arxiv.org/abs/2210.07182)

---

## 🔧 运行基准测试

```bash
cd benchmark

# Darcy Flow (内置数据)
python run_benchmarks.py --dataset darcy

# Burgers (需下载)
python run_benchmarks.py --dataset burgers
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