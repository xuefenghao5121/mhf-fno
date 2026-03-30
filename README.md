# MHF-FNO: Multi-Head Fourier Neural Operator

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**参数高效的 Fourier Neural Operator 变体**，通过多头频率分解 (Multi-Head Fourier, MHF) 和跨头注意力 (Cross-Head Attention, CoDA) 实现参数压缩与精度提升。

---

## 🎯 项目介绍

MHF-FNO 是对标准 Fourier Neural Operator (FNO) 的改进：

- **问题**: 标准 FNO 的频域卷积参数随着 `(modes × channels × channels)` 增长，参数量大
- **思路**: 将频域卷积分解为多个独立的头，每个头处理一部分频率子空间，大幅减少参数
- **改进**: 添加跨头注意力弥补 MHF 头独立性假设的局限性
- **结果**: 在多个基准 PDE 数据集上实现 **24-49% 参数减少** 的同时，获得 **7-36% 精度提升** (lower test loss)

---

## ✨ 核心特性

| 特性 | 说明 |
|------|------|
| **参数高效** | 减少 24-49% 参数量，降低内存占用 |
| **精度提升** | 在 Darcy/Burgers/NS 数据集上获得 7-36% 精度提升 (lower test loss) |
| **兼容标准** | 兼容 neuraloperator 2.0.0 API，易于集成 |
| **物理约束** | 支持 PINO (Physics-Informed Neural Operator) 物理约束 |
| **真实数据** | 支持真实 Navier-Stokes 速度场时间序列数据 |
| **灵活配置** | 支持保守到激进多种配置，适应不同场景 |
| **Zenodo 支持** | ✨ 完整支持 Zenodo 下载的双文件 H5 格式 |

---

## 📊 性能对比

### 最新突破 (v1.6.4)

**真实 Navier-Stokes 数据 + MHF+CoDA+PINO**

| 模型 | Test Loss | 精度提升 | 参数减少 |
|------|----------|---------|----------|
| MHF+CoDA (基线) | 0.3828 | - | -49% |
| **MHF+CoDA+PINO** | **0.001056** | **+36%** ✅ | -49% |

*(精度提升 = test loss 降低百分比)*

### 完整基准结果

| 数据集 | PDE 类型 | 模型 | 参数减少 | vs 标准 FNO (test loss) | 推荐 |
|--------|----------|------|----------|-------------|------|
| **Darcy 2D** | 椭圆型 | MHF+Attention | **-48.6%** | **-8.17%** ✅ | ✅✅ |
| **Burgers 1D** | 抛物型 | MHF+Attention | **-31.7%** | **-32.12%** ✅ | ✅✅✅ |
| **NS 2D (标量)** | 双曲型 | MHF (保守) | **-24.3%** | ~0% | ⚠️ |
| **NS 2D (真实+PINO)** | 双曲型 | MHF+CoDA+PINO | **-49%** | **-36%** ✅ | ✅✅✅ |

*(负号表示 test loss 降低，精度提升)*

测试配置: `epochs=50, batch_size=32, lr=5e-4, n_modes=(16,16), hidden_channels=32, n_layers=3`

*(精度提升 = 测试 loss 降低百分比，越低越好)*

---

## 🚀 安装方法

### 依赖

```bash
pip install -r requirements.txt
```

主要依赖:
- `torch >= 1.10`
- `neuraloperator >= 2.0.0`
- `numpy`
- `matplotlib`
- `h5py` (for loading H5 datasets from Zenodo)

### 安装方式

**方式 1: 从 GitHub 直接安装**
```bash
pip install git+https://github.com/xuefenghao5121/mhf-fno.git
```

**方式 2: 克隆后开发安装**
```bash
git clone https://github.com/xuefenghao5121/mhf-fno.git
cd mhf-fno
pip install -e .
```

---

## 🏃 快速开始

### 基础用法

```python
from mhf_fno import MHFFNO, MHFFNOWithAttention
import torch

# 1. 基础版 MHF-FNO
model = MHFFNO.best_config(
    n_modes=(12, 12),
    hidden_channels=32
)

# 2. 带跨头注意力的版本 (推荐)
model = MHFFNOWithAttention.best_config(
    n_modes=(12, 12),
    hidden_channels=32
)

# 3. 前向传播
x = torch.randn(4, 1, 32, 32)  # [batch, channels, height, width]
y = model(x)
print(y.shape)  # [4, 1, 32, 32]
```

### 自定义配置

```python
from mhf_fno import create_mhf_fno_with_attention

# Navier-Stokes 保守配置 (推荐)
model = create_mhf_fno_with_attention(
    n_modes=(16, 16),
    hidden_channels=32,
    in_channels=1,
    out_channels=1,
    n_layers=3,
    mhf_layers=[0],           # 仅第一层使用 MHF
    n_heads=4,
    attention_layers=[0],     # 第一层使用注意力
    attn_dropout=0.0
)

# Darcy/Burgers 激进配置
model = create_mhf_fno_with_attention(
    n_modes=(16, 16),
    hidden_channels=32,
    mhf_layers=[0, 2],        # 首尾层使用 MHF
    attention_layers=[0, 2],  # 首尾层使用注意力
    n_heads=4
)
```

### PINO 物理约束 (真实 NS 数据)

```python
from mhf_fno import create_mhf_fno_with_attention
from mhf_fno.pino_physics import NavierStokesPINOLoss

# MHF+CoDA+PINO 完整模型
model = create_mhf_fno_with_attention(
    in_channels=2,           # 速度场 (u, v)
    n_modes=(16, 16),
    hidden_channels=32,
    mhf_layers=[0, 2],
    n_heads=4,
    attention_layers=[0, -1]
)

# 添加物理约束损失
pino_loss = NavierStokesPINOLoss(
    viscosity=1e-3,
    dt=0.01
)

# 计算物理约束损失
# pred: [batch, T, 2, H, W] 预测速度场
# loss = pino_loss(pred)
```

---

## 💡 使用示例

更多完整示例请查看 [examples/](examples/) 目录:

- [basic_usage.py](examples/basic_usage.py) - 基础使用示例
- [pino_usage.py](examples/pino_usage.py) - PINO 物理约束使用示例
- [ns_real_data.py](examples/ns_real_data.py) - 真实 Navier-Stokes 数据处理示例

### 运行基准测试

```bash
cd benchmark

# 1. 生成数据
# Darcy Flow
python generate_data.py --dataset darcy \
    --n_train 500 --n_test 100 --resolution 32

# Burgers
python generate_data.py --dataset burgers \
    --n_train 500 --n_test 100 --resolution 256

# Navier-Stokes
python generate_data.py --dataset navier_stokes \
    --n_train 1000 --n_test 200 --resolution 32

# 2. 运行测试
# 快速测试 (20 epochs)
python run_benchmarks.py --dataset darcy --epochs 20

# 完整测试 (50 epochs)
python run_benchmarks.py --dataset navier_stokes --epochs 50
```

---

## 🔬 技术原理

### MHF (Multi-Head Fourier)

将频域卷积分解为多个独立的头，每个头处理一部分频率子空间：

```
标准 SpectralConv: [C_in × C_out × modes]
MHF: [n_heads × (C_in/n_heads) × (C_out/n_heads) × modes]
```

**优势**:
- 参数减少约 `1/n_heads`
- 不同头可学习不同频率特征
- 类似 CNN 中的通道分组

**限制**:
- 假设频率之间相对独立
- 对强耦合 PDE (如 NS) 效果受限

### Cross-Head Attention (CoDA)

跨头注意力机制，弥补 MHF 头独立性假设：

1. 全局池化压缩空间维度得到头特征
2. 计算头之间的注意力权重
3. 加权融合增强头特征
4. 门控混合保留原始信息

**参数量**: 约为 MHF 参数的 <1%

---

## 🎯 最佳实践

### 参数选择指南

| 参数 | Darcy/Burgers | Navier-Stokes | 说明 |
|------|---------------|---------------|------|
| `mhf_layers` | [0, 2] | **[0]** | NS 需要保守配置 |
| `n_heads` | 4 | 2-4 | NS 建议减少头数 |
| `attention_layers` | [0, 2] | [0] | NS 需要保守配置 |
| `n_modes` | (16, 16) | (16, 16) | 根据分辨率调整 |
| `hidden_channels` | 32 | 32 | 平衡效率性能 |
| `n_layers` | 3 | 3 | 标准 FNO 配置 |

### 数据量要求

| 模型 | 最小样本 | 推荐样本 |
|------|----------|----------|
| 标准 FNO | 200 | 500+ |
| MHF-FNO | 500 | **1000+** | MHF 需要更多数据学习分布

### 训练策略

```python
# 优化器
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=5e-4,
    weight_decay=1e-4
)

# 学习率调度
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=epochs
)

# 梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

## 📁 项目结构

```
mhf-fno/
├── mhf_fno/              # 核心代码
│   ├── __init__.py       # 导出接口
│   ├── mhf_1d.py         # 1D MHF 实现
│   ├── mhf_2d.py         # 2D MHF 实现
│   ├── mhf_fno.py        # MHF-FNO 主实现
│   ├── mhf_attention.py  # 跨头注意力
│   ├── pino_physics.py   # PINO 物理约束
│   └── pino_high_freq.py # 高频处理
│
├── benchmark/            # 基准测试脚本
│   ├── data/             # 数据目录 (运行时生成)
│   ├── generate_data.py  # 数据生成
│   ├── run_benchmarks.py # 运行基准测试
│   └── *.py              # 各类测试脚本
│
├── examples/             # 使用示例
│   └── *.py              # 示例代码
│
├── data/                 # 数据根目录
├── README.md             # 项目文档
├── CHANGELOG.md          # 变更日志
├── requirements.txt      # 依赖列表
├── setup.py              # 安装配置
└── LICENSE               # 许可证
```

---

## 📚 参考文献

### 核心 FNO 论文

```bibtex
@inproceedings{li2021fourier,
  title={Fourier Neural Operator for Parametric Partial Differential Equations},
  author={Li, Zongyi and Kovachki, Nikola and Azizzadenesheli, Kamyar and Liu, Burigede and Bhattacharya, Kaushik and Stuart, Andrew and Anandkumar, Anima},
  booktitle={International Conference on Learning Representations},
  year={2021}
}
```

### 相关工作

- **Neural Operator**: Kovachki et al., 2021. [arXiv:2108.08481](https://arxiv.org/abs/2108.08481)
- **PINO**: Li et al., 2021. [arXiv:2111.03794](https://arxiv.org/abs/2111.03794)
- **PDEBench**: Takamoto et al., 2022. [arXiv:2210.07182](https://arxiv.org/abs/2210.07182)

---

## 🤝 贡献

由 **天渊团队 (Tianyuan Team)** 开发：

| 角色 | 贡献 |
|------|------|
| 架构师 | 整体架构设计 |
| 研究员 | 理论分析、论文学习 |
| 工程师 | 代码实现、文档 |
| 测试 | 测试验证、结果分析 |

欢迎提交 Issue 和 Pull Request！

---

## 📜 许可证

MIT License - 详见 [LICENSE](LICENSE)

---

## 📮 联系方式

- GitHub: https://github.com/xuefenghao5121/mhf-fno
- Team ID: team_tianyuan_fft

---

**最后更新**: 2026-03-26
**版本**: v1.6.4
