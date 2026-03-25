# MHF-FNO: Multi-Head Fourier Neural Operator

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**参数高效的 Fourier Neural Operator 变体**，通过多头频率分解 (MHF) 和跨头注意力 (Cross-Head Attention) 实现：

- ✅ **参数减少 24-49%**
- ✅ **性能提升 7-32%** (Darcy/Burgers)
- ✅ **兼容 NeuralOperator 2.0.0**
- ⚠️ **NS 方程**需保守配置或物理约束

---

## 📊 性能基准

### 测试配置
- epochs=50, batch_size=32, lr=5e-4
- n_modes=(16,16), hidden_channels=32, n_layers=3

### 结果汇总

| 数据集 | PDE 类型 | 模型 | 参数减少 | vs FNO | 推荐 |
|--------|----------|------|----------|--------|------|
| **Darcy 2D** | 椭圆型 | MHF+Attention | **-48.6%** | **+8.17%** | ✅✅ |
| **Burgers 1D** | 抛物型 | MHF+Attention | **-31.7%** | **+32.12%** | ✅✅✅ |
| **NS 2D** | 双曲型 | MHF (保守) | **-24.3%** | ~0% | ⚠️ |

### NS 优化测试 (2026-03-26)

**保守配置**: `mhf_layers=[0]` (仅第一层使用 MHF)

| 模型 | 参数量 | 参数减少 | Test Loss | vs FNO |
|------|--------|----------|-----------|--------|
| FNO | 453,361 | - | 0.3753 | 基准 |
| MHF (保守) | 343,142 | **-24.3%** | ~0.3756 | ~0% |
| MHF (原始) | 232,177 | **-48.8%** | 0.3849 | -2.56% |

**结论**:
- 保守配置在 NS 上与 FNO 基本持平，参数减少 24%
- 原始配置参数减少更显著 (49%)，但性能略有下降
- **建议**: NS 方程使用保守配置 + PINO 物理约束

---

## 🚀 快速开始

### 安装

```bash
# 从 GitHub 安装
pip install git+https://github.com/xuefenghao5121/mhf-fno.git

# 或克隆安装
git clone https://github.com/xuefenghao5121/mhf-fno.git
cd mhf-fno
pip install -e .
```

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

# NS 方程保守配置 (推荐)
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

### Cross-Head Attention

跨头注意力机制，弥补 MHF 头独立性假设：

```python
# 轻量级 SENet 风格注意力
class CrossHeadAttention:
    def forward(self, head_outputs):
        # 1. 全局池化 [B, n_heads, C, H, W] -> [B, n_heads, C]
        pooled = global_avg_pool(head_outputs)

        # 2. 头间注意力 [B, n_heads, n_heads]
        attn = softmax(Q @ K.T / sqrt(d))

        # 3. 加权融合
        enhanced = attn @ V

        # 4. 门控混合
        return gate * enhanced + (1-gate) * original
```

**参数量**: 约为 MHF 参数的 <1%

---

## 📖 适用场景

### 推荐使用 MHF-FNO

| PDE 类型 | 示例 | 推荐配置 | 参数减少 | 性能提升 |
|----------|------|----------|----------|----------|
| **抛物型** | Burgers, 扩散 | mhf_layers=[0,2] | ~32% | **+32%** |
| **椭圆型** | Darcy, 热传导 | mhf_layers=[0,2] | ~49% | **+8%** |

### 谨慎使用 MHF-FNO

| PDE 类型 | 示例 | 推荐配置 | 参数减少 | 性能变化 |
|----------|------|----------|----------|----------|
| **双曲型** | Navier-Stokes | mhf_layers=[0] | ~24% | ~0% |

**原因**: NS 方程存在强频率耦合，MHF 的独立性假设不完全成立。

**建议**:
1. 使用保守配置 (`mhf_layers=[0]`)
2. 增加 PINO 物理约束
3. 使用更多训练数据 (1000+ 样本)

---

## 🎯 最佳实践

### 参数选择

| 参数 | Darcy/Burgers | Navier-Stokes | 说明 |
|------|---------------|---------------|------|
| `mhf_layers` | [0, 2] | **[0]** | NS 需保守 |
| `n_heads` | 4 | 2-4 | NS 可减少 |
| `attention_layers` | [0, 2] | [0] | NS 需保守 |
| `n_modes` | (16, 16) | (16, 16) | 根据分辨率调整 |
| `hidden_channels` | 32 | 32 | 平衡效率 |
| `n_layers` | 3 | 3 | 标准 FNO 配置 |

### 数据量要求

| 模型 | 最小样本 | 推荐样本 | 说明 |
|------|----------|----------|------|
| FNO | 200 | 500+ | 标准需求 |
| MHF-FNO | 500 | **1000+** | MHF 需更多数据 |

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
│   ├── mhf_fno.py        # MHF-FNO 主实现
│   └── mhf_attention.py  # 跨头注意力
│
├── benchmark/            # 基准测试
│   ├── generate_data.py  # 数据生成
│   ├── run_benchmarks.py # 运行测试
│   └── test_ns_*.py      # NS 优化测试
│
├── docs/                 # 文档
│   └── paper-notes/      # 论文笔记 (不上传 GitHub)
│
├── examples/             # 使用示例
├── data/                 # 数据目录 (需生成)
├── README.md
├── TIANYUAN_CONFIG.md    # 详细配置文档
├── NS_OPTIMIZATION_SUMMARY.md  # NS 优化报告
├── requirements.txt
├── setup.py
└── LICENSE
```

---

## 🔧 运行基准测试

### 1. 生成数据

```bash
cd benchmark

# Darcy Flow
python generate_data.py --dataset darcy \
    --n_train 500 --n_test 100 --resolution 32

# Burgers
python generate_data.py --dataset burgers \
    --n_train 500 --n_test 100 --resolution 256

# Navier-Stokes
python generate_data.py --dataset navier_stokes \
    --n_train 1000 --n_test 200 --resolution 32
```

### 2. 运行测试

```bash
# 快速测试 (20 epochs)
python run_benchmarks.py --dataset darcy --epochs 20

# 完整测试 (50 epochs)
python run_benchmarks.py --dataset navier_stokes --epochs 50

# NS 优化测试 (保守配置)
python test_ns_with_existing_data.py
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

| 角色 | 成员 | 贡献 |
|------|------|------|
| 架构师 | 天渊 | 整体架构设计 |
| 研究员 | 天池 | 理论分析、论文学习 |
| 工程师 | 天渠 | 代码实现、文档 |
| 测试 | 天井 | 测试验证、结果分析 |

欢迎提交 Issue 和 Pull Request！

---

## 📜 License

MIT License - 详见 [LICENSE](LICENSE)

---

## 📮 联系方式

- GitHub: https://github.com/xuefenghao5121/mhf-fno
- Team ID: team_tianyuan_fft

---

**最后更新**: 2026-03-26
**版本**: v1.2.0
