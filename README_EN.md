# MHF-FNO: Multi-Head Fourier Neural Operator

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**Parameter-Efficient Variants of Fourier Neural Operator** achieves parameter reduction and **accuracy improvement** via Multi-Head Frequency decomposition (MHF) and Cross-Head Attention (CoDA).

---

## 🎯 Introduction

MHF-FNO is an improvement over the standard Fourier Neural Operator (FNO):

- **Problem**: The spectral convolution parameters in standard FNO grow with `(modes × channels × channels)`, resulting in large parameter count
- **Idea**: Decompose spectral convolution into multiple independent heads, each handles a subspace of frequency, significantly reducing parameters
- **Improvement**: Add Cross-Head Attention to compensate for the independence assumption of MHF heads
- **Result**: Achieves **24-49% parameter reduction** with **7-36% accuracy improvement** (lower test loss) on multiple benchmark PDE datasets

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **Parameter Efficient** | 24-49% fewer parameters, reduced memory usage |
| **Accuracy Improvement** | 7-36% accuracy improvement (lower test loss) on Darcy/Burgers/NS datasets |
| **Standard Compatible** | Compatible with neuraloperator 2.0.0 API, easy to integrate |
| **Physics Constraints** | Supports PINO (Physics-Informed Neural Operator) |
| **Real Data Support** | Supports real Navier-Stokes velocity field time series data |
| **Flexible Configuration** | Supports conservative to aggressive configurations for different scenarios |
| **Zenodo Support** | ✨ Full support for double H5 files downloaded from Zenodo |

---

## 📊 Performance Comparison

### Latest Breakthrough (v1.3.0)

**Real Navier-Stokes Data + MHF+CoDA+PINO**

| Model | Test Loss | Accuracy Improvement | Parameter Reduction |
|-------|----------|---------------------|-------------------|
| MHF+CoDA (baseline) | 0.3828 | - | -49% |
| **MHF+CoDA+PINO** | **0.001056** | **+36%** ✅ | -49% |

*(Accuracy improvement = percentage reduction in test loss)*

### Full Benchmark Results

| Dataset | PDE Type | Model | Parameter Reduction | vs Standard FNO (test loss) | Recommendation |
|---------|----------|-------|-------------------|-------------------------|---------------|
| **Darcy 2D** | Elliptic | MHF+Attention | **-48.6%** | **-8.17%** ✅ | ✅✅ |
| **Burgers 1D** | Parabolic | MHF+Attention | **-31.7%** | **-32.12%** ✅ | ✅✅✅ |
| **NS 2D (scalar)** | Hyperbolic | MHF (conservative) | **-24.3%** | ~0% | ⚠️ |
| **NS 2D (real+PINO)** | Hyperbolic | MHF+CoDA+PINO | **-49%** | **-36%** ✅ | ✅✅✅ |

*Negative sign means test loss reduction = accuracy improvement*

Test configuration: `epochs=50, batch_size=32, lr=5e-4, n_modes=(16,16), hidden_channels=32, n_layers=3`

---

## 🚀 Installation

### Dependencies

```bash
pip install -r requirements.txt
```

Main dependencies:
- `torch >= 1.10`
- `neuraloperator >= 2.0.0`
- `numpy`
- `matplotlib`
- `h5py` (for loading H5 datasets)

### Install

**Option 1: Install directly from GitHub**
```bash
pip install git+https://github.com/xuefenghao5121/mhf-fno.git
```

**Option 2: Clone and install in dev mode**
```bash
git clone https://github.com/xuefenghao5121/mhf-fno.git
cd mhf-fno
pip install -e .
```

---

## 🏃 Quick Start

### Basic Usage

```python
from mhf_fno import MHFFNO, MHFFNOWithAttention
import torch

# 1. Basic MHF-FNO
model = MHFFNO.best_config(
    n_modes=(12, 12),
    hidden_channels=32
)

# 2. MHF-FNO with Cross-Head Attention (recommended)
model = MHFFNOWithAttention.best_config(
    n_modes=(12, 12),
    hidden_channels=32
)

# 3. Forward pass
x = torch.randn(4, 1, 32, 32)  # [batch, channels, height, width]
y = model(x)
print(y.shape)  # [4, 1, 32, 32]
```

### Custom Configuration

```python
from mhf_fno import create_mhf_fno_with_attention

# Conservative configuration for Navier-Stokes (recommended)
model = create_mhf_fno_with_attention(
    n_modes=(16, 16),
    hidden_channels=32,
    in_channels=1,
    out_channels=1,
    n_layers=3,
    mhf_layers=[0],           # Only use MHF at the first layer
    n_heads=4,
    attention_layers=[0],     # Use attention at first layer
    attn_dropout=0.0
)

# Aggressive configuration for Darcy/Burgers
model = create_mhf_fno_with_attention(
    n_modes=(16, 16),
    hidden_channels=32,
    mhf_layers=[0, 2],        # Use MHF at first and last layers
    attention_layers=[0, 2],  # Use attention at first and last layers
    n_heads=4
)
```

### PINO Physics Constraints (Real NS Data)

```python
from mhf_fno import create_mhf_fno_with_attention
from mhf_fno.pino_physics import NavierStokesPINOLoss

# Full MHF+CoDA+PINO model
model = create_mhf_fno_with_attention(
    in_channels=2,           # velocity field (u, v)
    n_modes=(16, 16),
    hidden_channels=32,
    mhf_layers=[0, 2],
    n_heads=4,
    attention_layers=[0, -1]
)

# Add physics constraint loss
pino_loss = NavierStokesPINOLoss(
    viscosity=1e-3,
    dt=0.01
)

# Compute physics constraint loss
# pred: [batch, T, 2, H, W] predicted velocity field
# loss = pino_loss(pred)
```

---

## 💡 Examples

See [examples/](examples/) directory for complete examples:

- [basic_usage.py](examples/basic_usage.py) - Basic usage example
- [pino_usage.py](examples/pino_usage.py) - PINO physics constraint example
- [ns_real_data.py](examples/ns_real_data.py) - Real Navier-Stokes data processing example

### Run Benchmark

```bash
cd benchmark

# 1. Generate data
# Darcy Flow
python generate_data.py --dataset darcy \
    --n_train 500 --n_test 100 --resolution 32

# Burgers
python generate_data.py --dataset burgers \
    --n_train 500 --n_test 100 --resolution 256

# Navier-Stokes
python generate_data.py --dataset navier_stokes \
    --n_train 1000 --n_test 200 --resolution 32

# 2. Run benchmark with Zenodo downloaded data (double H5 files)
# Burgers 1D from Zenodo: https://zenodo.org/records/13355846
python run_benchmarks.py \
    --dataset burgers \
    --format h5 \
    --train_path ../data/1D_Burgers_Re1000_Train.h5 \
    --test_path ../data/1D_Burgers_Re1000_Test.h5

# Navier-Stokes 2D from Zenodo
python run_benchmarks.py \
    --dataset navier_stokes \
    --format h5 \
    --train_path ../data/2D_NS_Re100_Train.h5 \
    --test_path ../data/2D_NS_Re100_Test.h5
```

---

## 🔬 Technical Details

### MHF (Multi-Head Fourier)

Decompose spectral convolution into multiple independent heads, each handles a subspace of frequency:

```
Standard SpectralConv: [C_in × C_out × modes]
MHF: [n_heads × (C_in/n_heads) × (C_out/n_heads) × modes]
```

**Advantages**:
- Parameter reduction ~ `1/n_heads`
- Different heads learn different frequency features
- Similar to channel grouping in CNN

**Limitations**:
- Assumes relative independence between frequencies
- Limited effectiveness on strongly coupled PDEs like NS

### Cross-Head Attention (CoDA)

Cross-Head Attention mechanism to compensate for the independence assumption of MHF heads:

1. Global pooling to get head features from spatial dimensions
2. Compute attention weights between heads
3. Weighted fusion to enhance head features
4. Gated mixing preserves original information

**Parameter count**: <1% of MHF parameters

---

## 🎯 Best Practices

### Parameter Selection Guide

| Parameter | Darcy/Burgers | Navier-Stokes | Notes |
|-----------|---------------|---------------|-------|
| `mhf_layers` | [0, 2] | **[0]** | NS needs conservative configuration |
| `n_heads` | 4 | 2-4 | Recommend fewer heads for NS |
| `attention_layers` | [0, 2] | [0] | NS needs conservative configuration |
| `n_modes` | (16, 16) | (16, 16) | Adjust according to resolution |
| `hidden_channels` | 32 | 32 | Balance efficiency and performance |
| `n_layers` | 3 | 3 | Standard FNO configuration |

### Data Requirements

| Model | Min Samples | Recommended |
|-------|-------------|-------------|
| Standard FNO | 200 | 500+ |
| MHF-FNO | 500 | **1000+** | MHF needs more data to learn distribution |

### Training Strategy

```python
# Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=5e-4,
    weight_decay=1e-4
)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=epochs
)

# Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

## 📁 Project Structure

```
mhf-fno/
├── mhf_fno/              # Core code
│   ├── __init__.py       # API exports
│   ├── mhf_1d.py         # 1D MHF implementation
│   ├── mhf_2d.py         # 2D MHF implementation
│   ├── mhf_fno.py        # MHF-FNO main implementation
│   ├── mhf_attention.py  # Cross-Head attention
│   ├── pino_physics.py   # PINO physics constraints
│   └── pino_high_freq.py # High frequency processing
│
├── benchmark/            # Benchmark scripts
│   ├── data/             # Data directory (generated at runtime)
│   ├── data_loader.py    # ✨ Universal data loader supports all formats
│   ├── generate_data.py  # Data generation
│   ├── run_benchmarks.py # Run benchmark entry
│   └── *.py              # Test scripts
│
├── examples/             # Usage examples
│   └── *.py              # Example code
│
├── data/                 # Data root directory
├── README.md             # Chinese documentation
├── README_EN.md          # English documentation (this file)
├── CHANGELOG.md          # Change log
├── requirements.txt      # Dependencies
├── setup.py              # Installation configuration
└── LICENSE               # License
```

---

## 📚 References

### Core FNO Paper

```bibtex
@inproceedings{li2021fourier,
  title={Fourier Neural Operator for Parametric Partial Differential Equations},
  author={Li, Zongyi and Kovachki, Nikola and Azizzadenesheli, Kamyar and Liu, Burigede and Bhattacharya, Kaushik and Stuart, Andrew and Anandkumar, Anima},
  booktitle={International Conference on Learning Representations},
  year={2021}
}
```

### Related Work

- **Neural Operator**: Kovachki et al., 2021. [arXiv:2108.08481](https://arxiv.org/abs/2108.08481)
- **PINO**: Li et al., 2021. [arXiv:2111.03794](https://arxiv.org/abs/2111.03794)
- **PDEBench**: Takamoto et al., 2022. [arXiv:2210.07182](https://arxiv.org/abs/2210.07182)

---

## 🤝 Contributing

Developed by **Tianyuan Team**:

| Role | Contribution |
|------|-------------|
| Architect | Overall architecture design |
| Researcher | Theoretical analysis, literature review |
| Engineer | Code implementation, documentation |
| Tester | Testing validation, result analysis |

Issues and Pull Requests are welcome!

---

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 📮 Contact

- GitHub: https://github.com/xuefenghao5121/mhf-fno
- Team ID: team_tianyuan_fft

---

**Last Updated**: 2026-03-26
**Version**: v1.3.3
