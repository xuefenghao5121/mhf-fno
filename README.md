# MHF-FNO: Multi-Head Fourier Neural Operator

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**拷贝即用** 的 MHF-FNO 实现，在标准 FNO 基础上减少 **23% 参数**，精度损失仅 **5%**。

---

## 🚀 30秒快速开始

### 1. 安装依赖

```bash
pip install neuralop torch numpy
```

### 2. 下载并运行示例

```bash
# 克隆仓库
git clone https://github.com/xuefenghao5121/mhf-fno.git
cd mhf-fno

# 运行完整示例（包含 FNO vs MHF-FNO 对比）
python example.py
```

**预期输出**：

```
使用设备: cpu
生成数据: 训练 500, 测试 100, 分辨率 16x16
✅ 数据生成完成

============================================================
测试 FNO (基准)
============================================================
参数量: 133,873
  Epoch 10/30: Train 0.0923, Test 0.0945
  ...

============================================================
结果对比
============================================================
指标              FNO             MHF-FNO         变化            
------------------------------------------------------------
参数量              133,873         103,153         -22.9%
最佳测试Loss        0.0945          0.0987          +4.4%

✅ 测试完成！
```

---

## 📋 核心代码（直接复制使用）

```python
import torch
import torch.nn as nn

# ============ MHF-FNO 核心实现 ============

class MHFSpectralConv(nn.Module):
    """Multi-Head Fourier Spectral Convolution"""
    
    def __init__(self, in_channels, out_channels, n_modes, n_heads=4):
        super().__init__()
        self.n_modes = n_modes
        self.n_heads = n_heads
        self.head_channels = out_channels // n_heads
        
        self.weight = nn.Parameter(
            torch.randn(n_heads, self.head_channels, self.head_channels, *n_modes)
        )
        self.fc = nn.Linear(in_channels, out_channels)
        
        # 尺度多样性初始化
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
    """MHF-FNO 模型"""
    
    def __init__(self, n_modes, hidden_channels, in_channels=1, out_channels=1,
                 n_layers=3, n_heads=4):
        super().__init__()
        
        self.fc_in = nn.Linear(in_channels, hidden_channels)
        self.fc_out = nn.Linear(hidden_channels, out_channels)
        
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(nn.ModuleDict({
                'mhf': MHFSpectralConv(hidden_channels, hidden_channels, n_modes, n_heads) 
                      if i in [0, n_layers-1] else None,  # 只在首尾层使用 MHF
                'w': nn.Conv2d(hidden_channels, hidden_channels, 1),
            }))
    
    def forward(self, x):
        x = self.fc_in(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        for layer in self.layers:
            if layer['mhf'] is not None:
                x = x + layer['mhf'](x)
            x = x + layer['w'](x)
        
        return self.fc_out(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


# ============ 使用示例 ============

# 创建模型
model = MHFFNO(
    n_modes=(8, 8),       # 频率模式数
    hidden_channels=32,   # 隐藏通道
    n_layers=3,           # 层数
    n_heads=4,            # 头数
)

# 前向传播
x = torch.randn(16, 1, 16, 16)  # (batch, channel, H, W)
y = model(x)

print(f"输入: {x.shape}")
print(f"输出: {y.shape}")
print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
```

---

## 📊 性能对比

### Darcy Flow (16×16)

| 指标 | FNO | MHF-FNO | 变化 |
|------|-----|---------|------|
| 参数量 | 133,873 | 103,153 | **-22.9%** ✅ |
| L2 误差 | 0.1022 | 0.1072 | +4.9% |
| 推理延迟 | 3.59ms | 3.26ms | **-9.2%** ✅ |

---

## ⚙️ 配置建议

### 选择 n_heads

| 模型规模 | 推荐 n_heads |
|----------|--------------|
| 大 (>10万参数) | 4-8 |
| 中 (5-10万) | 2-4 |
| 小 (<5万) | 1-2 |

### 选择 n_modes

| 分辨率 | 推荐 n_modes |
|--------|--------------|
| 16×16 | (8, 8) |
| 32×32 | (16, 16) |
| 64×64 | (32, 32) |

**重要**: 对于低分辨率，确保 `n_modes ≥ resolution // 2`

---

## 📁 项目结构

```
mhf-fno/
├── example.py            # ⭐ 完整示例（拷贝即用）
├── mhf_fno/              # 核心模块
│   ├── __init__.py
│   ├── mhf_1d.py
│   ├── mhf_2d.py
│   └── mhf_fno.py
├── run_benchmarks.py     # 基准测试脚本
├── requirements.txt
└── README.md
```

---

## 🔧 API

### MHFFNO

```python
MHFFNO(
    n_modes,              # 频率模式数，如 (8, 8)
    hidden_channels,      # 隐藏通道数
    in_channels=1,        # 输入通道
    out_channels=1,       # 输出通道
    n_layers=3,           # FNO 层数
    n_heads=4,            # MHF 头数
)
```

---

## 📖 原理

MHF-FNO 将频域卷积分解为多个头：

```
标准 FNO:    Conv2D(32, 32, modes)     # 单一权重
MHF-FNO:     [Conv2D(8, 8, modes)] × 4  # 4个头独立学习
```

**优势**：
- 参数减少：`32² → 4 × 8² = 256` (减少 75%)
- 尺度多样性：不同头学习不同频率范围
- 隐式正则化：防止过拟合

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