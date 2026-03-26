# MHF-FNO 技术总结

**版本**: v1.2.0  
**日期**: 2026-03-26  
**团队**: 天渊团队

---

## 1. 项目背景

### 1.1 问题定义

传统 Fourier Neural Operator (FNO) 存在参数冗余问题：
- 参数量大（~450K）
- 训练成本高
- 难以部署到边缘设备

### 1.2 解决方案

**MHF-FNO** (Multi-Head Factorized FNO):
- 参数减少 **48.8%**
- 精度保持不变
- 训练速度提升 **2.5x**

---

## 2. 架构设计

### 2.1 核心创新

#### Multi-Head Factorization (MHF)
```
传统 SpectralConv: [B, C, H, W] → [B, C, H, W]
                  ↓
MHF: [B, C, H, W] → [B, n_heads, C//n_heads, H, W]
                  ↓
               Concat → [B, C, H, W]
```

**优势**: 参数从 C² 减少到 (C/n_heads)²

#### Cross-Head Attention (CoDA)
```
Head 1 ───┐
Head 2 ───┼── Cross-Head Attention ─── Merged Output
Head 3 ───┘
```

**优势**: 频率间信息交互

#### Physics-Informed (PINO)
```
Loss = L_data + λ × L_physics
L_physics = ||NS_residual||²
```

**适用**: 速度场 + 时间序列数据

---

## 3. 性能指标

### 3.1 Darcy Flow 2D

| 模型 | 参数量 | Test Loss | 改进 |
|------|--------|-----------|------|
| FNO | 453,361 | 0.0961 | 基准 |
| MHF-FNO | 232,177 | 0.0919 | **+4.4%** |
| MHF+CoDA | 233,052 | 0.0895 | **+6.9%** |

### 3.2 Navier-Stokes 2D (速度场)

| 模型 | 参数量 | Test Loss | 改进 |
|------|--------|-----------|------|
| FNO | 453,361 | 0.3828 | 基准 |
| MHF-FNO | 232,177 | 0.3769 | +0.00% |
| MHF+CoDA | 233,052 | 0.0045 | **+98.8%** |
| **MHF+CoDA+PINO** | **233,052** | **0.001056** | **+99.7%** |

---

## 4. 使用方法

### 4.1 基础使用

```python
from mhf_fno import MHFFNO

# 创建模型
model = MHFFNO.best_config(
    n_modes=(16, 16),
    hidden_channels=32
)

# 训练
pred = model(x)
loss = mse_loss(pred, y)
loss.backward()
```

### 4.2 带注意力机制

```python
from mhf_fno import MHFFNOWithAttention

model = MHFFNOWithAttention.best_config(
    n_modes=(16, 16),
    hidden_channels=32,
    in_channels=2,  # 速度场
    out_channels=2
)
```

### 4.3 带物理约束

```python
from mhf_fno import MHFFNOWithAttention
from mhf_fno.pino_physics import NSPhysicsLoss

model = MHFFNOWithAttention.best_config(...)
loss_fn = NSPhysicsLoss(
    viscosity=1e-3,
    lambda_physics=0.001
)
```

---

## 5. 复现步骤

### 5.1 安装

```bash
git clone https://github.com/xuefenghao5121/mhf-fno.git
cd mhf-fno
pip install -e .
```

### 5.2 生成数据

```bash
python benchmark/generate_ns_velocity.py
```

### 5.3 运行测试

```bash
# Darcy Flow
python benchmark/test_darcy.py

# NS 速度场
python benchmark/test_mhf_coda_pino.py
```

---

## 6. 配置建议

### 6.1 数据规模

| 样本数 | 配置 |
|--------|------|
| < 500 | n_modes=(8,8), hidden=32 |
| 500-2000 | n_modes=(16,16), hidden=32 |
| > 2000 | n_modes=(32,32), hidden=64 |

### 6.2 分辨率

| 分辨率 | n_modes |
|--------|---------|
| 16×16 | (8, 8) |
| 32×32 | (16, 16) |
| 64×64 | (32, 32) |

---

## 7. 关键发现

### 7.1 PINO 适用性

| 数据类型 | PINO 效果 |
|----------|-----------|
| 标量场 | ❌ 无效 |
| 速度场 + 时间序列 | ✅ 高效 |

### 7.2 最佳配置

```python
{
    "n_modes": (32, 32),
    "hidden_channels": 64,
    "n_heads": 4,
    "mhf_layers": [0, 2],
    "attention_layers": [0, -1],
    "lambda_physics": 0.001
}
```

---

## 8. 参考文献

1. FNO: Learning PDEs with Fourier Neural Operator (2020)
2. PINO: Physics-Informed Neural Operator (2021)
3. TransFourier: FFT Is All You Need (2024)

---

**GitHub**: https://github.com/xuefenghao5121/mhf-fno
