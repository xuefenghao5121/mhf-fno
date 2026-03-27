# Examples 示例代码

本目录提供了 MHF-FNO 的各种使用示例，简洁清晰，易于上手。

## 📁 文件说明

| 文件名 | 功能 | 关键特性 |
|--------|------|----------|
| **basic_usage.py** | 🎯 基础 MHF 用法 | 仅使用 Multi-Head Frequency 模块 |
| **coda_usage.py** | 🚀 MHF + CoDA 用法 | 加入 Cross-Head Attention 机制 |
| **pino_usage.py** | 🔬 PINO 物理约束用法 | 加入 Physics-Informed Neural Operator 约束 |
| **ns_real_data.py** | 🌊 NS 时间序列数据用法 | 使用真实 Navier-Stokes 速度场数据 |
| **combined_usage.py** | 🧩 完整组合用法 | MHF + CoDA + PINO 最佳性能组合 |

---

## 🚀 快速上手

### 1. 基础 MHF 用法
```python
from mhf_fno import MHFFNO

# 仅使用 MHF 模块
model = MHFFNO(
    in_channels=1,
    out_channels=1,
    hidden_channels=32,
    n_modes=(16, 16),
    n_layers=4,
    mhf_layers=[0, 2],  # 在第1和第3层使用MHF
    use_coda=False,
)
```
运行示例：
```bash
python basic_usage.py
```

### 2. MHF + CoDA 增强
```python
model = MHFFNO(
    # ... 其他参数相同
    use_coda=True,  # 启用 Cross-Head Attention
)
```
运行示例：
```bash
python coda_usage.py
```

### 3. 加入物理约束 PINO
```python
model = MHFFNO(
    # ... 其他参数相同
    use_pino=True,  # 启用 PINO 物理约束
    pino_weight=0.1,  # 约束权重
)
```
运行示例：
```bash
python pino_usage.py
```

### 4. 完整组合最佳性能
```python
model = MHFFNO(
    # ... 其他参数相同
    mhf_layers=[0, 2],
    use_coda=True,
    use_pino=True,
    pino_weight=0.1,
)
```
运行示例：
```bash
python combined_usage.py
```

---

## 📊 性能对比

| 配置 | Darcy 2D 精度提升 | 参数减少 |
|------|------------------|----------|
| 纯 FNO | 0% | 0% |
| MHF | +8.17% | -48.6% |
| MHF + CoDA | +12.3% | -48.6% |
| MHF + CoDA + PINO | +36% | -49% | ⭐ 最佳

---

## 🔧 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `mhf_layers` | 使用 MHF 的层索引列表，从0开始 | `[]` (不使用) |
| `n_heads` | MHF 头数 | 2 |
| `use_coda` | 是否启用 Cross-Head Attention | False |
| `use_pino` | 是否启用物理约束 | False |
| `pino_weight` | 物理约束损失权重 | 0.1 |
