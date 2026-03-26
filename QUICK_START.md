# MHF-FNO 快速开始

5 分钟快速上手 MHF-FNO（Multi-Head Fourier Neural Operator）

---

## 📦 1. 安装

### 方法 1: 从 GitHub 安装（推荐）

```bash
pip install git+https://github.com/xuefenghao5121/mhf-fno.git
```

### 方法 2: 克隆安装

```bash
git clone https://github.com/xuefenghao5121/mhf-fno.git
cd mhf-fno
pip install -e .
```

### 依赖

```bash
pip install torch numpy neuralop
```

---

## 🚀 2. 基础使用

### 创建模型

```python
from mhf_fno import create_mhf_fno_with_attention

# 创建模型
model = create_mhf_fno_with_attention(
    n_modes=(16, 16),      # 频率模式数 = resolution // 2
    hidden_channels=32,    # 隐藏通道数
    in_channels=1,         # 输入通道
    out_channels=1,        # 输出通道
    mhf_layers=[0, 2],     # 哪些层使用 MHF
    n_heads=4,             # MHF 头数
    attention_layers=[0, -1]  # 注意力层
)

# 查看参数量
params = sum(p.numel() for p in model.parameters())
print(f"参数量: {params:,}")
```

### 训练示例

```python
import torch
import torch.nn as nn

# 准备数据
x = torch.randn(8, 1, 32, 32)  # [batch, channels, height, width]
y = torch.randn(8, 1, 32, 32)

# 训练
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

model.train()
optimizer.zero_grad()
y_pred = model(x)
loss = criterion(y_pred, y)
loss.backward()
optimizer.step()

print(f"Loss: {loss.item():.6f}")
```

---

## 🎯 3. 配置选择

### Darcy 2D / Burgers 1D（激进配置）

```python
# 推荐：性能提升 7-32%，参数减少 30-50%
model = create_mhf_fno_with_attention(
    n_modes=(16, 16),
    hidden_channels=32,
    mhf_layers=[0, 2],      # 首尾层使用 MHF
    n_heads=4,
    attention_layers=[0, 2]
)
```

**适用**: 椭圆型/抛物型 PDE（Darcy, Burgers, 热传导）

### Navier-Stokes 2D（保守配置）

```python
# 推荐：性能持平，参数减少 24%
model = create_mhf_fno_with_attention(
    n_modes=(16, 16),
    hidden_channels=32,
    mhf_layers=[0],         # 仅第一层使用 MHF
    n_heads=2,
    attention_layers=[0]
)
```

**适用**: 双曲型 PDE（Navier-Stokes, 对流扩散）

---

## 🔬 4. PINO 物理约束（实验性）

### 基础用法

```python
from mhf_fno.pino_high_freq import HighFreqPINOLoss

# 创建 PINO 损失
pino_loss = HighFreqPINOLoss(
    lambda_physics=0.0001,  # 物理损失权重
    threshold_ratio=0.5
)

# 训练循环
y_pred = model(x)
data_loss = criterion(y_pred, y)
physics_loss = pino_loss(y_pred)
total_loss = data_loss + physics_loss
```

### 参数调优

| 数据集 | 推荐 lambda_physics | 说明 |
|--------|-------------------|------|
| Darcy 2D | 1e-4 ~ 1e-3 | 静态场，效果明显 |
| Burgers 1D | 1e-5 ~ 1e-4 | 时间演化，适中 |
| NS 2D | 1e-5 ~ 1e-4 | 复杂流动，保守 |

---

## 📊 5. 性能基准

| 数据集 | vs FNO | 参数减少 | 推荐配置 |
|--------|--------|----------|----------|
| **Darcy 2D** | **+8.17%** | **-49%** | mhf=[0,2], heads=4 |
| **Burgers 1D** | **+32%** | **-32%** | mhf=[0,2], heads=4 |
| **NS 2D** | ~0% | **-24%** | mhf=[0], heads=2 |

---

## 📁 6. 完整示例

### 运行示例代码

```bash
# 基础示例
cd examples
python basic_usage.py

# PINO 示例
python pino_usage.py

# NS 真实数据
python ns_real_data.py
```

### 运行基准测试

```bash
cd benchmark

# 生成数据
python generate_data.py --dataset darcy --n_train 500

# 运行测试
python run_benchmarks.py --dataset darcy --epochs 50
```

---

## 🆘 7. 常见问题

### Q: 如何选择 `n_modes`？

**A**: 通常设置为 `resolution // 2`：
- 32x32 数据 → `n_modes=(16, 16)`
- 64x64 数据 → `n_modes=(32, 32)`

### Q: 训练不收敛怎么办？

**A**: 尝试以下方法：
1. 降低学习率（5e-4 → 1e-4）
2. 数据归一化（重要！）
3. 使用保守配置（`mhf_layers=[0]`）
4. 增加训练数据（推荐 1000+ 样本）

### Q: NS 方程为什么性能持平？

**A**: NS 存在强频率耦合，MHF 的独立性假设受限。使用保守配置 + PINO 可改善。

### Q: PINO 何时有效？

**A**: 适用于：
- ✅ 有明确物理方程
- ✅ 数据质量好
- ✅ 需要外推/泛化

---

## 📚 8. 下一步

### 深入学习

- **完整文档**: `README.md`
- **配置详解**: `TIANYUAN_CONFIG.md`
- **NS 优化**: `NS_OPTIMIZATION_SUMMARY.md`
- **PINO 报告**: `PINO_OPTIMIZATION_REPORT.md`

### 进阶主题

1. **自定义数据**: 修改 `examples/ns_real_data.py`
2. **PINO 调优**: 调整 `lambda_physics` 参数
3. **架构改进**: 实现新的注意力机制

### 贡献

欢迎提交 Issue 和 Pull Request！

---

## 📮 联系方式

- **GitHub**: https://github.com/xuefenghao5121/mhf-fno
- **Team**: 天渊团队 (team_tianyuan_fft)
- **版本**: v1.2.0

---

**最后更新**: 2026-03-26

**快速开始，快速上手！** 🚀
