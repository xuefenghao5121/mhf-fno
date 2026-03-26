# Examples - 使用示例

本目录包含 MHF-FNO 的完整使用示例，帮助您快速上手。

## 📁 文件说明

| 文件 | 说明 | 难度 |
|------|------|------|
| `basic_usage.py` | 基础使用示例，展示核心功能 | ⭐ 入门 |
| `pino_usage.py` | PINO 物理约束示例（实验性） | ⭐⭐ 进阶 |
| `ns_real_data.py` | Navier-Stokes 真实数据示例 | ⭐⭐⭐ 高级 |
| `example.py` | 完整训练测试示例 | ⭐⭐ 进阶 |

## 🚀 快速开始

### 1. 基础使用（推荐新手）

```bash
python basic_usage.py
```

**内容**:
- 创建 MHF-FNO 模型
- 前向传播和训练循环
- 不同场景的推荐配置
- 参数对比分析

**预期输出**:
```
✅ 模型创建成功
   参数量: 108,772
   配置: MHF layers=[0,2], Heads=4, Attention=[0,-1]

📋 使用总结
...
```

### 2. PINO 物理约束（进阶）

```bash
python pino_usage.py
```

**内容**:
- PINO 损失创建和使用
- 数据损失 + 物理损失组合
- lambda_physics 参数调优
- 自适应 lambda 调度

**适用场景**:
- ✅ 需要物理约束的问题
- ✅ Darcy Flow, 热传导等
- ⚠️ 实验性功能，需要调优

### 3. Navier-Stokes 真实数据（高级）

```bash
python ns_real_data.py
```

**内容**:
- 真实 NS 数据加载（PDEBench 格式）
- NS 推荐配置（保守）
- 完整训练流程
- PINO 可选集成
- 模型保存和加载

**数据格式**:
```python
# PyTorch 格式
data = torch.load('ns_data.pt')
x = data['x']  # [N, T, C, H, W] 或 [N, C, H, W]

# HDF5 格式（PDEBench）
import h5py
with h5py.File('ns_data.h5', 'r') as f:
    x = torch.from_numpy(f['input'][:])
```

## 📊 配置选择指南

### Darcy 2D / Burgers 1D（激进配置）

```python
model = create_mhf_fno_with_attention(
    n_modes=(16, 16),
    hidden_channels=32,
    mhf_layers=[0, 2],      # 首尾层
    n_heads=4,
    attention_layers=[0, 2]
)
```

**效果**: 参数减少 30-50%，性能提升 7-32%

### Navier-Stokes 2D（保守配置）

```python
model = create_mhf_fno_with_attention(
    n_modes=(16, 16),
    hidden_channels=32,
    mhf_layers=[0],         # 仅第一层
    n_heads=2,
    attention_layers=[0]
)
```

**效果**: 参数减少 24%，性能持平

## 🔧 运行要求

### 依赖

```bash
pip install torch numpy neuralop
```

### 可选依赖

```bash
# HDF5 数据支持
pip install h5py

# PDEBench 数据
pip install pydoe
```

### 硬件要求

| 模型大小 | 最小 GPU | 推荐 GPU | 训练时间 (50 epochs) |
|----------|----------|----------|---------------------|
| 小 (16x16) | 2GB | 4GB | ~5 分钟 |
| 中 (32x32) | 4GB | 8GB | ~15 分钟 |
| 大 (64x64) | 8GB | 16GB | ~1 小时 |

## 📖 学习路径

### 初学者

1. 运行 `basic_usage.py` - 了解基础概念
2. 阅读 `../README.md` - 理解整体架构
3. 尝试自己的数据集

### 进阶用户

1. 运行 `pino_usage.py` - 学习 PINO 约束
2. 阅读 `../TIANYUAN_CONFIG.md` - 深入配置参数
3. 调优 lambda_physics 参数

### 高级用户

1. 运行 `ns_real_data.py` - 真实 NS 测试
2. 实现 PINO 残差计算（时间序列）
3. 贡献新的示例或改进

## 🆘 常见问题

### Q: 为什么 NS 要用保守配置？

A: NS 方程存在强频率耦合，MHF 的独立性假设不完全成立。保守配置（仅第一层使用 MHF）能保持性能持平，同时减少 24% 参数。

### Q: PINO 什么时候有效？

A: PINO 适用于：
- ✅ 有明确物理方程的问题
- ✅ 数据质量好、噪声小
- ✅ 需要外推或泛化的场景

### Q: 如何选择 n_modes？

A: 通常设置为 `resolution // 2`，例如：
- 32x32 数据 → n_modes=(16, 16)
- 64x64 数据 → n_modes=(32, 32)

### Q: 训练不收敛怎么办？

A: 尝试：
1. 降低学习率（5e-4 → 1e-4）
2. 增加数据归一化
3. 使用更保守的配置（mhf_layers=[0]）
4. 增加训练数据（推荐 1000+ 样本）

## 📚 更多资源

- **完整文档**: `../README.md`
- **快速开始**: `../QUICK_START.md`
- **配置详解**: `../TIANYUAN_CONFIG.md`
- **NS 优化报告**: `../NS_OPTIMIZATION_SUMMARY.md`
- **PINO 报告**: `../PINO_OPTIMIZATION_REPORT.md`

## 🤝 贡献

欢迎提交新的示例或改进现有示例！

---

**天渊团队 (Tianyuan Team)**  
Team ID: team_tianyuan_fft
