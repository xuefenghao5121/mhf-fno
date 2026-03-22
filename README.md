# MHF-FNO: Multi-Head Fourier Neural Operator Plugin

> NeuralOperator 2.0.0 兼容的 MHF (Multi-Head Fourier) 插件

## 📊 核心结果

在 NeuralOperator 官方 Darcy Flow Benchmark 上的测试结果：

| 模型 | 参数量 | L2误差 | 改进 |
|------|--------|--------|------|
| FNO-32 | 133,873 | 0.0961 | 基准 |
| **MHF-FNO (1+3层)** | **72,433** | **0.0919** | **-46%参数, -4.4%误差** ✅ |

---

## 🚀 安装

### 方法 1: 从 GitHub 安装 (推荐)

```bash
pip install git+https://github.com/xuefenghao5121/mhf-fno.git
```

### 方法 2: 手动安装

```bash
# 克隆仓库
git clone https://github.com/xuefenghao5121/mhf-fno.git
cd mhf-fno

# 安装
pip install -e .
```

### 依赖

```bash
pip install torch neuraloperator>=2.0.0
```

---

## 📚 NeuralOperator 集成指南

### 方法 1: 使用预设的最佳配置 (推荐)

```python
from mhf_fno import MHFFNO

# 一行创建最佳配置的模型
model = MHFFNO.best_config()

# 自定义参数
model = MHFFNO.best_config(
    n_modes=(16, 16),      # 更高分辨率
    hidden_channels=64,     # 更大模型
    in_channels=3,          # 多通道输入
    out_channels=1          # 输出通道
)
```

### 方法 2: 混合层配置

```python
from mhf_fno import create_hybrid_fno

# 指定哪些层使用 MHF
model = create_hybrid_fno(
    n_modes=(8, 8),
    hidden_channels=32,
    n_layers=3,
    mhf_layers=[0, 2],  # 第1和第3层使用 MHF
    n_heads=4
)
```

### 方法 3: 作为 FNO 的 conv_module 使用

```python
from neuralop.models import FNO
from mhf_fno import MHFSpectralConv
from functools import partial

# 创建 MHF 卷积层工厂
MHFConv = partial(MHFSpectralConv, n_heads=4)

# 全部层使用 MHF
model = FNO(
    n_modes=(8, 8),
    hidden_channels=32,
    in_channels=1,
    out_channels=1,
    n_layers=3,
    conv_module=MHFConv  # 替换所有 SpectralConv
)
```

---

## 🔥 完整训练示例

### 基础训练

```python
import torch
from torch.utils.data import DataLoader
from mhf_fno import MHFFNO

# 1. 创建模型
model = MHFFNO.best_config()

# 2. 准备数据
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# 3. 定义损失和优化器
from neuralop.losses.data_losses import LpLoss
loss_fn = LpLoss(d=2, p=2, reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# 4. 训练循环
for epoch in range(100):
    model.train()
    for batch in train_loader:
        x, y = batch['x'], batch['y']
        
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
    
    scheduler.step()
    
    # 评估
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for batch in test_loader:
            x, y = batch['x'], batch['y']
            pred = model(x)
            total_loss += loss_fn(pred, y).item()
        print(f"Epoch {epoch}: Test Loss = {total_loss / len(test_loader):.4f}")
```

### 使用 NeuralOperator 官方 Trainer

```python
from neuralop.training import Trainer
from neuralop.losses.data_losses import LpLoss
from mhf_fno import MHFFNO

# 创建模型
model = MHFFNO.best_config()

# 使用官方 Trainer
trainer = Trainer(
    model=model,
    n_epochs=100,
    learning_rate=1e-3,
    training_loss=LpLoss(d=2, p=2),
    eval_losses={'h1': LpLoss(d=2, p=2)},
)

# 开始训练
trainer.train(
    train_loader=train_loader,
    test_loaders={'test': test_loader},
)
```

### 使用官方 Darcy Flow 数据集

```python
from neuralop.data.datasets import load_darcy_flow_small
from mhf_fno import MHFFNO

# 加载官方数据集 (支持 16x16 和 32x32)
train_loader, test_loaders, data_processor = load_darcy_flow_small(
    n_train=1000,
    n_tests=[100, 100],
    batch_size=32,
    test_batch_sizes=[32, 32],
    test_resolutions=[16, 32]
)

# 创建模型
model = MHFFNO.best_config()

# 训练...
```

---

## 🎯 不同场景的配置建议

### 场景 1: 追求最高精度

```python
# 使用更大的 hidden_channels，保持混合配置
model = create_hybrid_fno(
    n_modes=(16, 16),
    hidden_channels=64,     # 更大模型
    n_layers=4,
    mhf_layers=[0, 2, 3],   # 更多层使用 MHF
    n_heads=8               # 更多头
)
```

### 场景 2: 追求最小参数量

```python
# 全部层使用 MHF
model = create_hybrid_fno(
    n_modes=(8, 8),
    hidden_channels=32,
    n_layers=3,
    mhf_layers=[0, 1, 2],   # 全部使用 MHF
    n_heads=8               # 更多头 = 更少参数
)
```

### 场景 3: 平衡性能与精度 (推荐)

```python
# 最佳配置：第1+3层使用 MHF
model = MHFFNO.best_config()
```

---

## 📊 详细测试结果

### 不同层配置对比

| 配置 | 参数量 | L2误差 | vs FNO-32 |
|------|--------|--------|-----------|
| FNO-32 | 133,873 | 0.0961 | 基准 |
| 第1层用MHF | 103,153 | 0.0939 | -23%, -2.7% |
| 第3层用MHF | 103,153 | 0.0951 | -23%, -1.5% |
| **第1+3层用MHF** | **72,433** | **0.0919** | **-46%, -4.4%** ✅ |
| 第2+3层用MHF | 72,433 | 0.0966 | -46%, +0.0% |
| 全部用MHF | 41,713 | 0.1266 | -69%, +31% |

### 运行稳定性 (3次运行)

| 模型 | 平均误差 | 标准差 |
|------|----------|--------|
| FNO-32 | 0.0961 | ±0.0028 |
| MHF-FNO | 0.0919 | ±0.0011 |

---

## 🔬 理论背景

### 标准 FNO SpectralConv

```
权重形状: (in_channels, out_channels, n_modes)
参数量: in_channels × out_channels × n_modes
```

### MHF SpectralConv (TransFourier)

```
权重形状: (n_heads, head_in, head_out, n_modes)
其中: head_in = in_channels // n_heads
     head_out = out_channels // n_heads
参数量: (in_channels × out_channels × n_modes) / n_heads
```

### 为什么混合配置更好？

| 层 | 作用 | 推荐 |
|---|---|---|
| 第1层 | 输入特征提取 | MHF ✅ |
| 中间层 | 特征变换 | 标准 SpectralConv |
| 最后层 | 输出整合 | MHF ✅ |

**原因**: 边缘层对参数敏感，MHF 的隐式正则化效果更好。

---

## 💡 设计理念与实际发现

### 原始设计理念 (TransFourier 论文)

MHF 最初来自 TransFourier 论文，核心目标是解决 Self-Attention 的复杂度问题：

```
Self-Attention: y_i = Σ_j α_ij · x_j  → O(n²) 复杂度
FFT 混合:      Y = W ⊙ FFT(x)         → O(n log n) 复杂度
```

**多头设计**:
- 每个头学习不同的频域权重
- 多头提供表达能力补偿
- 参数减少 `n_heads` 倍

### 实际测试发现

我们在 NeuralOperator 官方 FNO 上的测试揭示了重要偏差：

| 设计预期 | 实际测试 | 结论 |
|----------|----------|------|
| 全部层用 MHF | 全部用 MHF → 误差+31% ❌ | **预期偏差** |
| 多头越多越好 | n_heads=8 不如 n_heads=4 | **需要平衡** |
| 主要优势是参数效率 | 隐式正则化更关键 | **核心价值** |

### 关键洞察

#### 1. MHF 的真正价值：隐式正则化

```
FNO-32 训练曲线:
  Epoch 20:  L2=0.1184
  Epoch 100: L2=0.0960  (持续下降，但有过拟合趋势)

MHF-FNO 训练曲线:
  Epoch 20:  L2=0.1168
  Epoch 100: L2=0.0906  (稳定收敛)
```

**参数少 → 防止过拟合 → 更好泛化**

#### 2. 混合配置优于全 MHF

| 配置 | 参数量 | L2误差 | 分析 |
|------|--------|--------|------|
| 全部 MHF | 41,713 (-69%) | 0.1266 (+31%) | 参数太少，表达能力不足 |
| 第1+3层 MHF | 72,433 (-46%) | 0.0919 (-4.4%) | **平衡点** ✅ |
| 第1层 MHF | 103,153 (-23%) | 0.0939 (-2.7%) | 保守方案 |

**最佳策略**: 边缘层用 MHF (正则化)，中间层保留标准 (表达能力)

#### 3. 官方框架 vs 自定义 FNO

| 框架 | FNO 参数 | MHF 参数 | MHF 精度改进 |
|------|----------|----------|--------------|
| 自定义简化 FNO | 123,073 | 30,913 | **+11%** ✅✅ |
| 官方 neuralop FNO | 133,873 | 72,433 | **+4.4%** ✅ |

**原因**: 官方 FNO 有额外组件 (channel_mlp, skips)，稀释了 MHF 的优势

### 设计理念修正

```diff
- 原始理念: MHF = 全部替换 SpectralConv
- 目标: 参数效率 + 表达能力

+ 修正理念: MHF = 混合配置 (边缘层替换)
+ 目标: 参数效率 + 稳定性 + 防过拟合
```

### 实践建议

1. **首选混合配置**: `mhf_layers=[0, -1]` (第一层和最后一层)
2. **头数选择**: `n_heads=4` 是安全选择，更多头需要更多 `hidden_channels`
3. **官方框架**: 使用 `create_hybrid_fno()` 而非全部替换
4. **过拟合场景**: MHF 特别适合小数据集、高维任务

---

## 🔧 API 参考

### MHFSpectralConv

```python
class MHFSpectralConv(SpectralConv):
    """
    多头频谱卷积层，完全兼容 NeuralOperator API
    
    参数:
        in_channels: 输入通道数
        out_channels: 输出通道数
        n_modes: 频率模式数 (tuple for 2D, int for 1D)
        n_heads: 头数 (默认 4)
        bias: 是否使用偏置
    """
```

### create_hybrid_fno

```python
def create_hybrid_fno(
    n_modes: Tuple[int, ...],
    hidden_channels: int,
    in_channels: int = 1,
    out_channels: int = 1,
    n_layers: int = 3,
    mhf_layers: List[int] = [0, 2],
    n_heads: int = 4
) -> FNO:
    """创建混合 FNO 模型"""
```

### MHFFNO 预设配置

```python
class MHFFNO:
    @staticmethod
    def best_config(n_modes=(8, 8), hidden_channels=32, ...):
        """最佳配置: 第1+3层使用 MHF"""
    
    @staticmethod
    def full_mhf(n_modes=(8, 8), hidden_channels=32, ...):
        """全部层使用 MHF (不推荐)"""
```

---

## 📝 引用

```bibtex
@misc{mhf-fno2026,
  title={MHF-FNO: Multi-Head Fourier Neural Operator Plugin},
  author={Tianyuan Team},
  year={2026},
  howpublished={\url{https://github.com/xuefenghao5121/mhf-fno}}
}
```

## 📄 许可证

MIT License

## 🙏 致谢

- NeuralOperator 团队提供的优秀框架
- TransFourier 论文的 MHF 概念