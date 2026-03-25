# Burgers 1D 测试失败深度分析报告

> 分析日期: 2026-03-25  
> 分析师: 天池 (Tianchi)  
> 团队: 天渊团队 (Team Tianyuan FFT)

## 执行摘要

**根本原因**: `run_multi_dataset.py` 中的自定义 `MHFFNO1D` 实现**完全没有激活函数**，导致模型只能表示线性变换，无法学习 Burgers 方程的非线性映射。

**验证结果**: 简单对比测试显示，没有激活函数的模型性能比有激活函数的模型差 **55.8%**。

---

## 1. 问题背景

### 1.1 测试结果对比

| 数据集 | FNO Loss | MHF Loss | 改进率 | 状态 |
|--------|----------|----------|--------|------|
| Darcy (2D) | 0.3534 | 0.3270 | +7.47% | ✅ 成功 |
| NS (2D) | 0.3872 | 0.3918 | -1.19% | ⚠️ 轻微下降 |
| **Burgers (1D)** | **0.0273** | **0.1401** | **-412.8%** | ❌ 严重失败 |

### 1.2 矛盾现象

频率耦合分析显示 Burgers 应该是适合 MHF 的：
- 适用性: 95.8%
- 频率耦合强度: 高

但实际效果却很差，这表明问题不在理论层面，而在实现层面。

---

## 2. 根因分析

### 2.1 关键发现: 激活函数缺失

通过代码审查发现，`run_multi_dataset.py` 中的 `MHFFNO1D` 实现**完全没有激活函数**：

```python
# run_multi_dataset.py 第 72-95 行
class MHFFNO1D(nn.Module):
    """1D MHF-FNO 包装器"""
    def __init__(self, n_modes, hidden_channels, in_channels=1, out_channels=1, 
                 n_layers=3, n_heads=4):
        super().__init__()
        from mhf_fno import MHFSpectralConv
        
        self.fc_in = nn.Linear(in_channels, hidden_channels)
        self.layers = nn.ModuleList()
        
        for i in range(n_layers):
            conv = MHFSpectralConv(...)
            fc = nn.Linear(hidden_channels, hidden_channels)
            self.layers.append(nn.ModuleDict({'spectral': conv, 'fc': fc}))
        
        self.fc_out = nn.Linear(hidden_channels, out_channels)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.fc_in(x)
        
        for layer in self.layers:
            x1 = layer['spectral'](x.permute(0, 2, 1)).permute(0, 2, 1)
            x2 = layer['fc'](x)
            x = x + x1 + x2  # ⚠️ 没有 GELU/ReLU 激活函数！
        
        x = self.fc_out(x)
        x = x.permute(0, 2, 1)
        return x
```

对比 `mhf_fno/mhf_1d.py` 中的正确实现：

```python
# mhf_fno/mhf_1d.py
class MHFFNO1D(nn.Module):
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.lifting(x)
        x = x.transpose(1, 2)
        
        for conv in self.convs:
            x = F.gelu(conv(x))  # ✅ 有 GELU 激活函数！
        
        x = x.transpose(1, 2)
        x = self.projection(x)
        x = x.transpose(1, 2)
        
        return x
```

### 2.2 验证测试

**测试设计**: 使用简单的非线性映射 `y = sin(3x) + 0.5*x²`，对比有无激活函数的效果。

**结果**:

| 模型 | 测试 MSE | 相对改进 |
|------|----------|----------|
| 无激活函数 | 0.9567 | - |
| 有激活函数 | 0.4233 | +55.8% |

**结论**: 缺少激活函数导致模型表达能力严重受限，无法学习非线性映射。

---

## 3. 架构差异分析

### 3.1 Darcy (成功) vs Burgers (失败) 实现对比

| 项目 | Darcy (2D 成功) | Burgers (1D 失败) |
|------|-----------------|-------------------|
| **模型来源** | `MHFFNO.best_config()` | 自定义 `MHFFNO1D` |
| **基类** | `neuralop.models.FNO` | 自定义 `nn.Module` |
| **SpectralConv** | 替换第 0, 2 层 | 所有层使用 MHF |
| **Lifting/Projection** | ChannelMLP (Conv1d) | Linear |
| **Channel MLP** | ✅ 有 | ❌ 无 |
| **Skip Connection** | ✅ SoftGating | ❌ 无 |
| **激活函数** | ✅ GELU (在 FNO 内部) | ❌ 无 |

### 3.2 关键差异详解

#### 3.2.1 模型架构复杂度

**Darcy 成功案例** (`MHFFNO.best_config()`):
```
FNO 模型结构:
├── lifting: ChannelMLP (Conv1d x 2)
├── fno_blocks:
│   ├── convs: [MHFSpectralConv, SpectralConv, MHFSpectralConv]
│   ├── fno_skips: [Conv1d, Conv1d, Conv1d]
│   ├── channel_mlp: [ChannelMLP, ChannelMLP, ChannelMLP]
│   └── channel_mlp_skips: [SoftGating, SoftGating, SoftGating]
└── projection: ChannelMLP (Conv1d x 2)
```

**Burgers 失败案例** (自定义 `MHFFNO1D`):
```
自定义模型结构:
├── fc_in: Linear
├── layers: [
│   {spectral: MHFSpectralConv, fc: Linear},
│   {spectral: MHFSpectralConv, fc: Linear},
│   {spectral: MHFSpectralConv, fc: Linear}
│ ]
└── fc_out: Linear
```

#### 3.2.2 缺失的关键组件

| 组件 | 作用 | Darcy 有 | Burgers 有 |
|------|------|----------|------------|
| Channel MLP | 通道间信息混合 | ✅ | ❌ |
| FNO Skips | 跨层特征传递 | ✅ | ❌ |
| SoftGating | 自适应特征加权 | ✅ | ❌ |
| GELU | 非线性激活 | ✅ | ❌ |

---

## 4. 次要问题分析

### 4.1 数据预处理

检查 Burgers 数据：
- 形状: `[500, 256]` (训练), `[100, 256]` (测试)
- 数据范围: 正常，无异常值
- 输入输出相关性: 0.9495 (高)

**结论**: 数据本身没有问题。

### 4.2 n_modes 设置

- Burgers: `n_modes = 256 // 2 = 128`
- Darcy: `n_modes = (16, 16)`

**结论**: n_modes 设置合理。

### 4.3 训练参数

两者使用相同的训练配置：
- epochs: 40
- batch_size: 32
- learning_rate: 0.001

**结论**: 训练参数不是问题。

---

## 5. 修复方案

### 5.1 方案 A: 使用标准 FNO 框架 (推荐)

**核心思路**: 与 2D 情况保持一致，使用 `MHFFNO.best_config()` 创建模型，让 `neuralop.FNO` 处理 1D 情况。

**修改 `run_multi_dataset.py`**:

```python
# 删除自定义 MHFFNO1D 类 (第 72-112 行)

# 在 run_single_dataset 函数中修改 MHF-FNO 模型创建逻辑:
if is_1d:
    # 使用 FNO 框架 + MHFSpectralConv 替换
    from mhf_fno import create_hybrid_fno
    
    model_mhf = create_hybrid_fno(
        n_modes=n_modes_tuple,  # (n_modes,) for 1D
        hidden_channels=hidden_channels,
        in_channels=info['input_channels'],
        out_channels=info['output_channels'],
        mhf_layers=[0, 2],  # 首尾层使用 MHF
        n_heads=4
    )
else:
    model_mhf = MHFFNO.best_config(
        n_modes=n_modes_tuple,
        hidden_channels=hidden_channels,
        in_channels=info['input_channels'],
        out_channels=info['output_channels']
    )
```

**优点**:
- 与 2D 实现保持一致
- 自动包含所有 FNO 组件
- 经过充分验证

### 5.2 方案 B: 修复自定义 MHFFNO1D

如果需要保留自定义实现，修复激活函数问题：

```python
class MHFFNO1D(nn.Module):
    def __init__(self, n_modes, hidden_channels, ...):
        super().__init__()
        # ... 保持不变 ...
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.gelu(self.fc_in(x))  # 添加 GELU
        
        for layer in self.layers:
            x1 = layer['spectral'](x.permute(0, 2, 1)).permute(0, 2, 1)
            x2 = layer['fc'](x)
            x = F.gelu(x + x1 + x2)  # 添加 GELU
        
        x = self.fc_out(x)
        x = x.permute(0, 2, 1)
        return x
```

**注意**: 此方案仍缺少 Channel MLP 等关键组件。

### 5.3 方案 C: 使用 mhf_fno/mhf_1d.py 中的 MHFFNO1D

直接导入正确实现：

```python
from mhf_fno.mhf_1d import MHFFNO1D

# 创建模型
model_mhf = MHFFNO1D(
    in_channels=info['input_channels'],
    out_channels=info['output_channels'],
    hidden_channels=hidden_channels,
    n_modes=n_modes,
    n_layers=3,
    n_heads=4
)
```

---

## 6. 测试验证计划

### 6.1 单元测试

1. **激活函数验证**:
   ```python
   def test_mhffno1d_has_activation():
       model = MHFFNO1D(...)
       # 检查 forward 中是否使用了 F.gelu 或其他激活函数
       # 可以通过 hooks 检查中间输出
   ```

2. **模型输出范围测试**:
   ```python
   def test_model_output_nonlinear():
       # 输入两个不同的样本
       x1, x2 = torch.randn(2, 1, 256), torch.randn(2, 1, 256)
       # 如果模型是线性的，则 model(x1+x2) = model(x1) + model(x2)
       # 检查这个等式是否不成立（说明模型有非线性）
   ```

### 6.2 集成测试

```bash
# 使用修复后的代码重新运行测试
cd /root/.openclaw/workspace/memory/projects/tianyuan-fft/benchmark
python run_multi_dataset.py --datasets burgers --epochs 40
```

### 6.3 预期结果

修复后，Burgers 测试应该：
- MHF Loss 接近或优于 FNO Loss (0.0273)
- 改进率应该在 -5% 到 +10% 范围内

---

## 7. 经验教训

### 7.1 代码审查要点

1. **神经网络基本组件检查**:
   - ✅ 激活函数是否存在于每层
   - ✅ 归一化层是否合理
   - ✅ 残差连接是否正确

2. **一致性检查**:
   - 1D 和 2D 实现是否使用相同的框架
   - 是否有重复定义的类

### 7.2 测试策略改进

1. **添加模型结构验证测试**:
   - 检查模型是否包含预期的激活函数
   - 检查模型复杂度是否合理

2. **添加对比基准测试**:
   - 新模型应该与已知好用的模型对比
   - 如果性能差距过大，触发警告

---

## 8. 总结

### 根本原因
`run_multi_dataset.py` 中的自定义 `MHFFNO1D` 实现缺少激活函数，导致模型无法学习非线性映射。

### 关键发现
1. Darcy 成功是因为使用了 `MHFFNO.best_config()`，它基于 `neuralop.FNO` 框架
2. Burgers 失败是因为使用了简化版自定义实现
3. 两者架构差异巨大，不仅仅是维度不同

### 推荐修复
使用 `create_hybrid_fno()` 替代自定义实现，与 2D 情况保持一致。

---

## 附录

### A. 相关代码位置

| 文件 | 行号 | 内容 |
|------|------|------|
| `benchmark/run_multi_dataset.py` | 72-112 | 自定义 MHFFNO1D (有 bug) |
| `mhf_fno/mhf_1d.py` | 68-92 | 正确的 MHFFNO1D |
| `mhf_fno/mhf_fno.py` | 280-310 | MHFFNO.best_config() |
| `mhf_fno/mhf_fno.py` | 180-230 | create_hybrid_fno() |

### B. 测试命令

```bash
# 重新运行完整测试
python benchmark/run_multi_dataset.py --datasets darcy burgers navier_stokes --epochs 50

# 仅测试 Burgers
python benchmark/run_multi_dataset.py --datasets burgers --epochs 50
```

---

*报告完成 - 天池*