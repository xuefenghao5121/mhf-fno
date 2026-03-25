# MHF-FNO 代码审查报告

**审查人**: 天渠 (天渊团队 - developer)  
**审查日期**: 2026-03-25  
**代码版本**: v1.1.0

---

## 📋 审查范围

| 文件 | 状态 |
|------|------|
| `mhf_fno/mhf_fno.py` | ✅ 正确 |
| `mhf_fno/mhf_1d.py` | ✅ 正确 |
| `mhf_fno/mhf_2d.py` | ✅ 正确 |
| `benchmark/run_benchmarks.py` | ⚠️ 已修复 |
| `benchmark/compare_fno_mhf_tfno.py` | ✅ 正确 |
| `benchmark/generate_data.py` | ✅ 正确 |
| `examples/example.py` | ⚠️ 已修复 |

---

## 🔴 发现的问题

### 问题 1: `run_benchmarks.py` MHFFNO 实现错误 (严重)

**位置**: `benchmark/run_benchmarks.py` 原第 68-88 行

**问题描述**:
非 MHF 层使用 `nn.Identity()` 替代频域卷积，导致跳过频域处理：
```python
# 原代码 (错误)
if use_mhf:
    conv = MHFSpectralConv(...)
else:
    conv = nn.Identity()  # ❌ 错误
```

**影响**: 混合配置下模型行为与标准 FNO 不一致，基准测试结果不准确。

**修复方案**: 使用核心库的 `create_hybrid_fno` 函数，确保非 MHF 层正确使用标准 SpectralConv。

**状态**: ✅ 已修复

---

### 问题 2: `run_benchmarks.py` MHFSpectralConv 实现不一致 (严重)

**位置**: `benchmark/run_benchmarks.py` 原第 14-59 行

**问题描述**:
1. 使用实部/虚部分离存储复数权重，而非核心实现的 `torch.cfloat`
2. 在 `forward` 中添加了额外的 `self.fc` 线性变换

**影响**: 
- 参数量和计算行为与核心实现不一致
- 基准测试结果无法反映真实 MHF-FNO 性能

**修复方案**: 导入核心库的 `MHFSpectralConv` 和 `create_hybrid_fno`。

**状态**: ✅ 已修复

---

### 问题 3: `examples/example.py` 独立实现与核心不一致 (中等)

**位置**: `examples/example.py` 原第 34-100 行

**问题描述**:
示例代码包含完全独立的 `MHFFNO` 类实现，与核心库 `create_hybrid_fno` 不一致。

**影响**: 用户可能使用错误的实现方式。

**修复方案**: 重写示例代码，使用核心库 API。

**状态**: ✅ 已修复

---

## ✅ 核心实现审查结果

### `mhf_fno/mhf_fno.py` 

**维度正确性**: ✅ 通过
- 1D: 输入 (B, C, L) → 输出 (B, C, L)
- 2D: 输入 (B, C, H, W) → 输出 (B, C, H, W)
- einsum 公式正确：`'bhif,hiof->bhof'` (1D), `'bhiXY,hioXY->bhoXY'` (2D)

**数值稳定性**: ✅ 通过
- 使用 `min()` 确保模式数不超过频率维度
- rfft/rfft2 输出长度处理正确
- 复数权重使用 `torch.cfloat` 类型

**梯度流**: ✅ 通过
- 梯度正确传播到输入
- 权重梯度正常

**NeuralOperator 兼容性**: ✅ 通过
- 继承自 `SpectralConv`，API 兼容
- `create_hybrid_fno` 正确替换 FNO 层的卷积

**回退机制**: ✅ 通过
- 通道数不能被 n_heads 整除时，自动回退到标准卷积
- 发出 UserWarning 提示用户

---

## 📊 测试结果

```
============================================================
测试 MHFSpectralConv 维度正确性
============================================================

[1] 1D 测试
输入: torch.Size([2, 32, 64]), 输出: torch.Size([2, 32, 64])
✅ 1D 形状正确

[2] 2D 测试
输入: torch.Size([2, 32, 16, 16]), 输出: torch.Size([2, 32, 16, 16])
✅ 2D 形状正确

[3] 梯度流测试
输入梯度形状: torch.Size([2, 32, 16, 16])
✅ 梯度传播正确

[4] 回退机制测试
✅ 回退机制正确

============================================================
测试 create_hybrid_fno
============================================================
输入: torch.Size([2, 1, 16, 16]), 输出: torch.Size([2, 1, 16, 16])
✅ create_hybrid_fno 正确

============================================================
验证 compare_fno_mhf_tfno.py 中的用法
============================================================
FNO 参数量: 133,873
MHF-FNO 参数量: 92,913
参数减少: 30.6%
✅ compare_fno_mhf_tfno.py 用法正确
```

---

## 🔧 已应用的修复

### 1. `benchmark/run_benchmarks.py`
- 删除独立的 `MHFSpectralConv` 类
- 删除独立的 `MHFFNO` 类
- 导入核心库：`from mhf_fno import MHFSpectralConv, create_hybrid_fno`
- 使用核心库创建模型

### 2. `examples/example.py`
- 删除独立的 `MHFSpectralConv` 类
- 删除独立的 `MHFFNO` 类
- 使用核心库的 `create_hybrid_fno` 函数
- 添加正确的导入说明

---

## 📝 建议改进

### 1. 代码整合
`mhf_1d.py` 和 `mhf_2d.py` 提供了独立但功能相似的实现。建议：
- 保留作为轻量级独立使用选项
- 在文档中明确说明核心实现是 `mhf_fno.py`

### 2. 单元测试
建议添加正式的单元测试文件 `tests/test_mhf_fno.py`，覆盖：
- 维度正确性测试
- 梯度流测试
- 边界条件测试
- 回退机制测试

### 3. 文档
建议添加 API 文档，说明：
- `MHFSpectralConv` 参数和用法
- `create_hybrid_fno` 最佳实践
- `MHFFNO.best_config` 和 `MHFFNO.light_config` 预设

---

## ✅ 结论

核心实现 `mhf_fno/mhf_fno.py` 正确无误，维度处理、梯度流和 NeuralOperator 兼容性均通过测试。

测试代码和示例代码中发现的实现不一致问题已修复。修复后的代码统一使用核心库 API，确保行为一致性。

**审查状态**: ✅ 通过 (已修复所有问题)