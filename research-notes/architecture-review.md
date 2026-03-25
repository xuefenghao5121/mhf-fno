# MHF-FNO 架构审查报告

> **审查员**: 天渊（架构师）  
> **日期**: 2026-03-25  
> **团队**: 天渊团队

---

## 一、项目结构审查

### 1.1 目录组织

```
mhf-fno/
├── mhf_fno/              # 核心模块 ✅
│   ├── __init__.py       # API 入口
│   ├── mhf_fno.py        # 主要实现 (与 NeuralOperator 集成)
│   ├── mhf_1d.py         # 独立 1D 实现
│   └── mhf_2d.py         # 独立 2D 实现
├── benchmark/            # 性能测试 ✅
├── examples/             # 使用示例 ✅
├── research-notes/       # 研究笔记 ✅
├── setup.py              # 包配置 ✅
└── requirements.txt      # 依赖声明 ✅
```

### 1.2 优点

| 方面 | 评价 |
|------|------|
| 模块划分 | ✅ 清晰分离核心代码、测试、示例 |
| 包结构 | ✅ 标准 Python 包，支持 pip 安装 |
| 文档 | ✅ README 详尽，包含论文参考和快速开始 |
| Git 管理 | ✅ 有 .gitignore，版本历史清晰 |

### 1.3 问题

| 问题 | 严重性 | 建议 |
|------|--------|------|
| **代码重复**: `mhf_1d.py`/`mhf_2d.py` 与 `mhf_fno.py` 功能重叠 | 🔴 高 | 统一到 `MHFSpectralConv`，移除独立实现或标记为 deprecated |
| **缺少测试目录**: 无 `tests/`，测试代码混在示例中 | 🔴 高 | 创建 `tests/` 目录，使用 pytest 组织单元测试 |
| **缺少类型标记文件**: 无 `py.typed` | 🟡 中 | 添加 `py.typed` 支持 mypy 类型检查 |
| **缺少 CI/CD**: 无 GitHub Actions 配置 | 🟡 中 | 添加 `.github/workflows/` 配置自动化测试 |

---

## 二、核心架构审查

### 2.1 MHFSpectralConv 设计

```python
class MHFSpectralConv(SpectralConv):
    """
    继承 NeuralOperator 的 SpectralConv，添加多头机制
    """
```

**优点：**

1. **兼容性好** - 继承官方 `SpectralConv`，与 NeuralOperator 2.0 完全兼容
2. **优雅降级** - 当 `in_channels % n_heads != 0` 时自动回退到标准卷积
3. **性能优化** - 使用 `view` 替代 `reshape`，预分配输出张量，`einsum` 矩阵乘法

**问题：**

```python
# 问题1: 删除父类属性再重建，不够优雅
super().__init__(...)
del self.weight  # 删除父类权重
self.weight = nn.Parameter(...)  # 重新创建多头权重
```

**建议**：考虑组合模式替代继承：

```python
class MHFSpectralConv(nn.Module):
    def __init__(self, ...):
        super().__init__()
        self.spectral_conv = SpectralConv(...)  # 组合而非继承
        # 只创建多头特定的参数
```

### 2.2 与 NeuralOperator FNO 集成

**优点：**

```python
def create_hybrid_fno(..., mhf_layers=[0, 2]):
    model = FNO(...)  # 创建标准 FNO
    for layer_idx in mhf_layers:
        model.fno_blocks.convs[layer_idx] = MHFSpectralConv(...)  # 替换指定层
    return model
```

- 非侵入式设计，不修改 NeuralOperator 源码
- 支持灵活的层替换策略

**问题：**

- 直接访问 `model.fno_blocks.convs` 假设了 FNO 内部结构，如果 NeuralOperator 更新 API 可能会失败
- 建议：添加版本检查或 try-except 兼容处理

### 2.3 混合架构 (mhf_layers)

**当前实现：**

```python
mhf_layers=[0, 2]  # 首尾层使用 MHF，中间层保持标准卷积
```

**研究发现：**
- 这种配置在 Darcy Flow 上效果最佳
- 参数减少 30.6%，精度损失仅 1.4%

**建议：**
- 将这个最佳实践固化为默认配置
- 添加配置验证，确保 `mhf_layers` 索引有效

---

## 三、API 设计审查

### 3.1 工厂方法模式

**当前设计：**

```python
class MHFFNO:
    @staticmethod
    def best_config(...) -> FNO:
        return create_hybrid_fno(...)
    
    @staticmethod  
    def light_config(...) -> FNO:
        return create_hybrid_fno(...)
```

**问题：**

1. **反模式** - Python 惯例推荐使用模块级工厂函数，而非静态方法类
2. **混淆** - `MHFFNO` 类名暗示是模型类，但实际返回 `FNO` 实例

**建议**：

```python
# 方案1: 模块级函数（推荐）
def create_mhf_fno_best_config(...):
    ...

# 方案2: 真正的模型类
class MHFFNO(nn.Module):
    def __init__(self, ...):
        self.model = create_hybrid_fno(...)
    
    def forward(self, x):
        return self.model(x)
```

### 3.2 参数命名一致性

| 参数 | `create_hybrid_fno` | `MHFFNO.best_config` | 问题 |
|------|---------------------|----------------------|------|
| `n_modes` | `Tuple[int, ...]` | `Tuple[int, ...]` | ✅ 一致 |
| `n_heads` | 有 | 无 | ⚠️ `best_config` 硬编码为 4 |
| `mhf_layers` | 有 | 无 | ⚠️ `best_config` 硬编码为 [0, 2] |

**建议**：统一参数暴露，让用户可以在所有入口点自定义所有参数

### 3.3 默认值分散

```python
# mhf_fno.py
n_heads: int = 4  # 默认 4

# create_hybrid_fno
if mhf_layers is None:
    mhf_layers = [0, n_layers - 1]  # 默认首尾层

# README 推荐
n_heads=2  # 推荐 2！
```

**问题**：默认值与最佳实践不一致

**建议**：

```python
# 在模块级别定义默认配置
DEFAULT_N_HEADS = 2  # 基于优化测试的最佳值
DEFAULT_MHF_LAYERS = [0, 2]  # 首尾层策略
```

---

## 四、扩展性审查

### 4.1 添加新变体

**当前难度**：中等

需要修改的核心文件：
- `mhf_fno.py` - 添加新的卷积类
- `__init__.py` - 导出新类
- `create_hybrid_fno` - 添加新参数支持

**建议**：引入策略模式

```python
class SpectralConvStrategy(Enum):
    STANDARD = "standard"
    MHF = "mhf"
    TFNO = "tfno"  # 未来扩展

def create_fno(strategy: SpectralConvStrategy, ...):
    if strategy == SpectralConvStrategy.MHF:
        return create_hybrid_fno(...)
    # ...
```

### 4.2 支持不同 PDE 类型

**当前状态**：
- ✅ Darcy Flow 2D - 最佳效果
- ⚠️ Burgers 1D - 一般效果
- ❌ Navier-Stokes 2D - 不推荐
- ❌ 3D PDE - 不支持

**建议**：添加 PDE 类型适配器

```python
class PDEAdapter:
    """根据 PDE 类型自动选择最佳配置"""
    
    @staticmethod
    def get_recommended_config(pde_type: str) -> dict:
        configs = {
            "darcy_2d": {"n_heads": 2, "mhf_layers": [0, 2]},
            "burgers_1d": {"n_heads": 4, "mhf_layers": [0, 1, 2]},
            "navier_stokes": {"use_mhf": False},  # 不推荐
        }
        return configs.get(pde_type, {})
```

### 4.3 调试和测试支持

**缺失的功能：**

1. **可视化工具** - 无频谱权重可视化
2. **调试钩子** - 无中间层输出访问
3. **性能分析** - 无内置计时工具

**建议**：

```python
class MHFSpectralConv(SpectralConv):
    def forward(self, x, return_freq=False):
        # ...
        if return_freq:
            return x_out, {'freq_in': x_freq, 'freq_out': out_freq}
        return x_out
```

---

## 五、与业界最佳实践对比

### 5.1 PyTorch 风格

| 实践 | 当前状态 | 建议 |
|------|----------|------|
| `nn.Module` 基类 | ✅ 遵循 | - |
| `nn.Parameter` | ✅ 遵循 | - |
| `extra_repr()` | ✅ 实现 | - |
| 类型注解 | ⚠️ 部分 | 补充返回类型 |
| 文档字符串 | ⚠️ 部分 | 使用 Google/NumPy 风格 |
| 工厂方法 | ❌ 静态方法类 | 改用模块级函数 |

### 5.2 NeuralOperator 集成

| 实践 | 当前状态 | 建议 |
|------|----------|------|
| 继承 `SpectralConv` | ✅ | - |
| API 兼容 | ✅ | 添加版本检查 |
| 错误处理 | ⚠️ 只有警告 | 添加异常处理 |

### 5.3 代码质量

| 实践 | 当前状态 | 建议 |
|------|----------|------|
| 类型检查 (mypy) | ❌ 无配置 | 添加 `py.typed` 和 mypy 配置 |
| 代码格式化 | ❌ 无配置 | 添加 black/isort 配置 |
| 单元测试 | ❌ 无 | 添加 pytest 测试 |
| CI/CD | ❌ 无 | 添加 GitHub Actions |

---

## 六、改进建议优先级

### P0 - 必须修复

1. **移除重复代码** - 统一 `mhf_1d.py`/`mhf_2d.py` 到 `MHFSpectralConv`
2. **添加测试目录** - 创建 `tests/` 并添加单元测试
3. **修复默认值** - 将 `n_heads=4` 改为 `n_heads=2`（与最佳实践一致）

### P1 - 应该改进

1. **重构工厂模式** - 将 `MHFFNO` 改为模块级函数
2. **添加版本兼容** - 检查 NeuralOperator 版本
3. **添加类型标记** - 创建 `py.typed`

### P2 - 可以优化

1. **添加 CI/CD** - GitHub Actions 自动测试
2. **添加可视化** - 频谱权重可视化工具
3. **添加适配器** - PDE 类型自动配置

---

## 七、架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    MHF-FNO Architecture                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   MHFFNO     │    │create_hybrid │    │MHFSpectralConv│ │
│  │  (工厂类)    │───▶│    _fno()    │───▶│  (核心层)    │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│         │                   │                    │         │
│         │                   ▼                    │         │
│         │           ┌──────────────┐             │         │
│         │           │   FNO Model  │             │         │
│         │           │ (NeuralOperator)          │         │
│         │           └──────────────┘             │         │
│         │                   │                    │         │
│         │                   ▼                    │         │
│         │           ┌──────────────┐             │         │
│         └──────────▶│fno_blocks.convs│◀──────────┘         │
│                     └──────────────┘                       │
│                                                             │
│  层替换策略: mhf_layers=[0,2]                               │
│  ┌─────────────────────────────────────────────────┐       │
│  │ Layer 0: MHFSpectralConv (n_heads=2)            │       │
│  │ Layer 1: SpectralConv (标准)                    │       │
│  │ Layer 2: MHFSpectralConv (n_heads=2)            │       │
│  └─────────────────────────────────────────────────┘       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 八、总结

### 架构优点

1. **创新设计** - 多头频谱卷积有效减少参数量
2. **兼容性好** - 与 NeuralOperator 无缝集成
3. **实用性强** - 提供工厂方法和预设配置
4. **文档完善** - README 和研究笔记详尽

### 架构问题

1. **代码重复** - 独立 1D/2D 实现与主类重复
2. **测试缺失** - 无单元测试框架
3. **API 不一致** - 参数默认值与最佳实践不符
4. **扩展性不足** - 缺少抽象层和适配器设计

### 与业界差距

| 方面 | 差距 | 改进方向 |
|------|------|----------|
| 测试覆盖 | 落后 | 添加 pytest + coverage |
| CI/CD | 缺失 | 添加 GitHub Actions |
| 类型系统 | 基础 | 添加 py.typed + mypy |
| 代码风格 | 无规范 | 添加 black + isort |

---

*审查完成 - 天渊*