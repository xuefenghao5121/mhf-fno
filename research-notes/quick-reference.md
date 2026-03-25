# FFT 神经算子研究快速参考

> 天池 (Tianchi) | 天渊团队 | 2026-03-25 (更新)

## 核心论文速查

| 论文 | arXiv | 核心贡献 | 适用场景 |
|------|-------|----------|----------|
| **FNO** | 2010.08895 | 频谱卷积算子 | 通用 PDE |
| **DeepONet** | 1910.03193 | Branch+Trunk 架构 | 算子学习 |
| **PINO** | 2111.03794 | 物理约束 | 无数据场景 |
| **MG-TFNO** | - | Tucker分解压缩 | 高分辨率 |
| **GINO** | - | 几何感知 | 3D复杂几何 |
| **CoDA-NO** | - | Codomain注意力 | 多物理场 |

## 关键研究发现 (2026-03-25)

### 频率耦合问题 ⚠️

**核心发现**: MHF 的多头独立处理破坏了 NS 方程的频率耦合机制。

| 配置 | vs FNO | 结论 |
|------|--------|------|
| MHF-[0,2] (首尾层) | -3.5% | ✅ 有效 |
| MHF-[0,1,2] (全层) | +5.6% | ❌ 失败 |
| CoDA-[0,2] | -4.4% | ✅ 最佳 |
| 数据增强 (1000样本) | -6.35% | ✅ 显著提升 |

**理论解释**: 
- NS 方程对流项产生非线性频率混合
- 中间标准卷积充当"频率耦合恢复层"
- 全层 MHF 无恢复层，耦合彻底破坏

### 最佳实践

```python
# NS 方程推荐配置
config = {
    'n_modes': (12, 12),
    'hidden_channels': 32,
    'n_heads': 4,
    'attention_type': 'coda',
    'mhf_layers': [0, 2],  # 关键: 首尾层！
    'data_size': 1000,
    'epochs': 300,
}
```

### 已排除的策略

| 策略 | 结果 | 原因 |
|------|------|------|
| ❌ 全层 MHF | +5.6% | 破坏频率耦合 |
| ❌ 全层 CoDA | +21.7% | 注意力过度叠加 |
| ❌ n_heads=2 | -1.52% | 表达力不足 |

## FNO 变体对比

| 方法 | 频率耦合机制 | 与 MHF+CoDA 关系 |
|------|-------------|-----------------|
| **TFNO** | Tucker 核心张量保持连接 | 启发：跨头连接 |
| **GINO** | 图+FNO 混合 | 启发：局部-全局分离 |
| **CoDA-NO** | Codomain 注意力 | 已集成 |

## 优化路径

```
当前: -6.35% (迭代 3)
       │
       ▼ 迭代 4 (测试中)
  -7.5% ~ -8.5%
       │
       ▼ + 物理约束/架构改进
  -10% ✅ 目标达成
```

## 研究笔记索引

| 文件 | 主题 |
|------|------|
| `frequency-coupling-analysis.md` | 频率耦合问题深度分析 |
| `fno-variants-deep-analysis.md` | TFNO/GINO/CoDA 深度分析 |
| `data-efficiency-analysis.md` | 数据效率与增强策略 |
| `research-summary.md` | 研究总结报告 |

## 项目代码

```python
# MHF-FNO 最佳配置
from mhf_fno import MHFFNO
model = MHFFNO.best_config(n_modes=(8, 8), hidden_channels=32)
```

详见: `/root/.openclaw/workspace/memory/projects/tianyuan-fft/`