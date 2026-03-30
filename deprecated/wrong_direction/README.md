# 错误方向说明

## ❌ 已废弃的代码

这些代码基于对 CODA 的错误理解，已被废弃。

### 错误理解

**错误的 CODA 定义**：
- ❌ CODA = Curriculum and Data Augmentation（课程学习与数据增强）

**正确的 CODA 定义**：
- ✅ CODA = Cross-Head Attention（跨头注意力机制）

### 废弃的文件

| 文件 | 原因 | 行数 |
|------|------|------|
| `afno.py` | 基于错误理解创建 | 400 |
| `afno_curriculum.py` | 基于错误理解创建 | 500 |

### 废弃的文档

| 文件 | 原因 |
|------|------|
| `docs/research/coda_experiment_design.md` | 需要大幅修改 CODA 定义 |
| `docs/research/CODA_RESEARCH_COMPLETE.md` | 需要重新编写 |
| `docs/research/tfno_afno_analysis.md` | 需要删除错误 CODA 章节 |
| `docs/research/afno_fs_design.md` | 基于 AFNO-FS，部分内容可复用 |
| `docs/research/afno_fs_experiment_report.md` | 基于 AFNO-FS 实验报告 |

### 正确的实现

**MHF + CODA 的正确实现在**：
- `mhf_fno/mhf_attention.py` - Cross-Head Attention（CoDA）完整实现
- `mhf_fno/mhf_fno.py` - MHF-FNO 核心实现
- `mhf_fno/pino_physics.py` - PINO 物理约束

### 正确的 CODA 理解

**CODA = Cross-Head Attention（跨头注意力机制）**

- **实现位置**：`mhf_fno/mhf_attention.py`
- **核心类**：
  - `CrossHeadAttention` - 跨头注意力模块
  - `MHFSpectralConvWithAttention` - 带注意力的 MHF 频谱卷积
  - `MHFFNOWithAttention` - 带注意力的 MHF-FNO 模型

- **设计目的**：
  - 解决 MHF 的头之间完全独立、无法跨频率交互的问题
  - 实现**跨频率交互**（不同频率头之间的信息交换）
  - 保持参数高效（参数量 < MHF 卷积的 1%）

---

*废弃日期: 2026-03-30*
*原因: CODA 理解错误*
