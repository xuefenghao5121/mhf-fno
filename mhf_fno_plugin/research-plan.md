# 天渊团队 - 研究计划

> **核心目标**: 优化传统Transformer结构，在纯CPU环境下使用FFT库实现高效AI推理

---

## 研究背景

### 核心需求 ⭐

**因果性（Causality）是大模型训练推理的必要条件！**

| 阶段 | 因果性要求 |
|------|-----------|
| **训练** | 必须保证，否则模型无法正确学习自回归 |
| **推理** | 必须保证，否则生成质量严重下降 |
| **大模型** | 绝对必要，影响scaling和性能 |

### 问题定义

| 现状 | 挑战 |
|------|------|
| **Self-Attention** | O(n²)复杂度，CPU上效率低 |
| **内存带宽** | Attention矩阵占用大量内存 |
| **GPU依赖** | 大多数优化依赖GPU并行 |
| **推理成本** | 长序列推理计算成本高 |
| **FFT天然全局** | 不满足因果性，需要特殊处理 |

### 解决方案方向

**使用FFT替代Self-Attention，并严格保证因果性**

| 优势 | 说明 |
|------|------|
| **复杂度** | O(n log n) vs O(n²) |
| **CPU友好** | FFT库在CPU上高度优化 |
| **内存效率** | 无需存储Attention矩阵 |
| **并行性** | FFT天然可并行 |
| **因果性保证** | 非对称padding + 截断 |

---

## 技术路线

### 阶段1: 理论验证 (W1-W2)

**目标**: 确认FFT替代Attention的可行性

| 任务 | 交付物 |
|------|--------|
| 研读TransFourier论文 | 核心技术分析报告 |
| 研读FNO论文 | 频域建模方法论 |
| 分析FFT库性能 | FFTW/Intel MKL基准测试 |
| 因果性实现方案 | 频域因果掩码设计 |

**关键技术问题**:
1. 如何在频域实现因果性？
2. 如何保持位置信息？
3. 如何处理变长序列？

### 阶段2: 原型实现 (W3-W4)

**目标**: 实现最小可行原型

| 任务 | 交付物 |
|------|--------|
| 实现MHF模块 | Python/PyTorch代码 |
| 实现因果掩码 | 频域因果机制 |
| 小规模验证 | WikiText-2测试 |
| 性能基准 | CPU vs GPU对比 |

**基准测试**:
```python
# 测试序列长度
lengths = [512, 1024, 2048, 4096, 8192]

# 对比项目
- 标准Transformer Self-Attention
- FFT-based Attention
- FNet (Google)

# 测试指标
- 推理延迟 (ms)
- 内存占用 (MB)
- 困惑度 (PPL)
```

### 阶段3: CPU优化 (W5-W6)

**目标**: 针对CPU进行深度优化

| 任务 | 交付物 |
|------|--------|
| FFT库选型 | 性能对比报告 |
| SIMD优化 | AVX2/AVX-512利用 |
| 多线程并行 | OpenMP/线程池 |
| 缓存优化 | 内存访问模式优化 |

**CPU优化策略**:

```
1. FFT库选择
   - FFTW: 通用高效
   - Intel MKL: Intel CPU优化
   - Apple vDSP: ARM优化
   - PocketFFT: 轻量级

2. 并行策略
   - 批量FFT并行
   - 多头并行
   - 序列维度并行

3. 内存优化
   - In-place FFT
   - 内存池
   - 预分配buffer
```

### 阶段4: 模型验证 (W7-W8)

**目标**: 在实际模型上验证效果

| 任务 | 交付物 |
|------|--------|
| 集成到LLM | 小型语言模型测试 |
| 长序列测试 | 8K+ token推理 |
| 精度评估 | 与原始模型对比 |
| 论文撰写 | 技术报告/论文 |

**测试模型**:
- GPT-2 Small (124M)
- 小型自定义模型
- 对比基线

---

## 关键技术细节

### 1. FFT替代Self-Attention

**标准Self-Attention**:
```python
def self_attention(Q, K, V):
    # Q, K, V: (batch, heads, seq_len, d_k)
    scores = Q @ K.transpose(-2, -1) / sqrt(d_k)  # O(n²)
    attn = softmax(scores)
    return attn @ V
```

**FFT-based Attention**:
```python
def fft_attention(Q, K, V):
    # 频域混合
    Q_freq = fft(Q, dim=-2)  # O(n log n)
    K_freq = fft(K, dim=-2)
    
    # 频域操作
    mixed = Q_freq * K_freq.conj()  # 逐元素乘法
    
    # 回到时域
    output = ifft(mixed, dim=-2)
    return output.real @ V
```

### 2. 因果性实现

**挑战**: FFT是全局操作，需要实现因果掩码

**TransFourier方案**:
```python
def causal_fft_attention(x):
    # 非对称padding
    x_padded = F.pad(x, (0, seq_len))  # 右侧补零
    
    # FFT
    x_freq = fft(x_padded)
    
    # 频域混合
    mixed = freq_mix(x_freq)
    
    # IFFT + 截断
    output = ifft(mixed)[:, :, :seq_len, :]
    
    return output
```

### 3. CPU优化

**FFTW使用**:
```c
#include <fftw3.h>

// 规划最优FFT
fftw_plan plan = fftw_plan_dft_1d(n, in, out, FFTW_FORWARD, FFTW_MEASURE);

// 执行FFT
fftw_execute(plan);

// 清理
fftw_destroy_plan(plan);
```

**Intel MKL**:
```c
#include <mkl.h>

// 使用MKL的FFT
DFTI_DESCRIPTOR_HANDLE descriptor;
DftiCreateDescriptor(&descriptor, DFTI_DOUBLE, DFTI_COMPLEX, 1, n);
DftiSetValue(descriptor, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
DftiCommitDescriptor(descriptor);
DftiComputeForward(descriptor, input, output);
```

---

## 预期成果

### 性能目标

| 序列长度 | 标准Attention | FFT-Attention | 加速比 |
|---------|--------------|---------------|--------|
| 512 | 1x | 0.8x | 0.8 |
| 1024 | 4x | 1.5x | 2.7x |
| 2048 | 16x | 3x | 5.3x |
| 4096 | 64x | 6x | 10.7x |
| 8192 | 256x | 12x | 21.3x |

### 精度目标

| 指标 | 目标 |
|------|------|
| 困惑度下降 | < 5% |
| 长序列保持率 | > 90% |
| 内存减少 | > 50% |

---

## 资源需求

### 计算资源

| 资源 | 用途 |
|------|------|
| **CPU服务器** | 基准测试、性能优化 |
| **开发机** | 原型开发、调试 |

### 软件依赖

```
# Python
torch >= 2.0
numpy
scipy  # FFT

# C/C++
fftw3
mkl  # 可选
openmp

# 框架
neuraloperator  # 参考实现
```

---

## 风险与缓解

| 风险 | 缓解措施 |
|------|----------|
| **精度下降** | 混合架构（FFT + 局部Attention） |
| **因果性问题** | 参考TransFourier的频域因果掩码 |
| **长序列位置编码** | 相对位置编码或RoPE适配 |
| **CPU性能不达预期** | 多级优化（SIMD + 多线程 + 缓存） |

---

## 时间线

```
W1-W2: 理论验证 ████████░░░░░░░░░░░░ 10%
W3-W4: 原型实现 ░░░░░░░░████████░░░░ 40%
W5-W6: CPU优化  ░░░░░░░░░░░░░░██████ 70%
W7-W8: 模型验证 ░░░░░░░░░░░░░░░░░░██ 100%
```

---

*制定日期: 2026-03-20*
*团队: 天渊团队*