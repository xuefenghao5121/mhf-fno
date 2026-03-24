# 天渊团队 (Tianyuan Team)

> 频域之渊，无穷探索

## 团队名称

**天渊团队 (Tianyuan Team)**

- **命名来源**：天渊是二十八宿之一，象征深邃与探索
- **研究方向**：FFT替代Transformer架构设计
- **核心理念**：以频域计算替代时域注意力，突破序列长度限制

## 研究论文

### 1. TransFourier: FFT Is All You Need

- **论文链接**: https://openreview.net/forum?id=TSHMAEItPc
- **关键词**: FFT, Attention-Free, Autoregressive
- **核心创新**:
  1. 完全用FFT替代masked self-attention
  2. O(L log L)复杂度（vs Transformer的O(L²)）
  3. 频域因果掩码实现自回归
  4. 无需自定义CUDA内核

### 2. Fourier Neural Operators Explained ⭐ 新增

- **论文链接**: https://arxiv.org/abs/2512.01421
- **作者**: Valentin Duruisseaux, Jean Kossaifi, Anima Anandkumar (NVIDIA/Caltech)
- **规模**: 96页, 27张图
- **核心价值**:
  - 全面而实用的FNO指南
  - 统一数学原理与实现策略
  - NeuralOperator 2.0.0库集成
  - 解决文献中的常见误解
- **学习重点**:
  - 算子理论基础
  - 频谱参数化
  - 离散化不变性
  - 实现细节

## 研究方向

### 1. 架构设计

| 方向 | 内容 |
|------|------|
| **FFT替代Self-Attention** | Multi-Head Fourier (MHF) 模块设计 |
| **频域因果掩码** | 自回归生成的关键技术 |
| **位置编码替代** | FFT天然解决位置外推问题 |
| **长序列建模** | 突破Transformer长度限制 |

### 2. 性能优化

| 方向 | 内容 |
|------|------|
| **计算效率** | O(L log L) vs O(L²) |
| **内存优化** | 无需存储attention matrix |
| **硬件友好** | 标准FFT算子，无需自定义CUDA |
| **并行化** | FFT天然并行 |

### 3. 应用场景

| 场景 | 潜力 |
|------|------|
| **长文本生成** | 百万token级别 |
| **时间序列预测** | 金融、气象、医疗 |
| **音频处理** | 语音识别、合成 |
| **DNA序列分析** | 基因组学应用 |

## 团队成员

### 天渊 ⭐ 首席架构师
- **职责**: 架构设计、核心算法
- **技能**: 深度学习、信号处理、FFT理论
- **模型**: qwen3-max + thinking

### 天渠 ⭐ 算法研究员
- **职责**: 算法实现、实验验证
- **技能**: PyTorch、CUDA、性能优化
- **模型**: qwen3-max + thinking

### 天池 ⭐ 理论分析师
- **职责**: 数学推导、理论证明
- **技能**: 傅里叶分析、泛函分析、信息论
- **模型**: qwen3-max + thinking

### 天井 ⭐ 应用工程师
- **职责**: 应用落地、系统集成
- **技能**: 工程实现、部署优化
- **模型**: qwen3.5-plus

## 当前任务

### Phase 1: 论文深度理解（W1-W2）

- [ ] 完整阅读TransFourier论文
- [ ] 理解MHF模块设计
- [ ] 分析频域因果掩码机制
- [ ] 对比实验结果

### Phase 2: 原型实现（W3-W4）

- [ ] 实现MHF模块
- [ ] 实现频域因果掩码
- [ ] 小规模验证实验
- [ ] 性能基准测试

### Phase 3: 优化改进（W5-W8）

- [ ] 探索改进方向
- [ ] 长序列扩展实验
- [ ] 多模态应用测试
- [ ] 论文撰写

## 相关工作

| 论文/模型 | 关系 |
|----------|------|
| FNet (2021) | 早期FFT用于Transformer的工作 |
| Mamba | SSM架构，同样关注长序列 |
| Hyena | 长卷积替代注意力 |
| RWKV | 线性注意力RNN |

## 文件结构

```
memory/projects/tianyuan-fft/
├── TEAM.md           # 团队说明
├── paper-analysis.md # 论文分析
├── experiments/      # 实验代码
├── notes/            # 研究笔记
└── deliverables/     # 交付成果
```

---

*创建日期: 2026-03-20*
*创建者: 鸡你太美*