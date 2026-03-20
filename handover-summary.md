# 工作交接总结

> 接收日期: 2026-03-20
> 来源: 鸡你太美

---

## 一、项目概述

### 核心目标

将 **Engram 确定性记忆架构** + **TransFourier 频域混合技术** 结合，优化 **NeuralOperator 2.0.0** 的 Darcy Flow 基准问题。

### 关键成果预期

| 指标 | 目标 |
|------|------|
| **精度** | L2 误差 ≤ 1%（与标准FNO相当）|
| **CPU吞吐量** | 提升 2-3 倍 |
| **训练时间** | 减少 20% 以上 |
| **超分辨率** | 支持训练低分辨率、推理高分辨率 |

---

## 二、技术背景

### 核心技术来源

| 技术 | 来源 | 核心贡献 |
|------|------|---------|
| **Engram** | DeepSeek论文 | 确定性哈希查表记忆，O(1)检索，可卸载到CPU内存 |
| **TransFourier** | 2025年论文 | MHF替代自注意力，O(L log L)复杂度，因果掩码支持 |
| **NeuralOperator 2.0.0** | PKU-CMEGroup | FNO/TFNO实现，内置benchmark，ONNX导出支持 |

### 本项目创新点

1. **双域Engram记忆**
   - 空间域Engram：记忆局部物理模式
   - 频域Engram：记忆重复频率成分

2. **MHF替换SpectralConv**
   - 用TransFourier的Multi-Head Fourier模块替代传统谱卷积

3. **频率域因果掩码**
   - 将TransFourier的因果技术引入时间序列PDE预测

4. **CPU推理优化**
   - Engram表卸载到CPU DRAM
   - ONNX Runtime / OpenVINO + 异步预取

---

## 三、总体架构

```
                    输入 (物理场)
                         │
   ┌─────────────────────┼─────────────────────┐
   ▼                     ▼                     ▼
┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│空间域Engram │   │ MHF模块     │   │频域Engram   │
│(局部模式)   │   │(TransFourier)│   │(频率模式)   │
└─────────────┘   └─────────────┘   └─────────────┘
   │                     │                     │
   └─────────────────────┼─────────────────────┘
                         ▼
               ┌─────────────────┐
               │   门控融合层     │
               │ (α, β, γ权重)   │
               └─────────────────┘
                         │
                         ▼
                    输出 (物理场)
```

---

## 四、实施阶段

### 阶段1：基线建立（2周）

- [ ] 安装NeuralOperator 2.0.0，跑通Darcy Flow训练
- [ ] 分析Darcy Flow数据集（局部模式分布统计）
- [ ] 编写数据加载器（支持多分辨率采样）

### 阶段2：Engram与MHF集成（3周）

- [ ] 实现空间域Engram模块
- [ ] 实现MHF模块（基于TransFourier）
- [ ] 实现频域Engram模块
- [ ] 实现门控融合层
- [ ] 在Darcy Flow上验证精度

### 阶段3：CPU推理优化（2周）

- [ ] 导出ONNX模型
- [ ] 将Engram表迁移到CPU内存
- [ ] 实现异步预取
- [ ] 使用ONNX Runtime/OpenVINO进行CPU推理
- [ ] 量化门控MLP

### 阶段4：实验验证与文档（2周）

- [ ] 多分辨率测试（64×64训练，128×128/256×256推理）
- [ ] 门控激活可视化
- [ ] 消融实验
- [ ] 撰写技术报告
- [ ] 代码整理（GitHub仓库）

---

## 五、评估指标

| 类别 | 指标 | 目标值 |
|------|------|--------|
| **精度** | 相对L2误差 | ≤ 基线FNO（允许+1%）|
| **训练** | 训练时间 | 减少 ≥ 20% |
| **训练** | 显存占用 | 减少 ≥ 50% |
| **推理** | CPU吞吐量 | 提升 2-3倍 |
| **泛化** | 超分辨率误差 | ≤ 基线的1.2倍 |

---

## 六、技术细节

### 空间域Engram设计

```python
# 局部窗口（如3×3）划分
# 使用确定性哈希（xxhash）映射到嵌入表索引
# 嵌入表放置在CPU内存，按需取用

class SpatialEngram:
    def __init__(self, window_size=3, num_patterns=10000):
        self.window_size = window_size
        self.embedding_table = nn.Embedding(num_patterns, embed_dim)
    
    def forward(self, x):
        # 1. 提取局部窗口
        windows = extract_windows(x, self.window_size)
        # 2. 哈希索引
        indices = xxhash(windows) % self.num_patterns
        # 3. 查表
        embeddings = self.embedding_table(indices)
        return embeddings
```

### MHF模块设计

```python
# 基于TransFourier的设计
# 多头FFT + 频域权重 + IFFT

class MHFModule(nn.Module):
    def __init__(self, d_model, n_heads, n_modes):
        self.freq_weights = nn.Parameter(
            torch.randn(n_heads, n_modes, d_model // n_heads, dtype=torch.cfloat)
        )
    
    def forward(self, x):
        # 1. 多头拆分
        # 2. FFT
        x_freq = torch.fft.rfft(x, dim=-2)
        # 3. 频域混合
        mixed = x_freq * self.freq_weights
        # 4. IFFT
        return torch.fft.irfft(mixed, dim=-2)
```

### CPU推理优化策略

| 组件 | 存储位置 | 优化手段 |
|------|----------|----------|
| Engram表 | CPU DRAM | 异步预取 + 多级缓存 |
| MHF权重 | CPU (ONNX) | FFTW/MKL加速 |
| 门控MLP | CPU | INT8量化 |

---

## 七、依赖环境

```
PyTorch >= 2.0
ONNX Runtime
OpenVINO (可选)
neuraloperator >= 2.0.0
```

---

## 八、交接注意事项

1. **数据路径**: 确保 `data/darcy_square` 目录存在
2. **哈希函数**: 使用确定性哈希（xxhash），注意跨平台一致性
3. **预取实现**: 使用 `CUDAStream` 或 `torch.cuda.Stream`
4. **门控可视化**: 训练过程中记录 α, β, γ 的均值

---

*接收人: 天渊团队*
*日期: 2026-03-20*