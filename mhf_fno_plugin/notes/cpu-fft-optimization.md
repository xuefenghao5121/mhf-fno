# FFT CPU优化技术调研

> 目标: 在纯CPU环境下实现高效的FFT-based Transformer推理

---

## 1. FFT库选型

### 1.1 主流FFT库对比

| 库 | 语言 | 优势 | 劣势 | 许可证 |
|---|------|------|------|--------|
| **FFTW** | C | 通用高效，自适应优化 | 首次运行需要规划 | GPL |
| **Intel MKL** | C/Fortran | Intel CPU最优 | 仅限x86，收费 | 商业 |
| **Apple vDSP** | C | ARM/M系列最优 | 仅限Apple | BSD |
| **PocketFFT** | C++ | 轻量级，易集成 | 性能稍弱 | BSD-3 |
| **KissFFT** | C | 极简实现 | 性能一般 | BSD-3 |
| **pyFFTW** | Python | PyTorch/NumPy集成 | Python开销 | BSD |

### 1.2 性能基准

**测试环境**: 
- CPU: Intel Xeon / AMD EPYC / Apple M系列
- 序列长度: 1024, 4096, 16384
- 测量: 单次FFT时间 (μs)

**预期性能排名** (Intel CPU):
1. Intel MKL (最快)
2. FFTW (接近MKL)
3. PocketFFT
4. KissFFT

**预期性能排名** (ARM/Apple):
1. Apple vDSP
2. FFTW (ARM优化版)
3. PocketFFT

---

## 2. CPU优化策略

### 2.1 SIMD向量化

| 指令集 | 平台 | 宽度 | FFT加速 |
|--------|------|------|---------|
| **AVX-512** | Intel Xeon | 512-bit | 2-4x |
| **AVX2** | Intel/AMD | 256-bit | 1.5-2x |
| **NEON** | ARM | 128-bit | 1.5-2x |
| **SVE** | ARM (新) | 可变 | 2-4x |

**实现方式**:
```c
// FFTW自动使用SIMD
fftw_plan plan = fftw_plan_dft_1d(n, in, out, 
    FFTW_FORWARD, 
    FFTW_MEASURE | FFTW_UNALIGNED);  // 允许非对齐访问
```

### 2.2 多线程并行

**并行策略**:

```python
# 1. 批量并行（最有效）
# 多个序列同时FFT
batch_size = 32
for batch in range(0, total, batch_size):
    parallel_fft(sequences[batch:batch+batch_size])

# 2. 多头并行
# 不同head的FFT并行
with ThreadPoolExecutor(max_workers=num_heads) as executor:
    futures = [executor.submit(fft_head, h) for h in heads]

# 3. 序列分段并行
# 长序列分段处理
chunk_size = 1024
for i in range(0, seq_len, chunk_size):
    process_chunk(seq[i:i+chunk_size])
```

**FFTW多线程**:
```c
#include <fftw3.h>
#include <omp.h>

// 初始化多线程FFTW
fftw_init_threads();
fftw_plan_with_nthreads(omp_get_max_threads());

// 后续所有FFT自动并行
```

### 2.3 缓存优化

**内存访问模式**:

```python
# 差: 跨步访问
for i in range(seq_len):
    for j in range(d_model):
        x[i, j] = fft(x[i, j])

# 好: 连续访问
for j in range(d_model):
    x[:, j] = fft(x[:, j])

# 最好: 批量处理
x = fft(x, axis=0)  # 单次调用处理所有
```

**内存布局**:
```python
# 行优先 (PyTorch默认)
x = x.contiguous()  # 确保连续内存

# 预分配buffer避免重复分配
buffer = torch.empty(batch, heads, seq_len, d_model)
```

---

## 3. FFT在Transformer中的应用

### 3.1 标准Attention vs FFT Attention

**计算复杂度对比**:

| 操作 | 标准Attention | FFT Attention |
|------|--------------|---------------|
| Q·Kᵀ | O(n²d) | - |
| Softmax | O(n²) | - |
| Attn·V | O(n²d) | - |
| FFT | - | O(nd log n) |
| IFFT | - | O(nd log n) |
| **总计** | **O(n²d)** | **O(nd log n)** |

**内存占用对比**:

| 操作 | 标准Attention | FFT Attention |
|------|--------------|---------------|
| Attention矩阵 | O(n²) | 不需要 |
| 中间结果 | O(n²) | O(nd) |
| **总计** | **O(n²)** | **O(nd)** |

### 3.2 关键实现

**PyTorch实现**:
```python
import torch
import torch.fft

class FFTAttention(torch.nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        # 频域权重
        self.freq_weight = torch.nn.Parameter(
            torch.randn(n_heads, self.head_dim, self.head_dim) * 0.02
        )
    
    def forward(self, x, mask=None):
        B, L, D = x.shape
        
        # Reshape to heads
        x = x.view(B, L, self.n_heads, self.head_dim)
        x = x.permute(0, 2, 1, 3)  # (B, heads, L, d)
        
        # FFT along sequence dimension
        x_freq = torch.fft.rfft(x, dim=2)
        
        # Frequency mixing
        x_mixed = torch.einsum('bhld,hde->bhle', 
                               x_freq, 
                               self.freq_weight.to(x_freq.dtype))
        
        # IFFT
        x_out = torch.fft.irfft(x_mixed, n=L, dim=2)
        
        # Reshape back
        x_out = x_out.permute(0, 2, 1, 3).contiguous()
        x_out = x_out.view(B, L, D)
        
        return x_out
```

**C++实现 (FFTW)**:
```cpp
#include <fftw3.h>
#include <vector>

void fft_attention_forward(
    const float* input,
    float* output,
    int batch_size,
    int seq_len,
    int n_heads,
    int head_dim
) {
    int n = seq_len;
    int total = batch_size * n_heads * head_dim;
    
    // 分配FFTW内存
    fftwf_complex* in = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * n);
    fftwf_complex* out = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * n);
    
    // 规划FFT（重用计划）
    fftwf_plan plan_fwd = fftwf_plan_dft_1d(n, in, out, FFTW_FORWARD, FFTW_MEASURE);
    fftwf_plan plan_bwd = fftwf_plan_dft_1d(n, out, in, FFTW_BACKWARD, FFTW_MEASURE);
    
    // 对每个batch, head, dim执行FFT
    #pragma omp parallel for
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < n_heads; h++) {
            for (int d = 0; d < head_dim; d++) {
                // 复制输入
                for (int i = 0; i < n; i++) {
                    in[i][0] = input[((b * n_heads + h) * seq_len + i) * head_dim + d];
                    in[i][1] = 0.0f;
                }
                
                // 执行FFT
                fftwf_execute(plan_fwd);
                
                // TODO: 频域混合
                
                // 执行IFFT
                fftwf_execute(plan_bwd);
                
                // 复制输出
                for (int i = 0; i < n; i++) {
                    output[((b * n_heads + h) * seq_len + i) * head_dim + d] = in[i][0] / n;
                }
            }
        }
    }
    
    // 清理
    fftwf_destroy_plan(plan_fwd);
    fftwf_destroy_plan(plan_bwd);
    fftwf_free(in);
    fftwf_free(out);
}
```

---

## 4. CPU vs GPU性能对比

### 4.1 理论分析

| 因素 | GPU | CPU |
|------|-----|-----|
| **峰值算力** | 高 (Tensor Core) | 中 (AVX-512) |
| **内存带宽** | 高 (HBM) | 中 (DDR4/5) |
| **延迟** | 高 | 低 |
| **并行性** | 大规模并行 | 中等并行 |
| **FFT效率** | 中 (cuFFT) | 高 (FFTW/MKL) |

### 4.2 FFT的特殊性

**FFT在CPU上相对高效的原因**:

1. **内存访问模式**: FFT有规律的内存访问，适合CPU缓存
2. **算法复杂度**: O(n log n)比矩阵乘法O(n³)更友好
3. **库优化**: FFTW/MKL经过几十年优化
4. **无GPU开销**: 避免PCIe传输和内核启动

### 4.3 预期性能

**序列长度 8192, d_model 1024**:

| 平台 | 标准Attention | FFT Attention |
|------|--------------|---------------|
| **GPU (A100)** | 50ms | 15ms |
| **CPU (Xeon 32核)** | 2000ms | 100ms |
| **CPU加速比** | 1x (基准) | 20x |

---

## 5. 实验计划

### 5.1 基准测试

```bash
# FFT库性能测试
python benchmark_fft_libs.py --libs fftw,mkl,pocketfft --sizes 512,1024,2048,4096,8192

# Attention性能测试
python benchmark_attention.py --types standard,fft,fnet --seq-lens 512,1024,2048,4096
```

### 5.2 端到端测试

```python
# 模型配置
model_config = {
    'd_model': 512,
    'n_heads': 8,
    'n_layers': 6,
    'vocab_size': 50257,
}

# 测试配置
test_config = {
    'seq_lens': [512, 1024, 2048, 4096, 8192],
    'batch_sizes': [1, 2, 4, 8],
    'metrics': ['latency', 'memory', 'throughput'],
}
```

---

## 6. 参考文献

1. FFTW: http://www.fftw.org/
2. Intel MKL: https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html
3. TransFourier: https://openreview.net/forum?id=TSHMAEItPc
4. FNet: https://arxiv.org/abs/2105.03824

---

*调研日期: 2026-03-20*
*团队: 天渊团队*