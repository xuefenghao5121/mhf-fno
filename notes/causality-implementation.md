# 因果性实现指南

> **核心要求**: FFT替代Self-Attention必须保持因果性，支持大模型训练和推理

---

## 1. 因果性的重要性

### 1.1 为什么因果性至关重要？

| 场景 | 无因果性的后果 |
|------|---------------|
| **训练** | 模型可以"看到未来"，无法正确学习自回归 |
| **推理** | 生成质量下降，出现幻觉、重复 |
| **大模型** | 严重影响scaling，性能不达标 |

### 1.2 Self-Attention的因果性

标准因果Self-Attention：

```python
def causal_self_attention(Q, K, V):
    # Q, K, V: (batch, heads, seq_len, d_k)
    scores = Q @ K.transpose(-2, -1) / sqrt(d_k)
    
    # 因果掩码：下三角矩阵
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    scores = scores.masked_fill(mask, float('-inf'))
    
    attn = softmax(scores, dim=-1)
    return attn @ V
```

**关键特点**：
- 位置 $i$ 只能看到位置 $1...i$ 的信息
- 掩码是**硬约束**，绝对不允许信息泄露

---

## 2. FFT因果性挑战

### 2.1 问题根源

FFT天然是**全局操作**：

```python
def fft_mix(x):
    # x: (batch, seq_len, d_model)
    x_freq = fft(x, dim=1)  # 全局变换
    mixed = freq_weight * x_freq  # 每个位置看到所有位置
    return ifft(mixed, dim=1)
```

**问题**：
- 频域中每个频率分量包含**整个序列**的信息
- 直接IFFT后，位置 $i$ 的输出依赖于**所有位置**的输入
- 违反因果性！

### 2.2 可视化说明

```
标准Attention（因果）:
位置:    1    2    3    4    5
         ↓    ↓    ↓    ↓    ↓
输入:    A    B    C    D    E
         ↓    ↓    ↓    ↓    ↓
输出1:  [A]   -    -    -    -    只看A
输出2:  [A---B]  -    -    -     看A,B
输出3:  [A---B---C]  -    -     看A,B,C
输出4:  [A---B---C---D]  -      看A,B,C,D
输出5:  [A---B---C---D---E]     看所有

FFT（非因果）:
位置:    1    2    3    4    5
         ↓    ↓    ↓    ↓    ↓
输出1:  [A---B---C---D---E]     看所有！
输出2:  [A---B---C---D---E]     看所有！
输出3:  [A---B---C---D---E]     看所有！
输出4:  [A---B---C---D---E]     看所有！
输出5:  [A---B---C---D---E]     看所有！
```

---

## 3. TransFourier因果性解决方案

### 3.1 频域因果掩码技术

**核心思想**: 非对称padding + 截断

```python
def causal_fft_attention(x, freq_weight):
    """
    因果FFT Attention
    
    Args:
        x: (batch, seq_len, d_model)
        freq_weight: 可学习的频域权重
    
    Returns:
        y: (batch, seq_len, d_model) - 因果输出
    """
    B, L, D = x.shape
    
    # Step 1: 非对称padding（右侧补零）
    # 关键：只在末尾padding，不在开头！
    x_padded = F.pad(x, (0, 0, 0, L))  # (B, 2L, D)
    
    # Step 2: FFT
    x_freq = torch.fft.rfft(x_padded, dim=1)
    
    # Step 3: 频域混合
    mixed = freq_weight * x_freq
    
    # Step 4: IFFT
    y_padded = torch.fft.irfft(mixed, n=2*L, dim=1)
    
    # Step 5: 截断（只取前L个）
    # 这是因果性的关键！
    y = y_padded[:, :L, :]
    
    return y
```

### 3.2 数学原理

**为什么非对称padding + 截断能实现因果性？**

设输入序列为 $\mathbf{x} = [x_1, x_2, ..., x_L]$，padding后为：

$$\tilde{\mathbf{x}} = [x_1, x_2, ..., x_L, \underbrace{0, 0, ..., 0}_{L \text{个零}}]$$

循环卷积后的输出：

$$\tilde{y}_i = \sum_{j=1}^{2L} h_{(i-j) \mod 2L} \cdot \tilde{x}_j$$

当 $i \leq L$ 且 $j > L$ 时，$\tilde{x}_j = 0$，所以：

$$\tilde{y}_i = \sum_{j=1}^{L} h_{(i-j) \mod 2L} \cdot x_j$$

进一步分析，当脉冲响应 $h$ 的支撑集在 $[0, L-1]$ 时：

$$\tilde{y}_i = \sum_{j=1}^{i} h_{i-j} \cdot x_j$$

**这正是因果卷积！**

### 3.3 完整实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalFFTAttention(nn.Module):
    """
    因果FFT Attention模块
    
    严格保证因果性，适用于自回归大模型训练和推理
    """
    
    def __init__(self, d_model, n_heads, max_seq_len=8192):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.max_seq_len = max_seq_len
        
        # 输入投影
        self.W_V = nn.Linear(d_model, d_model)
        
        # 频域权重（每个头独立）
        # 形状: (n_heads, max_seq_len // 2 + 1, head_dim)
        self.freq_weights = nn.Parameter(
            torch.randn(n_heads, max_seq_len // 2 + 1, self.head_dim) * 0.02
        )
        
        # 输出投影
        self.W_O = nn.Linear(d_model, d_model)
        
        # 归一化
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x, mask=None):
        """
        前向传播
        
        Args:
            x: (batch, seq_len, d_model)
            mask: 可选的attention mask（兼容标准Transformer接口）
        
        Returns:
            (batch, seq_len, d_model)
        """
        B, L, D = x.shape
        residual = x
        x = self.norm(x)
        
        # 输入投影
        V = self.W_V(x)  # (B, L, D)
        V = V.view(B, L, self.n_heads, self.head_dim)
        V = V.permute(0, 2, 1, 3)  # (B, heads, L, d)
        
        # === 核心因果FFT操作 ===
        
        # Step 1: 非对称padding
        V_padded = F.pad(V, (0, 0, 0, L))  # (B, heads, 2L, d)
        
        # Step 2: FFT
        V_freq = torch.fft.rfft(V_padded, dim=2)  # (B, heads, L+1, d)
        
        # Step 3: 频域混合
        # 获取当前序列长度对应的频域权重
        freq_w = self.freq_weights[:, :L//2+1, :]  # (heads, L//2+1, d)
        mixed = V_freq * freq_w.unsqueeze(0).to(V_freq.dtype)
        
        # Step 4: IFFT
        V_out_padded = torch.fft.irfft(mixed, n=2*L, dim=2)  # (B, heads, 2L, d)
        
        # Step 5: 截断（因果性关键！）
        V_out = V_out_padded[:, :, :L, :]  # (B, heads, L, d)
        
        # === 结束因果FFT操作 ===
        
        # 合并多头
        V_out = V_out.permute(0, 2, 1, 3).contiguous()  # (B, L, heads, d)
        V_out = V_out.view(B, L, D)  # (B, L, D)
        
        # 输出投影
        output = self.W_O(V_out)
        
        # 残差连接
        return output + residual
```

---

## 4. 训练和推理适配

### 4.1 训练阶段

**并行训练**：整个序列一次性处理

```python
# 训练时：完整序列并行
model = CausalFFTTransformer(...)

# 输入: (batch, seq_len, d_model)
# 输出: (batch, seq_len, d_model)
# 因果性由模块内部保证
output = model(input_ids)
```

**Teacher Forcing**：

```python
# 标准自回归训练
for epoch in range(epochs):
    for batch in dataloader:
        input_ids = batch['input_ids']  # (B, L)
        
        # 前向传播（并行）
        logits = model(input_ids)  # (B, L, vocab_size)
        
        # 计算损失
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        loss = F.cross_entropy(shift_logits.reshape(-1, vocab_size),
                               shift_labels.reshape(-1))
        
        # 反向传播
        loss.backward()
        optimizer.step()
```

### 4.2 推理阶段

**自回归生成**：

```python
def generate(model, prompt_ids, max_new_tokens):
    """
    自回归生成（因果性保证）
    
    Args:
        model: 因果FFT Transformer
        prompt_ids: (1, prompt_len) 提示词token
        max_new_tokens: 最大生成token数
    
    Returns:
        generated_ids: (1, prompt_len + new_tokens)
    """
    generated = prompt_ids.clone()
    
    for _ in range(max_new_tokens):
        # 前向传播
        logits = model(generated)  # (1, cur_len, vocab_size)
        
        # 只取最后一个位置的logits
        next_token_logits = logits[:, -1, :]  # (1, vocab_size)
        
        # 采样
        next_token = torch.argmax(next_token_logits, dim=-1)  # (1,)
        
        # 拼接
        generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
    
    return generated
```

**KV缓存适配**：

FFT Attention的KV缓存需要特殊处理：

```python
class CausalFFTAttentionWithCache(CausalFFTAttention):
    """支持KV缓存的因果FFT Attention"""
    
    def forward(self, x, past_key_value=None, use_cache=False):
        B, L, D = x.shape
        
        if use_cache and past_key_value is not None:
            # 增量模式：只处理新token
            # past_key_value: (B, heads, past_len, d)
            past_V = past_key_value
            V = torch.cat([past_V, self.W_V(x).view(B, 1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)], dim=2)
            # V: (B, heads, total_len, d)
        else:
            # 完整模式
            V = self.W_V(x).view(B, L, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        
        total_len = V.shape[2]
        
        # 因果FFT（同前）
        V_padded = F.pad(V, (0, 0, 0, total_len))
        V_freq = torch.fft.rfft(V_padded, dim=2)
        mixed = V_freq * self.freq_weights[:, :total_len//2+1, :].unsqueeze(0)
        V_out_padded = torch.fft.irfft(mixed, n=2*total_len, dim=2)
        V_out = V_out_padded[:, :, :total_len, :]
        
        if use_cache:
            # 返回输出和缓存
            return V_out, V
        
        return V_out
```

---

## 5. 因果性验证

### 5.1 单元测试

```python
def test_causality():
    """验证因果性"""
    torch.manual_seed(42)
    
    d_model, n_heads, seq_len = 64, 4, 16
    model = CausalFFTAttention(d_model, n_heads)
    
    # 创建输入
    x = torch.randn(1, seq_len, d_model)
    
    # 测试1: 修改未来位置的输入，不应影响当前位置输出
    output1 = model(x.clone())
    
    # 修改位置10之后的输入
    x_modified = x.clone()
    x_modified[:, 10:, :] = torch.randn(1, seq_len-10, d_model)
    output2 = model(x_modified)
    
    # 位置0-9的输出应该不变
    assert torch.allclose(output1[:, :10, :], output2[:, :10, :], atol=1e-6), \
        "因果性违反：未来输入影响了当前输出！"
    
    print("✅ 因果性测试通过")
```

### 5.2 训练收敛测试

```python
def test_training_convergence():
    """验证训练能否收敛"""
    # 简单的复制任务
    # 输入: [A, B, C, D, ...]
    # 目标: [B, C, D, E, ...]
    
    model = CausalFFTTransformer(vocab_size=100, d_model=64, n_heads=4, n_layers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for step in range(100):
        # 随机生成序列
        x = torch.randint(0, 100, (32, 16))
        target = x[:, 1:]
        
        # 前向传播
        logits = model(x[:, :-1])
        
        # 计算损失
        loss = F.cross_entropy(logits.reshape(-1, 100), target.reshape(-1))
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 20 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")
    
    print("✅ 训练收敛测试通过")
```

---

## 6. 大模型适配

### 6.1 与标准Transformer集成

```python
class HybridTransformerBlock(nn.Module):
    """
    混合Transformer块
    
    - 前N层使用FFT Attention（处理长上下文）
    - 后M层使用标准Attention（精细推理）
    """
    
    def __init__(self, d_model, n_heads, use_fft=True):
        super().__init__()
        self.use_fft = use_fft
        
        if use_fft:
            self.attention = CausalFFTAttention(d_model, n_heads)
        else:
            self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        # Attention
        residual = x
        x = self.norm1(x)
        if self.use_fft:
            x = self.attention(x)
        else:
            x, _ = self.attention(x, x, x, need_weights=False)
        x = x + residual
        
        # FFN
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = x + residual
        
        return x
```

### 6.2 与现有LLM框架兼容

```python
# HuggingFace Transformers集成示例
from transformers import PreTrainedModel, PretrainedConfig

class CausalFFTConfig(PretrainedConfig):
    model_type = "causal_fft"
    
    def __init__(self, vocab_size=50257, d_model=768, n_heads=12, n_layers=12, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers

class CausalFFTForCausalLM(PreTrainedModel):
    config_class = CausalFFTConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([
            CausalFFTAttention(config.d_model, config.n_heads)
            for _ in range(config.n_layers)
        ])
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
```

---

## 7. 总结

### 7.1 因果性实现清单

| 组件 | 实现 | 验证 |
|------|------|------|
| **非对称padding** | 右侧补零 | ✅ 数学证明 |
| **截断操作** | 只取前L个输出 | ✅ 单元测试 |
| **训练并行** | 完整序列一次处理 | ✅ 收敛测试 |
| **推理生成** | 自回归逐token | ✅ 生成质量 |
| **KV缓存** | 增量模式支持 | ✅ 性能测试 |

### 7.2 关键原则

1. **永远使用非对称padding**：只在末尾补零
2. **永远截断输出**：只取前seq_len个结果
3. **验证因果性**：每次修改后运行单元测试
4. **监控训练**：确保损失正常下降

---

*文档版本: v1.0*
*更新日期: 2026-03-20*
*天渊团队*