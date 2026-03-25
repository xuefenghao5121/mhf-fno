# 架构优化详细建议

> **分析人**: 天渊（架构师）
> **日期**: 2026-03-25

---

## 一、优化优先级矩阵

| 优化方向 | 预期收益 | 实现难度 | 风险 | 优先级 |
|----------|----------|----------|------|--------|
| 数据扩展 (2000 样本) | 1-2% | 低 | 低 | P0 |
| 超参数优化 (bottleneck, gate) | 0.5-1% | 中 | 低 | P0 |
| 训练策略优化 | 0.5-1% | 低 | 低 | P1 |
| 频域注意力 | 1-3% | 中 | 中 | P1 |
| 自适应头数 | 1-2% | 高 | 中 | P2 |
| PINO 风格物理约束 | 2-4% | 高 | 中 | P2 |

---

## 二、P0 优化详细建议

### 2.1 数据扩展

#### 当前状态
- 训练样本: 1000
- 测试样本: 200
- 参数/样本比: 140

#### 扩展方案

```python
# 方案 A: 渐进扩展
train_samples = [1000, 2000, 5000]
for n_samples in train_samples:
    results = run_experiment(n_samples=n_samples)
    if results.vs_fno <= -10:
        break

# 方案 B: 激进扩展
train_samples = 5000  # 直接跳到足够大的数据集
```

#### 预期效果

| 样本数 | 预期 vs FNO | 参数/样本 |
|--------|-------------|-----------|
| 1000 | -6.35% | 140 |
| 2000 | -7.5% ~ -8.5% | 70 |
| 5000 | -8.5% ~ -10% | 28 |

### 2.2 超参数优化

#### 当前配置

```python
MHFFNO(
    n_modes=(12, 12),
    hidden_channels=32,
    n_layers=3,
    n_heads=4,
    mhf_layers=[0, 2],
    attention=CoDAStyleAttention(
        bottleneck=4,        # 当前值
        gate_init=0.1        # 当前值
    )
)
```

#### 优化空间

**Bottleneck 大小**

| 值 | 压缩率 | 注意力容量 | 预期效果 |
|----|--------|-----------|----------|
| 2 | 75% | 低 | 可能信息损失 |
| 4 | 50% | 中 | 当前最佳 |
| 6 | 25% | 高 | 可能过拟合 |
| 8 | 0% | 最高 | 等于无压缩 |

**Gate 初始值**

| 值 | 初始增强强度 | 预期效果 |
|----|-------------|----------|
| 0.05 | 弱 | 更稳定，但可能收敛慢 |
| 0.1 | 中 | 当前值 |
| 0.2 | 强 | 可能更快的改进 |
| 0.3 | 很强 | 可能不稳定 |

#### 搜索策略

```python
# 网格搜索
bottleneck_options = [2, 4, 6]
gate_init_options = [0.05, 0.1, 0.2]

results = {}
for bn in bottleneck_options:
    for gi in gate_init_options:
        model = create_model(bottleneck=bn, gate_init=gi)
        results[(bn, gi)] = train_and_evaluate(model)

# 找最佳组合
best_config = min(results, key=lambda x: results[x]['loss'])
```

---

## 三、P1 优化详细建议

### 3.1 训练策略优化

#### 学习率调度

```python
# 当前: CosineAnnealing
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

# 建议: Warmup + Cosine
def get_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs  # 线性预热
        else:
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

scheduler = get_warmup_cosine_scheduler(optimizer, warmup_epochs=5, total_epochs=200)
```

#### Epochs 调整

| Epochs | 当前效果 | 建议 |
|--------|----------|------|
| 50 | -3.73% (200样本) | 数据量少时足够 |
| 100 | 待测试 | 建议测试 |
| 200 | 待测试 | 数据量大时推荐 |
| 500 | 可能过拟合 | 不推荐 |

### 3.2 频域注意力

#### 设计思路

当前 CoDA 在空间域（IFFT 后）做注意力。改为在频域直接做注意力可能更有效：

```python
class FrequencyDomainCoDA(nn.Module):
    """在频域直接做跨头注意力"""
    
    def __init__(self, n_heads, head_dim, bottleneck=4):
        super().__init__()
        
        # 复数处理: 分离实部和虚部
        self.real_compress = nn.Linear(head_dim, bottleneck)
        self.imag_compress = nn.Linear(head_dim, bottleneck)
        
        # 跨头注意力
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=bottleneck,
            num_heads=2,
            batch_first=True
        )
        
        # 重建
        self.real_expand = nn.Linear(bottleneck, head_dim)
        self.imag_expand = nn.Linear(bottleneck, head_dim)
        
        self.gate = nn.Parameter(torch.ones(1) * 0.1)
    
    def forward(self, x_freq):
        """
        Args:
            x_freq: [B, n_heads, head_dim, H, W] 复数张量
        """
        # 分离实部虚部
        x_real = x_freq.real
        x_imag = x_freq.imag
        
        # 空间池化
        x_real_pool = x_real.mean(dim=(-2, -1))  # [B, n_heads, head_dim]
        x_imag_pool = x_imag.mean(dim=(-2, -1))
        
        # 分别压缩
        real_comp = self.real_compress(x_real_pool)
        imag_comp = self.imag_compress(x_imag_pool)
        
        # 跨头注意力
        real_attn, _ = self.cross_attn(real_comp, real_comp, real_comp)
        imag_attn, _ = self.cross_attn(imag_comp, imag_comp, imag_comp)
        
        # 重建
        real_out = self.real_expand(real_attn)
        imag_out = self.imag_expand(imag_attn)
        
        # 计算增量
        delta_real = real_out - x_real_pool
        delta_imag = imag_out - x_imag_pool
        
        # 广播并应用
        delta_real = delta_real.unsqueeze(-1).unsqueeze(-1)
        delta_imag = delta_imag.unsqueeze(-1).unsqueeze(-1)
        
        out_real = x_real + self.gate * delta_real
        out_imag = x_imag + self.gate * delta_imag
        
        return torch.complex(out_real, out_imag)
```

#### 预期优势

1. **信息保留**: 避免 FFT → IFFT → FFT 的信息损失
2. **直接耦合**: 直接在频率空间恢复耦合
3. **计算效率**: 复数运算可能更高效

---

## 四、P2 优化详细建议

### 4.1 自适应头数

#### 设计思路

不同频率范围可能需要不同的头数：

```python
class AdaptiveMHFSpectralConv(nn.Module):
    """根据频率自适应调整头数"""
    
    def __init__(self, in_channels, out_channels, n_modes, n_heads=4):
        super().__init__()
        
        self.n_modes = n_modes
        
        # 低频用少头，高频用多头
        self.low_freq_heads = max(1, n_heads // 2)   # 低频: 2 头
        self.high_freq_heads = n_heads               # 高频: 4 头
        
        # 分界点
        self.freq_threshold = n_modes[0] // 2
        
        # 低频权重 (少头，更大容量)
        self.low_weight = nn.Parameter(
            torch.randn(self.low_freq_heads, 
                       in_channels // self.low_freq_heads,
                       out_channels // self.low_freq_heads,
                       self.freq_threshold)
        )
        
        # 高频权重 (多头，细粒度)
        self.high_weight = nn.Parameter(
            torch.randn(self.high_freq_heads,
                       in_channels // self.high_freq_heads,
                       out_channels // self.high_freq_heads,
                       n_modes[0] - self.freq_threshold)
        )
    
    def forward(self, x):
        # FFT
        x_freq = torch.fft.rfft2(x, dim=(-2, -1))
        
        # 分离低频和高频
        x_low = x_freq[:, :, :self.freq_threshold, :]
        x_high = x_freq[:, :, self.freq_threshold:, :]
        
        # 分别处理
        out_low = self._process_low_freq(x_low)
        out_high = self._process_high_freq(x_high)
        
        # 合并
        out_freq = torch.cat([out_low, out_high], dim=2)
        
        # IFFT
        return torch.fft.irfft2(out_freq, s=x.shape[-2:])
```

#### 预期优势

1. **低频**: 用少头，保持大尺度涡的耦合
2. **高频**: 用多头，细化处理小尺度涡

### 4.2 PINO 风格物理约束

#### 设计思路

将 NS 方程的物理约束加入损失函数：

```python
class PhysicsInformedLoss(nn.Module):
    """物理约束损失函数"""
    
    def __init__(self, viscosity=1e-3, weight=0.1):
        super().__init__()
        self.viscosity = viscosity
        self.weight = weight
    
    def compute_ns_residual(self, u_pred):
        """计算 NS 方程残差"""
        # u_pred: [B, T, 2, H, W] (速度场)
        
        # 时间导数
        u_t = torch.autograd.grad(
            u_pred.sum(), 
            torch.arange(u_pred.shape[1], device=u_pred.device),
            create_graph=True
        )[0]
        
        # 空间导数
        u_x = torch.autograd.grad(u_pred.sum(), ..., create_graph=True)[0]
        u_y = torch.autograd.grad(u_pred.sum(), ..., create_graph=True)[0]
        
        # 对流项
        convection = u_pred[..., 0] * u_x + u_pred[..., 1] * u_y
        
        # 扩散项
        u_xx = torch.autograd.grad(u_x.sum(), ..., create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y.sum(), ..., create_graph=True)[0]
        diffusion = self.viscosity * (u_xx + u_yy)
        
        # 残差
        residual = u_t + convection - diffusion
        
        return (residual ** 2).mean()
    
    def forward(self, u_pred, u_true):
        # 数据损失
        data_loss = F.mse_loss(u_pred, u_true)
        
        # 物理损失
        physics_loss = self.compute_ns_residual(u_pred)
        
        # 组合
        return data_loss + self.weight * physics_loss
```

#### 预期优势

1. **物理一致性**: 确保输出满足 NS 方程
2. **正则化效果**: 物理约束作为强正则化
3. **数据效率**: 可能减少对数据量的需求

---

## 五、实验设计建议

### 5.1 A/B 测试框架

```python
def run_ablation_study():
    """A/B 消融实验"""
    
    configs = {
        'baseline': {
            'data_size': 1000,
            'bottleneck': 4,
            'gate_init': 0.1,
            'epochs': 50
        },
        'data_2k': {
            'data_size': 2000,
            'bottleneck': 4,
            'gate_init': 0.1,
            'epochs': 50
        },
        'hyp_opt': {
            'data_size': 1000,
            'bottleneck': 6,
            'gate_init': 0.2,
            'epochs': 50
        },
        'combined': {
            'data_size': 2000,
            'bottleneck': 6,
            'gate_init': 0.2,
            'epochs': 100
        }
    }
    
    results = {}
    for name, config in configs.items():
        results[name] = train_and_evaluate(config)
    
    return results
```

### 5.2 收敛分析

```python
def analyze_convergence(history):
    """分析收敛特性"""
    
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Loss 曲线
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['test_loss'], label='Test')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    
    # 最佳 Epoch
    best_epoch = np.argmin(history['test_loss'])
    axes[0].axvline(best_epoch, color='r', linestyle='--', label=f'Best: {best_epoch}')
    
    # 学习率
    axes[1].plot(history['lr'])
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Learning Rate')
    
    # 门控值变化
    if 'gate_value' in history:
        axes[2].plot(history['gate_value'])
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Gate Value')
    
    plt.tight_layout()
    plt.savefig('convergence_analysis.png')
```

---

## 六、风险评估

### 6.1 高风险优化

| 优化 | 风险 | 缓解措施 |
|------|------|----------|
| PINO 物理约束 | 实现复杂，可能引入 Bug | 先在小数据集验证 |
| 自适应头数 | 可能破坏现有架构 | 保持回退选项 |
| 频域注意力 | 复数运算可能有数值问题 | 使用稳定实现 |

### 6.2 建议顺序

1. **先执行低风险优化**: 数据扩展 + 超参数
2. **验证后再尝试中风险**: 训练策略 + 频域注意力
3. **最后才考虑高风险**: 物理约束 + 自适应头数

---

*建议完成 - 天渊*
*2026-03-25 16:57*