# MHF 驱动的新算子设计

## 🎯 设计理念

**核心问题**: MHF 不应该只是"替换"现有 SpectralConv，而应该作为设计新算子的核心组件。

**MHF 的特征**:
1. 通道分离 → 多样化频域特征
2. 参数减少 → 隐式正则化
3. 独立权重 → 可学习频率模式

---

## 算子 1: MHF-FNO → MHF-ResFNO

### 问题分析
标准 FNO 使用 SpectralConv，MHF 替换后边缘层效果好，但中间层不稳定。

### 新设计: MHF-ResFNO

```python
class MHFResBlock(nn.Module):
    """MHF 残差块"""
    def __init__(self, channels, n_modes, n_heads=4):
        super().__init__()
        # MHF 处理低频
        self.mhf = MHFSpectralConv(channels, channels, n_modes, n_heads)
        # 标准 Conv 处理高频残差
        self.high_freq = nn.Conv2d(channels, channels, 1)
        # 门控融合
        self.gate = nn.Parameter(torch.ones(1) * 0.5)
    
    def forward(self, x):
        low_freq = self.mhf(x)
        high_freq = self.high_freq(x)
        return self.gate * low_freq + (1 - self.gate) * high_freq + x
```

**创新点**:
- MHF 专注低频特征（全局趋势）
- 标准 Conv 处理高频残差（局部细节）
- 门控融合 + 残差连接

---

## 算子 2: TFNO + MHF → MHF-TensorFNO

### 问题分析
TFNO 已使用张量化压缩，直接替换 MHF 会增加参数。

### 新设计: MHF-TensorFNO

**核心思想**: 不替换，而是**在张量化之前应用 MHF**

```python
class MHFTensorSpectralConv(nn.Module):
    """MHF + 张量化 SpectralConv"""
    def __init__(self, in_ch, out_ch, n_modes, n_heads=4, rank=0.5):
        super().__init__()
        # 先用 MHF 分离通道
        self.n_heads = n_heads
        head_ch = in_ch // n_heads
        
        # 每个头独立的低秩分解
        self.head_weights = nn.ModuleList([
            TensorizedWeight(head_ch, head_ch, n_modes, rank)
            for _ in range(n_heads)
        ])
    
    def forward(self, x):
        B, C, H, W = x.shape
        # 分头
        x_heads = x.view(B, self.n_heads, C // self.n_heads, H, W)
        
        # 各头独立处理
        out_heads = []
        for i, weight in enumerate(self.head_weights):
            out_heads.append(weight(x_heads[:, i]))
        
        return torch.cat(out_heads, dim=1)
```

**创新点**:
- MHF 的通道分离 + TFNO 的张量化
- 每个头独立低秩分解
- 参数: (in_ch × out_ch × modes) / n_heads × rank

---

## 算子 3: UNO + MHF → MHF-PyramidFNO

### 问题分析
UNO 有多尺度结构，直接替换可能破坏尺度关系。

### 新设计: MHF-PyramidFNO

**核心思想**: 不同尺度使用不同的 MHF 配置

```python
class MHFPyramidFNO(nn.Module):
    """多尺度 MHF"""
    def __init__(self, ...):
        super().__init__()
        
        # 编码器 (高分辨率 → 低分辨率)
        # 使用更多 heads，捕获全局
        self.encoder_mhf = nn.ModuleList([
            MHFSpectralConv(ch, ch, modes, n_heads=8)  # 多头
            for ch, modes in encoder_config
        ])
        
        # 瓶颈 (最低分辨率)
        # 使用标准 SpectralConv，保持精度
        self.bottleneck = SpectralConv(bottleneck_ch, bottleneck_ch, modes)
        
        # 解码器 (低分辨率 → 高分辨率)
        # 使用更少 heads，保留细节
        self.decoder_mhf = nn.ModuleList([
            MHFSpectralConv(ch, ch, modes, n_heads=2)  # 少头
            for ch, modes in decoder_config
        ])
    
    def forward(self, x):
        # 编码: 多头 MHF → 全局特征
        skips = []
        for enc in self.encoder_mhf:
            x = enc(x)
            skips.append(x)
        
        # 瓶颈: 标准 SpectralConv → 精度
        x = self.bottleneck(x)
        
        # 解码: 少头 MHF → 细节保留
        for dec, skip in zip(self.decoder_mhf, reversed(skips)):
            x = dec(x + skip)
        
        return x
```

**创新点**:
- 编码器用多头 MHF（全局特征）
- 瓶颈用标准 SpectralConv（精度）
- 解码器用少头 MHF（细节保留）

---

## 算子 4: MHF-FNO3D → MHF-Separable3D

### 问题分析
3D FFT 内存占用高，MHF 可以减少参数，但 3D 频域更复杂。

### 新设计: MHF-Separable3D

**核心思想**: 可分离 3D FFT + MHF

```python
class MHFSeparable3D(nn.Module):
    """可分离 3D MHF"""
    def __init__(self, channels, n_modes_3d, n_heads=4):
        super().__init__()
        
        # 分离的 1D FFT
        self.mhf_x = MHFSpectralConv1D(channels, n_modes_3d[0], n_heads)
        self.mhf_y = MHFSpectralConv1D(channels, n_modes_3d[1], n_heads)
        self.mhf_z = MHFSpectralConv1D(channels, n_modes_3d[2], n_heads)
        
        # 融合
        self.fusion = nn.Conv3d(channels * 3, channels, 1)
    
    def forward(self, x):
        # x: (B, C, D, H, W)
        
        # 分别处理三个方向
        x_x = self.mhf_x(x.transpose(-1, -3)).transpose(-1, -3)  # X 方向
        x_y = self.mhf_y(x.transpose(-1, -2)).transpose(-1, -2)  # Y 方向
        x_z = self.mhf_z(x)                                       # Z 方向
        
        # 融合
        return self.fusion(torch.cat([x_x, x_y, x_z], dim=1))
```

**创新点**:
- 3D 问题分解为 3 个 1D MHF
- 内存: O(n³) → O(3n)
- 每个方向独立的频率模式

---

## 🎯 设计原则总结

| 算子 | MHF 角色 | 新设计核心 |
|------|----------|-----------|
| FNO | 低频处理器 | MHF + 高频残差 |
| TFNO | 张量化前置 | 通道分离 + 低秩分解 |
| UNO | 多尺度适配 | 编码多头/解码少头 |
| FNO3D | 可分离核心 | 3×1D MHF |

---

## 📝 下一步计划

1. 实现 MHF-ResFNO
2. 实现 MHF-PyramidFNO
3. 对比测试新算子 vs 原算子 + MHF 替换