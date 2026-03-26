#!/usr/bin/env python3
"""
Navier-Stokes 效果分析 + 优化方案验证

问题分析:
---------
1. 频率耦合特性
   - NS是双曲型PDE，存在湍流和涡旋结构
   - 高低频之间存在强烈耦合（能量级联）
   - MHF的独立头假设与NS物理特性冲突

2. 当前CoDA的局限
   - 注意力在空域进行，无法直接恢复频域耦合
   - 门控机制可能无法学习到正确的频率交互

优化方案:
---------
Plan A: 频域注意力 (Spectral Attention)
   - 在频域空间进行跨头注意力
   - 直接学习频率间的交互关系

Plan B: 自适应混合层 (Adaptive Mixing)
   - 每层学习MHF和标准卷积的混合比例
   - 模型自动决定何处需要频率耦合

Plan C: 残差频率连接 (Residual Frequency Connection)
   - 将标准卷积的残差连接到MHF层
   - 保留部分频率耦合能力

作者: Tianyuan Team - 天池
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralop.losses.data_losses import LpLoss
from neuralop.models import FNO
from neuralop.layers.spectral_convolution import SpectralConv

torch.set_num_threads(os.cpu_count() or 1)

sys.path.insert(0, str(Path(__file__).parent.parent))
from mhf_fno import MHFSpectralConv, create_hybrid_fno


# ============================================================================
# Plan A: 频域注意力
# ============================================================================

class SpectralCrossHeadAttention(nn.Module):
    """
    频域跨头注意力
    
    在频域空间进行跨头交互，直接学习频率间的耦合关系。
    """
    
    def __init__(self, n_heads, channels_per_head, n_modes):
        super().__init__()
        self.n_heads = n_heads
        self.channels_per_head = channels_per_head
        self.n_modes = n_modes
        
        # 频域注意力参数
        # Q, K, V 在频率维度上操作
        self.freq_attn = nn.MultiheadAttention(
            embed_dim=channels_per_head,
            num_heads=min(4, channels_per_head // 2),
            batch_first=True
        )
        
        # 频率位置编码
        self.freq_pos = nn.Parameter(
            torch.randn(1, n_modes[0], channels_per_head) * 0.02
        )
        
        # 门控
        self.gate = nn.Parameter(torch.zeros(1))
    
    def forward(self, x_freq):
        """
        Args:
            x_freq: [B, n_heads, C_per_head, H, W] 频域表示
        
        Returns:
            增强的频域表示
        """
        B, nH, C, H, W = x_freq.shape
        
        # 只处理低频部分 (n_modes)
        m = min(self.n_modes[0], H)
        
        # 重塑为注意力输入: [B*n_heads, C, m] -> [B*n_heads, m, C]
        x_low = x_freq[:, :, :, :m, 0]  # 取第一个频率维度 [B, nH, C, m]
        x_low = x_low.permute(0, 1, 3, 2)  # [B, nH, m, C]
        x_low = x_low.reshape(B * nH, m, C)
        
        # 添加频率位置编码
        x_low = x_low + self.freq_pos[:, :m, :]
        
        # 频域自注意力
        attn_out, _ = self.freq_attn(x_low, x_low, x_low)
        
        # 门控融合
        gate = torch.sigmoid(self.gate)
        x_low = gate * attn_out + (1 - gate) * x_low
        
        # 重塑回原形状
        x_low = x_low.reshape(B, nH, m, C).permute(0, 1, 3, 2)
        
        # 创建输出
        out = x_freq.clone()
        out[:, :, :, :m, 0] = x_low
        
        return out


class MHFSpectralConvV2(nn.Module):
    """
    MHF v2: 带频域注意力的多头频谱卷积
    
    改进:
    1. 在频域进行跨头注意力
    2. 保留频域表示，避免空域转换损失
    """
    
    def __init__(self, in_channels, out_channels, n_modes, n_heads=4, use_spectral_attn=True):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes = n_modes if isinstance(n_modes, tuple) else (n_modes,)
        self.n_heads = n_heads
        self.use_mhf = in_channels % n_heads == 0 and out_channels % n_heads == 0
        
        if self.use_mhf:
            self.head_in = in_channels // n_heads
            self.head_out = out_channels // n_heads
            
            # MHF 权重
            weight_shape = (n_heads, self.head_in, self.head_out, n_modes[0], n_modes[1] // 2 + 1)
            self.weight = nn.Parameter(
                torch.randn(*weight_shape, dtype=torch.cfloat) * 0.01
            )
            
            # 频域注意力
            if use_spectral_attn:
                self.spectral_attn = SpectralCrossHeadAttention(
                    n_heads=n_heads,
                    channels_per_head=self.head_out,
                    n_modes=n_modes
                )
            else:
                self.spectral_attn = None
        else:
            # 回退到标准卷积
            weight_shape = (in_channels, out_channels, n_modes[0], n_modes[1] // 2 + 1)
            self.weight = nn.Parameter(
                torch.randn(*weight_shape, dtype=torch.cfloat) * 0.01
            )
            self.spectral_attn = None
        
        if True:
            self.bias = nn.Parameter(torch.zeros(out_channels, 1, 1))
        else:
            self.bias = None
    
    def transform(self, x, output_shape=None):
        """兼容 FNO block 的 transform 接口"""
        return self.forward(x)
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # 2D FFT
        x_freq = torch.fft.rfft2(x, dim=(-2, -1))
        freq_H, freq_W = x_freq.shape[-2], x_freq.shape[-1]
        
        m_x = min(self.n_modes[0], freq_H)
        m_y = min(self.weight.shape[-1], freq_W)
        
        if self.use_mhf:
            # 多头处理
            x_freq = x_freq.view(B, self.n_heads, self.head_in, freq_H, freq_W)
            
            # 多头卷积
            out_freq = torch.zeros(
                B, self.n_heads, self.head_out, freq_H, freq_W,
                dtype=x_freq.dtype, device=x.device
            )
            out_freq[:, :, :, :m_x, :m_y] = torch.einsum(
                'bhiXY,hioXY->bhoXY',
                x_freq[:, :, :, :m_x, :m_y],
                self.weight[:, :, :, :m_x, :m_y]
            )
            
            # 频域注意力
            if self.spectral_attn is not None:
                out_freq = self.spectral_attn(out_freq)
            
            out_freq = out_freq.reshape(B, self.out_channels, freq_H, freq_W)
        else:
            out_freq = torch.zeros(
                B, self.out_channels, freq_H, freq_W,
                dtype=x_freq.dtype, device=x.device
            )
            out_freq[:, :, :m_x, :m_y] = torch.einsum(
                'biXY,ioXY->boXY',
                x_freq[:, :, :m_x, :m_y],
                self.weight[:, :, :m_x, :m_y]
            )
        
        # IFFT
        x_out = torch.fft.irfft2(out_freq, s=(H, W), dim=(-2, -1))
        
        if self.bias is not None:
            x_out = x_out + self.bias
        
        return x_out


# ============================================================================
# Plan B: 自适应混合层
# ============================================================================

class AdaptiveMixingConv(nn.Module):
    """
    自适应混合层
    
    学习 MHF 和标准卷积的混合比例，自动适应不同 PDE 的频率耦合需求。
    """
    
    def __init__(self, in_channels, out_channels, n_modes, n_heads=4):
        super().__init__()
        
        # MHF 分支
        self.mhf = MHFSpectralConv(
            in_channels=in_channels,
            out_channels=out_channels,
            n_modes=n_modes,
            n_heads=n_heads
        )
        
        # 标准卷积分支
        self.standard = SpectralConv(
            in_channels=in_channels,
            out_channels=out_channels,
            n_modes=n_modes
        )
        
        # 自适应混合参数 (可学习)
        self.mix_weight = nn.Parameter(torch.tensor(0.3))  # 初始偏向MHF
        
    def transform(self, x, output_shape=None):
        """兼容 FNO block 的 transform 接口"""
        return self.forward(x)
    
    def forward(self, x):
        # 两个分支
        mhf_out = self.mhf(x)
        std_out = self.standard(x)
        
        # 自适应混合
        alpha = torch.sigmoid(self.mix_weight)
        return alpha * mhf_out + (1 - alpha) * std_out


def create_adaptive_fno(n_modes, hidden_channels, in_channels=1, out_channels=1, 
                        n_layers=3, n_heads=4, mixing_layers=None):
    """创建自适应混合 FNO"""
    from neuralop.models import FNO
    
    model = FNO(
        n_modes=n_modes,
        hidden_channels=hidden_channels,
        in_channels=in_channels,
        out_channels=out_channels,
        n_layers=n_layers
    )
    
    if mixing_layers is None:
        mixing_layers = [0, n_layers - 1]
    
    for idx in mixing_layers:
        if idx < n_layers:
            model.fno_blocks.convs[idx] = AdaptiveMixingConv(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                n_modes=n_modes,
                n_heads=n_heads
            )
    
    return model


# ============================================================================
# Plan C: 残差频率连接
# ============================================================================

class MHFWithResidual(nn.Module):
    """
    带残差频率连接的 MHF
    
    保留标准卷积的残差路径，确保频率耦合信息不丢失。
    """
    
    def __init__(self, in_channels, out_channels, n_modes, n_heads=4, residual_weight=0.2):
        super().__init__()
        
        self.mhf = MHFSpectralConv(
            in_channels=in_channels,
            out_channels=out_channels,
            n_modes=n_modes,
            n_heads=n_heads
        )
        
        # 残差卷积 (压缩通道以节省参数)
        self.residual = SpectralConv(
            in_channels=in_channels,
            out_channels=out_channels,
            n_modes=n_modes
        )
        
        self.residual_weight = residual_weight
    
    def transform(self, x, output_shape=None):
        """兼容 FNO block 的 transform 接口"""
        return self.forward(x)
    
    def forward(self, x):
        mhf_out = self.mhf(x)
        res_out = self.residual(x)
        return mhf_out + self.residual_weight * res_out


def create_residual_fno(n_modes, hidden_channels, in_channels=1, out_channels=1,
                        n_layers=3, n_heads=4, residual_layers=None, residual_weight=0.2):
    """创建带残差频率连接的 FNO"""
    from neuralop.models import FNO
    
    model = FNO(
        n_modes=n_modes,
        hidden_channels=hidden_channels,
        in_channels=in_channels,
        out_channels=out_channels,
        n_layers=n_layers
    )
    
    if residual_layers is None:
        residual_layers = [0, n_layers - 1]
    
    for idx in residual_layers:
        if idx < n_layers:
            model.fno_blocks.convs[idx] = MHFWithResidual(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                n_modes=n_modes,
                n_heads=n_heads,
                residual_weight=residual_weight
            )
    
    return model


# ============================================================================
# 测试函数
# ============================================================================

def load_data():
    """加载数据"""
    train_path = Path(__file__).parent / 'data' / 'ns_train_32_large.pt'
    test_path = Path(__file__).parent / 'data' / 'ns_test_32_large.pt'
    
    if not train_path.exists():
        return None
    
    train_data = torch.load(train_path, weights_only=False)
    test_data = torch.load(test_path, weights_only=False)
    
    if isinstance(train_data, dict):
        train_x = train_data.get('x', train_data.get('train_x'))
        train_y = train_data.get('y', train_data.get('train_y'))
    else:
        train_x, train_y = train_data[0], train_data[1]
    
    if isinstance(test_data, dict):
        test_x = test_data.get('x', test_data.get('test_x'))
        test_y = test_data.get('y', test_data.get('test_y'))
    else:
        test_x, test_y = test_data[0], test_data[1]
    
    if train_x.dim() == 3:
        train_x = train_x.unsqueeze(1)
        train_y = train_y.unsqueeze(1)
    if test_x.dim() == 3:
        test_x = test_x.unsqueeze(1)
        test_y = test_y.unsqueeze(1)
    
    return train_x.float(), train_y.float(), test_x.float(), test_y.float()


def train(model, train_x, train_y, test_x, test_y, epochs=50, lr=1e-3, verbose=True):
    """训练函数"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = LpLoss(d=2, p=2, reduction='mean')
    
    best_loss = float('inf')
    n_train = train_x.shape[0]
    batch_size = 32
    
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train)
        
        for i in range(0, n_train, batch_size):
            bx = train_x[perm[i:i+batch_size]]
            by = train_y[perm[i:i+batch_size]]
            
            optimizer.zero_grad()
            loss = loss_fn(model(bx), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        scheduler.step()
        
        model.eval()
        with torch.no_grad():
            test_loss = loss_fn(model(test_x), test_y).item()
        
        if test_loss < best_loss:
            best_loss = test_loss
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: Test {test_loss:.4f} (best: {best_loss:.4f})")
    
    return best_loss


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def main():
    print("="*70)
    print("Navier-Stokes 优化方案验证")
    print("="*70)
    
    # 加载数据
    data = load_data()
    if data is None:
        print("无法加载数据")
        return
    
    train_x, train_y, test_x, test_y = data
    
    # 使用部分数据进行快速测试
    train_x, train_y = train_x[:500], train_y[:500]
    test_x, test_y = test_x[:100], test_y[:100]
    
    resolution = train_x.shape[-1]
    n_modes = (resolution // 4, resolution // 4)
    in_ch = train_x.shape[1]
    out_ch = train_y.shape[1]
    
    print(f"\n数据: {train_x.shape[0]} 训练, {test_x.shape[0]} 测试, 分辨率 {resolution}x{resolution}")
    print(f"n_modes: {n_modes}")
    
    results = {}
    
    # ========== 基准测试 ==========
    print(f"\n{'='*70}")
    print("基准测试")
    print(f"{'='*70}")
    
    # FNO
    print("\n[1/5] FNO (基准)...")
    torch.manual_seed(42)
    model_fno = FNO(n_modes=n_modes, hidden_channels=32, in_channels=in_ch, out_channels=out_ch, n_layers=3)
    params_fno = count_params(model_fno)
    loss_fno = train(model_fno, train_x, train_y, test_x, test_y, epochs=50)
    print(f"  参数: {params_fno:,}, 最佳Loss: {loss_fno:.4f}")
    results['FNO'] = {'params': params_fno, 'loss': loss_fno}
    
    # MHF-FNO
    print("\n[2/5] MHF-FNO...")
    torch.manual_seed(42)
    model_mhf = create_hybrid_fno(n_modes=n_modes, hidden_channels=32, in_channels=in_ch, out_channels=out_ch, mhf_layers=[0, 2])
    params_mhf = count_params(model_mhf)
    loss_mhf = train(model_mhf, train_x, train_y, test_x, test_y, epochs=50)
    improvement = (loss_fno - loss_mhf) / loss_fno * 100
    print(f"  参数: {params_mhf:,} ({(params_mhf/params_fno-1)*100:+.1f}%), 最佳Loss: {loss_mhf:.4f} ({improvement:+.2f}%)")
    results['MHF-FNO'] = {'params': params_mhf, 'loss': loss_mhf, 'improvement': improvement}
    
    # ========== 优化方案测试 ==========
    print(f"\n{'='*70}")
    print("优化方案测试")
    print(f"{'='*70}")
    
    # Plan A: 频域注意力
    print("\n[3/5] Plan A: 频域注意力...")
    torch.manual_seed(42)
    model_a = FNO(n_modes=n_modes, hidden_channels=32, in_channels=in_ch, out_channels=out_ch, n_layers=3)
    for idx in [0, 2]:
        model_a.fno_blocks.convs[idx] = MHFSpectralConvV2(
            in_channels=32, out_channels=32, n_modes=n_modes, n_heads=4, use_spectral_attn=True
        )
    params_a = count_params(model_a)
    loss_a = train(model_a, train_x, train_y, test_x, test_y, epochs=50)
    improvement_a = (loss_fno - loss_a) / loss_fno * 100
    print(f"  参数: {params_a:,}, 最佳Loss: {loss_a:.4f} ({improvement_a:+.2f}%)")
    results['PlanA_SpectralAttn'] = {'params': params_a, 'loss': loss_a, 'improvement': improvement_a}
    
    # Plan B: 自适应混合
    print("\n[4/5] Plan B: 自适应混合...")
    torch.manual_seed(42)
    model_b = create_adaptive_fno(n_modes=n_modes, hidden_channels=32, in_channels=in_ch, out_channels=out_ch, mixing_layers=[0, 2])
    params_b = count_params(model_b)
    loss_b = train(model_b, train_x, train_y, test_x, test_y, epochs=50)
    improvement_b = (loss_fno - loss_b) / loss_fno * 100
    # 检查混合权重
    for name, param in model_b.named_parameters():
        if 'mix_weight' in name:
            print(f"  学习到的混合权重: α = {torch.sigmoid(param).item():.4f}")
    print(f"  参数: {params_b:,}, 最佳Loss: {loss_b:.4f} ({improvement_b:+.2f}%)")
    results['PlanB_AdaptiveMixing'] = {'params': params_b, 'loss': loss_b, 'improvement': improvement_b}
    
    # Plan C: 残差频率连接
    print("\n[5/5] Plan C: 残差频率连接...")
    torch.manual_seed(42)
    model_c = create_residual_fno(n_modes=n_modes, hidden_channels=32, in_channels=in_ch, out_channels=out_ch, residual_layers=[0, 2], residual_weight=0.2)
    params_c = count_params(model_c)
    loss_c = train(model_c, train_x, train_y, test_x, test_y, epochs=50)
    improvement_c = (loss_fno - loss_c) / loss_fno * 100
    print(f"  参数: {params_c:,}, 最佳Loss: {loss_c:.4f} ({improvement_c:+.2f}%)")
    results['PlanC_Residual'] = {'params': params_c, 'loss': loss_c, 'improvement': improvement_c}
    
    # ========== 结果汇总 ==========
    print(f"\n{'='*70}")
    print("结果汇总")
    print(f"{'='*70}")
    
    print(f"\n{'方案':<25} {'参数量':<12} {'参数变化':<12} {'Test Loss':<12} {'vs FNO':<12}")
    print("-"*70)
    
    for name, res in results.items():
        param_change = (res['params'] / params_fno - 1) * 100
        improvement = res.get('improvement', (loss_fno - res['loss']) / loss_fno * 100)
        marker = "✅" if improvement > 0 else "⚠️"
        print(f"{name:<25} {res['params']:<12,} {param_change:+.1f}%{'':>5} {res['loss']:<12.4f} {improvement:+.2f}% {marker}")
    
    # 最佳方案
    best_name = min(results.items(), key=lambda x: x[1]['loss'])[0]
    print(f"\n🏆 最佳方案: {best_name}")
    
    # 保存结果
    output_path = Path(__file__).parent.parent / 'ns_optimization_analysis.json'
    with open(output_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'config': {
                'n_train': train_x.shape[0],
                'n_test': test_x.shape[0],
                'resolution': resolution,
                'n_modes': n_modes,
                'epochs': 50
            },
            'results': results,
            'best_plan': best_name
        }, f, indent=2)
    
    print(f"\n✅ 结果已保存到: {output_path}")
    
    return results


if __name__ == '__main__':
    main()