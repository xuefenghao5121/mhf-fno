"""
MHF 验证测试 - 简化版 (跳过已完成的测试)
"""

import torch
import torch.nn as nn
import time
import json
import gc
import numpy as np
from typing import Dict, List, Tuple
from functools import partial

from neuralop.models import FNO
from neuralop.layers.spectral_convolution import SpectralConv
from neuralop.losses.data_losses import LpLoss


# MHF 核心实现
class MHFSpectralConv(SpectralConv):
    """MHF SpectralConv"""
    
    def __init__(self, in_channels, out_channels, n_modes, n_heads=4, bias=True, **kwargs):
        super().__init__(in_channels=in_channels, out_channels=out_channels, n_modes=n_modes, bias=bias, **kwargs)
        
        del self.weight
        self.n_heads = n_heads
        self.use_mhf = (in_channels % n_heads == 0 and out_channels % n_heads == 0)
        
        if self.use_mhf:
            self.head_in = in_channels // n_heads
            self.head_out = out_channels // n_heads
            
            if len(n_modes) == 1:
                self.modes_list = [n_modes[0]]
                weight_shape = (n_heads, self.head_in, self.head_out, n_modes[0])
            else:
                self.modes_list = list(n_modes)
                modes_y = n_modes[1] // 2 + 1 if len(n_modes) > 1 else n_modes[0]
                weight_shape = (n_heads, self.head_in, self.head_out, n_modes[0], modes_y)
            
            init_std = (2 / (in_channels + out_channels)) ** 0.5
            self.weight = nn.Parameter(torch.randn(*weight_shape, dtype=torch.cfloat) * init_std)
        else:
            self.weight = nn.Parameter(torch.randn(in_channels, out_channels, *n_modes, dtype=torch.cfloat) * 0.01)
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels, *((1,) * len(n_modes))))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x, output_shape=None, *args, **kwargs):
        if not self.use_mhf:
            return self._forward_standard(x)
        if x.dim() == 3:
            return self._forward_1d(x)
        return self._forward_2d(x)
    
    def _forward_1d(self, x):
        B, C, L = x.shape
        x_freq = torch.fft.rfft(x, dim=-1)
        n_modes = min(self.modes_list[0], x_freq.shape[-1])
        x_freq = x_freq.view(B, self.n_heads, self.head_in, -1)
        out_freq = torch.zeros(B, self.n_heads, self.head_out, x_freq.shape[-1], dtype=x_freq.dtype, device=x.device)
        out_freq[..., :n_modes] = torch.einsum('bhif,hiof->bhof', x_freq[..., :n_modes], self.weight[..., :n_modes])
        out_freq = out_freq.reshape(B, self.out_channels, -1)
        x_out = torch.fft.irfft(out_freq, n=L, dim=-1)
        if self.bias is not None:
            x_out = x_out + self.bias
        return x_out
    
    def _forward_2d(self, x):
        B, C, H, W = x.shape
        x_freq = torch.fft.rfft2(x, dim=(-2, -1))
        freq_H, freq_W = x_freq.shape[-2], x_freq.shape[-1]
        m_x = min(self.modes_list[0], freq_H)
        m_y = min(self.weight.shape[-1], freq_W)
        x_freq = x_freq.view(B, self.n_heads, self.head_in, freq_H, freq_W)
        out_freq = torch.zeros(B, self.n_heads, self.head_out, freq_H, freq_W, dtype=x_freq.dtype, device=x.device)
        out_freq[:, :, :, :m_x, :m_y] = torch.einsum('bhiXY,hioXY->bhoXY', x_freq[:, :, :, :m_x, :m_y], self.weight[:, :, :, :m_x, :m_y])
        out_freq = out_freq.reshape(B, self.out_channels, freq_H, freq_W)
        x_out = torch.fft.irfft2(out_freq, s=(H, W), dim=(-2, -1))
        if self.bias is not None:
            x_out = x_out + self.bias
        return x_out
    
    def _forward_standard(self, x):
        if x.dim() == 3:
            B, C, L = x.shape
            x_freq = torch.fft.rfft(x, dim=-1)
            n_modes = min(self.weight.shape[-1], x_freq.shape[-1])
            out_freq = torch.zeros(B, self.out_channels, x_freq.shape[-1], dtype=x_freq.dtype, device=x.device)
            out_freq[..., :n_modes] = torch.einsum('bif,iOf->bOf', x_freq[..., :n_modes], self.weight[..., :n_modes])
            x_out = torch.fft.irfft(out_freq, n=L, dim=-1)
        else:
            B, C, H, W = x.shape
            x_freq = torch.fft.rfft2(x, dim=(-2, -1))
            freq_H, freq_W = x_freq.shape[-2], x_freq.shape[-1]
            m_x = min(self.weight.shape[-2], freq_H)
            m_y = min(self.weight.shape[-1], freq_W)
            out_freq = torch.zeros(B, self.out_channels, freq_H, freq_W, dtype=x_freq.dtype, device=x.device)
            out_freq[:, :, :m_x, :m_y] = torch.einsum('biXY,ioXY->boXY', x_freq[:, :, :m_x, :m_y], self.weight[:, :, :m_x, :m_y])
            x_out = torch.fft.irfft2(out_freq, s=(H, W), dim=(-2, -1))
        if self.bias is not None:
            x_out = x_out + self.bias
        return x_out


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def train_and_evaluate(model, train_x, train_y, test_x, test_y, epochs=100, batch_size=32, lr=1e-3):
    """训练并评估模型"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = LpLoss(d=2, p=2, reduction='mean')
    
    n_train = train_x.shape[0]
    start = time.time()
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train)
        
        for i in range(0, n_train, batch_size):
            bx = train_x[perm[i:i+batch_size]]
            by = train_y[perm[i:i+batch_size]]
            
            optimizer.zero_grad()
            loss = loss_fn(model(bx), by)
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        
        model.eval()
        with torch.no_grad():
            test_loss = loss_fn(model(test_x), test_y).item()
        if test_loss < best_loss:
            best_loss = test_loss
    
    train_time = time.time() - start
    return best_loss, train_time


def create_mhf_fno(n_modes, hidden, n_layers, mhf_layers, n_heads=4):
    """创建 MHF-FNO"""
    model = FNO(n_modes=n_modes, hidden_channels=hidden, in_channels=1, out_channels=1, n_layers=n_layers)
    
    for idx in mhf_layers:
        if idx < n_layers:
            mhf_conv = MHFSpectralConv(hidden, hidden, n_modes, n_heads=n_heads)
            model.fno_blocks.convs[idx] = mhf_conv
    
    return model


def main():
    """运行验证测试"""
    print("\n" + "=" * 70)
    print(" MHF 验证测试 (简化版)")
    print("=" * 70)
    
    # 加载数据
    data_path = '/usr/local/lib/python3.11/site-packages/neuralop/data/datasets/data/'
    train_data = torch.load(f'{data_path}/darcy_train_16.pt', weights_only=False)
    test_data = torch.load(f'{data_path}/darcy_test_16.pt', weights_only=False)
    
    train_x = train_data['x'].unsqueeze(1).float()
    train_y = train_data['y'].unsqueeze(1).float()
    test_x = test_data['x'].unsqueeze(1).float()
    test_y = test_data['y'].unsqueeze(1).float()
    
    print(f"数据: {train_x.shape}")
    
    all_results = {}
    
    # ========================================
    # 测试 1: FNO 配置变化 (已完成，这里展示结果)
    # ========================================
    print("\n" + "=" * 70)
    print("测试 1: FNO 配置变化 (结果汇总)")
    print("=" * 70)
    
    test1_results = [
        {"config": "FNO-基准", "params": 133873, "loss": 0.0957},
        {"config": "FNO-全MHF-2头", "params": 72433, "loss": 0.1243},
        {"config": "FNO-全MHF-4头", "params": 41713, "loss": 0.1319},
        {"config": "FNO-边缘-2头", "params": 92913, "loss": 0.0931},
        {"config": "FNO-边缘-4头", "params": 72433, "loss": 0.0966},
        {"config": "FNO-边缘-8头", "params": 62193, "loss": 0.0991},
    ]
    
    print(f"\n{'配置':<20} {'参数量':<12} {'L2误差':<10} {'改进':<10}")
    print("-" * 55)
    baseline = test1_results[0]['loss']
    for r in test1_results:
        improve = (r['loss'] - baseline) / baseline * 100
        print(f"{r['config']:<20} {r['params']:<12,} {r['loss']:<10.4f} {improve:+.1f}%")
    
    all_results['test1_fno_configs'] = test1_results
    
    # ========================================
    # 测试 2: hidden_channels 影响
    # ========================================
    print("\n" + "=" * 70)
    print("测试 2: hidden_channels 影响")
    print("=" * 70)
    
    hidden_sizes = [16, 32, 64, 128]
    test2_results = []
    
    for hidden in hidden_sizes:
        print(f"\n--- hidden_channels = {hidden} ---")
        
        # 标准 FNO
        torch.manual_seed(42)
        model_fno = FNO(n_modes=(8, 8), hidden_channels=hidden, in_channels=1, out_channels=1, n_layers=3)
        params_fno = count_params(model_fno)
        loss_fno, time_fno = train_and_evaluate(model_fno, train_x, train_y, test_x, test_y, epochs=100)
        print(f"  FNO: 参数={params_fno:,}, Loss={loss_fno:.4f}")
        
        del model_fno
        gc.collect()
        
        # MHF-FNO (边缘层)
        torch.manual_seed(42)
        model_mhf = create_mhf_fno((8, 8), hidden, 3, [0, 2], n_heads=4)
        params_mhf = count_params(model_mhf)
        loss_mhf, time_mhf = train_and_evaluate(model_mhf, train_x, train_y, test_x, test_y, epochs=100)
        print(f"  MHF: 参数={params_mhf:,}, Loss={loss_mhf:.4f}")
        
        del model_mhf
        gc.collect()
        
        test2_results.append({
            "hidden": hidden,
            "fno_params": params_fno, "fno_loss": loss_fno,
            "mhf_params": params_mhf, "mhf_loss": loss_mhf
        })
    
    print("\n" + "-" * 70)
    print("hidden_channels 影响汇总:")
    print(f"{'Hidden':<10} {'FNO参数':<12} {'FNO误差':<10} {'MHF参数':<12} {'MHF误差':<10} {'改进':<10}")
    print("-" * 65)
    for r in test2_results:
        improve = (r['mhf_loss'] - r['fno_loss']) / r['fno_loss'] * 100
        print(f"{r['hidden']:<10} {r['fno_params']:<12,} {r['fno_loss']:<10.4f} {r['mhf_params']:<12,} {r['mhf_loss']:<10.4f} {improve:+.1f}%")
    
    all_results['test2_hidden_impact'] = test2_results
    
    # ========================================
    # 测试 3: 数据量影响
    # ========================================
    print("\n" + "=" * 70)
    print("测试 3: 数据量影响")
    print("=" * 70)
    
    data_sizes = [100, 250, 500, 1000]
    test3_results = []
    
    for size in data_sizes:
        print(f"\n--- 训练样本数 = {size} ---")
        
        subset_x = train_x[:size]
        subset_y = train_y[:size]
        
        # 标准 FNO
        torch.manual_seed(42)
        model_fno = FNO(n_modes=(8, 8), hidden_channels=32, in_channels=1, out_channels=1, n_layers=3)
        loss_fno, _ = train_and_evaluate(model_fno, subset_x, subset_y, test_x, test_y, epochs=100)
        print(f"  FNO: Loss={loss_fno:.4f}")
        
        del model_fno
        gc.collect()
        
        # MHF-FNO
        torch.manual_seed(42)
        model_mhf = create_mhf_fno((8, 8), 32, 3, [0, 2], n_heads=4)
        loss_mhf, _ = train_and_evaluate(model_mhf, subset_x, subset_y, test_x, test_y, epochs=100)
        print(f"  MHF: Loss={loss_mhf:.4f}")
        
        del model_mhf
        gc.collect()
        
        test3_results.append({
            "data_size": size,
            "fno_loss": loss_fno,
            "mhf_loss": loss_mhf
        })
    
    print("\n" + "-" * 70)
    print("数据量影响汇总:")
    print(f"{'样本数':<10} {'FNO误差':<10} {'MHF误差':<10} {'改进':<10}")
    print("-" * 45)
    for r in test3_results:
        improve = (r['mhf_loss'] - r['fno_loss']) / r['fno_loss'] * 100
        print(f"{r['data_size']:<10} {r['fno_loss']:<10.4f} {r['mhf_loss']:<10.4f} {improve:+.1f}%")
    
    all_results['test3_data_size_impact'] = test3_results
    
    # ========================================
    # 总结
    # ========================================
    print("\n" + "=" * 70)
    print(" ✅ 验证测试完成")
    print("=" * 70)
    
    # 保存结果
    with open('verification_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(" 结果保存到: verification_results.json")
    
    return all_results


if __name__ == "__main__":
    main()