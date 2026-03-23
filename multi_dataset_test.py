"""
MHF-FNO 多数据集完整测试

测试数据集:
1. Darcy Flow 16×16 (已完成)
2. Darcy Flow 32×32
3. Burgers 1D
"""

import torch
import torch.nn as nn
import time
import json
import gc
from neuralop.models import FNO
from neuralop.layers.spectral_convolution import SpectralConv
from neuralop.losses.data_losses import LpLoss
from neuralop.data.datasets import load_darcy_flow_small, load_mini_burgers_1dtime
import numpy as np


class MHFSpectralConv(SpectralConv):
    """MHF SpectralConv"""
    
    def __init__(self, in_channels, out_channels, n_modes, n_heads=4, bias=True, **kwargs):
        if isinstance(n_modes, list):
            n_modes = tuple(n_modes)
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


def train_and_evaluate(model, train_loader, test_loader, epochs=100, lr=1e-3, is_1d=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = LpLoss(d=1 if is_1d else 2, p=2, reduction='mean')
    
    start = time.time()
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            if isinstance(batch, dict):
                x = batch['x']
                y = batch['y']
            else:
                x, y = batch
            
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        
        model.eval()
        with torch.no_grad():
            total_loss = 0
            count = 0
            for batch in test_loader:
                if isinstance(batch, dict):
                    x = batch['x']
                    y = batch['y']
                else:
                    x, y = batch
                total_loss += loss_fn(model(x), y).item()
                count += 1
            test_loss = total_loss / count
        
        if test_loss < best_loss:
            best_loss = test_loss
    
    return best_loss, time.time() - start


def test_darcy_16():
    """测试 Darcy Flow 16×16"""
    print("\n" + "=" * 70)
    print("数据集 1: Darcy Flow 16×16")
    print("=" * 70)
    
    # 加载数据
    train_loader, test_loaders, _ = load_darcy_flow_small(
        n_train=1000, n_tests=[100], batch_size=32, test_batch_sizes=[32], test_resolutions=[16]
    )
    test_loader = test_loaders[16]  # key 是分辨率
    
    results = []
    
    # 标准 FNO
    print("\n[1/2] 标准 FNO...")
    torch.manual_seed(42)
    model_fno = FNO(n_modes=(8, 8), hidden_channels=32, in_channels=1, out_channels=1, n_layers=3)
    params_fno = count_params(model_fno)
    loss_fno, time_fno = train_and_evaluate(model_fno, train_loader, test_loader, epochs=100)
    print(f"  参数: {params_fno:,}, Loss: {loss_fno:.4f}, 时间: {time_fno:.1f}s")
    results.append({"name": "FNO", "params": params_fno, "loss": loss_fno, "time": time_fno})
    del model_fno
    gc.collect()
    
    # MHF-FNO
    print("\n[2/2] MHF-FNO (边缘层)...")
    torch.manual_seed(42)
    model_mhf = FNO(n_modes=(8, 8), hidden_channels=32, in_channels=1, out_channels=1, n_layers=3)
    model_mhf.fno_blocks.convs[0] = MHFSpectralConv(32, 32, (8, 8), n_heads=4)
    model_mhf.fno_blocks.convs[2] = MHFSpectralConv(32, 32, (8, 8), n_heads=4)
    params_mhf = count_params(model_mhf)
    loss_mhf, time_mhf = train_and_evaluate(model_mhf, train_loader, test_loader, epochs=100)
    print(f"  参数: {params_mhf:,}, Loss: {loss_mhf:.4f}, 时间: {time_mhf:.1f}s")
    results.append({"name": "MHF-FNO", "params": params_mhf, "loss": loss_mhf, "time": time_mhf})
    del model_mhf
    gc.collect()
    
    return results


def test_darcy_32():
    """测试 Darcy Flow 32×32"""
    print("\n" + "=" * 70)
    print("数据集 2: Darcy Flow 32×32")
    print("=" * 70)
    
    try:
        train_loader, test_loaders, _ = load_darcy_flow_small(
            n_train=1000, n_tests=[100], batch_size=16, test_batch_sizes=[16], test_resolutions=[32]
        )
        test_loader = test_loaders[32]  # key 是分辨率
    except Exception as e:
        print(f"  ❌ 数据加载失败: {e}")
        return None
    
    results = []
    
    # 标准 FNO
    print("\n[1/2] 标准 FNO...")
    torch.manual_seed(42)
    model_fno = FNO(n_modes=(16, 16), hidden_channels=32, in_channels=1, out_channels=1, n_layers=3)
    params_fno = count_params(model_fno)
    loss_fno, time_fno = train_and_evaluate(model_fno, train_loader, test_loader, epochs=100)
    print(f"  参数: {params_fno:,}, Loss: {loss_fno:.4f}, 时间: {time_fno:.1f}s")
    results.append({"name": "FNO", "params": params_fno, "loss": loss_fno, "time": time_fno})
    del model_fno
    gc.collect()
    
    # MHF-FNO
    print("\n[2/2] MHF-FNO (边缘层)...")
    torch.manual_seed(42)
    model_mhf = FNO(n_modes=(16, 16), hidden_channels=32, in_channels=1, out_channels=1, n_layers=3)
    model_mhf.fno_blocks.convs[0] = MHFSpectralConv(32, 32, (16, 16), n_heads=4)
    model_mhf.fno_blocks.convs[2] = MHFSpectralConv(32, 32, (16, 16), n_heads=4)
    params_mhf = count_params(model_mhf)
    loss_mhf, time_mhf = train_and_evaluate(model_mhf, train_loader, test_loader, epochs=100)
    print(f"  参数: {params_mhf:,}, Loss: {loss_mhf:.4f}, 时间: {time_mhf:.1f}s")
    results.append({"name": "MHF-FNO", "params": params_mhf, "loss": loss_mhf, "time": time_mhf})
    del model_mhf
    gc.collect()
    
    return results


def main():
    print("\n" + "=" * 70)
    print(" MHF-FNO 多数据集完整测试")
    print("=" * 70)
    
    all_results = {}
    
    # 测试 Darcy 16
    all_results['darcy_16'] = test_darcy_16()
    
    # 测试 Darcy 32
    all_results['darcy_32'] = test_darcy_32()
    
    # 汇总结果
    print("\n" + "=" * 70)
    print(" 📊 测试结果汇总")
    print("=" * 70)
    
    for dataset, results in all_results.items():
        if results is None:
            continue
        
        print(f"\n{dataset}:")
        print(f"{'模型':<15} {'参数量':<12} {'L2误差':<10} {'改进':<10}")
        print("-" * 50)
        
        baseline = results[0]['loss']
        for r in results:
            improve = (r['loss'] - baseline) / baseline * 100
            print(f"{r['name']:<15} {r['params']:<12,} {r['loss']:<10.4f} {improve:+.1f}%")
    
    # 保存结果
    with open('multi_dataset_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print("\n结果已保存到: multi_dataset_results.json")
    
    return all_results


if __name__ == "__main__":
    main()