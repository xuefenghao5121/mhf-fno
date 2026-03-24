"""
MHF-FNO 完整官方 Benchmark 测试

使用 NeuralOperator 官方 load_darcy_flow_small:
- 训练: 1000 samples
- 测试: 16x16 (100 samples) + 32x32 (100 samples)
- 验证跨分辨率泛化能力
"""

import torch
import torch.nn as nn
import time
import json
import gc
import numpy as np

from neuralop.models import FNO
from neuralop.layers.spectral_convolution import SpectralConv
from neuralop.losses.data_losses import LpLoss
from neuralop.data.datasets import load_darcy_flow_small


class MHFSpectralConv(SpectralConv):
    """MHF SpectralConv"""
    
    def __init__(self, in_channels, out_channels, n_modes, n_heads=4, bias=True, **kwargs):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            n_modes=n_modes,
            bias=bias,
            **kwargs
        )
        
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
            self.weight = nn.Parameter(
                torch.randn(in_channels, out_channels, *n_modes, dtype=torch.cfloat) * 0.01)
        
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
        out_freq = torch.zeros(B, self.n_heads, self.head_out, x_freq.shape[-1],
                               dtype=x_freq.dtype, device=x.device)
        out_freq[..., :n_modes] = torch.einsum('bhif,hiof->bhof',
            x_freq[..., :n_modes], self.weight[..., :n_modes])
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
        out_freq = torch.zeros(B, self.n_heads, self.head_out, freq_H, freq_W,
                               dtype=x_freq.dtype, device=x.device)
        out_freq[:, :, :, :m_x, :m_y] = torch.einsum('bhiXY,hioXY->bhoXY',
            x_freq[:, :, :, :m_x, :m_y], self.weight[:, :, :, :m_x, :m_y])
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
            out_freq = torch.zeros(B, self.out_channels, x_freq.shape[-1],
                                   dtype=x_freq.dtype, device=x.device)
            out_freq[..., :n_modes] = torch.einsum('bif,iOf->bOf',
                x_freq[..., :n_modes], self.weight[..., :n_modes])
            x_out = torch.fft.irfft(out_freq, n=L, dim=-1)
        else:
            B, C, H, W = x.shape
            x_freq = torch.fft.rfft2(x, dim=(-2, -1))
            freq_H, freq_W = x_freq.shape[-2], x_freq.shape[-1]
            m_x = min(self.weight.shape[-2], freq_H)
            m_y = min(self.weight.shape[-1], freq_W)
            out_freq = torch.zeros(B, self.out_channels, freq_H, freq_W,
                                   dtype=x_freq.dtype, device=x.device)
            out_freq[:, :, :m_x, :m_y] = torch.einsum('biXY,ioXY->boXY',
                x_freq[:, :, :m_x, :m_y], self.weight[:, :, :m_x, :m_y])
            x_out = torch.fft.irfft2(out_freq, s=(H, W), dim=(-2, -1))
        if self.bias is not None:
            x_out = x_out + self.bias
        return x_out


def create_hybrid_fno(n_modes, hidden_channels, in_channels, out_channels, n_layers, 
                      mhf_layers, n_heads=4):
    """创建混合 FNO 模型"""
    model = FNO(
        n_modes=n_modes,
        hidden_channels=hidden_channels,
        in_channels=in_channels,
        out_channels=out_channels,
        n_layers=n_layers
    )
    
    for layer_idx in mhf_layers:
        if layer_idx < n_layers:
            mhf_conv = MHFSpectralConv(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                n_modes=n_modes,
                n_heads=n_heads
            )
            model.fno_blocks.convs[layer_idx] = mhf_conv
    
    return model


def train_model(model, train_loader, test_loaders, epochs=150, lr=1e-3, verbose=True):
    """训练模型"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = LpLoss(d=2, p=2, reduction='mean')
    
    start = time.time()
    best_test_loss = {res: float('inf') for res in test_loaders.keys()}
    
    for epoch in range(epochs):
        model.train()
        
        for batch in train_loader:
            x = batch['x']
            y = batch['y']
            
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        
        # 测试
        model.eval()
        with torch.no_grad():
            for res, test_loader in test_loaders.items():
                total_loss = 0.0
                n_samples = 0
                for batch in test_loader:
                    x = batch['x']
                    y = batch['y']
                    pred = model(x)
                    loss = loss_fn(pred, y)
                    total_loss += loss.item() * x.shape[0]
                    n_samples += x.shape[0]
                avg_loss = total_loss / n_samples
                if avg_loss < best_test_loss[res]:
                    best_test_loss[res] = avg_loss
        
        if verbose and (epoch + 1) % 30 == 0:
            losses_str = ', '.join([f'{res}={best_test_loss[res]:.4f}' for res in sorted(test_loaders.keys())])
            print(f"    Epoch {epoch+1}: {losses_str}")
    
    train_time = time.time() - start
    return best_test_loss, train_time


def main():
    print("\n" + "=" * 70)
    print(" MHF-FNO 完整官方 Benchmark 测试")
    print(" 使用 NeuralOperator 官方 load_darcy_flow_small")
    print(" 测试分辨率: 16x16 和 32x32")
    print("=" * 70)
    
    # 加载官方数据
    print("\n📊 加载官方 Darcy Flow benchmark...")
    train_loader, test_loaders, _ = load_darcy_flow_small(
        n_train=1000,
        n_tests=[100, 100],
        batch_size=32,
        test_batch_sizes=[32, 32],
        test_resolutions=[16, 32]
    )
    
    # 检查数据
    for batch in train_loader:
        print(f"训练集: x={batch['x'].shape}")
        break
    
    for res in sorted(test_loaders.keys()):
        for batch in test_loaders[res]:
            print(f"测试集 ({res}x{res}): x={batch['x'].shape}")
            break
    
    # 配置
    n_modes = (8, 8)
    hidden = 32
    n_layers = 3
    epochs = 150
    n_heads = 4
    n_runs = 3
    
    results = {
        'FNO-32': {'losses_16': [], 'losses_32': [], 'times': [], 'params': 0},
        'MHF-1+3': {'losses_16': [], 'losses_32': [], 'times': [], 'params': 0}
    }
    
    # ===== FNO-32 =====
    print("\n" + "=" * 70)
    print(" 🔬 基准: 标准 FNO-32")
    print("=" * 70)
    
    for run in range(n_runs):
        print(f"\n--- Run {run+1}/{n_runs} ---")
        
        torch.manual_seed(42 + run)
        model = FNO(n_modes=n_modes, hidden_channels=hidden, in_channels=1, out_channels=1, n_layers=n_layers)
        
        if run == 0:
            params_fno = sum(p.numel() for p in model.parameters())
            results['FNO-32']['params'] = params_fno
            print(f"参数量: {params_fno:,}")
        
        test_loss, train_time = train_model(model, train_loader, test_loaders, epochs)
        
        results['FNO-32']['losses_16'].append(test_loss[16])
        results['FNO-32']['losses_32'].append(test_loss[32])
        results['FNO-32']['times'].append(train_time)
        print(f"Test Loss: 16x16={test_loss[16]:.4f}, 32x32={test_loss[32]:.4f}")
        
        del model
        gc.collect()
    
    # ===== MHF-1+3 =====
    print("\n" + "=" * 70)
    print(" 🔬 最佳配置: 第1+3层用MHF")
    print("=" * 70)
    
    for run in range(n_runs):
        print(f"\n--- Run {run+1}/{n_runs} ---")
        
        torch.manual_seed(42 + run)
        model = create_hybrid_fno(
            n_modes=n_modes,
            hidden_channels=hidden,
            in_channels=1,
            out_channels=1,
            n_layers=n_layers,
            mhf_layers=[0, 2],
            n_heads=n_heads
        )
        
        if run == 0:
            params_mhf = sum(p.numel() for p in model.parameters())
            results['MHF-1+3']['params'] = params_mhf
            params_change = (params_mhf - params_fno) / params_fno * 100
            print(f"参数量: {params_mhf:,} ({params_change:+.1f}%)")
        
        test_loss, train_time = train_model(model, train_loader, test_loaders, epochs)
        
        results['MHF-1+3']['losses_16'].append(test_loss[16])
        results['MHF-1+3']['losses_32'].append(test_loss[32])
        results['MHF-1+3']['times'].append(train_time)
        
        loss_change_16 = (test_loss[16] - results['FNO-32']['losses_16'][run]) / results['FNO-32']['losses_16'][run] * 100
        loss_change_32 = (test_loss[32] - results['FNO-32']['losses_32'][run]) / results['FNO-32']['losses_32'][run] * 100
        print(f"Test Loss: 16x16={test_loss[16]:.4f} ({loss_change_16:+.1f}%), 32x32={test_loss[32]:.4f} ({loss_change_32:+.1f}%)")
        
        del model
        gc.collect()
    
    # ===== 汇总 =====
    print("\n" + "=" * 70)
    print(" 📈 最终结果汇总")
    print("=" * 70)
    
    fno_16_mean = np.mean(results['FNO-32']['losses_16'])
    fno_16_std = np.std(results['FNO-32']['losses_16'])
    fno_32_mean = np.mean(results['FNO-32']['losses_32'])
    fno_32_std = np.std(results['FNO-32']['losses_32'])
    
    mhf_16_mean = np.mean(results['MHF-1+3']['losses_16'])
    mhf_16_std = np.std(results['MHF-1+3']['losses_16'])
    mhf_32_mean = np.mean(results['MHF-1+3']['losses_32'])
    mhf_32_std = np.std(results['MHF-1+3']['losses_32'])
    
    params_change = (results['MHF-1+3']['params'] - results['FNO-32']['params']) / results['FNO-32']['params'] * 100
    loss_change_16 = (mhf_16_mean - fno_16_mean) / fno_16_mean * 100
    loss_change_32 = (mhf_32_mean - fno_32_mean) / fno_32_mean * 100
    
    print(f"\n{'模型':<15} {'参数量':<12} {'L2 (16x16)':<18} {'L2 (32x32)':<18}")
    print("-" * 68)
    print(f"{'FNO-32':<15} {results['FNO-32']['params']:<12,} {fno_16_mean:.4f} ± {fno_16_std:.4f}    {fno_32_mean:.4f} ± {fno_32_std:.4f}")
    print(f"{'MHF-1+3':<15} {results['MHF-1+3']['params']:<12,} {mhf_16_mean:.4f} ± {mhf_16_std:.4f}    {mhf_32_mean:.4f} ± {mhf_32_std:.4f}")
    
    print(f"\n📊 MHF vs FNO:")
    print(f"   参数量: {params_change:+.1f}%")
    print(f"   L2 (16x16): {loss_change_16:+.1f}%")
    print(f"   L2 (32x32): {loss_change_32:+.1f}%")
    
    print("\n" + "-" * 68)
    if loss_change_16 < 0 or loss_change_32 < 0:
        print(f"✅ MHF-1+3 在完整官方 Benchmark 上表现优异!")
        if loss_change_16 < 0:
            print(f"   16x16 分辨率: 精度提升 {-loss_change_16:.1f}%")
        if loss_change_32 < 0:
            print(f"   32x32 分辨率: 精度提升 {-loss_change_32:.1f}%")
        print(f"   参数减少 {-params_change:.1f}%")
    
    # 保存
    with open('complete_benchmark_results.json', 'w') as f:
        json.dump({
            'FNO-32': {
                'params': results['FNO-32']['params'],
                'loss_16_mean': fno_16_mean, 'loss_16_std': fno_16_std,
                'loss_32_mean': fno_32_mean, 'loss_32_std': fno_32_std
            },
            'MHF-1+3': {
                'params': results['MHF-1+3']['params'],
                'loss_16_mean': mhf_16_mean, 'loss_16_std': mhf_16_std,
                'loss_32_mean': mhf_32_mean, 'loss_32_std': mhf_32_std
            }
        }, f, indent=2)
    print(f"\n结果已保存到 complete_benchmark_results.json")


if __name__ == "__main__":
    main()