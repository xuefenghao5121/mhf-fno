"""
MHF-FNO 深度调优实验

目标: 找到精度接近或超越 FNO 的 MHF-FNO 配置

策略:
1. 增大 hidden_channels (64, 96, 128)
2. 调整 n_heads (2, 4, 8)
3. 保持参数量在合理范围内
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import gc
import json
from functools import partial

from neuralop.models import FNO
from neuralop.layers.spectral_convolution import SpectralConv
from neuralop.losses.data_losses import LpLoss


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
        
        if in_channels % n_heads != 0 or out_channels % n_heads != 0:
            self.use_mhf = False
            self.weight = nn.Parameter(
                torch.randn(in_channels, out_channels, *n_modes, dtype=torch.cfloat) * 0.01)
        else:
            self.use_mhf = True
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


def train_model(model, train_x, train_y, test_x, test_y, epochs=100, batch_size=32, lr=1e-3, verbose=True):
    """训练模型"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    loss_fn = LpLoss(d=2, p=2, reduction='mean')
    
    n_train = train_x.shape[0]
    start = time.time()
    
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
        
        if verbose and (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                test_loss = loss_fn(model(test_x), test_y).item()
            print(f"    Epoch {epoch+1}: Test Loss = {test_loss:.4f}")
            model.train()
    
    model.eval()
    with torch.no_grad():
        final_loss = loss_fn(model(test_x), test_y).item()
    
    train_time = time.time() - start
    return final_loss, train_time


def main():
    print("\n" + "=" * 70)
    print(" MHF-FNO 深度调优实验")
    print(" 目标: 找到精度接近或超越 FNO 的配置")
    print("=" * 70)
    
    # 加载数据
    print("\n📊 加载数据...")
    data_path = '/usr/local/lib/python3.11/site-packages/neuralop/data/datasets/data/'
    train_data = torch.load(f'{data_path}/darcy_train_16.pt', weights_only=False)
    test_data = torch.load(f'{data_path}/darcy_test_16.pt', weights_only=False)
    
    train_x = train_data['x'].unsqueeze(1).float()
    train_y = train_data['y'].unsqueeze(1).float()
    test_x = test_data['x'].unsqueeze(1).float()
    test_y = test_data['y'].unsqueeze(1).float()
    
    print(f"   训练集: {train_x.shape}")
    print(f"   测试集: {test_x.shape}")
    
    # 配置
    n_modes = (8, 8)
    n_layers = 3
    epochs = 100
    batch_size = 32
    
    results = []
    
    # ===== 基准 FNO (hidden=32) =====
    print("\n" + "=" * 70)
    print(" 🔬 基准测试: 标准 FNO")
    print("=" * 70)
    
    for hidden in [32, 64]:
        print(f"\nFNO (hidden={hidden}):")
        model = FNO(n_modes=n_modes, hidden_channels=hidden, in_channels=1, out_channels=1, n_layers=n_layers)
        params = sum(p.numel() for p in model.parameters())
        loss, train_time = train_model(model, train_x, train_y, test_x, test_y, epochs, batch_size)
        
        print(f"  参数: {params:,}, Loss: {loss:.4f}, 时间: {train_time:.1f}s")
        
        results.append({
            'name': f'FNO-h{hidden}',
            'hidden': hidden,
            'n_heads': 1,
            'params': params,
            'loss': loss,
            'time': train_time,
            'is_baseline': True
        })
        
        del model
        gc.collect()
    
    # 保存基准
    baseline_32 = results[0]
    baseline_64 = results[1]
    
    # ===== MHF-FNO 调优 =====
    print("\n" + "=" * 70)
    print(" 🔬 MHF-FNO 调优实验")
    print("=" * 70)
    
    # 测试配置: (hidden, n_heads)
    configs = [
        # 目标: 参数量接近 FNO-32 (133K)，精度更好
        (64, 8),     # 预计 ~104K
        (80, 8),     # 预计 ~160K
        (96, 8),     # 预计 ~230K
        (96, 16),    # 更激进的 head 分离
        (128, 8),    # 预计 ~400K
        (128, 16),   # 预计 ~200K
        (160, 8),    # 大模型
        (160, 16),   # 预计 ~300K
    ]
    
    for hidden, n_heads in configs:
        print(f"\nMHF (hidden={hidden}, n_heads={n_heads}):")
        
        # 检查 hidden 是否能被 n_heads 整除
        if hidden % n_heads != 0:
            print(f"  ⚠️ 跳过: {hidden} 不能被 {n_heads} 整除")
            continue
        
        try:
            MHFConv = partial(MHFSpectralConv, n_heads=n_heads)
            model = FNO(
                n_modes=n_modes,
                hidden_channels=hidden,
                in_channels=1,
                out_channels=1,
                n_layers=n_layers,
                conv_module=MHFConv
            )
            
            params = sum(p.numel() for p in model.parameters())
            loss, train_time = train_model(model, train_x, train_y, test_x, test_y, epochs, batch_size)
            
            # 相对基准的变化
            params_change_32 = (params - baseline_32['params']) / baseline_32['params'] * 100
            loss_change_32 = (loss - baseline_32['loss']) / baseline_32['loss'] * 100
            params_change_64 = (params - baseline_64['params']) / baseline_64['params'] * 100
            loss_change_64 = (loss - baseline_64['loss']) / baseline_64['loss'] * 100
            
            print(f"  参数: {params:,}, Loss: {loss:.4f}, 时间: {train_time:.1f}s")
            print(f"  vs FNO-32: 参数 {params_change_32:+.1f}%, 误差 {loss_change_32:+.1f}%")
            print(f"  vs FNO-64: 参数 {params_change_64:+.1f}%, 误差 {loss_change_64:+.1f}%")
            
            results.append({
                'name': f'MHF-h{hidden}-he{n_heads}',
                'hidden': hidden,
                'n_heads': n_heads,
                'params': params,
                'loss': loss,
                'time': train_time,
                'is_baseline': False,
                'params_change_32': params_change_32,
                'loss_change_32': loss_change_32,
                'params_change_64': params_change_64,
                'loss_change_64': loss_change_64
            })
            
            del model
            gc.collect()
            
        except Exception as e:
            print(f"  ❌ 错误: {e}")
    
    # ===== 汇总 =====
    print("\n" + "=" * 70)
    print(" 📈 调优结果汇总")
    print("=" * 70)
    
    print(f"\n{'模型':<20} {'参数量':<12} {'L2误差':<10} {'vs FNO-32':<15} {'vs FNO-64':<15}")
    print("-" * 75)
    
    for r in results:
        if r['is_baseline']:
            print(f"{r['name']:<20} {r['params']:<12,} {r['loss']:<10.4f} {'(基准)':<15} {'--':<15}")
        else:
            vs32 = f"{r['params_change_32']:+.1f}%/{r['loss_change_32']:+.1f}%"
            vs64 = f"{r['params_change_64']:+.1f}%/{r['loss_change_64']:+.1f}%"
            print(f"{r['name']:<20} {r['params']:<12,} {r['loss']:<10.4f} {vs32:<15} {vs64:<15}")
    
    # 找最佳配置
    print("\n" + "-" * 70)
    
    mhf_results = [r for r in results if not r['is_baseline']]
    
    # 精度最佳
    best_loss = min(mhf_results, key=lambda x: x['loss'])
    print(f"✅ 最佳精度: {best_loss['name']}")
    print(f"   参数: {best_loss['params']:,}, Loss: {best_loss['loss']:.4f}")
    
    # 效率最佳 (误差改善/参数增加比例)
    def efficiency_score(r):
        # 负的参数变化（减少）和负的误差变化（改善）都是好的
        return r['loss_change_32'] / max(abs(r['params_change_32']), 1)
    
    best_eff = min(mhf_results, key=efficiency_score)
    print(f"\n✅ 最佳效率: {best_eff['name']}")
    print(f"   参数: {best_eff['params']:,}, Loss: {best_eff['loss']:.4f}")
    
    # 是否超越 FNO
    print("\n" + "-" * 70)
    better_than_32 = [r for r in mhf_results if r['loss'] < baseline_32['loss']]
    better_than_64 = [r for r in mhf_results if r['loss'] < baseline_64['loss']]
    
    if better_than_32:
        print(f"🏆 超越 FNO-32 的配置:")
        for r in sorted(better_than_32, key=lambda x: x['loss']):
            print(f"   {r['name']}: Loss={r['loss']:.4f}, 参数={r['params']:,}")
    else:
        print("⚠️ 尚未找到超越 FNO-32 的配置")
    
    if better_than_64:
        print(f"🏆 超越 FNO-64 的配置:")
        for r in sorted(better_than_64, key=lambda x: x['loss']):
            print(f"   {r['name']}: Loss={r['loss']:.4f}, 参数={r['params']:,}")
    else:
        print("⚠️ 尚未找到超越 FNO-64 的配置")
    
    # 保存结果
    with open('deep_tuning_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n结果已保存到 deep_tuning_results.json")
    
    return results


if __name__ == "__main__":
    main()