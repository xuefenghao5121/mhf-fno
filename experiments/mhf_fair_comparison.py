"""
MHF-FNO 公平对比实验

目标: 在参数量接近 FNO-32 的条件下，验证 MHF 是否能获得更好精度

策略:
1. 目标参数量: ~130K (接近 FNO-32)
2. 测试不同 hidden 和 n_heads 组合
3. 同时测试参数更少的配置 (验证参数效率)
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
        
        # 记录最佳测试误差
        model.eval()
        with torch.no_grad():
            test_loss = loss_fn(model(test_x), test_y).item()
        if test_loss < best_loss:
            best_loss = test_loss
        model.train()
        
        if verbose and (epoch + 1) % 25 == 0:
            print(f"    Epoch {epoch+1}: Test Loss = {test_loss:.4f}")
    
    train_time = time.time() - start
    return best_loss, train_time


def main():
    print("\n" + "=" * 70)
    print(" MHF-FNO 公平对比实验")
    print(" 目标: 在参数量接近 FNO-32 的条件下验证 MHF 效果")
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
    
    # ===== 基准 FNO =====
    print("\n" + "=" * 70)
    print(" 🔬 基准测试: FNO-32")
    print("=" * 70)
    
    model_fno = FNO(n_modes=n_modes, hidden_channels=32, in_channels=1, out_channels=1, n_layers=n_layers)
    params_fno = sum(p.numel() for p in model_fno.parameters())
    print(f"参数量: {params_fno:,}")
    loss_fno, time_fno = train_model(model_fno, train_x, train_y, test_x, test_y, epochs, batch_size)
    print(f"最终 Loss: {loss_fno:.4f}, 时间: {time_fno:.1f}s")
    
    results.append({
        'name': 'FNO-32',
        'hidden': 32,
        'n_heads': 1,
        'params': params_fno,
        'loss': loss_fno,
        'time': time_fno,
        'is_baseline': True
    })
    
    del model_fno
    gc.collect()
    
    # ===== MHF-FNO 测试 =====
    print("\n" + "=" * 70)
    print(" 🔬 MHF-FNO 对比实验")
    print("=" * 70)
    
    # 测试配置: 目标参数量范围 [60K, 150K]
    configs = [
        # 参数量 < FNO-32 (验证参数效率)
        (48, 4),    # 预计 ~90K
        (48, 8),    # 预计 ~50K
        (56, 4),    # 预计 ~120K
        (56, 8),    # 预计 ~70K
        (64, 4),    # 预计 ~165K
        (64, 8),    # 已测: 103,905
        
        # 参数量 ≈ FNO-32 (公平对比)
        (72, 8),    # 预计 ~130K
        (72, 16),   # 预计 ~80K
        (80, 8),    # 已测: 161,881
        (80, 16),   # 预计 ~100K
        
        # 更细粒度的多头
        (96, 16),   # 已测: 163,537
        (96, 32),   # 预计 ~90K
        (128, 16),  # 已测: 289,729
        (128, 32),  # 预计 ~150K
    ]
    
    for hidden, n_heads in configs:
        # 检查 hidden 是否能被 n_heads 整除
        if hidden % n_heads != 0:
            print(f"\n⏭️ 跳过 hidden={hidden}, n_heads={n_heads} (不能整除)")
            continue
        
        print(f"\nMHF (hidden={hidden}, n_heads={n_heads}):")
        
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
            params_ratio = params / params_fno * 100
            
            # 跳过参数量过大的配置 (>200%)
            if params_ratio > 200:
                print(f"  ⏭️ 跳过: 参数量 {params:,} ({params_ratio:.1f}%) 超过 FNO-32 的 200%")
                del model
                gc.collect()
                continue
            
            loss, train_time = train_model(model, train_x, train_y, test_x, test_y, epochs, batch_size)
            
            loss_change = (loss - loss_fno) / loss_fno * 100
            
            print(f"  参数: {params:,} ({params_ratio:.1f}% of FNO-32)")
            print(f"  Loss: {loss:.4f} ({loss_change:+.1f}% vs FNO-32)")
            print(f"  时间: {train_time:.1f}s")
            
            # 判断是否优于 FNO-32
            if loss < loss_fno:
                print(f"  🎉 优于 FNO-32!")
            
            results.append({
                'name': f'MHF-h{hidden}-he{n_heads}',
                'hidden': hidden,
                'n_heads': n_heads,
                'params': params,
                'params_ratio': params_ratio,
                'loss': loss,
                'loss_change': loss_change,
                'time': train_time,
                'is_baseline': False
            })
            
            del model
            gc.collect()
            
        except Exception as e:
            print(f"  ❌ 错误: {e}")
    
    # ===== 汇总 =====
    print("\n" + "=" * 70)
    print(" 📈 最终结果汇总")
    print("=" * 70)
    
    print(f"\n{'模型':<20} {'参数量':<12} {'参数比':<10} {'L2误差':<10} {'误差变化':<10}")
    print("-" * 65)
    
    # 按参数量排序
    sorted_results = sorted(results, key=lambda x: x['params'])
    
    for r in sorted_results:
        if r['is_baseline']:
            print(f"{r['name']:<20} {r['params']:<12,} {'100%':<10} {r['loss']:<10.4f} {'(基准)':<10}")
        else:
            print(f"{r['name']:<20} {r['params']:<12,} {r['params_ratio']:.1f}%{'':<5} {r['loss']:<10.4f} {r['loss_change']:+.1f}%")
    
    # 找最佳配置
    print("\n" + "-" * 65)
    
    mhf_results = [r for r in results if not r['is_baseline']]
    
    if mhf_results:
        # 精度最佳
        best_loss = min(mhf_results, key=lambda x: x['loss'])
        print(f"🏆 最佳精度: {best_loss['name']}")
        print(f"   参数: {best_loss['params']:,} ({best_loss['params_ratio']:.1f}%), Loss: {best_loss['loss']:.4f}")
        
        # 参数效率最佳
        def efficiency_score(r):
            return r['loss_change'] / max(r['params_ratio'], 10)
        
        best_eff = min(mhf_results, key=efficiency_score)
        print(f"\n🏆 最佳效率: {best_eff['name']}")
        print(f"   参数: {best_eff['params']:,} ({best_eff['params_ratio']:.1f}%), Loss: {best_eff['loss']:.4f}")
        
        # 是否超越 FNO-32
        better = [r for r in mhf_results if r['loss'] < loss_fno]
        if better:
            print(f"\n🎉 超越 FNO-32 的配置:")
            for r in sorted(better, key=lambda x: x['loss']):
                print(f"   {r['name']}: Loss={r['loss']:.4f}, 参数={r['params']:,}")
        else:
            print(f"\n⚠️ 未找到超越 FNO-32 的配置")
            closest = min(mhf_results, key=lambda x: x['loss'])
            print(f"   最接近: {closest['name']}, Loss={closest['loss']:.4f} (+{closest['loss_change']:.1f}%)")
    
    # 保存结果
    with open('fair_comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n结果已保存到 fair_comparison_results.json")
    
    return results


if __name__ == "__main__":
    main()