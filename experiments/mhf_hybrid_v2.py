"""
MHF-FNO 混合策略测试 v2

方案3: 关键层使用 MHF，非关键层保持标准 SpectralConv

策略: 直接替换 FNO 模型中的指定层
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import json
import gc

from neuralop.models import FNO
from neuralop.layers.spectral_convolution import SpectralConv
from neuralop.losses.data_losses import LpLoss


class MHFSpectralConv(SpectralConv):
    """MHF SpectralConv - 兼容 NeuralOperator 接口"""
    
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
    
    # 替换指定层
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


def train_model(model, train_x, train_y, test_x, test_y, epochs=100, batch_size=32, lr=1e-3):
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
        
        model.eval()
        with torch.no_grad():
            test_loss = loss_fn(model(test_x), test_y).item()
        if test_loss < best_loss:
            best_loss = test_loss
        model.train()
        
        if (epoch + 1) % 25 == 0:
            print(f"    Epoch {epoch+1}: Test Loss = {test_loss:.4f}")
    
    train_time = time.time() - start
    return best_loss, train_time


def main():
    print("\n" + "=" * 70)
    print(" MHF-FNO 混合策略测试 v2")
    print(" 方案3: 关键层使用 MHF，非关键层保持标准 SpectralConv")
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
    hidden = 32
    n_layers = 3
    epochs = 100
    batch_size = 32
    n_heads = 4
    
    results = []
    
    # ===== 基准 =====
    print("\n" + "=" * 70)
    print(" 🔬 基准: 标准 FNO-32")
    print("=" * 70)
    
    model_fno = FNO(n_modes=n_modes, hidden_channels=hidden, in_channels=1, out_channels=1, n_layers=n_layers)
    params_fno = sum(p.numel() for p in model_fno.parameters())
    print(f"参数量: {params_fno:,}")
    
    loss_fno, time_fno = train_model(model_fno, train_x, train_y, test_x, test_y, epochs, batch_size)
    print(f"最终 Loss: {loss_fno:.4f}")
    
    results.append({
        'name': 'FNO-32',
        'mhf_layers': [],
        'params': params_fno,
        'loss': loss_fno,
        'time': time_fno
    })
    
    del model_fno
    gc.collect()
    
    # ===== 混合策略 =====
    print("\n" + "=" * 70)
    print(" 🔬 混合策略测试")
    print("=" * 70)
    
    strategies = [
        ([0], "第1层用MHF"),
        ([1], "第2层用MHF"),
        ([2], "第3层用MHF"),
        ([0, 1], "第1+2层用MHF"),
        ([1, 2], "第2+3层用MHF"),
        ([0, 2], "第1+3层用MHF"),
        ([0, 1, 2], "全部用MHF"),
    ]
    
    for mhf_layers, desc in strategies:
        print(f"\n策略: {desc}")
        
        try:
            model = create_hybrid_fno(
                n_modes=n_modes,
                hidden_channels=hidden,
                in_channels=1,
                out_channels=1,
                n_layers=n_layers,
                mhf_layers=mhf_layers,
                n_heads=n_heads
            )
            
            params = sum(p.numel() for p in model.parameters())
            params_change = (params - params_fno) / params_fno * 100
            
            loss, train_time = train_model(model, train_x, train_y, test_x, test_y, epochs, batch_size)
            loss_change = (loss - loss_fno) / loss_fno * 100
            
            print(f"  参数: {params:,} ({params_change:+.1f}%)")
            print(f"  Loss: {loss:.4f} ({loss_change:+.1f}%)")
            
            if loss < loss_fno:
                print(f"  🎉 优于 FNO-32!")
            
            results.append({
                'name': desc,
                'mhf_layers': mhf_layers,
                'params': params,
                'params_change': params_change,
                'loss': loss,
                'loss_change': loss_change,
                'time': train_time
            })
            
            del model
            gc.collect()
            
        except Exception as e:
            print(f"  ❌ 错误: {e}")
    
    # ===== 汇总 =====
    print("\n" + "=" * 70)
    print(" 📈 结果汇总")
    print("=" * 70)
    
    print(f"\n{'策略':<20} {'参数量':<12} {'参数变化':<10} {'L2误差':<10} {'误差变化':<10}")
    print("-" * 65)
    
    for r in results:
        if 'params_change' not in r:
            print(f"{r['name']:<20} {r['params']:<12,} {'(基准)':<10} {r['loss']:<10.4f} {'--':<10}")
        else:
            print(f"{r['name']:<20} {r['params']:<12,} {r['params_change']:+.1f}%{'':<5} {r['loss']:<10.4f} {r['loss_change']:+.1f}%")
    
    # 找最佳
    mhf_results = [r for r in results if 'loss_change' in r]
    
    if mhf_results:
        best = min(mhf_results, key=lambda x: x['loss'])
        print(f"\n🏆 最佳策略: {best['name']}")
        print(f"   参数: {best['params']:,}, Loss: {best['loss']:.4f}")
        
        better = [r for r in mhf_results if r['loss'] < loss_fno]
        if better:
            print(f"\n🎉 超越 FNO-32 的策略:")
            for r in sorted(better, key=lambda x: x['loss']):
                print(f"   {r['name']}: Loss={r['loss']:.4f}")
        else:
            print(f"\n⚠️ 未找到超越 FNO-32 的策略")
    
    with open('hybrid_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n结果已保存到 hybrid_results.json")


if __name__ == "__main__":
    main()