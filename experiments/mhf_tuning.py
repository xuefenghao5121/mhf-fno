"""
MHF-FNO 超参数调优实验

测试配置:
1. n_heads = 2, 4, 8
2. hidden_channels = 32, 48, 64
3. 对比参数量、精度、训练时间
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import gc
import json
from itertools import product

# NeuralOperator 官方组件
from neuralop.models import FNO
from neuralop.layers.spectral_convolution import SpectralConv
from neuralop.losses.data_losses import LpLoss
from functools import partial


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


def train_model(model, train_x, train_y, test_x, test_y, epochs=100, batch_size=32, lr=1e-3):
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
    
    model.eval()
    with torch.no_grad():
        final_loss = loss_fn(model(test_x), test_y).item()
    
    train_time = time.time() - start
    return final_loss, train_time


def main():
    print("\n" + "=" * 70)
    print(" MHF-FNO 超参数调优实验")
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
    
    # 基准 FNO
    print("\n" + "-" * 50)
    print("🔬 基准: 标准 FNO (hidden=32)")
    print("-" * 50)
    
    model_fno = FNO(n_modes=n_modes, hidden_channels=32, in_channels=1, out_channels=1, n_layers=n_layers)
    params_fno = sum(p.numel() for p in model_fno.parameters())
    loss_fno, time_fno = train_model(model_fno, train_x, train_y, test_x, test_y, epochs, batch_size)
    print(f"参数量: {params_fno:,}, Loss: {loss_fno:.4f}, 时间: {time_fno:.1f}s")
    
    del model_fno
    gc.collect()
    
    # 调优实验
    configs = [
        # (hidden, n_heads)
        (32, 2),   # n_heads=2, 保持表达能力
        (32, 4),   # 原始配置
        (32, 8),   # n_heads=8, 更激进的分离
        (48, 4),   # 增加hidden
        (64, 4),   # 更多hidden
        (64, 8),   # 大hidden + 多head
    ]
    
    results = []
    
    print("\n" + "-" * 50)
    print("🔬 MHF-FNO 调优实验")
    print("-" * 50)
    
    for hidden, n_heads in configs:
        print(f"\n配置: hidden={hidden}, n_heads={n_heads}")
        
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
            
            # 计算改进
            params_change = (params - params_fno) / params_fno * 100
            loss_change = (loss - loss_fno) / loss_fno * 100
            
            print(f"  参数: {params:,} ({params_change:+.1f}%), Loss: {loss:.4f} ({loss_change:+.1f}%), 时间: {train_time:.1f}s")
            
            results.append({
                'hidden': hidden,
                'n_heads': n_heads,
                'params': params,
                'loss': loss,
                'time': train_time,
                'params_change': params_change,
                'loss_change': loss_change
            })
            
            del model
            gc.collect()
            
        except Exception as e:
            print(f"  ❌ 错误: {e}")
    
    # 汇总
    print("\n" + "=" * 70)
    print(" 📈 调优结果汇总")
    print("=" * 70)
    
    print(f"\n{'配置':<20} {'参数量':<12} {'L2误差':<10} {'参数变化':<10} {'误差变化':<10}")
    print("-" * 65)
    print(f"{'FNO (基准)':<20} {params_fno:<12,} {loss_fno:<10.4f} {'--':<10} {'--':<10}")
    
    for r in results:
        config = f"MHF h={r['hidden']},he={r['n_heads']}"
        print(f"{config:<20} {r['params']:<12,} {r['loss']:<10.4f} {r['params_change']:+.1f}%{'':<5} {r['loss_change']:+.1f}%")
    
    # 找最佳配置
    print("\n" + "-" * 60)
    
    # 精度最接近 FNO 的配置
    best_loss = min(results, key=lambda x: x['loss'])
    print(f"✅ 最佳精度: hidden={best_loss['hidden']}, n_heads={best_loss['n_heads']}")
    print(f"   参数: {best_loss['params']:,}, Loss: {best_loss['loss']:.4f}")
    
    # 参数效率最佳
    best_efficiency = min(results, key=lambda x: x['loss_change'] / abs(x['params_change']))
    print(f"\n✅ 最佳效率: hidden={best_efficiency['hidden']}, n_heads={best_efficiency['n_heads']}")
    print(f"   参数: {best_efficiency['params']:,}, Loss: {best_efficiency['loss']:.4f}")
    
    # 保存结果
    with open('tuning_results.json', 'w') as f:
        json.dump({
            'baseline': {'params': params_fno, 'loss': loss_fno},
            'results': results
        }, f, indent=2)
    print(f"\n结果已保存到 tuning_results.json")


if __name__ == "__main__":
    main()