"""
MHF-FNO 官方框架集成测试 (修正版)

关键修正:
1. LpLoss 使用 reduction='mean' 而非默认的 'sum'
2. 正确的相对 L2 误差计算
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import time

# NeuralOperator 官方组件
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


def load_data():
    """加载数据"""
    data_path = '/usr/local/lib/python3.11/site-packages/neuralop/data/datasets/data/'
    train_data = torch.load(f'{data_path}/darcy_train_16.pt', weights_only=False)
    test_data = torch.load(f'{data_path}/darcy_test_16.pt', weights_only=False)
    return (train_data['x'].unsqueeze(1).float(), train_data['y'].unsqueeze(1).float(),
            test_data['x'].unsqueeze(1).float(), test_data['y'].unsqueeze(1).float())


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def train_model(model, train_x, train_y, test_x, test_y, epochs=50, batch_size=32, lr=1e-3):
    """训练模型"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)
    
    # ⭐ 关键修正: 使用 reduction='mean'
    loss_fn = LpLoss(d=2, p=2, reduction='mean')
    
    n_train = train_x.shape[0]
    start = time.time()
    history = []
    
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train)
        
        epoch_loss = 0.0
        n_batches = 0
        
        for i in range(0, n_train, batch_size):
            bx = train_x[perm[i:i+batch_size]]
            by = train_y[perm[i:i+batch_size]]
            
            optimizer.zero_grad()
            loss = loss_fn(model(bx), by)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        
        # 测试
        model.eval()
        with torch.no_grad():
            test_loss = loss_fn(model(test_x), test_y).item()
        
        history.append(test_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Train={epoch_loss/n_batches:.4f}, Test={test_loss:.4f}")
    
    final_loss = history[-1]
    train_time = time.time() - start
    return final_loss, train_time, history


def main():
    print("\n" + "=" * 70)
    print(" MHF-FNO 官方框架集成测试 (修正版)")
    print("=" * 70)
    
    print("\n📊 加载数据...")
    train_x, train_y, test_x, test_y = load_data()
    print(f"   训练集: {train_x.shape}")
    print(f"   测试集: {test_x.shape}")
    
    # 配置
    hidden = 32
    n_modes = (8, 8)
    n_layers = 3
    n_heads = 4
    epochs = 50
    
    results = {}
    
    # ===== 测试 FNO =====
    print("\n" + "-" * 50)
    print("🔬 测试标准 FNO (官方框架)")
    print("-" * 50)
    
    model_fno = FNO(
        n_modes=n_modes, 
        hidden_channels=hidden, 
        in_channels=1, 
        out_channels=1, 
        n_layers=n_layers
    )
    params_fno = count_params(model_fno)
    print(f"参数量: {params_fno:,}")
    
    loss_fno, time_fno, hist_fno = train_model(
        model_fno, train_x, train_y, test_x, test_y, epochs
    )
    
    print(f"\n最终 Loss: {loss_fno:.4f}")
    print(f"训练时间: {time_fno:.1f}s")
    
    results['FNO'] = {
        'params': params_fno,
        'loss': loss_fno,
        'time': time_fno,
        'history': hist_fno
    }
    
    # ===== 测试 MHF-FNO =====
    print("\n" + "-" * 50)
    print("🔬 测试 MHF-FNO (官方框架)")
    print("-" * 50)
    
    from functools import partial
    MHFConv4Heads = partial(MHFSpectralConv, n_heads=n_heads)
    
    model_mhf = FNO(
        n_modes=n_modes, 
        hidden_channels=hidden, 
        in_channels=1, 
        out_channels=1, 
        n_layers=n_layers,
        conv_module=MHFConv4Heads
    )
    params_mhf = count_params(model_mhf)
    print(f"参数量: {params_mhf:,}")
    
    loss_mhf, time_mhf, hist_mhf = train_model(
        model_mhf, train_x, train_y, test_x, test_y, epochs
    )
    
    print(f"\n最终 Loss: {loss_mhf:.4f}")
    print(f"训练时间: {time_mhf:.1f}s")
    
    results['MHF-FNO'] = {
        'params': params_mhf,
        'loss': loss_mhf,
        'time': time_mhf,
        'history': hist_mhf
    }
    
    # ===== 汇总 =====
    print("\n" + "=" * 70)
    print(" 📈 汇总对比")
    print("=" * 70)
    
    print(f"\n{'模型':<12} {'参数量':<12} {'L2误差':<10} {'时间':<8}")
    print("-" * 50)
    
    for name, res in results.items():
        print(f"{name:<12} {res['params']:<12,} {res['loss']:<10.4f} {res['time']:<8.1f}s")
    
    # 改进分析
    p_change = (results['MHF-FNO']['params'] - results['FNO']['params']) / results['FNO']['params'] * 100
    l_change = (results['MHF-FNO']['loss'] - results['FNO']['loss']) / results['FNO']['loss'] * 100
    t_change = (results['MHF-FNO']['time'] - results['FNO']['time']) / results['FNO']['time'] * 100
    
    print(f"\n📊 MHF-FNO vs FNO:")
    print(f"   参数量变化: {p_change:+.1f}%")
    print(f"   L2误差变化: {l_change:+.1f}%")
    print(f"   训练时间变化: {t_change:+.1f}%")
    
    # 判断结果
    print("\n" + "=" * 70)
    passed = 0
    if p_change < -50:
        print(f" ✅ 参数效率大幅提升 ({-p_change:.1f}% 减少)")
        passed += 1
    if l_change < 5:
        print(f" ✅ 精度保持 (误差增加 {l_change:.1f}%)")
        passed += 1
    if t_change < 20:
        print(f" ✅ 训练效率保持")
        passed += 1
    
    print(f"\n 通过 {passed}/3 项指标")
    
    return results


if __name__ == "__main__":
    main()