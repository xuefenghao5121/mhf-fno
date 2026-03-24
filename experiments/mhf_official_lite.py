"""
MHF-FNO 官方框架集成测试 (轻量版)

针对资源受限环境优化:
1. 减少训练样本数
2. 更小的 batch size
3. 更少的 epochs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import gc

# NeuralOperator 官方组件
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


def main():
    print("\n" + "=" * 70)
    print(" MHF-FNO 官方框架集成测试 (轻量版)")
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
    
    # 只使用部分训练数据
    n_train = 200  # 减少训练样本
    train_x = train_x[:n_train]
    train_y = train_y[:n_train]
    
    print(f"   训练集: {train_x.shape}")
    print(f"   测试集: {test_x.shape}")
    
    # 配置
    hidden = 32
    n_modes = (8, 8)
    n_layers = 3
    n_heads = 4
    epochs = 30  # 减少 epochs
    batch_size = 16  # 减小 batch size
    
    # 使用 reduction='mean' 的 LpLoss
    loss_fn = LpLoss(d=2, p=2, reduction='mean')
    
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
    params_fno = sum(p.numel() for p in model_fno.parameters())
    print(f"参数量: {params_fno:,}")
    
    optimizer = torch.optim.Adam(model_fno.parameters(), lr=1e-3)
    
    print("训练中...")
    start = time.time()
    for epoch in range(epochs):
        model_fno.train()
        perm = torch.randperm(n_train)
        
        epoch_loss = 0.0
        n_batches = 0
        
        for i in range(0, n_train, batch_size):
            bx = train_x[perm[i:i+batch_size]]
            by = train_y[perm[i:i+batch_size]]
            
            optimizer.zero_grad()
            loss = loss_fn(model_fno(bx), by)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        if (epoch + 1) % 10 == 0:
            model_fno.eval()
            with torch.no_grad():
                test_loss = loss_fn(model_fno(test_x), test_y).item()
            print(f"Epoch {epoch+1}: Train={epoch_loss/n_batches:.4f}, Test={test_loss:.4f}")
    
    time_fno = time.time() - start
    model_fno.eval()
    with torch.no_grad():
        loss_fno = loss_fn(model_fno(test_x), test_y).item()
    
    print(f"\n最终 Loss: {loss_fno:.4f}, 时间: {time_fno:.1f}s")
    results['FNO'] = {'params': params_fno, 'loss': loss_fno, 'time': time_fno}
    
    # 清理内存
    del model_fno, optimizer
    gc.collect()
    
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
    params_mhf = sum(p.numel() for p in model_mhf.parameters())
    print(f"参数量: {params_mhf:,}")
    
    optimizer = torch.optim.Adam(model_mhf.parameters(), lr=1e-3)
    
    print("训练中...")
    start = time.time()
    for epoch in range(epochs):
        model_mhf.train()
        perm = torch.randperm(n_train)
        
        epoch_loss = 0.0
        n_batches = 0
        
        for i in range(0, n_train, batch_size):
            bx = train_x[perm[i:i+batch_size]]
            by = train_y[perm[i:i+batch_size]]
            
            optimizer.zero_grad()
            loss = loss_fn(model_mhf(bx), by)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        if (epoch + 1) % 10 == 0:
            model_mhf.eval()
            with torch.no_grad():
                test_loss = loss_fn(model_mhf(test_x), test_y).item()
            print(f"Epoch {epoch+1}: Train={epoch_loss/n_batches:.4f}, Test={test_loss:.4f}")
    
    time_mhf = time.time() - start
    model_mhf.eval()
    with torch.no_grad():
        loss_mhf = loss_fn(model_mhf(test_x), test_y).item()
    
    print(f"\n最终 Loss: {loss_mhf:.4f}, 时间: {time_mhf:.1f}s")
    results['MHF-FNO'] = {'params': params_mhf, 'loss': loss_mhf, 'time': time_mhf}
    
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
    
    print(f"\n📊 MHF-FNO vs FNO:")
    print(f"   参数量变化: {p_change:+.1f}%")
    print(f"   L2误差变化: {l_change:+.1f}%")
    
    if p_change < -50 and l_change < 10:
        print("\n✅ MHF-FNO 集成成功！")
    else:
        print("\n⚠️ 需要进一步调优")
    
    return results


if __name__ == "__main__":
    main()