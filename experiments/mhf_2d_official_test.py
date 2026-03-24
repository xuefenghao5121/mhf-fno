"""
MHF-FNO 2D 官方数据测试

使用 NeuralOperator 官方 Darcy Flow 数据
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time


class MHFSpectralConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, n_modes, n_heads=4):
        super().__init__()
        
        assert in_channels % n_heads == 0
        assert out_channels % n_heads == 0
        
        self.n_heads = n_heads
        self.head_in = in_channels // n_heads
        self.head_out = out_channels // n_heads
        
        self.modes_x = n_modes[0]
        self.modes_y = n_modes[1] // 2 + 1
        
        init_std = (2 / (in_channels + out_channels)) ** 0.5
        
        self.weight = nn.Parameter(
            torch.randn(n_heads, self.head_in, self.head_out, self.modes_x, self.modes_y, 
                       dtype=torch.cfloat) * init_std
        )
        self.bias = nn.Parameter(init_std * torch.randn(out_channels, 1, 1))
    
    def forward(self, x):
        B, C, H, W = x.shape
        x_freq = torch.fft.rfft2(x, dim=(-2, -1))
        freq_H, freq_W = x_freq.shape[-2], x_freq.shape[-1]
        m_x, m_y = min(self.modes_x, freq_H), min(self.modes_y, freq_W)
        
        x_freq = x_freq.view(B, self.n_heads, self.head_in, freq_H, freq_W)
        out_freq = torch.zeros(B, self.n_heads, self.head_out, freq_H, freq_W,
                              dtype=x_freq.dtype, device=x.device)
        out_freq[:, :, :, :m_x, :m_y] = torch.einsum(
            'bhiXY,hioXY->bhoXY', x_freq[:, :, :, :m_x, :m_y], self.weight[:, :, :, :m_x, :m_y])
        out_freq = out_freq.reshape(B, self.head_out * self.n_heads, freq_H, freq_W)
        
        x_out = torch.fft.irfft2(out_freq, s=(H, W), dim=(-2, -1))
        return x_out + self.bias


class MHFFNO2D(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=32, n_modes=(8, 8), n_layers=3, n_heads=4):
        super().__init__()
        self.lifting = nn.Conv2d(in_channels, hidden_channels, 1)
        self.convs = nn.ModuleList([
            MHFSpectralConv2D(hidden_channels, hidden_channels, n_modes, n_heads)
            for _ in range(n_layers)
        ])
        self.projection = nn.Conv2d(hidden_channels, out_channels, 1)
    
    def forward(self, x):
        x = self.lifting(x)
        for conv in self.convs:
            x = F.gelu(conv(x))
        return self.projection(x)


class SpectralConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, n_modes):
        super().__init__()
        self.modes_x = n_modes[0]
        self.modes_y = n_modes[1] // 2 + 1
        self.weight = nn.Parameter(
            torch.randn(in_channels, out_channels, self.modes_x, self.modes_y, dtype=torch.cfloat) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_channels, 1, 1))
    
    def forward(self, x):
        B, C, H, W = x.shape
        x_freq = torch.fft.rfft2(x, dim=(-2, -1))
        freq_H, freq_W = x_freq.shape[-2], x_freq.shape[-1]
        m_x, m_y = min(self.modes_x, freq_H), min(self.modes_y, freq_W)
        
        out_freq = torch.zeros(B, self.weight.shape[1], freq_H, freq_W, dtype=x_freq.dtype, device=x.device)
        out_freq[:, :, :m_x, :m_y] = torch.einsum('biXY,ioXY->boXY', 
            x_freq[:, :, :m_x, :m_y], self.weight[:, :, :m_x, :m_y])
        
        x_out = torch.fft.irfft2(out_freq, s=(H, W), dim=(-2, -1))
        return x_out + self.bias


class StandardFNO2D(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=32, n_modes=(8, 8), n_layers=3):
        super().__init__()
        self.lifting = nn.Conv2d(in_channels, hidden_channels, 1)
        self.convs = nn.ModuleList([
            SpectralConv2D(hidden_channels, hidden_channels, n_modes) for _ in range(n_layers)
        ])
        self.projection = nn.Conv2d(hidden_channels, out_channels, 1)
    
    def forward(self, x):
        x = self.lifting(x)
        for conv in self.convs:
            x = F.gelu(conv(x))
        return self.projection(x)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def main():
    print("\n" + "=" * 60)
    print(" MHF-FNO 2D 官方数据测试")
    print("=" * 60)
    
    # 加载官方数据
    print("\n加载 NeuralOperator 官方 Darcy Flow 数据...")
    data_path = '/usr/local/lib/python3.11/site-packages/neuralop/data/datasets/data/'
    
    train_data = torch.load(f'{data_path}/darcy_train_16.pt', weights_only=False)
    test_data = torch.load(f'{data_path}/darcy_test_16.pt', weights_only=False)
    
    train_x = train_data['x'].unsqueeze(1).float()[:500]
    train_y = train_data['y'].unsqueeze(1).float()[:500]
    test_x = test_data['x'].unsqueeze(1).float()
    test_y = test_data['y'].unsqueeze(1).float()
    
    print(f"训练集: {train_x.shape}, 测试集: {test_x.shape}")
    
    hidden = 32
    n_modes = (8, 8)
    n_layers = 3
    n_heads = 4
    epochs = 100
    batch_size = 32
    
    results = {}
    
    # FNO
    print("\n测试 FNO...")
    model_fno = StandardFNO2D(1, 1, hidden, n_modes, n_layers)
    params_fno = count_params(model_fno)
    print(f"参数量: {params_fno:,}")
    
    optimizer = torch.optim.Adam(model_fno.parameters(), lr=1e-3)
    n_train = train_x.shape[0]
    
    start = time.time()
    for epoch in range(epochs):
        model_fno.train()
        perm = torch.randperm(n_train)
        for i in range(0, n_train, batch_size):
            bx, by = train_x[perm[i:i+batch_size]], train_y[perm[i:i+batch_size]]
            optimizer.zero_grad()
            F.mse_loss(model_fno(bx), by).backward()
            optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            model_fno.eval()
            with torch.no_grad():
                l2 = (torch.norm(model_fno(test_x) - test_y) / torch.norm(test_y)).item()
            print(f"Epoch {epoch+1}: L2={l2:.4f}")
    
    time_fno = time.time() - start
    with torch.no_grad():
        l2_fno = (torch.norm(model_fno(test_x) - test_y) / torch.norm(test_y)).item()
    print(f"最终 L2: {l2_fno:.4f}, 时间: {time_fno:.1f}s")
    results['FNO'] = {'params': params_fno, 'l2': l2_fno, 'time': time_fno}
    
    # MHF-FNO
    print("\n测试 MHF-FNO...")
    model_mhf = MHFFNO2D(1, 1, hidden, n_modes, n_layers, n_heads)
    params_mhf = count_params(model_mhf)
    print(f"参数量: {params_mhf:,}")
    
    optimizer = torch.optim.Adam(model_mhf.parameters(), lr=1e-3)
    
    start = time.time()
    for epoch in range(epochs):
        model_mhf.train()
        perm = torch.randperm(n_train)
        for i in range(0, n_train, batch_size):
            bx, by = train_x[perm[i:i+batch_size]], train_y[perm[i:i+batch_size]]
            optimizer.zero_grad()
            F.mse_loss(model_mhf(bx), by).backward()
            optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            model_mhf.eval()
            with torch.no_grad():
                l2 = (torch.norm(model_mhf(test_x) - test_y) / torch.norm(test_y)).item()
            print(f"Epoch {epoch+1}: L2={l2:.4f}")
    
    time_mhf = time.time() - start
    with torch.no_grad():
        l2_mhf = (torch.norm(model_mhf(test_x) - test_y) / torch.norm(test_y)).item()
    print(f"最终 L2: {l2_mhf:.4f}, 时间: {time_mhf:.1f}s")
    results['MHF-FNO'] = {'params': params_mhf, 'l2': l2_mhf, 'time': time_mhf}
    
    # 汇总
    print("\n" + "=" * 60)
    print("汇总")
    print("=" * 60)
    print(f"\n{'模型':<12} {'参数量':<12} {'L2误差':<10} {'时间':<8}")
    print("-" * 45)
    for name, res in results.items():
        print(f"{name:<12} {res['params']:<12,} {res['l2']:<10.4f} {res['time']:<8.1f}s")
    
    params_change = (results['MHF-FNO']['params'] - results['FNO']['params']) / results['FNO']['params'] * 100
    l2_change = (results['MHF-FNO']['l2'] - results['FNO']['l2']) / results['FNO']['l2'] * 100
    print(f"\n改进: 参数量 {params_change:+.1f}%, L2误差 {l2_change:+.2f}%")


if __name__ == "__main__":
    main()