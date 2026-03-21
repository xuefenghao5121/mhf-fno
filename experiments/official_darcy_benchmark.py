"""
MHF-FNO vs FNO - NeuralOperator 官方 Darcy Flow Benchmark

使用官方数据全面对比：
1. 参数量
2. L2 误差
3. 训练时间
4. 收敛曲线
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import json


# ============================================
# MHF-FNO 2D
# ============================================

class MHFSpectralConv2D(nn.Module):
    """多头频谱卷积"""
    
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
            torch.randn(n_heads, self.head_in, self.head_out, 
                       self.modes_x, self.modes_y, dtype=torch.cfloat) * init_std
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
            'bhiXY,hioXY->bhoXY', 
            x_freq[:, :, :, :m_x, :m_y], 
            self.weight[:, :, :, :m_x, :m_y]
        )
        out_freq = out_freq.reshape(B, self.head_out * self.n_heads, freq_H, freq_W)
        
        x_out = torch.fft.irfft2(out_freq, s=(H, W), dim=(-2, -1))
        return x_out + self.bias


class MHFFNO2D(nn.Module):
    """MHF-FNO 2D"""
    
    def __init__(self, in_channels=1, out_channels=1, hidden=32, n_modes=(8, 8), 
                 n_layers=3, n_heads=4):
        super().__init__()
        
        self.lifting = nn.Conv2d(in_channels, hidden, 1)
        self.convs = nn.ModuleList([
            MHFSpectralConv2D(hidden, hidden, n_modes, n_heads)
            for _ in range(n_layers)
        ])
        self.projection = nn.Conv2d(hidden, out_channels, 1)
    
    def forward(self, x):
        x = self.lifting(x)
        for conv in self.convs:
            x = F.gelu(conv(x))
        return self.projection(x)


# ============================================
# 标准 FNO 2D
# ============================================

class SpectralConv2D(nn.Module):
    """标准频谱卷积"""
    
    def __init__(self, in_channels, out_channels, n_modes):
        super().__init__()
        
        self.modes_x = n_modes[0]
        self.modes_y = n_modes[1] // 2 + 1
        
        self.weight = nn.Parameter(
            torch.randn(in_channels, out_channels, 
                       self.modes_x, self.modes_y, dtype=torch.cfloat) * 0.01
        )
        self.bias = nn.Parameter(torch.zeros(out_channels, 1, 1))
    
    def forward(self, x):
        B, C, H, W = x.shape
        x_freq = torch.fft.rfft2(x, dim=(-2, -1))
        freq_H, freq_W = x_freq.shape[-2], x_freq.shape[-1]
        m_x, m_y = min(self.modes_x, freq_H), min(self.modes_y, freq_W)
        
        out_freq = torch.zeros(B, self.weight.shape[1], freq_H, freq_W,
                              dtype=x_freq.dtype, device=x.device)
        out_freq[:, :, :m_x, :m_y] = torch.einsum(
            'biXY,ioXY->boXY',
            x_freq[:, :, :m_x, :m_y],
            self.weight[:, :, :m_x, :m_y]
        )
        
        x_out = torch.fft.irfft2(out_freq, s=(H, W), dim=(-2, -1))
        return x_out + self.bias


class StandardFNO2D(nn.Module):
    """标准 FNO 2D"""
    
    def __init__(self, in_channels=1, out_channels=1, hidden=32, n_modes=(8, 8), n_layers=3):
        super().__init__()
        
        self.lifting = nn.Conv2d(in_channels, hidden, 1)
        self.convs = nn.ModuleList([
            SpectralConv2D(hidden, hidden, n_modes)
            for _ in range(n_layers)
        ])
        self.projection = nn.Conv2d(hidden, out_channels, 1)
    
    def forward(self, x):
        x = self.lifting(x)
        for conv in self.convs:
            x = F.gelu(conv(x))
        return self.projection(x)


# ============================================
# 测试函数
# ============================================

def count_params(model):
    return sum(p.numel() for p in model.parameters())


def train_and_test(model, train_x, train_y, test_x, test_y, epochs=100, batch_size=32, lr=1e-3):
    """训练并测试"""
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    
    n_train = train_x.shape[0]
    history = {'train_loss': [], 'test_l2': []}
    
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        
        perm = torch.randperm(n_train)
        train_x_perm = train_x[perm]
        train_y_perm = train_y[perm]
        
        epoch_loss = 0.0
        n_batches = 0
        
        for i in range(0, n_train, batch_size):
            bx = train_x_perm[i:i+batch_size]
            by = train_y_perm[i:i+batch_size]
            
            optimizer.zero_grad()
            pred = model(bx)
            loss = F.mse_loss(pred, by)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        history['train_loss'].append(epoch_loss / n_batches)
        
        # 测试
        model.eval()
        with torch.no_grad():
            pred = model(test_x)
            l2 = (torch.norm(pred - test_y) / torch.norm(test_y)).item()
        history['test_l2'].append(l2)
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}: Loss={epoch_loss/n_batches:.6f}, L2={l2:.4f}")
    
    train_time = time.time() - start_time
    
    return history, train_time


def main():
    print("\n" + "=" * 70)
    print(" MHF-FNO vs FNO - NeuralOperator 官方 Darcy Flow Benchmark")
    print("=" * 70)
    
    # 加载官方数据
    print("\n📊 加载官方 Darcy Flow 数据...")
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
    hidden = 32
    n_modes = (8, 8)
    n_layers = 3
    n_heads = 4
    epochs = 100
    
    results = {}
    
    # ===== 测试 FNO =====
    print("\n" + "-" * 50)
    print("🔬 测试标准 FNO")
    print("-" * 50)
    
    model_fno = StandardFNO2D(1, 1, hidden, n_modes, n_layers)
    params_fno = count_params(model_fno)
    print(f"参数量: {params_fno:,}")
    
    history_fno, time_fno = train_and_test(
        model_fno, train_x, train_y, test_x, test_y, epochs
    )
    
    l2_fno = history_fno['test_l2'][-1]
    print(f"\n最终 L2误差: {l2_fno:.4f}")
    print(f"训练时间: {time_fno:.1f}s")
    
    results['FNO'] = {
        'params': params_fno,
        'l2': l2_fno,
        'time': time_fno,
        'history': history_fno
    }
    
    # ===== 测试 MHF-FNO =====
    print("\n" + "-" * 50)
    print("🔬 测试 MHF-FNO")
    print("-" * 50)
    
    model_mhf = MHFFNO2D(1, 1, hidden, n_modes, n_layers, n_heads)
    params_mhf = count_params(model_mhf)
    print(f"参数量: {params_mhf:,}")
    
    history_mhf, time_mhf = train_and_test(
        model_mhf, train_x, train_y, test_x, test_y, epochs
    )
    
    l2_mhf = history_mhf['test_l2'][-1]
    print(f"\n最终 L2误差: {l2_mhf:.4f}")
    print(f"训练时间: {time_mhf:.1f}s")
    
    results['MHF-FNO'] = {
        'params': params_mhf,
        'l2': l2_mhf,
        'time': time_mhf,
        'history': history_mhf
    }
    
    # ===== 汇总 =====
    print("\n" + "=" * 70)
    print(" 📈 汇总对比")
    print("=" * 70)
    
    print(f"\n{'模型':<15} {'参数量':<12} {'L2误差':<10} {'训练时间':<10}")
    print("-" * 50)
    
    for name, res in results.items():
        print(f"{name:<15} {res['params']:<12,} {res['l2']:<10.4f} {res['time']:<10.1f}s")
    
    # 改进
    params_change = (results['MHF-FNO']['params'] - results['FNO']['params']) / results['FNO']['params'] * 100
    l2_change = (results['MHF-FNO']['l2'] - results['FNO']['l2']) / results['FNO']['l2'] * 100
    time_change = (results['MHF-FNO']['time'] - results['FNO']['time']) / results['FNO']['time'] * 100
    
    print(f"\n📊 MHF-FNO vs FNO:")
    print(f"   参数量: {params_change:+.1f}%")
    print(f"   L2误差: {l2_change:+.2f}%")
    print(f"   训练时间: {time_change:+.1f}%")
    
    # 判断
    print("\n" + "=" * 70)
    passed = 0
    if params_change < -50:
        print(f" ✅ 参数效率大幅提升 ({-params_change:.1f}% 减少)")
        passed += 1
    if l2_change < 0:
        print(f" ✅ 精度提升 ({-l2_change:.1f}% 改进)")
        passed += 1
    elif l2_change < 5:
        print(f" ✅ 精度保持 (误差增加 {l2_change:.1f}%)")
        passed += 1
    if time_change < 10:
        print(f" ✅ 训练效率保持")
        passed += 1
    
    print(f"\n 通过 {passed}/3 项指标")
    
    # 保存结果
    save_results = {
        'FNO': {'params': params_fno, 'l2': l2_fno, 'time': time_fno},
        'MHF-FNO': {'params': params_mhf, 'l2': l2_mhf, 'time': time_mhf}
    }
    
    with open('benchmark_results.json', 'w') as f:
        json.dump(save_results, f, indent=2)
    
    print(f"\n结果已保存到 benchmark_results.json")
    
    return results


if __name__ == "__main__":
    main()