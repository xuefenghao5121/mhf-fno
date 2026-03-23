"""
MHF-PyramidUNO: 完整实现

参数对比:
| Block | 标准 UNO | MHF-PyramidUNO | 减少 |
|-------|----------|----------------|------|
| 0 (编码) | 40,960 | 10,272 | -75% |
| 1 (编码) | 81,920 | 20,544 | -75% |
| 2 (瓶颈) | 163,840 | 163,840 | 0% (标准) |
| 3 (解码) | 327,680 | 81,984 | -75% |
| 4 (解码) | 122,880 | 30,752 | -75% |

总参数: 812,193 → 307,392 (-62%)
"""

import torch
import torch.nn as nn
from neuralop.layers.spectral_convolution import SpectralConv
from neuralop.losses.data_losses import LpLoss
from neuralop.models import UNO
import time


class AdaptiveMHFConv(nn.Module):
    """自适应 MHF 卷积"""
    
    def __init__(self, in_channels, out_channels, n_modes, n_heads=4):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        if isinstance(n_modes, list):
            n_modes = tuple(n_modes)
        self.n_modes = n_modes
        
        modes_x = n_modes[0]
        modes_y = n_modes[1] if len(n_modes) > 1 else n_modes[0]
        self.modes_x = modes_x
        self.modes_y = modes_y
        
        self.n_heads = n_heads
        self.use_mhf = (in_channels % n_heads == 0 and out_channels % n_heads == 0)
        
        if self.use_mhf:
            self.head_in = in_channels // n_heads
            self.head_out = out_channels // n_heads
            
            weight_shape = (n_heads, self.head_in, self.head_out, modes_x, modes_y)
            init_std = (2 / (in_channels + out_channels)) ** 0.5
            self.weight = nn.Parameter(torch.randn(*weight_shape, dtype=torch.cfloat) * init_std)
        else:
            self.weight = nn.Parameter(
                torch.randn(in_channels, out_channels, modes_x, modes_y, dtype=torch.cfloat) * 0.01
            )
        
        self.bias = nn.Parameter(torch.zeros(out_channels, 1, 1))
    
    def forward(self, x):
        B, C, H, W = x.shape
        x_freq = torch.fft.rfft2(x, dim=(-2, -1))
        freq_H, freq_W = x_freq.shape[-2], x_freq.shape[-1]
        
        m_x = min(self.modes_x, freq_H)
        m_y = min(self.modes_y, freq_W)
        
        if self.use_mhf:
            x_heads = x_freq.view(B, self.n_heads, self.head_in, freq_H, freq_W)
            out_heads = torch.zeros(B, self.n_heads, self.head_out, freq_H, freq_W,
                                    dtype=x_freq.dtype, device=x_freq.device)
            
            for h in range(self.n_heads):
                for mx in range(m_x):
                    for my in range(m_y):
                        out_heads[:, h, :, mx, my] = torch.einsum(
                            'bi,io->bo',
                            x_heads[:, h, :, mx, my],
                            self.weight[h, :, :, mx, my]
                        )
            
            out_freq = out_heads.reshape(B, self.out_channels, freq_H, freq_W)
        else:
            out_freq = torch.zeros(B, self.out_channels, freq_H, freq_W,
                                   dtype=x_freq.dtype, device=x_freq.device)
            for mx in range(m_x):
                for my in range(m_y):
                    out_freq[:, :, mx, my] = torch.einsum(
                        'bi,io->bo',
                        x_freq[:, :, mx, my],
                        self.weight[:, :, mx, my]
                    )
        
        x_out = torch.fft.irfft2(out_freq, s=(H, W), dim=(-2, -1))
        return x_out + self.bias


class MHFPyramidUNO(nn.Module):
    """
    MHF-PyramidUNO
    
    设计策略:
    - 编码器 (Block 0,1): MHF n_heads=8 (全局特征)
    - 瓶颈 (Block 2): 标准 SpectralConv (精度优先)
    - 解码器 (Block 3,4): MHF n_heads=4 (细节保留)
    """
    
    def __init__(self, in_channels=1, out_channels=1, hidden_channels=32, n_layers=5):
        super().__init__()
        
        self.n_layers = n_layers
        
        # Lifting
        self.lifting = nn.Conv2d(in_channels, hidden_channels, 1)
        
        # UNO 配置
        uno_out = [32, 64, 64, 64, 32]
        uno_modes = [(8, 5)] * 5
        
        # 构建 blocks
        self.blocks = nn.ModuleList()
        
        for i in range(n_layers):
            if i == 0:
                in_ch, out_ch = hidden_channels, uno_out[0]
            else:
                in_ch = uno_out[i-1] if i > 0 else hidden_channels
                out_ch = uno_out[i]
            
            # 根据位置选择策略
            if i < 2:  # 编码器
                stage = 'encoder'
                n_heads = 8
            elif i == 2:  # 瓶颈
                stage = 'bottleneck'
                n_heads = 1  # 标准
            else:  # 解码器
                stage = 'decoder'
                n_heads = 4
            
            if stage == 'bottleneck':
                # 标准 SpectralConv
                conv = SpectralConv(in_ch, out_ch, uno_modes[i])
            else:
                # MHF
                conv = AdaptiveMHFConv(in_ch, out_ch, uno_modes[i], n_heads)
            
            self.blocks.append(conv)
        
        # Projection
        self.projection = nn.Conv2d(uno_out[-1], out_channels, 1)
    
    def forward(self, x):
        x = self.lifting(x)
        
        for conv in self.blocks:
            x = conv(x)
        
        x = self.projection(x)
        return x


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def train_and_evaluate(model, train_x, train_y, test_x, test_y, epochs=100, batch_size=32, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
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
    
    return best_loss, time.time() - start


def main():
    print("\n" + "=" * 70)
    print(" MHF-PyramidUNO vs 标准 UNO 对比测试")
    print("=" * 70)
    
    # 加载数据
    data_path = '/usr/local/lib/python3.11/site-packages/neuralop/data/datasets/data/'
    train_data = torch.load(f'{data_path}/darcy_train_16.pt', weights_only=False)
    test_data = torch.load(f'{data_path}/darcy_test_16.pt', weights_only=False)
    
    train_x = train_data['x'].unsqueeze(1).float()
    train_y = train_data['y'].unsqueeze(1).float()
    test_x = test_data['x'].unsqueeze(1).float()
    test_y = test_data['y'].unsqueeze(1).float()
    
    print(f"数据: {train_x.shape}")
    
    results = []
    
    # 1. MHF-PyramidUNO
    print("\n[1/2] MHF-PyramidUNO...")
    torch.manual_seed(42)
    model_mhf = MHFPyramidUNO(in_channels=1, out_channels=1, hidden_channels=32, n_layers=5)
    params_mhf = count_params(model_mhf)
    print(f"  参数: {params_mhf:,}")
    loss_mhf, time_mhf = train_and_evaluate(model_mhf, train_x, train_y, test_x, test_y, epochs=100)
    print(f"  Loss: {loss_mhf:.4f}, 时间: {time_mhf:.1f}s")
    results.append({"name": "MHF-PyramidUNO", "params": params_mhf, "loss": loss_mhf, "time": time_mhf})
    
    # 2. 标准 UNO
    print("\n[2/2] 标准 UNO...")
    torch.manual_seed(42)
    model_uno = UNO(
        1, 1, 32,
        uno_out_channels=[32, 64, 64, 64, 32],
        uno_n_modes=[[8, 8], [8, 8], [8, 8], [8, 8], [8, 8]],
        uno_scalings=[[1.0, 1.0], [0.5, 0.5], [1, 1], [1, 1], [2, 2]],
        n_layers=5,
        channel_mlp_skip='linear',
    )
    params_uno = count_params(model_uno)
    print(f"  参数: {params_uno:,}")
    loss_uno, time_uno = train_and_evaluate(model_uno, train_x, train_y, test_x, test_y, epochs=100)
    print(f"  Loss: {loss_uno:.4f}, 时间: {time_uno:.1f}s")
    results.append({"name": "UNO-基准", "params": params_uno, "loss": loss_uno, "time": time_uno})
    
    # 汇总
    print("\n" + "=" * 70)
    print(" 对比结果")
    print("=" * 70)
    print(f"{'配置':<20} {'参数量':<12} {'参数减少':<10} {'L2误差':<10} {'精度改进':<10}")
    print("-" * 65)
    baseline_params = results[1]['params']
    baseline_loss = results[1]['loss']
    for r in results:
        params_change = (r['params'] - baseline_params) / baseline_params * 100
        loss_change = (r['loss'] - baseline_loss) / baseline_loss * 100
        print(f"{r['name']:<20} {r['params']:<12,} {params_change:+.1f}% {r['loss']:<10.4f} {loss_change:+.1f}%")
    
    return results


if __name__ == "__main__":
    main()