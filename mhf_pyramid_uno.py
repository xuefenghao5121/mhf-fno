"""
MHF-PyramidUNO: 针对 UNO 的 MHF 新设计

核心创新:
1. 编码器: 多头 MHF (捕获全局特征)
2. 瓶颈: 标准 SpectralConv (保持精度)
3. 解码器: 自适应 MHF (保留细节)
"""

import torch
import torch.nn as nn
from neuralop.layers.spectral_convolution import SpectralConv
from neuralop.losses.data_losses import LpLoss


class AdaptiveMHFConv(nn.Module):
    """
    自适应 MHF 卷积
    
    特点:
    - 自动适配不规则的 n_modes
    - 支持 in_channels != out_channels
    """
    
    def __init__(self, in_channels, out_channels, n_modes, n_heads=None):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes = n_modes if isinstance(n_modes, tuple) else tuple(n_modes)
        
        # 自动确定 n_heads
        if n_heads is None:
            # 找到最大公约数作为 head 数
            import math
            gcd = math.gcd(in_channels, out_channels)
            self.n_heads = min(gcd, 8)  # 最多 8 头
        else:
            self.n_heads = n_heads
        
        self.use_mhf = (in_channels % self.n_heads == 0 and out_channels % self.n_heads == 0)
        
        if self.use_mhf:
            self.head_in = in_channels // self.n_heads
            self.head_out = out_channels // self.n_heads
            
            # 适配 n_modes (rfft 后的第二维)
            modes_x = self.n_modes[0]
            modes_y = self.n_modes[1] if len(self.n_modes) > 1 else self.n_modes[0]
            
            weight_shape = (self.n_heads, self.head_in, self.head_out, modes_x, modes_y)
            init_std = (2 / (in_channels + out_channels)) ** 0.5
            self.weight = nn.Parameter(torch.randn(*weight_shape, dtype=torch.cfloat) * init_std)
            self.use_mhf = True
        else:
            # 回退到标准 SpectralConv
            modes_x = self.n_modes[0]
            modes_y = self.n_modes[1] if len(self.n_modes) > 1 else self.n_modes[0]
            self.weight = nn.Parameter(
                torch.randn(in_channels, out_channels, modes_x, modes_y, dtype=torch.cfloat) * 0.01
            )
            self.use_mhf = False
        
        self.bias = nn.Parameter(torch.zeros(out_channels, 1, 1))
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # FFT
        x_freq = torch.fft.rfft2(x, dim=(-2, -1))
        freq_H, freq_W = x_freq.shape[-2], x_freq.shape[-1]
        
        m_x = min(self.weight.shape[-3] if self.weight.dim() > 3 else self.weight.shape[-2], freq_H)
        m_y = min(self.weight.shape[-1], freq_W)
        
        if self.use_mhf:
            # MHF 路径
            x_freq = x_freq.view(B, self.n_heads, self.head_in, freq_H, freq_W)
            out_freq = torch.zeros(B, self.n_heads, self.head_out, freq_H, freq_W, 
                                   dtype=x_freq.dtype, device=x.device)
            out_freq[:, :, :, :m_x, :m_y] = torch.einsum(
                'bhiXY,hioXY->bhoXY', 
                x_freq[:, :, :, :m_x, :m_y], 
                self.weight[:, :, :, :m_x, :m_y]
            )
            out_freq = out_freq.reshape(B, self.out_channels, freq_H, freq_W)
        else:
            # 标准路径
            out_freq = torch.zeros(B, self.out_channels, freq_H, freq_W, 
                                   dtype=x_freq.dtype, device=x.device)
            out_freq[:, :, :m_x, :m_y] = torch.einsum(
                'biXY,ioXY->boXY', 
                x_freq[:, :, :m_x, :m_y], 
                self.weight[:, :, :m_x, :m_y]
            )
        
        # IFFT
        x_out = torch.fft.irfft2(out_freq, s=(H, W), dim=(-2, -1))
        x_out = x_out + self.bias
        
        return x_out


class MHFPyramidBlock(nn.Module):
    """
    MHF 金字塔块
    
    根据阶段自动选择配置:
    - 编码器: n_heads=8 (全局特征)
    - 瓶颈: 标准 SpectralConv (精度)
    - 解码器: n_heads=4 (细节保留)
    """
    
    def __init__(self, in_channels, out_channels, n_modes, stage='encoder'):
        super().__init__()
        
        self.stage = stage
        
        if stage == 'encoder':
            # 编码器: 多头 MHF
            self.conv = AdaptiveMHFConv(in_channels, out_channels, n_modes, n_heads=8)
        elif stage == 'bottleneck':
            # 瓶颈: 标准 SpectralConv
            self.conv = SpectralConv(in_channels, out_channels, n_modes)
        else:  # decoder
            # 解码器: 少头 MHF
            self.conv = AdaptiveMHFConv(in_channels, out_channels, n_modes, n_heads=4)
        
        self.act = nn.GELU()
    
    def forward(self, x):
        return self.act(self.conv(x))


class MHFPyramidUNO(nn.Module):
    """
    MHF-PyramidUNO
    
    结构:
    - Lifting: 输入投影
    - 编码器: 多头 MHF (全局)
    - 瓶颈: 标准 SpectralConv (精度)
    - 解码器: 少头 MHF (细节)
    - Projection: 输出投影
    """
    
    def __init__(self, in_channels=1, out_channels=1, hidden_channels=32, 
                 n_modes=(8, 5), n_layers=5):
        super().__init__()
        
        self.n_layers = n_layers
        
        # Lifting
        self.lifting = nn.Conv2d(in_channels, hidden_channels, 1)
        
        # 构建 UNO 结构
        # 简化版: 不使用 skip connections
        self.blocks = nn.ModuleList()
        
        # 编码器 (前 n_layers//2 层)
        encoder_layers = n_layers // 2
        for i in range(encoder_layers):
            if i == 0:
                in_ch = hidden_channels
                out_ch = hidden_channels * 2
            else:
                in_ch = hidden_channels * (2 ** i)
                out_ch = hidden_channels * (2 ** (i+1))
            
            self.blocks.append(MHFPyramidBlock(in_ch, out_ch, n_modes, stage='encoder'))
        
        # 瓶颈
        bottleneck_ch = hidden_channels * (2 ** encoder_layers)
        self.blocks.append(MHFPyramidBlock(bottleneck_ch, bottleneck_ch, n_modes, stage='bottleneck'))
        
        # 解码器
        decoder_layers = n_layers - encoder_layers - 1
        for i in range(decoder_layers):
            in_ch = bottleneck_ch // (2 ** i)
            out_ch = bottleneck_ch // (2 ** (i+1))
            self.blocks.append(MHFPyramidBlock(in_ch, out_ch, n_modes, stage='decoder'))
        
        # Projection
        self.projection = nn.Conv2d(hidden_channels, out_channels, 1)
    
    def forward(self, x):
        x = self.lifting(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.projection(x)
        
        return x


# ============================================
# 测试代码
# ============================================

def count_params(model):
    return sum(p.numel() for p in model.parameters())


def train_and_evaluate(model, train_x, train_y, test_x, test_y, epochs=100, batch_size=32, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = LpLoss(d=2, p=2, reduction='mean')
    
    n_train = train_x.shape[0]
    import time
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
    print(" MHF-PyramidUNO 新算子测试")
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
    model_mhf = MHFPyramidUNO(in_channels=1, out_channels=1, hidden_channels=32, n_modes=(8, 5), n_layers=5)
    params_mhf = count_params(model_mhf)
    print(f"  参数: {params_mhf:,}")
    loss_mhf, time_mhf = train_and_evaluate(model_mhf, train_x, train_y, test_x, test_y, epochs=100)
    print(f"  Loss: {loss_mhf:.4f}, 时间: {time_mhf:.1f}s")
    results.append({"name": "MHF-PyramidUNO", "params": params_mhf, "loss": loss_mhf, "time": time_mhf})
    
    # 2. 对比: 标准 UNO (简化版)
    print("\n[2/2] 标准 FNO (对比)...")
    from neuralop.models import FNO
    torch.manual_seed(42)
    model_fno = FNO(n_modes=(8, 8), hidden_channels=32, in_channels=1, out_channels=1, n_layers=3)
    params_fno = count_params(model_fno)
    loss_fno, time_fno = train_and_evaluate(model_fno, train_x, train_y, test_x, test_y, epochs=100)
    print(f"  参数: {params_fno:,}, Loss: {loss_fno:.4f}, 时间: {time_fno:.1f}s")
    results.append({"name": "FNO-基准", "params": params_fno, "loss": loss_fno, "time": time_fno})
    
    # 汇总
    print("\n" + "=" * 70)
    print(" 对比结果")
    print("=" * 70)
    print(f"{'配置':<20} {'参数量':<12} {'L2误差':<10} {'改进':<10}")
    print("-" * 55)
    baseline = results[1]['loss']  # FNO 作为基准
    for r in results:
        improve = (r['loss'] - baseline) / baseline * 100
        print(f"{r['name']:<20} {r['params']:<12,} {r['loss']:<10.4f} {improve:+.1f}%")
    
    return results


if __name__ == "__main__":
    main()