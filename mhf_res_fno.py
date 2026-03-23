"""
MHF-ResFNO: 新设计的算子

核心创新:
- MHF 处理低频 (全局趋势)
- 标准 Conv 处理高频残差 (局部细节)
- 门控融合 + 残差连接
"""

import torch
import torch.nn as nn
import time
import gc

from neuralop.models import FNO
from neuralop.layers.spectral_convolution import SpectralConv
from neuralop.losses.data_losses import LpLoss


# ============================================
# MHF 核心
# ============================================

class MHFSpectralConv(SpectralConv):
    """MHF SpectralConv"""
    
    def __init__(self, in_channels, out_channels, n_modes, n_heads=4, bias=True, **kwargs):
        if isinstance(n_modes, list):
            n_modes = tuple(n_modes)
        super().__init__(in_channels=in_channels, out_channels=out_channels, n_modes=n_modes, bias=bias, **kwargs)
        
        del self.weight
        self.n_heads = n_heads
        self.use_mhf = (in_channels % n_heads == 0 and out_channels % n_heads == 0)
        
        if self.use_mhf:
            self.head_in = in_channels // n_heads
            self.head_out = out_channels // n_heads
            
            modes_y = n_modes[1] // 2 + 1 if len(n_modes) > 1 else n_modes[0]
            weight_shape = (n_heads, self.head_in, self.head_out, n_modes[0], modes_y)
            
            init_std = (2 / (in_channels + out_channels)) ** 0.5
            self.weight = nn.Parameter(torch.randn(*weight_shape, dtype=torch.cfloat) * init_std)
        else:
            self.weight = nn.Parameter(torch.randn(in_channels, out_channels, n_modes[0], n_modes[1] // 2 + 1, dtype=torch.cfloat) * 0.01)
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels, 1, 1))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x, output_shape=None, *args, **kwargs):
        if not self.use_mhf:
            return self._forward_standard(x)
        return self._forward_2d(x)
    
    def _forward_2d(self, x):
        B, C, H, W = x.shape
        x_freq = torch.fft.rfft2(x, dim=(-2, -1))
        freq_H, freq_W = x_freq.shape[-2], x_freq.shape[-1]
        m_x = min(self.weight.shape[-3], freq_H)
        m_y = min(self.weight.shape[-1], freq_W)
        x_freq = x_freq.view(B, self.n_heads, self.head_in, freq_H, freq_W)
        out_freq = torch.zeros(B, self.n_heads, self.head_out, freq_H, freq_W, dtype=x_freq.dtype, device=x.device)
        out_freq[:, :, :, :m_x, :m_y] = torch.einsum('bhiXY,hioXY->bhoXY', x_freq[:, :, :, :m_x, :m_y], self.weight[:, :, :, :m_x, :m_y])
        out_freq = out_freq.reshape(B, self.out_channels, freq_H, freq_W)
        x_out = torch.fft.irfft2(out_freq, s=(H, W), dim=(-2, -1))
        if self.bias is not None:
            x_out = x_out + self.bias
        return x_out
    
    def _forward_standard(self, x):
        B, C, H, W = x.shape
        x_freq = torch.fft.rfft2(x, dim=(-2, -1))
        freq_H, freq_W = x_freq.shape[-2], x_freq.shape[-1]
        m_x = min(self.weight.shape[-3], freq_H)
        m_y = min(self.weight.shape[-1], freq_W)
        out_freq = torch.zeros(B, self.out_channels, freq_H, freq_W, dtype=x_freq.dtype, device=x.device)
        out_freq[:, :, :m_x, :m_y] = torch.einsum('biXY,ioXY->boXY', x_freq[:, :, :m_x, :m_y], self.weight[:, :, :m_x, :m_y])
        x_out = torch.fft.irfft2(out_freq, s=(H, W), dim=(-2, -1))
        if self.bias is not None:
            x_out = x_out + self.bias
        return x_out


# ============================================
# 新算子: MHF-ResBlock
# ============================================

class MHFResBlock(nn.Module):
    """
    MHF 残差块
    
    创新点:
    1. MHF 处理低频 (全局趋势)
    2. 高频残差路径保留细节
    3. 门控融合
    4. 残差连接
    """
    
    def __init__(self, channels, n_modes, n_heads=4, use_high_freq=True):
        super().__init__()
        
        # 低频路径: MHF
        self.mhf = MHFSpectralConv(channels, channels, n_modes, n_heads)
        
        # 高频路径: 可选
        self.use_high_freq = use_high_freq
        if use_high_freq:
            self.high_freq_conv = nn.Conv2d(channels, channels, 3, padding=1)
        
        # 门控融合
        self.gate = nn.Parameter(torch.ones(1) * 0.5)
        
        # 归一化
        self.norm = nn.GroupNorm(1, channels)
        
        # 激活
        self.act = nn.GELU()
    
    def forward(self, x):
        # 低频路径
        low_freq = self.mhf(x)
        
        # 高频路径
        if self.use_high_freq:
            high_freq = self.high_freq_conv(x)
        else:
            high_freq = 0
        
        # 门控融合 + 残差
        out = self.gate * low_freq + (1 - self.gate) * high_freq
        out = self.norm(out)
        out = self.act(out)
        
        return out + x


class MHFResFNO(nn.Module):
    """
    MHF-ResFNO: 完整模型
    
    结构:
    - Lifting (输入投影)
    - N 个 MHFResBlock
    - Projection (输出投影)
    """
    
    def __init__(self, in_channels, out_channels, hidden_channels, n_modes, n_layers=4, n_heads=4):
        super().__init__()
        
        # Lifting
        self.lifting = nn.Conv2d(in_channels, hidden_channels, 1)
        
        # MHF ResBlocks
        self.blocks = nn.ModuleList([
            MHFResBlock(hidden_channels, n_modes, n_heads)
            for _ in range(n_layers)
        ])
        
        # Projection
        self.projection = nn.Conv2d(hidden_channels, out_channels, 1)
    
    def forward(self, x):
        # Lifting
        x = self.lifting(x)
        
        # MHF ResBlocks
        for block in self.blocks:
            x = block(x)
        
        # Projection
        x = self.projection(x)
        
        return x


# ============================================
# 对比测试
# ============================================

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
    print(" MHF-ResFNO 新算子验证")
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
    
    # 1. 标准 FNO
    print("\n[1/4] 标准 FNO...")
    torch.manual_seed(42)
    model_fno = FNO(n_modes=(8, 8), hidden_channels=32, in_channels=1, out_channels=1, n_layers=4)
    params_fno = count_params(model_fno)
    loss_fno, time_fno = train_and_evaluate(model_fno, train_x, train_y, test_x, test_y, epochs=100)
    print(f"  参数: {params_fno:,}, Loss: {loss_fno:.4f}, 时间: {time_fno:.1f}s")
    results.append({"name": "FNO-基准", "params": params_fno, "loss": loss_fno, "time": time_fno})
    del model_fno
    gc.collect()
    
    # 2. MHF-FNO (替换)
    print("\n[2/4] MHF-FNO (替换)...")
    torch.manual_seed(42)
    model_mhf_fno = FNO(n_modes=(8, 8), hidden_channels=32, in_channels=1, out_channels=1, n_layers=4)
    for i in [0, 3]:  # 边缘层
        model_mhf_fno.fno_blocks.convs[i] = MHFSpectralConv(32, 32, (8, 8), n_heads=4)
    params_mhf_fno = count_params(model_mhf_fno)
    loss_mhf_fno, time_mhf_fno = train_and_evaluate(model_mhf_fno, train_x, train_y, test_x, test_y, epochs=100)
    print(f"  参数: {params_mhf_fno:,}, Loss: {loss_mhf_fno:.4f}, 时间: {time_mhf_fno:.1f}s")
    results.append({"name": "MHF-FNO-替换", "params": params_mhf_fno, "loss": loss_mhf_fno, "time": time_mhf_fno})
    del model_mhf_fno
    gc.collect()
    
    # 3. MHF-ResFNO (新设计)
    print("\n[3/4] MHF-ResFNO (新设计)...")
    torch.manual_seed(42)
    model_mhf_res = MHFResFNO(in_channels=1, out_channels=1, hidden_channels=32, n_modes=(8, 8), n_layers=4, n_heads=4)
    params_mhf_res = count_params(model_mhf_res)
    loss_mhf_res, time_mhf_res = train_and_evaluate(model_mhf_res, train_x, train_y, test_x, test_y, epochs=100)
    print(f"  参数: {params_mhf_res:,}, Loss: {loss_mhf_res:.4f}, 时间: {time_mhf_res:.1f}s")
    results.append({"name": "MHF-ResFNO", "params": params_mhf_res, "loss": loss_mhf_res, "time": time_mhf_res})
    del model_mhf_res
    gc.collect()
    
    # 4. MHF-ResFNO (无高频路径)
    print("\n[4/4] MHF-ResFNO (无高频)...")
    torch.manual_seed(42)
    model_mhf_res_lite = MHFResFNO(in_channels=1, out_channels=1, hidden_channels=32, n_modes=(8, 8), n_layers=4, n_heads=4)
    # 禁用高频路径
    for block in model_mhf_res_lite.blocks:
        block.use_high_freq = False
    params_mhf_res_lite = count_params(model_mhf_res_lite)
    loss_mhf_res_lite, time_mhf_res_lite = train_and_evaluate(model_mhf_res_lite, train_x, train_y, test_x, test_y, epochs=100)
    print(f"  参数: {params_mhf_res_lite:,}, Loss: {loss_mhf_res_lite:.4f}, 时间: {time_mhf_res_lite:.1f}s")
    results.append({"name": "MHF-ResFNO-精简", "params": params_mhf_res_lite, "loss": loss_mhf_res_lite, "time": time_mhf_res_lite})
    del model_mhf_res_lite
    gc.collect()
    
    # 汇总
    print("\n" + "=" * 70)
    print(" 对比结果")
    print("=" * 70)
    print(f"{'配置':<20} {'参数量':<12} {'L2误差':<10} {'改进':<10} {'时间':<10}")
    print("-" * 65)
    baseline = results[0]['loss']
    for r in results:
        improve = (r['loss'] - baseline) / baseline * 100
        print(f"{r['name']:<20} {r['params']:<12,} {r['loss']:<10.4f} {improve:+.1f}% {r['time']:<10.1f}s")
    
    return results


if __name__ == "__main__":
    main()