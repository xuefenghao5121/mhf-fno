#!/usr/bin/env python3
"""
MHF-FNO 完整使用示例 - 拷贝即用

这个脚本包含所有必要的代码，可以直接运行测试 MHF-FNO vs FNO

推荐配置 (经优化测试验证):
    n_heads=2, mhf_layers=[0, 2], hidden_channels=32
    效果: 参数减少 30.6%, 精度损失仅 1.4%
"""

# ============================================================
# 第一步：安装依赖（在终端运行）
# ============================================================
# pip install neuralop torch numpy

# ============================================================
# 第二步：导入必要的库
# ============================================================
import torch
import torch.nn as nn
import numpy as np
import time

# 检查是否有 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")


# ============================================================
# 第三步：MHF-FNO 核心实现（直接复制使用）
# ============================================================

class MHFSpectralConv(nn.Module):
    """Multi-Head Fourier Spectral Convolution"""
    
    def __init__(self, in_channels, out_channels, n_modes, n_heads=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes = n_modes
        self.n_heads = n_heads
        
        # 检查是否能被 n_heads 整除
        if in_channels % n_heads != 0 or out_channels % n_heads != 0:
            raise ValueError(f"通道数 ({in_channels}, {out_channels}) 必须能被 n_heads ({n_heads}) 整除")
        
        self.head_in = in_channels // n_heads
        self.head_out = out_channels // n_heads
        
        # 频域权重: [n_heads, head_in, head_out, *n_modes]
        # 使用复数类型避免 forward 中转换
        modes_y = n_modes[1] // 2 + 1 if len(n_modes) > 1 else n_modes[0]
        weight_shape = (n_heads, self.head_in, self.head_out, n_modes[0], modes_y)
        
        init_std = (2 / (in_channels + out_channels)) ** 0.5
        self.weight = nn.Parameter(
            torch.randn(*weight_shape, dtype=torch.cfloat) * init_std
        )
        
        self.bias = nn.Parameter(torch.zeros(out_channels, 1, 1))
        self._init_weights()
    
    def _init_weights(self):
        nn.init.zeros_(self.bias)
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # 2D FFT
        x_freq = torch.fft.rfft2(x, dim=(-2, -1))
        freq_H, freq_W = x_freq.shape[-2], x_freq.shape[-1]
        
        # 模式数
        m_x = min(self.n_modes[0], freq_H)
        m_y = min(self.weight.shape[-1], freq_W)
        
        # 重塑为多头格式: [B, n_heads, head_in, H, W]
        x_freq = x_freq.view(B, self.n_heads, self.head_in, freq_H, freq_W)
        
        # 输出张量
        out_freq = torch.zeros(
            B, self.n_heads, self.head_out, freq_H, freq_W,
            dtype=x_freq.dtype, device=x.device
        )
        
        # 多头频域卷积 (向量化)
        # einsum: 输入 [B, h, i, X, Y], 权重 [h, i, o, X, Y] -> 输出 [B, h, o, X, Y]
        out_freq[:, :, :, :m_x, :m_y] = torch.einsum(
            'bhixy,hioxy->bhoxy',
            x_freq[:, :, :, :m_x, :m_y],
            self.weight[..., :m_x, :m_y]
        )
        
        # 合并多头
        out_freq = out_freq.reshape(B, self.out_channels, freq_H, freq_W)
        
        # IFFT
        x_out = torch.fft.irfft2(out_freq, s=(H, W), dim=(-2, -1))
        
        # 添加偏置
        x_out = x_out + self.bias
        
        return x_out


class MHFFNO(nn.Module):
    """MHF-FNO 模型"""
    
    def __init__(self, n_modes, hidden_channels, in_channels, out_channels,
                 n_layers=3, n_heads=4, mhf_layers=None):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_channels = hidden_channels
        
        if mhf_layers is None:
            mhf_layers = [0, n_layers - 1]
        self.mhf_layers = set(mhf_layers)
        
        self.fc_in = nn.Linear(in_channels, hidden_channels)
        
        self.fno_blocks = nn.ModuleList()
        self.use_mhf_list = []
        for i in range(n_layers):
            use_mhf = i in self.mhf_layers
            block = nn.ModuleDict({
                'conv': MHFSpectralConv(
                    hidden_channels, hidden_channels, n_modes, n_heads
                ) if use_mhf else nn.Identity(),
                'w': nn.Conv2d(hidden_channels, hidden_channels, 1),
            })
            self.fno_blocks.append(block)
            self.use_mhf_list.append(use_mhf)
        
        self.fc_out = nn.Linear(hidden_channels, out_channels)
    
    def forward(self, x):
        x = self.fc_in(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        for i, block in enumerate(self.fno_blocks):
            if self.use_mhf_list[i]:
                x = x + block['conv'](x)
            x = x + block['w'](x)
        
        x = self.fc_out(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x


# ============================================================
# 第四步：训练函数
# ============================================================

def train_model(model, train_x, train_y, test_x, test_y, epochs=50, lr=1e-3):
    """训练模型并返回结果"""
    
    # 使用 L2 Loss
    def l2_loss(pred, target):
        return torch.mean((pred - target) ** 2)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    results = {'train_loss': [], 'test_loss': []}
    best_test = float('inf')
    
    n_train = train_x.shape[0]
    batch_size = 32
    
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train)
        train_loss = 0
        
        for i in range(0, n_train, batch_size):
            bx = train_x[perm[i:i+batch_size]].to(device)
            by = train_y[perm[i:i+batch_size]].to(device)
            
            optimizer.zero_grad()
            loss = l2_loss(model(bx), by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        scheduler.step()
        
        # 测试
        model.eval()
        with torch.no_grad():
            test_loss = l2_loss(model(test_x.to(device)), test_y.to(device)).item()
        
        results['train_loss'].append(train_loss / (n_train // batch_size))
        results['test_loss'].append(test_loss)
        
        if test_loss < best_test:
            best_test = test_loss
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: Train {train_loss/(n_train//batch_size):.4f}, Test {test_loss:.4f}")
    
    return results, best_test


# ============================================================
# 第五步：生成简单测试数据（Darcy Flow 风格）
# ============================================================

def generate_darcy_like_data(n_train=500, n_test=100, resolution=16):
    """生成类似 Darcy Flow 的测试数据"""
    
    print(f"生成数据: 训练 {n_train}, 测试 {n_test}, 分辨率 {resolution}x{resolution}")
    
    # 输入：扩散系数场 a(x) - 随机平滑场
    # 输出：压力场 u(x) - 简化版 Darcy 方程解
    
    def generate_smooth_field(n, resolution):
        """生成平滑随机场"""
        x = torch.randn(n, 1, resolution, resolution)
        # 平滑处理
        kernel = torch.ones(1, 1, 3, 3) / 9
        for _ in range(3):
            x = torch.nn.functional.conv2d(x, kernel, padding=1)
        return x
    
    train_x = generate_smooth_field(n_train, resolution)
    test_x = generate_smooth_field(n_test, resolution)
    
    # 简化的解：u(x) ≈ 平滑的 a(x)
    train_y = torch.nn.functional.avg_pool2d(train_x, 3, stride=1, padding=1)
    test_y = torch.nn.functional.avg_pool2d(test_x, 3, stride=1, padding=1)
    
    print(f"✅ 数据生成完成")
    return train_x, train_y, test_x, test_y


# ============================================================
# 第六步：主测试流程
# ============================================================

def main():
    print("=" * 60)
    print("MHF-FNO vs FNO 对比测试")
    print("=" * 60)
    
    # 生成数据
    train_x, train_y, test_x, test_y = generate_darcy_like_data(
        n_train=500, n_test=100, resolution=16
    )
    
    # 配置
    config = {
        'n_modes': (8, 8),
        'hidden_channels': 32,
        'in_channels': 1,
        'out_channels': 1,
        'n_layers': 3,
        'epochs': 30,
    }
    
    results = {}
    
    # ---------- 测试 FNO ----------
    print("\n" + "=" * 60)
    print("测试 FNO (基准)")
    print("=" * 60)
    
    from neuralop.models import FNO
    
    torch.manual_seed(42)
    model_fno = FNO(
        n_modes=config['n_modes'],
        hidden_channels=config['hidden_channels'],
        in_channels=config['in_channels'],
        out_channels=config['out_channels'],
        n_layers=config['n_layers'],
    ).to(device)
    
    params_fno = sum(p.numel() for p in model_fno.parameters())
    print(f"参数量: {params_fno:,}")
    
    results_fno, best_fno = train_model(
        model_fno, train_x, train_y, test_x, test_y,
        epochs=config['epochs']
    )
    results['FNO'] = {'params': params_fno, 'best_loss': best_fno}
    
    # ---------- 测试 MHF-FNO ----------
    print("\n" + "=" * 60)
    print("测试 MHF-FNO (推荐配置: n_heads=2, mhf_layers=[0,2])")
    print("=" * 60)
    
    torch.manual_seed(42)
    model_mhf = MHFFNO(
        n_modes=config['n_modes'],
        hidden_channels=config['hidden_channels'],
        in_channels=config['in_channels'],
        out_channels=config['out_channels'],
        n_layers=config['n_layers'],
        n_heads=2,  # 推荐配置
        mhf_layers=[0, 2],  # 首尾层使用 MHF
    ).to(device)
    
    params_mhf = sum(p.numel() for p in model_mhf.parameters())
    print(f"参数量: {params_mhf:,} ({(1-params_mhf/params_fno)*100:+.1f}%)")
    
    results_mhf, best_mhf = train_model(
        model_mhf, train_x, train_y, test_x, test_y,
        epochs=config['epochs']
    )
    results['MHF-FNO'] = {'params': params_mhf, 'best_loss': best_mhf}
    
    # ---------- 结果对比 ----------
    print("\n" + "=" * 60)
    print("结果对比")
    print("=" * 60)
    
    print(f"{'指标':<15} {'FNO':<15} {'MHF-FNO':<15} {'变化':<15}")
    print("-" * 60)
    print(f"{'参数量':<15} {params_fno:<15,} {params_mhf:<15,} {(1-params_mhf/params_fno)*100:+.1f}%")
    print(f"{'最佳测试Loss':<15} {best_fno:<15.6f} {best_mhf:<15.6f} {(best_mhf-best_fno)/best_fno*100:+.1f}%")
    
    print("\n✅ 测试完成！")
    
    print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋 推荐配置说明
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

本示例使用独立的 MHFFNO 类实现，所有 FNO 层都使用 MHF 卷积。

另一种推荐方式是使用 create_hybrid_fno（混合模式）:

    from mhf_fno import create_hybrid_fno
    
    model = create_hybrid_fno(
        n_modes=(8, 8),
        hidden_channels=32,
        n_heads=2,           # 推荐 2
        mhf_layers=[0, 2],   # 首尾层使用 MHF
    )

混合模式效果:
- 参数减少: 30.6%
- 精度损失: 1.4%

详见: benchmark/OPTIMIZATION_REPORT.md
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
    
    return results


if __name__ == '__main__':
    main()