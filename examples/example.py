#!/usr/bin/env python3
"""
MHF-FNO 完整使用示例 - 拷贝即用

这个脚本包含所有必要的代码，可以直接运行测试 MHF-FNO vs FNO
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
        self.head_channels = out_channels // n_heads
        
        # 频域权重
        self.weight = nn.Parameter(
            torch.randn(n_heads, self.head_channels, self.head_channels, *n_modes)
        )
        self.fc = nn.Linear(in_channels, out_channels)
        self._init_weights()
    
    def _init_weights(self):
        with torch.no_grad():
            for h in range(self.n_heads):
                scale = 0.01 * (2 ** h)
                nn.init.normal_(self.weight[h], mean=0, std=scale)
            nn.init.xavier_normal_(self.fc.weight)
            nn.init.zeros_(self.fc.bias)
    
    def forward(self, x):
        batch_size = x.shape[0]
        x_ft = torch.fft.rfftn(x, dim=(-2, -1))
        
        out_ft = torch.zeros(
            batch_size, self.out_channels, *x_ft.shape[-2:],
            dtype=x_ft.dtype, device=x.device
        )
        
        for h in range(self.n_heads):
            start = h * self.head_channels
            end = start + self.head_channels
            for i in range(min(self.n_modes[0], x_ft.shape[-2])):
                for j in range(min(self.n_modes[1], x_ft.shape[-1])):
                    out_ft[:, start:end, i, j] = torch.einsum(
                        'bi,bio->bo',
                        x_ft[:, :, i, j],
                        self.weight[h, :, :, i, j]
                    )
        
        x = torch.fft.irfftn(out_ft, dim=(-2, -1))
        x = self.fc(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x


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
        for i in range(n_layers):
            use_mhf = i in self.mhf_layers
            block = nn.ModuleDict({
                'conv': MHFSpectralConv(
                    hidden_channels, hidden_channels, n_modes, n_heads
                ) if use_mhf else nn.Identity(),
                'w': nn.Conv2d(hidden_channels, hidden_channels, 1),
                'use_mhf': use_mhf,
            })
            self.fno_blocks.append(block)
        
        self.fc_out = nn.Linear(hidden_channels, out_channels)
    
    def forward(self, x):
        x = self.fc_in(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        for block in self.fno_blocks:
            if block['use_mhf']:
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
    print("测试 MHF-FNO")
    print("=" * 60)
    
    torch.manual_seed(42)
    model_mhf = MHFFNO(
        n_modes=config['n_modes'],
        hidden_channels=config['hidden_channels'],
        in_channels=config['in_channels'],
        out_channels=config['out_channels'],
        n_layers=config['n_layers'],
        n_heads=4,
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
    
    return results


if __name__ == '__main__':
    main()