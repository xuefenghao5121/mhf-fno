#!/usr/bin/env python3
"""
MHF-FNO 完整使用示例 - 拷贝即用

这个脚本展示如何使用核心库的 MHF-FNO 实现。

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
import sys
from pathlib import Path

# 添加父目录到路径，导入核心库
sys.path.insert(0, str(Path(__file__).parent.parent))
from mhf_fno import create_hybrid_fno, MHFFNO, get_device

# 检查是否有 GPU
device = get_device()
print(f"使用设备: {device}")


# ============================================================
# 第三步：训练函数
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
# 第四步：生成简单测试数据（Darcy Flow 风格）
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
# 第五步：主测试流程
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
    
    # ---------- 测试 MHF-FNO (使用核心库) ----------
    print("\n" + "=" * 60)
    print("测试 MHF-FNO (推荐配置: n_heads=2, mhf_layers=[0,2])")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    # 使用核心库的 create_hybrid_fno 函数
    model_mhf = create_hybrid_fno(
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

本示例使用核心库的 create_hybrid_fno 函数创建混合 FNO 模型。

两种创建方式:

1. 混合模式 (推荐):
    from mhf_fno import create_hybrid_fno
    
    model = create_hybrid_fno(
        n_modes=(8, 8),
        hidden_channels=32,
        n_heads=2,           # 推荐 2
        mhf_layers=[0, 2],   # 首尾层使用 MHF
    )

2. 预设配置:
    from mhf_fno import MHFFNO
    
    model = MHFFNO.best_config(n_modes=(8, 8), hidden_channels=32)

混合模式效果:
- 参数减少: 30.6%
- 精度损失: 1.4%

详见: benchmark/OPTIMIZATION_REPORT.md
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
    
    return results


if __name__ == '__main__':
    main()