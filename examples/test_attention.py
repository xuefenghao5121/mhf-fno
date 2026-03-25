#!/usr/bin/env python3
"""
MHF-FNO 跨头注意力测试

对比测试：
1. FNO (基准)
2. MHF-FNO (无注意力)
3. MHF-FNO + Cross-Head Attention (带注意力)

预期结果：
- MHF-FNO: 参数减少，精度略有下降
- MHF-FNO + Attention: 参数减少，精度提升（跨频率交互）
"""

import torch
import torch.nn as nn
import numpy as np
import time
import sys
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from mhf_fno import (
    create_hybrid_fno, 
    MHFFNO, 
    MHFFNOWithAttention,
    get_device
)


def l2_loss(pred, target):
    """L2 Loss"""
    return torch.mean((pred - target) ** 2)


def relative_l2_error(pred, target):
    """相对 L2 误差"""
    return torch.norm(pred - target) / torch.norm(target)


def generate_darcy_flow_data(n_train=500, n_test=100, resolution=16, complexity='simple'):
    """
    生成 Darcy Flow 风格的测试数据
    
    复杂度说明：
    - simple: 简单平滑场
    - medium: 多尺度特征
    - complex: 强频率耦合（需要跨头注意力）
    """
    print(f"生成数据: 训练 {n_train}, 测试 {n_test}, 分辨率 {resolution}x{resolution}, 复杂度 {complexity}")
    
    def generate_smooth_field(n, res):
        """平滑随机场"""
        x = torch.randn(n, 1, res, res)
        kernel = torch.ones(1, 1, 3, 3) / 9
        for _ in range(3):
            x = torch.nn.functional.conv2d(x, kernel, padding=1)
        return x
    
    def generate_multiscale_field(n, res):
        """多尺度随机场（需要跨频率交互）"""
        # 低频分量
        low = generate_smooth_field(n, res)
        
        # 高频分量
        high = torch.randn(n, 1, res, res)
        kernel = torch.tensor([[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]], dtype=torch.float32).reshape(1, 1, 3, 3)
        high = torch.nn.functional.conv2d(high, kernel, padding=1)
        high = torch.nn.functional.avg_pool2d(high, 2, stride=1, padding=1)
        
        # 混合（需要跨频率耦合）
        x = 0.7 * low + 0.3 * high
        return x
    
    def generate_complex_field(n, res):
        """复杂场（强频率耦合）"""
        x = torch.randn(n, 1, res, res)
        
        # 添加多个频率分量
        for freq in [2, 4, 8]:
            pattern = torch.sin(torch.linspace(0, freq * 2 * np.pi, res))
            pattern = pattern.reshape(1, 1, res, 1).expand(n, 1, res, res)
            x = x + 0.1 * pattern
        
        # 添加交互项
        x = x + 0.1 * torch.sin(x * 3)
        
        return x
    
    # 根据复杂度选择生成方式
    generators = {
        'simple': generate_smooth_field,
        'medium': generate_multiscale_field,
        'complex': generate_complex_field
    }
    
    gen_func = generators.get(complexity, generate_smooth_field)
    
    train_x = gen_func(n_train, resolution)
    test_x = gen_func(n_test, resolution)
    
    # 模拟 Darcy 方程的解（简化版）
    # 真实情况下需要求解 PDE
    if complexity == 'simple':
        train_y = torch.nn.functional.avg_pool2d(train_x, 3, stride=1, padding=1)
        test_y = torch.nn.functional.avg_pool2d(test_x, 3, stride=1, padding=1)
    elif complexity == 'medium':
        # 多尺度解：需要处理不同频率
        low_pass = torch.nn.functional.avg_pool2d(train_x, 3, stride=1, padding=1)
        high_pass = train_x - low_pass
        train_y = 0.6 * low_pass + 0.4 * torch.nn.functional.avg_pool2d(high_pass, 3, stride=1, padding=1)
        
        low_pass = torch.nn.functional.avg_pool2d(test_x, 3, stride=1, padding=1)
        high_pass = test_x - low_pass
        test_y = 0.6 * low_pass + 0.4 * torch.nn.functional.avg_pool2d(high_pass, 3, stride=1, padding=1)
    else:  # complex
        # 复杂解：非线性耦合
        train_y = torch.nn.functional.avg_pool2d(train_x, 3, stride=1, padding=1)
        train_y = train_y + 0.2 * torch.tanh(train_x)
        train_y = train_y + 0.1 * torch.sin(train_x * 2)
        
        test_y = torch.nn.functional.avg_pool2d(test_x, 3, stride=1, padding=1)
        test_y = test_y + 0.2 * torch.tanh(test_x)
        test_y = test_y + 0.1 * torch.sin(test_x * 2)
    
    print(f"✅ 数据生成完成")
    return train_x, train_y, test_x, test_y


def train_model(model, train_x, train_y, test_x, test_y, epochs=50, lr=1e-3, device='cpu'):
    """训练模型"""
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    results = {'train_loss': [], 'test_loss': [], 'relative_error': []}
    best_test = float('inf')
    best_error = float('inf')
    
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
            pred = model(test_x.to(device))
            test_loss = l2_loss(pred, test_y.to(device)).item()
            rel_error = relative_l2_error(pred, test_y.to(device)).item()
        
        results['train_loss'].append(train_loss / (n_train // batch_size))
        results['test_loss'].append(test_loss)
        results['relative_error'].append(rel_error)
        
        if test_loss < best_test:
            best_test = test_loss
            best_error = rel_error
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: Train {train_loss/(n_train//batch_size):.4f}, "
                  f"Test {test_loss:.4f}, RelError {rel_error:.4f}")
    
    return results, best_test, best_error


def run_comparison(complexity='medium', epochs=30):
    """运行对比测试"""
    
    print("=" * 70)
    print(f"MHF-FNO 跨头注意力对比测试 (复杂度: {complexity})")
    print("=" * 70)
    
    device = get_device()
    print(f"设备: {device}")
    
    # 生成数据
    train_x, train_y, test_x, test_y = generate_darcy_flow_data(
        n_train=500, n_test=100, resolution=16, complexity=complexity
    )
    
    # 配置
    config = {
        'n_modes': (8, 8),
        'hidden_channels': 32,
        'in_channels': 1,
        'out_channels': 1,
        'n_layers': 3,
        'epochs': epochs,
        'n_heads': 4,
    }
    
    results = {}
    
    # ========== 测试 FNO (基准) ==========
    print("\n" + "=" * 70)
    print("测试 1: FNO (基准)")
    print("=" * 70)
    
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
    
    _, best_fno, error_fno = train_model(
        model_fno, train_x, train_y, test_x, test_y,
        epochs=config['epochs'], device=device
    )
    results['FNO'] = {'params': params_fno, 'best_loss': best_fno, 'rel_error': error_fno}
    
    # ========== 测试 MHF-FNO (无注意力) ==========
    print("\n" + "=" * 70)
    print("测试 2: MHF-FNO (无注意力)")
    print("=" * 70)
    
    torch.manual_seed(42)
    model_mhf = create_hybrid_fno(
        n_modes=config['n_modes'],
        hidden_channels=config['hidden_channels'],
        in_channels=config['in_channels'],
        out_channels=config['out_channels'],
        n_layers=config['n_layers'],
        n_heads=config['n_heads'],
        mhf_layers=[0, 2],
    ).to(device)
    
    params_mhf = sum(p.numel() for p in model_mhf.parameters())
    print(f"参数量: {params_mhf:,} ({(1-params_mhf/params_fno)*100:+.1f}%)")
    
    _, best_mhf, error_mhf = train_model(
        model_mhf, train_x, train_y, test_x, test_y,
        epochs=config['epochs'], device=device
    )
    results['MHF-FNO'] = {'params': params_mhf, 'best_loss': best_mhf, 'rel_error': error_mhf}
    
    # ========== 测试 MHF-FNO + Attention ==========
    print("\n" + "=" * 70)
    print("测试 3: MHF-FNO + Cross-Head Attention")
    print("=" * 70)
    
    torch.manual_seed(42)
    model_attn = MHFFNOWithAttention.best_config(
        n_modes=config['n_modes'],
        hidden_channels=config['hidden_channels'],
        in_channels=config['in_channels'],
        out_channels=config['out_channels'],
        n_heads=config['n_heads'],
    ).to(device)
    
    params_attn = sum(p.numel() for p in model_attn.parameters())
    print(f"参数量: {params_attn:,} ({(1-params_attn/params_fno)*100:+.1f}%)")
    
    _, best_attn, error_attn = train_model(
        model_attn, train_x, train_y, test_x, test_y,
        epochs=config['epochs'], device=device
    )
    results['MHF-FNO+Attn'] = {'params': params_attn, 'best_loss': best_attn, 'rel_error': error_attn}
    
    # ========== 结果汇总 ==========
    print("\n" + "=" * 70)
    print("结果汇总")
    print("=" * 70)
    
    print(f"{'模型':<20} {'参数量':>12} {'参数变化':>10} {'测试Loss':>12} {'相对误差':>12}")
    print("-" * 70)
    
    for name, res in results.items():
        param_change = (1 - res['params']/params_fno) * 100
        print(f"{name:<20} {res['params']:>12,} {param_change:>+10.1f}% "
              f"{res['best_loss']:>12.6f} {res['rel_error']:>12.4f}")
    
    # 计算改进
    print("\n" + "=" * 70)
    print("注意力机制效果分析")
    print("=" * 70)
    
    loss_improve = (best_mhf - best_attn) / best_mhf * 100
    error_improve = (error_mhf - error_attn) / error_mhf * 100
    
    print(f"MHF-FNO vs MHF-FNO+Attn:")
    print(f"  Loss 改进: {loss_improve:+.2f}%")
    print(f"  相对误差改进: {error_improve:+.2f}%")
    
    if loss_improve > 0:
        print(f"\n✅ 跨头注意力有效！在 {complexity} 数据集上提升了 {loss_improve:.1f}%")
    else:
        print(f"\n⚠️ 在 {complexity} 数据集上，注意力机制未带来明显改进")
    
    return results


def main():
    """主函数"""
    
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║                 MHF-FNO 跨头注意力机制测试                              ║
╚══════════════════════════════════════════════════════════════════════════╝

本测试验证跨头注意力机制的效果：

1. FNO (基准): 标准 Fourier Neural Operator
2. MHF-FNO: 多头频域卷积，头之间独立
3. MHF-FNO + Attention: 添加跨头注意力，允许频率交互

预期结果：
- 简单数据: 注意力效果不明显
- 中等复杂度: 注意力带来一定改进
- 复杂数据: 注意力显著提升性能（跨频率耦合）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
    
    # 测试不同复杂度
    complexities = ['simple', 'medium', 'complex']
    all_results = {}
    
    for comp in complexities:
        print(f"\n{'='*70}")
        print(f"测试复杂度: {comp}")
        print(f"{'='*70}")
        
        all_results[comp] = run_comparison(complexity=comp, epochs=30)
    
    # 最终总结
    print("\n" + "=" * 70)
    print("最终总结")
    print("=" * 70)
    
    print("\n不同复杂度下注意力机制的效果：")
    print(f"{'复杂度':<15} {'MHF-FNO Loss':>15} {'MHF+Attn Loss':>15} {'改进':>10}")
    print("-" * 60)
    
    for comp, res in all_results.items():
        mhf_loss = res['MHF-FNO']['best_loss']
        attn_loss = res['MHF-FNO+Attn']['best_loss']
        improve = (mhf_loss - attn_loss) / mhf_loss * 100
        print(f"{comp:<15} {mhf_loss:>15.6f} {attn_loss:>15.6f} {improve:>+10.2f}%")
    
    print("\n✅ 测试完成！")


if __name__ == '__main__':
    main()