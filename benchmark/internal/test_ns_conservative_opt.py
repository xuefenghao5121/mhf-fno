#!/usr/bin/env python3
"""
NS 数据集优化测试 - 保守配置 (mhf_layers=[0])

根据诊断报告推荐:
- mhf_layers=[0] (仅第1层使用MHF，适合强频率耦合的NS方程)
- n_train=1000 (MHF需要更多数据)
- epochs=50 (充分训练)
- n_modes=(16,16) (保持与之前测试一致)
"""

import json
import sys
import time
from pathlib import Path

import torch
from neuralop.losses.data_losses import LpLoss
from neuralop.models import FNO

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from mhf_fno import create_mhf_fno_with_attention

# 强制刷新输出
import functools
print = functools.partial(print, flush=True)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def generate_ns_data(n_train=1000, n_test=200, resolution=32):
    """生成 NS 数据"""
    print(f"\n📊 生成 Navier-Stokes 数据 (n_train={n_train}, n_test={n_test})...")
    
    # 导入数据生成函数
    sys.path.insert(0, str(Path(__file__).parent))
    from generate_data import generate_navier_stokes_data
    
    # 生成数据
    train_x, train_y = generate_navier_stokes_data(
        n_samples=n_train,
        resolution=resolution,
        viscosity=1e-3,
        n_steps=100,
        device='cpu'
    )
    
    test_x, test_y = generate_navier_stokes_data(
        n_samples=n_test,
        resolution=resolution,
        viscosity=1e-3,
        n_steps=100,
        device='cpu'
    )
    
    # 保存
    data_dir = Path(__file__).parent / 'data'
    data_dir.mkdir(exist_ok=True)
    
    train_path = data_dir / 'ns_train_32_large.pt'
    test_path = data_dir / 'ns_test_32_large.pt'
    
    torch.save({'x': train_x, 'y': train_y}, train_path)
    torch.save({'x': test_x, 'y': test_y}, test_path)
    
    print(f"✅ 数据已保存: {train_path}, {test_path}")
    
    return train_x, train_y, test_x, test_y


def load_or_generate_data(n_train=1000, n_test=200):
    """加载或生成数据"""
    train_path = Path(__file__).parent / 'data' / 'ns_train_32_large.pt'
    test_path = Path(__file__).parent / 'data' / 'ns_test_32_large.pt'
    
    if train_path.exists() and test_path.exists():
        print("\n📊 加载已有 Navier-Stokes 数据 (1000 samples)...")
        train_data = torch.load(train_path, weights_only=False)
        test_data = torch.load(test_path, weights_only=False)
        
        if isinstance(train_data, dict):
            train_x = train_data.get('x', train_data.get('train_x'))
            train_y = train_data.get('y', train_data.get('train_y'))
        else:
            train_x, train_y = train_data[0], train_data[1]
        
        if isinstance(test_data, dict):
            test_x = test_data.get('x', test_data.get('test_x'))
            test_y = test_data.get('y', test_data.get('test_y'))
        else:
            test_x, test_y = test_data[0], test_data[1]
        
        # 检查数据量
        if train_x.shape[0] < n_train:
            print(f"⚠️  数据量不足 ({train_x.shape[0]} < {n_train})，重新生成...")
            return generate_ns_data(n_train, n_test)
        
        print(f"✅ 数据加载成功: 训练 {train_x.shape[0]}, 测试 {test_x.shape[0]}")
    else:
        print("\n⚠️  数据文件不存在，开始生成...")
        train_x, train_y, test_x, test_y = generate_ns_data(n_train, n_test)
    
    # 确保维度
    if train_x.dim() == 3:
        train_x = train_x.unsqueeze(1)
        train_y = train_y.unsqueeze(1)
    if test_x.dim() == 3:
        test_x = test_x.unsqueeze(1)
        test_y = test_y.unsqueeze(1)
    
    train_x = train_x.float()
    train_y = train_y.float()
    test_x = test_x.float()
    test_y = test_y.float()
    
    print(f"   分辨率: {train_x.shape[-1]}x{train_x.shape[-1]}")
    
    return train_x, train_y, test_x, test_y


def train_model(model, train_x, train_y, test_x, test_y, epochs=50, model_name="Model"):
    """训练模型"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = LpLoss(d=2, p=2, reduction='mean')
    
    best_test_loss = float('inf')
    best_epoch = 0
    n_train = train_x.shape[0]
    batch_size = 32
    
    train_losses = []
    test_losses = []
    
    start_time = time.time()
    print(f"\n  开始训练 {model_name} (epochs={epochs}, n_train={n_train})...")
    
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train)
        
        train_loss = 0
        batch_count = 0
        
        for i in range(0, n_train, batch_size):
            bx = train_x[perm[i:i+batch_size]]
            by = train_y[perm[i:i+batch_size]]
            
            optimizer.zero_grad()
            loss = loss_fn(model(bx), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            batch_count += 1
        
        scheduler.step()
        avg_train_loss = train_loss / batch_count
        train_losses.append(avg_train_loss)
        
        # 测试
        model.eval()
        with torch.no_grad():
            test_loss = loss_fn(model(test_x), test_y).item()
        test_losses.append(test_loss)
        
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_epoch = epoch + 1
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}: Train {avg_train_loss:.4f}, Test {test_loss:.4f} (best: {best_test_loss:.4f} @ epoch {best_epoch})")
    
    total_time = time.time() - start_time
    
    return {
        'best_test_loss': best_test_loss,
        'best_epoch': best_epoch,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'total_time': total_time,
        'avg_epoch_time': total_time / epochs
    }


def main():
    print("="*70)
    print("NS 数据集优化测试 - 保守配置 (mhf_layers=[0])")
    print("="*70)
    print("📋 优化策略:")
    print("   - mhf_layers=[0] (仅第1层，适合强频率耦合)")
    print("   - n_train=1000 (MHF需要更多数据)")
    print("   - epochs=50 (充分训练)")
    print("   - n_modes=(16,16) (保持一致)")
    print("="*70)
    
    # 加载或生成数据
    train_x, train_y, test_x, test_y = load_or_generate_data(n_train=1000, n_test=200)
    
    in_channels = train_x.shape[1]
    out_channels = train_y.shape[1]
    
    # 配置
    n_modes = (16, 16)
    hidden_channels = 32
    n_layers = 3
    
    results = {
        'config': {
            'epochs': 50,
            'n_train': train_x.shape[0],
            'n_test': test_x.shape[0],
            'batch_size': 32,
            'learning_rate': 5e-4,
            'n_modes': n_modes,
            'hidden_channels': hidden_channels,
            'n_layers': n_layers
        },
        'results': {}
    }
    
    # ========================================
    # 1. FNO Baseline
    # ========================================
    print("\n" + "="*70)
    print("测试 FNO Baseline")
    print("="*70)
    
    fno = FNO(
        n_modes=n_modes,
        hidden_channels=hidden_channels,
        in_channels=in_channels,
        out_channels=out_channels,
        n_layers=n_layers
    )
    
    fno_params = count_parameters(fno)
    print(f"参数量: {fno_params:,}")
    
    fno_result = train_model(fno, train_x, train_y, test_x, test_y, epochs=50, model_name="FNO")
    
    print(f"\n✅ FNO 完成:")
    print(f"  最佳测试损失: {fno_result['best_test_loss']:.4f} (epoch {fno_result['best_epoch']})")
    print(f"  训练时间: {fno_result['total_time']:.1f}s")
    
    results['results']['FNO'] = {
        'parameters': fno_params,
        **fno_result
    }
    
    # ========================================
    # 2. MHF-FNO 保守配置 (mhf_layers=[0])
    # ========================================
    print("\n" + "="*70)
    print("测试 MHF-FNO 保守配置 (mhf_layers=[0])")
    print("="*70)
    
    mhf_fno = create_mhf_fno_with_attention(
        n_modes=n_modes,
        hidden_channels=hidden_channels,
        in_channels=in_channels,
        out_channels=out_channels,
        n_layers=n_layers,
        mhf_layers=[0],  # ⭐ 保守配置：仅第1层
        n_heads=4,
        head_dim=16,
        attention_layers=[0],
        bottleneck=4,
        gate_init=0.1,
        positional_embedding='grid'
    )
    
    mhf_params = count_parameters(mhf_fno)
    param_reduction = (1 - mhf_params / fno_params) * 100
    print(f"参数量: {mhf_params:,} ({param_reduction:.1f}% 减少)")
    
    mhf_result = train_model(mhf_fno, train_x, train_y, test_x, test_y, epochs=50, model_name="MHF-FNO (conservative)")
    
    improvement = (1 - mhf_result['best_test_loss'] / fno_result['best_test_loss']) * 100
    
    print(f"\n✅ MHF-FNO 完成:")
    print(f"  最佳测试损失: {mhf_result['best_test_loss']:.4f} (epoch {mhf_result['best_epoch']})")
    print(f"  vs FNO: {improvement:+.2f}%")
    print(f"  训练时间: {mhf_result['total_time']:.1f}s")
    
    results['results']['MHF-FNO-Conservative'] = {
        'parameters': mhf_params,
        'param_reduction_pct': param_reduction,
        'improvement_vs_fno': improvement,
        **mhf_result
    }
    
    # ========================================
    # 3. 对比原始配置 (mhf_layers=[0,2])
    # ========================================
    print("\n" + "="*70)
    print("测试 MHF-FNO 原始配置 (mhf_layers=[0,2]) - 对照组")
    print("="*70)
    
    mhf_fno_original = create_mhf_fno_with_attention(
        n_modes=n_modes,
        hidden_channels=hidden_channels,
        in_channels=in_channels,
        out_channels=out_channels,
        n_layers=n_layers,
        mhf_layers=[0, 2],  # 原始配置
        n_heads=4,
        head_dim=16,
        attention_layers=[0, 2],
        bottleneck=4,
        gate_init=0.1,
        positional_embedding='grid'
    )
    
    mhf_orig_params = count_parameters(mhf_fno_original)
    param_reduction_orig = (1 - mhf_orig_params / fno_params) * 100
    print(f"参数量: {mhf_orig_params:,} ({param_reduction_orig:.1f}% 减少)")
    
    mhf_orig_result = train_model(mhf_fno_original, train_x, train_y, test_x, test_y, epochs=50, model_name="MHF-FNO (original)")
    
    improvement_orig = (1 - mhf_orig_result['best_test_loss'] / fno_result['best_test_loss']) * 100
    
    print(f"\n✅ MHF-FNO (original) 完成:")
    print(f"  最佳测试损失: {mhf_orig_result['best_test_loss']:.4f} (epoch {mhf_orig_result['best_epoch']})")
    print(f"  vs FNO: {improvement_orig:+.2f}%")
    print(f"  训练时间: {mhf_orig_result['total_time']:.1f}s")
    
    results['results']['MHF-FNO-Original'] = {
        'parameters': mhf_orig_params,
        'param_reduction_pct': param_reduction_orig,
        'improvement_vs_fno': improvement_orig,
        **mhf_orig_result
    }
    
    # ========================================
    # 总结
    # ========================================
    print("\n" + "="*70)
    print("📊 测试总结")
    print("="*70)
    
    print(f"\n{'模型':<30} {'参数量':>12} {'参数减少':>10} {'测试损失':>10} {'vs FNO':>10}")
    print("-"*70)
    
    print(f"{'FNO Baseline':<30} {fno_params:>12,} {'-':>10} {fno_result['best_test_loss']:>10.4f} {'-':>10}")
    print(f"{'MHF-FNO (保守)':<30} {mhf_params:>12,} {param_reduction:>9.1f}% {mhf_result['best_test_loss']:>10.4f} {improvement:>9.2f}%")
    print(f"{'MHF-FNO (原始)':<30} {mhf_orig_params:>12,} {param_reduction_orig:>9.1f}% {mhf_orig_result['best_test_loss']:>10.4f} {improvement_orig:>9.2f}%")
    
    # 保存结果
    results['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
    
    output_file = Path(__file__).parent.parent / 'ns_conservative_opt_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ 结果已保存到: {output_file}")
    
    # 判断优化是否成功
    print("\n" + "="*70)
    print("🎯 优化结论")
    print("="*70)
    
    if improvement >= 0:
        print(f"✅ 保守配置优化成功！MHF-FNO 在 NS 数据集上优于 FNO {improvement:.2f}%")
        print(f"   同时参数量减少了 {param_reduction:.1f}%")
    else:
        print(f"⚠️  保守配置未能超越 FNO ({improvement:.2f}%)")
        print(f"   但参数量减少了 {param_reduction:.1f}%")
        
    if improvement > improvement_orig:
        print(f"\n✅ 保守配置 (mhf_layers=[0]) 优于原始配置 (mhf_layers=[0,2])")
        print(f"   改进幅度: {improvement - improvement_orig:.2f}%")
    else:
        print(f"\n⚠️  保守配置未能优于原始配置")
    
    print("\n" + "="*70)


if __name__ == '__main__':
    main()
