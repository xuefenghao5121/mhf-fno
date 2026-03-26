#!/usr/bin/env python3
"""
NS 数据集快速测试 - n=1000, epochs=30

目标: 快速验证数据量扩展的效果
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


def load_ns_data():
    """加载 NS 数据 (1000 samples)"""
    print("\n📊 加载 Navier-Stokes 数据 (1000 samples)...")
    
    train_path = Path(__file__).parent / 'data' / 'ns_train_32_large.pt'
    test_path = Path(__file__).parent / 'data' / 'ns_test_32_large.pt'
    
    if not train_path.exists():
        print(f"❌ 数据文件不存在: {train_path}")
        return None
    
    train_data = torch.load(train_path, weights_only=False)
    test_data = torch.load(test_path, weights_only=False)
    
    # 解析数据
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
    
    print(f"✅ 数据加载成功: 训练 {train_x.shape[0]}, 测试 {test_x.shape[0]}")
    print(f"   分辨率: {train_x.shape[-1]}x{train_x.shape[-1]}")
    
    return train_x, train_y, test_x, test_y


def train_model(model, train_x, train_y, test_x, test_y, epochs=30):
    """训练模型"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = LpLoss(d=2, p=2, reduction='mean')
    
    best_test_loss = float('inf')
    n_train = train_x.shape[0]
    batch_size = 32
    
    print(f"  开始训练 (epochs={epochs}, n_train={n_train})...")
    
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
        
        # 测试
        model.eval()
        with torch.no_grad():
            test_loss = loss_fn(model(test_x), test_y).item()
        
        if test_loss < best_test_loss:
            best_test_loss = test_loss
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: Train {train_loss/batch_count:.4f}, Test {test_loss:.4f} (best: {best_test_loss:.4f})")
    
    return best_test_loss


def main():
    print("="*60)
    print("NS 数据集快速测试 - n=1000, epochs=30")
    print("="*60)
    
    # 加载数据
    data = load_ns_data()
    if data is None:
        return
    
    train_x, train_y, test_x, test_y = data
    
    resolution = train_x.shape[-1]
    n_modes = (resolution // 4, resolution // 4)
    in_channels = train_x.shape[1]
    out_channels = train_y.shape[1]
    
    results = {}
    
    # ========== FNO Baseline ==========
    print(f"\n{'='*60}")
    print("测试 FNO Baseline")
    print(f"{'='*60}")
    
    torch.manual_seed(42)
    model_fno = FNO(
        n_modes=n_modes,
        hidden_channels=32,
        in_channels=in_channels,
        out_channels=out_channels,
        n_layers=3,
    )
    
    params_fno = count_parameters(model_fno)
    print(f"参数量: {params_fno:,}")
    
    t0 = time.time()
    loss_fno = train_model(model_fno, train_x, train_y, test_x, test_y, epochs=30)
    time_fno = time.time() - t0
    
    print(f"\n✅ FNO 完成:")
    print(f"  最佳测试损失: {loss_fno:.4f}")
    print(f"  训练时间: {time_fno:.1f}s")
    
    results['FNO'] = {
        'parameters': params_fno,
        'best_test_loss': loss_fno,
        'training_time': time_fno,
    }
    
    # ========== MHF+CoDA ==========
    print(f"\n{'='*60}")
    print("测试 MHF+CoDA")
    print(f"{'='*60}")
    
    torch.manual_seed(42)
    model_mhf = create_mhf_fno_with_attention(
        n_modes=n_modes,
        hidden_channels=32,
        in_channels=in_channels,
        out_channels=out_channels,
        n_layers=3,
        n_heads=4,
        mhf_layers=[0, 2],
        attention_layers=[0, 2]
    )
    
    params_mhf = count_parameters(model_mhf)
    param_reduction = (params_fno - params_mhf) / params_fno * 100
    print(f"参数量: {params_mhf:,} ({param_reduction:.1f}% 减少)")
    
    t0 = time.time()
    loss_mhf = train_model(model_mhf, train_x, train_y, test_x, test_y, epochs=30)
    time_mhf = time.time() - t0
    
    improvement = (loss_fno - loss_mhf) / loss_fno * 100
    marker = "✅" if improvement > 0 else "⚠️"
    
    print(f"\n✅ MHF+CoDA 完成:")
    print(f"  最佳测试损失: {loss_mhf:.4f}")
    print(f"  训练时间: {time_mhf:.1f}s")
    print(f"  vs FNO: {improvement:+.2f}% {marker}")
    
    results['MHF+CoDA'] = {
        'parameters': params_mhf,
        'best_test_loss': loss_mhf,
        'training_time': time_mhf,
        'param_reduction': param_reduction,
        'improvement_vs_fno': improvement,
    }
    
    # ========== 汇总 ==========
    print(f"\n{'='*60}")
    print("结果汇总")
    print(f"{'='*60}")
    
    print(f"\n{'模型':<15} {'参数量':<12} {'测试Loss':<12} {'vs FNO':<12}")
    print("-"*50)
    print(f"{'FNO':<15} {params_fno:<12,} {loss_fno:<12.4f} {'基准':<12}")
    print(f"{'MHF+CoDA':<15} {params_mhf:<12,} {loss_mhf:<12.4f} {improvement:+.2f}% {marker}")
    
    # 保存结果
    output_path = Path(__file__).parent.parent / 'ns_test_1000_quick_results.json'
    with open(output_path, 'w') as f:
        json.dump({
            'config': {
                'n_train': train_x.shape[0],
                'n_test': test_x.shape[0],
                'epochs': 30,
                'resolution': resolution,
            },
            'results': results,
        }, f, indent=2)
    
    print(f"\n✅ 结果已保存到: {output_path}")


if __name__ == '__main__':
    main()
