#!/usr/bin/env python3
"""
NS 数据集快速优化测试

快速验证几个关键配置，使用较少的 epoch。
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import torch
from neuralop.losses.data_losses import LpLoss
from neuralop.models import FNO

# CPU 优化
torch.set_num_threads(os.cpu_count() or 1)

# 导入项目模块
sys.path.insert(0, str(Path(__file__).parent.parent))
from mhf_fno import create_hybrid_fno, create_mhf_fno_with_attention


def load_local_ns_data(n_train=500, n_test=100):
    """加载本地 NS 数据"""
    print(f"\n📊 加载本地 Navier-Stokes 数据...")
    
    # 首先尝试大版本
    train_path_large = Path(__file__).parent / 'data' / 'ns_train_32_large.pt'
    test_path_large = Path(__file__).parent / 'data' / 'ns_test_32_large.pt'
    
    if train_path_large.exists():
        print(f"   使用大版本数据 (1000 samples)")
        train_data = torch.load(train_path_large, weights_only=False)
        test_data = torch.load(test_path_large, weights_only=False)
    else:
        # 使用标准版本
        data_dir = Path(__file__).parent.parent / 'data'
        train_path = data_dir / 'ns_train_32.pt'
        test_path = data_dir / 'ns_test_32.pt'
        
        if not train_path.exists():
            print(f"❌ 数据文件不存在: {train_path}")
            return None
            
        train_data = torch.load(train_path, weights_only=False)
        test_data = torch.load(test_path, weights_only=False)
    
    # 解析数据格式
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
    
    # 确保维度正确
    if train_x.dim() == 3:
        train_x = train_x.unsqueeze(1)
        train_y = train_y.unsqueeze(1)
    if test_x.dim() == 3:
        test_x = test_x.unsqueeze(1)
        test_y = test_y.unsqueeze(1)
    
    # 转换类型
    train_x = train_x.float()
    train_y = train_y.float()
    test_x = test_x.float()
    test_y = test_y.float()
    
    # 限制样本数
    train_x = train_x[:n_train]
    train_y = train_y[:n_train]
    test_x = test_x[:n_test]
    test_y = test_y[:n_test]
    
    resolution = train_x.shape[-1]
    
    info = {
        'name': 'Navier-Stokes (本地)',
        'resolution': f'{resolution}x{resolution}',
        'n_train': train_x.shape[0],
        'n_test': test_x.shape[0],
        'input_channels': train_x.shape[1],
        'output_channels': train_y.shape[1],
        'n_modes': (resolution // 4, resolution // 4),
    }
    
    print(f"✅ 加载成功: 训练 {train_x.shape[0]}, 测试 {test_x.shape[0]}, 分辨率 {resolution}x{resolution}")
    return train_x, train_y, test_x, test_y, info


def train_model(model, train_x, train_y, test_x, test_y, epochs=30, batch_size=32, lr=1e-3, verbose=True):
    """训练模型"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = LpLoss(d=2, p=2, reduction='mean')
    
    results = {'train_losses': [], 'test_losses': [], 'best_test_loss': float('inf')}
    
    n_train = train_x.shape[0]
    
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
        
        model.eval()
        with torch.no_grad():
            test_loss = loss_fn(model(test_x), test_y).item()
        
        results['train_losses'].append(train_loss / batch_count)
        results['test_losses'].append(test_loss)
        
        if test_loss < results['best_test_loss']:
            results['best_test_loss'] = test_loss
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: Train {train_loss/batch_count:.4f}, Test {test_loss:.4f} (best: {results['best_test_loss']:.4f})")
    
    return results


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def main():
    print("="*60)
    print("NS 快速优化测试")
    print("="*60)
    
    # 加载数据
    data = load_local_ns_data(n_train=500, n_test=100)
    if data is None:
        return
    
    train_x, train_y, test_x, test_y, info = data
    
    results = {}
    
    # 测试配置
    test_configs = [
        ('FNO (基准)', 'FNO', {}),
        ('MHF-FNO [0,2]', 'MHF', {'mhf_layers': [0, 2]}),
        ('MHF+CoDA [0,2]', 'CoDA', {'mhf_layers': [0, 2]}),
        ('MHF+CoDA [0,1,2]', 'CoDA', {'mhf_layers': [0, 1, 2]}),
        ('MHF+CoDA n_heads=8', 'CoDA', {'mhf_layers': [0, 2], 'n_heads': 8}),
    ]
    
    for name, model_type, extra_args in test_configs:
        print(f"\n{'='*60}")
        print(f"测试: {name}")
        print(f"{'='*60}")
        
        torch.manual_seed(42)
        
        if model_type == 'FNO':
            model = FNO(
                n_modes=info['n_modes'],
                hidden_channels=32,
                in_channels=info['input_channels'],
                out_channels=info['output_channels'],
                n_layers=3,
            )
        elif model_type == 'MHF':
            model = create_hybrid_fno(
                n_modes=info['n_modes'],
                hidden_channels=32,
                in_channels=info['input_channels'],
                out_channels=info['output_channels'],
                n_layers=3,
                n_heads=extra_args.get('n_heads', 4),
                mhf_layers=extra_args.get('mhf_layers', [0, 2])
            )
        else:  # CoDA
            model = create_mhf_fno_with_attention(
                n_modes=info['n_modes'],
                hidden_channels=32,
                in_channels=info['input_channels'],
                out_channels=info['output_channels'],
                n_layers=3,
                n_heads=extra_args.get('n_heads', 4),
                mhf_layers=extra_args.get('mhf_layers', [0, 2]),
                attention_layers=extra_args.get('mhf_layers', [0, 2])
            )
        
        params = count_parameters(model)
        print(f"参数量: {params:,}")
        
        train_results = train_model(model, train_x, train_y, test_x, test_y, epochs=50)
        
        results[name] = {
            'parameters': params,
            'best_test_loss': train_results['best_test_loss'],
        }
    
    # 汇总
    print(f"\n{'='*60}")
    print("结果汇总")
    print(f"{'='*60}")
    
    baseline = results['FNO (基准)']['best_test_loss']
    
    print(f"\n{'配置':<25} {'参数量':<12} {'测试Loss':<12} {'vs FNO':<12}")
    print("-"*60)
    
    for name, res in results.items():
        improvement = (baseline - res['best_test_loss']) / baseline * 100
        marker = "✅" if improvement > 0 else "⚠️"
        print(f"{name:<25} {res['parameters']:<12,} {res['best_test_loss']:<12.4f} {improvement:+.2f}% {marker}")
    
    # 保存
    output_path = Path(__file__).parent.parent / 'ns_quick_results.json'
    with open(output_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': results,
        }, f, indent=2)
    
    print(f"\n✅ 结果已保存到: {output_path}")


if __name__ == '__main__':
    main()