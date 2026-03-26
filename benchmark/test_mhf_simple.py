#!/usr/bin/env python3
"""
MHF-FNO 简化对比测试 - 无PINO物理约束

测试MHF-FNO vs FNO的基础性能
"""

import json
import sys
from pathlib import Path

import torch
from neuralop.losses.data_losses import LpLoss
from neuralop.models import FNO

sys.path.insert(0, str(Path(__file__).parent.parent))
from mhf_fno import create_mhf_fno_with_attention

import functools
print = functools.partial(print, flush=True)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def load_data():
    """加载数据"""
    train_path = Path(__file__).parent / 'data' / 'ns_train_32_large.pt'
    test_path = Path(__file__).parent / 'data' / 'ns_test_32_large.pt'
    
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
    
    if train_x.dim() == 3:
        train_x = train_x.unsqueeze(1)
        train_y = train_y.unsqueeze(1)
    if test_x.dim() == 3:
        test_x = test_x.unsqueeze(1)
        test_y = test_y.unsqueeze(1)
    
    return train_x.float(), train_y.float(), test_x.float(), test_y.float()


def train(model, train_x, train_y, test_x, test_y, epochs=30):
    """训练"""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    loss_fn = LpLoss(d=2, p=2, reduction='mean')
    test_fn = LpLoss(d=2, p=2)
    
    n_train = train_x.shape[0]
    batch_size = 32
    
    best_test = float('inf')
    
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train)
        
        train_loss = 0
        for i in range(0, n_train, batch_size):
            bx = train_x[perm[i:i+batch_size]]
            by = train_y[perm[i:i+batch_size]]
            
            optimizer.zero_grad()
            y_pred = model(bx)
            loss = loss_fn(y_pred, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        scheduler.step()
        
        # 测试
        model.eval()
        with torch.no_grad():
            test_pred = model(test_x)
            test_loss = test_fn(test_pred, test_y).item()
        
        if test_loss < best_test:
            best_test = test_loss
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:2d}: train={train_loss/(n_train//batch_size):.6f}, test={test_loss:.6f}, best={best_test:.6f}")
    
    return best_test


def main():
    print("=" * 80)
    print("MHF-FNO vs FNO 基础对比 (无PINO)")
    print("=" * 80)
    
    train_x, train_y, test_x, test_y = load_data()
    print(f"\n✅ 数据: 训练 {train_x.shape[0]}, 测试 {test_x.shape[0]}")
    
    device = torch.device('cpu')
    
    results = {}
    
    # 1. FNO
    print("\n" + "=" * 80)
    print("1️⃣  FNO")
    print("=" * 80)
    
    model_fno = FNO(
        n_modes=(16, 16),
        hidden_channels=32,
        in_channels=1,
        out_channels=1,
        n_layers=4
    ).to(device)
    
    print(f"参数量: {count_parameters(model_fno):,}")
    loss_fno = train(model_fno, train_x, train_y, test_x, test_y, epochs=30)
    
    results['FNO'] = {'loss': loss_fno, 'params': count_parameters(model_fno)}
    print(f"\n✅ FNO: {loss_fno:.6f}")
    
    # 2. MHF-FNO
    print("\n" + "=" * 80)
    print("2️⃣  MHF-FNO")
    print("=" * 80)
    
    model_mhf = create_mhf_fno_with_attention(
        n_modes=(16, 16),
        hidden_channels=32,
        in_channels=1,
        out_channels=1,
        n_layers=4,
        n_heads=2,
        mhf_layers=[0],
        attention_layers=[0]
    ).to(device)
    
    print(f"参数量: {count_parameters(model_mhf):,}")
    loss_mhf = train(model_mhf, train_x, train_y, test_x, test_y, epochs=30)
    
    results['MHF-FNO'] = {'loss': loss_mhf, 'params': count_parameters(model_mhf)}
    print(f"\n✅ MHF-FNO: {loss_mhf:.6f}")
    
    # 对比
    print("\n" + "=" * 80)
    print("📊 结果对比")
    print("=" * 80)
    
    baseline = results['FNO']['loss']
    for name, res in results.items():
        vs_fno = ((res['loss'] - baseline) / baseline) * 100
        print(f"{name:<15} {res['loss']:.6f}  ({vs_fno:+.2f}%)  params={res['params']:,}")
    
    # 保存
    output = {
        'FNO': {'test_loss': results['FNO']['loss'], 'params': results['FNO']['params']},
        'MHF-FNO': {'test_loss': results['MHF-FNO']['loss'], 'params': results['MHF-FNO']['params']}
    }
    
    with open(Path(__file__).parent / 'mhf_baseline_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n💾 结果已保存到 mhf_baseline_results.json")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
