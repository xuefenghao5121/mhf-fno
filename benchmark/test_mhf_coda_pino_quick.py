#!/usr/bin/env python3
"""
MHF+CoDA+PINO 快速测试（简化版）

使用更小的配置进行快速验证

作者: 天渊团队
日期: 2026-03-26
"""

import sys
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from mhf_fno import MHFFNOWithAttention


def load_ns_velocity_data(data_path, n_train, n_test, time_steps=5):
    """加载 NS 速度场数据"""
    data = torch.load(data_path, weights_only=False)
    
    velocity = data['velocity']  # [N, T, 2, H, W]
    velocity = velocity[:n_train+n_test, :time_steps]
    
    # 分割
    train_velocity = velocity[:n_train]
    test_velocity = velocity[n_train:n_train+n_test]
    
    # 准备输入输出对：(u^t, u^{t+1})
    train_x = train_velocity[:, :-1].reshape(-1, 2, 64, 64)
    train_y = train_velocity[:, 1:].reshape(-1, 2, 64, 64)
    test_x = test_velocity[:, :-1].reshape(-1, 2, 64, 64)
    test_y = test_velocity[:, 1:].reshape(-1, 2, 64, 64)
    
    return train_x, train_y, test_x, test_y


def train_model(model, train_x, train_y, test_x, test_y, config, model_name):
    """训练模型"""
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    loss_fn = nn.MSELoss()
    
    best_test_loss = float('inf')
    
    print(f"\n训练 {model_name}...")
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    n_train = train_x.shape[0]
    batch_size = config['batch_size']
    
    for epoch in range(config['epochs']):
        model.train()
        perm = torch.randperm(n_train)
        train_loss = 0
        batch_count = 0
        
        for i in range(0, n_train, batch_size):
            bx = train_x[perm[i:i+batch_size]]
            by = train_y[perm[i:i+batch_size]]
            
            optimizer.zero_grad()
            pred = model(bx)
            loss = loss_fn(pred, by)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            batch_count += 1
        
        # 测试
        model.eval()
        with torch.no_grad():
            test_pred = model(test_x)
            test_loss = loss_fn(test_pred, test_y).item()
            
            if test_loss < best_test_loss:
                best_test_loss = test_loss
        
        if (epoch + 1) % 2 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{config['epochs']}: Train {train_loss/batch_count:.4f}, Test {test_loss:.4f}")
    
    return best_test_loss


def main():
    print("=" * 70)
    print("MHF+CoDA+PINO 快速测试")
    print("=" * 70)
    
    config = {
        'n_train': 50,
        'n_test': 20,
        'time_steps': 5,
        'epochs': 10,
        'batch_size': 8,
        'learning_rate': 0.0001,
    }
    
    # 加载数据
    data_path = Path(__file__).parent / 'data' / 'ns_real_velocity.pt'
    print(f"\n加载数据: {data_path}")
    train_x, train_y, test_x, test_y = load_ns_velocity_data(
        data_path,
        config['n_train'],
        config['n_test'],
        config['time_steps']
    )
    print(f"训练集: {train_x.shape}")
    print(f"测试集: {test_x.shape}")
    
    # 测试 MHF+CoDA
    print("\n" + "=" * 60)
    print("MHF+CoDA（带跨头注意力）")
    print("=" * 60)
    model = MHFFNOWithAttention.best_config(
        n_modes=(16, 16),
        hidden_channels=32,
        in_channels=2,
        out_channels=2
    )
    loss = train_model(model, train_x, train_y, test_x, test_y, config, "MHF+CoDA")
    print(f"\n✓ MHF+CoDA 最佳测试损失: {loss:.4f}")
    
    # 保存结果
    results = {
        'test': 'MHF+CoDA on real NS velocity data (quick)',
        'config': config,
        'mhf_coda_loss': loss,
        'success': True
    }
    
    output_path = Path(__file__).parent.parent / 'mhf_coda_quick_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ 结果已保存: {output_path}")


if __name__ == "__main__":
    main()
