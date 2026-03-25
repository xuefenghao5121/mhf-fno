#!/usr/bin/env python3
"""
扩展对比测试 - 更多 epochs 和不同配置
"""

import sys
import time
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from neuralop.models import FNO

sys.path.insert(0, str(Path(__file__).parent.parent))

from mhf_fno import create_hybrid_fno
from mhf_fno.mhf_attention_v2 import create_mhf_fno_v2


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def train_and_eval(model, train_x, train_y, test_x, test_y, epochs, batch_size, lr, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.MSELoss()
    
    best_test_loss = float('inf')
    best_epoch = 0
    
    for epoch in range(epochs):
        # Train
        model.train()
        perm = torch.randperm(train_x.shape[0], device=device)
        
        for i in range(0, train_x.shape[0], batch_size):
            idx = perm[i:i+batch_size]
            bx, by = train_x[idx].to(device), train_y[idx].to(device)
            
            optimizer.zero_grad()
            output = model(bx)
            loss = loss_fn(output, by)
            
            if not (torch.isnan(loss) or torch.isinf(loss)):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        
        # Eval
        model.eval()
        test_loss = 0
        n_test = 0
        with torch.no_grad():
            for i in range(0, test_x.shape[0], batch_size):
                bx, by = test_x[i:i+batch_size].to(device), test_y[i:i+batch_size].to(device)
                loss = loss_fn(model(bx), by)
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    test_loss += loss.item()
                    n_test += 1
        
        test_loss /= max(n_test, 1)
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_epoch = epoch + 1
        
        scheduler.step()
        
        if (epoch + 1) % 25 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: Test {test_loss:.6f} (best: {best_test_loss:.6f} @ {best_epoch})")
    
    return best_test_loss, best_epoch


def main():
    device = torch.device('cpu')
    print("扩展对比测试 (100 epochs)")
    print("="*60)
    
    # 加载数据
    data_path = Path('./data')
    train_data = torch.load(data_path / 'ns_train_32.pt', weights_only=False, map_location='cpu')
    test_data = torch.load(data_path / 'ns_test_32.pt', weights_only=False, map_location='cpu')
    
    train_x = train_data['x'].unsqueeze(1).float()
    train_y = train_data['y'].unsqueeze(1).float()
    test_x = test_data['x'].unsqueeze(1).float()
    test_y = test_data['y'].unsqueeze(1).float()
    
    # 归一化
    train_x = (train_x - train_x.mean(dim=(-2, -1), keepdim=True)) / (train_x.std(dim=(-2, -1), keepdim=True) + 1e-8)
    train_y = (train_y - train_y.mean(dim=(-2, -1), keepdim=True)) / (train_y.std(dim=(-2, -1), keepdim=True) + 1e-8)
    test_x = (test_x - test_x.mean(dim=(-2, -1), keepdim=True)) / (test_x.std(dim=(-2, -1), keepdim=True) + 1e-8)
    test_y = (test_y - test_y.mean(dim=(-2, -1), keepdim=True)) / (test_y.std(dim=(-2, -1), keepdim=True) + 1e-8)
    
    print(f"训练集: {train_x.shape}, 测试集: {test_x.shape}")
    
    # 配置
    n_modes = (12, 12)
    hidden_channels = 32
    n_heads = 4
    epochs = 100
    batch_size = 16
    lr = 1e-3
    
    results = {}
    
    # 只测试最有希望的变体
    test_variants = ['none', 'senet', 'coda']
    
    for attn_type in test_variants:
        print(f"\n{'='*50}")
        print(f"测试: {attn_type}")
        print(f"{'='*50}")
        
        torch.manual_seed(42)
        
        if attn_type == 'none':
            model = create_hybrid_fno(n_modes, hidden_channels, n_heads=n_heads).to(device)
        else:
            model = create_mhf_fno_v2(n_modes, hidden_channels, n_heads=n_heads, attention_type=attn_type).to(device)
        
        params = count_parameters(model)
        print(f"参数量: {params:,}")
        
        t0 = time.time()
        best_loss, best_epoch = train_and_eval(model, train_x, train_y, test_x, test_y, epochs, batch_size, lr, device)
        train_time = time.time() - t0
        
        print(f"\n最佳 Loss: {best_loss:.6f} @ Epoch {best_epoch}, 时间: {train_time:.1f}s")
        
        results[attn_type] = {
            'params': params,
            'best_loss': best_loss,
            'best_epoch': best_epoch,
            'time': train_time
        }
    
    # 汇总
    print(f"\n{'='*60}")
    print("最终汇总")
    print(f"{'='*60}")
    
    baseline = results['none']['best_loss']
    print(f"\n{'模型':<15} {'参数':<10} {'最佳Loss':<12} {'最佳Epoch':<10} {'vs基线':<10}")
    print("-" * 60)
    
    for attn_type, r in results.items():
        diff = (r['best_loss'] - baseline) / baseline * 100
        print(f"{attn_type:<15} {r['params']:<10,} {r['best_loss']:<12.6f} {r['best_epoch']:<10} {diff:+.2f}%")
    
    # 结论
    best_type = min(results.keys(), key=lambda x: results[x]['best_loss'])
    improvement = (baseline - results[best_type]['best_loss']) / baseline * 100
    
    print(f"\n{'='*60}")
    print(f"结论")
    print(f"{'='*60}")
    print(f"最佳方案: {best_type}")
    print(f"相比纯 MHF-FNO 改善: {improvement:.2f}%")
    
    # 保存
    output = {
        'timestamp': datetime.now().isoformat(),
        'results': results
    }
    with open('./results/extended_comparison.json', 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n结果已保存")


if __name__ == '__main__':
    main()