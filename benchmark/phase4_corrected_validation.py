#!/usr/bin/env python3
"""
Phase 4: 修正版测试

发现：全层 CoDA [0,1,2] 效果变差，需要调整测试方案

新测试：
1. 基准: MHF-FNO [0,2]
2. 首尾层 CoDA [0,2] (不是全层)
3. 中间层单独测试
4. 100 epochs 完整训练
"""

import sys
import time
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from neuralop.models import FNO

sys.path.insert(0, str(Path(__file__).parent.parent))

from mhf_fno import create_hybrid_fno
from mhf_fno.mhf_attention_v2 import create_mhf_fno_v2
from mhf_fno.attention_variants import CoDAStyleAttention


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def train_and_eval(model, train_x, train_y, test_x, test_y, epochs, batch_size, lr, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.MSELoss()
    
    best_test_loss = float('inf')
    best_epoch = 0
    
    for epoch in range(epochs):
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
            print(f"  Epoch {epoch+1}/{epochs}: {test_loss:.6f} (best: {best_test_loss:.6f} @ {best_epoch})", flush=True)
    
    return best_test_loss, best_epoch


def main():
    device = torch.device('cpu')
    
    print("=" * 70, flush=True)
    print("Phase 4: 修正版测试 (100 epochs)", flush=True)
    print("=" * 70, flush=True)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    
    # 加载数据
    data_path = Path(__file__).parent.parent / 'data'
    train_data = torch.load(data_path / 'ns_train_32.pt', weights_only=False, map_location='cpu')
    test_data = torch.load(data_path / 'ns_test_32.pt', weights_only=False, map_location='cpu')
    
    train_x = train_data['x'].unsqueeze(1).float()
    train_y = train_data['y'].unsqueeze(1).float()
    test_x = test_data['x'].unsqueeze(1).float()
    test_y = test_data['y'].unsqueeze(1).float()
    
    train_x = (train_x - train_x.mean(dim=(-2, -1), keepdim=True)) / (train_x.std(dim=(-2, -1), keepdim=True) + 1e-8)
    train_y = (train_y - train_y.mean(dim=(-2, -1), keepdim=True)) / (train_y.std(dim=(-2, -1), keepdim=True) + 1e-8)
    test_x = (test_x - test_x.mean(dim=(-2, -1), keepdim=True)) / (test_x.std(dim=(-2, -1), keepdim=True) + 1e-8)
    test_y = (test_y - test_y.mean(dim=(-2, -1), keepdim=True)) / (test_y.std(dim=(-2, -1), keepdim=True) + 1e-8)
    
    print(f"数据: train={train_x.shape}, test={test_x.shape}", flush=True)
    
    n_modes = (12, 12)
    hidden_channels = 32
    n_heads = 4
    batch_size = 16
    lr = 1e-3
    epochs = 100
    
    results = {}
    
    # 1. FNO 基准
    print("\n[1/5] FNO 基准...", flush=True)
    torch.manual_seed(42)
    model = FNO(n_modes=n_modes, hidden_channels=hidden_channels, in_channels=1, out_channels=1, n_layers=3).to(device)
    t0 = time.time()
    loss, epoch = train_and_eval(model, train_x, train_y, test_x, test_y, epochs, batch_size, lr, device)
    results['FNO'] = {'loss': loss, 'epoch': epoch, 'time': time.time()-t0, 'params': count_parameters(model)}
    print(f"  => {loss:.6f} @ {epoch}", flush=True)
    
    # 2. MHF-FNO [0,2] (基准)
    print("\n[2/5] MHF-FNO [0,2] 基准...", flush=True)
    torch.manual_seed(42)
    model = create_hybrid_fno(n_modes, hidden_channels, n_heads=n_heads, mhf_layers=[0, 2]).to(device)
    t0 = time.time()
    loss, epoch = train_and_eval(model, train_x, train_y, test_x, test_y, epochs, batch_size, lr, device)
    results['MHF-[0,2]'] = {'loss': loss, 'epoch': epoch, 'time': time.time()-t0, 'params': count_parameters(model)}
    print(f"  => {loss:.6f} @ {epoch}", flush=True)
    
    # 3. CoDA [0,2] (首尾层，不是全层!)
    print("\n[3/5] CoDA [0,2] (首尾层)...", flush=True)
    torch.manual_seed(42)
    model = create_mhf_fno_v2(n_modes, hidden_channels, n_heads=n_heads, attention_type='coda', mhf_layers=[0, 2]).to(device)
    t0 = time.time()
    loss, epoch = train_and_eval(model, train_x, train_y, test_x, test_y, epochs, batch_size, lr, device)
    results['CoDA-[0,2]'] = {'loss': loss, 'epoch': epoch, 'time': time.time()-t0, 'params': count_parameters(model)}
    print(f"  => {loss:.6f} @ {epoch}", flush=True)
    
    # 4. MHF-FNO [0,1,2] (全层 MHF，无注意力)
    print("\n[4/5] MHF-FNO [0,1,2] (全层 MHF)...", flush=True)
    torch.manual_seed(42)
    model = create_hybrid_fno(n_modes, hidden_channels, n_heads=n_heads, mhf_layers=[0, 1, 2]).to(device)
    t0 = time.time()
    loss, epoch = train_and_eval(model, train_x, train_y, test_x, test_y, epochs, batch_size, lr, device)
    results['MHF-[0,1,2]'] = {'loss': loss, 'epoch': epoch, 'time': time.time()-t0, 'params': count_parameters(model)}
    print(f"  => {loss:.6f} @ {epoch}", flush=True)
    
    # 5. 更大 hidden_channels (64)
    print("\n[5/5] MHF-FNO [0,2] + hidden=64...", flush=True)
    torch.manual_seed(42)
    model = create_hybrid_fno(n_modes, 64, n_heads=4, mhf_layers=[0, 2]).to(device)
    t0 = time.time()
    loss, epoch = train_and_eval(model, train_x, train_y, test_x, test_y, epochs, batch_size, lr, device)
    results['MHF-64ch'] = {'loss': loss, 'epoch': epoch, 'time': time.time()-t0, 'params': count_parameters(model)}
    print(f"  => {loss:.6f} @ {epoch}", flush=True)
    
    # 汇总
    print("\n" + "=" * 70, flush=True)
    print("汇总结果", flush=True)
    print("=" * 70, flush=True)
    
    fno_loss = results['FNO']['loss']
    
    print(f"\n{'配置':<15} {'参数':<10} {'Loss':<12} {'Epoch':<6} {'vs FNO':<10}", flush=True)
    print("-" * 60, flush=True)
    
    for name, r in results.items():
        diff = (r['loss'] - fno_loss) / fno_loss * 100
        print(f"{name:<15} {r['params']:<10,} {r['loss']:<12.6f} {r['epoch']:<6} {diff:+.2f}%", flush=True)
    
    # 最佳
    best_name = min(results.keys(), key=lambda x: results[x]['loss'])
    best = results[best_name]
    best_diff = (best['loss'] - fno_loss) / fno_loss * 100
    
    print(f"\n最佳配置: {best_name}", flush=True)
    print(f"最佳 Loss: {best['loss']:.6f}", flush=True)
    print(f"vs FNO: {best_diff:+.2f}%", flush=True)
    
    # 目标检查
    target = -10.0
    if best_diff <= target:
        print(f"\n✅ 目标达成! ({best_diff:+.2f}% ≤ {target}%)", flush=True)
    else:
        gap = best_diff - target
        print(f"\n❌ 目标未达成，还差 {gap:.2f}%", flush=True)
    
    # 保存
    output = {
        'timestamp': datetime.now().isoformat(),
        'results': results,
        'summary': {
            'best_config': best_name,
            'best_loss': best['loss'],
            'vs_fno_percent': best_diff,
            'target_met': best_diff <= target
        }
    }
    
    output_path = Path(__file__).parent.parent / 'results' / 'phase4_corrected_validation.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n结果已保存: {output_path}", flush=True)


if __name__ == '__main__':
    main()