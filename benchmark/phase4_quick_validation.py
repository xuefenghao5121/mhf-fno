#!/usr/bin/env python3
"""
Phase 4: 快速验证测试 (减少 epochs)

快速验证架构师建议的效果
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
    
    return best_test_loss, best_epoch


def main():
    device = torch.device('cpu')
    
    print("=" * 70, flush=True)
    print("Phase 4: 快速验证测试 (50 epochs)", flush=True)
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
    
    # 配置
    n_modes = (12, 12)
    hidden_channels = 32
    n_heads = 4
    batch_size = 16
    lr = 1e-3
    epochs = 50  # 减少 epochs
    
    results = {}
    
    # 基准 FNO
    print("\n[1/4] 测试 FNO 基准...", flush=True)
    torch.manual_seed(42)
    model_fno = FNO(n_modes=n_modes, hidden_channels=hidden_channels, in_channels=1, out_channels=1, n_layers=3).to(device)
    t0 = time.time()
    fno_loss, fno_epoch = train_and_eval(model_fno, train_x, train_y, test_x, test_y, epochs, batch_size, lr, device)
    print(f"  FNO: {fno_loss:.6f} @ {fno_epoch}, time={time.time()-t0:.1f}s", flush=True)
    results['FNO'] = {'loss': fno_loss, 'epoch': fno_epoch}
    
    # 基准 MHF-FNO [0,2]
    print("\n[2/4] 测试 MHF-FNO 基准...", flush=True)
    torch.manual_seed(42)
    model_mhf = create_hybrid_fno(n_modes, hidden_channels, n_heads=n_heads, mhf_layers=[0, 2]).to(device)
    t0 = time.time()
    mhf_loss, mhf_epoch = train_and_eval(model_mhf, train_x, train_y, test_x, test_y, epochs, batch_size, lr, device)
    print(f"  MHF-FNO: {mhf_loss:.6f} @ {mhf_epoch}, time={time.time()-t0:.1f}s", flush=True)
    results['MHF-FNO'] = {'loss': mhf_loss, 'epoch': mhf_epoch}
    
    # 全层 CoDA [0,1,2]
    print("\n[3/4] 测试全层 CoDA...", flush=True)
    torch.manual_seed(42)
    model_coda = create_mhf_fno_v2(n_modes, hidden_channels, n_heads=n_heads, attention_type='coda', mhf_layers=[0, 1, 2]).to(device)
    t0 = time.time()
    coda_loss, coda_epoch = train_and_eval(model_coda, train_x, train_y, test_x, test_y, epochs, batch_size, lr, device)
    print(f"  全层CoDA: {coda_loss:.6f} @ {coda_epoch}, time={time.time()-t0:.1f}s", flush=True)
    results['CoDA-Full'] = {'loss': coda_loss, 'epoch': coda_epoch}
    
    # CoDA bottleneck 调优
    print("\n[4/4] 测试 CoDA bottleneck 调优...", flush=True)
    for bn in [2, 4, 6]:
        # 创建带自定义 bottleneck 的模型
        model_bn = FNO(n_modes=n_modes, hidden_channels=hidden_channels, in_channels=1, out_channels=1, n_layers=3)
        
        from mhf_fno.mhf_attention_v2 import MHFSpectralConvV2
        
        for layer_idx in [0, 1, 2]:
            class CustomCoDAConv(MHFSpectralConvV2):
                def __init__(self, *args, bottleneck=4, **kwargs):
                    super().__init__(*args, attention_type='coda', **kwargs)
                    if self.use_attention:
                        self.attention = CoDAStyleAttention(
                            n_heads=self.n_heads,
                            head_dim=self.head_out,
                            bottleneck=bottleneck
                        )
            
            mhf_conv = CustomCoDAConv(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                n_modes=n_modes,
                n_heads=n_heads,
                bottleneck=bn
            )
            model_bn.fno_blocks.convs[layer_idx] = mhf_conv
        
        model_bn = model_bn.to(device)
        torch.manual_seed(42)
        bn_loss, bn_epoch = train_and_eval(model_bn, train_x, train_y, test_x, test_y, epochs, batch_size, lr, device)
        print(f"  CoDA-bn{bn}: {bn_loss:.6f} @ {bn_epoch}", flush=True)
        results[f'CoDA-bn{bn}'] = {'loss': bn_loss, 'epoch': bn_epoch, 'bottleneck': bn}
    
    # 汇总
    print("\n" + "=" * 70, flush=True)
    print("汇总结果", flush=True)
    print("=" * 70, flush=True)
    
    print(f"\n{'配置':<15} {'Loss':<12} {'Epoch':<8} {'vs FNO':<10}", flush=True)
    print("-" * 50, flush=True)
    
    for name, r in results.items():
        diff = (r['loss'] - fno_loss) / fno_loss * 100
        print(f"{name:<15} {r['loss']:<12.6f} {r['epoch']:<8} {diff:+.2f}%", flush=True)
    
    # 最佳配置
    best_name = min(results.keys(), key=lambda x: results[x]['loss'])
    best_loss = results[best_name]['loss']
    best_diff = (best_loss - fno_loss) / fno_loss * 100
    
    print(f"\n最佳配置: {best_name}", flush=True)
    print(f"最佳 Loss: {best_loss:.6f}", flush=True)
    print(f"vs FNO: {best_diff:+.2f}%", flush=True)
    
    # 保存
    output = {
        'timestamp': datetime.now().isoformat(),
        'epochs': epochs,
        'results': results,
        'summary': {
            'best_config': best_name,
            'best_loss': best_loss,
            'vs_fno_percent': best_diff
        }
    }
    
    output_path = Path(__file__).parent.parent / 'results' / 'phase4_quick_validation.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n结果已保存: {output_path}", flush=True)


if __name__ == '__main__':
    main()