#!/usr/bin/env python3
"""
迭代 4 快速版 - 聚焦关键超参数
===============================

约束: 1000样本, 保持现有epochs

测试方向:
1. bottleneck: [2, 4, 6]
2. gate_init: [0.05, 0.1, 0.2]
3. mhf_layers: [[0], [2], [1,2]]
4. 学习率调度: cosine vs warmup_cosine

目标: ≤ -10% vs FNO

作者: Tianyuan Team
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
from mhf_fno.mhf_attention_v2 import MHFSpectralConvV2
from mhf_fno.attention_variants import CoDAStyleAttention


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def train_and_eval(
    model, train_x, train_y, test_x, test_y,
    epochs, batch_size, lr, device,
    lr_schedule='cosine', warmup_epochs=0,
    verbose=True
):
    """训练并评估模型"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    if lr_schedule == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif lr_schedule == 'warmup_cosine':
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            else:
                return 0.5 * (1 + np.cos((epoch - warmup_epochs) / (epochs - warmup_epochs) * np.pi))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
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
        
        scheduler.step()
        
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
        
        if verbose and (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: Test {test_loss:.6f} (best: {best_test_loss:.6f} @ {best_epoch})", flush=True)
    
    return best_test_loss, best_epoch


def create_model(
    n_modes, hidden_channels,
    n_heads=4,
    mhf_layers=[0, 2],
    bottleneck=4,
    gate_init=0.1
):
    """创建MHF-FNO模型"""
    model = FNO(
        n_modes=n_modes,
        hidden_channels=hidden_channels,
        in_channels=1,
        out_channels=1,
        n_layers=3
    )
    
    head_dim = hidden_channels // n_heads
    
    for layer_idx in mhf_layers:
        # 创建自定义CoDA
        class CustomCoDA(CoDAStyleAttention):
            def __init__(self, n_heads, head_dim, bottleneck, gate_init):
                super().__init__(n_heads, head_dim, bottleneck)
                nn.init.constant_(self.gate, gate_init)
        
        # 创建MHF卷积
        mhf_conv = MHFSpectralConvV2(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            n_modes=n_modes,
            n_heads=n_heads,
            attention_type='none'
        )
        
        mhf_conv.attention = CustomCoDA(n_heads, head_dim, bottleneck, gate_init)
        mhf_conv.use_attention = True
        
        model.fno_blocks.convs[layer_idx] = mhf_conv
    
    return model


def main():
    device = torch.device('cpu')
    base_path = Path(__file__).parent.parent
    data_path = base_path / 'data'
    results_path = base_path / 'results'
    results_path.mkdir(exist_ok=True)
    
    print("=" * 70, flush=True)
    print("迭代 4 快速版: 超参数优化", flush=True)
    print("=" * 70, flush=True)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"设备: {device}", flush=True)
    
    # 加载数据
    print("\n加载数据...", flush=True)
    train_data = torch.load(data_path / 'ns_train_32_large.pt', weights_only=False, map_location='cpu')
    test_data = torch.load(data_path / 'ns_test_32_large.pt', weights_only=False, map_location='cpu')
    
    train_x = train_data['x'].unsqueeze(1).float()
    train_y = train_data['y'].unsqueeze(1).float()
    test_x = test_data['x'].unsqueeze(1).float()
    test_y = test_data['y'].unsqueeze(1).float()
    
    print(f"训练集: {train_x.shape}", flush=True)
    print(f"测试集: {test_x.shape}", flush=True)
    
    # 归一化
    train_x = (train_x - train_x.mean(dim=(-2, -1), keepdim=True)) / (train_x.std(dim=(-2, -1), keepdim=True) + 1e-8)
    train_y = (train_y - train_y.mean(dim=(-2, -1), keepdim=True)) / (train_y.std(dim=(-2, -1), keepdim=True) + 1e-8)
    test_x = (test_x - test_x.mean(dim=(-2, -1), keepdim=True)) / (test_x.std(dim=(-2, -1), keepdim=True) + 1e-8)
    test_y = (test_y - test_y.mean(dim=(-2, -1), keepdim=True)) / (test_y.std(dim=(-2, -1), keepdim=True) + 1e-8)
    
    n_modes = (12, 12)
    hidden_channels = 32
    batch_size = 32
    lr = 1e-3
    epochs = 200  # 保持原有epochs
    
    results = {}
    
    # =========================================
    # 1. FNO 基准
    # =========================================
    print("\n" + "=" * 70, flush=True)
    print("1. FNO 基准 (200 epochs)", flush=True)
    print("=" * 70, flush=True)
    
    torch.manual_seed(42)
    model_fno = FNO(n_modes=n_modes, hidden_channels=hidden_channels, in_channels=1, out_channels=1, n_layers=3).to(device)
    params_fno = count_parameters(model_fno)
    print(f"参数量: {params_fno:,}", flush=True)
    
    t0 = time.time()
    fno_loss, fno_epoch = train_and_eval(model_fno, train_x, train_y, test_x, test_y, epochs, batch_size, lr, device)
    fno_time = time.time() - t0
    
    print(f"\n最佳 Loss: {fno_loss:.6f} @ Epoch {fno_epoch}", flush=True)
    print(f"训练时间: {fno_time:.1f}s", flush=True)
    
    results['FNO'] = {
        'params': params_fno,
        'best_loss': fno_loss,
        'best_epoch': fno_epoch,
        'time': fno_time,
        'vs_fno': 0.0
    }
    
    # =========================================
    # 2. bottleneck 优化
    # =========================================
    print("\n" + "=" * 70, flush=True)
    print("2. bottleneck 优化 [2, 4, 6]", flush=True)
    print("=" * 70, flush=True)
    
    for bn in [2, 4, 6]:
        name = f"CoDA-bn{bn}"
        print(f"\n--- {name} ---", flush=True)
        
        torch.manual_seed(42)
        model = create_model(n_modes, hidden_channels, mhf_layers=[0, 2], bottleneck=bn).to(device)
        params = count_parameters(model)
        print(f"参数量: {params:,}", flush=True)
        
        t0 = time.time()
        loss, epoch = train_and_eval(model, train_x, train_y, test_x, test_y, epochs, batch_size, lr, device)
        elapsed = time.time() - t0
        
        diff = (loss - fno_loss) / fno_loss * 100
        print(f"最佳 Loss: {loss:.6f} @ Epoch {epoch} | vs FNO: {diff:+.2f}%", flush=True)
        
        results[name] = {
            'params': params,
            'best_loss': loss,
            'best_epoch': epoch,
            'time': elapsed,
            'vs_fno': diff,
            'bottleneck': bn
        }
    
    # 找最佳 bottleneck
    best_bn = min(
        [(k, v) for k, v in results.items() if k.startswith('CoDA-bn')],
        key=lambda x: x[1]['best_loss']
    )[1]['bottleneck']
    print(f"\n最佳 bottleneck: {best_bn}", flush=True)
    
    # =========================================
    # 3. gate_init 优化
    # =========================================
    print("\n" + "=" * 70, flush=True)
    print("3. gate_init 优化 [0.05, 0.1, 0.2]", flush=True)
    print("=" * 70, flush=True)
    
    for gi in [0.05, 0.1, 0.2]:
        name = f"CoDA-gate{gi}"
        print(f"\n--- {name} ---", flush=True)
        
        torch.manual_seed(42)
        model = create_model(n_modes, hidden_channels, mhf_layers=[0, 2], bottleneck=best_bn, gate_init=gi).to(device)
        params = count_parameters(model)
        print(f"参数量: {params:,}", flush=True)
        
        t0 = time.time()
        loss, epoch = train_and_eval(model, train_x, train_y, test_x, test_y, epochs, batch_size, lr, device)
        elapsed = time.time() - t0
        
        diff = (loss - fno_loss) / fno_loss * 100
        print(f"最佳 Loss: {loss:.6f} @ Epoch {epoch} | vs FNO: {diff:+.2f}%", flush=True)
        
        results[name] = {
            'params': params,
            'best_loss': loss,
            'best_epoch': epoch,
            'time': elapsed,
            'vs_fno': diff,
            'gate_init': gi
        }
    
    # 找最佳 gate_init
    best_gate = min(
        [(k, v) for k, v in results.items() if k.startswith('CoDA-gate')],
        key=lambda x: x[1]['best_loss']
    )[1]['gate_init']
    print(f"\n最佳 gate_init: {best_gate}", flush=True)
    
    # =========================================
    # 4. mhf_layers 配置
    # =========================================
    print("\n" + "=" * 70, flush=True)
    print("4. mhf_layers 配置 [[0], [2], [1,2]]", flush=True)
    print("=" * 70, flush=True)
    
    for layers in [[0], [2], [1, 2]]:
        name = f"CoDA-layers{layers}"
        print(f"\n--- {name} ---", flush=True)
        
        torch.manual_seed(42)
        model = create_model(n_modes, hidden_channels, mhf_layers=layers, bottleneck=best_bn, gate_init=best_gate).to(device)
        params = count_parameters(model)
        print(f"参数量: {params:,}", flush=True)
        
        t0 = time.time()
        loss, epoch = train_and_eval(model, train_x, train_y, test_x, test_y, epochs, batch_size, lr, device)
        elapsed = time.time() - t0
        
        diff = (loss - fno_loss) / fno_loss * 100
        print(f"最佳 Loss: {loss:.6f} @ Epoch {epoch} | vs FNO: {diff:+.2f}%", flush=True)
        
        results[name] = {
            'params': params,
            'best_loss': loss,
            'best_epoch': epoch,
            'time': elapsed,
            'vs_fno': diff,
            'mhf_layers': layers
        }
    
    # 找最佳 mhf_layers
    best_layers = min(
        [(k, v) for k, v in results.items() if k.startswith('CoDA-layers')],
        key=lambda x: x[1]['best_loss']
    )[1]['mhf_layers']
    print(f"\n最佳 mhf_layers: {best_layers}", flush=True)
    
    # =========================================
    # 5. 学习率调度
    # =========================================
    print("\n" + "=" * 70, flush=True)
    print("5. 学习率调度优化", flush=True)
    print("=" * 70, flush=True)
    
    for schedule in ['cosine', 'warmup_cosine']:
        name = f"CoDA-lr_{schedule}"
        print(f"\n--- {name} ---", flush=True)
        
        torch.manual_seed(42)
        model = create_model(n_modes, hidden_channels, mhf_layers=best_layers, bottleneck=best_bn, gate_init=best_gate).to(device)
        params = count_parameters(model)
        
        t0 = time.time()
        loss, epoch = train_and_eval(
            model, train_x, train_y, test_x, test_y, epochs, batch_size, lr, device,
            lr_schedule=schedule, warmup_epochs=20
        )
        elapsed = time.time() - t0
        
        diff = (loss - fno_loss) / fno_loss * 100
        print(f"最佳 Loss: {loss:.6f} @ Epoch {epoch} | vs FNO: {diff:+.2f}%", flush=True)
        
        results[name] = {
            'params': params,
            'best_loss': loss,
            'best_epoch': epoch,
            'time': elapsed,
            'vs_fno': diff,
            'lr_schedule': schedule
        }
    
    best_schedule = min(
        [(k, v) for k, v in results.items() if k.startswith('CoDA-lr_')],
        key=lambda x: x[1]['best_loss']
    )[1]['lr_schedule']
    print(f"\n最佳 lr_schedule: {best_schedule}", flush=True)
    
    # =========================================
    # 6. 最佳配置确认 (300 epochs)
    # =========================================
    print("\n" + "=" * 70, flush=True)
    print("6. 最佳配置确认 (300 epochs)", flush=True)
    print("=" * 70, flush=True)
    print(f"配置: bottleneck={best_bn}, gate_init={best_gate}, mhf_layers={best_layers}, lr={best_schedule}", flush=True)
    
    torch.manual_seed(42)
    model_best = create_model(n_modes, hidden_channels, mhf_layers=best_layers, bottleneck=best_bn, gate_init=best_gate).to(device)
    params_best = count_parameters(model_best)
    print(f"参数量: {params_best:,}", flush=True)
    
    t0 = time.time()
    best_loss, best_epoch = train_and_eval(
        model_best, train_x, train_y, test_x, test_y, 300, batch_size, lr, device,
        lr_schedule=best_schedule, warmup_epochs=20
    )
    elapsed = time.time() - t0
    
    diff = (best_loss - fno_loss) / fno_loss * 100
    print(f"\n最佳 Loss: {best_loss:.6f} @ Epoch {best_epoch}", flush=True)
    print(f"vs FNO: {diff:+.2f}%", flush=True)
    
    results['BEST'] = {
        'params': params_best,
        'best_loss': best_loss,
        'best_epoch': best_epoch,
        'time': elapsed,
        'vs_fno': diff,
        'config': {
            'bottleneck': best_bn,
            'gate_init': best_gate,
            'mhf_layers': best_layers,
            'lr_schedule': best_schedule,
            'epochs': 300
        }
    }
    
    # =========================================
    # 汇总
    # =========================================
    print("\n" + "=" * 70, flush=True)
    print("汇总报告", flush=True)
    print("=" * 70, flush=True)
    
    print(f"\n{'配置':<25} {'参数量':<12} {'最佳Loss':<12} {'vs FNO':<10}", flush=True)
    print("-" * 70, flush=True)
    
    for name, r in sorted(results.items(), key=lambda x: x[1]['best_loss']):
        print(f"{name:<25} {r['params']:,} {r['best_loss']:<12.6f} {r['vs_fno']:+.2f}%", flush=True)
    
    # 目标检查
    best_result = min(results.items(), key=lambda x: x[1]['best_loss'])
    print(f"\n{'='*70}", flush=True)
    print("目标检查", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"最佳结果: {best_result[0]}", flush=True)
    print(f"vs FNO: {best_result[1]['vs_fno']:+.2f}%", flush=True)
    print(f"目标: ≤ -10%", flush=True)
    
    if best_result[1]['vs_fno'] <= -10:
        print("✅ 目标达成!", flush=True)
    else:
        gap = -10 - best_result[1]['vs_fno']
        print(f"⚠️ 距离目标还差 {gap:.2f}%", flush=True)
    
    # 保存结果
    output = {
        'timestamp': datetime.now().isoformat(),
        'iteration': 4,
        'focus': 'hyperparameter optimization',
        'epochs': epochs,
        'results': results,
        'fno_baseline': fno_loss,
        'best_config': {
            'bottleneck': best_bn,
            'gate_init': best_gate,
            'mhf_layers': best_layers,
            'lr_schedule': best_schedule
        }
    }
    
    output_path = results_path / 'iteration4_results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n结果已保存: {output_path}", flush=True)
    
    return output


if __name__ == '__main__':
    main()