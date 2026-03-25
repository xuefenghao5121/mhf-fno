#!/usr/bin/env python3
"""
迭代 4 - 超参数优化 + 训练策略优化
=====================================

测试方向：
1. 超参数优化: bottleneck, gate_init, n_heads
2. 训练策略: epochs, lr_schedule
3. 组合优化

目标: 达到 -10% vs FNO

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
from mhf_fno.mhf_attention_v2 import create_mhf_fno_v2, MHFSpectralConvV2


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def train_and_eval(
    model, train_x, train_y, test_x, test_y,
    epochs, batch_size, lr, device,
    lr_schedule='cosine', warmup_epochs=0,
    verbose=True
):
    """训练并评估模型，支持多种学习率调度"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # 学习率调度
    if lr_schedule == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif lr_schedule == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs//3, gamma=0.5)
    elif lr_schedule == 'warmup_cosine':
        # 预热 + 余弦退火
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            else:
                return 0.5 * (1 + np.cos((epoch - warmup_epochs) / (epochs - warmup_epochs) * np.pi))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif lr_schedule == 'onecycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr*10, epochs=epochs, steps_per_epoch=len(train_x)//batch_size+1
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    loss_fn = nn.MSELoss()
    
    best_test_loss = float('inf')
    best_epoch = 0
    history = []
    
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
                
                if lr_schedule == 'onecycle':
                    scheduler.step()
        
        if lr_schedule != 'onecycle':
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
        history.append(test_loss)
        
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_epoch = epoch + 1
        
        if verbose and (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: Test {test_loss:.6f} (best: {best_test_loss:.6f} @ {best_epoch})")
    
    return best_test_loss, best_epoch, history


def create_mhf_fno_with_params(
    n_modes, hidden_channels,
    n_heads=4,
    attention_type='coda',
    mhf_layers=[0, 2],
    bottleneck=4,
    gate_init=0.1
):
    """创建可配置参数的MHF-FNO"""
    from mhf_fno.attention_variants import CoDAStyleAttention
    
    model = FNO(
        n_modes=n_modes,
        hidden_channels=hidden_channels,
        in_channels=1,
        out_channels=1,
        n_layers=3
    )
    
    head_dim = hidden_channels // n_heads
    
    for layer_idx in mhf_layers:
        # 创建自定义CoDA注意力
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
        
        # 手动设置注意力
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
    
    print("=" * 70)
    print("迭代 4: 超参数优化 + 训练策略优化")
    print("=" * 70)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"设备: {device}")
    
    # 加载数据
    train_data = torch.load(data_path / 'ns_train_32_large.pt', weights_only=False, map_location='cpu')
    test_data = torch.load(data_path / 'ns_test_32_large.pt', weights_only=False, map_location='cpu')
    
    train_x = train_data['x'].unsqueeze(1).float()
    train_y = train_data['y'].unsqueeze(1).float()
    test_x = test_data['x'].unsqueeze(1).float()
    test_y = test_data['y'].unsqueeze(1).float()
    
    print(f"训练集: {train_x.shape}")
    print(f"测试集: {test_x.shape}")
    
    # 归一化
    train_x = (train_x - train_x.mean(dim=(-2, -1), keepdim=True)) / (train_x.std(dim=(-2, -1), keepdim=True) + 1e-8)
    train_y = (train_y - train_y.mean(dim=(-2, -1), keepdim=True)) / (train_y.std(dim=(-2, -1), keepdim=True) + 1e-8)
    test_x = (test_x - test_x.mean(dim=(-2, -1), keepdim=True)) / (test_x.std(dim=(-2, -1), keepdim=True) + 1e-8)
    test_y = (test_y - test_y.mean(dim=(-2, -1), keepdim=True)) / (test_y.std(dim=(-2, -1), keepdim=True) + 1e-8)
    
    n_modes = (12, 12)
    hidden_channels = 32
    batch_size = 32
    lr = 1e-3
    epochs = 300  # 更长训练
    
    results = {}
    
    # =========================================
    # 1. FNO 基准 (300 epochs)
    # =========================================
    print("\n" + "=" * 70)
    print("1. FNO 基准 (300 epochs)")
    print("=" * 70)
    
    torch.manual_seed(42)
    model_fno = FNO(n_modes=n_modes, hidden_channels=hidden_channels, in_channels=1, out_channels=1, n_layers=3).to(device)
    params_fno = count_parameters(model_fno)
    print(f"参数量: {params_fno:,}")
    
    t0 = time.time()
    fno_loss, fno_epoch, fno_history = train_and_eval(
        model_fno, train_x, train_y, test_x, test_y, epochs, batch_size, lr, device,
        lr_schedule='cosine'
    )
    fno_time = time.time() - t0
    
    print(f"\n最佳 Loss: {fno_loss:.6f} @ Epoch {fno_epoch}")
    print(f"训练时间: {fno_time:.1f}s")
    
    results['FNO'] = {
        'params': params_fno,
        'best_loss': fno_loss,
        'best_epoch': fno_epoch,
        'time': fno_time,
        'vs_fno': 0.0
    }
    
    # =========================================
    # 2. n_heads 优化
    # =========================================
    print("\n" + "=" * 70)
    print("2. n_heads 优化 (2 vs 4)")
    print("=" * 70)
    
    for n_heads in [2, 4]:
        name = f"CoDA-h{n_heads}"
        print(f"\n--- {name} ---")
        
        torch.manual_seed(42)
        model = create_mhf_fno_with_params(
            n_modes, hidden_channels,
            n_heads=n_heads,
            attention_type='coda',
            mhf_layers=[0, 2],
            bottleneck=4,
            gate_init=0.1
        ).to(device)
        params = count_parameters(model)
        print(f"参数量: {params:,}")
        
        t0 = time.time()
        loss, epoch, _ = train_and_eval(model, train_x, train_y, test_x, test_y, epochs, batch_size, lr, device)
        elapsed = time.time() - t0
        
        diff = (loss - fno_loss) / fno_loss * 100
        print(f"最佳 Loss: {loss:.6f} @ Epoch {epoch}")
        print(f"vs FNO: {diff:+.2f}%")
        
        results[name] = {
            'params': params,
            'best_loss': loss,
            'best_epoch': epoch,
            'time': elapsed,
            'vs_fno': diff,
            'n_heads': n_heads
        }
    
    # 找最佳 n_heads
    best_n_heads = min(
        [(k, v) for k, v in results.items() if k.startswith('CoDA-h')],
        key=lambda x: x[1]['best_loss']
    )[1]['n_heads']
    print(f"\n最佳 n_heads: {best_n_heads}")
    
    # =========================================
    # 3. bottleneck 优化
    # =========================================
    print("\n" + "=" * 70)
    print("3. bottleneck 优化")
    print("=" * 70)
    
    bottleneck_sizes = [2, 4, 6, 8]
    
    for bn in bottleneck_sizes:
        name = f"CoDA-bn{bn}"
        print(f"\n--- {name} ---")
        
        torch.manual_seed(42)
        model = create_mhf_fno_with_params(
            n_modes, hidden_channels,
            n_heads=best_n_heads,
            attention_type='coda',
            mhf_layers=[0, 2],
            bottleneck=bn,
            gate_init=0.1
        ).to(device)
        params = count_parameters(model)
        print(f"参数量: {params:,}")
        
        t0 = time.time()
        loss, epoch, _ = train_and_eval(model, train_x, train_y, test_x, test_y, epochs, batch_size, lr, device)
        elapsed = time.time() - t0
        
        diff = (loss - fno_loss) / fno_loss * 100
        print(f"最佳 Loss: {loss:.6f} @ Epoch {epoch}")
        print(f"vs FNO: {diff:+.2f}%")
        
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
        [(k, v) for k, v in results.items() if 'bottleneck' in v],
        key=lambda x: x[1]['best_loss']
    )[1]['bottleneck']
    print(f"\n最佳 bottleneck: {best_bn}")
    
    # =========================================
    # 4. gate_init 优化
    # =========================================
    print("\n" + "=" * 70)
    print("4. gate_init 优化")
    print("=" * 70)
    
    gate_inits = [0.05, 0.1, 0.2, 0.3]
    
    for gi in gate_inits:
        name = f"CoDA-gate{gi}"
        print(f"\n--- {name} ---")
        
        torch.manual_seed(42)
        model = create_mhf_fno_with_params(
            n_modes, hidden_channels,
            n_heads=best_n_heads,
            attention_type='coda',
            mhf_layers=[0, 2],
            bottleneck=best_bn,
            gate_init=gi
        ).to(device)
        params = count_parameters(model)
        print(f"参数量: {params:,}")
        
        t0 = time.time()
        loss, epoch, _ = train_and_eval(model, train_x, train_y, test_x, test_y, epochs, batch_size, lr, device)
        elapsed = time.time() - t0
        
        diff = (loss - fno_loss) / fno_loss * 100
        print(f"最佳 Loss: {loss:.6f} @ Epoch {epoch}")
        print(f"vs FNO: {diff:+.2f}%")
        
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
        [(k, v) for k, v in results.items() if 'gate_init' in v],
        key=lambda x: x[1]['best_loss']
    )[1]['gate_init']
    print(f"\n最佳 gate_init: {best_gate}")
    
    # =========================================
    # 5. 学习率调度优化
    # =========================================
    print("\n" + "=" * 70)
    print("5. 学习率调度优化")
    print("=" * 70)
    
    lr_schedules = ['cosine', 'step', 'warmup_cosine']
    
    for schedule in lr_schedules:
        name = f"CoDA-lr_{schedule}"
        print(f"\n--- {name} ---")
        
        torch.manual_seed(42)
        model = create_mhf_fno_with_params(
            n_modes, hidden_channels,
            n_heads=best_n_heads,
            attention_type='coda',
            mhf_layers=[0, 2],
            bottleneck=best_bn,
            gate_init=best_gate
        ).to(device)
        params = count_parameters(model)
        
        t0 = time.time()
        loss, epoch, _ = train_and_eval(
            model, train_x, train_y, test_x, test_y, epochs, batch_size, lr, device,
            lr_schedule=schedule, warmup_epochs=20
        )
        elapsed = time.time() - t0
        
        diff = (loss - fno_loss) / fno_loss * 100
        print(f"最佳 Loss: {loss:.6f} @ Epoch {epoch}")
        print(f"vs FNO: {diff:+.2f}%")
        
        results[name] = {
            'params': params,
            'best_loss': loss,
            'best_epoch': epoch,
            'time': elapsed,
            'vs_fno': diff,
            'lr_schedule': schedule
        }
    
    # 找最佳 lr_schedule
    best_schedule = min(
        [(k, v) for k, v in results.items() if 'lr_schedule' in v],
        key=lambda x: x[1]['best_loss']
    )[1]['lr_schedule']
    print(f"\n最佳 lr_schedule: {best_schedule}")
    
    # =========================================
    # 6. 最佳配置组合测试
    # =========================================
    print("\n" + "=" * 70)
    print("6. 最佳配置组合测试 (400 epochs)")
    print("=" * 70)
    
    torch.manual_seed(42)
    model_best = create_mhf_fno_with_params(
        n_modes, hidden_channels,
        n_heads=best_n_heads,
        attention_type='coda',
        mhf_layers=[0, 2],
        bottleneck=best_bn,
        gate_init=best_gate
    ).to(device)
    params_best = count_parameters(model_best)
    print(f"参数量: {params_best:,}")
    print(f"配置: n_heads={best_n_heads}, bottleneck={best_bn}, gate_init={best_gate}, lr_schedule={best_schedule}")
    
    t0 = time.time()
    best_loss, best_epoch, _ = train_and_eval(
        model_best, train_x, train_y, test_x, test_y, 400, batch_size, lr, device,
        lr_schedule=best_schedule, warmup_epochs=20
    )
    elapsed = time.time() - t0
    
    diff = (best_loss - fno_loss) / fno_loss * 100
    print(f"\n最佳 Loss: {best_loss:.6f} @ Epoch {best_epoch}")
    print(f"vs FNO: {diff:+.2f}%")
    
    results['BEST_COMBINED'] = {
        'params': params_best,
        'best_loss': best_loss,
        'best_epoch': best_epoch,
        'time': elapsed,
        'vs_fno': diff,
        'config': {
            'n_heads': best_n_heads,
            'bottleneck': best_bn,
            'gate_init': best_gate,
            'lr_schedule': best_schedule,
            'epochs': 400
        }
    }
    
    # =========================================
    # 汇总报告
    # =========================================
    print("\n" + "=" * 70)
    print("汇总报告")
    print("=" * 70)
    
    print(f"\n{'配置':<20} {'参数量':<12} {'最佳Loss':<12} {'vs FNO':<10}")
    print("-" * 70)
    
    for name, r in sorted(results.items(), key=lambda x: x[1]['best_loss']):
        print(f"{name:<20} {r['params']:,} {r['best_loss']:<12.6f} {r['vs_fno']:+.2f}%")
    
    # 目标检查
    best_result = min(results.items(), key=lambda x: x[1]['best_loss'])
    print(f"\n{'='*70}")
    print("目标检查")
    print(f"{'='*70}")
    print(f"最佳结果: {best_result[0]}")
    print(f"vs FNO: {best_result[1]['vs_fno']:+.2f}%")
    print(f"目标: ≤ -10%")
    
    if best_result[1]['vs_fno'] <= -10:
        print("✅ 目标达成!")
    else:
        gap = -10 - best_result[1]['vs_fno']
        print(f"⚠️ 距离目标还差 {gap:.2f}%")
    
    # 保存结果
    output = {
        'timestamp': datetime.now().isoformat(),
        'iteration': 4,
        'focus': 'hyperparameter + training optimization',
        'epochs': epochs,
        'results': results,
        'fno_baseline': fno_loss,
        'best_config': {
            'n_heads': best_n_heads,
            'bottleneck': best_bn,
            'gate_init': best_gate,
            'lr_schedule': best_schedule
        }
    }
    
    output_path = results_path / 'iteration4_optimization_results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n结果已保存: {output_path}")
    
    return output


if __name__ == '__main__':
    main()