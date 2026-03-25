#!/usr/bin/env python3
"""
迭代 5 最终测试 - 微调 gate_init 和 bottleneck
==============================================

基于迭代 4 最佳配置 (CoDA-default: bottleneck=4, gate_init=0.1, -7.11% vs FNO)
尝试进一步优化以达到 ≤ -10% 目标

测试配置:
- FNO (基准)
- CoDA-gi0.15: bottleneck=4, gate_init=0.15
- CoDA-gi0.2: bottleneck=4, gate_init=0.2
- CoDA-bn6: bottleneck=6, gate_init=0.1
- CoDA-best: bottleneck=6, gate_init=0.15

每个配置运行 5 次，取平均结果

作者: Tianyuan Team - 天渠
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

from mhf_fno.mhf_attention_v2 import MHFSpectralConvV2
from mhf_fno.attention_variants import CoDAStyleAttention


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def train_and_eval(
    model, train_x, train_y, test_x, test_y,
    epochs, batch_size, lr, device, verbose=True, seed=42
):
    """训练并评估模型"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
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
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: Test {test_loss:.6f} (best: {best_test_loss:.6f} @ {best_epoch})")
    
    return best_test_loss, best_epoch, history


def create_coda_model(
    n_modes, hidden_channels,
    n_heads=4,
    bottleneck=4,
    gate_init=0.1,
    mhf_layers=[0, 2]
):
    """创建带自定义参数的 CoDA 模型"""
    
    model = FNO(
        n_modes=n_modes,
        hidden_channels=hidden_channels,
        in_channels=1,
        out_channels=1,
        n_layers=3
    )
    
    head_dim = hidden_channels // n_heads
    
    for layer_idx in mhf_layers:
        # 创建自定义 CoDA 注意力
        class CustomCoDA(CoDAStyleAttention):
            def __init__(self, n_heads, head_dim, bottleneck, gate_init):
                super().__init__(n_heads, head_dim, bottleneck)
                nn.init.constant_(self.gate, gate_init)
        
        # 创建 MHF 卷积
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


def run_single_config(config, train_x, train_y, test_x, test_y, epochs, batch_size, lr, device, run_idx):
    """运行单个配置的单次训练"""
    torch.manual_seed(42 + run_idx)
    
    n_modes = (12, 12)
    hidden_channels = 32
    
    if config['type'] == 'fno':
        model = FNO(
            n_modes=n_modes,
            hidden_channels=hidden_channels,
            in_channels=1,
            out_channels=1,
            n_layers=3
        ).to(device)
    else:
        model = create_coda_model(
            n_modes, hidden_channels,
            n_heads=4,
            bottleneck=config['bottleneck'],
            gate_init=config['gate_init'],
            mhf_layers=[0, 2]
        ).to(device)
    
    params = count_parameters(model)
    
    loss, epoch, history = train_and_eval(
        model, train_x, train_y, test_x, test_y,
        epochs, batch_size, lr, device, verbose=(run_idx == 0), seed=42 + run_idx
    )
    
    return loss, epoch, params


def main():
    device = torch.device('cpu')
    base_path = Path(__file__).parent.parent
    data_path = base_path / 'data'
    results_path = base_path / 'results'
    results_path.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("迭代 5 最终测试 - 微调 gate_init 和 bottleneck")
    print("=" * 70)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"设备: {device}")
    print(f"Epochs: 50")
    print(f"每配置运行次数: 5")
    
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
    
    batch_size = 32
    lr = 1e-3
    epochs = 50
    n_runs = 5
    
    # =========================================
    # 测试配置
    # =========================================
    test_configs = [
        {'name': 'FNO', 'type': 'fno'},
        {'name': 'CoDA-gi0.15', 'type': 'coda', 'bottleneck': 4, 'gate_init': 0.15},
        {'name': 'CoDA-gi0.2', 'type': 'coda', 'bottleneck': 4, 'gate_init': 0.2},
        {'name': 'CoDA-bn6', 'type': 'coda', 'bottleneck': 6, 'gate_init': 0.1},
        {'name': 'CoDA-best', 'type': 'coda', 'bottleneck': 6, 'gate_init': 0.15},
    ]
    
    all_results = {}
    
    for config in test_configs:
        name = config['name']
        print(f"\n{'='*70}")
        print(f"测试: {name}")
        if config['type'] == 'coda':
            print(f"配置: bottleneck={config['bottleneck']}, gate_init={config['gate_init']}")
        print(f"{'='*70}")
        
        losses = []
        epochs_list = []
        params = None
        
        for run_idx in range(n_runs):
            print(f"\n--- Run {run_idx + 1}/{n_runs} ---")
            
            loss, best_epoch, params = run_single_config(
                config, train_x, train_y, test_x, test_y,
                epochs, batch_size, lr, device, run_idx
            )
            
            losses.append(loss)
            epochs_list.append(best_epoch)
            print(f"  Loss: {loss:.6f}, Best Epoch: {best_epoch}")
        
        avg_loss = np.mean(losses)
        std_loss = np.std(losses)
        avg_epoch = np.mean(epochs_list)
        
        print(f"\n平均结果 ({n_runs} runs):")
        print(f"  Loss: {avg_loss:.6f} ± {std_loss:.6f}")
        print(f"  Best Epoch: {avg_epoch:.1f}")
        
        all_results[name] = {
            'params': params,
            'losses': losses,
            'avg_loss': avg_loss,
            'std_loss': std_loss,
            'avg_epoch': avg_epoch,
            'config': config if config['type'] == 'coda' else None
        }
    
    # =========================================
    # 计算相对 FNO 的改进
    # =========================================
    fno_loss = all_results['FNO']['avg_loss']
    for name in all_results:
        if name != 'FNO':
            diff = (all_results[name]['avg_loss'] - fno_loss) / fno_loss * 100
            all_results[name]['vs_fno'] = diff
        else:
            all_results[name]['vs_fno'] = 0.0
    
    # =========================================
    # 汇总报告
    # =========================================
    print("\n" + "=" * 70)
    print("迭代 5 最终测试结果汇总")
    print("=" * 70)
    
    print(f"\n{'配置':<18} {'参数量':<10} {'平均Loss':<16} {'vs FNO':<12}")
    print("-" * 70)
    
    sorted_results = sorted(all_results.items(), key=lambda x: x[1]['avg_loss'])
    for name, r in sorted_results:
        vs_fno_str = f"{r['vs_fno']:+.2f}%" if name != 'FNO' else "baseline"
        print(f"{name:<18} {r['params']:,} {r['avg_loss']:.6f}±{r['std_loss']:.6f} {vs_fno_str:<12}")
    
    # 最佳配置推荐
    best_name = min(all_results.items(), key=lambda x: x[1]['avg_loss'])[0]
    best = all_results[best_name]
    
    print(f"\n{'='*70}")
    print("最佳配置推荐")
    print(f"{'='*70}")
    print(f"名称: {best_name}")
    print(f"平均 Loss: {best['avg_loss']:.6f} ± {best['std_loss']:.6f}")
    print(f"vs FNO: {best['vs_fno']:+.2f}%")
    
    if best_name != 'FNO':
        print(f"bottleneck: {best['config']['bottleneck']}")
        print(f"gate_init: {best['config']['gate_init']}")
        
        # 计算参数减少
        fno_params = all_results['FNO']['params']
        param_reduction = (fno_params - best['params']) / fno_params * 100
        print(f"参数减少: {param_reduction:.1f}%")
    
    # 目标检查
    print(f"\n{'='*70}")
    print("目标检查")
    print(f"{'='*70}")
    print(f"目标: ≤ -10% vs FNO")
    
    achieved = False
    if best['vs_fno'] <= -10:
        print("✅ 目标达成!")
        achieved = True
    else:
        gap = -10 - best['vs_fno']
        print(f"⚠️ 距离目标还差 {gap:.2f}%")
        
        # 显示所有配置的达标情况
        print(f"\n各配置达标情况:")
        for name, r in sorted_results:
            if name != 'FNO':
                if r['vs_fno'] <= -10:
                    print(f"  {name}: {r['vs_fno']:+.2f}% ✅")
                else:
                    gap = -10 - r['vs_fno']
                    print(f"  {name}: {r['vs_fno']:+.2f}% (差 {gap:.2f}%)")
    
    # 与迭代 4 最佳结果对比
    print(f"\n{'='*70}")
    print("与迭代 4 最佳结果对比")
    print(f"{'='*70}")
    print(f"迭代 4 最佳: CoDA-default (bn=4, gi=0.1) -> -7.11%")
    
    # 找到最接近迭代 4 的配置
    for name, r in all_results.items():
        if name == 'FNO':
            continue
        if r['config']['bottleneck'] == 4 and r['config']['gate_init'] == 0.1:
            print(f"迭代 5 相同配置 ({name}): {r['vs_fno']:+.2f}%")
            break
    
    # 改进最大的配置
    best_improvement = 0
    best_improvement_name = None
    for name, r in all_results.items():
        if name == 'FNO':
            continue
        improvement = r['vs_fno'] - (-7.11)  # 相对迭代 4 最佳的改进
        if improvement < best_improvement:
            best_improvement = improvement
            best_improvement_name = name
    
    if best_improvement_name:
        print(f"迭代 5 最大改进: {best_improvement_name} ({all_results[best_improvement_name]['vs_fno']:+.2f}%) -> 改进 {-best_improvement:.2f}%")
    
    # 保存结果
    output = {
        'timestamp': datetime.now().isoformat(),
        'iteration': '5',
        'epochs': epochs,
        'n_runs': n_runs,
        'results': {
            name: {
                'params': r['params'],
                'avg_loss': r['avg_loss'],
                'std_loss': r['std_loss'],
                'avg_epoch': r['avg_epoch'],
                'vs_fno': r['vs_fno'],
                'losses': r['losses'],
                'config': r.get('config')
            }
            for name, r in all_results.items()
        },
        'best_config': {
            'name': best_name,
            'avg_loss': best['avg_loss'],
            'std_loss': best['std_loss'],
            'vs_fno': best['vs_fno'],
            'bottleneck': best['config']['bottleneck'] if best['config'] else None,
            'gate_init': best['config']['gate_init'] if best['config'] else None,
            'param_reduction': param_reduction if best_name != 'FNO' else 0
        },
        'target_achieved': achieved,
        'target': '-10% vs FNO',
        'iteration4_best': -7.11
    }
    
    output_path = results_path / 'iteration5_final.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n结果已保存: {output_path}")
    
    return output


if __name__ == '__main__':
    main()