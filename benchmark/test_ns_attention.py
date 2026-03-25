#!/usr/bin/env python3
"""
Navier-Stokes 数据集上的 MHF-FNO + 跨头注意力测试

对比三种模型:
1. FNO (基准)
2. MHF-FNO (原版)
3. MHF-FNO + Attention (新实现)

测试目标: 验证跨头注意力能否改善 NS 数据集上的效果
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

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from mhf_fno import create_hybrid_fno
from mhf_fno.mhf_attention import create_mhf_fno_with_attention


def count_parameters(model):
    """计算参数量"""
    return sum(p.numel() for p in model.parameters())


def train_epoch(model, train_x, train_y, optimizer, loss_fn, batch_size, device):
    """训练一个 epoch"""
    model.train()
    n_train = train_x.shape[0]
    perm = torch.randperm(n_train, device=device)
    total_loss = 0
    n_batches = 0
    
    for i in range(0, n_train, batch_size):
        idx = perm[i:i+batch_size]
        bx = train_x[idx].to(device)
        by = train_y[idx].to(device)
        
        optimizer.zero_grad()
        output = model(bx)
        loss = loss_fn(output, by)
        
        if torch.isnan(loss) or torch.isinf(loss):
            continue
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / max(n_batches, 1)


def evaluate(model, test_x, test_y, loss_fn, batch_size, device):
    """评估模型"""
    model.eval()
    n_test = test_x.shape[0]
    total_loss = 0
    n_batches = 0
    
    with torch.no_grad():
        for i in range(0, n_test, batch_size):
            bx = test_x[i:i+batch_size].to(device)
            by = test_y[i:i+batch_size].to(device)
            loss = loss_fn(model(bx), by)
            if not (torch.isnan(loss) or torch.isinf(loss)):
                total_loss += loss.item()
                n_batches += 1
    
    return total_loss / max(n_batches, 1)


def run_experiment(config):
    """运行完整实验"""
    
    device = torch.device(config.get('device', 'cpu'))
    print(f"使用设备: {device}")
    
    # 加载数据
    print(f"\n{'='*70}")
    print(f"加载 Navier-Stokes 数据集")
    print(f"{'='*70}")
    
    data_path = Path(config['data_dir'])
    train_data = torch.load(data_path / 'ns_train_32.pt', weights_only=False, map_location='cpu')
    test_data = torch.load(data_path / 'ns_test_32.pt', weights_only=False, map_location='cpu')
    
    train_x = train_data['x'].unsqueeze(1).float()  # [N, 1, H, W]
    train_y = train_data['y'].unsqueeze(1).float()
    test_x = test_data['x'].unsqueeze(1).float()
    test_y = test_data['y'].unsqueeze(1).float()
    
    # 数据归一化 (按样本)
    train_x = (train_x - train_x.mean(dim=(-2, -1), keepdim=True)) / (train_x.std(dim=(-2, -1), keepdim=True) + 1e-8)
    train_y = (train_y - train_y.mean(dim=(-2, -1), keepdim=True)) / (train_y.std(dim=(-2, -1), keepdim=True) + 1e-8)
    test_x = (test_x - test_x.mean(dim=(-2, -1), keepdim=True)) / (test_x.std(dim=(-2, -1), keepdim=True) + 1e-8)
    test_y = (test_y - test_y.mean(dim=(-2, -1), keepdim=True)) / (test_y.std(dim=(-2, -1), keepdim=True) + 1e-8)
    
    print(f"训练集: {train_x.shape}")
    print(f"测试集: {test_x.shape}")
    
    # 模型配置
    n_modes = (12, 12)  # 减小模式数
    hidden_channels = 32
    in_channels = 1
    out_channels = 1
    n_layers = 3
    n_heads = 4
    
    results = {
        'config': config,
        'timestamp': datetime.now().isoformat(),
        'dataset': {
            'train_samples': train_x.shape[0],
            'test_samples': test_x.shape[0],
            'resolution': f"{train_x.shape[-1]}x{train_x.shape[-2]}",
        },
        'models': {}
    }
    
    # 使用 MSE 损失函数 (更稳定)
    loss_fn = nn.MSELoss()
    
    # ========================================================================
    # 测试 1: FNO (基准)
    # ========================================================================
    print(f"\n{'='*70}")
    print(f"测试 1: FNO (基准)")
    print(f"{'='*70}")
    
    torch.manual_seed(config['seed'])
    model_fno = FNO(
        n_modes=n_modes,
        hidden_channels=hidden_channels,
        in_channels=in_channels,
        out_channels=out_channels,
        n_layers=n_layers,
    ).to(device)
    
    params_fno = count_parameters(model_fno)
    print(f"参数量: {params_fno:,}")
    
    optimizer = torch.optim.AdamW(model_fno.parameters(), lr=config['lr'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    
    fno_train_losses = []
    fno_test_losses = []
    
    t0 = time.time()
    for epoch in range(config['epochs']):
        train_loss = train_epoch(model_fno, train_x, train_y, optimizer, loss_fn, config['batch_size'], device)
        test_loss = evaluate(model_fno, test_x, test_y, loss_fn, config['batch_size'], device)
        scheduler.step()
        
        fno_train_losses.append(train_loss)
        fno_test_losses.append(test_loss)
        
        if (epoch + 1) % 25 == 0:
            print(f"  Epoch {epoch+1}/{config['epochs']}: Train {train_loss:.6f}, Test {test_loss:.6f}")
    
    fno_time = time.time() - t0
    fno_best_loss = min(fno_test_losses)
    
    print(f"\n最佳测试 Loss: {fno_best_loss:.6f}")
    print(f"训练时间: {fno_time:.1f}s")
    
    results['models']['FNO'] = {
        'parameters': params_fno,
        'best_test_loss': fno_best_loss,
        'final_train_loss': fno_train_losses[-1],
        'train_time_s': fno_time,
    }
    
    # ========================================================================
    # 测试 2: MHF-FNO (原版)
    # ========================================================================
    print(f"\n{'='*70}")
    print(f"测试 2: MHF-FNO (原版)")
    print(f"{'='*70}")
    
    torch.manual_seed(config['seed'])
    model_mhf = create_hybrid_fno(
        n_modes=n_modes,
        hidden_channels=hidden_channels,
        in_channels=in_channels,
        out_channels=out_channels,
        n_layers=n_layers,
        n_heads=n_heads,
        mhf_layers=[0, 2],
    ).to(device)
    
    params_mhf = count_parameters(model_mhf)
    print(f"参数量: {params_mhf:,} ({(1-params_mhf/params_fno)*100:.1f}% 减少)")
    
    optimizer = torch.optim.AdamW(model_mhf.parameters(), lr=config['lr'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    
    mhf_train_losses = []
    mhf_test_losses = []
    
    t0 = time.time()
    for epoch in range(config['epochs']):
        train_loss = train_epoch(model_mhf, train_x, train_y, optimizer, loss_fn, config['batch_size'], device)
        test_loss = evaluate(model_mhf, test_x, test_y, loss_fn, config['batch_size'], device)
        scheduler.step()
        
        mhf_train_losses.append(train_loss)
        mhf_test_losses.append(test_loss)
        
        if (epoch + 1) % 25 == 0:
            print(f"  Epoch {epoch+1}/{config['epochs']}: Train {train_loss:.6f}, Test {test_loss:.6f}")
    
    mhf_time = time.time() - t0
    mhf_best_loss = min(mhf_test_losses)
    
    print(f"\n最佳测试 Loss: {mhf_best_loss:.6f}")
    print(f"vs FNO: {(mhf_best_loss - fno_best_loss) / fno_best_loss * 100:+.1f}%")
    print(f"训练时间: {mhf_time:.1f}s")
    
    results['models']['MHF-FNO'] = {
        'parameters': params_mhf,
        'best_test_loss': mhf_best_loss,
        'final_train_loss': mhf_train_losses[-1],
        'train_time_s': mhf_time,
    }
    
    # ========================================================================
    # 测试 3: MHF-FNO + Attention (新实现)
    # ========================================================================
    print(f"\n{'='*70}")
    print(f"测试 3: MHF-FNO + Cross-Head Attention")
    print(f"{'='*70}")
    
    torch.manual_seed(config['seed'])
    model_attn = create_mhf_fno_with_attention(
        n_modes=n_modes,
        hidden_channels=hidden_channels,
        in_channels=in_channels,
        out_channels=out_channels,
        n_layers=n_layers,
        n_heads=n_heads,
        mhf_layers=[0, 2],
        attention_layers=[0, 2],
        attn_dropout=0.0,
    ).to(device)
    
    params_attn = count_parameters(model_attn)
    print(f"参数量: {params_attn:,} ({(1-params_attn/params_fno)*100:.1f}% vs FNO)")
    print(f"vs MHF-FNO: {(params_attn - params_mhf):+,} 参数增量")
    
    optimizer = torch.optim.AdamW(model_attn.parameters(), lr=config['lr'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    
    attn_train_losses = []
    attn_test_losses = []
    
    t0 = time.time()
    for epoch in range(config['epochs']):
        train_loss = train_epoch(model_attn, train_x, train_y, optimizer, loss_fn, config['batch_size'], device)
        test_loss = evaluate(model_attn, test_x, test_y, loss_fn, config['batch_size'], device)
        scheduler.step()
        
        attn_train_losses.append(train_loss)
        attn_test_losses.append(test_loss)
        
        if (epoch + 1) % 25 == 0:
            print(f"  Epoch {epoch+1}/{config['epochs']}: Train {train_loss:.6f}, Test {test_loss:.6f}")
    
    attn_time = time.time() - t0
    attn_best_loss = min(attn_test_losses)
    
    print(f"\n最佳测试 Loss: {attn_best_loss:.6f}")
    print(f"vs FNO: {(attn_best_loss - fno_best_loss) / fno_best_loss * 100:+.1f}%")
    print(f"vs MHF-FNO: {(attn_best_loss - mhf_best_loss) / mhf_best_loss * 100:+.1f}%")
    print(f"训练时间: {attn_time:.1f}s")
    
    results['models']['MHF+Attention'] = {
        'parameters': params_attn,
        'best_test_loss': attn_best_loss,
        'final_train_loss': attn_train_losses[-1],
        'train_time_s': attn_time,
    }
    
    # ========================================================================
    # 汇总报告
    # ========================================================================
    print(f"\n{'='*70}")
    print(f"汇总报告")
    print(f"{'='*70}")
    
    print(f"\n{'模型':<20} {'参数量':<12} {'最佳Loss':<12} {'vs FNO':<12}")
    print(f"{'-'*60}")
    print(f"{'FNO (基准)':<20} {params_fno:<12,} {fno_best_loss:<12.6f} {'基准':<12}")
    print(f"{'MHF-FNO':<20} {params_mhf:<12,} {mhf_best_loss:<12.6f} {(mhf_best_loss - fno_best_loss) / fno_best_loss * 100:+.1f}%")
    print(f"{'MHF+Attention':<20} {params_attn:<12,} {attn_best_loss:<12.6f} {(attn_best_loss - fno_best_loss) / fno_best_loss * 100:+.1f}%")
    
    # 判断实验结论
    print(f"\n{'='*70}")
    print(f"实验结论")
    print(f"{'='*70}")
    
    if attn_best_loss < mhf_best_loss:
        improvement = (mhf_best_loss - attn_best_loss) / mhf_best_loss * 100
        print(f"✅ 跨头注意力有效! 相比 MHF-FNO 改善 {improvement:.1f}%")
    else:
        degradation = (attn_best_loss - mhf_best_loss) / mhf_best_loss * 100
        print(f"❌ 跨头注意力未改善，相比 MHF-FNO 差 {degradation:.1f}%")
    
    if attn_best_loss < fno_best_loss:
        improvement = (fno_best_loss - attn_best_loss) / fno_best_loss * 100
        print(f"✅ MHF+Attention 超越 FNO 基准 {improvement:.1f}%")
    else:
        degradation = (attn_best_loss - fno_best_loss) / fno_best_loss * 100
        print(f"⚠️ MHF+Attention 未超越 FNO 基准，差 {degradation:.1f}%")
    
    # 保存结果
    output_path = Path(config['output_dir']) / 'ns_attention_results.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n结果已保存: {output_path}")
    
    return results


if __name__ == '__main__':
    config = {
        'data_dir': './data',
        'output_dir': './results',
        'epochs': 100,
        'batch_size': 16,
        'lr': 1e-3,
        'seed': 42,
        'device': 'cpu',
    }
    
    run_experiment(config)