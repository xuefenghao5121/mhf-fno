#!/usr/bin/env python3
"""
PINO Round 1 测试: 高频噪声惩罚

测试目标:
- MHF-PINO (高频噪声惩罚) test_loss < 0.383 (vs MHF-FNO baseline)
- FNO baseline ≈ 0.38 (验证测试正确)

测试配置:
- 数据集: Navier-Stokes 32×32 (500 train, 100 test)
- Epochs: 50
- lambda_physics: 0.0001
- freq_threshold: 0.5

作者: 天渊团队
日期: 2026-03-26
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np

# 项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from mhf_fno import MHFFNO
from mhf_fno.pino_high_freq import HighFreqPINOLoss
from neuralop.models import FNO
from neuralop.losses.data_losses import LpLoss


def load_ns_data(data_path, n_train, n_test):
    """加载 Navier-Stokes 数据"""
    data = torch.load(data_path, weights_only=False)
    
    if isinstance(data, dict):
        train_x = data.get('x', data.get('train_x'))
        train_y = data.get('y', data.get('train_y'))
    else:
        train_x, train_y = data[0], data[1]
    
    if train_x.dim() == 3:
        train_x = train_x.unsqueeze(1).float()
        train_y = train_y.unsqueeze(1).float()
    else:
        train_x = train_x.float()
        train_y = train_y.float()
    
    # 分割训练集和测试集
    # 从可用样本中分割: n_train=400, n_test=100
    actual_n_train = min(n_train, len(train_x) - n_test) if len(train_x) > n_test else len(train_x) // 2
    train_x_split = train_x[:actual_n_train]
    train_y_split = train_y[:actual_n_train]
    test_x = train_x[actual_n_train:actual_n_train+n_test]
    test_y = train_y[actual_n_train:actual_n_train+n_test]
    
    return train_x_split, train_y_split, test_x, test_y
    
    return train_x, train_y, test_x, test_y


def train_model(model, train_x, train_y, test_x, test_y, config, model_name, loss_fn=None):
    """训练模型"""
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    
    # 默认使用 LpLoss (2D)
    if loss_fn is None:
        loss_fn = LpLoss(d=2, p=2, reduction='mean')
    
    results = {
        'train_losses': [],
        'test_losses': [],
        'epoch_times': [],
    }
    
    n_train = train_x.shape[0]
    batch_size = config['batch_size']
    
    print(f"\n训练 {model_name}...")
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(config['epochs']):
        t0 = time.time()
        model.train()
        
        # 随机打乱
        perm = torch.randperm(n_train)
        train_loss = 0
        batch_count = 0
        
        for i in range(0, n_train, batch_size):
            bx = train_x[perm[i:i+batch_size]]
            by = train_y[perm[i:i+batch_size]]
            
            optimizer.zero_grad()
            pred = model(bx)
            
            # 使用 PINO loss 或标准 loss
            if isinstance(loss_fn, HighFreqPINOLoss):
                loss = loss_fn(pred, by)
            else:
                loss = loss_fn(pred, by)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            batch_count += 1
        
        scheduler.step()
        epoch_time = time.time() - t0
        
        # 测试
        model.eval()
        with torch.no_grad():
            # 测试时只用 LpLoss，不用物理约束
            test_loss_fn = LpLoss(d=2, p=2, reduction='mean')
            test_loss = test_loss_fn(model(test_x), test_y).item()
        
        results['train_losses'].append(train_loss / batch_count)
        results['test_losses'].append(test_loss)
        results['epoch_times'].append(epoch_time)
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{config['epochs']}: "
                  f"Train {train_loss/batch_count:.4f}, "
                  f"Test {test_loss:.4f}, "
                  f"Time {epoch_time:.1f}s")
    
    return results


def main():
    print("="*70)
    print("PINO Round 1 测试: 高频噪声惩罚")
    print("="*70)
    
    # 配置
    config = {
        'n_train': 500,
        'n_test': 100,
        'epochs': 50,
        'batch_size': 32,
        'learning_rate': 0.0005,
        'seed': 42,
        'device': 'cpu',
    }
    
    print(f"\n配置: {json.dumps(config, indent=2)}")
    
    # 设置随机种子
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # 加载数据
    data_path = Path(__file__).parent.parent / 'data' / 'ns_train_32.pt'
    print(f"\n加载数据: {data_path}")
    
    train_x, train_y, test_x, test_y = load_ns_data(
        data_path, config['n_train'], config['n_test']
    )
    
    print(f"  训练集: {train_x.shape}")
    print(f"  测试集: {test_x.shape}")
    
    # 模型配置
    in_channels = train_x.shape[1]
    out_channels = train_y.shape[1]
    hidden_channels = 32
    n_modes = (16, 16)
    
    results = {
        'config': config,
        'timestamp': datetime.now().isoformat(),
        'models': {}
    }
    
    # =========================================================================
    # 测试 1: FNO (基准)
    # =========================================================================
    print(f"\n{'='*60}")
    print("测试 FNO (基准)")
    print(f"{'='*60}")
    
    torch.manual_seed(config['seed'])
    model_fno = FNO(
        n_modes=n_modes,
        hidden_channels=hidden_channels,
        in_channels=in_channels,
        out_channels=out_channels,
        n_layers=3,
    )
    
    train_fno = train_model(model_fno, train_x, train_y, test_x, test_y, config, "FNO")
    
    results['models']['FNO'] = {
        'parameters': sum(p.numel() for p in model_fno.parameters()),
        'best_test_loss': min(train_fno['test_losses']),
        'final_test_loss': train_fno['test_losses'][-1],
        'avg_epoch_time': np.mean(train_fno['epoch_times']),
    }
    
    fno_loss = results['models']['FNO']['best_test_loss']
    print(f"\n✓ FNO 最佳测试损失: {fno_loss:.4f}")
    
    # 验证基线是否正常
    if fno_loss > 0.5:
        print(f"⚠️ 警告: FNO loss = {fno_loss:.4f} >> 0.5，测试可能有问题！")
        return
    
    # =========================================================================
    # 测试 2: MHF-FNO (无 PINO)
    # =========================================================================
    print(f"\n{'='*60}")
    print("测试 MHF-FNO (无 PINO)")
    print(f"{'='*60}")
    
    torch.manual_seed(config['seed'])
    model_mhf = MHFFNO.best_config(
        n_modes=n_modes,
        hidden_channels=hidden_channels,
        in_channels=in_channels,
        out_channels=out_channels,
    )
    
    train_mhf = train_model(model_mhf, train_x, train_y, test_x, test_y, config, "MHF-FNO")
    
    results['models']['MHF-FNO'] = {
        'parameters': sum(p.numel() for p in model_mhf.parameters()),
        'best_test_loss': min(train_mhf['test_losses']),
        'final_test_loss': train_mhf['test_losses'][-1],
        'avg_epoch_time': np.mean(train_mhf['epoch_times']),
        'param_reduction': (1 - sum(p.numel() for p in model_mhf.parameters()) / 
                                 sum(p.numel() for p in model_fno.parameters())) * 100,
    }
    
    mhf_loss = results['models']['MHF-FNO']['best_test_loss']
    print(f"\n✓ MHF-FNO 最佳测试损失: {mhf_loss:.4f}")
    
    # =========================================================================
    # 测试 3: MHF-PINO (高频噪声惩罚)
    # =========================================================================
    print(f"\n{'='*60}")
    print("测试 MHF-PINO (高频噪声惩罚)")
    print(f"{'='*60}")
    
    torch.manual_seed(config['seed'])
    model_pino = MHFFNO.best_config(
        n_modes=n_modes,
        hidden_channels=hidden_channels,
        in_channels=in_channels,
        out_channels=out_channels,
    )
    
    # 使用高频噪声惩罚 loss
    pino_loss_fn = HighFreqPINOLoss(
        lambda_physics=0.0001,
        freq_threshold=0.5
    )
    
    train_pino = train_model(
        model_pino, train_x, train_y, test_x, test_y, config, 
        "MHF-PINO", loss_fn=pino_loss_fn
    )
    
    results['models']['MHF-PINO'] = {
        'parameters': sum(p.numel() for p in model_pino.parameters()),
        'best_test_loss': min(train_pino['test_losses']),
        'final_test_loss': train_pino['test_losses'][-1],
        'avg_epoch_time': np.mean(train_pino['epoch_times']),
        'param_reduction': (1 - sum(p.numel() for p in model_pino.parameters()) / 
                                 sum(p.numel() for p in model_fno.parameters())) * 100,
        'pino_config': {
            'lambda_physics': 0.0001,
            'freq_threshold': 0.5,
        }
    }
    
    pino_loss = results['models']['MHF-PINO']['best_test_loss']
    print(f"\n✓ MHF-PINO 最佳测试损失: {pino_loss:.4f}")
    
    # =========================================================================
    # 结果对比
    # =========================================================================
    print(f"\n{'='*70}")
    print("结果对比")
    print(f"{'='*70}")
    
    print(f"\n{'模型':<15} {'参数量':>12} {'参数减少':>10} {'最佳Loss':>12} {'vs MHF-FNO':>12}")
    print(f"{'-'*60}")
    
    for name in ['FNO', 'MHF-FNO', 'MHF-PINO']:
        model_res = results['models'][name]
        params = model_res['parameters']
        loss = model_res['best_test_loss']
        
        if name == 'FNO':
            param_str = '-'
            vs_str = '基准'
        else:
            param_red = model_res.get('param_reduction', 0)
            param_str = f"{param_red:.1f}%"
            vs_mhf = (loss - mhf_loss) / mhf_loss * 100
            vs_str = f"{vs_mhf:+.2f}%"
        
        print(f"{name:<15} {params:>12,} {param_str:>10} {loss:>12.4f} {vs_str:>12}")
    
    # =========================================================================
    # 成功判断
    # =========================================================================
    print(f"\n{'='*70}")
    print("测试结论")
    print(f"{'='*70}")
    
    success = pino_loss < mhf_loss
    
    if success:
        print(f"✅ 成功! MHF-PINO ({pino_loss:.4f}) < MHF-FNO ({mhf_loss:.4f})")
        print(f"   改进: {(mhf_loss - pino_loss) / mhf_loss * 100:.2f}%")
    else:
        print(f"❌ 失败! MHF-PINO ({pino_loss:.4f}) >= MHF-FNO ({mhf_loss:.4f})")
        print(f"   恶化: {(pino_loss - mhf_loss) / mhf_loss * 100:.2f}%")
        print(f"\n下一步: 尝试 Round 2 (自适应 lambda)")
    
    # 保存结果
    output_file = Path(__file__).parent.parent / 'pino_round1_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else x)
    
    print(f"\n✅ 结果已保存: {output_file}")
    
    return success


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
