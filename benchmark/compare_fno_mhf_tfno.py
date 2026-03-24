#!/usr/bin/env python3
"""
Darcy Flow 对比测试: FNO vs MHF-FNO vs TFNO

使用方法:
    python compare_fno_mhf_tfno.py
"""

import time
import torch
import torch.nn as nn
from neuralop.models import FNO, TFNO
from neuralop.losses.data_losses import LpLoss
from pathlib import Path
import sys

# 导入 MHF-FNO
sys.path.insert(0, str(Path(__file__).parent.parent))
from mhf_fno import create_hybrid_fno


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def train(model, train_x, train_y, test_x, test_y, epochs=30, lr=1e-3):
    """训练模型"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = LpLoss(d=2, p=2)
    
    t0 = time.time()
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(train_x))
        
        for i in range(0, len(train_x), 32):
            bx = train_x[perm[i:i+32]]
            by = train_y[perm[i:i+32]]
            
            optimizer.zero_grad()
            loss = loss_fn(model(bx), by)
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        
        # 测试
        model.eval()
        with torch.no_grad():
            test_loss = loss_fn(model(test_x), test_y).item()
        
        if test_loss < best_loss:
            best_loss = test_loss
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: test_loss={test_loss:.6f}")
    
    return {'time': time.time() - t0, 'loss': best_loss}


def main():
    print("=" * 60)
    print("Darcy Flow 对比测试: FNO vs MHF-FNO vs TFNO")
    print("=" * 60)
    
    # 加载数据
    data_path = Path(__file__).parent / 'data'
    
    train_data = torch.load(data_path / 'darcy_train_16.pt', weights_only=False)
    test_data = torch.load(data_path / 'darcy_test_16.pt', weights_only=False)
    
    train_x = train_data['x'].unsqueeze(1).float()
    train_y = train_data['y'].unsqueeze(1).float()
    test_x = test_data['x'].unsqueeze(1).float()
    test_y = test_data['y'].unsqueeze(1).float()
    
    print(f"\n数据: 训练 {train_x.shape}, 测试 {test_x.shape}")
    
    n_modes = (8, 8)
    hidden = 32
    results = {}
    
    # ========== FNO ==========
    print(f"\n{'='*60}")
    print("[1] FNO (基准)")
    print("=" * 60)
    
    torch.manual_seed(42)
    fno = FNO(
        n_modes=n_modes,
        in_channels=1,
        out_channels=1,
        hidden_channels=hidden,
        n_layers=3,
    )
    params_fno = count_params(fno)
    print(f"参数量: {params_fno:,}")
    
    r_fno = train(fno, train_x, train_y, test_x, test_y)
    results['FNO'] = {'params': params_fno, **r_fno}
    print(f"✅ Loss: {r_fno['loss']:.6f}")
    
    # ========== MHF-FNO ==========
    print(f"\n{'='*60}")
    print("[2] MHF-FNO (n_heads=2, mhf_layers=[0,2])")
    print("=" * 60)
    
    torch.manual_seed(42)
    mhf = create_hybrid_fno(
        n_modes=n_modes,
        hidden_channels=hidden,
        n_heads=2,
        mhf_layers=[0, 2]
    )
    params_mhf = count_params(mhf)
    print(f"参数量: {params_mhf:,} ({(1-params_mhf/params_fno)*100:+.1f}%)")
    
    r_mhf = train(mhf, train_x, train_y, test_x, test_y)
    results['MHF-FNO'] = {'params': params_mhf, **r_mhf}
    print(f"✅ Loss: {r_mhf['loss']:.6f}")
    
    # ========== TFNO ==========
    print(f"\n{'='*60}")
    print("[3] TFNO (rank=0.5)")
    print("=" * 60)
    
    torch.manual_seed(42)
    tfno = TFNO(
        n_modes=n_modes,
        in_channels=1,
        out_channels=1,
        hidden_channels=hidden,
        n_layers=3,
        rank=0.5,  # 50% 参数
    )
    params_tfno = count_params(tfno)
    print(f"参数量: {params_tfno:,} ({(1-params_tfno/params_fno)*100:+.1f}%)")
    
    r_tfno = train(tfno, train_x, train_y, test_x, test_y)
    results['TFNO'] = {'params': params_tfno, **r_tfno}
    print(f"✅ Loss: {r_tfno['loss']:.6f}")
    
    # ========== 汇总 ==========
    print(f"\n{'='*60}")
    print("结果汇总")
    print("=" * 60)
    
    print(f"\n{'模型':<15} {'参数量':>12} {'压缩率':>10} {'L2 Loss':>12} {'vs FNO':>10}")
    print("-" * 60)
    
    for name, r in results.items():
        compression = (1 - r['params'] / params_fno) * 100
        vs_fno = (r['loss'] / results['FNO']['loss'] - 1) * 100
        print(f"{name:<15} {r['params']:>12,} {compression:>9.1f}% {r['loss']:>12.6f} {vs_fno:>+9.1f}%")
    
    # 推荐
    print(f"\n{'='*60}")
    print("推荐")
    print("=" * 60)
    
    best = min(results.items(), key=lambda x: x[1]['loss'])
    print(f"\n✅ 最佳模型: {best[0]}")
    print(f"   参数量: {best[1]['params']:,}")
    print(f"   L2 Loss: {best[1]['loss']:.6f}")


if __name__ == '__main__':
    main()