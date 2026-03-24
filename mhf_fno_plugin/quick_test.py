"""
MHF-FNO 快速泛化测试 (减少 epoch 数以快速完成)
"""

import torch
import torch.nn as nn
from neuralop.models import FNO
from neuralop.losses.data_losses import LpLoss
import numpy as np
from mhf_fno import MHFSpectralConv
import time


class ScaleDiverseMHF(MHFSpectralConv):
    """尺度多样性初始化"""
    def __init__(self, in_channels, out_channels, n_modes, n_heads=4):
        super().__init__(in_channels, out_channels, n_modes, n_heads)
        with torch.no_grad():
            for h in range(n_heads):
                scale = 0.01 * (2 ** h)
                nn.init.normal_(self.weight[h], mean=0, std=scale)


def quick_train(model, train_x, train_y, test_x, test_y, epochs=30, batch_size=32, lr=1e-3):
    """快速训练"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = LpLoss(d=2, p=2, reduction='mean')
    
    n_train = train_x.shape[0]
    best_train = float('inf')
    best_test = float('inf')
    
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train)
        
        epoch_train_loss = 0
        batch_count = 0
        
        for i in range(0, n_train, batch_size):
            bx = train_x[perm[i:i+batch_size]]
            by = train_y[perm[i:i+batch_size]]
            
            optimizer.zero_grad()
            loss = loss_fn(model(bx), by)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
            batch_count += 1
        
        scheduler.step()
        
        model.eval()
        with torch.no_grad():
            train_loss = epoch_train_loss / batch_count
            test_loss = loss_fn(model(test_x), test_y).item()
        
        if train_loss < best_train:
            best_train = train_loss
        if test_loss < best_test:
            best_test = test_loss
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: Train {train_loss:.4f}, Test {test_loss:.4f}")
    
    return best_train, best_test


def main():
    print("=" * 70)
    print(" MHF-FNO 快速泛化测试 (Darcy Flow)")
    print("=" * 70)
    
    # 加载数据
    data_path = '/usr/local/lib/python3.11/site-packages/neuralop/data/datasets/data/'
    train_data = torch.load(f'{data_path}/darcy_train_16.pt', weights_only=False)
    test_data = torch.load(f'{data_path}/darcy_test_16.pt', weights_only=False)
    
    train_x = train_data['x'].unsqueeze(1).float()
    train_y = train_data['y'].unsqueeze(1).float()
    test_x = test_data['x'].unsqueeze(1).float()
    test_y = test_data['y'].unsqueeze(1).float()
    
    print(f"\n训练集: {train_x.shape}")
    print(f"测试集: {test_x.shape}")
    
    results = []
    
    # 测试 1: 标准 FNO
    print("\n[1/3] 标准 FNO...")
    torch.manual_seed(42)
    model_fno = FNO(n_modes=(8, 8), hidden_channels=32, in_channels=1, out_channels=1, n_layers=3)
    params_fno = sum(p.numel() for p in model_fno.parameters())
    print(f"参数: {params_fno:,}")
    
    start = time.time()
    train_fno, test_fno = quick_train(model_fno, train_x, train_y, test_x, test_y, epochs=30)
    time_fno = time.time() - start
    
    gap_fno = test_fno - train_fno
    gap_ratio_fno = (gap_fno / train_fno) * 100
    print(f"训练: {train_fno:.4f}, 测试: {test_fno:.4f}, 泛化差距: {gap_ratio_fno:.1f}%")
    results.append({"name": "FNO", "params": params_fno, "train": train_fno, "test": test_fno, "gap": gap_ratio_fno})
    
    del model_fno
    
    # 测试 2: MHF-FNO (随机)
    print("\n[2/3] MHF-FNO (随机初始化)...")
    torch.manual_seed(42)
    model_mhf = FNO(n_modes=(8, 8), hidden_channels=32, in_channels=1, out_channels=1, n_layers=3)
    model_mhf.fno_blocks.convs[0] = MHFSpectralConv(32, 32, (8, 8), n_heads=4)
    params_mhf = sum(p.numel() for p in model_mhf.parameters())
    print(f"参数: {params_mhf:,}")
    
    start = time.time()
    train_mhf, test_mhf = quick_train(model_mhf, train_x, train_y, test_x, test_y, epochs=30)
    time_mhf = time.time() - start
    
    gap_mhf = test_mhf - train_mhf
    gap_ratio_mhf = (gap_mhf / train_mhf) * 100
    print(f"训练: {train_mhf:.4f}, 测试: {test_mhf:.4f}, 泛化差距: {gap_ratio_mhf:.1f}%")
    results.append({"name": "MHF-FNO", "params": params_mhf, "train": train_mhf, "test": test_mhf, "gap": gap_ratio_mhf})
    
    del model_mhf
    
    # 测试 3: MHF-FNO (尺度)
    print("\n[3/3] MHF-FNO (尺度初始化)...")
    torch.manual_seed(42)
    model_scale = FNO(n_modes=(8, 8), hidden_channels=32, in_channels=1, out_channels=1, n_layers=3)
    model_scale.fno_blocks.convs[0] = ScaleDiverseMHF(32, 32, (8, 8), n_heads=4)
    params_scale = sum(p.numel() for p in model_scale.parameters())
    print(f"参数: {params_scale:,}")
    
    start = time.time()
    train_scale, test_scale = quick_train(model_scale, train_x, train_y, test_x, test_y, epochs=30)
    time_scale = time.time() - start
    
    gap_scale = test_scale - train_scale
    gap_ratio_scale = (gap_scale / train_scale) * 100
    print(f"训练: {train_scale:.4f}, 测试: {test_scale:.4f}, 泛化差距: {gap_ratio_scale:.1f}%")
    results.append({"name": "MHF-FNO (尺度)", "params": params_scale, "train": train_scale, "test": test_scale, "gap": gap_ratio_scale})
    
    # 汇总
    print("\n" + "=" * 70)
    print(" 📊 Darcy Flow 泛化能力测试结果")
    print("=" * 70)
    
    print(f"\n{'模型':<20} {'参数':<12} {'训练Loss':<10} {'测试Loss':<10} {'泛化差距%':<10}")
    print("-" * 65)
    
    for r in results:
        print(f"{r['name']:<20} {r['params']:<12,} {r['train']:<10.4f} {r['test']:<10.4f} {r['gap']:<10.1f}%")
    
    # 对比
    print("\n与基准 (FNO) 对比:")
    baseline = results[0]
    for r in results[1:]:
        test_improve = (baseline['test'] - r['test']) / baseline['test'] * 100
        gap_improve = (baseline['gap'] - r['gap']) / baseline['gap'] * 100
        
        print(f"\n{r['name']}:")
        print(f"  测试精度: {test_improve:+.2f}%")
        print(f"  泛化差距: {gap_improve:+.2f}%")
    
    return results


if __name__ == "__main__":
    main()