"""
MHF-FNO 在 Navier-Stokes 方程上的测试

Navier-Stokes 方程:
∂u/∂t + (u·∇)u = -∇p + ν∇²u + f
∇·u = 0

特点：
- 非线性 PDE
- 时序预测
- 湍流等复杂现象
- 比 Darcy 复杂得多
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


def train_and_track(model, train_loader, test_loader, epochs=50, lr=1e-4):
    """训练并跟踪泛化能力"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = LpLoss(d=2, p=2, reduction='mean')
    
    best_train = float('inf')
    best_test = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0
        train_count = 0
        
        for batch in train_loader:
            x = batch['x']
            y = batch['y']
            
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            
            train_loss_sum += loss.item()
            train_count += 1
        
        scheduler.step()
        
        # 计算训练和测试 Loss
        model.eval()
        with torch.no_grad():
            train_loss = train_loss_sum / train_count
            
            test_loss_sum = 0
            test_count = 0
            for batch in test_loader:
                x = batch['x']
                y = batch['y']
                test_loss_sum += loss_fn(model(x), y).item()
                test_count += 1
            test_loss = test_loss_sum / test_count
        
        if train_loss < best_train:
            best_train = train_loss
        if test_loss < best_test:
            best_test = test_loss
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: Train {train_loss:.4f}, Test {test_loss:.4f}")
    
    return best_train, best_test


def main():
    print("=" * 70)
    print(" MHF-FNO 在 Navier-Stokes 方程上的测试")
    print("=" * 70)
    
    # 加载 Navier-Stokes 数据
    print("\n加载 Navier-Stokes 数据（首次运行会自动下载）...")
    from neuralop.data.datasets import load_navier_stokes_pt
    
    try:
        train_loader, test_loaders, data_processor = load_navier_stokes_pt(
            n_train=1000,
            n_tests=[200],
            batch_size=16,
            test_batch_sizes=[16],
            train_resolution=128,
            test_resolutions=[128],
        )
        test_loader = test_loaders[128]
    except Exception as e:
        print(f"加载失败: {e}")
        print("请稍后重试，或使用其他数据集")
        return
    
    # 查看数据形状
    for batch in train_loader:
        x = batch['x']
        y = batch['y']
        print(f"输入形状: {x.shape}")
        print(f"输出形状: {y.shape}")
        break
    
    results = []
    
    # 测试 1: 标准 FNO
    print("\n[1/3] 标准 FNO...")
    torch.manual_seed(42)
    model_fno = FNO(n_modes=(32, 32), hidden_channels=64, in_channels=3, out_channels=3, n_layers=4)
    params_fno = sum(p.numel() for p in model_fno.parameters())
    print(f"参数: {params_fno:,}")
    
    start = time.time()
    train_fno, test_fno = train_and_track(model_fno, train_loader, test_loader, epochs=50)
    time_fno = time.time() - start
    
    gap_fno = test_fno - train_fno
    gap_ratio_fno = (gap_fno / train_fno) * 100
    print(f"训练: {train_fno:.4f}, 测试: {test_fno:.4f}, 泛化差距: {gap_ratio_fno:.1f}%")
    results.append({"name": "FNO", "params": params_fno, "train": train_fno, "test": test_fno, "gap": gap_ratio_fno})
    
    del model_fno
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # 测试 2: MHF-FNO (随机)
    print("\n[2/3] MHF-FNO (随机)...")
    torch.manual_seed(42)
    model_mhf = FNO(n_modes=(32, 32), hidden_channels=64, in_channels=3, out_channels=3, n_layers=4)
    model_mhf.fno_blocks.convs[0] = MHFSpectralConv(64, 64, (32, 32), n_heads=4)
    params_mhf = sum(p.numel() for p in model_mhf.parameters())
    print(f"参数: {params_mhf:,}")
    
    start = time.time()
    train_mhf, test_mhf = train_and_track(model_mhf, train_loader, test_loader, epochs=50)
    time_mhf = time.time() - start
    
    gap_mhf = test_mhf - train_mhf
    gap_ratio_mhf = (gap_mhf / train_mhf) * 100
    print(f"训练: {train_mhf:.4f}, 测试: {test_mhf:.4f}, 泛化差距: {gap_ratio_mhf:.1f}%")
    results.append({"name": "MHF-FNO", "params": params_mhf, "train": train_mhf, "test": test_mhf, "gap": gap_ratio_mhf})
    
    del model_mhf
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # 测试 3: MHF-FNO (尺度)
    print("\n[3/3] MHF-FNO (尺度初始化)...")
    torch.manual_seed(42)
    model_scale = FNO(n_modes=(32, 32), hidden_channels=64, in_channels=3, out_channels=3, n_layers=4)
    model_scale.fno_blocks.convs[0] = ScaleDiverseMHF(64, 64, (32, 32), n_heads=4)
    params_scale = sum(p.numel() for p in model_scale.parameters())
    print(f"参数: {params_scale:,}")
    
    start = time.time()
    train_scale, test_scale = train_and_track(model_scale, train_loader, test_loader, epochs=50)
    time_scale = time.time() - start
    
    gap_scale = test_scale - train_scale
    gap_ratio_scale = (gap_scale / train_scale) * 100
    print(f"训练: {train_scale:.4f}, 测试: {test_scale:.4f}, 泛化差距: {gap_ratio_scale:.1f}%")
    results.append({"name": "MHF-FNO (尺度)", "params": params_scale, "train": train_scale, "test": test_scale, "gap": gap_ratio_scale})
    
    # 汇总
    print("\n" + "=" * 70)
    print(" 📊 Navier-Stokes 测试结果")
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


if __name__ == "__main__":
    main()