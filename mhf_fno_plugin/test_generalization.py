"""
MHF-FNO 泛化能力测试

测量：
1. 训练 Loss
2. 测试 Loss
3. 泛化差距 = 测试 Loss - 训练 Loss

假设：MHF 的正则化效果会减小泛化差距
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


def train_and_track(model, train_x, train_y, test_x, test_y, epochs=100, batch_size=32, lr=1e-3):
    """训练并跟踪训练/测试 Loss"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = LpLoss(d=2, p=2, reduction='mean')
    
    n_train = train_x.shape[0]
    
    train_losses = []
    test_losses = []
    
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
        
        # 计算训练 Loss
        model.eval()
        with torch.no_grad():
            train_loss = epoch_train_loss / batch_count
        
        # 计算测试 Loss
        with torch.no_grad():
            test_loss = loss_fn(model(test_x), test_y).item()
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
        if train_loss < best_train:
            best_train = train_loss
        if test_loss < best_test:
            best_test = test_loss
    
    return best_train, best_test, train_losses, test_losses


def main():
    print("=" * 70)
    print(" MHF-FNO 泛化能力测试")
    print("=" * 70)
    
    # 加载数据
    data_path = '/usr/local/lib/python3.11/site-packages/neuralop/data/datasets/data/'
    train_data = torch.load(f'{data_path}/darcy_train_16.pt', weights_only=False)
    test_data = torch.load(f'{data_path}/darcy_test_16.pt', weights_only=False)
    
    train_x = train_data['x'].unsqueeze(1).float()
    train_y = train_data['y'].unsqueeze(1).float()
    test_x = test_data['x'].unsqueeze(1).float()
    test_y = test_data['y'].unsqueeze(1).float()
    
    print(f"训练集: {train_x.shape}")
    print(f"测试集: {test_x.shape}")
    
    results = []
    
    # 配置列表
    configs = [
        ("FNO (基准)", None, None),
        ("MHF-FNO (随机)", MHFSpectralConv, None),
        ("MHF-FNO (尺度)", ScaleDiverseMHF, None),
    ]
    
    for name, MHFClass, _ in configs:
        print(f"\n{'='*70}")
        print(f" {name}")
        print("=" * 70)
        
        torch.manual_seed(42)
        model = FNO(n_modes=(8, 8), hidden_channels=32, in_channels=1, out_channels=1, n_layers=3)
        
        params = sum(p.numel() for p in model.parameters())
        print(f"参数: {params:,}")
        
        if MHFClass is not None:
            model.fno_blocks.convs[0] = MHFClass(32, 32, (8, 8), n_heads=4)
        
        start = time.time()
        best_train, best_test, train_losses, test_losses = train_and_track(
            model, train_x, train_y, test_x, test_y, epochs=100
        )
        train_time = time.time() - start
        
        # 计算泛化差距
        generalization_gap = best_test - best_train
        gap_ratio = (generalization_gap / best_train) * 100
        
        print(f"\n训练 Loss: {best_train:.4f}")
        print(f"测试 Loss: {best_test:.4f}")
        print(f"泛化差距: {generalization_gap:.4f} ({gap_ratio:.1f}%)")
        print(f"训练时间: {train_time:.1f}s")
        
        results.append({
            "name": name,
            "params": params,
            "train_loss": best_train,
            "test_loss": best_test,
            "gap": generalization_gap,
            "gap_ratio": gap_ratio,
            "train_losses": train_losses,
            "test_losses": test_losses
        })
        
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # 汇总
    print("\n" + "=" * 70)
    print(" 📊 泛化能力对比")
    print("=" * 70)
    
    print(f"\n{'模型':<20} {'训练Loss':<10} {'测试Loss':<10} {'泛化差距':<12} {'差距%':<8}")
    print("-" * 65)
    
    baseline = results[0]
    
    for r in results:
        improvement = (baseline['test_loss'] - r['test_loss']) / baseline['test_loss'] * 100
        gap_improve = (baseline['gap'] - r['gap']) / baseline['gap'] * 100 if baseline['gap'] != 0 else 0
        
        print(f"{r['name']:<20} {r['train_loss']:<10.4f} {r['test_loss']:<10.4f} {r['gap']:<12.4f} {r['gap_ratio']:<8.1f}%")
    
    print("\n" + "=" * 70)
    print(" 💡 分析")
    print("=" * 70)
    
    # 找出泛化能力最好的
    best_generalization = min(results, key=lambda x: x['gap_ratio'])
    best_test = min(results, key=lambda x: x['test_loss'])
    
    print(f"\n最佳泛化能力: {best_generalization['name']}")
    print(f"  泛化差距: {best_generalization['gap']:.4f} ({best_generalization['gap_ratio']:.1f}%)")
    
    print(f"\n最佳测试精度: {best_test['name']}")
    print(f"  测试 Loss: {best_test['test_loss']:.4f}")
    
    # 对比基准
    print("\n与基准 (FNO) 对比:")
    for r in results[1:]:
        gap_improve = (baseline['gap_ratio'] - r['gap_ratio']) / baseline['gap_ratio'] * 100
        test_improve = (baseline['test_loss'] - r['test_loss']) / baseline['test_loss'] * 100
        
        print(f"\n{r['name']}:")
        print(f"  测试精度变化: {test_improve:+.2f}%")
        print(f"  泛化差距变化: {gap_improve:+.2f}%")
        
        if gap_improve > 0:
            print(f"  ✅ 泛化能力提升 {gap_improve:.1f}%")
        else:
            print(f"  ❌ 泛化能力下降 {-gap_improve:.1f}%")


if __name__ == "__main__":
    main()