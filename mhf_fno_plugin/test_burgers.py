"""
MHF-FNO 在 Burgers 方程上的测试

Burgers 方程:
∂u/∂t + u∂u/∂x = ν∂²u/∂x²

特点：
- 非线性 PDE
- 时序预测问题
- 比 Navier-Stokes 简单但有非线性
"""

import torch
import torch.nn as nn
from neuralop.models import FNO
from neuralop.losses.data_losses import LpLoss
import numpy as np
from mhf_fno import MHFSpectralConv
import time


def analyze_diversity(model):
    """分析多头多样性"""
    mhf = model.fno_blocks.convs[0]
    weight = mhf.weight
    n_heads = weight.shape[0]
    
    head_weights = []
    for h in range(n_heads):
        w = weight[h].detach().cpu().numpy()
        w_avg = np.mean(np.abs(w), axis=(2, 3))
        head_weights.append(w_avg.flatten())
    
    similarity_matrix = np.zeros((n_heads, n_heads))
    for i in range(n_heads):
        for j in range(n_heads):
            sim = np.dot(head_weights[i], head_weights[j]) / (
                np.linalg.norm(head_weights[i]) * np.linalg.norm(head_weights[j]) + 1e-8
            )
            similarity_matrix[i, j] = sim
    
    off_diag = similarity_matrix[~np.eye(n_heads, dtype=bool)]
    return np.mean(off_diag), 1 - np.mean(off_diag)


def train_model(model, train_loader, test_loader, epochs=50, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = LpLoss(d=2, p=2, reduction='mean')
    
    best_loss = float('inf')
    start = time.time()
    
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            x = batch['x']
            y = batch['y']
            
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        
        model.eval()
        with torch.no_grad():
            total_loss = 0
            count = 0
            for batch in test_loader:
                x = batch['x']
                y = batch['y']
                total_loss += loss_fn(model(x), y).item()
                count += 1
            test_loss = total_loss / count
        
        if test_loss < best_loss:
            best_loss = test_loss
    
    return best_loss, time.time() - start


def main():
    print("=" * 70)
    print(" MHF-FNO 在 Burgers 方程上的测试")
    print("=" * 70)
    
    # 加载 Burgers 数据
    print("\n加载 Burgers 数据...")
    from neuralop.data.datasets import load_mini_burgers_1dtime
    
    train_loader, test_loader, data_processor = load_mini_burgers_1dtime(
        data_path='/usr/local/lib/python3.11/site-packages/neuralop/datasets/data',
        n_train=1000,
        n_test=200,
        batch_size=32,
        test_batch_size=32,
    )
    
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
    model_fno = FNO(n_modes=(16,), hidden_channels=64, in_channels=1, out_channels=1, n_layers=3)
    params_fno = sum(p.numel() for p in model_fno.parameters())
    print(f"参数: {params_fno:,}")
    loss_fno, time_fno = train_model(model_fno, train_loader, test_loader, epochs=50)
    print(f"Loss: {loss_fno:.4f}, 时间: {time_fno:.1f}s")
    results.append({"name": "FNO", "params": params_fno, "loss": loss_fno})
    del model_fno
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # 测试 2: MHF-FNO (随机初始化)
    print("\n[2/3] MHF-FNO (随机初始化)...")
    torch.manual_seed(42)
    model_mhf = FNO(n_modes=(16,), hidden_channels=64, in_channels=1, out_channels=1, n_layers=3)
    model_mhf.fno_blocks.convs[0] = MHFSpectralConv(64, 64, (16,), n_heads=4)
    params_mhf = sum(p.numel() for p in model_mhf.parameters())
    print(f"参数: {params_mhf:,}")
    loss_mhf, time_mhf = train_model(model_mhf, train_loader, test_loader, epochs=50)
    sim_mhf, div_mhf = analyze_diversity(model_mhf)
    print(f"Loss: {loss_mhf:.4f}, 多样性: {div_mhf:.4f}")
    results.append({"name": "MHF-FNO", "params": params_mhf, "loss": loss_mhf, "diversity": div_mhf})
    del model_mhf
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # 测试 3: MHF-FNO (尺度多样性初始化)
    print("\n[3/3] MHF-FNO (尺度多样性初始化)...")
    
    class ScaleDiverseMHF(MHFSpectralConv):
        def __init__(self, in_channels, out_channels, n_modes, n_heads=4):
            super().__init__(in_channels, out_channels, n_modes, n_heads)
            with torch.no_grad():
                for h in range(n_heads):
                    scale = 0.01 * (2 ** h)
                    nn.init.normal_(self.weight[h], mean=0, std=scale)
    
    torch.manual_seed(42)
    model_mhf_div = FNO(n_modes=(16,), hidden_channels=64, in_channels=1, out_channels=1, n_layers=3)
    model_mhf_div.fno_blocks.convs[0] = ScaleDiverseMHF(64, 64, (16,), n_heads=4)
    params_mhf_div = sum(p.numel() for p in model_mhf_div.parameters())
    print(f"参数: {params_mhf_div:,}")
    loss_mhf_div, time_mhf_div = train_model(model_mhf_div, train_loader, test_loader, epochs=50)
    sim_mhf_div, div_mhf_div = analyze_diversity(model_mhf_div)
    print(f"Loss: {loss_mhf_div:.4f}, 多样性: {div_mhf_div:.4f}")
    results.append({"name": "MHF-FNO (尺度)", "params": params_mhf_div, "loss": loss_mhf_div, "diversity": div_mhf_div})
    del model_mhf_div
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # 汇总
    print("\n" + "=" * 70)
    print(" 📊 测试结果")
    print("=" * 70)
    
    print(f"\n{'模型':<20} {'参数量':<12} {'Loss':<10} {'多样性':<10}")
    print("-" * 55)
    
    for r in results:
        div = r.get('diversity', '-')
        print(f"{r['name']:<20} {r['params']:<12,} {r['loss']:<10.4f} {div}")


if __name__ == "__main__":
    main()