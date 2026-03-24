"""
MHF-FNO 优化测试 v2 - 调整配置以平衡精度和效率

目标：
- L2 误差 ≤ 标准 FNO
- 参数效率 ≥ 30%
- 训练速度相当
- CPU 推理更高效

策略：
- 使用更大的 n_modes 来补偿 MHF 的精度损失
- 对比不同配置找到最佳平衡点
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


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def measure_inference_time(model, x, device='cpu', n_runs=100, warmup=10):
    model = model.to(device)
    model.eval()
    x = x.to(device)
    
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
    
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            start = time.perf_counter()
            _ = model(x)
            end = time.perf_counter()
            times.append((end - start) * 1000)
    
    return np.mean(times), np.std(times)


def quick_train(model, train_x, train_y, test_x, test_y, epochs=50, batch_size=32, lr=1e-3, device='cpu'):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = LpLoss(d=2, p=2, reduction='mean')
    
    train_x, train_y = train_x.to(device), train_y.to(device)
    test_x, test_y = test_x.to(device), test_y.to(device)
    
    n_train = train_x.shape[0]
    best_test = float('inf')
    epoch_times = []
    final_train = 0
    final_test = 0
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        model.train()
        perm = torch.randperm(n_train)
        
        epoch_train = 0
        batch_count = 0
        
        for i in range(0, n_train, batch_size):
            bx = train_x[perm[i:i+batch_size]]
            by = train_y[perm[i:i+batch_size]]
            
            optimizer.zero_grad()
            loss = loss_fn(model(bx), by)
            loss.backward()
            optimizer.step()
            
            epoch_train += loss.item()
            batch_count += 1
        
        scheduler.step()
        epoch_times.append(time.time() - epoch_start)
        
        model.eval()
        with torch.no_grad():
            train_loss = epoch_train / batch_count
            test_loss = loss_fn(model(test_x), test_y).item()
        
        final_train = train_loss
        final_test = test_loss
        
        if test_loss < best_test:
            best_test = test_loss
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: Train={train_loss:.6f}, Test={test_loss:.6f}, Time={epoch_times[-1]:.2f}s")
    
    return best_test, final_train, final_test, np.mean(epoch_times)


def main():
    print("=" * 70)
    print(" MHF-FNO 优化测试 v2 - 平衡精度与效率")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")
    
    # 加载数据
    print("\n加载 Darcy Flow 数据...")
    data_path = '/usr/local/lib/python3.11/site-packages/neuralop/data/datasets/data/'
    try:
        train_data = torch.load(f'{data_path}/darcy_train_16.pt', weights_only=False)
        test_data = torch.load(f'{data_path}/darcy_test_16.pt', weights_only=False)
    except:
        import neuralop
        pkg_path = neuralop.__path__[0]
        data_path = f'{pkg_path}/data/datasets/data/'
        train_data = torch.load(f'{data_path}/darcy_train_16.pt', weights_only=False)
        test_data = torch.load(f'{data_path}/darcy_test_16.pt', weights_only=False)
    
    train_x = train_data['x'].unsqueeze(1).float()
    train_y = train_data['y'].unsqueeze(1).float()
    test_x = test_data['x'].unsqueeze(1).float()
    test_y = test_data['y'].unsqueeze(1).float()
    
    print(f"训练集: {train_x.shape}")
    print(f"测试集: {test_x.shape}")
    
    epochs = 50
    results = []
    sample_input = test_x[:1]
    
    # ==================== 测试 1: FNO (基准) ====================
    print("\n" + "=" * 50)
    print(" [1/5] FNO (基准) - n_modes=(8,8), hidden=32")
    print("=" * 50)
    
    torch.manual_seed(42)
    model_fno = FNO(n_modes=(8, 8), hidden_channels=32, in_channels=1, out_channels=1, n_layers=3)
    params_fno = count_parameters(model_fno)
    print(f"参数量: {params_fno:,}")
    
    best_fno, train_fno, test_fno, time_fno = quick_train(model_fno, train_x, train_y, test_x, test_y, epochs=epochs, device=device)
    cpu_fno, cpu_std_fno = measure_inference_time(model_fno, sample_input)
    print(f"CPU 延迟: {cpu_fno:.2f} ± {cpu_std_fno:.2f} ms")
    
    results.append({
        "name": "FNO (基准)",
        "config": "n_modes=(8,8), hidden=32",
        "params": params_fno,
        "train": train_fno,
        "test": test_fno,
        "epoch_time": time_fno,
        "cpu": cpu_fno
    })
    
    del model_fno
    
    # ==================== 测试 2: MHF-FNO 输入层 (标准配置) ====================
    print("\n" + "=" * 50)
    print(" [2/5] MHF-FNO (输入层) - 标准配置")
    print("=" * 50)
    
    torch.manual_seed(42)
    model_mhf_std = FNO(n_modes=(8, 8), hidden_channels=32, in_channels=1, out_channels=1, n_layers=3)
    model_mhf_std.fno_blocks.convs[0] = ScaleDiverseMHF(32, 32, (8, 8), n_heads=4)
    params_mhf_std = count_parameters(model_mhf_std)
    print(f"参数量: {params_mhf_std:,}")
    
    best_std, train_std, test_std, time_std = quick_train(model_mhf_std, train_x, train_y, test_x, test_y, epochs=epochs, device=device)
    cpu_std, cpu_std_std = measure_inference_time(model_mhf_std, sample_input)
    print(f"CPU 延迟: {cpu_std:.2f} ± {cpu_std_std:.2f} ms")
    
    results.append({
        "name": "MHF-FNO (输入层)",
        "config": "n_modes=(8,8), hidden=32",
        "params": params_mhf_std,
        "train": train_std,
        "test": test_std,
        "epoch_time": time_std,
        "cpu": cpu_std
    })
    
    del model_mhf_std
    
    # ==================== 测试 3: MHF-FNO 输入层 (增大 n_modes) ====================
    print("\n" + "=" * 50)
    print(" [3/5] MHF-FNO (输入层) - 增大 n_modes=(12,12)")
    print("=" * 50)
    
    torch.manual_seed(42)
    model_mhf_large = FNO(n_modes=(12, 12), hidden_channels=32, in_channels=1, out_channels=1, n_layers=3)
    model_mhf_large.fno_blocks.convs[0] = ScaleDiverseMHF(32, 32, (12, 12), n_heads=4)
    params_mhf_large = count_parameters(model_mhf_large)
    print(f"参数量: {params_mhf_large:,}")
    
    best_large, train_large, test_large, time_large = quick_train(model_mhf_large, train_x, train_y, test_x, test_y, epochs=epochs, device=device)
    cpu_large, cpu_std_large = measure_inference_time(model_mhf_large, sample_input)
    print(f"CPU 延迟: {cpu_large:.2f} ± {cpu_std_large:.2f} ms")
    
    results.append({
        "name": "MHF-FNO (n_modes=12)",
        "config": "n_modes=(12,12), hidden=32",
        "params": params_mhf_large,
        "train": train_large,
        "test": test_large,
        "epoch_time": time_large,
        "cpu": cpu_large
    })
    
    del model_mhf_large
    
    # ==================== 测试 4: MHF-FNO 边缘层 (减小 hidden_channels) ====================
    print("\n" + "=" * 50)
    print(" [4/5] MHF-FNO (边缘层) - hidden=24")
    print("=" * 50)
    
    torch.manual_seed(42)
    model_mhf_edge = FNO(n_modes=(8, 8), hidden_channels=24, in_channels=1, out_channels=1, n_layers=3)
    model_mhf_edge.fno_blocks.convs[0] = ScaleDiverseMHF(24, 24, (8, 8), n_heads=4)
    model_mhf_edge.fno_blocks.convs[-1] = ScaleDiverseMHF(24, 24, (8, 8), n_heads=4)
    params_mhf_edge = count_parameters(model_mhf_edge)
    print(f"参数量: {params_mhf_edge:,}")
    
    best_edge, train_edge, test_edge, time_edge = quick_train(model_mhf_edge, train_x, train_y, test_x, test_y, epochs=epochs, device=device)
    cpu_edge, cpu_std_edge = measure_inference_time(model_mhf_edge, sample_input)
    print(f"CPU 延迟: {cpu_edge:.2f} ± {cpu_std_edge:.2f} ms")
    
    results.append({
        "name": "MHF-FNO (边缘层, h=24)",
        "config": "n_modes=(8,8), hidden=24",
        "params": params_mhf_edge,
        "train": train_edge,
        "test": test_edge,
        "epoch_time": time_edge,
        "cpu": cpu_edge
    })
    
    del model_mhf_edge
    
    # ==================== 测试 5: MHF-FNO 输入层 (n_heads=8) ====================
    print("\n" + "=" * 50)
    print(" [5/5] MHF-FNO (输入层) - n_heads=8")
    print("=" * 50)
    
    torch.manual_seed(42)
    model_mhf_heads = FNO(n_modes=(8, 8), hidden_channels=32, in_channels=1, out_channels=1, n_layers=3)
    model_mhf_heads.fno_blocks.convs[0] = ScaleDiverseMHF(32, 32, (8, 8), n_heads=8)
    params_mhf_heads = count_parameters(model_mhf_heads)
    print(f"参数量: {params_mhf_heads:,}")
    
    best_heads, train_heads, test_heads, time_heads = quick_train(model_mhf_heads, train_x, train_y, test_x, test_y, epochs=epochs, device=device)
    cpu_heads, cpu_std_heads = measure_inference_time(model_mhf_heads, sample_input)
    print(f"CPU 延迟: {cpu_heads:.2f} ± {cpu_std_heads:.2f} ms")
    
    results.append({
        "name": "MHF-FNO (n_heads=8)",
        "config": "n_modes=(8,8), hidden=32, heads=8",
        "params": params_mhf_heads,
        "train": train_heads,
        "test": test_heads,
        "epoch_time": time_heads,
        "cpu": cpu_heads
    })
    
    # ==================== 结果汇总 ====================
    print("\n" + "=" * 70)
    print(" 📊 MHF-FNO 优化测试结果 v2")
    print("=" * 70)
    
    baseline = results[0]
    
    print(f"\n{'模型':<25} {'配置':<25} {'参数':<10} {'测试Loss':<12} {'参数减少':<10} {'Loss变化':<10}")
    print("-" * 95)
    
    for r in results:
        param_change = (r['params'] - baseline['params']) / baseline['params'] * 100
        loss_change = (r['test'] - baseline['test']) / baseline['test'] * 100
        
        param_str = f"{param_change:+.1f}%" if r != baseline else "基准"
        loss_str = f"{loss_change:+.2f}%" if r != baseline else "基准"
        
        print(f"{r['name']:<25} {r['config']:<25} {r['params']:<10,} {r['test']:<12.6f} {param_str:<10} {loss_str:<10}")
    
    # 目标达成分析
    print("\n" + "=" * 70)
    print(" ✅ 目标达成分析")
    print("=" * 70)
    
    for r in results[1:]:
        param_reduction = (baseline['params'] - r['params']) / baseline['params'] * 100
        loss_change = (r['test'] - baseline['test']) / baseline['test'] * 100
        speed_change = (r['epoch_time'] - baseline['epoch_time']) / baseline['epoch_time'] * 100
        cpu_change = (r['cpu'] - baseline['cpu']) / baseline['cpu'] * 100
        
        print(f"\n{r['name']}:")
        
        # L2 误差
        if loss_change <= 0:
            print(f"  ✅ L2 误差: {loss_change:+.2f}% (优于基准)")
        elif loss_change <= 5:
            print(f"  ✓ L2 误差: {loss_change:+.2f}% (接近基准)")
        else:
            print(f"  ○ L2 误差: {loss_change:+.2f}% (略高于基准)")
        
        # 参数效率
        if param_reduction >= 30:
            print(f"  ✅ 参数效率: 减少 {param_reduction:.1f}% (达标)")
        elif param_reduction > 0:
            print(f"  ○ 参数效率: 减少 {param_reduction:.1f}% (未达30%目标)")
        else:
            print(f"  ✗ 参数效率: 增加 {-param_reduction:.1f}%")
        
        # 训练速度
        if speed_change <= 0:
            print(f"  ✅ 训练速度: {speed_change:+.1f}% (更快)")
        elif speed_change <= 10:
            print(f"  ✓ 训练速度: {speed_change:+.1f}% (相当)")
        else:
            print(f"  ○ 训练速度: {speed_change:+.1f}% (较慢)")
        
        # CPU 推理
        if cpu_change < 0:
            print(f"  ✅ CPU 推理: {cpu_change:+.1f}% (更快)")
        else:
            print(f"  ○ CPU 推理: {cpu_change:+.1f}% (略慢)")
    
    # 最佳配置推荐
    print("\n" + "=" * 70)
    print(" 🏆 最佳配置推荐")
    print("=" * 70)
    
    # 找到参数减少 >= 30% 且 L2 误差最小的配置
    valid_results = [r for r in results[1:] if (baseline['params'] - r['params']) / baseline['params'] >= 0.3]
    
    if valid_results:
        best = min(valid_results, key=lambda x: x['test'])
        param_reduction = (baseline['params'] - best['params']) / baseline['params'] * 100
        loss_change = (best['test'] - baseline['test']) / baseline['test'] * 100
        
        print(f"\n推荐配置: {best['name']}")
        print(f"  配置: {best['config']}")
        print(f"  参数量: {best['params']:,} (减少 {param_reduction:.1f}%)")
        print(f"  L2 误差: {best['test']:.6f} (变化 {loss_change:+.2f}%)")
        print(f"  训练时间: {best['epoch_time']:.2f}s/epoch")
        print(f"  CPU 推理: {best['cpu']:.2f}ms")
    else:
        # 没有达到 30% 参数减少的配置，推荐 L2 误差最小的
        best = min(results[1:], key=lambda x: x['test'])
        param_reduction = (baseline['params'] - best['params']) / baseline['params'] * 100
        loss_change = (best['test'] - baseline['test']) / baseline['test'] * 100
        
        print(f"\n推荐配置: {best['name']}")
        print(f"  配置: {best['config']}")
        print(f"  参数量: {best['params']:,} (减少 {param_reduction:.1f}%)")
        print(f"  L2 误差: {best['test']:.6f} (变化 {loss_change:+.2f}%)")
        print(f"  训练时间: {best['epoch_time']:.2f}s/epoch")
        print(f"  CPU 推理: {best['cpu']:.2f}ms")
        print(f"\n  ⚠️ 注意: 参数减少未达 30%，但 L2 误差最优")
    
    return results


if __name__ == "__main__":
    main()