"""
MHF-FNO 优化测试 v3 - 寻找最佳平衡点

策略：
1. 使用 MHF-FNO (输入层) 作为基础
2. 调整 hidden_channels 找到参数减少 30%+ 且 L2 误差接近的配置
3. 测试混合配置（部分层使用 MHF）
"""

import torch
import torch.nn as nn
from neuralop.models import FNO
from neuralop.losses.data_losses import LpLoss
import numpy as np
from mhf_fno import MHFSpectralConv
import time


class ScaleDiverseMHF(MHFSpectralConv):
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
    print(" MHF-FNO 优化测试 v3 - 最佳平衡点")
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
    
    # 基准参数量
    baseline_params = 133873  # FNO n_modes=(8,8), hidden=32, n_layers=3
    
    # ==================== 测试 1: FNO (基准) ====================
    print("\n" + "=" * 50)
    print(" [1/4] FNO (基准)")
    print("=" * 50)
    
    torch.manual_seed(42)
    model_fno = FNO(n_modes=(8, 8), hidden_channels=32, in_channels=1, out_channels=1, n_layers=3)
    params_fno = count_parameters(model_fno)
    print(f"参数量: {params_fno:,}")
    
    best_fno, train_fno, test_fno, time_fno = quick_train(model_fno, train_x, train_y, test_x, test_y, epochs=epochs, device=device)
    cpu_fno, _ = measure_inference_time(model_fno, sample_input)
    print(f"CPU 延迟: {cpu_fno:.2f} ms")
    
    results.append({
        "name": "FNO (基准)",
        "params": params_fno,
        "train": train_fno,
        "test": test_fno,
        "epoch_time": time_fno,
        "cpu": cpu_fno
    })
    
    del model_fno
    
    # ==================== 测试 2: MHF-FNO (输入层, hidden=28) ====================
    print("\n" + "=" * 50)
    print(" [2/4] MHF-FNO (输入层, hidden=28)")
    print("=" * 50)
    
    torch.manual_seed(42)
    model_h28 = FNO(n_modes=(8, 8), hidden_channels=28, in_channels=1, out_channels=1, n_layers=3)
    model_h28.fno_blocks.convs[0] = ScaleDiverseMHF(28, 28, (8, 8), n_heads=4)
    params_h28 = count_parameters(model_h28)
    print(f"参数量: {params_h28:,}")
    
    best_h28, train_h28, test_h28, time_h28 = quick_train(model_h28, train_x, train_y, test_x, test_y, epochs=epochs, device=device)
    cpu_h28, _ = measure_inference_time(model_h28, sample_input)
    print(f"CPU 延迟: {cpu_h28:.2f} ms")
    
    results.append({
        "name": "MHF-FNO (h=28)",
        "params": params_h28,
        "train": train_h28,
        "test": test_h28,
        "epoch_time": time_h28,
        "cpu": cpu_h28
    })
    
    del model_h28
    
    # ==================== 测试 3: MHF-FNO (输入层, hidden=26) ====================
    print("\n" + "=" * 50)
    print(" [3/4] MHF-FNO (输入层, hidden=26)")
    print("=" * 50)
    
    torch.manual_seed(42)
    model_h26 = FNO(n_modes=(8, 8), hidden_channels=26, in_channels=1, out_channels=1, n_layers=3)
    model_h26.fno_blocks.convs[0] = ScaleDiverseMHF(26, 26, (8, 8), n_heads=4)
    params_h26 = count_parameters(model_h26)
    print(f"参数量: {params_h26:,}")
    
    best_h26, train_h26, test_h26, time_h26 = quick_train(model_h26, train_x, train_y, test_x, test_y, epochs=epochs, device=device)
    cpu_h26, _ = measure_inference_time(model_h26, sample_input)
    print(f"CPU 延迟: {cpu_h26:.2f} ms")
    
    results.append({
        "name": "MHF-FNO (h=26)",
        "params": params_h26,
        "train": train_h26,
        "test": test_h26,
        "epoch_time": time_h26,
        "cpu": cpu_h26
    })
    
    del model_h26
    
    # ==================== 测试 4: MHF-FNO (输入层, hidden=25) ====================
    print("\n" + "=" * 50)
    print(" [4/4] MHF-FNO (输入层, hidden=25)")
    print("=" * 50)
    
    torch.manual_seed(42)
    model_h25 = FNO(n_modes=(8, 8), hidden_channels=25, in_channels=1, out_channels=1, n_layers=3)
    model_h25.fno_blocks.convs[0] = ScaleDiverseMHF(25, 25, (8, 8), n_heads=4)
    params_h25 = count_parameters(model_h25)
    print(f"参数量: {params_h25:,}")
    
    best_h25, train_h25, test_h25, time_h25 = quick_train(model_h25, train_x, train_y, test_x, test_y, epochs=epochs, device=device)
    cpu_h25, _ = measure_inference_time(model_h25, sample_input)
    print(f"CPU 延迟: {cpu_h25:.2f} ms")
    
    results.append({
        "name": "MHF-FNO (h=25)",
        "params": params_h25,
        "train": train_h25,
        "test": test_h25,
        "epoch_time": time_h25,
        "cpu": cpu_h25
    })
    
    # ==================== 结果汇总 ====================
    print("\n" + "=" * 70)
    print(" 📊 MHF-FNO 优化测试结果 v3")
    print("=" * 70)
    
    baseline = results[0]
    
    print(f"\n{'模型':<20} {'参数':<10} {'参数减少':<10} {'训练Loss':<12} {'测试Loss':<12} {'Loss变化':<10} {'CPU延迟':<10}")
    print("-" * 95)
    
    for r in results:
        param_change = (baseline['params'] - r['params']) / baseline['params'] * 100
        loss_change = (r['test'] - baseline['test']) / baseline['test'] * 100
        
        param_str = f"{param_change:+.1f}%" if r != baseline else "基准"
        loss_str = f"{loss_change:+.2f}%" if r != baseline else "基准"
        
        print(f"{r['name']:<20} {r['params']:<10,} {param_str:<10} {r['train']:<12.6f} {r['test']:<12.6f} {loss_str:<10} {r['cpu']:.2f}ms")
    
    # 找到最佳配置
    print("\n" + "=" * 70)
    print(" 🏆 目标达成分析")
    print("=" * 70)
    
    # 筛选参数减少 >= 30% 的配置
    valid_configs = []
    for r in results[1:]:
        param_reduction = (baseline['params'] - r['params']) / baseline['params'] * 100
        loss_change = (r['test'] - baseline['test']) / baseline['test'] * 100
        speed_change = (r['epoch_time'] - baseline['epoch_time']) / baseline['epoch_time'] * 100
        cpu_change = (r['cpu'] - baseline['cpu']) / baseline['cpu'] * 100
        
        if param_reduction >= 30:
            valid_configs.append({
                "name": r['name'],
                "params": r['params'],
                "param_reduction": param_reduction,
                "loss_change": loss_change,
                "test_loss": r['test'],
                "speed_change": speed_change,
                "cpu_change": cpu_change,
                "cpu": r['cpu']
            })
        
        print(f"\n{r['name']}:")
        print(f"  参数减少: {param_reduction:.1f}% {'✅' if param_reduction >= 30 else '○'}")
        print(f"  L2 误差变化: {loss_change:+.2f}% {'✅' if loss_change <= 5 else '○'}")
        print(f"  训练速度: {speed_change:+.1f}%")
        print(f"  CPU 推理: {cpu_change:+.1f}%")
    
    if valid_configs:
        print("\n" + "=" * 70)
        print(" ✅ 参数减少 ≥30% 的配置")
        print("=" * 70)
        
        for c in sorted(valid_configs, key=lambda x: x['test_loss']):
            print(f"\n{c['name']}:")
            print(f"  参数减少: {c['param_reduction']:.1f}%")
            print(f"  L2 误差变化: {c['loss_change']:+.2f}%")
            print(f"  测试 Loss: {c['test_loss']:.6f}")
            print(f"  CPU 推理: {c['cpu']:.2f}ms")
    else:
        print("\n⚠️ 没有配置达到参数减少 ≥30%")
    
    # 最终推荐
    print("\n" + "=" * 70)
    print(" 📌 最终推荐")
    print("=" * 70)
    
    # 找到 L2 误差最小的配置
    best = min(results[1:], key=lambda x: x['test'])
    param_reduction = (baseline['params'] - best['params']) / baseline['params'] * 100
    loss_change = (best['test'] - baseline['test']) / baseline['test'] * 100
    
    print(f"\n推荐配置: {best['name']}")
    print(f"  参数量: {best['params']:,} (减少 {param_reduction:.1f}%)")
    print(f"  L2 误差: {best['test']:.6f} (变化 {loss_change:+.2f}%)")
    print(f"  CPU 推理: {best['cpu']:.2f}ms")
    
    return results


if __name__ == "__main__":
    main()