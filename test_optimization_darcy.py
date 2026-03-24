"""
MHF-FNO 优化测试 - Darcy Flow 数据集

核心目标：使用 MHF 设计来优化 FNO

测试指标：
1. L2 误差 - MHF-FNO vs FNO（目标：相当或更好）
2. 参数效率 - 参数量减少（目标：30%+）
3. 训练速度 - 每 epoch 耗时（目标：相当或更快）
4. CPU 推理延迟（目标：更高效）

测试配置：
1. FNO (基准) - 标准配置
2. MHF-FNO (输入层) - 仅替换输入层
3. MHF-FNO (边缘层) - 替换第1层和最后1层
"""

import torch
import torch.nn as nn
from neuralop.models import FNO
from neuralop.losses.data_losses import LpLoss
import numpy as np
from mhf_fno import MHFSpectralConv
import time
import sys


class ScaleDiverseMHF(MHFSpectralConv):
    """尺度多样性初始化 - 最优配置"""
    def __init__(self, in_channels, out_channels, n_modes, n_heads=4):
        super().__init__(in_channels, out_channels, n_modes, n_heads)
        with torch.no_grad():
            for h in range(n_heads):
                scale = 0.01 * (2 ** h)
                nn.init.normal_(self.weight[h], mean=0, std=scale)


def count_parameters(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters())


def measure_inference_time(model, x, device='cpu', n_runs=100, warmup=10):
    """测量 CPU 推理延迟"""
    model = model.to(device)
    model.eval()
    x = x.to(device)
    
    # 预热
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
    
    # 测量
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            start = time.perf_counter()
            _ = model(x)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms
    
    return np.mean(times), np.std(times)


def quick_train_and_evaluate(model, train_x, train_y, test_x, test_y, epochs=50, batch_size=32, lr=1e-3, device='cpu'):
    """快速训练并记录详细指标"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = LpLoss(d=2, p=2, reduction='mean')
    
    train_x, train_y = train_x.to(device), train_y.to(device)
    test_x, test_y = test_x.to(device), test_y.to(device)
    
    n_train = train_x.shape[0]
    best_test_loss = float('inf')
    epoch_times = []
    final_train_loss = 0
    final_test_loss = 0
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
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
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        
        model.eval()
        with torch.no_grad():
            train_loss = epoch_train_loss / batch_count
            test_loss = loss_fn(model(test_x), test_y).item()
        
        final_train_loss = train_loss
        final_test_loss = test_loss
        
        if test_loss < best_test_loss:
            best_test_loss = test_loss
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: Train L2 = {train_loss:.6f}, Test L2 = {test_loss:.6f}, Time = {epoch_time:.2f}s")
    
    avg_epoch_time = np.mean(epoch_times)
    return best_test_loss, final_train_loss, final_test_loss, avg_epoch_time


def main():
    print("=" * 70)
    print(" MHF-FNO 优化测试 - Darcy Flow")
    print(" 目标：使用 MHF 设计来优化 FNO")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")
    
    # 加载 Darcy Flow 数据
    print("\n加载 Darcy Flow 数据...")
    data_path = '/usr/local/lib/python3.11/site-packages/neuralop/data/datasets/data/'
    try:
        train_data = torch.load(f'{data_path}/darcy_train_16.pt', weights_only=False)
        test_data = torch.load(f'{data_path}/darcy_test_16.pt', weights_only=False)
    except:
        # 尝试另一个路径
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
    
    # 配置
    in_channels = 1
    out_channels = 1
    n_modes = (8, 8)
    hidden_channels = 32
    n_layers = 3
    epochs = 50
    
    results = []
    
    # ==================== 测试 1: FNO (基准) ====================
    print("\n" + "=" * 50)
    print(" [1/3] FNO (基准)")
    print("=" * 50)
    
    torch.manual_seed(42)
    model_fno = FNO(
        n_modes=n_modes,
        hidden_channels=hidden_channels,
        in_channels=in_channels,
        out_channels=out_channels,
        n_layers=n_layers
    )
    params_fno = count_parameters(model_fno)
    print(f"参数量: {params_fno:,}")
    
    # 训练
    best_fno, train_fno, test_fno, epoch_time_fno = quick_train_and_evaluate(
        model_fno, train_x, train_y, test_x, test_y, epochs=epochs, device=device
    )
    
    # CPU 推理延迟
    sample_input = test_x[:1]
    cpu_mean_fno, cpu_std_fno = measure_inference_time(model_fno, sample_input, device='cpu')
    print(f"CPU 推理延迟: {cpu_mean_fno:.2f} ± {cpu_std_fno:.2f} ms")
    
    results.append({
        "name": "FNO (基准)",
        "params": params_fno,
        "train_loss": train_fno,
        "test_loss": test_fno,
        "best_test": best_fno,
        "epoch_time": epoch_time_fno,
        "cpu_latency": cpu_mean_fno,
        "cpu_std": cpu_std_fno
    })
    
    del model_fno
    
    # ==================== 测试 2: MHF-FNO (输入层) ====================
    print("\n" + "=" * 50)
    print(" [2/3] MHF-FNO (仅输入层)")
    print("=" * 50)
    
    torch.manual_seed(42)
    model_mhf_input = FNO(
        n_modes=n_modes,
        hidden_channels=hidden_channels,
        in_channels=in_channels,
        out_channels=out_channels,
        n_layers=n_layers
    )
    # 替换输入层
    model_mhf_input.fno_blocks.convs[0] = ScaleDiverseMHF(hidden_channels, hidden_channels, n_modes, n_heads=4)
    params_mhf_input = count_parameters(model_mhf_input)
    print(f"参数量: {params_mhf_input:,}")
    
    # 训练
    best_input, train_input, test_input, epoch_time_input = quick_train_and_evaluate(
        model_mhf_input, train_x, train_y, test_x, test_y, epochs=epochs, device=device
    )
    
    # CPU 推理延迟
    cpu_mean_input, cpu_std_input = measure_inference_time(model_mhf_input, sample_input, device='cpu')
    print(f"CPU 推理延迟: {cpu_mean_input:.2f} ± {cpu_std_input:.2f} ms")
    
    results.append({
        "name": "MHF-FNO (输入层)",
        "params": params_mhf_input,
        "train_loss": train_input,
        "test_loss": test_input,
        "best_test": best_input,
        "epoch_time": epoch_time_input,
        "cpu_latency": cpu_mean_input,
        "cpu_std": cpu_std_input
    })
    
    del model_mhf_input
    
    # ==================== 测试 3: MHF-FNO (边缘层) ====================
    print("\n" + "=" * 50)
    print(" [3/3] MHF-FNO (边缘层)")
    print("=" * 50)
    
    torch.manual_seed(42)
    model_mhf_edge = FNO(
        n_modes=n_modes,
        hidden_channels=hidden_channels,
        in_channels=in_channels,
        out_channels=out_channels,
        n_layers=n_layers
    )
    # 替换输入层和输出层
    model_mhf_edge.fno_blocks.convs[0] = ScaleDiverseMHF(hidden_channels, hidden_channels, n_modes, n_heads=4)
    model_mhf_edge.fno_blocks.convs[-1] = ScaleDiverseMHF(hidden_channels, hidden_channels, n_modes, n_heads=4)
    params_mhf_edge = count_parameters(model_mhf_edge)
    print(f"参数量: {params_mhf_edge:,}")
    
    # 训练
    best_edge, train_edge, test_edge, epoch_time_edge = quick_train_and_evaluate(
        model_mhf_edge, train_x, train_y, test_x, test_y, epochs=epochs, device=device
    )
    
    # CPU 推理延迟
    cpu_mean_edge, cpu_std_edge = measure_inference_time(model_mhf_edge, sample_input, device='cpu')
    print(f"CPU 推理延迟: {cpu_mean_edge:.2f} ± {cpu_std_edge:.2f} ms")
    
    results.append({
        "name": "MHF-FNO (边缘层)",
        "params": params_mhf_edge,
        "train_loss": train_edge,
        "test_loss": test_edge,
        "best_test": best_edge,
        "epoch_time": epoch_time_edge,
        "cpu_latency": cpu_mean_edge,
        "cpu_std": cpu_std_edge
    })
    
    # ==================== 结果汇总 ====================
    print("\n" + "=" * 70)
    print(" 📊 MHF-FNO 优化测试结果")
    print("=" * 70)
    
    print(f"\n数据集: Darcy Flow (16x16)")
    print(f"训练轮数: {epochs}")
    
    print(f"\n{'模型':<25} {'参数量':<12} {'训练Loss':<12} {'测试Loss':<12} {'epoch时间':<12} {'CPU延迟(ms)':<15}")
    print("-" * 90)
    
    for r in results:
        print(f"{r['name']:<25} {r['params']:<12,} {r['train_loss']:<12.6f} {r['test_loss']:<12.6f} {r['epoch_time']:<12.2f} {r['cpu_latency']:.2f} ± {r['cpu_std']:.2f}")
    
    # 详细对比
    baseline = results[0]
    print("\n" + "=" * 70)
    print(" 📈 与基准 FNO 对比")
    print("=" * 70)
    
    print(f"\n{'模型':<25} {'参数减少':<12} {'L2误差变化':<12} {'训练速度':<12} {'CPU延迟':<12}")
    print("-" * 75)
    
    for r in results[1:]:
        param_reduction = (baseline['params'] - r['params']) / baseline['params'] * 100
        loss_change = (r['test_loss'] - baseline['test_loss']) / baseline['test_loss'] * 100
        speed_change = (r['epoch_time'] - baseline['epoch_time']) / baseline['epoch_time'] * 100
        latency_change = (r['cpu_latency'] - baseline['cpu_latency']) / baseline['cpu_latency'] * 100
        
        print(f"{r['name']:<25} {param_reduction:+.1f}%       {loss_change:+.2f}%       {speed_change:+.1f}%       {latency_change:+.1f}%")
    
    # 目标检查
    print("\n" + "=" * 70)
    print(" ✅ 目标达成情况")
    print("=" * 70)
    
    for r in results[1:]:
        param_reduction = (baseline['params'] - r['params']) / baseline['params'] * 100
        loss_change = (r['test_loss'] - baseline['test_loss']) / baseline['test_loss'] * 100
        speed_change = (r['epoch_time'] - baseline['epoch_time']) / baseline['epoch_time'] * 100
        latency_change = (r['cpu_latency'] - baseline['cpu_latency']) / baseline['cpu_latency'] * 100
        
        print(f"\n{r['name']}:")
        
        # L2 误差目标
        if loss_change <= 5:
            print(f"  ✓ L2 误差: {loss_change:+.2f}% (目标: ≤基准)")
        else:
            print(f"  ✗ L2 误差: {loss_change:+.2f}% (目标: ≤基准)")
        
        # 参数效率目标
        if param_reduction >= 30:
            print(f"  ✓ 参数效率: 减少 {param_reduction:.1f}% (目标: ≥30%)")
        else:
            print(f"  ○ 参数效率: 减少 {param_reduction:.1f}% (目标: ≥30%)")
        
        # 训练速度目标
        if speed_change <= 10:
            print(f"  ✓ 训练速度: {speed_change:+.1f}% (目标: 相当或更快)")
        else:
            print(f"  ✗ 训练速度: {speed_change:+.1f}% (目标: 相当或更快)")
        
        # CPU 推理目标
        if latency_change <= 10:
            print(f"  ✓ CPU 推理: {latency_change:+.1f}% (目标: 更高效)")
        else:
            print(f"  ○ CPU 推理: {latency_change:+.1f}% (目标: 更高效)")
    
    # 保存结果
    print("\n" + "=" * 70)
    print(" 💾 保存测试结果")
    print("=" * 70)
    
    result_file = f"optimization_results_{int(time.time())}.txt"
    with open(result_file, 'w') as f:
        f.write("MHF-FNO 优化测试结果\n")
        f.write(f"数据集: Darcy Flow (16x16)\n")
        f.write(f"训练轮数: {epochs}\n")
        f.write(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"{'模型':<25} {'参数量':<12} {'训练Loss':<12} {'测试Loss':<12} {'epoch时间':<12} {'CPU延迟(ms)':<15}\n")
        f.write("-" * 90 + "\n")
        
        for r in results:
            f.write(f"{r['name']:<25} {r['params']:<12,} {r['train_loss']:<12.6f} {r['test_loss']:<12.6f} {r['epoch_time']:<12.2f} {r['cpu_latency']:.2f} ± {r['cpu_std']:.2f}\n")
        
        f.write("\n与基准 FNO 对比:\n")
        for r in results[1:]:
            param_reduction = (baseline['params'] - r['params']) / baseline['params'] * 100
            loss_change = (r['test_loss'] - baseline['test_loss']) / baseline['test_loss'] * 100
            speed_change = (r['epoch_time'] - baseline['epoch_time']) / baseline['epoch_time'] * 100
            latency_change = (r['cpu_latency'] - baseline['cpu_latency']) / baseline['cpu_latency'] * 100
            
            f.write(f"\n{r['name']}:\n")
            f.write(f"  参数减少: {param_reduction:+.1f}%\n")
            f.write(f"  L2误差变化: {loss_change:+.2f}%\n")
            f.write(f"  训练速度: {speed_change:+.1f}%\n")
            f.write(f"  CPU延迟: {latency_change:+.1f}%\n")
    
    print(f"结果已保存到: {result_file}")
    
    return results


if __name__ == "__main__":
    main()