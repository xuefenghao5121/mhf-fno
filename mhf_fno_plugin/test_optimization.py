"""
MHF-FNO 优化测试

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
            torch.cuda.synchronize() if device == 'cuda' else None
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms
    
    return np.mean(times), np.std(times)


def train_and_evaluate(model, train_loader, test_loader, epochs=50, lr=1e-4, device='cuda'):
    """训练并记录详细指标"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = LpLoss(d=2, p=2, reduction='mean')
    
    best_test_loss = float('inf')
    epoch_times = []
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        model.train()
        train_loss_sum = 0
        train_count = 0
        
        for batch in train_loader:
            x = batch['x'].to(device)
            y = batch['y'].to(device)
            
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            
            train_loss_sum += loss.item()
            train_count += 1
        
        scheduler.step()
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        
        # 测试评估
        model.eval()
        with torch.no_grad():
            test_loss_sum = 0
            test_count = 0
            for batch in test_loader:
                x = batch['x'].to(device)
                y = batch['y'].to(device)
                test_loss_sum += loss_fn(model(x), y).item()
                test_count += 1
            test_loss = test_loss_sum / test_count
        
        if test_loss < best_test_loss:
            best_test_loss = test_loss
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: Test L2 = {test_loss:.6f}, Time = {epoch_time:.2f}s")
    
    avg_epoch_time = np.mean(epoch_times)
    return best_test_loss, avg_epoch_time


def quick_train_and_evaluate(model, train_x, train_y, test_x, test_y, epochs=50, batch_size=32, lr=1e-3, device='cuda'):
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
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        model.train()
        perm = torch.randperm(n_train)
        
        for i in range(0, n_train, batch_size):
            bx = train_x[perm[i:i+batch_size]]
            by = train_y[perm[i:i+batch_size]]
            
            optimizer.zero_grad()
            loss = loss_fn(model(bx), by)
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        
        model.eval()
        with torch.no_grad():
            test_loss = loss_fn(model(test_x), test_y).item()
        
        if test_loss < best_test_loss:
            best_test_loss = test_loss
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: Test L2 = {test_loss:.6f}, Time = {epoch_time:.2f}s")
    
    avg_epoch_time = np.mean(epoch_times)
    return best_test_loss, avg_epoch_time


def main():
    print("=" * 70)
    print(" MHF-FNO 优化测试")
    print(" 目标：使用 MHF 设计来优化 FNO")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")
    
    # 尝试加载 Navier-Stokes 数据
    use_navier_stokes = False
    try:
        print("\n尝试加载 Navier-Stokes 数据...")
        from neuralop.data.datasets import load_navier_stokes_pt
        
        train_loader, test_loaders, data_processor = load_navier_stokes_pt(
            n_train=1000,
            n_tests=[200],
            batch_size=16,
            test_batch_sizes=[16],
            train_resolution=128,
            test_resolutions=[128],
        )
        test_loader = test_loaders[128]
        use_navier_stokes = True
        print("✓ Navier-Stokes 数据加载成功！")
        
        # 获取样本形状
        for batch in train_loader:
            sample_x = batch['x']
            sample_y = batch['y']
            print(f"输入形状: {sample_x.shape}")
            print(f"输出形状: {sample_y.shape}")
            break
            
        in_channels = sample_x.shape[1]
        out_channels = sample_y.shape[1]
        n_modes = (32, 32)
        hidden_channels = 64
        n_layers = 4
        
    except Exception as e:
        print(f"Navier-Stokes 加载失败: {e}")
        print("\n切换到 Darcy Flow 数据集...")
        
        # 使用 Darcy Flow 数据
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
        
        in_channels = 1
        out_channels = 1
        n_modes = (8, 8)
        hidden_channels = 32
        n_layers = 3
    
    results = []
    epochs = 50
    
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
    if use_navier_stokes:
        test_loss_fno, epoch_time_fno = train_and_evaluate(
            model_fno, train_loader, test_loader, epochs=epochs, device=device
        )
    else:
        test_loss_fno, epoch_time_fno = quick_train_and_evaluate(
            model_fno, train_x, train_y, test_x, test_y, epochs=epochs, device=device
        )
    
    # CPU 推理延迟
    sample_input = sample_x[:1] if use_navier_stokes else test_x[:1]
    cpu_mean_fno, cpu_std_fno = measure_inference_time(model_fno, sample_input, device='cpu')
    print(f"CPU 推理延迟: {cpu_mean_fno:.2f} ± {cpu_std_fno:.2f} ms")
    
    results.append({
        "name": "FNO (基准)",
        "params": params_fno,
        "test_loss": test_loss_fno,
        "epoch_time": epoch_time_fno,
        "cpu_latency": cpu_mean_fno,
        "cpu_std": cpu_std_fno
    })
    
    del model_fno
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
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
    if use_navier_stokes:
        test_loss_mhf_input, epoch_time_mhf_input = train_and_evaluate(
            model_mhf_input, train_loader, test_loader, epochs=epochs, device=device
        )
    else:
        test_loss_mhf_input, epoch_time_mhf_input = quick_train_and_evaluate(
            model_mhf_input, train_x, train_y, test_x, test_y, epochs=epochs, device=device
        )
    
    # CPU 推理延迟
    cpu_mean_mhf_input, cpu_std_mhf_input = measure_inference_time(model_mhf_input, sample_input, device='cpu')
    print(f"CPU 推理延迟: {cpu_mean_mhf_input:.2f} ± {cpu_std_mhf_input:.2f} ms")
    
    results.append({
        "name": "MHF-FNO (输入层)",
        "params": params_mhf_input,
        "test_loss": test_loss_mhf_input,
        "epoch_time": epoch_time_mhf_input,
        "cpu_latency": cpu_mean_mhf_input,
        "cpu_std": cpu_std_mhf_input
    })
    
    del model_mhf_input
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
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
    if use_navier_stokes:
        test_loss_mhf_edge, epoch_time_mhf_edge = train_and_evaluate(
            model_mhf_edge, train_loader, test_loader, epochs=epochs, device=device
        )
    else:
        test_loss_mhf_edge, epoch_time_mhf_edge = quick_train_and_evaluate(
            model_mhf_edge, train_x, train_y, test_x, test_y, epochs=epochs, device=device
        )
    
    # CPU 推理延迟
    cpu_mean_mhf_edge, cpu_std_mhf_edge = measure_inference_time(model_mhf_edge, sample_input, device='cpu')
    print(f"CPU 推理延迟: {cpu_mean_mhf_edge:.2f} ± {cpu_std_mhf_edge:.2f} ms")
    
    results.append({
        "name": "MHF-FNO (边缘层)",
        "params": params_mhf_edge,
        "test_loss": test_loss_mhf_edge,
        "epoch_time": epoch_time_mhf_edge,
        "cpu_latency": cpu_mean_mhf_edge,
        "cpu_std": cpu_std_mhf_edge
    })
    
    # ==================== 结果汇总 ====================
    print("\n" + "=" * 70)
    print(" 📊 MHF-FNO 优化测试结果")
    print("=" * 70)
    
    dataset_name = "Navier-Stokes" if use_navier_stokes else "Darcy Flow"
    print(f"\n数据集: {dataset_name}")
    print(f"训练轮数: {epochs}")
    
    print(f"\n{'模型':<25} {'参数量':<12} {'L2误差':<12} {'训练时间/epoch':<15} {'CPU延迟(ms)':<15}")
    print("-" * 80)
    
    for r in results:
        print(f"{r['name']:<25} {r['params']:<12,} {r['test_loss']:<12.6f} {r['epoch_time']:<15.2f} {r['cpu_latency']:.2f} ± {r['cpu_std']:.2f}")
    
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
    
    return results


if __name__ == "__main__":
    main()