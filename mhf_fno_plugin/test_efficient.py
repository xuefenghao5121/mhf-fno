"""
MHF-FNO 快速性能测试 - Darcy Flow 16x16

核心目标验证（快速版）：
1. L2 误差 ≤ 标准 FNO
2. 参数效率：减少 30%+
3. 训练速度：相当或更快
"""

import torch
import torch.nn as nn
from neuralop.models import FNO
from neuralop.losses.data_losses import LpLoss
from neuralop.data.datasets import load_darcy_flow_small
import numpy as np
from mhf_fno import MHFSpectralConv, create_hybrid_fno
import time


class ScaleDiverseMHF(MHFSpectralConv):
    def __init__(self, in_channels, out_channels, n_modes, n_heads=4):
        super().__init__(in_channels, out_channels, n_modes, n_heads)
        with torch.no_grad():
            for h in range(n_heads):
                nn.init.normal_(self.weight[h], mean=0, std=0.01 * (2 ** h))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def measure_latency(model, x, n_runs=50):
    model.eval()
    with torch.no_grad():
        for _ in range(5):
            _ = model(x)
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            start = time.perf_counter()
            _ = model(x)
            times.append((time.perf_counter() - start) * 1000)
    return np.mean(times), np.std(times)


def train_model(model, train_loader, test_loader, epochs=50, device='cpu'):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = LpLoss(d=2, p=2, reduction='mean')
    
    best_test = float('inf')
    epoch_times = []
    final_train, final_test = 0, 0
    
    for epoch in range(epochs):
        start = time.time()
        model.train()
        train_sum = 0
        for batch in train_loader:
            x, y = batch['x'].to(device), batch['y'].to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            train_sum += loss.item()
        scheduler.step()
        epoch_times.append(time.time() - start)
        
        model.eval()
        with torch.no_grad():
            train_loss = train_sum / len(train_loader)
            test_sum = 0
            for batch in test_loader:
                x, y = batch['x'].to(device), batch['y'].to(device)
                test_sum += loss_fn(model(x), y).item()
            test_loss = test_sum / len(test_loader)
        
        final_train, final_test = train_loss, test_loss
        if test_loss < best_test:
            best_test = test_loss
    
    return best_test, final_train, final_test, np.mean(epoch_times)


def main():
    print("=" * 70)
    print(" MHF-FNO 性能测试 - Darcy Flow 16x16")
    print("=" * 70)
    
    device = 'cpu'
    print(f"设备: {device}")
    
    # 加载数据
    print("\n加载数据...")
    train_loader, test_loaders, _ = load_darcy_flow_small(
        n_train=1000, n_tests=[200], batch_size=32, test_batch_sizes=[32], test_resolutions=[16]
    )
    test_loader = test_loaders[16]
    
    # 配置
    n_modes = (8, 8)
    hidden_channels = 32
    epochs = 50
    
    results = []
    
    # 测试配置列表
    configs = [
        ('FNO (基准)', lambda: FNO(n_modes=n_modes, hidden_channels=hidden_channels, in_channels=1, out_channels=1, n_layers=4)),
        ('MHF-FNO (输入层)', lambda: (lambda m: (setattr(m.fno_blocks.convs[0], '__class__', MHFSpectralConv) or m))(FNO(n_modes=n_modes, hidden_channels=hidden_channels, in_channels=1, out_channels=1, n_layers=4)) if False else (lambda: (
            m := FNO(n_modes=n_modes, hidden_channels=hidden_channels, in_channels=1, out_channels=1, n_layers=4),
            setattr(m.fno_blocks, 'convs', nn.ModuleList([MHFSpectralConv(hidden_channels, hidden_channels, n_modes, 4)] + list(m.fno_blocks.convs[1:]))),
            m
        )[-1])()),
    ]
    
    # 测试 1: 标准 FNO
    print("\n[1/4] FNO (基准)")
    torch.manual_seed(42)
    model_fno = FNO(n_modes=n_modes, hidden_channels=hidden_channels, in_channels=1, out_channels=1, n_layers=4)
    params_fno = count_parameters(model_fno)
    print(f"参数: {params_fno:,}")
    
    for batch in train_loader:
        sample = batch['x'][:1]
        break
    lat_fno, _ = measure_latency(model_fno, sample)
    print(f"延迟: {lat_fno:.2f}ms")
    
    start = time.time()
    best_fno, train_fno, test_fno, epoch_fno = train_model(model_fno, train_loader, test_loader, epochs)
    total_fno = time.time() - start
    print(f"结果: Train={train_fno:.4f}, Test={test_fno:.4f}, 时间={total_fno:.1f}s")
    
    results.append({'name': 'FNO', 'params': params_fno, 'train': train_fno, 'test': test_fno, 'best': best_fno, 'epoch': epoch_fno, 'lat': lat_fno})
    del model_fno
    
    # 测试 2: MHF-FNO (输入层)
    print("\n[2/4] MHF-FNO (输入层)")
    torch.manual_seed(42)
    model_mhf1 = FNO(n_modes=n_modes, hidden_channels=hidden_channels, in_channels=1, out_channels=1, n_layers=4)
    model_mhf1.fno_blocks.convs[0] = MHFSpectralConv(hidden_channels, hidden_channels, n_modes, 4)
    params_mhf1 = count_parameters(model_mhf1)
    print(f"参数: {params_mhf1:,}")
    
    lat_mhf1, _ = measure_latency(model_mhf1, sample)
    print(f"延迟: {lat_mhf1:.2f}ms")
    
    start = time.time()
    best_mhf1, train_mhf1, test_mhf1, epoch_mhf1 = train_model(model_mhf1, train_loader, test_loader, epochs)
    total_mhf1 = time.time() - start
    print(f"结果: Train={train_mhf1:.4f}, Test={test_mhf1:.4f}, 时间={total_mhf1:.1f}s")
    
    results.append({'name': 'MHF(输入层)', 'params': params_mhf1, 'train': train_mhf1, 'test': test_mhf1, 'best': best_mhf1, 'epoch': epoch_mhf1, 'lat': lat_mhf1})
    del model_mhf1
    
    # 测试 3: MHF-FNO (混合层)
    print("\n[3/4] MHF-FNO (混合层 1+3)")
    torch.manual_seed(42)
    model_mhf2 = create_hybrid_fno(n_modes=n_modes, hidden_channels=hidden_channels, in_channels=1, out_channels=1, n_layers=4, mhf_layers=[0, 2], n_heads=4)
    params_mhf2 = count_parameters(model_mhf2)
    print(f"参数: {params_mhf2:,}")
    
    lat_mhf2, _ = measure_latency(model_mhf2, sample)
    print(f"延迟: {lat_mhf2:.2f}ms")
    
    start = time.time()
    best_mhf2, train_mhf2, test_mhf2, epoch_mhf2 = train_model(model_mhf2, train_loader, test_loader, epochs)
    total_mhf2 = time.time() - start
    print(f"结果: Train={train_mhf2:.4f}, Test={test_mhf2:.4f}, 时间={total_mhf2:.1f}s")
    
    results.append({'name': 'MHF(混合层)', 'params': params_mhf2, 'train': train_mhf2, 'test': test_mhf2, 'best': best_mhf2, 'epoch': epoch_mhf2, 'lat': lat_mhf2})
    del model_mhf2
    
    # 测试 4: MHF-FNO (尺度初始化)
    print("\n[4/4] MHF-FNO (尺度初始化)")
    torch.manual_seed(42)
    model_mhf3 = FNO(n_modes=n_modes, hidden_channels=hidden_channels, in_channels=1, out_channels=1, n_layers=4)
    model_mhf3.fno_blocks.convs[0] = ScaleDiverseMHF(hidden_channels, hidden_channels, n_modes, 4)
    params_mhf3 = count_parameters(model_mhf3)
    print(f"参数: {params_mhf3:,}")
    
    lat_mhf3, _ = measure_latency(model_mhf3, sample)
    print(f"延迟: {lat_mhf3:.2f}ms")
    
    start = time.time()
    best_mhf3, train_mhf3, test_mhf3, epoch_mhf3 = train_model(model_mhf3, train_loader, test_loader, epochs)
    total_mhf3 = time.time() - start
    print(f"结果: Train={train_mhf3:.4f}, Test={test_mhf3:.4f}, 时间={total_mhf3:.1f}s")
    
    results.append({'name': 'MHF(尺度)', 'params': params_mhf3, 'train': train_mhf3, 'test': test_mhf3, 'best': best_mhf3, 'epoch': epoch_mhf3, 'lat': lat_mhf3})
    del model_mhf3
    
    # 汇总
    print("\n" + "=" * 70)
    print(" 📊 测试结果汇总")
    print("=" * 70)
    
    baseline = results[0]
    print(f"\n{'模型':<18} {'参数':<10} {'训练L2':<10} {'测试L2':<10} {'最佳L2':<10} {'Epoch时间':<10} {'延迟ms':<8}")
    print("-" * 76)
    
    for r in results:
        print(f"{r['name']:<18} {r['params']:<10,} {r['train']:<10.4f} {r['test']:<10.4f} {r['best']:<10.4f} {r['epoch']:<10.2f}s {r['lat']:<8.2f}")
    
    print("\n" + "=" * 70)
    print(" 📈 与基准对比")
    print("=" * 70)
    
    for r in results[1:]:
        param_red = (1 - r['params']/baseline['params']) * 100
        test_diff = (r['test'] - baseline['test']) / baseline['test'] * 100
        speed_diff = (r['epoch'] - baseline['epoch']) / baseline['epoch'] * 100
        lat_diff = (r['lat'] - baseline['lat']) / baseline['lat'] * 100
        
        print(f"\n{r['name']}:")
        print(f"  {'✅' if param_red >= 30 else '⚠️'} 参数减少: {param_red:.1f}%")
        print(f"  {'✅' if test_diff <= 0 else '❌'} L2误差: {test_diff:+.2f}%")
        print(f"  {'✅' if abs(speed_diff) <= 10 else '⚠️'} 训练速度: {speed_diff:+.1f}%")
        print(f"  {'✅' if lat_diff <= 0 else '⚠️'} 推理延迟: {lat_diff:+.1f}%")
    
    # 核心目标
    print("\n" + "=" * 70)
    print(" 🎯 核心目标达成情况")
    print("=" * 70)
    
    param_met = any((1-r['params']/baseline['params'])*100 >= 30 for r in results[1:])
    l2_met = any(r['test'] <= baseline['test'] for r in results[1:])
    speed_met = any(abs((r['epoch']-baseline['epoch'])/baseline['epoch']*100) <= 10 for r in results[1:])
    
    print(f"\n1. 参数减少 30%+: {'✅ 达成' if param_met else '❌ 未达成'}")
    print(f"2. L2误差 ≤ FNO: {'✅ 达成' if l2_met else '❌ 未达成'}")
    print(f"3. 训练速度相当: {'✅ 达成' if speed_met else '❌ 未达成'}")
    
    # 保存结果
    timestamp = int(time.time())
    with open(f'mhf_fno_results_{timestamp}.txt', 'w') as f:
        f.write(f"MHF-FNO 性能测试 - Darcy Flow 16x16\n")
        f.write(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"{'模型':<18} {'参数':<10} {'训练L2':<10} {'测试L2':<10}\n")
        f.write("-" * 50 + "\n")
        for r in results:
            f.write(f"{r['name']:<18} {r['params']:<10,} {r['train']:<10.4f} {r['test']:<10.4f}\n")
        f.write(f"\n核心目标:\n")
        f.write(f"1. 参数减少30%+: {'达成' if param_met else '未达成'}\n")
        f.write(f"2. L2误差≤FNO: {'达成' if l2_met else '未达成'}\n")
        f.write(f"3. 速度相当: {'达成' if speed_met else '未达成'}\n")
    
    print(f"\n结果已保存: mhf_fno_results_{timestamp}.txt")


if __name__ == "__main__":
    main()