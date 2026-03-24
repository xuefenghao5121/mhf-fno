"""
MHF-FNO 测试 - Darcy Flow 16x16 (替代 Navier-Stokes)

核心目标验证：
1. L2 误差 ≤ 标准 FNO（相当或更好）
2. 参数效率：减少 30%+
3. 训练速度：相当或更快

使用可用的 16x16 数据进行测试，重点验证 MHF-FNO 的参数效率和精度
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
    """测量推理延迟"""
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


def train_model(model, train_loader, test_loader, epochs=100, lr=1e-3, device='cpu'):
    """训练模型并记录详细指标"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = LpLoss(d=2, p=2, reduction='mean')
    
    best_test_loss = float('inf')
    epoch_times = []
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Training
        model.train()
        train_loss_sum = 0
        train_count = 0
        
        for batch in train_loader:
            x = batch['x'].to(device)
            y = batch['y'].to(device)
            
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            
            train_loss_sum += loss.item()
            train_count += 1
        
        scheduler.step()
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            train_loss = train_loss_sum / train_count
            
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
        
        if (epoch + 1) % 25 == 0:
            print(f"  Epoch {epoch+1}: Train={train_loss:.6f}, Test={test_loss:.6f}, Time={epoch_time:.2f}s")
    
    return {
        'best_test': best_test_loss,
        'final_train': train_loss,
        'final_test': test_loss,
        'avg_epoch_time': np.mean(epoch_times)
    }


def main():
    print("=" * 80)
    print(" MHF-FNO 性能测试 - Darcy Flow 16x16")
    print(" 目标: L2误差≤FNO, 参数减少30%+, 速度相当")
    print("=" * 80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")
    
    # 加载数据
    print("\n加载 Darcy Flow 16x16 数据...")
    train_loader, test_loaders, data_processor = load_darcy_flow_small(
        n_train=1000,
        n_tests=[200],
        batch_size=32,
        test_batch_sizes=[32],
        test_resolutions=[16]
    )
    test_loader = test_loaders[16]
    
    # 获取输入输出形状
    for batch in train_loader:
        sample_x = batch['x']
        sample_y = batch['y']
        print(f"输入形状: {sample_x.shape}")
        print(f"输出形状: {sample_y.shape}")
        break
    
    # 模型配置
    n_modes = (8, 8)
    hidden_channels = 32
    in_channels = 1
    out_channels = 1
    epochs = 100
    
    results = []
    
    # ============ 测试 1: 标准 FNO (基准) ============
    print("\n" + "=" * 60)
    print("[1/5] 标准 FNO (基准)")
    print("=" * 60)
    
    torch.manual_seed(42)
    model_fno = FNO(
        n_modes=n_modes,
        hidden_channels=hidden_channels,
        in_channels=in_channels,
        out_channels=out_channels,
        n_layers=4
    )
    params_fno = count_parameters(model_fno)
    print(f"参数量: {params_fno:,}")
    
    # 测量推理延迟
    inf_mean, inf_std = measure_inference_time(model_fno, sample_x[:1])
    print(f"推理延迟: {inf_mean:.2f} ± {inf_std:.2f} ms")
    
    start_time = time.time()
    result_fno = train_model(model_fno, train_loader, test_loader, epochs=epochs, lr=1e-3, device=device)
    total_time_fno = time.time() - start_time
    
    results.append({
        'name': 'FNO (基准)',
        'params': params_fno,
        'train_loss': result_fno['final_train'],
        'test_loss': result_fno['final_test'],
        'best_test': result_fno['best_test'],
        'epoch_time': result_fno['avg_epoch_time'],
        'total_time': total_time_fno,
        'inference_ms': inf_mean
    })
    
    del model_fno
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # ============ 测试 2: MHF-FNO (输入层) ============
    print("\n" + "=" * 60)
    print("[2/5] MHF-FNO (输入层)")
    print("=" * 60)
    
    torch.manual_seed(42)
    model_mhf_in = FNO(
        n_modes=n_modes,
        hidden_channels=hidden_channels,
        in_channels=in_channels,
        out_channels=out_channels,
        n_layers=4
    )
    model_mhf_in.fno_blocks.convs[0] = MHFSpectralConv(
        hidden_channels, hidden_channels, n_modes, n_heads=4
    )
    params_mhf_in = count_parameters(model_mhf_in)
    print(f"参数量: {params_mhf_in:,}")
    
    inf_mean, inf_std = measure_inference_time(model_mhf_in, sample_x[:1])
    print(f"推理延迟: {inf_mean:.2f} ± {inf_std:.2f} ms")
    
    start_time = time.time()
    result_mhf_in = train_model(model_mhf_in, train_loader, test_loader, epochs=epochs, lr=1e-3, device=device)
    total_time_mhf_in = time.time() - start_time
    
    results.append({
        'name': 'MHF-FNO (输入层)',
        'params': params_mhf_in,
        'train_loss': result_mhf_in['final_train'],
        'test_loss': result_mhf_in['final_test'],
        'best_test': result_mhf_in['best_test'],
        'epoch_time': result_mhf_in['avg_epoch_time'],
        'total_time': total_time_mhf_in,
        'inference_ms': inf_mean
    })
    
    del model_mhf_in
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # ============ 测试 3: MHF-FNO (混合层 - 最佳配置) ============
    print("\n" + "=" * 60)
    print("[3/5] MHF-FNO (混合层 - 第1+3层)")
    print("=" * 60)
    
    torch.manual_seed(42)
    model_mhf_mix = create_hybrid_fno(
        n_modes=n_modes,
        hidden_channels=hidden_channels,
        in_channels=in_channels,
        out_channels=out_channels,
        n_layers=4,
        mhf_layers=[0, 2],
        n_heads=4
    )
    params_mhf_mix = count_parameters(model_mhf_mix)
    print(f"参数量: {params_mhf_mix:,}")
    
    inf_mean, inf_std = measure_inference_time(model_mhf_mix, sample_x[:1])
    print(f"推理延迟: {inf_mean:.2f} ± {inf_std:.2f} ms")
    
    start_time = time.time()
    result_mhf_mix = train_model(model_mhf_mix, train_loader, test_loader, epochs=epochs, lr=1e-3, device=device)
    total_time_mhf_mix = time.time() - start_time
    
    results.append({
        'name': 'MHF-FNO (混合层)',
        'params': params_mhf_mix,
        'train_loss': result_mhf_mix['final_train'],
        'test_loss': result_mhf_mix['final_test'],
        'best_test': result_mhf_mix['best_test'],
        'epoch_time': result_mhf_mix['avg_epoch_time'],
        'total_time': total_time_mhf_mix,
        'inference_ms': inf_mean
    })
    
    del model_mhf_mix
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # ============ 测试 4: MHF-FNO (尺度初始化) ============
    print("\n" + "=" * 60)
    print("[4/5] MHF-FNO (尺度初始化)")
    print("=" * 60)
    
    torch.manual_seed(42)
    model_mhf_scale = FNO(
        n_modes=n_modes,
        hidden_channels=hidden_channels,
        in_channels=in_channels,
        out_channels=out_channels,
        n_layers=4
    )
    model_mhf_scale.fno_blocks.convs[0] = ScaleDiverseMHF(
        hidden_channels, hidden_channels, n_modes, n_heads=4
    )
    params_mhf_scale = count_parameters(model_mhf_scale)
    print(f"参数量: {params_mhf_scale:,}")
    
    inf_mean, inf_std = measure_inference_time(model_mhf_scale, sample_x[:1])
    print(f"推理延迟: {inf_mean:.2f} ± {inf_std:.2f} ms")
    
    start_time = time.time()
    result_mhf_scale = train_model(model_mhf_scale, train_loader, test_loader, epochs=epochs, lr=1e-3, device=device)
    total_time_mhf_scale = time.time() - start_time
    
    results.append({
        'name': 'MHF-FNO (尺度)',
        'params': params_mhf_scale,
        'train_loss': result_mhf_scale['final_train'],
        'test_loss': result_mhf_scale['final_test'],
        'best_test': result_mhf_scale['best_test'],
        'epoch_time': result_mhf_scale['avg_epoch_time'],
        'total_time': total_time_mhf_scale,
        'inference_ms': inf_mean
    })
    
    del model_mhf_scale
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # ============ 测试 5: 增大 hidden_channels 的 MHF-FNO ============
    print("\n" + "=" * 60)
    print("[5/5] MHF-FNO (增强版 - 匹配参数量)")
    print("=" * 60)
    
    torch.manual_seed(42)
    # 计算 hidden_channels 以匹配 FNO 参数量
    target_params = params_fno
    # 粗略估计: MHF-FNO 输入层参数约为 FNO 的 1/n_heads
    # 总参数主要由 hidden_channels 决定
    enhanced_hidden = 48  # 增大 hidden_channels
    
    model_mhf_enhanced = FNO(
        n_modes=n_modes,
        hidden_channels=enhanced_hidden,
        in_channels=in_channels,
        out_channels=out_channels,
        n_layers=4
    )
    model_mhf_enhanced.fno_blocks.convs[0] = MHFSpectralConv(
        enhanced_hidden, enhanced_hidden, n_modes, n_heads=4
    )
    params_mhf_enhanced = count_parameters(model_mhf_enhanced)
    print(f"参数量: {params_mhf_enhanced:,} (目标: ~{target_params:,})")
    
    inf_mean, inf_std = measure_inference_time(model_mhf_enhanced, sample_x[:1])
    print(f"推理延迟: {inf_mean:.2f} ± {inf_std:.2f} ms")
    
    start_time = time.time()
    result_mhf_enhanced = train_model(model_mhf_enhanced, train_loader, test_loader, epochs=epochs, lr=1e-3, device=device)
    total_time_mhf_enhanced = time.time() - start_time
    
    results.append({
        'name': 'MHF-FNO (增强版)',
        'params': params_mhf_enhanced,
        'train_loss': result_mhf_enhanced['final_train'],
        'test_loss': result_mhf_enhanced['final_test'],
        'best_test': result_mhf_enhanced['best_test'],
        'epoch_time': result_mhf_enhanced['avg_epoch_time'],
        'total_time': total_time_mhf_enhanced,
        'inference_ms': inf_mean
    })
    
    del model_mhf_enhanced
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # ============ 汇总结果 ============
    print("\n" + "=" * 80)
    print(" 📊 MHF-FNO 性能测试结果汇总")
    print("=" * 80)
    
    baseline = results[0]
    
    print(f"\n{'模型':<22} {'参数量':<12} {'训练Loss':<12} {'测试Loss':<12} {'最佳测试':<12} {'Epoch时间':<10} {'推理ms':<10}")
    print("-" * 100)
    
    for r in results:
        print(f"{r['name']:<22} {r['params']:<12,} {r['train_loss']:<12.6f} {r['test_loss']:<12.6f} {r['best_test']:<12.6f} {r['epoch_time']:<10.2f}s {r['inference_ms']:<10.2f}")
    
    print("\n" + "=" * 80)
    print(" 📈 与基准 (FNO) 对比")
    print("=" * 80)
    
    for r in results[1:]:
        param_reduce = (1 - r['params'] / baseline['params']) * 100
        test_diff = (r['test_loss'] - baseline['test_loss']) / baseline['test_loss'] * 100
        speed_diff = (r['epoch_time'] - baseline['epoch_time']) / baseline['epoch_time'] * 100
        inf_diff = (r['inference_ms'] - baseline['inference_ms']) / baseline['inference_ms'] * 100
        
        param_status = '✅' if param_reduce >= 30 else ('⚠️' if param_reduce > 0 else '❌')
        test_status = '✅' if test_diff <= 0 else '❌'
        speed_status = '✅' if abs(speed_diff) <= 10 else '⚠️'
        inf_status = '✅' if inf_diff <= 0 else '⚠️'
        
        print(f"\n{r['name']}:")
        print(f"  {param_status} 参数减少: {param_reduce:.1f}%")
        print(f"  {test_status} L2误差: {test_diff:+.2f}% (目标: ≤0%)")
        print(f"  {speed_status} 训练速度: {speed_diff:+.1f}% (目标: ±10%)")
        print(f"  {inf_status} 推理延迟: {inf_diff:+.1f}%")
    
    # 核心目标达成情况
    print("\n" + "=" * 80)
    print(" 🎯 核心目标达成情况")
    print("=" * 80)
    
    best_param_reduce = max((1 - r['params'] / baseline['params']) * 100 for r in results[1:])
    best_l2 = min(r['test_loss'] for r in results[1:])
    best_l2_diff = (best_l2 - baseline['test_loss']) / baseline['test_loss'] * 100
    
    goals_met = {
        'param': any((1 - r['params'] / baseline['params']) * 100 >= 30 for r in results[1:]),
        'l2': any(r['test_loss'] <= baseline['test_loss'] for r in results[1:]),
        'speed': any(abs((r['epoch_time'] - baseline['epoch_time']) / baseline['epoch_time'] * 100) <= 10 for r in results[1:])
    }
    
    print(f"\n1. 参数减少 30%+: {'✅ 达成' if goals_met['param'] else '❌ 未达成'} (最佳: {best_param_reduce:.1f}%)")
    print(f"2. L2 误差 ≤ FNO: {'✅ 达成' if goals_met['l2'] else '❌ 未达成'} (最佳差距: {best_l2_diff:+.2f}%)")
    print(f"3. 训练速度相当: {'✅ 达成' if goals_met['speed'] else '❌ 未达成'}")
    
    # 推荐配置
    print("\n" + "=" * 80)
    print(" 🏆 最佳配置推荐")
    print("=" * 80)
    
    # 找到最佳平衡点
    best_balance = None
    best_score = float('inf')
    
    for r in results[1:]:
        # 综合评分: 参数效率 + 精度 + 速度
        param_score = max(0, (1 - r['params'] / baseline['params']) * 100)  # 参数减少越多越好
        test_score = max(0, -((r['test_loss'] - baseline['test_loss']) / baseline['test_loss'] * 100))  # 精度提升越多越好
        speed_score = max(0, -((r['epoch_time'] - baseline['epoch_time']) / baseline['epoch_time'] * 100))  # 速度越快越好
        
        # 综合分数 (参数效率权重0.4, 精度权重0.4, 速度权重0.2)
        score = -(0.4 * param_score + 0.4 * test_score + 0.2 * speed_score)
        
        if score < best_score:
            best_score = score
            best_balance = r
    
    if best_balance:
        param_reduce = (1 - best_balance['params'] / baseline['params']) * 100
        test_diff = (best_balance['test_loss'] - baseline['test_loss']) / baseline['test_loss'] * 100
        
        print(f"\n推荐配置: {best_balance['name']}")
        print(f"  - 参数减少: {param_reduce:.1f}%")
        print(f"  - L2误差: {test_diff:+.2f}%")
        print(f"  - 训练速度: {best_balance['epoch_time']:.2f}s/epoch")
    
    # 保存结果
    timestamp = int(time.time())
    with open(f'ns_test_results_{timestamp}.txt', 'w') as f:
        f.write("MHF-FNO 性能测试结果\n")
        f.write(f"数据集: Darcy Flow 16x16\n")
        f.write(f"训练样本: 1000, 测试样本: 200\n")
        f.write(f"训练轮数: {epochs}\n")
        f.write(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"{'模型':<22} {'参数量':<12} {'训练Loss':<12} {'测试Loss':<12} {'Epoch时间':<10}\n")
        f.write("-" * 70 + "\n")
        for r in results:
            f.write(f"{r['name']:<22} {r['params']:<12,} {r['train_loss']:<12.6f} {r['test_loss']:<12.6f} {r['epoch_time']:<10.2f}s\n")
        
        f.write("\n与基准 FNO 对比:\n")
        for r in results[1:]:
            param_reduce = (1 - r['params'] / baseline['params']) * 100
            test_diff = (r['test_loss'] - baseline['test_loss']) / baseline['test_loss'] * 100
            f.write(f"\n{r['name']}:\n")
            f.write(f"  参数减少: {param_reduce:.1f}%\n")
            f.write(f"  L2误差变化: {test_diff:+.2f}%\n")
        
        f.write("\n核心目标达成情况:\n")
        f.write(f"1. 参数减少 30%+: {'达成' if goals_met['param'] else '未达成'}\n")
        f.write(f"2. L2误差 ≤ FNO: {'达成' if goals_met['l2'] else '未达成'}\n")
        f.write(f"3. 训练速度相当: {'达成' if goals_met['speed'] else '未达成'}\n")
    
    print(f"\n结果已保存到: ns_test_results_{timestamp}.txt")


if __name__ == "__main__":
    main()