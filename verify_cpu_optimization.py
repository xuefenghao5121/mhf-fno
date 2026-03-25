#!/usr/bin/env python3
"""验证 CPU 优化效果

运行方式:
    python verify_cpu_optimization.py
"""
import os
import torch
import time
import numpy as np

print("="*60)
print("CPU 优化验证")
print("="*60)

# 检查线程配置
print(f"\n📊 CPU 核心数: {os.cpu_count()}")
print(f"📊 PyTorch 默认线程数: {torch.get_num_threads()}")

# 设置优化
torch.set_num_threads(os.cpu_count())
print(f"✅ 优化后线程数: {torch.get_num_threads()}")

# 测试矩阵乘法性能
def benchmark_matmul(size, n_runs=10):
    a = torch.randn(size, size)
    b = torch.randn(size, size)
    
    # 预热
    _ = a @ b
    
    # 计时
    times = []
    for _ in range(n_runs):
        t0 = time.time()
        _ = a @ b
        times.append(time.time() - t0)
    
    return np.mean(times) * 1000  # ms

print(f"\n{'='*60}")
print("矩阵乘法性能测试")
print("="*60)

size = 1024
print(f"\n矩阵大小: {size}x{size}")

torch.set_num_threads(2)
time_2_threads = benchmark_matmul(size)
print(f"  🔴 2 线程: {time_2_threads:.2f} ms")

torch.set_num_threads(os.cpu_count())
time_n_threads = benchmark_matmul(size)
print(f"  🟢 {os.cpu_count()} 线程: {time_n_threads:.2f} ms")
print(f"  ⚡ 加速比: {time_2_threads / time_n_threads:.2f}x")

# 测试 FFT 性能
def benchmark_fft(size, n_runs=10):
    x = torch.randn(1, 32, size, size)
    
    # 预热
    _ = torch.fft.rfft2(x)
    
    times = []
    for _ in range(n_runs):
        t0 = time.time()
        _ = torch.fft.rfft2(x)
        times.append(time.time() - t0)
    
    return np.mean(times) * 1000

print(f"\n{'='*60}")
print("FFT 性能测试 (MHF-FNO 核心操作)")
print("="*60)

size = 64
print(f"\n张量大小: [1, 32, {size}, {size}]")

torch.set_num_threads(2)
time_2_threads = benchmark_fft(size)
print(f"  🔴 2 线程: {time_2_threads:.2f} ms")

torch.set_num_threads(os.cpu_count())
time_n_threads = benchmark_fft(size)
print(f"  🟢 {os.cpu_count()} 线程: {time_n_threads:.2f} ms")
print(f"  ⚡ 加速比: {time_2_threads / time_n_threads:.2f}x")

# 测试完整 MHF-FNO 前向传播
print(f"\n{'='*60}")
print("MHF-FNO 前向传播性能测试")
print("="*60)

try:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from mhf_fno import MHFFNO
    
    model = MHFFNO.best_config(n_modes=(8, 8), hidden_channels=32)
    x = torch.randn(16, 1, 32, 32)
    
    # 预热
    with torch.no_grad():
        _ = model(x)
    
    def benchmark_forward(n_runs=10):
        times = []
        with torch.no_grad():
            for _ in range(n_runs):
                t0 = time.time()
                _ = model(x)
                times.append(time.time() - t0)
        return np.mean(times) * 1000
    
    torch.set_num_threads(2)
    time_2_threads = benchmark_forward()
    print(f"\nBatch size: 16, Resolution: 32x32")
    print(f"  🔴 2 线程: {time_2_threads:.2f} ms")
    
    torch.set_num_threads(os.cpu_count())
    time_n_threads = benchmark_forward()
    print(f"  🟢 {os.cpu_count()} 线程: {time_n_threads:.2f} ms")
    print(f"  ⚡ 加速比: {time_2_threads / time_n_threads:.2f}x")
    
except Exception as e:
    print(f"⚠️  无法测试 MHF-FNO: {e}")

# 总结
print(f"\n{'='*60}")
print("验证总结")
print("="*60)

torch.set_num_threads(os.cpu_count())
print(f"""
✅ CPU 优化已启用

配置信息:
  - CPU 核心数: {os.cpu_count()}
  - PyTorch 线程数: {torch.get_num_threads()}
  - OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'Not set')}

⚠️  当前机器只有 {os.cpu_count()} 个核心
   如果您在 256 核机器上运行，请确保:
   1. 检查 os.cpu_count() 返回 256
   2. PyTorch 线程数设置为 256
   3. CPU 利用率应该接近 100%

建议:
  1. 在训练脚本开头添加:
     torch.set_num_threads(os.cpu_count())
     
  2. 或设置环境变量:
     export OMP_NUM_THREADS=$(nproc)
     export MKL_NUM_THREADS=$(nproc)
     
  3. 运行时使用:
     ./run_cpu_optimized.sh python your_script.py
""")

print("="*60)