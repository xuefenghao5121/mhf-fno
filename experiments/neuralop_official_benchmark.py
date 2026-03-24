"""
NeuralOperator 官方 Benchmark 测试

使用 NeuralOperator 官方 Darcy Flow 数据集进行严格测试

对比模型：
1. FNO (NeuralOperator 官方实现)
2. MHF-FNO (TransFourier 风格)

评估指标：
- 相对 L2 误差
- 训练时间
- 参数量
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
from typing import Dict, Tuple

# 设置路径
sys.path.insert(0, '/root/.openclaw/workspace/memory/projects/tianyuan-fft/experiments')

# 导入模型
from neuralop.models import FNO
from engram_mhf_fno_fixed import EngramMHFFNO

print("✅ 模型导入成功")


def load_official_darcy_data():
    """加载 NeuralOperator 官方 Darcy Flow 数据集"""
    
    data_path = '/usr/local/lib/python3.11/site-packages/neuralop/data/datasets/data/'
    
    train_data = torch.load(f'{data_path}/darcy_train_16.pt', weights_only=False)
    test_data = torch.load(f'{data_path}/darcy_test_16.pt', weights_only=False)
    
    # x: (N, H, W) - 输入场（扩散系数 a）
    # y: (N, H, W) - 输出场（解 u）
    # 注意：原始数据可能是 bool 类型，需要转换为 float32
    train_x = train_data['x'].unsqueeze(1).float()  # (N, 1, H, W)
    train_y = train_data['y'].unsqueeze(1).float()
    test_x = test_data['x'].unsqueeze(1).float()
    test_y = test_data['y'].unsqueeze(1).float()
    
    return train_x, train_y, test_x, test_y


class Trainer:
    """通用训练器"""
    
    def __init__(self, model, lr=1e-3):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=50, gamma=0.5
        )
    
    def train_epoch(self, x, y, batch_size=32):
        """训练一个 epoch"""
        self.model.train()
        n = x.shape[0]
        perm = torch.randperm(n)
        x, y = x[perm], y[perm]
        
        total_loss = 0.0
        n_batches = n // batch_size
        
        for i in range(n_batches):
            bx = x[i*batch_size:(i+1)*batch_size]
            by = y[i*batch_size:(i+1)*batch_size]
            
            self.optimizer.zero_grad()
            pred = self.model(bx)
            loss = F.mse_loss(pred, by)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        self.scheduler.step()
        return total_loss / n_batches
    
    def evaluate(self, x, y, batch_size=32):
        """评估模型"""
        self.model.eval()
        
        total_l2 = 0.0
        n = x.shape[0]
        n_batches = (n + batch_size - 1) // batch_size
        
        with torch.no_grad():
            for i in range(0, n, batch_size):
                bx = x[i:i+batch_size]
                by = y[i:i+batch_size]
                
                pred = self.model(bx)
                l2 = torch.norm(pred - by) / torch.norm(by)
                total_l2 += l2.item() * bx.shape[0]
        
        return total_l2 / n


def run_benchmark(
    n_train: int = 500,
    n_test: int = 100,
    epochs: int = 100,
    batch_size: int = 32,
    hidden_channels: int = 32,
    n_modes: int = 8,
    n_layers: int = 3,
    n_heads: int = 4,
):
    """运行官方 Benchmark"""
    
    print("\n" + "=" * 70)
    print(" NeuralOperator 官方 Darcy Flow Benchmark")
    print("=" * 70)
    
    # 加载数据
    print("\n📊 加载官方数据集...")
    train_x, train_y, test_x, test_y = load_official_darcy_data()
    
    # 截取指定数量
    train_x, train_y = train_x[:n_train], train_y[:n_train]
    test_x, test_y = test_x[:n_test], test_y[:n_test]
    
    print(f"   训练集: {train_x.shape}")
    print(f"   测试集: {test_x.shape}")
    
    results = {}
    
    # ===== 测试 FNO (官方) =====
    print("\n" + "-" * 50)
    print("🔬 测试 FNO (NeuralOperator 官方实现)")
    print("-" * 50)
    
    model_fno = FNO(
        n_modes=(n_modes, n_modes),
        hidden_channels=hidden_channels,
        in_channels=1,
        out_channels=1,
        n_layers=n_layers,
    )
    
    n_params_fno = sum(p.numel() for p in model_fno.parameters())
    print(f"   参数量: {n_params_fno:,}")
    
    trainer_fno = Trainer(model_fno)
    
    start_time = time.time()
    for epoch in range(epochs):
        train_loss = trainer_fno.train_epoch(train_x, train_y, batch_size)
        if (epoch + 1) % 20 == 0:
            val_l2 = trainer_fno.evaluate(test_x, test_y, batch_size)
            print(f"   Epoch {epoch+1}: Loss={train_loss:.6f}, L2={val_l2:.4f}")
    train_time_fno = time.time() - start_time
    
    final_l2_fno = trainer_fno.evaluate(test_x, test_y, batch_size)
    print(f"\n   ✅ 最终 L2 误差: {final_l2_fno:.4f}")
    print(f"   ⏱️ 训练时间: {train_time_fno:.2f}s")
    
    results['FNO'] = {
        'params': n_params_fno,
        'l2': final_l2_fno,
        'time': train_time_fno,
    }
    
    # ===== 测试 Engram-MHF-FNO =====
    print("\n" + "-" * 50)
    print("🔬 测试 Engram-MHF-FNO (完整架构)")
    print("-" * 50)
    
    model_engram = EngramMHFFNO(
        in_channels=1,
        out_channels=1,
        hidden_channels=hidden_channels,
        n_modes=(n_modes, n_modes),
        n_layers=n_layers,
        n_heads=n_heads,
        window_size=3,
        num_patterns=1000,  # 减少 pattern 数量
    )
    
    n_params_engram = sum(p.numel() for p in model_engram.parameters())
    print(f"   参数量: {n_params_engram:,}")
    
    trainer_engram = Trainer(model_engram)
    
    start_time = time.time()
    for epoch in range(epochs):
        train_loss = trainer_engram.train_epoch(train_x, train_y, batch_size)
        if (epoch + 1) % 20 == 0:
            val_l2 = trainer_engram.evaluate(test_x, test_y, batch_size)
            print(f"   Epoch {epoch+1}: Loss={train_loss:.6f}, L2={val_l2:.4f}")
    train_time_engram = time.time() - start_time
    
    final_l2_engram = trainer_engram.evaluate(test_x, test_y, batch_size)
    print(f"\n   ✅ 最终 L2 误差: {final_l2_engram:.4f}")
    print(f"   ⏱️ 训练时间: {train_time_engram:.2f}s")
    
    results['Engram-MHF-FNO'] = {
        'params': n_params_engram,
        'l2': final_l2_engram,
        'time': train_time_engram,
    }
    
    # ===== 汇总对比 =====
    print("\n" + "=" * 70)
    print(" 📈 汇总对比")
    print("=" * 70)
    
    print(f"\n{'模型':<15} {'参数量':<15} {'L2误差':<12} {'训练时间':<12}")
    print("-" * 60)
    
    for name, res in results.items():
        print(f"{name:<15} {res['params']:<15,} {res['l2']:<12.4f} {res['time']:<12.2f}s")
    
    # 计算改进百分比
    params_change = (results['Engram-MHF-FNO']['params'] - results['FNO']['params']) / results['FNO']['params'] * 100
    l2_change = (results['Engram-MHF-FNO']['l2'] - results['FNO']['l2']) / results['FNO']['l2'] * 100
    time_change = (results['Engram-MHF-FNO']['time'] - results['FNO']['time']) / results['FNO']['time'] * 100
    
    print(f"\n📊 Engram-MHF-FNO vs FNO:")
    print(f"   参数量变化: {params_change:+.1f}%")
    print(f"   L2 误差变化: {l2_change:+.2f}%")
    print(f"   训练时间变化: {time_change:+.2f}%")
    
    # 判断测试是否通过
    print("\n" + "=" * 70)
    if l2_change <= 5:  # L2 误差增加不超过 5%
        print(" ✅ Engram-MHF-FNO 精度达标 (L2 误差增加 < 5%)")
    else:
        print(f" ⚠️ Engram-MHF-FNO 精度需改进 (L2 误差增加 {l2_change:.1f}%)")
    
    if params_change < 0:
        print(f" ✅ Engram-MHF-FNO 参数效率更高 (减少 {-params_change:.1f}%)")
    
    return results


if __name__ == "__main__":
    # 运行 benchmark
    results = run_benchmark(
        n_train=500,
        n_test=100,
        epochs=100,
        batch_size=32,
        hidden_channels=32,
        n_modes=8,
        n_layers=3,
        n_heads=4,
    )