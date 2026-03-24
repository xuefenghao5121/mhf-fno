#!/usr/bin/env python3
"""
MHF-FNO 优化配置验证测试

验证三种优化配置对 Burgers 数据集精度的影响：
1. n_heads=2 (减少注意力头)
2. n_modes=12 (增加模式数)
3. n_heads=2 + n_modes=12 (组合优化)

基准: n_heads=4, n_modes=8 (参数减少 29.4%, L2 变化 +304%)
"""

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# ============================================================================
# MHF-FNO 核心实现 (可配置版)
# ============================================================================

class MHFSpectralConv1D(nn.Module):
    """1D 多头频域卷积"""
    
    def __init__(self, in_channels, out_channels, n_modes, n_heads=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes = n_modes
        self.n_heads = n_heads
        
        # 确保 out_channels 可以被 n_heads 整除
        if out_channels % n_heads != 0:
            # 自动调整 n_heads
            n_heads = self._find_valid_n_heads(out_channels, n_heads)
            print(f"  ⚠️ 调整 n_heads: {self.n_heads} -> {n_heads} (out_channels={out_channels})")
            self.n_heads = n_heads
        
        self.head_channels = out_channels // n_heads
        
        # 频域权重 - 每个头有独立的权重矩阵
        self.weights = nn.Parameter(
            torch.randn(n_heads, in_channels, self.head_channels, n_modes, dtype=torch.cfloat)
        )
        
        # 初始化
        self._init_weights()
    
    def _find_valid_n_heads(self, out_channels, target_n_heads):
        """找到最大的有效 n_heads (能整除 out_channels)"""
        for h in [target_n_heads, 2, 1]:
            if out_channels % h == 0:
                return h
        return 1
    
    def _init_weights(self):
        with torch.no_grad():
            for h in range(self.n_heads):
                scale = 0.01 * (2 ** h)
                nn.init.normal_(self.weights[h], mean=0, std=scale)
    
    def forward(self, x):
        # x: (batch, in_channels, spatial)
        batch_size = x.shape[0]
        spatial_size = x.shape[-1]
        
        # FFT
        x_ft = torch.fft.rfft(x, dim=-1)  # (batch, in_channels, freq)
        
        # 多头频域处理
        out_ft = torch.zeros(
            batch_size, self.out_channels, x_ft.shape[-1],
            dtype=x_ft.dtype, device=x.device
        )
        
        for h in range(self.n_heads):
            start = h * self.head_channels
            end = start + self.head_channels
            
            for i in range(min(self.n_modes, x_ft.shape[-1])):
                out_ft[:, start:end, i] = torch.einsum(
                    'bi,io->bo',
                    x_ft[:, :, i],
                    self.weights[h, :, :, i]
                )
        
        # IFFT
        x = torch.fft.irfft(out_ft, n=spatial_size, dim=-1)
        
        return x


class MHFFNO1D(nn.Module):
    """1D MHF-FNO 模型 (可配置版)"""
    
    def __init__(self, n_modes, hidden_channels, in_channels, out_channels,
                 n_layers=3, n_heads=4):
        super().__init__()
        
        self.n_layers = n_layers
        self.hidden_channels = hidden_channels
        self.n_modes = n_modes
        self.n_heads = n_heads
        
        # 输入投影
        self.fc_in = nn.Linear(in_channels, hidden_channels)
        
        # FNO 层
        self.fno_blocks = nn.ModuleList()
        self.use_mhf_layers = []
        for i in range(n_layers):
            use_mhf = (i == 0 or i == n_layers - 1)
            self.use_mhf_layers.append(use_mhf)
            
            if use_mhf:
                conv = MHFSpectralConv1D(
                    hidden_channels, hidden_channels, n_modes, n_heads
                )
            else:
                conv = nn.Identity()
            
            block = nn.ModuleDict({
                'conv': conv,
                'w': nn.Conv1d(hidden_channels, hidden_channels, 1),
            })
            self.fno_blocks.append(block)
        
        # 输出投影
        self.fc_out = nn.Linear(hidden_channels, out_channels)
    
    def forward(self, x):
        # x: (batch, channels, spatial)
        # 输入投影
        x = self.fc_in(x.permute(0, 2, 1)).permute(0, 2, 1)
        
        # FNO 层
        for i, block in enumerate(self.fno_blocks):
            if self.use_mhf_layers[i]:
                x = x + block['conv'](x)
            x = x + block['w'](x)
        
        # 输出投影
        x = self.fc_out(x.permute(0, 2, 1)).permute(0, 2, 1)
        
        return x


class FNO1D(nn.Module):
    """标准 1D FNO 模型"""
    
    def __init__(self, n_modes, hidden_channels, in_channels, out_channels, n_layers=3):
        super().__init__()
        
        self.n_layers = n_layers
        self.hidden_channels = hidden_channels
        
        self.fc_in = nn.Linear(in_channels, hidden_channels)
        
        self.fno_blocks = nn.ModuleList()
        for _ in range(n_layers):
            self.fno_blocks.append(nn.ModuleDict({
                'spectral': SpectralConv1d(hidden_channels, hidden_channels, n_modes),
                'w': nn.Conv1d(hidden_channels, hidden_channels, 1),
            }))
        
        self.fc_out = nn.Linear(hidden_channels, out_channels)
    
    def forward(self, x):
        x = self.fc_in(x.permute(0, 2, 1)).permute(0, 2, 1)
        
        for block in self.fno_blocks:
            x = x + block['spectral'](x) + block['w'](x)
        
        x = self.fc_out(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x


class SpectralConv1d(nn.Module):
    """标准 1D 频域卷积"""
    
    def __init__(self, in_channels, out_channels, n_modes):
        super().__init__()
        self.n_modes = n_modes
        
        self.scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, n_modes, dtype=torch.cfloat)
        )
    
    def forward(self, x):
        batchsize = x.shape[0]
        
        x_ft = torch.fft.rfft(x)
        
        out_ft = torch.zeros(batchsize, x_ft.shape[-2], x_ft.shape[-1],
                             device=x.device, dtype=torch.cfloat)
        
        for i in range(min(self.n_modes, x_ft.shape[-1])):
            out_ft[:, :, i] = torch.einsum("bi,io->bo", x_ft[:, :, i], self.weights[:, :, i])
        
        x = torch.fft.irfft(out_ft, n=x.shape[-1])
        return x


# ============================================================================
# 加载数据
# ============================================================================

def load_burgers_data():
    """加载 Burgers 数据"""
    data_path = Path.home() / '.neuralop' / 'data'
    
    train_data = torch.load(data_path / 'burgers_train_16.pt')
    test_data = torch.load(data_path / 'burgers_test_16.pt')
    
    train_x = train_data['x']  # (n_train, 1, 16)
    train_y = train_data['y']  # (n_train, 32, 16)
    test_x = test_data['x']
    test_y = test_data['y']
    
    return train_x, train_y, test_x, test_y


# ============================================================================
# 训练
# ============================================================================

def train_model(model, train_x, train_y, test_x, test_y, config):
    """训练模型"""
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    
    results = {
        'train_losses': [],
        'test_losses': [],
        'epoch_times': [],
    }
    
    n_train = train_x.shape[0]
    batch_size = config['batch_size']
    
    for epoch in range(config['epochs']):
        t0 = time.time()
        model.train()
        
        perm = torch.randperm(n_train)
        train_loss = 0
        batch_count = 0
        
        for i in range(0, n_train, batch_size):
            bx = train_x[perm[i:i+batch_size]]
            by = train_y[perm[i:i+batch_size]]
            
            optimizer.zero_grad()
            pred = model(bx)
            
            # 预测单时间步
            loss = torch.mean((pred.squeeze(1) - by[:, 0, :]) ** 2)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            batch_count += 1
        
        scheduler.step()
        epoch_time = time.time() - t0
        
        # 测试
        model.eval()
        with torch.no_grad():
            pred = model(test_x)
            test_loss = torch.mean((pred.squeeze(1) - test_y[:, 0, :]) ** 2).item()
        
        results['train_losses'].append(train_loss / batch_count)
        results['test_losses'].append(test_loss)
        results['epoch_times'].append(epoch_time)
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{config['epochs']}: "
                  f"Train {train_loss/batch_count:.6f}, "
                  f"Test {test_loss:.6f}, "
                  f"Time {epoch_time:.1f}s")
    
    return results


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


# ============================================================================
# 测试配置
# ============================================================================

def test_configuration(name, n_modes, n_heads, train_x, train_y, test_x, test_y, 
                       base_params, base_l2, config):
    """测试单个配置"""
    print(f"\n{'='*60}")
    print(f"测试: {name}")
    print(f"  n_modes={n_modes}, n_heads={n_heads}")
    print(f"{'='*60}")
    
    torch.manual_seed(config['seed'])
    
    model = MHFFNO1D(
        n_modes=n_modes,
        hidden_channels=32,
        in_channels=1,
        out_channels=1,
        n_layers=3,
        n_heads=n_heads,
    )
    
    params = count_parameters(model)
    params_change = (params - base_params) / base_params * 100
    
    print(f"参数量: {params:,} ({params_change:+.1f}% vs FNO)")
    
    # 训练
    results = train_model(model, train_x, train_y, test_x, test_y, config)
    
    best_l2 = min(results['test_losses'])
    l2_change = (best_l2 - base_l2) / base_l2 * 100
    
    print(f"\n结果:")
    print(f"  最佳 L2: {best_l2:.6e} ({l2_change:+.1f}% vs FNO)")
    print(f"  平均 epoch 时间: {np.mean(results['epoch_times']):.2f}s")
    
    return {
        'name': name,
        'n_modes': n_modes,
        'n_heads': n_heads,
        'params': params,
        'params_change': params_change,
        'best_l2': best_l2,
        'l2_change': l2_change,
        'avg_epoch_time': np.mean(results['epoch_times']),
        'train_losses': results['train_losses'],
        'test_losses': results['test_losses'],
    }


def main():
    config = {
        'epochs': 50,
        'batch_size': 32,
        'learning_rate': 1e-3,
        'seed': 42,
    }
    
    print("="*60)
    print("MHF-FNO 优化配置验证测试")
    print("="*60)
    print(f"配置: {config}")
    print(f"目标: 验证 n_heads 和 n_modes 对精度的影响")
    
    # 加载数据
    print("\n📊 加载 Burgers 数据...")
    train_x, train_y, test_x, test_y = load_burgers_data()
    
    print(f"训练集: x {train_x.shape}, y {train_y.shape}")
    print(f"测试集: x {test_x.shape}, y {test_y.shape}")
    
    # ========== 先测试 FNO 基准 ==========
    print(f"\n{'='*60}")
    print("测试 FNO (基准)")
    print(f"{'='*60}")
    
    torch.manual_seed(config['seed'])
    model_fno = FNO1D(
        n_modes=8,
        hidden_channels=32,
        in_channels=1,
        out_channels=1,
        n_layers=3,
    )
    
    base_params = count_parameters(model_fno)
    print(f"参数量: {base_params:,}")
    
    results_fno = train_model(model_fno, train_x, train_y, test_x, test_y, config)
    base_l2 = min(results_fno['test_losses'])
    
    print(f"\nFNO 基准: L2 = {base_l2:.6e}")
    
    # ========== 测试当前配置 (基准) ==========
    print(f"\n{'='*60}")
    print("测试 MHF-FNO 当前配置 (n_heads=4, n_modes=8)")
    print(f"{'='*60}")
    
    torch.manual_seed(config['seed'])
    model_mhf_base = MHFFNO1D(
        n_modes=8,
        hidden_channels=32,
        in_channels=1,
        out_channels=1,
        n_layers=3,
        n_heads=4,
    )
    
    params_mhf_base = count_parameters(model_mhf_base)
    print(f"参数量: {params_mhf_base:,} ({(1-params_mhf_base/base_params)*100:.1f}% reduction)")
    
    results_mhf_base = train_model(model_mhf_base, train_x, train_y, test_x, test_y, config)
    l2_mhf_base = min(results_mhf_base['test_losses'])
    
    print(f"\nMHF-FNO 基准: L2 = {l2_mhf_base:.6e} ({(l2_mhf_base-base_l2)/base_l2*100:+.1f}% vs FNO)")
    
    # ========== 测试优化配置 ==========
    all_results = {
        'config': config,
        'timestamp': datetime.now().isoformat(),
        'base_fno': {
            'params': base_params,
            'best_l2': base_l2,
        },
        'current_mhf': {
            'params': params_mhf_base,
            'params_reduction': f"{(1-params_mhf_base/base_params)*100:.1f}%",
            'best_l2': l2_mhf_base,
            'l2_change_vs_fno': f"{(l2_mhf_base-base_l2)/base_l2*100:+.1f}%",
        },
        'optimization_tests': [],
    }
    
    # 测试 1: n_heads=2
    result1 = test_configuration(
        "优化1: n_heads=2",
        n_modes=8,
        n_heads=2,
        train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y,
        base_params=base_params, base_l2=base_l2, config=config
    )
    all_results['optimization_tests'].append(result1)
    
    # 测试 2: n_modes=12
    result2 = test_configuration(
        "优化2: n_modes=12",
        n_modes=12,
        n_heads=4,
        train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y,
        base_params=base_params, base_l2=base_l2, config=config
    )
    all_results['optimization_tests'].append(result2)
    
    # 测试 3: 组合优化
    result3 = test_configuration(
        "优化3: n_heads=2 + n_modes=12",
        n_modes=12,
        n_heads=2,
        train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y,
        base_params=base_params, base_l2=base_l2, config=config
    )
    all_results['optimization_tests'].append(result3)
    
    # ========== 分析结果 ==========
    print(f"\n{'='*60}")
    print("📊 优化验证结果汇总")
    print(f"{'='*60}")
    
    print(f"\n{'配置':<30} {'参数':<10} {'参数变化':<12} {'最佳L2':<15} {'L2变化':<12}")
    print("-"*80)
    print(f"{'FNO (基准)':<30} {base_params:<10,} {'--':<12} {base_l2:<15.6e} {'--':<12}")
    print(f"{'MHF-FNO 当前 (n_heads=4, n_modes=8)':<30} {params_mhf_base:<10,} "
          f"{(1-params_mhf_base/base_params)*100:+.1f}%{'':<6} {l2_mhf_base:<15.6e} "
          f"{(l2_mhf_base-base_l2)/base_l2*100:+.1f}%")
    
    for r in all_results['optimization_tests']:
        print(f"{r['name']:<30} {r['params']:<10,} {r['params_change']:+.1f}%{'':<6} "
              f"{r['best_l2']:<15.6e} {r['l2_change']:+.1f}%")
    
    # 分析结论
    print(f"\n{'='*60}")
    print("📋 分析结论")
    print(f"{'='*60}")
    
    # 检查优化是否有效
    current_l2 = l2_mhf_base
    
    # 找出最佳配置
    best_config = min(all_results['optimization_tests'], key=lambda x: x['best_l2'])
    
    if best_config['l2_change'] < current_l2 / base_l2 * 100 - 100:
        improvement = abs(best_config['l2_change'] - (l2_mhf_base/base_l2*100 - 100))
        print(f"✅ 优化有效: {best_config['name']} 相比当前配置 L2 改善 {improvement:.1f}%")
        print(f"   参数量: {best_config['params']:,} ({best_config['params_change']:+.1f}% vs FNO)")
        print(f"   L2: {best_config['best_l2']:.6e} ({best_config['l2_change']:+.1f}% vs FNO)")
    else:
        print(f"⚠️ 优化效果不明显")
        for r in all_results['optimization_tests']:
            if r['l2_change'] > l2_mhf_base / base_l2 * 100 - 100:
                print(f"   {r['name']}: L2 变化 {r['l2_change']:+.1f}% (未改善)")
    
    # 验证假设
    print(f"\n假设验证:")
    print(f"1. 小模型 n_heads=4 过多 → 减少 n_heads=2 应改善精度")
    heads2 = next(r for r in all_results['optimization_tests'] if 'n_heads=2' in r['name'] and 'n_modes=12' not in r['name'])
    if heads2['best_l2'] < l2_mhf_base:
        print(f"   ✅ n_heads=2 改善: {heads2['best_l2']:.6e} < {l2_mhf_base:.6e}")
    else:
        print(f"   ❌ n_heads=2 未改善: {heads2['best_l2']:.6e} >= {l2_mhf_base:.6e}")
    
    print(f"2. n_modes=8 对低分辨率不够 → 增加 n_modes=12 应改善精度")
    modes12 = next(r for r in all_results['optimization_tests'] if 'n_modes=12' in r['name'] and 'n_heads=2' not in r['name'])
    if modes12['best_l2'] < l2_mhf_base:
        print(f"   ✅ n_modes=12 改善: {modes12['best_l2']:.6e} < {l2_mhf_base:.6e}")
    else:
        print(f"   ❌ n_modes=12 未改善: {modes12['best_l2']:.6e} >= {l2_mhf_base:.6e}")
    
    print(f"3. 组合优化应达到最佳平衡")
    combo = next(r for r in all_results['optimization_tests'] if 'n_heads=2' in r['name'] and 'n_modes=12' in r['name'])
    if combo['best_l2'] < l2_mhf_base and combo['params'] < base_params:
        print(f"   ✅ 组合优化有效: L2={combo['best_l2']:.6e}, 参数={combo['params']:,}")
    else:
        print(f"   ⚠️ 组合优化需权衡: L2={combo['best_l2']:.6e}, 参数={combo['params']:,}")
    
    # 保存结果
    output_path = Path('optimization_validation_results.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✅ 结果已保存到: {output_path}")
    
    return all_results


if __name__ == '__main__':
    main()