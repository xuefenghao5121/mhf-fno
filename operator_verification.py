"""
NeuralOperator 算子 MHF 验证套件

逐个算子验证 MHF 的适用性：
1. FNO - 已验证，补充更多测试
2. TFNO - 检查是否支持 MHF
3. UNO - 检查多尺度位置
"""

import torch
import torch.nn as nn
import time
import json
import gc
import numpy as np
from typing import Dict, List, Tuple, Optional

from neuralop.models import FNO, TFNO, UNO
from neuralop.layers.spectral_convolution import SpectralConv
from neuralop.losses.data_losses import LpLoss


# ============================================
# MHF 核心实现
# ============================================

class MHFSpectralConv(SpectralConv):
    """MHF SpectralConv - 兼容 NeuralOperator API"""
    
    def __init__(self, in_channels, out_channels, n_modes, n_heads=4, bias=True, **kwargs):
        super().__init__(in_channels=in_channels, out_channels=out_channels, n_modes=n_modes, bias=bias, **kwargs)
        
        del self.weight
        self.n_heads = n_heads
        self.use_mhf = (in_channels % n_heads == 0 and out_channels % n_heads == 0)
        
        if self.use_mhf:
            self.head_in = in_channels // n_heads
            self.head_out = out_channels // n_heads
            
            if len(n_modes) == 1:
                self.modes_list = [n_modes[0]]
                weight_shape = (n_heads, self.head_in, self.head_out, n_modes[0])
            else:
                self.modes_list = list(n_modes)
                modes_y = n_modes[1] // 2 + 1 if len(n_modes) > 1 else n_modes[0]
                weight_shape = (n_heads, self.head_in, self.head_out, n_modes[0], modes_y)
            
            init_std = (2 / (in_channels + out_channels)) ** 0.5
            self.weight = nn.Parameter(torch.randn(*weight_shape, dtype=torch.cfloat) * init_std)
        else:
            self.weight = nn.Parameter(torch.randn(in_channels, out_channels, *n_modes, dtype=torch.cfloat) * 0.01)
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels, *((1,) * len(n_modes))))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x, output_shape=None, *args, **kwargs):
        if not self.use_mhf:
            return self._forward_standard(x)
        if x.dim() == 3:
            return self._forward_1d(x)
        return self._forward_2d(x)
    
    def _forward_1d(self, x):
        B, C, L = x.shape
        x_freq = torch.fft.rfft(x, dim=-1)
        n_modes = min(self.modes_list[0], x_freq.shape[-1])
        x_freq = x_freq.view(B, self.n_heads, self.head_in, -1)
        out_freq = torch.zeros(B, self.n_heads, self.head_out, x_freq.shape[-1], dtype=x_freq.dtype, device=x.device)
        out_freq[..., :n_modes] = torch.einsum('bhif,hiof->bhof', x_freq[..., :n_modes], self.weight[..., :n_modes])
        out_freq = out_freq.reshape(B, self.out_channels, -1)
        x_out = torch.fft.irfft(out_freq, n=L, dim=-1)
        if self.bias is not None:
            x_out = x_out + self.bias
        return x_out
    
    def _forward_2d(self, x):
        B, C, H, W = x.shape
        x_freq = torch.fft.rfft2(x, dim=(-2, -1))
        freq_H, freq_W = x_freq.shape[-2], x_freq.shape[-1]
        m_x = min(self.modes_list[0], freq_H)
        m_y = min(self.weight.shape[-1], freq_W)
        x_freq = x_freq.view(B, self.n_heads, self.head_in, freq_H, freq_W)
        out_freq = torch.zeros(B, self.n_heads, self.head_out, freq_H, freq_W, dtype=x_freq.dtype, device=x.device)
        out_freq[:, :, :, :m_x, :m_y] = torch.einsum('bhiXY,hioXY->bhoXY', x_freq[:, :, :, :m_x, :m_y], self.weight[:, :, :, :m_x, :m_y])
        out_freq = out_freq.reshape(B, self.out_channels, freq_H, freq_W)
        x_out = torch.fft.irfft2(out_freq, s=(H, W), dim=(-2, -1))
        if self.bias is not None:
            x_out = x_out + self.bias
        return x_out
    
    def _forward_standard(self, x):
        if x.dim() == 3:
            B, C, L = x.shape
            x_freq = torch.fft.rfft(x, dim=-1)
            n_modes = min(self.weight.shape[-1], x_freq.shape[-1])
            out_freq = torch.zeros(B, self.out_channels, x_freq.shape[-1], dtype=x_freq.dtype, device=x.device)
            out_freq[..., :n_modes] = torch.einsum('bif,iOf->bOf', x_freq[..., :n_modes], self.weight[..., :n_modes])
            x_out = torch.fft.irfft(out_freq, n=L, dim=-1)
        else:
            B, C, H, W = x.shape
            x_freq = torch.fft.rfft2(x, dim=(-2, -1))
            freq_H, freq_W = x_freq.shape[-2], x_freq.shape[-1]
            m_x = min(self.weight.shape[-2], freq_H)
            m_y = min(self.weight.shape[-1], freq_W)
            out_freq = torch.zeros(B, self.out_channels, freq_H, freq_W, dtype=x_freq.dtype, device=x.device)
            out_freq[:, :, :m_x, :m_y] = torch.einsum('biXY,ioXY->boXY', x_freq[:, :, :m_x, :m_y], self.weight[:, :, :m_x, :m_y])
            x_out = torch.fft.irfft2(out_freq, s=(H, W), dim=(-2, -1))
        if self.bias is not None:
            x_out = x_out + self.bias
        return x_out


# ============================================
# 验证工具
# ============================================

def count_params(model):
    return sum(p.numel() for p in model.parameters())


def train_and_evaluate(model, train_x, train_y, test_x, test_y, epochs=100, batch_size=32, lr=1e-3):
    """训练并评估模型"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = LpLoss(d=2, p=2, reduction='mean')
    
    n_train = train_x.shape[0]
    start = time.time()
    best_loss = float('inf')
    
    for epoch in range(epochs):
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
        
        model.eval()
        with torch.no_grad():
            test_loss = loss_fn(model(test_x), test_y).item()
        if test_loss < best_loss:
            best_loss = test_loss
    
    train_time = time.time() - start
    return best_loss, train_time


# ============================================
# 算子验证测试
# ============================================

def test_fno_mhf(train_x, train_y, test_x, test_y):
    """
    测试 1: FNO + MHF
    
    已知结论: 边缘层 MHF 效果最好
    """
    print("\n" + "=" * 70)
    print("算子 1: FNO + MHF 验证")
    print("=" * 70)
    
    results = []
    
    # 基准 FNO
    print("\n[1/3] 标准 FNO...")
    torch.manual_seed(42)
    model_fno = FNO(n_modes=(8, 8), hidden_channels=32, in_channels=1, out_channels=1, n_layers=3)
    params_fno = count_params(model_fno)
    loss_fno, time_fno = train_and_evaluate(model_fno, train_x, train_y, test_x, test_y, epochs=100)
    print(f"  参数: {params_fno:,}, Loss: {loss_fno:.4f}, 时间: {time_fno:.1f}s")
    results.append({"name": "FNO-基准", "params": params_fno, "loss": loss_fno, "time": time_fno})
    del model_fno
    gc.collect()
    
    # MHF-FNO (边缘层)
    print("\n[2/3] MHF-FNO (边缘层, n_heads=4)...")
    torch.manual_seed(42)
    model_mhf = FNO(n_modes=(8, 8), hidden_channels=32, in_channels=1, out_channels=1, n_layers=3)
    # 替换第 0 和 2 层
    model_mhf.fno_blocks.convs[0] = MHFSpectralConv(32, 32, (8, 8), n_heads=4)
    model_mhf.fno_blocks.convs[2] = MHFSpectralConv(32, 32, (8, 8), n_heads=4)
    params_mhf = count_params(model_mhf)
    loss_mhf, time_mhf = train_and_evaluate(model_mhf, train_x, train_y, test_x, test_y, epochs=100)
    print(f"  参数: {params_mhf:,}, Loss: {loss_mhf:.4f}, 时间: {time_mhf:.1f}s")
    results.append({"name": "MHF-FNO-边缘", "params": params_mhf, "loss": loss_mhf, "time": time_mhf})
    del model_mhf
    gc.collect()
    
    # MHF-FNO (全部层)
    print("\n[3/3] MHF-FNO (全部层, n_heads=4)...")
    torch.manual_seed(42)
    model_mhf_full = FNO(n_modes=(8, 8), hidden_channels=32, in_channels=1, out_channels=1, n_layers=3)
    for i in range(3):
        model_mhf_full.fno_blocks.convs[i] = MHFSpectralConv(32, 32, (8, 8), n_heads=4)
    params_mhf_full = count_params(model_mhf_full)
    loss_mhf_full, time_mhf_full = train_and_evaluate(model_mhf_full, train_x, train_y, test_x, test_y, epochs=100)
    print(f"  参数: {params_mhf_full:,}, Loss: {loss_mhf_full:.4f}, 时间: {time_mhf_full:.1f}s")
    results.append({"name": "MHF-FNO-全部", "params": params_mhf_full, "loss": loss_mhf_full, "time": time_mhf_full})
    del model_mhf_full
    gc.collect()
    
    # 汇总
    print("\n" + "-" * 70)
    print("FNO 验证结果:")
    print(f"{'配置':<20} {'参数量':<12} {'L2误差':<10} {'改进':<10}")
    print("-" * 55)
    baseline = results[0]['loss']
    for r in results:
        improve = (r['loss'] - baseline) / baseline * 100
        print(f"{r['name']:<20} {r['params']:<12,} {r['loss']:<10.4f} {improve:+.1f}%")
    
    return results


def test_tfno_mhf(train_x, train_y, test_x, test_y):
    """
    测试 2: TFNO + MHF
    
    TFNO 使用张量化压缩，需要检查是否支持 MHF
    """
    print("\n" + "=" * 70)
    print("算子 2: TFNO + MHF 验证")
    print("=" * 70)
    
    results = []
    
    # 检查 TFNO 结构
    print("\n[检查] TFNO 内部结构...")
    try:
        torch.manual_seed(42)
        model_tfno = TFNO(n_modes=(8, 8), hidden_channels=32, in_channels=1, out_channels=1, n_layers=3)
        
        # 检查是否有 fno_blocks
        if hasattr(model_tfno, 'fno_blocks'):
            print("  ✅ TFNO 有 fno_blocks 属性")
            if hasattr(model_tfno.fno_blocks, 'convs'):
                print(f"  ✅ 有 convs 层: {len(model_tfno.fno_blocks.convs)} 层")
        
        # 基准 TFNO
        print("\n[1/2] 标准 TFNO...")
        params_tfno = count_params(model_tfno)
        loss_tfno, time_tfno = train_and_evaluate(model_tfno, train_x, train_y, test_x, test_y, epochs=100)
        print(f"  参数: {params_tfno:,}, Loss: {loss_tfno:.4f}, 时间: {time_tfno:.1f}s")
        results.append({"name": "TFNO-基准", "params": params_tfno, "loss": loss_tfno, "time": time_tfno})
        
        # 尝试添加 MHF
        print("\n[2/2] TFNO + MHF (边缘层)...")
        del model_tfno
        gc.collect()
        
        torch.manual_seed(42)
        model_tfno_mhf = TFNO(n_modes=(8, 8), hidden_channels=32, in_channels=1, out_channels=1, n_layers=3)
        
        # 尝试替换
        try:
            model_tfno_mhf.fno_blocks.convs[0] = MHFSpectralConv(32, 32, (8, 8), n_heads=4)
            model_tfno_mhf.fno_blocks.convs[2] = MHFSpectralConv(32, 32, (8, 8), n_heads=4)
            print("  ✅ MHF 替换成功")
            
            params_mhf = count_params(model_tfno_mhf)
            loss_mhf, time_mhf = train_and_evaluate(model_tfno_mhf, train_x, train_y, test_x, test_y, epochs=100)
            print(f"  参数: {params_mhf:,}, Loss: {loss_mhf:.4f}, 时间: {time_mhf:.1f}s")
            results.append({"name": "TFNO+MHF-边缘", "params": params_mhf, "loss": loss_mhf, "time": time_mhf})
        except Exception as e:
            print(f"  ❌ MHF 替换失败: {e}")
            results.append({"name": "TFNO+MHF", "error": str(e)})
        
        del model_tfno_mhf
        gc.collect()
        
    except Exception as e:
        print(f"  ❌ TFNO 初始化失败: {e}")
        results.append({"name": "TFNO", "error": str(e)})
    
    # 汇总
    if results and 'error' not in results[0]:
        print("\n" + "-" * 70)
        print("TFNO 验证结果:")
        print(f"{'配置':<20} {'参数量':<12} {'L2误差':<10} {'改进':<10}")
        print("-" * 55)
        baseline = results[0]['loss']
        for r in results:
            if 'error' not in r:
                improve = (r['loss'] - baseline) / baseline * 100
                print(f"{r['name']:<20} {r['params']:<12,} {r['loss']:<10.4f} {improve:+.1f}%")
            else:
                print(f"{r['name']:<20} ERROR: {r['error']}")
    
    return results


def test_uno_mhf(train_x, train_y, test_x, test_y):
    """
    测试 3: UNO + MHF
    
    UNO 有多尺度结构，需要确定最佳 MHF 位置
    """
    print("\n" + "=" * 70)
    print("算子 3: UNO + MHF 验证")
    print("=" * 70)
    
    results = []
    
    # 检查 UNO 结构
    print("\n[检查] UNO 内部结构...")
    try:
        torch.manual_seed(42)
        model_uno = UNO(
            in_channels=1,
            out_channels=1,
            hidden_channels=32,
            lifting_channels=256,
            projection_channels=256,
            n_layers=5,
            uno_out_channels=[32, 64, 64, 32],  # 多尺度输出通道
            uno_n_modes=[(8,8), (8,8), (8,8), (8,8)]  # 多尺度模式数
        )
        
        # 检查结构
        if hasattr(model_uno, 'fno_blocks'):
            print("  ✅ UNO 有 fno_blocks 属性")
            if hasattr(model_uno.fno_blocks, 'convs'):
                n_convs = len(model_uno.fno_blocks.convs)
                print(f"  ✅ 有 {n_convs} 个 conv 层")
                # 打印每层的形状
                for i, conv in enumerate(model_uno.fno_blocks.convs):
                    if hasattr(conv, 'in_channels'):
                        print(f"     层{i}: in={conv.in_channels}, out={conv.out_channels}")
        
        # 基准 UNO
        print("\n[1/3] 标准 UNO...")
        params_uno = count_params(model_uno)
        loss_uno, time_uno = train_and_evaluate(model_uno, train_x, train_y, test_x, test_y, epochs=100)
        print(f"  参数: {params_uno:,}, Loss: {loss_uno:.4f}, 时间: {time_uno:.1f}s")
        results.append({"name": "UNO-基准", "params": params_uno, "loss": loss_uno, "time": time_uno})
        
        del model_uno
        gc.collect()
        
        # UNO + MHF (瓶颈层)
        print("\n[2/3] UNO + MHF (瓶颈层)...")
        torch.manual_seed(42)
        model_uno_bottleneck = UNO(
            in_channels=1,
            out_channels=1,
            hidden_channels=32,
            lifting_channels=256,
            projection_channels=256,
            n_layers=5,
            uno_out_channels=[32, 64, 64, 32],
            uno_n_modes=[(8,8), (8,8), (8,8), (8,8)]
        )
        
        # 找到中间层
        n_layers = len(model_uno_bottleneck.fno_blocks.convs)
        mid_layer = n_layers // 2
        
        try:
            # 只替换中间层
            conv = model_uno_bottleneck.fno_blocks.convs[mid_layer]
            if hasattr(conv, 'in_channels') and conv.in_channels % 4 == 0:
                model_uno_bottleneck.fno_blocks.convs[mid_layer] = MHFSpectralConv(
                    conv.in_channels, conv.out_channels, (8, 8), n_heads=4
                )
                print(f"  ✅ 替换中间层 {mid_layer}")
            else:
                print(f"  ⚠️ 中间层通道数不能被 4 整除")
            
            params_bottleneck = count_params(model_uno_bottleneck)
            loss_bottleneck, time_bottleneck = train_and_evaluate(model_uno_bottleneck, train_x, train_y, test_x, test_y, epochs=100)
            print(f"  参数: {params_bottleneck:,}, Loss: {loss_bottleneck:.4f}, 时间: {time_bottleneck:.1f}s")
            results.append({"name": "UNO+MHF-瓶颈", "params": params_bottleneck, "loss": loss_bottleneck, "time": time_bottleneck})
        except Exception as e:
            print(f"  ❌ MHF 替换失败: {e}")
            results.append({"name": "UNO+MHF-瓶颈", "error": str(e)})
        
        del model_uno_bottleneck
        gc.collect()
        
        # UNO + MHF (边缘层)
        print("\n[3/3] UNO + MHF (边缘层)...")
        torch.manual_seed(42)
        model_uno_edge = UNO(
            in_channels=1,
            out_channels=1,
            hidden_channels=32,
            lifting_channels=256,
            projection_channels=256,
            n_layers=5,
            uno_out_channels=[32, 64, 64, 32],
            uno_n_modes=[(8,8), (8,8), (8,8), (8,8)]
        )
        
        try:
            # 替换第一层和最后一层
            for idx in [0, n_layers-1]:
                conv = model_uno_edge.fno_blocks.convs[idx]
                if hasattr(conv, 'in_channels') and conv.in_channels % 4 == 0:
                    model_uno_edge.fno_blocks.convs[idx] = MHFSpectralConv(
                        conv.in_channels, conv.out_channels, (8, 8), n_heads=4
                    )
                    print(f"  ✅ 替换层 {idx}")
            
            params_edge = count_params(model_uno_edge)
            loss_edge, time_edge = train_and_evaluate(model_uno_edge, train_x, train_y, test_x, test_y, epochs=100)
            print(f"  参数: {params_edge:,}, Loss: {loss_edge:.4f}, 时间: {time_edge:.1f}s")
            results.append({"name": "UNO+MHF-边缘", "params": params_edge, "loss": loss_edge, "time": time_edge})
        except Exception as e:
            print(f"  ❌ MHF 替换失败: {e}")
            results.append({"name": "UNO+MHF-边缘", "error": str(e)})
        
        del model_uno_edge
        gc.collect()
        
    except Exception as e:
        print(f"  ❌ UNO 初始化失败: {e}")
        results.append({"name": "UNO", "error": str(e)})
    
    # 汇总
    if results and 'error' not in results[0]:
        print("\n" + "-" * 70)
        print("UNO 验证结果:")
        print(f"{'配置':<20} {'参数量':<12} {'L2误差':<10} {'改进':<10}")
        print("-" * 55)
        baseline = results[0]['loss']
        for r in results:
            if 'error' not in r:
                improve = (r['loss'] - baseline) / baseline * 100
                print(f"{r['name']:<20} {r['params']:<12,} {r['loss']:<10.4f} {improve:+.1f}%")
            else:
                print(f"{r['name']:<20} ERROR: {r['error']}")
    
    return results


def main():
    """运行所有算子验证"""
    print("\n" + "=" * 70)
    print(" NeuralOperator 算子 MHF 验证套件")
    print("=" * 70)
    
    # 加载数据
    data_path = '/usr/local/lib/python3.11/site-packages/neuralop/data/datasets/data/'
    train_data = torch.load(f'{data_path}/darcy_train_16.pt', weights_only=False)
    test_data = torch.load(f'{data_path}/darcy_test_16.pt', weights_only=False)
    
    train_x = train_data['x'].unsqueeze(1).float()
    train_y = train_data['y'].unsqueeze(1).float()
    test_x = test_data['x'].unsqueeze(1).float()
    test_y = test_data['y'].unsqueeze(1).float()
    
    print(f"数据: {train_x.shape}")
    
    all_results = {}
    
    # 测试 1: FNO
    all_results['fno'] = test_fno_mhf(train_x, train_y, test_x, test_y)
    
    # 测试 2: TFNO
    all_results['tfno'] = test_tfno_mhf(train_x, train_y, test_x, test_y)
    
    # 测试 3: UNO
    all_results['uno'] = test_uno_mhf(train_x, train_y, test_x, test_y)
    
    # 最终汇总
    print("\n" + "=" * 70)
    print(" ✅ 所有算子验证完成")
    print("=" * 70)
    
    # 保存结果
    with open('operator_verification_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(" 结果保存到: operator_verification_results.json")
    
    return all_results


if __name__ == "__main__":
    main()