#!/usr/bin/env python3
"""
快速测试三种架构变体
"""

import torch
import torch.nn as nn
import numpy as np
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

# 导入MHF-FNO with CoDA
from mhf_fno import create_mhf_fno_with_attention, CrossHeadAttention
from neuralop.models import FNO

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_variant_a():
    """变体A: 所有层都加CoDA"""
    print("\n🔬 测试变体A: 所有Fourier层都加CoDA（全连接）")
    
    model = create_mhf_fno_with_attention(
        n_modes=(8, 8),
        hidden_channels=32,
        in_channels=1,
        out_channels=1,
        n_layers=4,
        mhf_layers=[0, 1, 2, 3],
        n_heads=2,
        attention_layers=[0, 1, 2, 3]
    )
    
    params = count_parameters(model)
    print(f"参数量: {params:,}")
    
    # 测试前向传播
    x = torch.randn(2, 1, 32, 32)
    y = model(x)
    print(f"前向传播: 输入 {x.shape} -> 输出 {y.shape}")
    
    return model, params

def test_variant_b():
    """变体B: 渐进式CoDA"""
    print("\n🔬 测试变体B: 只在最后2层加CoDA（渐进式）")
    
    model = create_mhf_fno_with_attention(
        n_modes=(8, 8),
        hidden_channels=32,
        in_channels=1,
        out_channels=1,
        n_layers=4,
        mhf_layers=[0, 1, 2, 3],
        n_heads=2,
        attention_layers=[2, 3]  # 只在最后2层加CoDA
    )
    
    params = count_parameters(model)
    print(f"参数量: {params:,}")
    
    # 测试前向传播
    x = torch.randn(2, 1, 32, 32)
    y = model(x)
    print(f"前向传播: 输入 {x.shape} -> 输出 {y.shape}")
    
    return model, params

def test_variant_c():
    """变体C: 并行CoDA"""
    print("\n🔬 测试变体C: CoDA作为独立模块，与MHF并行（分离式）")
    
    # 基础FNO模块 - 修改为输出32通道，然后通过1x1卷积降维
    fno_hidden = FNO(
        n_modes=(8, 8),
        hidden_channels=32,
        in_channels=1,
        out_channels=32,  # 输出32通道用于融合
        n_layers=4
    )
    
    # 最终输出层
    fno_output = nn.Conv2d(32, 1, kernel_size=1)
    
    # 独立的CoDA模块 - 使用投影层将输入映射到多头格式
    n_heads = 2
    channels_for_coda = 32  # CoDA路径的通道数
    
    # 输入投影层
    input_proj = nn.Conv2d(1, channels_for_coda, kernel_size=1)
    
    # CoDA模块
    channels_per_head = channels_for_coda // n_heads
    coda = CrossHeadAttention(
        n_heads=n_heads,
        channels_per_head=channels_per_head,
        reduction=4
    )
    
    # 融合层
    fusion = nn.Conv2d(32 + channels_for_coda, 32, kernel_size=1)
    
    class ParallelCoDA(nn.Module):
        def __init__(self, fno_hidden, fno_output, input_proj, coda, fusion):
            super().__init__()
            self.fno_hidden = fno_hidden
            self.fno_output = fno_output
            self.input_proj = input_proj
            self.coda = coda
            self.fusion = fusion
            
        def forward(self, x):
            # FNO路径
            fno_hidden_out = self.fno_hidden(x)  # [B, 32, H, W]
            
            # CoDA路径
            # 1. 投影到CoDA所需的通道数
            coda_input = self.input_proj(x)  # [B, channels_for_coda, H, W]
            
            b, c_coda, h, w = coda_input.shape
            n_heads = self.coda.n_heads
            c_per_head = c_coda // n_heads
            
            # 2. 重塑为多头格式
            coda_input_heads = coda_input.view(b, n_heads, c_per_head, h, w)
            coda_out = self.coda(coda_input_heads)
            coda_out = coda_out.view(b, c_coda, h, w)
            
            # 3. 融合
            combined = torch.cat([fno_hidden_out, coda_out], dim=1)
            fused = self.fusion(combined)  # [B, 32, H, W]
            
            # 4. 最终输出
            output = self.fno_output(fused)  # [B, 1, H, W]
            
            return output
    
    model = ParallelCoDA(fno_hidden, fno_output, input_proj, coda, fusion)
    params = count_parameters(model)
    print(f"参数量: {params:,}")
    
    # 测试前向传播
    x = torch.randn(2, 1, 32, 32)
    y = model(x)
    print(f"前向传播: 输入 {x.shape} -> 输出 {y.shape}")
    
    return model, params

def test_baseline():
    """FNO基线"""
    print("\n🔬 测试FNO Baseline")
    
    model = FNO(
        n_modes=(8, 8),
        hidden_channels=32,
        in_channels=1,
        out_channels=1,
        n_layers=4
    )
    
    params = count_parameters(model)
    print(f"参数量: {params:,}")
    
    # 测试前向传播
    x = torch.randn(2, 1, 32, 32)
    y = model(x)
    print(f"前向传播: 输入 {x.shape} -> 输出 {y.shape}")
    
    return model, params

def test_mhf_without_coda():
    """MHF-FNO without CoDA"""
    print("\n🔬 测试MHF-FNO (无CoDA)")
    
    model = create_mhf_fno_with_attention(
        n_modes=(8, 8),
        hidden_channels=32,
        in_channels=1,
        out_channels=1,
        n_layers=4,
        mhf_layers=[0, 1, 2, 3],
        n_heads=2,
        attention_layers=[]  # 不使用CoDA
    )
    
    params = count_parameters(model)
    print(f"参数量: {params:,}")
    
    # 测试前向传播
    x = torch.randn(2, 1, 32, 32)
    y = model(x)
    print(f"前向传播: 输入 {x.shape} -> 输出 {y.shape}")
    
    return model, params

def main():
    print("="*80)
    print("快速测试三种架构变体")
    print("="*80)
    
    results = {}
    
    # 测试所有变体
    results['FNO_Baseline'] = test_baseline()
    results['MHF_NoCoDA'] = test_mhf_without_coda()
    results['VariantA_AllLayers'] = test_variant_a()
    results['VariantB_Progressive'] = test_variant_b()
    results['VariantC_Parallel'] = test_variant_c()
    
    # 汇总结果
    print("\n" + "="*80)
    print("架构变体测试汇总")
    print("="*80)
    
    print(f"\n{'模型':<25} {'参数量':<15} {'相对FNO减少':<15}")
    print("-"*60)
    
    baseline_params = results['FNO_Baseline'][1]
    
    for name, (model, params) in results.items():
        reduction = (1 - params / baseline_params) * 100
        print(f"{name:<25} {params:<15,} {reduction:+.1f}%")
    
    # 简单的推理速度测试
    print("\n🔧 推理速度测试 (单样本, 10次运行平均)")
    print("-"*60)
    
    x_test = torch.randn(1, 1, 32, 32)
    
    for name, (model, params) in results.items():
        model.eval()
        
        # 预热
        with torch.no_grad():
            _ = model(x_test)
        
        # 计时
        times = []
        with torch.no_grad():
            for _ in range(10):
                t0 = time.time()
                _ = model(x_test)
                times.append(time.time() - t0)
        
        avg_time = np.mean(times) * 1000
        print(f"{name:<25} {avg_time:.2f} ms")
    
    print("\n✅ 架构变体测试完成")
    
    # 保存结果
    import json
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_file = f"results/architecture_test_{timestamp}.json"
    
    Path("results").mkdir(exist_ok=True)
    
    with open(result_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'results': {
                name: {
                    'parameters': params,
                    'relative_reduction': (1 - params / baseline_params) * 100
                }
                for name, (model, params) in results.items()
            },
            'baseline_parameters': baseline_params
        }, f, indent=2)
    
    print(f"📁 结果已保存到: {result_file}")

if __name__ == '__main__':
    main()