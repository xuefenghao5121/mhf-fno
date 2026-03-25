#!/usr/bin/env python3
"""
Navier-Stokes + CoDA 详细测试

测试内容:
1. NS + CoDA 完整测试 (最佳配置)
2. 参数敏感性分析:
   - bottleneck ∈ [2, 4, 6, 8]
   - gate_init ∈ [0.05, 0.1, 0.2, 0.5]

使用方法:
    python test_ns_coda_detailed.py --data_path ../data/ns_train_32_large.pt
    python test_ns_coda_detailed.py --sensitivity
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
import sys
import warnings

import numpy as np
import torch
import torch.nn as nn
from neuralop.losses.data_losses import LpLoss
from neuralop.models import FNO

# 导入 MHF-FNO 核心库
sys.path.insert(0, str(Path(__file__).parent.parent))
from mhf_fno import (
    MHFFNO, 
    create_hybrid_fno,
    MHFSpectralConv,
    MHFSpectralConvWithAttention,
    create_mhf_fno_with_attention
)


# ============================================================================
# 模型构建
# ============================================================================

def create_mhf_fno_with_attention_v2(
    n_modes,
    hidden_channels,
    in_channels=1,
    out_channels=1,
    n_layers=3,
    n_heads=4,
    mhf_layers=None,
    bottleneck=4,
    gate_init=0.1,
    **kwargs
):
    """
    创建带自定义 CoDA 参数的 MHF-FNO 模型
    
    Args:
        n_modes: 频率模式数
        hidden_channels: 隐藏通道数
        in_channels: 输入通道数
        out_channels: 输出通道数
        n_layers: 层数
        n_heads: 多头数量
        mhf_layers: 使用 MHF 的层索引
        bottleneck: CoDA 瓶颈大小
        gate_init: 门控初始化值
    """
    if mhf_layers is None:
        mhf_layers = [0, n_layers - 1]
    
    # 验证配置
    if hidden_channels % n_heads != 0:
        warnings.warn(
            f"hidden_channels ({hidden_channels}) 不能被 n_heads ({n_heads}) 整除，"
            f"MHF 将回退到标准卷积。",
            UserWarning
        )
    
    # 创建基础 FNO 模型
    model = FNO(
        n_modes=n_modes,
        hidden_channels=hidden_channels,
        in_channels=in_channels,
        out_channels=out_channels,
        n_layers=n_layers
    )
    
    # 替换指定层的卷积为带注意力的 MHF
    for layer_idx in mhf_layers:
        if layer_idx < n_layers:
            # 创建带自定义参数的 MHF 卷积
            mhf_conv = MHFSpectralConvWithAttention(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                n_modes=n_modes,
                n_heads=n_heads,
                use_attention=True,
                attn_reduction=bottleneck,  # 使用 bottleneck 参数
                attn_dropout=0.0
            )
            
            # 设置门控初始化值
            with torch.no_grad():
                mhf_conv.cross_head_attn.gate.fill_(gate_init)
            
            model.fno_blocks.convs[layer_idx] = mhf_conv
    
    return model


# ============================================================================
# 数据加载
# ============================================================================

def load_navier_stokes_data(train_path, test_path=None, n_train=None, n_test=None):
    """加载 Navier-Stokes 数据"""
    print(f"\n📊 加载 Navier-Stokes 数据...")
    
    # 加载训练数据
    train_data = torch.load(train_path, weights_only=False)
    
    if isinstance(train_data, dict):
        train_x = train_data.get('x', train_data.get('train_x'))
        train_y = train_data.get('y', train_data.get('train_y'))
    else:
        train_x, train_y = train_data[0], train_data[1]
    
    # 确保维度正确
    if train_x.dim() == 3:
        train_x = train_x.unsqueeze(1)
        train_y = train_y.unsqueeze(1)
    
    train_x = train_x.float()
    train_y = train_y.float()
    
    # 加载测试数据
    if test_path and Path(test_path).exists():
        test_data = torch.load(test_path, weights_only=False)
        if isinstance(test_data, dict):
            test_x = test_data.get('x', test_data.get('test_x'))
            test_y = test_data.get('y', test_data.get('test_y'))
        else:
            test_x, test_y = test_data[0], test_data[1]
        
        if test_x.dim() == 3:
            test_x = test_x.unsqueeze(1)
            test_y = test_y.unsqueeze(1)
        test_x = test_x.float()
        test_y = test_y.float()
    else:
        # 从训练数据分割
        print("  测试文件不存在，从训练数据分割")
        split_idx = int(len(train_x) * 0.8)
        test_x = train_x[split_idx:]
        test_y = train_y[split_idx:]
        train_x = train_x[:split_idx]
        train_y = train_y[:split_idx]
    
    # 限制样本数
    if n_train:
        train_x = train_x[:n_train]
        train_y = train_y[:n_train]
    if n_test:
        test_x = test_x[:n_test]
        test_y = test_y[:n_test]
    
    print(f"  训练集: {train_x.shape}")
    print(f"  测试集: {test_x.shape}")
    
    return train_x, train_y, test_x, test_y


# ============================================================================
# 训练和评估
# ============================================================================

def train_model(model, train_x, train_y, test_x, test_y, config, verbose=True, 
                model_name="Model"):
    """训练模型"""
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['epochs']
    )
    loss_fn = LpLoss(d=2, p=2, reduction='mean')
    
    results = {
        'train_losses': [],
        'test_losses': [],
        'epoch_times': [],
    }
    
    n_train = train_x.shape[0]
    batch_size = config['batch_size']
    
    best_test_loss = float('inf')
    best_epoch = 0
    
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
            loss = loss_fn(model(bx), by)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            batch_count += 1
        
        scheduler.step()
        epoch_time = time.time() - t0
        
        model.eval()
        with torch.no_grad():
            test_loss = loss_fn(model(test_x), test_y).item()
        
        results['train_losses'].append(train_loss / batch_count)
        results['test_losses'].append(test_loss)
        results['epoch_times'].append(epoch_time)
        
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_epoch = epoch + 1
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"  [{model_name}] Epoch {epoch+1}/{config['epochs']}: "
                  f"Train {train_loss/batch_count:.4f}, "
                  f"Test {test_loss:.4f}, "
                  f"Best {best_test_loss:.4f} @ epoch {best_epoch}")
    
    results['best_test_loss'] = best_test_loss
    results['best_epoch'] = best_epoch
    
    return results


def count_parameters(model):
    """计算参数量"""
    return sum(p.numel() for p in model.parameters())


def get_model_info(model, model_name="Model"):
    """获取模型详细信息"""
    total_params = count_parameters(model)
    
    # 统计各部分参数
    fno_params = 0
    attention_params = 0
    
    for name, module in model.named_modules():
        if 'cross_head_attn' in name:
            attention_params += sum(p.numel() for p in module.parameters())
        elif 'spectral' in name.lower() or 'fno_blocks' in name:
            fno_params += sum(p.numel() for p in module.parameters(recurse=False))
    
    return {
        'name': model_name,
        'total_params': total_params,
        'attention_params': attention_params,
        'fno_params': fno_params,
    }


# ============================================================================
# 测试函数
# ============================================================================

def run_best_config_test(train_x, train_y, test_x, test_y, config):
    """运行最佳配置测试"""
    
    print(f"\n{'='*70}")
    print("测试 1: NS + CoDA 最佳配置测试")
    print(f"{'='*70}")
    
    resolution = train_x.shape[-1]
    n_modes = (resolution // 2, resolution // 2)
    
    results = {'best_config': {}}
    
    # --------------------------------------------------
    # 测试 FNO (基准)
    # --------------------------------------------------
    print(f"\n--- FNO (基准) ---")
    torch.manual_seed(config['seed'])
    
    model_fno = FNO(
        n_modes=n_modes,
        hidden_channels=32,
        in_channels=1,
        out_channels=1,
        n_layers=3,
    )
    
    info_fno = get_model_info(model_fno, "FNO")
    print(f"  参数量: {info_fno['total_params']:,}")
    
    train_fno = train_model(
        model_fno, train_x, train_y, test_x, test_y, config,
        verbose=True, model_name="FNO"
    )
    
    results['best_config']['FNO'] = {
        'info': info_fno,
        'results': train_fno,
    }
    
    # --------------------------------------------------
    # 测试纯 MHF-FNO
    # --------------------------------------------------
    print(f"\n--- MHF-FNO (无 CoDA) ---")
    torch.manual_seed(config['seed'])
    
    model_mhf = MHFFNO.best_config(
        n_modes=n_modes,
        hidden_channels=32,
        in_channels=1,
        out_channels=1,
    )
    
    info_mhf = get_model_info(model_mhf, "MHF-FNO")
    param_reduction = (1 - info_mhf['total_params'] / info_fno['total_params']) * 100
    print(f"  参数量: {info_mhf['total_params']:,} ({param_reduction:.1f}% reduction)")
    
    train_mhf = train_model(
        model_mhf, train_x, train_y, test_x, test_y, config,
        verbose=True, model_name="MHF-FNO"
    )
    
    results['best_config']['MHF-FNO'] = {
        'info': info_mhf,
        'results': train_mhf,
    }
    
    # --------------------------------------------------
    # 测试 MHF-FNO + CoDA (最佳配置)
    # --------------------------------------------------
    print(f"\n--- MHF-FNO + CoDA (最佳配置) ---")
    print(f"  配置: n_modes={n_modes}, hidden=32, n_heads=4, mhf_layers=[0,2]")
    print(f"        bottleneck=4, gate_init=0.1")
    
    torch.manual_seed(config['seed'])
    
    model_coda = create_mhf_fno_with_attention_v2(
        n_modes=n_modes,
        hidden_channels=32,
        in_channels=1,
        out_channels=1,
        n_layers=3,
        n_heads=4,
        mhf_layers=[0, 2],
        bottleneck=4,
        gate_init=0.1
    )
    
    info_coda = get_model_info(model_coda, "MHF+CoDA")
    param_reduction_coda = (1 - info_coda['total_params'] / info_fno['total_params']) * 100
    print(f"  参数量: {info_coda['total_params']:,} ({param_reduction_coda:.1f}% reduction)")
    print(f"  注意力参数: {info_coda['attention_params']:,} ({info_coda['attention_params']/info_coda['total_params']*100:.2f}%)")
    
    train_coda = train_model(
        model_coda, train_x, train_y, test_x, test_y, config,
        verbose=True, model_name="MHF+CoDA"
    )
    
    results['best_config']['MHF+CoDA'] = {
        'info': info_coda,
        'results': train_coda,
    }
    
    # --------------------------------------------------
    # 对比分析
    # --------------------------------------------------
    print(f"\n{'='*70}")
    print("结果对比")
    print(f"{'='*70}")
    
    print(f"\n{'模型':<15} {'参数量':<15} {'参数减少':<12} {'最佳Loss':<12} {'最佳Epoch':<10} {'vs FNO':<10}")
    print(f"{'-'*70}")
    
    fno_loss = results['best_config']['FNO']['results']['best_test_loss']
    
    for name in ['FNO', 'MHF-FNO', 'MHF+CoDA']:
        info = results['best_config'][name]['info']
        res = results['best_config'][name]['results']
        
        param_red = (1 - info['total_params'] / info_fno['total_params']) * 100
        improvement = (fno_loss - res['best_test_loss']) / fno_loss * 100
        
        print(f"{name:<15} {info['total_params']:<15,} {param_red:<12.1f}% "
              f"{res['best_test_loss']:<12.4f} {res['best_epoch']:<10} {improvement:+.2f}%")
    
    return results


def run_sensitivity_analysis(train_x, train_y, test_x, test_y, config):
    """运行参数敏感性分析"""
    
    print(f"\n{'='*70}")
    print("测试 2: 参数敏感性分析")
    print(f"{'='*70}")
    
    resolution = train_x.shape[-1]
    n_modes = (resolution // 2, resolution // 2)
    
    results = {
        'bottleneck_sensitivity': {},
        'gate_init_sensitivity': {},
    }
    
    # --------------------------------------------------
    # Bottleneck 敏感性分析
    # --------------------------------------------------
    print(f"\n--- Bottleneck 敏感性分析 ---")
    print(f"  测试值: [2, 4, 6, 8]")
    
    bottleneck_values = [2, 4, 6, 8]
    
    for bn in bottleneck_values:
        print(f"\n  Bottleneck = {bn}")
        torch.manual_seed(config['seed'])
        
        model = create_mhf_fno_with_attention_v2(
            n_modes=n_modes,
            hidden_channels=32,
            in_channels=1,
            out_channels=1,
            n_layers=3,
            n_heads=4,
            mhf_layers=[0, 2],
            bottleneck=bn,
            gate_init=0.1
        )
        
        info = get_model_info(model, f"bottleneck={bn}")
        
        train_res = train_model(
            model, train_x, train_y, test_x, test_y, config,
            verbose=False, model_name=f"BN={bn}"
        )
        
        results['bottleneck_sensitivity'][bn] = {
            'info': info,
            'results': train_res,
        }
        
        print(f"    参数量: {info['total_params']:,}, "
              f"注意力参数: {info['attention_params']:,}, "
              f"最佳Loss: {train_res['best_test_loss']:.4f} @ epoch {train_res['best_epoch']}")
    
    # --------------------------------------------------
    # Gate_init 敏感性分析
    # --------------------------------------------------
    print(f"\n--- Gate_init 敏感性分析 ---")
    print(f"  测试值: [0.05, 0.1, 0.2, 0.5]")
    
    gate_init_values = [0.05, 0.1, 0.2, 0.5]
    
    for gi in gate_init_values:
        print(f"\n  Gate_init = {gi}")
        torch.manual_seed(config['seed'])
        
        model = create_mhf_fno_with_attention_v2(
            n_modes=n_modes,
            hidden_channels=32,
            in_channels=1,
            out_channels=1,
            n_layers=3,
            n_heads=4,
            mhf_layers=[0, 2],
            bottleneck=4,
            gate_init=gi
        )
        
        info = get_model_info(model, f"gate_init={gi}")
        
        train_res = train_model(
            model, train_x, train_y, test_x, test_y, config,
            verbose=False, model_name=f"GI={gi}"
        )
        
        results['gate_init_sensitivity'][str(gi)] = {
            'info': info,
            'results': train_res,
        }
        
        print(f"    最佳Loss: {train_res['best_test_loss']:.4f} @ epoch {train_res['best_epoch']}")
    
    # --------------------------------------------------
    # 敏感性分析汇总
    # --------------------------------------------------
    print(f"\n{'='*70}")
    print("敏感性分析汇总")
    print(f"{'='*70}")
    
    print(f"\n--- Bottleneck 影响 ---")
    print(f"{'Bottleneck':<12} {'参数量':<15} {'注意力参数':<15} {'最佳Loss':<12} {'相对变化':<10}")
    print(f"{'-'*60}")
    
    bn_base_loss = results['bottleneck_sensitivity'][4]['results']['best_test_loss']
    
    for bn in bottleneck_values:
        res = results['bottleneck_sensitivity'][bn]
        rel_change = (res['results']['best_test_loss'] - bn_base_loss) / bn_base_loss * 100
        print(f"{bn:<12} {res['info']['total_params']:<15,} "
              f"{res['info']['attention_params']:<15,} "
              f"{res['results']['best_test_loss']:<12.4f} {rel_change:+.2f}%")
    
    print(f"\n--- Gate_init 影响 ---")
    print(f"{'Gate_init':<12} {'最佳Loss':<12} {'相对变化':<10}")
    print(f"{'-'*40}")
    
    gi_base_loss = results['gate_init_sensitivity']['0.1']['results']['best_test_loss']
    
    for gi in gate_init_values:
        res = results['gate_init_sensitivity'][str(gi)]
        rel_change = (res['results']['best_test_loss'] - gi_base_loss) / gi_base_loss * 100
        print(f"{gi:<12} {res['results']['best_test_loss']:<12.4f} {rel_change:+.2f}%")
    
    return results


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Navier-Stokes + CoDA 详细测试')
    parser.add_argument('--data_path', type=str, default='../data/ns_train_32_large.pt',
                       help='训练数据路径')
    parser.add_argument('--test_path', type=str, default='../data/ns_test_32_large.pt',
                       help='测试数据路径')
    parser.add_argument('--n_train', type=int, default=1000, help='训练集大小')
    parser.add_argument('--n_test', type=int, default=200, help='测试集大小')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--output', type=str, default='ns_coda_detailed_results.json',
                       help='输出文件')
    parser.add_argument('--sensitivity', action='store_true',
                       help='运行参数敏感性分析')
    
    args = parser.parse_args()
    
    config = {
        'n_train': args.n_train,
        'n_test': args.n_test,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'seed': args.seed,
    }
    
    print("="*70)
    print("Navier-Stokes + CoDA 详细测试")
    print("="*70)
    print(f"配置: {config}")
    print(f"数据路径: {args.data_path}")
    
    # 加载数据
    train_x, train_y, test_x, test_y = load_navier_stokes_data(
        args.data_path, args.test_path, args.n_train, args.n_test
    )
    
    all_results = {
        'config': config,
        'data_info': {
            'train_shape': list(train_x.shape),
            'test_shape': list(test_x.shape),
        },
        'timestamp': datetime.now().isoformat(),
    }
    
    # 运行最佳配置测试
    best_config_results = run_best_config_test(
        train_x, train_y, test_x, test_y, config
    )
    all_results['best_config_test'] = best_config_results
    
    # 运行敏感性分析 (如果指定)
    if args.sensitivity:
        sensitivity_results = run_sensitivity_analysis(
            train_x, train_y, test_x, test_y, config
        )
        all_results['sensitivity_analysis'] = sensitivity_results
    
    # 保存结果
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✅ 结果已保存到: {output_path}")
    
    # 生成汇总报告
    print(f"\n{'='*70}")
    print("最终汇总")
    print(f"{'='*70}")
    
    fno_loss = all_results['best_config_test']['best_config']['FNO']['results']['best_test_loss']
    mhf_loss = all_results['best_config_test']['best_config']['MHF-FNO']['results']['best_test_loss']
    coda_loss = all_results['best_config_test']['best_config']['MHF+CoDA']['results']['best_test_loss']
    
    print(f"\n模型性能对比:")
    print(f"  FNO:         Loss = {fno_loss:.4f}")
    print(f"  MHF-FNO:     Loss = {mhf_loss:.4f} ({(mhf_loss-fno_loss)/fno_loss*100:+.2f}%)")
    print(f"  MHF+CoDA:    Loss = {coda_loss:.4f} ({(coda_loss-fno_loss)/fno_loss*100:+.2f}%)")
    
    if args.sensitivity:
        print(f"\n最佳参数:")
        bn_results = all_results['sensitivity_analysis']['bottleneck_sensitivity']
        best_bn = min(bn_results.items(), 
                      key=lambda x: x[1]['results']['best_test_loss'])
        print(f"  Bottleneck: {best_bn[0]} (Loss: {best_bn[1]['results']['best_test_loss']:.4f})")
        
        gi_results = all_results['sensitivity_analysis']['gate_init_sensitivity']
        best_gi = min(gi_results.items(), 
                      key=lambda x: x[1]['results']['best_test_loss'])
        print(f"  Gate_init:  {best_gi[0]} (Loss: {best_gi[1]['results']['best_test_loss']:.4f})")


if __name__ == '__main__':
    main()