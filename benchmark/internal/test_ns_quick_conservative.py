#!/usr/bin/env python3
"""
NS 快速优化测试 - 保守配置 (30 epochs, 验证改进)
"""

import json
import sys
import time
from pathlib import Path

import torch
from neuralop.losses.data_losses import LpLoss
from neuralop.models import FNO

sys.path.insert(0, str(Path(__file__).parent.parent))
from mhf_fno import create_mhf_fno_with_attention

import functools
print = functools.partial(print, flush=True)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def load_data():
    """加载数据"""
    train_path = Path(__file__).parent / 'data' / 'ns_train_32_large.pt'
    test_path = Path(__file__).parent / 'data' / 'ns_test_32_large.pt'
    
    if not train_path.exists():
        print("❌ 需要先运行数据生成")
        return None
    
    train_data = torch.load(train_path, weights_only=False)
    test_data = torch.load(test_path, weights_only=False)
    
    train_x = train_data.get('x', train_data.get('train_x')) if isinstance(train_data, dict) else train_data[0]
    train_y = train_data.get('y', train_data.get('train_y')) if isinstance(train_data, dict) else train_data[1]
    test_x = test_data.get('x', test_data.get('test_x')) if isinstance(test_data, dict) else test_data[0]
    test_y = test_data.get('y', test_data.get('test_y')) if isinstance(test_data, dict) else test_data[1]
    
    if train_x.dim() == 3:
        train_x = train_x.unsqueeze(1)
        train_y = train_y.unsqueeze(1)
    if test_x.dim() == 3:
        test_x = test_x.unsqueeze(1)
        test_y = test_y.unsqueeze(1)
    
    return train_x.float(), train_y.float(), test_x.float(), test_y.float()


def train_model(model, train_x, train_y, test_x, test_y, epochs=30, name="Model"):
    """训练模型"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = LpLoss(d=2, p=2, reduction='mean')
    
    best_test_loss = float('inf')
    best_epoch = 0
    n_train = train_x.shape[0]
    batch_size = 32
    
    start_time = time.time()
    print(f"\n  训练 {name} (epochs={epochs}, n_train={n_train})...")
    
    for epoch in range(epochs):
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            batch_count += 1
        
        scheduler.step()
        
        # 测试
        model.eval()
        with torch.no_grad():
            test_loss = loss_fn(model(test_x), test_y).item()
        
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_epoch = epoch + 1
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:2d}/{epochs}: Train {train_loss/batch_count:.4f}, Test {test_loss:.4f} (best: {best_test_loss:.4f} @ {best_epoch})")
    
    total_time = time.time() - start_time
    print(f"  ✅ 完成: 最佳 {best_test_loss:.4f} (epoch {best_epoch}), 时间 {total_time:.1f}s")
    
    return best_test_loss, best_epoch, total_time


def main():
    print("="*70)
    print("NS 快速优化测试 - 保守配置 vs 原始配置")
    print("="*70)
    
    # 加载数据
    data = load_data()
    if data is None:
        return
    train_x, train_y, test_x, test_y = data
    
    print(f"\n✅ 数据: 训练 {train_x.shape[0]}, 测试 {test_x.shape[0]}, 分辨率 {train_x.shape[-1]}x{train_x.shape[-1]}")
    
    in_channels = train_x.shape[1]
    out_channels = train_y.shape[1]
    n_modes = (16, 16)
    hidden_channels = 32
    n_layers = 3
    epochs = 30
    
    results = {}
    
    # 1. FNO
    print("\n" + "="*70)
    print("1️⃣  FNO Baseline")
    print("="*70)
    fno = FNO(n_modes=n_modes, hidden_channels=hidden_channels, 
              in_channels=in_channels, out_channels=out_channels, n_layers=n_layers)
    fno_params = count_parameters(fno)
    print(f"参数量: {fno_params:,}")
    
    fno_loss, fno_epoch, fno_time = train_model(fno, train_x, train_y, test_x, test_y, epochs, "FNO")
    results['FNO'] = {'params': fno_params, 'loss': fno_loss, 'epoch': fno_epoch, 'time': fno_time}
    
    # 2. MHF-FNO 保守配置
    print("\n" + "="*70)
    print("2️⃣  MHF-FNO 保守配置 (mhf_layers=[0])")
    print("="*70)
    mhf_cons = create_mhf_fno_with_attention(
        n_modes=n_modes, hidden_channels=hidden_channels,
        in_channels=in_channels, out_channels=out_channels,
        n_layers=n_layers, mhf_layers=[0], n_heads=4,
        attention_layers=[0], bottleneck=4, gate_init=0.1
    )
    mhf_cons_params = count_parameters(mhf_cons)
    reduction_cons = (1 - mhf_cons_params / fno_params) * 100
    print(f"参数量: {mhf_cons_params:,} ({reduction_cons:.1f}% 减少)")
    
    mhf_cons_loss, mhf_cons_epoch, mhf_cons_time = train_model(
        mhf_cons, train_x, train_y, test_x, test_y, epochs, "MHF-FNO (保守)")
    improvement_cons = (1 - mhf_cons_loss / fno_loss) * 100
    results['MHF-Conservative'] = {
        'params': mhf_cons_params, 'loss': mhf_cons_loss, 'epoch': mhf_cons_epoch,
        'improvement': improvement_cons, 'time': mhf_cons_time
    }
    
    # 3. MHF-FNO 原始配置
    print("\n" + "="*70)
    print("3️⃣  MHF-FNO 原始配置 (mhf_layers=[0,2])")
    print("="*70)
    mhf_orig = create_mhf_fno_with_attention(
        n_modes=n_modes, hidden_channels=hidden_channels,
        in_channels=in_channels, out_channels=out_channels,
        n_layers=n_layers, mhf_layers=[0, 2], n_heads=4,
        attention_layers=[0, 2], bottleneck=4, gate_init=0.1
    )
    mhf_orig_params = count_parameters(mhf_orig)
    reduction_orig = (1 - mhf_orig_params / fno_params) * 100
    print(f"参数量: {mhf_orig_params:,} ({reduction_orig:.1f}% 减少)")
    
    mhf_orig_loss, mhf_orig_epoch, mhf_orig_time = train_model(
        mhf_orig, train_x, train_y, test_x, test_y, epochs, "MHF-FNO (原始)")
    improvement_orig = (1 - mhf_orig_loss / fno_loss) * 100
    results['MHF-Original'] = {
        'params': mhf_orig_params, 'loss': mhf_orig_loss, 'epoch': mhf_orig_epoch,
        'improvement': improvement_orig, 'time': mhf_orig_time
    }
    
    # 总结
    print("\n" + "="*70)
    print("📊 测试总结")
    print("="*70)
    print(f"\n{'模型':<25} {'参数量':>10} {'减少':>8} {'测试损失':>10} {'vs FNO':>8} {'最佳轮':>6}")
    print("-"*70)
    print(f"{'FNO':<25} {fno_params:>10,} {'-':>8} {fno_loss:>10.4f} {'-':>8} {fno_epoch:>6}")
    print(f"{'MHF-FNO (保守)':<25} {mhf_cons_params:>10,} {reduction_cons:>7.1f}% {mhf_cons_loss:>10.4f} {improvement_cons:>7.2f}% {mhf_cons_epoch:>6}")
    print(f"{'MHF-FNO (原始)':<25} {mhf_orig_params:>10,} {reduction_orig:>7.1f}% {mhf_orig_loss:>10.4f} {improvement_orig:>7.2f}% {mhf_orig_epoch:>6}")
    
    # 保存结果
    output = {
        'config': {'epochs': epochs, 'n_train': train_x.shape[0], 'n_test': test_x.shape[0]},
        'results': results,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    output_file = Path(__file__).parent.parent / 'ns_quick_opt_results.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✅ 结果已保存: {output_file}")
    
    # 结论
    print("\n" + "="*70)
    print("🎯 优化结论")
    print("="*70)
    
    if improvement_cons >= 0:
        print(f"✅ 保守配置成功！MHF-FNO (保守) 优于 FNO {improvement_cons:.2f}%")
        success = True
    else:
        print(f"⚠️  保守配置未能超越 FNO ({improvement_cons:.2f}%)")
        success = False
    
    if improvement_cons > improvement_orig:
        print(f"✅ 保守配置优于原始配置 {improvement_cons - improvement_orig:.2f}%")
    else:
        print(f"⚠️  保守配置未优于原始配置")
    
    # 记录到日志
    log_file = Path(__file__).parent.parent / 'ns_optimization_log.txt'
    with open(log_file, 'a') as f:
        f.write(f"\n{'='*70}\n")
        f.write(f"NS 优化测试 - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*70}\n")
        f.write(f"数据: n_train={train_x.shape[0]}, n_test={test_x.shape[0]}, resolution={train_x.shape[-1]}\n")
        f.write(f"FNO: {fno_loss:.4f} (epoch {fno_epoch})\n")
        f.write(f"MHF (保守): {mhf_cons_loss:.4f} (epoch {mhf_cons_epoch}), {improvement_cons:+.2f}%\n")
        f.write(f"MHF (原始): {mhf_orig_loss:.4f} (epoch {mhf_orig_epoch}), {improvement_orig:+.2f}%\n")
        f.write(f"结论: {'成功' if success else '需进一步优化'}\n")
    
    print(f"✅ 日志已更新: {log_file}")
    print("="*70)


if __name__ == '__main__':
    main()
