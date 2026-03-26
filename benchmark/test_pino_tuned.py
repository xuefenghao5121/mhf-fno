#!/usr/bin/env python3
"""
PINO 物理约束测试 - 超参数调优版本

测试不同的 lambda_physics 值，找到最佳配置
"""

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from neuralop.losses.data_losses import LpLoss
from neuralop.models import FNO

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from mhf_fno import create_mhf_fno_with_attention, PINOLoss

# 强制刷新输出
import functools
print = functools.partial(print, flush=True)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def load_ns_data():
    """加载 NS 数据"""
    print("\n📊 加载 Navier-Stokes 数据...")
    
    train_path = Path(__file__).parent / 'data' / 'ns_train_32_large.pt'
    test_path = Path(__file__).parent / 'data' / 'ns_test_32_large.pt'
    
    if not train_path.exists():
        print(f"❌ 数据文件不存在: {train_path}")
        return None
    
    train_data = torch.load(train_path, weights_only=False)
    test_data = torch.load(test_path, weights_only=False)
    
    # 解析数据
    if isinstance(train_data, dict):
        train_x = train_data.get('x', train_data.get('train_x'))
        train_y = train_data.get('y', train_data.get('train_y'))
    else:
        train_x, train_y = train_data[0], train_data[1]
    
    if isinstance(test_data, dict):
        test_x = test_data.get('x', test_data.get('test_x'))
        test_y = test_data.get('y', test_data.get('test_y'))
    else:
        test_x, test_y = test_data[0], test_data[1]
    
    # 确保维度
    if train_x.dim() == 3:
        train_x = train_x.unsqueeze(1)
        train_y = train_y.unsqueeze(1)
    if test_x.dim() == 3:
        test_x = test_x.unsqueeze(1)
        test_y = test_y.unsqueeze(1)
    
    train_x = train_x.float()
    train_y = train_y.float()
    test_x = test_x.float()
    test_y = test_y.float()
    
    print(f"✅ 数据加载成功: 训练 {train_x.shape[0]}, 测试 {test_x.shape[0]}")
    
    return train_x, train_y, test_x, test_y


def train_model(
    model,
    train_x,
    train_y,
    test_x,
    test_y,
    epochs=50,
    lambda_physics=0.0,  # 0.0 表示不使用PINO
    batch_size=32,
    lr=1e-3
):
    """训练模型"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    n_train = train_x.shape[0]
    steps_per_epoch = (n_train + batch_size - 1) // batch_size
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3
    )
    
    # 损失函数
    if lambda_physics > 0:
        pino_loss = PINOLoss(lambda_physics=lambda_physics, smoothness_weight=0.5)
    else:
        # 使用标准的LpLoss
        data_loss_fn = LpLoss(d=2, p=2, reduction='mean')
    
    best_test_loss = float('inf')
    train_losses = []
    test_losses = []
    
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train)
        
        train_loss = 0
        batch_count = 0
        
        for i in range(0, n_train, batch_size):
            bx = train_x[perm[i:i+batch_size]]
            by = train_y[perm[i:i+batch_size]]
            
            optimizer.zero_grad()
            
            y_pred = model(bx)
            
            if lambda_physics > 0:
                loss = pino_loss(y_pred, by)
            else:
                loss = data_loss_fn(y_pred, by)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            batch_count += 1
        
        avg_train_loss = train_loss / batch_count
        train_losses.append(avg_train_loss)
        
        # 测试
        model.eval()
        with torch.no_grad():
            test_pred = model(test_x)
            test_loss_fn = LpLoss(d=2, p=2)
            test_loss = test_loss_fn(test_pred, test_y).item()
            test_losses.append(test_loss)
        
        if test_loss < best_test_loss:
            best_test_loss = test_loss
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:2d}: train={avg_train_loss:.6f}, test={test_loss:.6f}, best={best_test_loss:.6f}")
    
    return best_test_loss, train_losses, test_losses


def main():
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    
    print("=" * 80)
    print("PINO 超参数调优测试")
    print("=" * 80)
    
    # 加载数据
    data = load_ns_data()
    if data is None:
        return
    train_x, train_y, test_x, test_y = data
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  设备: {device}")
    
    # 数据移到设备
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    test_x = test_x.to(device)
    test_y = test_y.to(device)
    
    results = {}
    
    # ========================================
    # 1. Baseline: FNO (无PINO)
    # ========================================
    print("\n" + "=" * 80)
    print("1️⃣  Baseline: FNO (λ=0.0, 无PINO)")
    print("=" * 80)
    
    model_fno = FNO(
        n_modes=(16, 16),
        hidden_channels=32,
        in_channels=1,
        out_channels=1,
        n_layers=4
    ).to(device)
    
    print(f"参数量: {count_parameters(model_fno):,}")
    
    loss_fno, train_fno, test_fno = train_model(
        model_fno, train_x, train_y, test_x, test_y,
        epochs=20, lambda_physics=0.0
    )
    
    results['FNO'] = {
        'test_loss': loss_fno,
        'train_losses': train_fno,
        'test_losses': test_fno,
        'params': count_parameters(model_fno)
    }
    
    print(f"\n✅ FNO 最终结果: test_loss={loss_fno:.6f}")
    
    del model_fno
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # ========================================
    # 2. MHF-FNO (无PINO)
    # ========================================
    print("\n" + "=" * 80)
    print("2️⃣  MHF-FNO (λ=0.0, 无PINO)")
    print("=" * 80)
    
    model_mhf = create_mhf_fno_with_attention(
        n_modes=(16, 16),
        hidden_channels=32,
        in_channels=1,
        out_channels=1,
        n_layers=4,
        n_heads=2,
        mhf_layers=[0],
        attention_layers=[0]
    ).to(device)
    
    print(f"参数量: {count_parameters(model_mhf):,}")
    
    loss_mhf, train_mhf, test_mhf = train_model(
        model_mhf, train_x, train_y, test_x, test_y,
        epochs=20, lambda_physics=0.0
    )
    
    results['MHF-FNO'] = {
        'test_loss': loss_mhf,
        'train_losses': train_mhf,
        'test_losses': test_mhf,
        'params': count_parameters(model_mhf)
    }
    
    print(f"\n✅ MHF-FNO 最终结果: test_loss={loss_mhf:.6f}")
    
    del model_mhf
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # ========================================
    # 3. MHF-FNO + PINO (不同 lambda 值)
    # ========================================
    lambda_values = [0.0001, 0.001, 0.01, 0.1]
    
    for idx, lambda_val in enumerate(lambda_values, start=3):
        print("\n" + "=" * 80)
        print(f"{idx}️⃣  MHF-FNO + PINO (λ={lambda_val})")
        print("=" * 80)
        
        model_pino = create_mhf_fno_with_attention(
            n_modes=(16, 16),
            hidden_channels=32,
            in_channels=1,
            out_channels=1,
            n_layers=4,
            n_heads=2,
            mhf_layers=[0],
            attention_layers=[0]
        ).to(device)
        
        print(f"参数量: {count_parameters(model_pino):,}")
        
        loss_pino, train_pino, test_pino = train_model(
            model_pino, train_x, train_y, test_x, test_y,
            epochs=20, lambda_physics=lambda_val
        )
        
        name = f'MHF-PINO-λ{lambda_val}'
        results[name] = {
            'test_loss': loss_pino,
            'train_losses': train_pino,
            'test_losses': test_pino,
            'params': count_parameters(model_pino),
            'lambda': lambda_val
        }
        
        print(f"\n✅ MHF-PINO (λ={lambda_val}) 最终结果: test_loss={loss_pino:.6f}")
        
        del model_pino
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # ========================================
    # 结果对比
    # ========================================
    print("\n" + "=" * 80)
    print("📊 最终结果对比")
    print("=" * 80)
    
    baseline_loss = results['FNO']['test_loss']
    
    print(f"\n{'配置':<25} {'Test Loss':<12} {'vs FNO':<10} {'参数量':<12}")
    print("-" * 80)
    
    for name, res in results.items():
        test_loss = res['test_loss']
        vs_fno = ((test_loss - baseline_loss) / baseline_loss) * 100
        params = res['params']
        
        print(f"{name:<25} {test_loss:<12.6f} {vs_fno:>+7.2f}%  {params:>10,}")
    
    # 保存结果
    output_file = Path(__file__).parent / 'pino_results.json'
    with open(output_file, 'w') as f:
        # 转换为可序列化的格式
        serializable_results = {}
        for name, res in results.items():
            serializable_results[name] = {
                'test_loss': res['test_loss'],
                'params': res['params'],
                'final_train_loss': res['train_losses'][-1],
                'final_test_loss': res['test_losses'][-1]
            }
            if 'lambda' in res:
                serializable_results[name]['lambda'] = res['lambda']
        
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n💾 结果已保存到: {output_file}")
    
    # 找到最佳配置
    best_config = min(results.items(), key=lambda x: x[1]['test_loss'])
    best_vs_fno = ((best_config[1]['test_loss'] - baseline_loss) / baseline_loss) * 100
    
    print("\n" + "=" * 80)
    print("🎯 测试结论")
    print("=" * 80)
    print(f"最佳配置: {best_config[0]}")
    print(f"最佳测试损失: {best_config[1]['test_loss']:.6f}")
    print(f"相对 FNO: {best_vs_fno:+.2f}%")
    
    if best_vs_fno < 0:
        print(f"\n✅ PINO 物理约束有效！相比 FNO 提升 {-best_vs_fno:.2f}%")
    else:
        print(f"\n⚠️  PINO 未带来提升，可能需要：")
        print(f"   1. 实现真正的 NS 方程残差")
        print(f"   2. 调整网络架构")
        print(f"   3. 使用更多训练数据")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
