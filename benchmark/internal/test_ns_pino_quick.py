#!/usr/bin/env python3
"""
NS 数据集测试 - PINO 物理约束 + 模型优化 (快速版本)

快速验证 PINO 优化效果
"""

import json
import sys
import time
from pathlib import Path

import torch
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
    """加载 NS 数据 (1000 samples)"""
    print("\n📊 加载 Navier-Stokes 数据 (1000 samples)...")
    
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
    print(f"   分辨率: {train_x.shape[-1]}x{train_x.shape[-1]}")
    
    return train_x, train_y, test_x, test_y


def train_model_quick(
    model,
    train_x,
    train_y,
    test_x,
    test_y,
    epochs=10,
    use_pino=True,
    lambda_physics=0.01
):
    """快速训练"""
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    
    n_train = train_x.shape[0]
    batch_size = 32
    steps_per_epoch = (n_train + batch_size - 1) // batch_size
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-3,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3
    )
    
    if use_pino:
        pino_loss = PINOLoss(
            lambda_physics=lambda_physics,
            smoothness_weight=0.5
        )
        print(f"  🔬 使用 PINO 损失: λ={lambda_physics}")
    else:
        loss_fn = LpLoss(d=2, p=2, reduction='mean')
        print(f"  📊 使用标准 LpLoss")
    
    best_test_loss = float('inf')
    best_epoch = 0
    grad_clip = 1.0
    
    print(f"  开始训练 (epochs={epochs}, use_pino={use_pino})...")
    
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
            
            if use_pino:
                loss = pino_loss(y_pred, by)
            else:
                loss = loss_fn(y_pred, by)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            batch_count += 1
        
        avg_train_loss = train_loss / batch_count
        
        # 测试
        model.eval()
        with torch.no_grad():
            test_pred = model(test_x)
            test_loss_fn = LpLoss(d=2, p=2)
            test_loss = test_loss_fn(test_pred, test_y).item()
        
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_epoch = epoch + 1
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:2d}: train={avg_train_loss:.6f}, test={test_loss:.6f}, best={best_test_loss:.6f}")
    
    return best_test_loss, best_epoch


def main():
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    
    print("=" * 80)
    print("NS 方程 PINO 优化测试 (快速版本)")
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
    # 1. Baseline: 标准 FNO
    # ========================================
    print("\n" + "=" * 80)
    print("1️⃣  Baseline: 标准 FNO")
    print("=" * 80)
    
    model_fno = FNO(
        n_modes=(16, 16),
        hidden_channels=32,
        in_channels=1,
        out_channels=1,
        n_layers=4
    ).to(device)
    
    print(f"参数量: {count_parameters(model_fno):,}")
    
    loss_fno, epoch_fno = train_model_quick(
        model_fno, train_x, train_y, test_x, test_y,
        epochs=10, use_pino=False
    )
    
    results['FNO'] = {
        'test_loss': loss_fno,
        'best_epoch': epoch_fno,
        'params': count_parameters(model_fno)
    }
    
    print(f"\n✅ FNO 结果: test_loss={loss_fno:.6f}")
    
    del model_fno
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # ========================================
    # 2. MHF-FNO + 优化配置 + PINO
    # ========================================
    print("\n" + "=" * 80)
    print("2️⃣  MHF-FNO 优化配置 + PINO 物理约束")
    print("=" * 80)

    model_pino = create_mhf_fno_with_attention(
        n_modes=(16, 16),
        hidden_channels=32,
        in_channels=1,
        out_channels=1,
        n_layers=4,
        n_heads=2,           # 减少头数 (4→2)
        mhf_layers=[0],      # 只用第一层 ([0,2]→[0])
        attention_layers=[0] # 注意力也只用在第一层
    ).to(device)
    
    print(f"参数量: {count_parameters(model_pino):,}")
    
    loss_pino, epoch_pino = train_model_quick(
        model_pino, train_x, train_y, test_x, test_y,
        epochs=10,
        use_pino=True,
        lambda_physics=0.01
    )
    
    results['MHF-FNO-PINO'] = {
        'test_loss': loss_pino,
        'best_epoch': epoch_pino,
        'params': count_parameters(model_pino)
    }
    
    print(f"\n✅ MHF-FNO+PINO 结果: test_loss={loss_pino:.6f}")
    
    del model_pino
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # ========================================
    # 结果对比
    # ========================================
    print("\n" + "=" * 80)
    print("📊 快速测试结果对比")
    print("=" * 80)
    
    baseline_loss = results['FNO']['test_loss']
    
    print(f"\n{'配置':<20} {'Test Loss':<12} {'vs FNO':<10} {'参数量':<12}")
    print("-" * 80)
    
    for name, res in results.items():
        test_loss = res['test_loss']
        vs_fno = ((test_loss - baseline_loss) / baseline_loss) * 100
        params = res['params']
        
        print(f"{name:<20} {test_loss:<12.6f} {vs_fno:>+7.2f}%  {params:>10,}")
    
    # 判断是否达到目标
    pino_vs_fno = ((results['MHF-FNO-PINO']['test_loss'] - baseline_loss) / baseline_loss) * 100
    
    print("\n" + "=" * 80)
    print("🎯 快速测试结论")
    print("=" * 80)
    
    if pino_vs_fno < 0:
        print(f"✅ 趋势正确！MHF-FNO+PINO 相比 FNO 提升 {-pino_vs_fno:.2f}%")
        print(f"   建议运行完整测试 (50 epochs) 获得最终结果")
    else:
        print(f"⚠️  MHF-FNO+PINO 相比 FNO 差距 {pino_vs_fno:.2f}%")
        print(f"   可能需要调整超参数")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
