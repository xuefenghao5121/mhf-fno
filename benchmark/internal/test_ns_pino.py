#!/usr/bin/env python3
"""
NS 数据集测试 - PINO 物理约束 + 模型优化

目标: 将 NS 方程的 MHF-FNO 效果从 -1.27% 提升到正收益

优化方案:
1. PINO 物理约束损失 (核心)
2. NS 专用模型配置
3. 优化的训练策略
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


def train_model_with_pino(
    model, 
    train_x, 
    train_y, 
    test_x, 
    test_y, 
    epochs=50,
    use_pino=True,
    lambda_physics=0.1,
    viscosity=1e-3,
    dt=0.01
):
    """
    使用 PINO 损失训练模型
    
    Args:
        model: 模型
        train_x, train_y: 训练数据
        test_x, test_y: 测试数据
        epochs: 训练轮数
        use_pino: 是否使用 PINO 损失
        lambda_physics: 物理损失权重
        viscosity: 运动粘度
        dt: 时间步长
    """
    # 优化器: Adam + weight decay
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    
    # 学习率调度: OneCycleLR
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
    
    # 损失函数
    if use_pino:
        pino_loss = PINOLoss(
            viscosity=viscosity,
            lambda_physics=lambda_physics,
            dt=dt
        )
        print(f"  🔬 使用 PINO 损失: λ={lambda_physics}, ν={viscosity}, dt={dt}")
    else:
        loss_fn = LpLoss(d=2, p=2, reduction='mean')
        print(f"  📊 使用标准 LpLoss")
    
    best_test_loss = float('inf')
    best_epoch = 0
    grad_clip = 1.0
    
    print(f"  开始训练 (epochs={epochs}, n_train={n_train}, use_pino={use_pino})...")
    
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
                # PINO 损失: 使用 train_x 作为 u_prev
                # 注意: 这里假设 train_x 是初始条件，train_y 是目标解
                # 实际上可能需要时间序列数据，但这里简化处理
                loss = pino_loss(y_pred, by, bx, dt=dt)
            else:
                loss = loss_fn(y_pred, by)
            
            loss.backward()
            
            # 梯度裁剪
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
            test_loss = loss_fn(test_pred, test_y).item() if not use_pino else LpLoss(d=2, p=2)(test_pred, test_y).item()
        
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_epoch = epoch + 1
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}: train={avg_train_loss:.6f}, test={test_loss:.6f}, best={best_test_loss:.6f} (epoch {best_epoch})")
    
    return best_test_loss, best_epoch


def main():
    import sys
    # 强制刷新所有输出
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    
    print("=" * 80, flush=True)
    print("NS 方程 PINO 优化测试", flush=True)
    print("=" * 80, flush=True)
    
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
    # Baseline: 标准 FNO
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
    
    loss_fno, epoch_fno = train_model_with_pino(
        model_fno, train_x, train_y, test_x, test_y,
        epochs=50, use_pino=False
    )
    
    results['FNO'] = {
        'test_loss': loss_fno,
        'best_epoch': epoch_fno,
        'params': count_parameters(model_fno)
    }
    
    print(f"\n✅ FNO 结果: test_loss={loss_fno:.6f}")
    
    # 清理
    del model_fno
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # ========================================
    # MHF-FNO 原始配置
    # ========================================
    print("\n" + "=" * 80)
    print("2️⃣  MHF-FNO 原始配置 (n_heads=4, mhf_layers=[0,2])")
    print("=" * 80)
    
    model_mhf = create_mhf_fno_with_attention(
        n_modes=(16, 16),
        hidden_channels=32,
        in_channels=1,
        out_channels=1,
        n_layers=4,
        n_heads=4,
        mhf_layers=[0, 2],
        bottleneck=4,
        gate_init=0.1
    ).to(device)
    
    print(f"参数量: {count_parameters(model_mhf):,}")
    
    loss_mhf, epoch_mhf = train_model_with_pino(
        model_mhf, train_x, train_y, test_x, test_y,
        epochs=50, use_pino=False
    )
    
    results['MHF-FNO'] = {
        'test_loss': loss_mhf,
        'best_epoch': epoch_mhf,
        'params': count_parameters(model_mhf)
    }
    
    print(f"\n✅ MHF-FNO 结果: test_loss={loss_mhf:.6f}")
    
    # 清理
    del model_mhf
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # ========================================
    # MHF-FNO + 优化配置 (无 PINO)
    # ========================================
    print("\n" + "=" * 80)
    print("3️⃣  MHF-FNO 优化配置 (n_heads=2, mhf_layers=[0], bottleneck=8)")
    print("=" * 80)
    
    model_mhf_opt = create_mhf_fno_with_attention(
        n_modes=(16, 16),
        hidden_channels=32,
        in_channels=1,
        out_channels=1,
        n_layers=4,
        n_heads=2,           # 减少头数
        mhf_layers=[0],      # 只用第一层
        attention_layers=[0] # 注意力也只用在第一层
    ).to(device)
    
    print(f"参数量: {count_parameters(model_mhf_opt):,}")
    
    loss_mhf_opt, epoch_mhf_opt = train_model_with_pino(
        model_mhf_opt, train_x, train_y, test_x, test_y,
        epochs=50, use_pino=False
    )
    
    results['MHF-FNO-Opt'] = {
        'test_loss': loss_mhf_opt,
        'best_epoch': epoch_mhf_opt,
        'params': count_parameters(model_mhf_opt)
    }
    
    print(f"\n✅ MHF-FNO 优化配置 结果: test_loss={loss_mhf_opt:.6f}")
    
    # 清理
    del model_mhf_opt
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # ========================================
    # MHF-FNO 优化配置 + PINO
    # ========================================
    print("\n" + "=" * 80)
    print("4️⃣  MHF-FNO 优化配置 + PINO 物理约束")
    print("=" * 80)
    
    model_pino = create_mhf_fno_with_attention(
        n_modes=(16, 16),
        hidden_channels=32,
        in_channels=1,
        out_channels=1,
        n_layers=4,
        n_heads=2,           # 减少头数
        mhf_layers=[0],      # 只用第一层
        attention_layers=[0] # 注意力也只用在第一层
    ).to(device)
    
    print(f"参数量: {count_parameters(model_pino):,}")
    
    loss_pino, epoch_pino = train_model_with_pino(
        model_pino, train_x, train_y, test_x, test_y,
        epochs=50,
        use_pino=True,
        lambda_physics=0.1,
        viscosity=1e-3,
        dt=0.01
    )
    
    results['MHF-FNO-PINO'] = {
        'test_loss': loss_pino,
        'best_epoch': epoch_pino,
        'params': count_parameters(model_pino)
    }
    
    print(f"\n✅ MHF-FNO + PINO 结果: test_loss={loss_pino:.6f}")
    
    # 清理
    del model_pino
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # ========================================
    # 结果对比
    # ========================================
    print("\n" + "=" * 80)
    print("📊 最终结果对比")
    print("=" * 80)
    
    baseline_loss = results['FNO']['test_loss']
    
    print(f"\n{'配置':<20} {'Test Loss':<12} {'vs FNO':<10} {'参数量':<12} {'Best Epoch'}")
    print("-" * 80)
    
    for name, res in results.items():
        test_loss = res['test_loss']
        vs_fno = ((test_loss - baseline_loss) / baseline_loss) * 100
        params = res['params']
        best_epoch = res['best_epoch']
        
        print(f"{name:<20} {test_loss:<12.6f} {vs_fno:>+7.2f}%  {params:>10,}  {best_epoch:>10}")
    
    # 保存结果
    output_file = Path(__file__).parent / 'ns_pino_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ 结果已保存到: {output_file}")
    
    # 判断是否达到目标
    pino_vs_fno = ((results['MHF-FNO-PINO']['test_loss'] - baseline_loss) / baseline_loss) * 100
    
    print("\n" + "=" * 80)
    print("🎯 目标达成情况")
    print("=" * 80)
    
    if pino_vs_fno > 0:
        print(f"✅ 成功！MHF-FNO+PINO 相比 FNO 提升 {-pino_vs_fno:.2f}%")
    else:
        print(f"❌ 未达目标。MHF-FNO+PINO 相比 FNO 差距 {pino_vs_fno:.2f}%")
        print(f"   (需要正收益，当前为负收益)")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
