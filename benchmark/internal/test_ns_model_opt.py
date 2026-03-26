#!/usr/bin/env python3
"""
NS 数据集测试 - 模型配置优化 (无 PINO)

目标: 验证模型配置优化本身的效果
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
from mhf_fno import create_mhf_fno_with_attention

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


def train_model(
    model,
    train_x,
    train_y,
    test_x,
    test_y,
    epochs=50,
    lr=5e-4
):
    """训练模型"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

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

    loss_fn = LpLoss(d=2, p=2, reduction='mean')

    best_test_loss = float('inf')
    best_epoch = 0
    grad_clip = 1.0

    print(f"  开始训练 (epochs={epochs})...")

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
            test_loss = loss_fn(test_pred, test_y).item()

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_epoch = epoch + 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}: train={avg_train_loss:.6f}, test={test_loss:.6f}, best={best_test_loss:.6f} (epoch {best_epoch})")

    return best_test_loss, best_epoch


def main():
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    print("=" * 80)
    print("NS 方程模型配置优化测试")
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

    loss_fno, epoch_fno = train_model(
        model_fno, train_x, train_y, test_x, test_y,
        epochs=50
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
    # 2. MHF-FNO 原始配置 (n_heads=4, mhf_layers=[0,2])
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
        attention_layers=[0, 2]
    ).to(device)

    print(f"参数量: {count_parameters(model_mhf):,}")

    loss_mhf, epoch_mhf = train_model(
        model_mhf, train_x, train_y, test_x, test_y,
        epochs=50
    )

    results['MHF-FNO'] = {
        'test_loss': loss_mhf,
        'best_epoch': epoch_mhf,
        'params': count_parameters(model_mhf)
    }

    print(f"\n✅ MHF-FNO 结果: test_loss={loss_mhf:.6f}")

    del model_mhf
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ========================================
    # 3. MHF-FNO 优化配置 (n_heads=2, mhf_layers=[0])
    # ========================================
    print("\n" + "=" * 80)
    print("3️⃣  MHF-FNO 优化配置 (n_heads=2, mhf_layers=[0])")
    print("=" * 80)

    model_mhf_opt = create_mhf_fno_with_attention(
        n_modes=(16, 16),
        hidden_channels=32,
        in_channels=1,
        out_channels=1,
        n_layers=4,
        n_heads=2,
        mhf_layers=[0],
        attention_layers=[0]
    ).to(device)

    print(f"参数量: {count_parameters(model_mhf_opt):,}")

    loss_mhf_opt, epoch_mhf_opt = train_model(
        model_mhf_opt, train_x, train_y, test_x, test_y,
        epochs=50
    )

    results['MHF-FNO-Opt'] = {
        'test_loss': loss_mhf_opt,
        'best_epoch': epoch_mhf_opt,
        'params': count_parameters(model_mhf_opt)
    }

    print(f"\n✅ MHF-FNO 优化配置 结果: test_loss={loss_mhf_opt:.6f}")

    del model_mhf_opt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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
    output_file = Path(__file__).parent / 'ns_model_opt_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ 结果已保存到: {output_file}")

    # 判断是否达到目标
    mhf_opt_vs_fno = ((results['MHF-FNO-Opt']['test_loss'] - baseline_loss) / baseline_loss) * 100

    print("\n" + "=" * 80)
    print("🎯 目标达成情况")
    print("=" * 80)

    if mhf_opt_vs_fno < 0:
        print(f"✅ 成功！MHF-FNO-Opt 相比 FNO 提升 {-mhf_opt_vs_fno:.2f}%")
    else:
        print(f"❌ 未达目标。MHF-FNO-Opt 相比 FNO 差距 {mhf_opt_vs_fno:.2f}%")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
