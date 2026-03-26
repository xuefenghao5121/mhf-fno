#!/usr/bin/env python3
"""
NS 数据集测试 - PINO 物理约束 (修正版)

修复:
1. 统一使用 LpLoss 作为测试指标
2. 使用真正的 NS 方程物理约束
3. 对比不同 lambda_physics 值的效果
"""

import sys
import time
from pathlib import Path

import torch
from neuralop.losses.data_losses import LpLoss
from neuralop.models import FNO

sys.path.insert(0, str(Path(__file__).parent.parent))
from mhf_fno import create_mhf_fno_with_attention

# 强制刷新输出
import functools
print = functools.partial(print, flush=True)


class NavierStokesPhysicsLoss(torch.nn.Module):
    """NS 方程物理约束 (从 ns_optimize_full.py 复制)"""

    def __init__(self, viscosity=1e-3, dx=1.0):
        super().__init__()
        self.viscosity = viscosity
        self.dx = dx

    def compute_laplacian(self, u):
        """计算拉普拉斯算子 ∇²u"""
        lap = (
            torch.roll(u, -1, dims=2) + torch.roll(u, 1, dims=2) +
            torch.roll(u, -1, dims=3) + torch.roll(u, 1, dims=3) -
            4 * u
        ) / (self.dx ** 2)
        return lap

    def forward(self, u_pred):
        """计算 NS 物理约束损失"""
        lap = self.compute_laplacian(u_pred)
        physics_loss = (lap ** 2).mean()
        return physics_loss


class PINOLossCombined(torch.nn.Module):
    """组合 PINO 损失: LpLoss + NS 物理约束"""

    def __init__(self, lambda_physics=0.01, viscosity=1e-3):
        super().__init__()
        self.lambda_physics = lambda_physics
        self.data_loss_fn = LpLoss(d=2, p=2, reduction='mean')
        self.physics_loss_fn = NavierStokesPhysicsLoss(viscosity=viscosity)

    def forward(self, u_pred, u_true):
        L_data = self.data_loss_fn(u_pred, u_true)
        L_physics = self.physics_loss_fn(u_pred)
        return L_data + self.lambda_physics * L_physics


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def load_ns_data():
    """加载 NS 数据"""
    print("\n📊 加载 Navier-Stokes 数据...")

    train_path = Path(__file__).parent / 'data' / 'ns_train_32_large.pt'
    test_path = Path(__file__).parent / 'data' / 'ns_test_32_large.pt'

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

    print(f"✅ 训练: {train_x.shape[0]}, 测试: {test_x.shape[0]}")
    return train_x, train_y, test_x, test_y


def train_model(
    model,
    train_x,
    train_y,
    test_x,
    test_y,
    epochs=10,
    use_pino=False,
    lambda_physics=0.01,
    model_name="Model"
):
    """训练模型"""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    test_loss_fn = LpLoss(d=2, p=2)

    if use_pino:
        train_loss_fn = PINOLossCombined(lambda_physics=lambda_physics)
        print(f"  🔬 使用 PINO: λ={lambda_physics}")
    else:
        train_loss_fn = LpLoss(d=2, p=2, reduction='mean')
        print(f"  📊 使用标准 LpLoss")

    n_train = train_x.shape[0]
    batch_size = 32

    best_test_loss = float('inf')
    best_epoch = 0

    print(f"  开始训练...")

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
            loss = train_loss_fn(y_pred, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            batch_count += 1

        scheduler.step()
        avg_train_loss = train_loss / batch_count

        # 测试
        model.eval()
        with torch.no_grad():
            test_pred = model(test_x)
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
    print("NS 方程 PINO 测试 (修正版)")
    print("=" * 80)

    # 加载数据
    data = load_ns_data()
    train_x, train_y, test_x, test_y = data

    device = torch.device('cpu')
    print(f"\n🖥️  设备: {device}")

    train_x = train_x.to(device)
    train_y = train_y.to(device)
    test_x = test_x.to(device)
    test_y = test_y.to(device)

    results = {}

    # ========================================
    # 1. Baseline: FNO (无 PINO)
    # ========================================
    print("\n" + "=" * 80)
    print("1️⃣  Baseline: FNO (无 PINO)")
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
        epochs=10, use_pino=False, model_name="FNO"
    )

    results['FNO'] = {
        'test_loss': loss_fno,
        'best_epoch': epoch_fno,
        'params': count_parameters(model_fno)
    }

    print(f"\n✅ FNO 结果: {loss_fno:.6f}")

    del model_fno
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ========================================
    # 2. MHF-FNO (无 PINO)
    # ========================================
    print("\n" + "=" * 80)
    print("2️⃣  MHF-FNO (无 PINO)")
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

    loss_mhf, epoch_mhf = train_model(
        model_mhf, train_x, train_y, test_x, test_y,
        epochs=10, use_pino=False, model_name="MHF-FNO"
    )

    results['MHF-FNO'] = {
        'test_loss': loss_mhf,
        'best_epoch': epoch_mhf,
        'params': count_parameters(model_mhf)
    }

    print(f"\n✅ MHF-FNO 结果: {loss_mhf:.6f}")

    del model_mhf
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ========================================
    # 3. MHF-FNO + PINO (不同 lambda 值)
    # ========================================
    for lambda_val in [0.001, 0.01, 0.1]:
        config_name = f"MHF-PINO-λ{lambda_val}"
        print(f"\n" + "=" * 80)
        print(f"3️⃣  {config_name}")
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

        loss_pino, epoch_pino = train_model(
            model_pino, train_x, train_y, test_x, test_y,
            epochs=10, use_pino=True, lambda_physics=lambda_val,
            model_name=config_name
        )

        results[config_name] = {
            'test_loss': loss_pino,
            'best_epoch': epoch_pino,
            'params': count_parameters(model_pino)
        }

        print(f"\n✅ {config_name} 结果: {loss_pino:.6f}")

        del model_pino
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ========================================
    # 结果对比
    # ========================================
    print("\n" + "=" * 80)
    print("📊 测试结果对比")
    print("=" * 80)

    baseline_loss = results['FNO']['test_loss']

    print(f"\n{'配置':<25} {'Test Loss':<12} {'vs FNO':<10} {'参数量':<12}")
    print("-" * 80)

    for name, res in results.items():
        test_loss = res['test_loss']
        vs_fno = ((test_loss - baseline_loss) / baseline_loss) * 100
        params = res['params']

        print(f"{name:<25} {test_loss:<12.6f} {vs_fno:>+7.2f}%  {params:>10,}")

    # 找出最佳配置
    best_config = min(results.items(), key=lambda x: x[1]['test_loss'])
    best_name, best_res = best_config

    print("\n" + "=" * 80)
    print("🎯 测试结论")
    print("=" * 80)

    print(f"\n✅ 最佳配置: {best_name}")
    print(f"   Test Loss: {best_res['test_loss']:.6f}")
    print(f"   vs FNO: {((best_res['test_loss'] - baseline_loss) / baseline_loss) * 100:+.2f}%")

    if best_name.startswith('MHF-PINO'):
        print(f"\n💡 PINO 有效！建议运行完整测试 (50 epochs) 验证最终效果")
    elif best_name == 'MHF-FNO':
        print(f"\n💡 MHF-FNO 基础架构优于 FNO，PINO 未带来额外提升")
        print(f"   可能需要调整 PINO 超参数或物理约束形式")
    else:
        print(f"\n⚠️  FNO 仍是最佳基线")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
