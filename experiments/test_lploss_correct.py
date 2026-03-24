"""
LpLoss 正确验证测试

关键发现：
1. 官方 LpLoss 默认 reduction='sum'，会累加所有样本
2. 需要使用 reduction='mean' 来获得平均相对误差
3. LpLoss.__call__ 默认调用 rel() 方法（相对误差）
"""

import torch
import torch.nn.functional as F
import math

print("\n" + "=" * 70)
print(" LpLoss 正确验证测试")
print("=" * 70)

from neuralop.losses.data_losses import LpLoss

# ============================================
# 测试 1: 理解 LpLoss 的计算方式
# ============================================
print("\n" + "-" * 50)
print("测试 1: 理解 LpLoss 计算方式")
print("-" * 50)

torch.manual_seed(42)
batch_size = 4
x = torch.randn(batch_size, 1, 16, 16)
y = torch.randn(batch_size, 1, 16, 16)

# 方式 1: 官方 LpLoss (reduction='sum')
loss_sum = LpLoss(d=2, p=2, reduction='sum')
result_sum = loss_sum(x, y)
print(f"官方 LpLoss (reduction='sum'): {result_sum.item():.6f}")

# 方式 2: 官方 LpLoss (reduction='mean')
loss_mean = LpLoss(d=2, p=2, reduction='mean')
result_mean = loss_mean(x, y)
print(f"官方 LpLoss (reduction='mean'): {result_mean.item():.6f}")

# 方式 3: 手动计算相对 L2 误差
def manual_rel_l2(x, y):
    """手动计算相对 L2 误差（每个样本单独计算，然后平均）"""
    batch_errors = []
    for i in range(x.shape[0]):
        err = torch.norm(x[i] - y[i]) / torch.norm(y[i])
        batch_errors.append(err.item())
    return sum(batch_errors) / len(batch_errors)

manual_result = manual_rel_l2(x, y)
print(f"手动计算相对 L2 (平均): {manual_result:.6f}")

# 方式 4: 整体张量的相对 L2
def tensor_rel_l2(x, y):
    """整体张量的相对 L2"""
    return (torch.norm(x - y) / torch.norm(y)).item()

tensor_result = tensor_rel_l2(x, y)
print(f"整体张量相对 L2: {tensor_result:.6f}")

print(f"\n分析:")
print(f"  - reduction='sum' 累加了 {batch_size} 个样本: {result_sum.item():.4f}")
print(f"  - reduction='mean' 平均: {result_mean.item():.4f}")
print(f"  - 手动平均相对 L2: {manual_result:.4f}")
print(f"  - reduction='mean' ≈ 手动平均: {abs(result_mean.item() - manual_result) < 0.01}")

# ============================================
# 测试 2: 验证 rel() 方法的公式
# ============================================
print("\n" + "-" * 50)
print("测试 2: 验证 rel() 公式")
print("-" * 50)

# 官方公式: ||x-y||_p / ||y||_p
# 对于 p=2: sqrt(sum((x-y)^2)) / sqrt(sum(y^2))

# 使用单个样本验证
x_single = torch.randn(1, 1, 4, 4)
y_single = torch.randn(1, 1, 4, 4)

loss_single = LpLoss(d=2, p=2, reduction='mean')
official = loss_single(x_single, y_single).item()

# 手动计算
diff_norm = torch.norm(x_single - y_single, p=2).item()
y_norm = torch.norm(y_single, p=2).item()
manual = diff_norm / y_norm

print(f"官方 LpLoss: {official:.6f}")
print(f"手动 ||x-y|| / ||y||: {manual:.6f}")
print(f"差异: {abs(official - manual):.6f}")

# ============================================
# 测试 3: 使用 reduction='mean' 进行模型训练
# ============================================
print("\n" + "-" * 50)
print("测试 3: 使用 reduction='mean' 训练简单模型")
print("-" * 50)

# 加载数据
data_path = '/usr/local/lib/python3.11/site-packages/neuralop/data/datasets/data/'
train_data = torch.load(f'{data_path}/darcy_train_16.pt', weights_only=False)
test_data = torch.load(f'{data_path}/darcy_test_16.pt', weights_only=False)

train_x = train_data['x'].unsqueeze(1).float()
train_y = train_data['y'].unsqueeze(1).float()
test_x = test_data['x'].unsqueeze(1).float()
test_y = test_data['y'].unsqueeze(1).float()

print(f"训练集: {train_x.shape}")
print(f"测试集: {test_x.shape}")

# 使用 reduction='mean'
loss_fn = LpLoss(d=2, p=2, reduction='mean')

# 初始误差
with torch.no_grad():
    init_train_loss = loss_fn(train_x, train_y).item()
    init_test_loss = loss_fn(test_x, test_y).item()
print(f"\n初始 Train Loss: {init_train_loss:.4f}")
print(f"初始 Test Loss: {init_test_loss:.4f}")

# 简单 CNN 模型
class SimpleCNN(torch.nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(1, hidden, 3, padding=1),
            torch.nn.GELU(),
            torch.nn.Conv2d(hidden, hidden, 3, padding=1),
            torch.nn.GELU(),
            torch.nn.Conv2d(hidden, 1, 3, padding=1)
        )
    
    def forward(self, x):
        return self.net(x)

model = SimpleCNN(hidden=32)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print("\n训练 20 epochs...")
for epoch in range(20):
    model.train()
    
    # 随机采样一个 batch
    idx = torch.randperm(train_x.shape[0])[:64]
    bx, by = train_x[idx], train_y[idx]
    
    optimizer.zero_grad()
    pred = model(bx)
    loss = loss_fn(pred, by)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 5 == 0:
        model.eval()
        with torch.no_grad():
            test_loss = loss_fn(model(test_x), test_y).item()
        print(f"Epoch {epoch+1}: Train={loss.item():.4f}, Test={test_loss:.4f}")

print("\n✅ 训练成功!")

# ============================================
# 测试 4: 对比两种 reduction 的效果
# ============================================
print("\n" + "-" * 50)
print("测试 4: 对比 reduction='sum' vs 'mean'")
print("-" * 50)

loss_sum = LpLoss(d=2, p=2, reduction='sum')
loss_mean = LpLoss(d=2, p=2, reduction='mean')

# 使用测试集
with torch.no_grad():
    sum_loss = loss_sum(test_x, test_y).item()
    mean_loss = loss_mean(test_x, test_y).item()

print(f"测试集大小: {test_x.shape[0]}")
print(f"reduction='sum': {sum_loss:.4f}")
print(f"reduction='mean': {mean_loss:.4f}")
print(f"sum / batch_size: {sum_loss / test_x.shape[0]:.4f}")
print(f"关系: sum ≈ mean × batch_size: {abs(sum_loss - mean_loss * test_x.shape[0]) < 1.0}")

print("\n" + "=" * 70)
print(" ✅ 关键结论")
print("=" * 70)
print("""
1. LpLoss 默认 reduction='sum'，会累加所有样本的误差
2. 训练时应使用 reduction='mean' 获得平均相对误差
3. rel() 方法计算: ||x-y||_p / ||y||_p (相对 Lp 误差)
4. abs() 方法计算: ||x-y||_p (绝对 Lp 误差，带积分权重)
""")