"""
简化测试: LpLoss 函数验证

目标：
1. 验证 NeuralOperator 官方 LpLoss 是否可用
2. 对比自定义 L2 Loss 和官方 LpLoss
3. 确认损失计算正确性
"""

import torch
import torch.nn.functional as F
import sys

# 测试官方 LpLoss 导入
print("\n" + "=" * 60)
print(" LpLoss 函数验证测试")
print("=" * 60)

try:
    from neuralop.losses.data_losses import LpLoss
    print("\n✅ 成功导入 LpLoss from neuralop.losses.data_losses")
except ImportError as e:
    print(f"\n❌ 导入失败: {e}")
    sys.exit(1)

# 测试 1: 创建 LpLoss 实例
print("\n" + "-" * 40)
print("测试 1: 创建 LpLoss 实例")
print("-" * 40)

try:
    # 1D LpLoss
    loss_fn_1d = LpLoss(d=1, p=2)
    print(f"✅ 1D LpLoss 创建成功: d=1, p=2")
    
    # 2D LpLoss
    loss_fn_2d = LpLoss(d=2, p=2)
    print(f"✅ 2D LpLoss 创建成功: d=2, p=2")
    
    # 默认 LpLoss
    loss_fn_default = LpLoss()
    print(f"✅ 默认 LpLoss 创建成功")
    
except Exception as e:
    print(f"❌ 创建失败: {e}")
    sys.exit(1)

# 测试 2: 简单的张量测试
print("\n" + "-" * 40)
print("测试 2: 简单张量测试")
print("-" * 40)

# 创建简单测试数据
batch_size = 4
x_1d = torch.randn(batch_size, 1, 64)
y_1d = torch.randn(batch_size, 1, 64)

x_2d = torch.randn(batch_size, 1, 16, 16)
y_2d = torch.randn(batch_size, 1, 16, 16)

try:
    # 1D 测试
    loss_1d = loss_fn_1d(x_1d, y_1d)
    print(f"✅ 1D LpLoss 计算: {loss_1d.item():.4f}")
    
    # 2D 测试
    loss_2d = loss_fn_2d(x_2d, y_2d)
    print(f"✅ 2D LpLoss 计算: {loss_2d.item():.4f}")
    
except Exception as e:
    print(f"❌ 计算失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试 3: 对比自定义 L2 Loss
print("\n" + "-" * 40)
print("测试 3: 对比自定义 L2 Loss")
print("-" * 40)

def custom_l2_loss(pred, target):
    """自定义 L2 Loss (相对误差)"""
    return torch.norm(pred - target) / torch.norm(target)

def custom_mse_loss(pred, target):
    """自定义 MSE Loss"""
    return F.mse_loss(pred, target)

# 计算各种损失
with torch.no_grad():
    # 使用相同的数据
    torch.manual_seed(42)
    x_test = torch.randn(10, 1, 16, 16)
    y_test = torch.randn(10, 1, 16, 16)
    
    # 官方 LpLoss
    lp_loss = LpLoss(d=2, p=2)
    lp_result = lp_loss(x_test, y_test).item()
    
    # 自定义 L2 Loss
    custom_l2 = custom_l2_loss(x_test, y_test).item()
    
    # 自定义 MSE Loss
    custom_mse = custom_mse_loss(x_test, y_test).item()
    
    print(f"官方 LpLoss (d=2, p=2): {lp_result:.6f}")
    print(f"自定义 L2 Loss:         {custom_l2:.6f}")
    print(f"自定义 MSE Loss:         {custom_mse:.6f}")
    
    # 检查 LpLoss 和自定义 L2 是否接近
    if abs(lp_result - custom_l2) < 0.1:
        print("\n✅ 官方 LpLoss 和自定义 L2 Loss 结果接近")
    else:
        print(f"\n⚠️ 差异较大: {abs(lp_result - custom_l2):.6f}")

# 测试 4: 使用实际 Darcy 数据
print("\n" + "-" * 40)
print("测试 4: 使用实际 Darcy 数据")
print("-" * 40)

try:
    data_path = '/usr/local/lib/python3.11/site-packages/neuralop/data/datasets/data/'
    
    # 加载训练数据
    train_data = torch.load(f'{data_path}/darcy_train_16.pt', weights_only=False)
    test_data = torch.load(f'{data_path}/darcy_test_16.pt', weights_only=False)
    
    train_x = train_data['x'].unsqueeze(1).float()
    train_y = train_data['y'].unsqueeze(1).float()
    test_x = test_data['x'].unsqueeze(1).float()
    test_y = test_data['y'].unsqueeze(1).float()
    
    print(f"训练集: {train_x.shape}")
    print(f"测试集: {test_x.shape}")
    
    # 计算 LpLoss
    loss_fn = LpLoss(d=2, p=2)
    
    # 使用一个 batch 计算损失
    batch_loss = loss_fn(train_x[:32], train_y[:32])
    print(f"\nBatch Loss (前32样本): {batch_loss.item():.6f}")
    
    # 计算全局 LpLoss
    full_loss = loss_fn(train_x, train_y)
    print(f"全局 Loss (全部训练集): {full_loss.item():.6f}")
    
    # 计算测试集 Loss
    test_loss = loss_fn(test_x, test_y)
    print(f"测试集 Loss: {test_loss.item():.6f}")
    
except Exception as e:
    print(f"❌ 数据加载失败: {e}")
    import traceback
    traceback.print_exc()

# 测试 5: 简单模型训练测试
print("\n" + "-" * 40)
print("测试 5: 简单模型训练测试")
print("-" * 40)

class SimpleFNO2D(torch.nn.Module):
    """简化版 FNO 2D"""
    def __init__(self, in_channels=1, out_channels=1, hidden=16):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, hidden, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(hidden, hidden, 3, padding=1)
        self.conv3 = torch.nn.Conv2d(hidden, out_channels, 3, padding=1)
    
    def forward(self, x):
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        return self.conv3(x)

try:
    # 创建简单模型
    model = SimpleFNO2D(hidden=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 使用 LpLoss
    loss_fn = LpLoss(d=2, p=2)
    
    # 训练 10 个 epoch
    print("训练 10 epochs...")
    for epoch in range(10):
        model.train()
        optimizer.zero_grad()
        pred = model(train_x[:32])
        loss = loss_fn(pred, train_y[:32])
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                test_loss = loss_fn(model(test_x), test_y)
            print(f"Epoch {epoch+1}: Train Loss={loss.item():.4f}, Test Loss={test_loss.item():.4f}")
    
    print("\n✅ 简单模型训练测试成功!")
    
except Exception as e:
    print(f"❌ 训练失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print(" ✅ LpLoss 验证测试完成!")
print("=" * 60)