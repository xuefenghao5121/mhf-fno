#!/usr/bin/env python3
"""
Darcy Flow 示例 - 使用最佳配置

数据集: Darcy Flow 32x32
真实数据路径: /home/huawei/Desktop/home/xuefenghao/workspace/mhf-data/darcy_train_32.pt
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from mhf_fno import MHFFNO
from neuralop.losses.data_losses import LpLoss

# 真实数据路径
DATA_PATH = "/home/huawei/Desktop/home/xuefenghao/workspace/mhf-data"

def load_darcy_data():
    """加载 Darcy Flow 32x32 数据"""
    print("📊 加载 Darcy Flow 32x32 数据...")
    
    train_path = f"{DATA_PATH}/darcy_train_32.pt"
    test_path = f"{DATA_PATH}/darcy_test_32.pt"
    
    # 加载训练数据
    train_data = torch.load(train_path, map_location='cpu')
    if isinstance(train_data, dict):
        train_x = train_data.get('x', train_data.get('input'))
        train_y = train_data.get('y', train_data.get('output', train_x))
    elif isinstance(train_data, (tuple, list)):
        train_x, train_y = train_data[0], train_data[1]
    else:
        train_x = train_data
        train_y = train_data
    
    # 加载测试数据
    test_data = torch.load(test_path, map_location='cpu')
    if isinstance(test_data, dict):
        test_x = test_data.get('x', test_data.get('input'))
        test_y = test_data.get('y', test_data.get('output', test_x))
    elif isinstance(test_data, (tuple, list)):
        test_x, test_y = test_data[0], test_data[1]
    else:
        test_x = test_data
        test_y = test_data
    
    # 转换为 float32 并确保维度正确
    train_x = train_x.float().unsqueeze(1)  # [N, 1, 32, 32]
    train_y = train_y.float().unsqueeze(1)
    test_x = test_x.float().unsqueeze(1)
    test_y = test_y.float().unsqueeze(1)
    
    print(f"✅ 数据加载成功")
    print(f"   训练集: {train_x.shape} -> {train_y.shape}")
    print(f"   测试集: {test_x.shape} -> {test_y.shape}")
    
    return train_x, train_y, test_x, test_y

def main():
    """主函数"""
    print("=" * 60)
    print("Darcy Flow 示例 - 最佳配置")
    print("=" * 60)
    
    # 1. 加载数据
    train_x, train_y, test_x, test_y = load_darcy_data()
    
    # 2. 创建模型 (最佳配置)
    print("\n🤖 创建 MHF-FNO 模型 (最佳配置)...")
    model = MHFFNO(
        in_channels=1,
        out_channels=1,
        hidden_channels=32,
        n_modes=(16, 16),  # 32x32 分辨率，使用一半的 modes
        n_layers=4,
        mhf_layers=[0, 2],  # 在第1和第3层使用 MHF
        n_heads=2,
        use_coda=True,      # 启用 Cross-Head Attention
        use_pino=False,     # Darcy 不需要 PINO
    )
    
    print(f"✅ 模型创建成功")
    print(f"   总参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 3. 训练配置
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    loss_fn = LpLoss(d=2, p=2)
    
    # 4. 简单训练循环
    print("\n🚀 开始训练 (5个epoch)...")
    model.train()
    for epoch in range(5):
        optimizer.zero_grad()
        pred = model(train_x)
        loss = loss_fn(pred, train_y)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if epoch % 1 == 0:
            print(f"   Epoch {epoch+1}/5: loss = {loss.item():.6f}")
    
    # 5. 测试
    print("\n🧪 测试模型...")
    model.eval()
    with torch.no_grad():
        test_pred = model(test_x)
        test_loss = loss_fn(test_pred, test_y)
        print(f"✅ 测试损失: {test_loss.item():.6f}")
    
    # 6. 保存模型
    output_path = "darcy_model_best.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'in_channels': 1,
            'out_channels': 1,
            'hidden_channels': 32,
            'n_modes': (16, 16),
            'n_layers': 4,
            'mhf_layers': [0, 2],
            'n_heads': 2,
            'use_coda': True,
        }
    }, output_path)
    print(f"💾 模型已保存到: {output_path}")
    
    print("\n" + "=" * 60)
    print("✅ Darcy Flow 示例完成!")
    print("=" * 60)

if __name__ == "__main__":
    main()