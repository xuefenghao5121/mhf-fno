#!/usr/bin/env python3
"""
Burgers 方程示例 - 使用最佳配置

数据集: Burgers 方程 (1D)
真实数据路径: /home/huawei/Desktop/home/xuefenghao/workspace/mhf-data/burgers/
"""

import sys
import torch
import torch.nn as nn
import numpy as np
from scipy.io import loadmat
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from mhf_fno import MHFFNO
from neuralop.losses.data_losses import LpLoss

# 真实数据路径
DATA_PATH = "/home/huawei/Desktop/home/xuefenghao/workspace/mhf-data/burgers"

def load_burgers_data():
    """加载 Burgers 方程数据"""
    print("📊 加载 Burgers 方程数据...")
    
    # 检查文件存在性
    burgers_files = list(Path(DATA_PATH).glob("*.mat"))
    if len(burgers_files) < 3:
        raise FileNotFoundError(f"Burgers 数据文件不足，需要3个 .mat 文件，找到 {len(burgers_files)} 个")
    
    print(f"   找到 {len(burgers_files)} 个 .mat 文件")
    
    # 加载第一个文件作为示例
    file_path = burgers_files[0]
    print(f"   加载文件: {file_path.name}")
    
    # 读取 .mat 文件
    data = loadmat(file_path)
    
    # 提取数据
    if 'a' in data and 'u' in data:
        # 格式: a 输入, u 输出
        x = data['a']  # 输入: [N, n_x]
        y = data['u']  # 输出: [N, n_x]
    elif 'input' in data and 'output' in data:
        # 格式: input/output
        x = data['input']
        y = data['output']
    else:
        # 尝试其他常见键
        keys = list(data.keys())
        # 过滤掉以 '__' 开头的系统键
        keys = [k for k in keys if not k.startswith('__')]
        if len(keys) >= 2:
            x = data[keys[0]]
            y = data[keys[1]]
        else:
            raise ValueError(f"无法识别的 .mat 文件格式: {file_path}")
    
    # 转换为 torch.Tensor
    x_tensor = torch.as_tensor(x, dtype=torch.float32)
    y_tensor = torch.as_tensor(y, dtype=torch.float32)
    
    # 分割训练集和测试集 (80/20)
    n_total = x_tensor.shape[0]
    n_train = int(n_total * 0.8)
    
    train_x = x_tensor[:n_train].unsqueeze(1)  # [N, 1, n_x]
    train_y = y_tensor[:n_train].unsqueeze(1)
    test_x = x_tensor[n_train:].unsqueeze(1)
    test_y = y_tensor[n_train:].unsqueeze(1)
    
    print(f"✅ 数据加载成功")
    print(f"   训练集: {train_x.shape} -> {train_y.shape}")
    print(f"   测试集: {test_x.shape} -> {test_y.shape}")
    print(f"   输入分辨率: {train_x.shape[-1]}")
    
    return train_x, train_y, test_x, test_y

def main():
    """主函数"""
    print("=" * 60)
    print("Burgers 方程示例 - 最佳配置")
    print("=" * 60)
    
    # 1. 加载数据
    train_x, train_y, test_x, test_y = load_burgers_data()
    
    # 获取输入分辨率
    n_x = train_x.shape[-1]
    n_modes = (n_x // 2,)  # 1D 数据集使用 tuple
    
    # 2. 创建模型 (最佳配置)
    print("\n🤖 创建 MHF-FNO 模型 (最佳配置)...")
    model = MHFFNO(
        in_channels=1,
        out_channels=1,
        hidden_channels=32,
        n_modes=n_modes,     # 1D 分辨率，使用一半的 modes
        n_layers=4,
        mhf_layers=[0, 2],   # 在第1和第3层使用 MHF
        n_heads=2,
        use_coda=True,       # 启用 Cross-Head Attention
        use_pino=False,      # Burgers 可选是否使用 PINO
    )
    
    print(f"✅ 模型创建成功")
    print(f"   总参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   n_modes: {n_modes}")
    
    # 3. 训练配置
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    loss_fn = LpLoss(d=1, p=2)  # 1D 数据
    
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
    output_path = "burgers_model_best.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'in_channels': 1,
            'out_channels': 1,
            'hidden_channels': 32,
            'n_modes': n_modes,
            'n_layers': 4,
            'mhf_layers': [0, 2],
            'n_heads': 2,
            'use_coda': True,
        }
    }, output_path)
    print(f"💾 模型已保存到: {output_path}")
    
    print("\n" + "=" * 60)
    print("✅ Burgers 方程示例完成!")
    print("=" * 60)

if __name__ == "__main__":
    main()