"""
验证：多头相似度是否与问题复杂度相关

实验设计：
1. 简单问题 (Darcy Flow) - 已测试，相似度 98%
2. 更复杂问题 (Navier-Stokes) - 需要多头多样性？
3. 人工多样性约束 - 强制头不同会怎样？
"""

import torch
import torch.nn as nn
from neuralop.models import FNO
from neuralop.losses.data_losses import LpLoss
import numpy as np
from mhf_fno import MHFSpectralConv


def analyze_diversity(mhf_layer, layer_name):
    """分析 MHF 层的多头多样性"""
    weight = mhf_layer.weight  # (n_heads, head_in, head_out, modes_x, modes_y)
    n_heads = weight.shape[0]
    
    # 提取每个头的权重
    head_weights = []
    for h in range(n_heads):
        w = weight[h].detach().cpu().numpy()
        w_abs = np.abs(w)
        w_avg = np.mean(w_abs, axis=(2, 3))
        head_weights.append(w_avg.flatten())
    
    # 计算相似度矩阵
    similarity_matrix = np.zeros((n_heads, n_heads))
    for i in range(n_heads):
        for j in range(n_heads):
            sim = np.dot(head_weights[i], head_weights[j]) / (
                np.linalg.norm(head_weights[i]) * np.linalg.norm(head_weights[j]) + 1e-8
            )
            similarity_matrix[i, j] = sim
    
    # 多样性指标
    off_diag = similarity_matrix[~np.eye(n_heads, dtype=bool)]
    avg_similarity = np.mean(off_diag)
    diversity_score = 1 - avg_similarity
    
    return avg_similarity, diversity_score


def train_model(model, train_x, train_y, test_x, test_y, epochs=100, batch_size=32, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = LpLoss(d=2, p=2, reduction='mean')
    
    n_train = train_x.shape[0]
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train)
        
        for i in range(0, n_train, batch_size):
            bx = train_x[perm[i:i+batch_size]]
            by = train_y[perm[i:i+batch_size]]
            
            optimizer.zero_grad()
            loss = loss_fn(model(bx), by)
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        
        model.eval()
        with torch.no_grad():
            test_loss = loss_fn(model(test_x), test_y).item()
        if test_loss < best_loss:
            best_loss = test_loss
    
    return best_loss


def main():
    print("=" * 70)
    print(" 验证：多头相似度与问题复杂度的关系")
    print("=" * 70)
    
    data_path = '/usr/local/lib/python3.11/site-packages/neuralop/data/datasets/data/'
    
    # 实验 1: Darcy Flow (简单)
    print("\n" + "=" * 70)
    print(" 实验 1: Darcy Flow (简单 PDE)")
    print("=" * 70)
    
    train_data = torch.load(f'{data_path}/darcy_train_16.pt', weights_only=False)
    test_data = torch.load(f'{data_path}/darcy_test_16.pt', weights_only=False)
    
    train_x = train_data['x'].unsqueeze(1).float()
    train_y = train_data['y'].unsqueeze(1).float()
    test_x = test_data['x'].unsqueeze(1).float()
    test_y = test_data['y'].unsqueeze(1).float()
    
    print(f"数据形状: {train_x.shape}")
    print("问题特点: 单一物理场、线性 PDE、简单边界条件")
    
    torch.manual_seed(42)
    model_darcy = FNO(n_modes=(8, 8), hidden_channels=32, in_channels=1, out_channels=1, n_layers=3)
    model_darcy.fno_blocks.convs[0] = MHFSpectralConv(32, 32, (8, 8), n_heads=4)
    
    loss_darcy = train_model(model_darcy, train_x, train_y, test_x, test_y, epochs=50)
    
    # 分析多样性
    sim_darcy, div_darcy = analyze_diversity(model_darcy.fno_blocks.convs[0], "输入层")
    
    print(f"\n训练 Loss: {loss_darcy:.4f}")
    print(f"多头相似度: {sim_darcy:.4f}")
    print(f"多样性得分: {div_darcy:.4f}")
    
    # 实验 2: 更复杂的输入（多通道）
    print("\n" + "=" * 70)
    print(" 实验 2: 多通道输入（模拟更复杂问题）")
    print("=" * 70)
    
    # 创建多通道输入（原始输入 + 梯度 + 拉普拉斯）
    train_x_multi = torch.cat([
        train_x,
        torch.gradient(train_x, dim=2)[0],  # x 梯度
        torch.gradient(train_x, dim=3)[0],  # y 梯度
    ], dim=1)
    
    test_x_multi = torch.cat([
        test_x,
        torch.gradient(test_x, dim=2)[0],
        torch.gradient(test_x, dim=3)[0],
    ], dim=1)
    
    print(f"多通道输入形状: {train_x_multi.shape}")
    print("通道: 原始 + x梯度 + y梯度")
    
    torch.manual_seed(42)
    model_multi = FNO(n_modes=(8, 8), hidden_channels=32, in_channels=3, out_channels=1, n_layers=3)
    model_multi.fno_blocks.convs[0] = MHFSpectralConv(32, 32, (8, 8), n_heads=4)
    
    loss_multi = train_model(model_multi, train_x_multi, train_y, test_x_multi, test_y, epochs=50)
    
    sim_multi, div_multi = analyze_diversity(model_multi.fno_blocks.convs[0], "输入层")
    
    print(f"\n训练 Loss: {loss_multi:.4f}")
    print(f"多头相似度: {sim_multi:.4f}")
    print(f"多样性得分: {div_multi:.4f}")
    
    # 实验 3: 更高分辨率
    print("\n" + "=" * 70)
    print(" 实验 3: 更高分辨率 (32×32)")
    print("=" * 70)
    
    try:
        train_data_32 = torch.load(f'{data_path}/darcy_train_32.pt', weights_only=False)
        test_data_32 = torch.load(f'{data_path}/darcy_test_32.pt', weights_only=False)
        
        train_x_32 = train_data_32['x'].unsqueeze(1).float()
        train_y_32 = train_data_32['y'].unsqueeze(1).float()
        test_x_32 = test_data_32['x'].unsqueeze(1).float()
        test_y_32 = test_data_32['y'].unsqueeze(1).float()
        
        print(f"数据形状: {train_x_32.shape}")
        print("问题特点: 更高分辨率，更多频率模式")
        
        torch.manual_seed(42)
        model_32 = FNO(n_modes=(16, 16), hidden_channels=32, in_channels=1, out_channels=1, n_layers=3)
        model_32.fno_blocks.convs[0] = MHFSpectralConv(32, 32, (16, 16), n_heads=4)
        
        loss_32 = train_model(model_32, train_x_32, train_y_32, test_x_32, test_y_32, epochs=50)
        
        sim_32, div_32 = analyze_diversity(model_32.fno_blocks.convs[0], "输入层")
        
        print(f"\n训练 Loss: {loss_32:.4f}")
        print(f"多头相似度: {sim_32:.4f}")
        print(f"多样性得分: {div_32:.4f}")
    except:
        print("Darcy 32×32 数据不可用")
        sim_32, div_32 = None, None
    
    # 汇总
    print("\n" + "=" * 70)
    print(" 📊 实验汇总")
    print("=" * 70)
    
    print(f"\n{'实验':<30} {'Loss':<10} {'相似度':<10} {'多样性':<10}")
    print("-" * 60)
    print(f"{'Darcy 16×16 (简单)':<30} {loss_darcy:.4f}     {sim_darcy:.4f}     {div_darcy:.4f}")
    print(f"{'多通道输入 (较复杂)':<30} {loss_multi:.4f}     {sim_multi:.4f}     {div_multi:.4f}")
    if sim_32 is not None:
        print(f"{'Darcy 32×32 (高分辨率)':<30} {loss_32:.4f}     {sim_32:.4f}     {div_32:.4f}")
    
    print("\n" + "=" * 70)
    print(" 💡 结论")
    print("=" * 70)
    
    if sim_multi < sim_darcy:
        print("""
✅ 假设验证：问题复杂度影响多头多样性

更复杂的输入（多通道）→ 多头相似度降低 → 多样性增加

原因分析:
- 简单问题只需要一种模式
- 复杂问题需要多个不同视角
- MHF 的多头在复杂问题上更有价值
""")
    else:
        print("""
❌ 假设未验证：问题复杂度可能不是主要因素

可能的原因:
- MHF 的设计本身不够鼓励多样性
- 需要显式的多样性约束
- 需要更好的初始化策略
""")


if __name__ == "__main__":
    main()