"""
不同头初始化策略对 MHF 多样性的影响

实验设计：
1. 随机初始化（当前）- 所有头随机
2. 正交初始化 - 强制头初始正交
3. 不同频率范围 - 每个头关注不同频率
4. 不同尺度 - 每个头不同权重尺度
"""

import torch
import torch.nn as nn
from neuralop.models import FNO
from neuralop.losses.data_losses import LpLoss
import numpy as np
from mhf_fno import MHFSpectralConv
import time


class OrthogonalInitMHF(MHFSpectralConv):
    """正交初始化的 MHF"""
    
    def __init__(self, in_channels, out_channels, n_modes, n_heads=4):
        super().__init__(in_channels, out_channels, n_modes, n_heads)
        
        # 重新初始化为正交
        with torch.no_grad():
            # 对每个频率维度独立正交初始化
            for h in range(n_heads):
                # 获取第 h 个头的权重
                w = self.weight[h]  # (head_in, head_out, modes_x, modes_y)
                
                # 对实部和虚部分别正交初始化
                w_real = w.real.reshape(-1, w.shape[-2] * w.shape[-1])
                w_imag = w.imag.reshape(-1, w.shape[-2] * w.shape[-1])
                
                # 使用 QR 分解创建正交矩阵
                q_real, _ = torch.linalg.qr(torch.randn_like(w_real))
                q_imag, _ = torch.linalg.qr(torch.randn_like(w_imag))
                
                # 重建复数权重
                w_new = torch.complex(
                    q_real.reshape(w.shape[:2] + (w.shape[2] * w.shape[3],)),
                    q_imag.reshape(w.shape[:2] + (w.shape[2] * w.shape[3],))
                ).reshape(w.shape)
                
                self.weight[h] = w_new


class FrequencyBandMHF(MHFSpectralConv):
    """不同频率范围的 MHF 初始化"""
    
    def __init__(self, in_channels, out_channels, n_modes, n_heads=4):
        super().__init__(in_channels, out_channels, n_modes, n_heads)
        
        modes_x, modes_y = n_modes[0], n_modes[1] // 2 + 1
        
        with torch.no_grad():
            for h in range(n_heads):
                w = self.weight[h]
                
                # 每个头关注不同的频率范围
                # 头 0: 低频 (0 ~ modes_x/4)
                # 头 1: 中低频 (modes_x/4 ~ modes_x/2)
                # 头 2: 中高频 (modes_x/2 ~ 3*modes_x/4)
                # 头 3: 高频 (3*modes_x/4 ~ modes_x)
                
                start = h * modes_x // n_heads
                end = (h + 1) * modes_x // n_heads
                
                # 只在对应频率范围有非零权重
                mask = torch.zeros_like(w)
                mask[:, :, start:end, :] = 1.0
                
                self.weight[h] = w * mask


class ScaleDiverseMHF(MHFSpectralConv):
    """不同尺度的 MHF 初始化"""
    
    def __init__(self, in_channels, out_channels, n_modes, n_heads=4):
        super().__init__(in_channels, out_channels, n_modes, n_heads)
        
        with torch.no_grad():
            for h in range(n_heads):
                # 不同头使用不同的初始化尺度
                scale = 0.01 * (2 ** h)  # 0.01, 0.02, 0.04, 0.08
                nn.init.normal_(self.weight[h], mean=0, std=scale)


class RandomInitMHF(MHFSpectralConv):
    """随机初始化（默认）"""
    pass


def analyze_diversity(model):
    """分析多头多样性"""
    mhf = model.fno_blocks.convs[0]
    weight = mhf.weight
    n_heads = weight.shape[0]
    
    head_weights = []
    for h in range(n_heads):
        w = weight[h].detach().cpu().numpy()
        w_avg = np.mean(np.abs(w), axis=(2, 3))
        head_weights.append(w_avg.flatten())
    
    similarity_matrix = np.zeros((n_heads, n_heads))
    for i in range(n_heads):
        for j in range(n_heads):
            sim = np.dot(head_weights[i], head_weights[j]) / (
                np.linalg.norm(head_weights[i]) * np.linalg.norm(head_weights[j]) + 1e-8
            )
            similarity_matrix[i, j] = sim
    
    off_diag = similarity_matrix[~np.eye(n_heads, dtype=bool)]
    return np.mean(off_diag), 1 - np.mean(off_diag)


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
    print(" 不同头初始化策略对 MHF 多样性的影响")
    print("=" * 70)
    
    # 加载数据
    data_path = '/usr/local/lib/python3.11/site-packages/neuralop/data/datasets/data/'
    train_data = torch.load(f'{data_path}/darcy_train_16.pt', weights_only=False)
    test_data = torch.load(f'{data_path}/darcy_test_16.pt', weights_only=False)
    
    train_x = train_data['x'].unsqueeze(1).float()
    train_y = train_data['y'].unsqueeze(1).float()
    test_x = test_data['x'].unsqueeze(1).float()
    test_y = test_data['y'].unsqueeze(1).float()
    
    print(f"数据: {train_x.shape}")
    
    # 初始化策略
    init_strategies = [
        ("随机初始化", RandomInitMHF),
        ("正交初始化", OrthogonalInitMHF),
        ("频率带分离", FrequencyBandMHF),
        ("尺度多样性", ScaleDiverseMHF),
    ]
    
    results = []
    
    for name, MHFClass in init_strategies:
        print(f"\n{'='*70}")
        print(f" 测试: {name}")
        print("=" * 70)
        
        torch.manual_seed(42)
        model = FNO(n_modes=(8, 8), hidden_channels=32, in_channels=1, out_channels=1, n_layers=3)
        model.fno_blocks.convs[0] = MHFClass(32, 32, (8, 8), n_heads=4)
        
        # 训练前多样性
        sim_before, div_before = analyze_diversity(model)
        print(f"训练前 - 相似度: {sim_before:.4f}, 多样性: {div_before:.4f}")
        
        # 训练
        start = time.time()
        loss = train_model(model, train_x, train_y, test_x, test_y, epochs=50)
        train_time = time.time() - start
        
        # 训练后多样性
        sim_after, div_after = analyze_diversity(model)
        print(f"训练后 - 相似度: {sim_after:.4f}, 多样性: {div_after:.4f}")
        print(f"训练 Loss: {loss:.4f}, 时间: {train_time:.1f}s")
        
        results.append({
            "name": name,
            "sim_before": sim_before,
            "div_before": div_before,
            "sim_after": sim_after,
            "div_after": div_after,
            "loss": loss,
            "time": train_time
        })
        
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # 汇总
    print("\n" + "=" * 70)
    print(" 📊 实验汇总")
    print("=" * 70)
    
    print(f"\n{'策略':<15} {'训练前相似度':<12} {'训练后相似度':<12} {'多样性变化':<12} {'Loss':<10}")
    print("-" * 65)
    
    for r in results:
        div_change = r['div_after'] - r['div_before']
        print(f"{r['name']:<15} {r['sim_before']:<12.4f} {r['sim_after']:<12.4f} {div_change:>+.4f}       {r['loss']:.4f}")
    
    print("\n" + "=" * 70)
    print(" 💡 分析")
    print("=" * 70)
    
    # 找出最佳初始化
    best = min(results, key=lambda x: x['loss'])
    most_diverse = max(results, key=lambda x: x['div_after'])
    
    print(f"\n最佳精度: {best['name']} (Loss: {best['loss']:.4f})")
    print(f"最高多样性: {most_diverse['name']} (多样性: {most_diverse['div_after']:.4f})")
    
    # 分析相似度变化
    print("\n相似度变化分析:")
    for r in results:
        sim_change = r['sim_after'] - r['sim_before']
        if sim_change > 0:
            print(f"  {r['name']}: 相似度增加 {sim_change:+.4f} → 头趋向相同")
        else:
            print(f"  {r['name']}: 相似度减少 {sim_change:+.4f} → 头保持差异")


if __name__ == "__main__":
    main()