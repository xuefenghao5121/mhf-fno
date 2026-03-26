#!/usr/bin/env python3
"""
真实 NS 数据 + MHF+CoDA+PINO 测试

目标: 验证在真实 NS 数据上，MHF+CoDA+PINO 是否有正收益
基线: MHF+CoDA test_loss = 0.3828
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys
import json
import time

sys.path.insert(0, str(Path(__file__).parent.parent))
from mhf_fno import create_mhf_fno_with_attention
from mhf_fno.pino_physics import NavierStokesPINOLoss

def main():
    print("=" * 70)
    print("真实 NS 数据 + MHF+CoDA+PINO 测试")
    print("=" * 70)
    
    # 1. 加载真实 NS 数据
    print("\n1. 加载真实 Navier-Stokes 数据")
    print("-" * 70)
    
    data_path = Path(__file__).parent / 'data' / 'ns_real_velocity.pt'
    if not data_path.exists():
        print(f"❌ 数据文件不存在: {data_path}")
        print("   请先运行 generate_ns_velocity.py 生成数据")
        return
    
    data = torch.load(data_path)
    u = data['velocity']  # [N, T, 2, H, W]
    viscosity = data['viscosity']
    dt = data['dt']
    
    print(f"✅ 数据加载成功")
    print(f"   形状: {u.shape}")
    print(f"   粘性系数: {viscosity}")
    print(f"   时间步长: {dt}")
    print(f"   速度场范围: [{u.min():.4f}, {u.max():.4f}]")
    
    # 2. 准备训练/测试数据
    print("\n2. 准备训练/测试数据")
    print("-" * 70)
    
    n_samples = u.shape[0]
    n_train = int(0.8 * n_samples)
    
    # 输入: 初始速度场
    # 输出: 最终速度场
    train_x = u[:n_train, 0]    # [N_train, 2, H, W]
    train_y = u[:n_train, -1]   # [N_train, 2, H, W]
    test_x = u[n_train:, 0]
    test_y = u[n_train:, -1]
    
    # 时间序列（用于 PINO）
    train_u_series = u[:n_train]  # [N_train, T, 2, H, W]
    
    print(f"训练集: {len(train_x)} 样本")
    print(f"测试集: {len(test_x)} 样本")
    print(f"输入形状: {train_x.shape}")
    print(f"输出形状: {train_y.shape}")
    
    # 3. 创建 MHF+CoDA 模型
    print("\n3. 创建 MHF+CoDA 模型")
    print("-" * 70)
    
    model = create_mhf_fno_with_attention(
        n_modes=(32, 32),        # resolution // 2 (64 / 2)
        hidden_channels=64,
        in_channels=2,           # 速度场有 2 个分量 (u, v)
        out_channels=2,
        n_layers=4,
        mhf_layers=[0, 2],       # MHF 层
        n_heads=4,
        attention_layers=[0, -1] # CoDA 层
    )
    
    params = sum(p.numel() for p in model.parameters())
    print(f"✅ 模型创建成功")
    print(f"   参数量: {params:,}")
    
    # 4. 创建 PINO 损失
    print("\n4. 创建 PINO 损失")
    print("-" * 70)
    
    pino_loss_fn = NavierStokesPINOLoss(
        viscosity=viscosity,
        lambda_divergence=0.1,
        dt=dt
    )
    print(f"✅ PINO 损失创建成功")
    print(f"   粘性系数: {viscosity}")
    print(f"   散度权重: 0.1")
    
    # 5. 训练配置
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    criterion = nn.MSELoss()
    
    # PINO 权重（从小开始）
    lambda_physics = 0.001
    
    print(f"\n训练配置:")
    print(f"   优化器: AdamW (lr=1e-3, weight_decay=1e-4)")
    print(f"   调度器: CosineAnnealingLR (T_max=50)")
    print(f"   PINO 权重: {lambda_physics}")
    print(f"   训练轮数: 50")
    print(f"   批量大小: 16")
    
    # 6. 训练循环
    print("\n5. 开始训练（MHF+CoDA+PINO）")
    print("-" * 70)
    
    epochs = 50
    batch_size = 16
    best_test_loss = float('inf')
    
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        
        # 随机采样
        indices = torch.randperm(len(train_x))
        total_data_loss = 0
        total_physics_loss = 0
        total_pde_loss = 0
        total_div_loss = 0
        n_batches = 0
        
        for i in range(0, len(train_x), batch_size):
            batch_idx = indices[i:i+batch_size]
            x_batch = train_x[batch_idx]
            y_batch = train_y[batch_idx]
            u_series_batch = train_u_series[batch_idx]
            
            # 前向传播
            y_pred = model(x_batch)
            
            # 数据损失
            data_loss = criterion(y_pred, y_batch)
            
            # 物理损失（使用时间序列）
            physics_loss, pde_loss, div_loss = pino_loss_fn(u_series_batch)
            
            # 总损失
            loss = data_loss + lambda_physics * physics_loss
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_data_loss += data_loss.item()
            total_physics_loss += physics_loss.item()
            total_pde_loss += pde_loss.item()
            total_div_loss += div_loss.item()
            n_batches += 1
        
        scheduler.step()
        
        # 测试
        if (epoch + 1) % 10 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                test_pred = model(test_x)
                test_loss = criterion(test_pred, test_y).item()
            
            if test_loss < best_test_loss:
                best_test_loss = test_loss
            
            print(f"Epoch {epoch+1:3d} | "
                  f"Data: {total_data_loss/n_batches:.6f} | "
                  f"Phys: {total_physics_loss/n_batches:.6f} | "
                  f"PDE: {total_pde_loss/n_batches:.6f} | "
                  f"Div: {total_div_loss/n_batches:.6f} | "
                  f"Test: {test_loss:.6f}")
    
    train_time = time.time() - start_time
    
    # 7. 最终结果
    print("\n" + "=" * 70)
    print("6. 最终测试结果")
    print("=" * 70)
    
    model.eval()
    with torch.no_grad():
        test_pred = model(test_x)
        final_loss = criterion(test_pred, test_y).item()
        
        # 计算相对误差
        relative_error = torch.norm(test_pred - test_y) / torch.norm(test_y)
    
    print(f"✅ MHF+CoDA+PINO 测试完成")
    print(f"   Final Test Loss: {final_loss:.6f}")
    print(f"   Best Test Loss: {best_test_loss:.6f}")
    print(f"   Relative Error: {relative_error.item():.6f}")
    print(f"   训练时间: {train_time:.2f}s")
    
    # 与基线对比
    baseline = 0.3828
    improvement = ((baseline / final_loss) - 1) * 100
    
    print(f"\n📊 与基线对比:")
    print(f"   基线 (MHF+CoDA): {baseline:.6f}")
    print(f"   当前 (MHF+CoDA+PINO): {final_loss:.6f}")
    print(f"   性能差异: {improvement:+.2f}%")
    
    if final_loss < baseline:
        print(f"\n🎉 正收益！PINO 在真实 NS 数据上有效！")
        success = True
    else:
        print(f"\n⚠️ 负收益，需要调整参数或数据")
        success = False
    
    # 8. 保存结果
    results = {
        'model': 'MHF+CoDA+PINO',
        'data': 'Real Navier-Stokes velocity field',
        'final_test_loss': final_loss,
        'best_test_loss': best_test_loss,
        'relative_error': relative_error.item(),
        'baseline': baseline,
        'improvement_percent': improvement,
        'success': success,
        'train_time': train_time,
        'config': {
            'n_modes': [32, 32],
            'hidden_channels': 64,
            'n_layers': 4,
            'mhf_layers': [0, 2],
            'n_heads': 4,
            'attention_layers': [0, -1],
            'lambda_physics': lambda_physics,
            'viscosity': viscosity,
            'dt': dt
        },
        'data_info': {
            'n_samples': n_samples,
            'n_train': n_train,
            'n_test': n_samples - n_train,
            'time_steps': u.shape[1],
            'resolution': u.shape[3]
        }
    }
    
    output_path = Path(__file__).parent.parent / 'real_ns_pino_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ 结果已保存到: {output_path}")
    
    return results

if __name__ == '__main__':
    main()
