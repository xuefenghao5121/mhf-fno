#!/usr/bin/env python3
"""
生成真实的 2D Navier-Stokes 时间序列数据（包含速度场）

NS 方程:
∂u/∂t + (u·∇)u = -∇p + ν∇²u
∇·u = 0 (不可压缩)

输出:
- velocity: [N, T, 2, H, W] - 速度场 (u_x, u_y)
- pressure: [N, T, 1, H, W] - 压力场
"""

import torch
import numpy as np
from pathlib import Path

def navier_stokes_2d(
    n_samples: int = 200,
    resolution: int = 64,
    time_steps: int = 20,
    viscosity: float = 1e-3,
    dt: float = 0.01
):
    """
    生成 2D Navier-Stokes 时间序列数据
    
    Returns:
        u: [N, T, 2, H, W] - 速度场 (u_x, u_y)
        p: [N, T, 1, H, W] - 压力场
    """
    print(f"生成 Navier-Stokes 数据...")
    print(f"  样本数: {n_samples}")
    print(f"  分辨率: {resolution}x{resolution}")
    print(f"  时间步: {time_steps}")
    print(f"  粘性系数: {viscosity}")
    print(f"  时间步长: {dt}")
    
    # 初始化速度场
    u = torch.zeros(n_samples, time_steps, 2, resolution, resolution)
    p = torch.zeros(n_samples, time_steps, 1, resolution, resolution)
    
    # 创建网格
    y_grid, x_grid = torch.meshgrid(
        torch.arange(resolution, dtype=torch.float32),
        torch.arange(resolution, dtype=torch.float32),
        indexing='ij'
    )
    
    # 初始条件（随机涡旋）
    print("\n初始化随机涡旋...")
    for i in range(n_samples):
        if (i + 1) % 50 == 0:
            print(f"  样本 {i+1}/{n_samples}")
        
        # 添加随机涡旋
        n_vortices = np.random.randint(2, 5)
        for _ in range(n_vortices):
            cx = np.random.randint(10, resolution - 10)
            cy = np.random.randint(10, resolution - 10)
            radius = np.random.randint(5, 15)
            strength = np.random.randn() * 2
            
            # 创建涡旋
            r = torch.sqrt((x_grid - cx)**2 + (y_grid - cy)**2) + 1e-6
            
            # 涡旋速度场（右手定则）
            u[i, 0, 0] += -strength * (y_grid - cy) / r * torch.exp(-r / radius)
            u[i, 0, 1] += strength * (x_grid - cx) / r * torch.exp(-r / radius)
    
    # 时间演化（简化的 NS 求解器）
    print("\n时间演化...")
    dx = 1.0  # 空间步长
    
    for t in range(1, time_steps):
        if (t + 1) % 5 == 0:
            print(f"  时间步 {t+1}/{time_steps}")
        
        u_prev = u[:, t-1]  # [N, 2, H, W]
        
        # 拉普拉斯算子（扩散项）
        # ∇²u = (u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] - 4u[i,j]) / dx²
        laplacian_u = (
            torch.roll(u_prev, -1, dims=2) + 
            torch.roll(u_prev, 1, dims=2) +
            torch.roll(u_prev, -1, dims=3) + 
            torch.roll(u_prev, 1, dims=3) -
            4 * u_prev
        ) / (dx ** 2)
        
        # 平流项 (u·∇)u
        # ∂u/∂x
        du_dx = (torch.roll(u_prev, -1, dims=3) - torch.roll(u_prev, 1, dims=3)) / (2 * dx)
        # ∂u/∂y
        du_dy = (torch.roll(u_prev, -1, dims=2) - torch.roll(u_prev, 1, dims=2)) / (2 * dx)
        
        # (u·∇)u_x = u * ∂u_x/∂x + v * ∂u_x/∂y
        advection_x = u_prev[:, 0:1] * du_dx[:, 0:1] + u_prev[:, 1:2] * du_dy[:, 0:1]
        # (u·∇)u_y = u * ∂u_y/∂x + v * ∂u_y/∂y
        advection_y = u_prev[:, 0:1] * du_dx[:, 1:2] + u_prev[:, 1:2] * du_dy[:, 1:2]
        
        advection = torch.cat([advection_x, advection_y], dim=1)
        
        # 时间步进
        # ∂u/∂t = - (u·∇)u + ν∇²u
        u[:, t] = u_prev + dt * (
            -advection +           # 平流项
            viscosity * laplacian_u # 扩散项
        )
        
        # 压力场（简化：Bernoulli 原理）
        # p + 0.5 * ρ * |u|² = const
        p[:, t, 0] = -0.5 * (u[:, t, 0]**2 + u[:, t, 1]**2)
    
    print(f"\n✅ 数据生成完成")
    print(f"  速度场: {u.shape}  # [N, T, 2, H, W]")
    print(f"  压力场: {p.shape}  # [N, T, 1, H, W]")
    
    # 统计信息
    print(f"\n统计信息:")
    print(f"  速度场范围: [{u.min():.4f}, {u.max():.4f}]")
    print(f"  压力场范围: [{p.min():.4f}, {p.max():.4f}]")
    print(f"  速度场均值: {u.mean():.4f}")
    print(f"  压力场均值: {p.mean():.4f}")
    
    return u, p

def main():
    # 生成数据
    u, p = navier_stokes_2d(
        n_samples=200,
        resolution=64,
        time_steps=20,
        viscosity=1e-3,
        dt=0.01
    )
    
    # 保存数据
    output_path = Path(__file__).parent / 'data' / 'ns_real_velocity.pt'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'velocity': u,
        'pressure': p,
        'viscosity': 1e-3,
        'dt': 0.01,
        'resolution': 64,
        'time_steps': 20,
        'n_samples': 200
    }, output_path)
    
    print(f"\n✅ 数据已保存到: {output_path}")
    print(f"  文件大小: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

if __name__ == '__main__':
    main()
