#!/usr/bin/env python3
"""
Navier-Stokes 方程 PINO 损失函数

NS 方程:
∂u/∂t + (u·∇)u = -∇p + ν∇²u
∇·u = 0 (不可压缩条件)

物理损失 = ||PDE 残差||² + λ * ||散度||²
"""

import torch
import torch.nn as nn

class NavierStokesPINOLoss(nn.Module):
    """
    Navier-Stokes 方程 PINO 损失
    
    NS 方程:
    ∂u/∂t + (u·∇)u = -∇p + ν∇²u
    ∇·u = 0 (不可压缩条件)
    
    物理损失 = ||PDE 残差||² + λ * ||散度||²
    """
    
    def __init__(
        self,
        viscosity: float = 1e-3,
        lambda_divergence: float = 0.1,
        dt: float = 0.01,
        dx: float = 1.0
    ):
        super().__init__()
        self.nu = viscosity
        self.lambda_div = lambda_divergence
        self.dt = dt
        self.dx = dx
    
    def compute_gradient(self, u, dim):
        """计算空间梯度 ∂u/∂x 或 ∂u/∂y"""
        return (torch.roll(u, -1, dims=dim) - torch.roll(u, 1, dims=dim)) / (2 * self.dx)
    
    def compute_laplacian(self, u):
        """计算拉普拉斯算子 ∇²u"""
        lap = (
            torch.roll(u, -1, dims=-2) + 
            torch.roll(u, 1, dims=-2) +
            torch.roll(u, -1, dims=-1) + 
            torch.roll(u, 1, dims=-1) -
            4 * u
        ) / (self.dx ** 2)
        return lap
    
    def compute_divergence(self, u):
        """
        计算散度 ∇·u
        u: [B, T, 2, H, W] (u_x, u_y)
        """
        du_dx = self.compute_gradient(u[..., 0, :, :], dim=-1)  # ∂u_x/∂x
        dv_dy = self.compute_gradient(u[..., 1, :, :], dim=-2)  # ∂u_y/∂y
        return du_dx + dv_dy
    
    def forward(self, u_pred, u_prev=None):
        """
        计算 NS 方程残差
        
        Args:
            u_pred: [B, T, 2, H, W] - 预测的速度场
            u_prev: [B, T, 2, H, W] - 前一时刻的速度场（用于计算时间导数）
        
        Returns:
            physics_loss: 物理约束损失
            pde_loss: PDE 残差损失
            div_loss: 散度损失
        """
        # 1. 时间导数 ∂u/∂t
        if u_prev is not None:
            du_dt = (u_pred - u_prev) / self.dt
        else:
            # 如果没有前一时刻，假设时间序列连续
            du_dt = (u_pred[:, 1:] - u_pred[:, :-1]) / self.dt
            u_pred = u_pred[:, 1:]  # 调整维度
        
        # 2. 平流项 (u·∇)u
        u_x = self.compute_gradient(u_pred[..., 0, :, :], dim=-1)  # ∂u/∂x
        u_y = self.compute_gradient(u_pred[..., 0, :, :], dim=-2)  # ∂u/∂y
        v_x = self.compute_gradient(u_pred[..., 1, :, :], dim=-1)  # ∂v/∂x
        v_y = self.compute_gradient(u_pred[..., 1, :, :], dim=-2)  # ∂v/∂y
        
        # (u·∇)u_x = u * ∂u/∂x + v * ∂u/∂y
        advection_x = u_pred[..., 0, :, :] * u_x + u_pred[..., 1, :, :] * u_y
        # (u·∇)u_y = u * ∂v/∂x + v * ∂v/∂y
        advection_y = u_pred[..., 0, :, :] * v_x + u_pred[..., 1, :, :] * v_y
        
        # 3. 扩散项 ν∇²u
        laplacian_u = self.compute_laplacian(u_pred)
        
        # 4. NS 方程残差
        # ∂u/∂t + (u·∇)u - ν∇²u = 0
        residual_x = du_dt[..., 0, :, :] + advection_x - self.nu * laplacian_u[..., 0, :, :]
        residual_y = du_dt[..., 1, :, :] + advection_y - self.nu * laplacian_u[..., 1, :, :]
        
        pde_loss = (residual_x ** 2).mean() + (residual_y ** 2).mean()
        
        # 5. 不可压缩条件 ∇·u = 0
        divergence = self.compute_divergence(u_pred)
        div_loss = (divergence ** 2).mean()
        
        # 总物理损失
        physics_loss = pde_loss + self.lambda_div * div_loss
        
        return physics_loss, pde_loss, div_loss

def test_pino_loss():
    """测试 PINO 损失函数"""
    print("=" * 70)
    print("测试 Navier-Stokes PINO 损失函数")
    print("=" * 70)
    
    # 创建测试数据
    batch_size = 4
    time_steps = 10
    resolution = 64
    
    # 随机速度场
    u = torch.randn(batch_size, time_steps, 2, resolution, resolution)
    
    # 创建 PINO 损失
    pino_loss_fn = NavierStokesPINOLoss(
        viscosity=1e-3,
        lambda_divergence=0.1,
        dt=0.01
    )
    
    # 计算损失
    physics_loss, pde_loss, div_loss = pino_loss_fn(u)
    
    print(f"✅ 测试成功")
    print(f"  Physics Loss: {physics_loss.item():.6f}")
    print(f"  PDE Loss: {pde_loss.item():.6f}")
    print(f"  Divergence Loss: {div_loss.item():.6f}")

if __name__ == '__main__':
    test_pino_loss()
