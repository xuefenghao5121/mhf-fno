#!/usr/bin/env python3
"""
Navier-Stokes 方程 PINO 损失函数 - 支持多种边界条件

NS 方程:
∂u/∂t + (u·∇)u = -∇p + ν∇²u
∇·u = 0 (不可压缩条件)

物理损失 = ||PDE 残差||² + λ * ||散度||²

支持的边界条件:
- periodic: 周期性边界条件
- dirichlet: Dirichlet边界条件 (固定值)
- neumann: Neumann边界条件 (固定导数)
"""

import torch
import torch.nn as nn
from enum import Enum
from typing import Optional, Tuple


class BoundaryCondition(str, Enum):
    """边界条件类型枚举"""
    PERIODIC = "periodic"      # 周期性边界: u(x+L) = u(x)
    DIRICHLET = "dirichlet"    # Dirichlet边界: u = u₀ (通常为零)
    NEUMANN = "neumann"        # Neumann边界: ∂u/∂n = 0


class NavierStokesPINOLoss(nn.Module):
    """
    Navier-Stokes 方程 PINO 损失 - 支持多种边界条件
    
    NS 方程:
    ∂u/∂t + (u·∇)u = -∇p + ν∇²u
    ∇·u = 0 (不可压缩条件)
    
    物理损失 = ||PDE 残差||² + λ * ||散度||²
    
    参数:
        viscosity: 运动粘度 ν
        lambda_divergence: 散度约束权重
        dt: 时间步长
        dx: 空间步长
        boundary_condition: 边界条件类型 ('periodic', 'dirichlet', 'neumann')
    """
    
    def __init__(
        self,
        viscosity: float = 1e-3,
        lambda_divergence: float = 0.1,
        dt: float = 0.01,
        dx: float = 1.0,
        boundary_condition: str = "periodic"
    ):
        super().__init__()
        self.nu = viscosity
        self.lambda_div = lambda_divergence
        self.dt = dt
        self.dx = dx
        
        # 验证并设置边界条件
        try:
            self.boundary_condition = BoundaryCondition(boundary_condition)
        except ValueError:
            valid_conditions = [e.value for e in BoundaryCondition]
            raise ValueError(
                f"不支持的边界条件 '{boundary_condition}'. "
                f"支持的条件: {valid_conditions}"
            )
        
        # 为Dirichlet边界预定义边界值 (默认零值)
        self.register_buffer('boundary_value', torch.zeros(1))
    
    def _apply_boundary_mask(self, u: torch.Tensor, boundary_width: int = 1) -> torch.Tensor:
        """
        应用边界掩码，将边界值设为指定值
        
        参数:
            u: 输入张量 [..., H, W]
            boundary_width: 边界宽度
        
        返回:
            应用边界条件后的张量
        """
        if self.boundary_condition == BoundaryCondition.PERIODIC:
            # 周期性边界不需要修改
            return u
        
        u_bounded = u.clone()
        
        # 应用边界条件
        if self.boundary_condition == BoundaryCondition.DIRICHLET:
            # Dirichlet: 边界设为固定值
            value = self.boundary_value.item()
            # 上下边界
            u_bounded[..., :boundary_width, :] = value
            u_bounded[..., -boundary_width:, :] = value
            # 左右边界
            u_bounded[..., :, :boundary_width] = value
            u_bounded[..., :, -boundary_width:] = value
            
        elif self.boundary_condition == BoundaryCondition.NEUMANN:
            # Neumann: 边界导数为0，使用最近邻插值
            # 上下边界: 复制边界内侧值
            u_bounded[..., :boundary_width, :] = u_bounded[..., boundary_width:boundary_width+1, :].expand_as(u_bounded[..., :boundary_width, :])
            u_bounded[..., -boundary_width:, :] = u_bounded[..., -boundary_width-1:-boundary_width, :].expand_as(u_bounded[..., -boundary_width:, :])
            # 左右边界
            u_bounded[..., :, :boundary_width] = u_bounded[..., :, boundary_width:boundary_width+1].expand_as(u_bounded[..., :, :boundary_width])
            u_bounded[..., :, -boundary_width:] = u_bounded[..., :, -boundary_width-1:-boundary_width].expand_as(u_bounded[..., :, -boundary_width:])
        
        return u_bounded
    
    def compute_gradient(self, u: torch.Tensor, dim: int) -> torch.Tensor:
        """
        计算空间梯度 ∂u/∂x 或 ∂u/∂y，考虑边界条件
        
        参数:
            u: 输入张量 [..., H, W]
            dim: 计算梯度的维度 (-1 for x, -2 for y)
        
        返回:
            梯度张量
        """
        if self.boundary_condition == BoundaryCondition.PERIODIC:
            # 周期性边界: 使用torch.roll
            return (torch.roll(u, -1, dims=dim) - torch.roll(u, 1, dims=dim)) / (2 * self.dx)
        
        # 对于非周期性边界，需要特殊处理边界点
        grad = torch.zeros_like(u)
        
        # 内部点: 使用中心差分
        if dim == -1:  # x方向
            grad[..., 1:-1] = (u[..., 2:] - u[..., :-2]) / (2 * self.dx)
            
            # 边界处理
            if self.boundary_condition == BoundaryCondition.DIRICHLET:
                # 左边界: 前向差分 (假设u=0在边界外)
                grad[..., 0] = (u[..., 1] - 0) / self.dx
                # 右边界: 后向差分
                grad[..., -1] = (0 - u[..., -2]) / self.dx
                
            elif self.boundary_condition == BoundaryCondition.NEUMANN:
                # Neumann: ∂u/∂n = 0, 边界值等于边界内值
                grad[..., 0] = 0.0  # 边界导数为0
                grad[..., -1] = 0.0
                
        elif dim == -2:  # y方向
            grad[..., 1:-1, :] = (u[..., 2:, :] - u[..., :-2, :]) / (2 * self.dx)
            
            # 边界处理
            if self.boundary_condition == BoundaryCondition.DIRICHLET:
                # 上边界
                grad[..., 0, :] = (u[..., 1, :] - 0) / self.dx
                # 下边界
                grad[..., -1, :] = (0 - u[..., -2, :]) / self.dx
                
            elif self.boundary_condition == BoundaryCondition.NEUMANN:
                grad[..., 0, :] = 0.0
                grad[..., -1, :] = 0.0
        
        return grad
    
    def compute_laplacian(self, u: torch.Tensor) -> torch.Tensor:
        """
        计算拉普拉斯算子 ∇²u，考虑边界条件
        
        参数:
            u: 输入张量 [..., H, W]
        
        返回:
            拉普拉斯算子结果
        """
        if self.boundary_condition == BoundaryCondition.PERIODIC:
            # 周期性边界: 使用torch.roll
            lap = (
                torch.roll(u, -1, dims=-2) + 
                torch.roll(u, 1, dims=-2) +
                torch.roll(u, -1, dims=-1) + 
                torch.roll(u, 1, dims=-1) -
                4 * u
            ) / (self.dx ** 2)
            return lap
        
        # 非周期性边界: 需要特殊处理边界点
        lap = torch.zeros_like(u)
        
        # 内部点: 标准5点差分
        lap[..., 1:-1, 1:-1] = (
            u[..., :-2, 1:-1] +    # 上
            u[..., 2:, 1:-1] +     # 下
            u[..., 1:-1, :-2] +    # 左
            u[..., 1:-1, 2:] -     # 右
            4 * u[..., 1:-1, 1:-1]
        ) / (self.dx ** 2)
        
        # 边界处理
        if self.boundary_condition == BoundaryCondition.DIRICHLET:
            # Dirichlet: 边界值已知，外部为0
            # 左右边界 (x方向)
            for i in range(1, u.shape[-2] - 1):
                # 左边界
                lap[..., i, 0] = (
                    u[..., i-1, 0] + u[..., i+1, 0] +
                    0 + u[..., i, 1] - 4 * u[..., i, 0]
                ) / (self.dx ** 2)
                # 右边界
                lap[..., i, -1] = (
                    u[..., i-1, -1] + u[..., i+1, -1] +
                    u[..., i, -2] + 0 - 4 * u[..., i, -1]
                ) / (self.dx ** 2)
            
            # 上下边界 (y方向)
            for j in range(1, u.shape[-1] - 1):
                # 上边界
                lap[..., 0, j] = (
                    0 + u[..., 1, j] +
                    u[..., 0, j-1] + u[..., 0, j+1] - 4 * u[..., 0, j]
                ) / (self.dx ** 2)
                # 下边界
                lap[..., -1, j] = (
                    u[..., -2, j] + 0 +
                    u[..., -1, j-1] + u[..., -1, j+1] - 4 * u[..., -1, j]
                ) / (self.dx ** 2)
            
            # 四个角点
            lap[..., 0, 0] = (0 + u[..., 1, 0] + 0 + u[..., 0, 1] - 4*u[..., 0, 0]) / (self.dx**2)
            lap[..., 0, -1] = (0 + u[..., 1, -1] + u[..., 0, -2] + 0 - 4*u[..., 0, -1]) / (self.dx**2)
            lap[..., -1, 0] = (u[..., -2, 0] + 0 + 0 + u[..., -1, 1] - 4*u[..., -1, 0]) / (self.dx**2)
            lap[..., -1, -1] = (u[..., -2, -1] + 0 + u[..., -1, -2] + 0 - 4*u[..., -1, -1]) / (self.dx**2)
        
        elif self.boundary_condition == BoundaryCondition.NEUMANN:
            # Neumann: 边界导数为0，使用镜像点法
            # 左右边界
            for i in range(1, u.shape[-2] - 1):
                # 左边界: ∂u/∂x = 0 => u[-1] = u[1]
                lap[..., i, 0] = (
                    u[..., i-1, 0] + u[..., i+1, 0] +
                    u[..., i, 1] + u[..., i, 1] - 4 * u[..., i, 0]
                ) / (self.dx ** 2)
                # 右边界
                lap[..., i, -1] = (
                    u[..., i-1, -1] + u[..., i+1, -1] +
                    u[..., i, -2] + u[..., i, -2] - 4 * u[..., i, -1]
                ) / (self.dx ** 2)
            
            # 上下边界
            for j in range(1, u.shape[-1] - 1):
                # 上边界: ∂u/∂y = 0 => u[0] = u[2]
                lap[..., 0, j] = (
                    u[..., 1, j] + u[..., 1, j] +
                    u[..., 0, j-1] + u[..., 0, j+1] - 4 * u[..., 0, j]
                ) / (self.dx ** 2)
                # 下边界
                lap[..., -1, j] = (
                    u[..., -2, j] + u[..., -2, j] +
                    u[..., -1, j-1] + u[..., -1, j+1] - 4 * u[..., -1, j]
                ) / (self.dx ** 2)
            
            # 四个角点 - 综合两个方向的镜像
            lap[..., 0, 0] = (
                2*u[..., 1, 0] + 2*u[..., 0, 1] - 4*u[..., 0, 0]
            ) / (self.dx**2)
            lap[..., 0, -1] = (
                2*u[..., 1, -1] + 2*u[..., 0, -2] - 4*u[..., 0, -1]
            ) / (self.dx**2)
            lap[..., -1, 0] = (
                2*u[..., -2, 0] + 2*u[..., -1, 1] - 4*u[..., -1, 0]
            ) / (self.dx**2)
            lap[..., -1, -1] = (
                2*u[..., -2, -1] + 2*u[..., -1, -2] - 4*u[..., -1, -1]
            ) / (self.dx**2)
        
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


def test_boundary_conditions():
    """测试不同边界条件的实现"""
    print("\n" + "=" * 70)
    print("测试不同边界条件的实现")
    print("=" * 70)
    
    batch_size = 2
    time_steps = 4
    resolution = 32
    
    # 创建一个平滑的测试场 (避免随机噪声)
    x = torch.linspace(0, 2*3.14159, resolution)
    y = torch.linspace(0, 2*3.14159, resolution)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    # 创建平滑的测试场
    u_base = torch.sin(X) * torch.cos(Y)
    u = u_base.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch_size, time_steps, 1, 1, 1)
    u = u.repeat(1, 1, 2, 1, 1)  # 复制到两个通道
    
    print(f"测试输入形状: {u.shape}")
    print(f"分辨率: {resolution}x{resolution}")
    
    # 测试每种边界条件
    for bc_type in ["periodic", "dirichlet", "neumann"]:
        print(f"\n{'='*50}")
        print(f"测试边界条件: {bc_type}")
        print('='*50)
        
        try:
            pino_loss = NavierStokesPINOLoss(
                viscosity=1e-3,
                lambda_divergence=0.1,
                dt=0.01,
                dx=1.0,
                boundary_condition=bc_type
            )
            
            # 测试梯度计算
            grad_x = pino_loss.compute_gradient(u[..., 0, :, :], dim=-1)
            grad_y = pino_loss.compute_gradient(u[..., 0, :, :], dim=-2)
            
            # 测试拉普拉斯算子
            laplacian = pino_loss.compute_laplacian(u[..., 0:1, :, :])
            
            # 测试完整损失
            physics_loss, pde_loss, div_loss = pino_loss(u)
            
            print(f"✅ {bc_type} 边界条件测试通过")
            print(f"  梯度 X 范围: [{grad_x.min().item():.6f}, {grad_x.max().item():.6f}]")
            print(f"  梯度 Y 范围: [{grad_y.min().item():.6f}, {grad_y.max().item():.6f}]")
            print(f"  拉普拉斯范围: [{laplacian.min().item():.6f}, {laplacian.max().item():.6f}]")
            print(f"  Physics Loss: {physics_loss.item():.6f}")
            print(f"  PDE Loss: {pde_loss.item():.6f}")
            print(f"  Divergence Loss: {div_loss.item():.6f}")
            
        except Exception as e:
            print(f"❌ {bc_type} 边界条件测试失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("所有边界条件测试完成")
    print("=" * 70)


def test_gradient_correctness():
    """测试梯度计算的正确性"""
    print("\n" + "=" * 70)
    print("测试梯度计算的正确性")
    print("=" * 70)
    
    resolution = 64
    dx = 2 * 3.14159 / resolution
    
    # 创建网格
    x = torch.linspace(0, 2*3.14159, resolution)
    y = torch.linspace(0, 2*3.14159, resolution)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    # 测试函数: u = sin(x) * cos(y)
    # ∂u/∂x = cos(x) * cos(y)
    # ∂u/∂y = -sin(x) * sin(y)
    u = torch.sin(X) * torch.cos(Y)
    expected_dudx = torch.cos(X) * torch.cos(Y)
    expected_dudy = -torch.sin(X) * torch.sin(Y)
    
    print(f"测试函数: u = sin(x) * cos(y)")
    print(f"解析解: ∂u/∂x = cos(x) * cos(y)")
    print(f"解析解: ∂u/∂y = -sin(x) * sin(y)")
    
    # 测试每种边界条件
    for bc_type in ["periodic", "dirichlet", "neumann"]:
        print(f"\n{'='*50}")
        print(f"边界条件: {bc_type}")
        print('='*50)
        
        pino_loss = NavierStokesPINOLoss(
            dx=dx,
            boundary_condition=bc_type
        )
        
        u_batch = u.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        
        dudx = pino_loss.compute_gradient(u_batch, dim=-1)
        dudy = pino_loss.compute_gradient(u_batch, dim=-2)
        
        # 计算内部点的误差 (避免边界影响)
        inner_slice = slice(5, -5)
        dudx_inner = dudx[0, 0, inner_slice, inner_slice]
        dudy_inner = dudy[0, 0, inner_slice, inner_slice]
        expected_dudx_inner = expected_dudx[inner_slice, inner_slice]
        expected_dudy_inner = expected_dudy[inner_slice, inner_slice]
        
        error_dudx = torch.abs(dudx_inner - expected_dudx_inner).mean()
        error_dudy = torch.abs(dudy_inner - expected_dudy_inner).mean()
        
        print(f"∂u/∂x 平均绝对误差 (内部点): {error_dudx.item():.6e}")
        print(f"∂u/∂y 平均绝对误差 (内部点): {error_dudy.item():.6e}")
        
        if error_dudx < 0.01 and error_dudy < 0.01:
            print("✅ 梯度计算精度满足要求")
        else:
            print("⚠️ 梯度计算误差较大")
    
    print("\n" + "=" * 70)


def test_laplacian_correctness():
    """测试拉普拉斯算子计算的正确性"""
    print("\n" + "=" * 70)
    print("测试拉普拉斯算子计算的正确性")
    print("=" * 70)
    
    resolution = 64
    dx = 2 * 3.14159 / resolution
    
    # 创建网格
    x = torch.linspace(0, 2*3.14159, resolution)
    y = torch.linspace(0, 2*3.14159, resolution)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    # 测试函数: u = sin(x) * sin(y)
    # ∇²u = ∂²u/∂x² + ∂²u/∂y² = -2*sin(x)*sin(y) = -2*u
    u = torch.sin(X) * torch.sin(Y)
    expected_lap = -2 * u
    
    print(f"测试函数: u = sin(x) * sin(y)")
    print(f"解析解: ∇²u = -2*sin(x)*sin(y) = -2*u")
    
    # 测试周期性边界条件 (其他边界条件需要特殊处理)
    print(f"\n{'='*50}")
    print(f"边界条件: periodic")
    print('='*50)
    
    pino_loss = NavierStokesPINOLoss(
        dx=dx,
        boundary_condition="periodic"
    )
    
    u_batch = u.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    lap = pino_loss.compute_laplacian(u_batch)
    
    # 计算误差
    error = torch.abs(lap[0, 0] - expected_lap).mean()
    
    print(f"拉普拉斯平均绝对误差: {error.item():.6e}")
    
    if error < 0.01:
        print("✅ 拉普拉斯算子计算精度满足要求")
    else:
        print("⚠️ 拉普拉斯算子计算误差较大")
    
    print("\n" + "=" * 70)


if __name__ == '__main__':
    # 运行所有测试
    test_pino_loss()
    test_boundary_conditions()
    test_gradient_correctness()
    test_laplacian_correctness()
