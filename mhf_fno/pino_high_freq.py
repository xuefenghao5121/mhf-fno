"""
PINO 高频噪声惩罚实现

Round 1 优化策略:
- 只惩罚高频分量（避免过度平滑）
- lambda_physics = 0.0001
- 目标：避免之前的过拟合问题

作者: 天渊团队
日期: 2026-03-26
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class HighFreqPINOLoss(nn.Module):
    """
    高频噪声惩罚 PINO 损失函数
    
    只惩罚高频分量，保留低频物理结构
    
    参数:
        lambda_physics: 物理损失权重，默认 0.0001
        freq_threshold: 频率阈值 (0-1)，默认 0.5 (Nyquist频率的50%)
    
    Example:
        >>> loss_fn = HighFreqPINOLoss(lambda_physics=0.0001)
        >>> loss = loss_fn(u_pred, u_true)
    """
    
    def __init__(self, lambda_physics: float = 0.0001, freq_threshold: float = 0.5):
        super().__init__()
        self.lambda_phy = lambda_physics
        self.freq_threshold = freq_threshold
    
    def compute_high_freq_penalty(self, u: torch.Tensor) -> torch.Tensor:
        """
        计算高频噪声惩罚
        
        只惩罚高频分量，保留低频物理结构
        
        Args:
            u: 预测场 [B, C, H, W]
        
        Returns:
            torch.Tensor: 高频噪声损失
        """
        # FFT (2D)
        u_freq = torch.fft.rfft2(u, norm='ortho')
        
        # 获取频率网格
        B, C, H, W = u.shape
        freq_h = torch.fft.fftfreq(H, device=u.device).abs()
        freq_w = torch.fft.rfftfreq(W, device=u.device).abs()
        
        # 归一化频率 (0-1)
        freq_h = freq_h / freq_h.max()
        freq_w = freq_w / freq_w.max()
        
        # 构造频率网格
        freq_grid = torch.sqrt(freq_h[:, None]**2 + freq_w[None, :]**2)
        
        # 归一化到 [0, 1]
        freq_grid = freq_grid / freq_grid.max()
        
        # 高频掩码 (超过阈值的频率)
        high_freq_mask = (freq_grid > self.freq_threshold).float()
        
        # 高频分量的幅值
        high_freq_amp = u_freq.abs() * high_freq_mask
        
        # 高频损失
        high_freq_loss = (high_freq_amp ** 2).mean()
        
        return high_freq_loss
    
    def forward(
        self,
        u_pred: torch.Tensor,
        u_true: torch.Tensor,
        u_prev: Optional[torch.Tensor] = None,
        dt: Optional[float] = None
    ) -> torch.Tensor:
        """
        计算总损失
        
        Args:
            u_pred: 模型预测 [B, C, H, W]
            u_true: 真实值 [B, C, H, W]
            u_prev: 上一时刻值（未使用）
            dt: 时间步长（未使用）
        
        Returns:
            torch.Tensor: 总损失 = L_data + λ × L_high_freq
        """
        # 数据损失 (MSE)
        L_data = F.mse_loss(u_pred, u_true)
        
        # 高频噪声惩罚
        L_high_freq = self.compute_high_freq_penalty(u_pred)
        
        # 总损失
        total_loss = L_data + self.lambda_phy * L_high_freq
        
        return total_loss


class AdaptiveHighFreqPINOLoss(nn.Module):
    """
    自适应高频噪声惩罚（Round 2）
    
    lambda从0.0001开始，每10 epoch增加1.5倍
    """
    
    def __init__(
        self,
        initial_lambda: float = 0.0001,
        growth_factor: float = 1.5,
        growth_interval: int = 10,
        max_lambda: float = 0.01,
        freq_threshold: float = 0.5
    ):
        super().__init__()
        self.lambda_phy = initial_lambda
        self.initial_lambda = initial_lambda
        self.growth_factor = growth_factor
        self.growth_interval = growth_interval
        self.max_lambda = max_lambda
        self.freq_threshold = freq_threshold
        self.epoch_count = 0
    
    def step_epoch(self):
        """每个epoch后调用，更新lambda"""
        self.epoch_count += 1
        if self.epoch_count % self.growth_interval == 0:
            self.lambda_phy = min(self.lambda_phy * self.growth_factor, self.max_lambda)
            print(f"[Adaptive PINO] Epoch {self.epoch_count}: lambda = {self.lambda_phy:.6f}")
    
    def compute_high_freq_penalty(self, u: torch.Tensor) -> torch.Tensor:
        """计算高频噪声惩罚（同HighFreqPINOLoss）"""
        u_freq = torch.fft.rfft2(u, norm='ortho')
        
        B, C, H, W = u.shape
        freq_h = torch.fft.fftfreq(H, device=u.device).abs()
        freq_w = torch.fft.rfftfreq(W, device=u.device).abs()
        
        freq_h = freq_h / freq_h.max()
        freq_w = freq_w / freq_w.max()
        
        freq_grid = torch.sqrt(freq_h[:, None]**2 + freq_w[None, :]**2)
        freq_grid = freq_grid / freq_grid.max()
        
        high_freq_mask = (freq_grid > self.freq_threshold).float()
        high_freq_amp = u_freq.abs() * high_freq_mask
        high_freq_loss = (high_freq_amp ** 2).mean()
        
        return high_freq_loss
    
    def forward(
        self,
        u_pred: torch.Tensor,
        u_true: torch.Tensor,
        u_prev: Optional[torch.Tensor] = None,
        dt: Optional[float] = None
    ) -> torch.Tensor:
        """计算总损失"""
        L_data = F.mse_loss(u_pred, u_true)
        L_high_freq = self.compute_high_freq_penalty(u_pred)
        total_loss = L_data + self.lambda_phy * L_high_freq
        return total_loss


# 测试代码
if __name__ == "__main__":
    print("测试高频噪声惩罚 PINO...")
    
    # 创建测试数据
    B, C, H, W = 4, 1, 32, 32
    u_pred = torch.randn(B, C, H, W)
    u_true = torch.randn(B, C, H, W)
    
    # 测试 HighFreqPINOLoss
    loss_fn = HighFreqPINOLoss(lambda_physics=0.0001, freq_threshold=0.5)
    loss = loss_fn(u_pred, u_true)
    print(f"✓ HighFreqPINOLoss: {loss.item():.6f}")
    
    # 测试 AdaptiveHighFreqPINOLoss
    adaptive_loss_fn = AdaptiveHighFreqPINOLoss(
        initial_lambda=0.0001,
        growth_factor=1.5,
        growth_interval=10
    )
    loss = adaptive_loss_fn(u_pred, u_true)
    print(f"✓ AdaptiveHighFreqPINOLoss (初始): {loss.item():.6f}")
    
    # 模拟epoch增长
    for epoch in range(1, 31):
        adaptive_loss_fn.step_epoch()
        loss = adaptive_loss_fn(u_pred, u_true)
        if epoch % 10 == 0:
            print(f"  Epoch {epoch}: lambda = {adaptive_loss_fn.lambda_phy:.6f}, loss = {loss.item():.6f}")
    
    print("\n✅ 测试通过！")
