"""
使用 NeuralOperator 官方框架测试 MHF-FNO

关键发现：
NeuralOperator 的 FNO 有 conv_module 参数，可以传入自定义的 SpectralConv！

这样我们可以：
1. 使用官方的训练器
2. 使用官方的数据加载
3. 只需要提供 MHF SpectralConv 模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

# NeuralOperator 官方组件
from neuralop.models import FNO
from neuralop.data.datasets import load_darcy_flow_small
from neuralop.training import trainer
from neuralop.layers.spectral_convolution import SpectralConv


class MHFSpectralConv(SpectralConv):
    """
    MHF SpectralConv - 兼容 NeuralOperator 的 SpectralConv 接口
    
    通过继承 SpectralConv，可以直接作为 conv_module 参数传入 FNO
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: Tuple[int, ...],
        n_heads: int = 4,
        bias: bool = True,
        **kwargs  # 忽略其他参数
    ):
        # 不调用父类初始化，完全重写
        nn.Module.__init__(self)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes = n_modes
        self.n_heads = n_heads
        
        # 确保通道数能被头数整除
        if in_channels % n_heads != 0 or out_channels % n_heads != 0:
            # 如果不能整除，降级到标准 SpectralConv
            self.use_mhf = False
            self.weight = nn.Parameter(
                torch.randn(in_channels, out_channels, *n_modes, dtype=torch.cfloat) * 0.01
            )
        else:
            self.use_mhf = True
            self.head_in = in_channels // n_heads
            self.head_out = out_channels // n_heads
            
            # 处理 n_modes
            if len(n_modes) == 1:
                # 1D
                self.modes = [n_modes[0]]
                weight_shape = (n_heads, self.head_in, self.head_out, n_modes[0])
            else:
                # 2D 或更高维
                self.modes = list(n_modes)
                modes_y = n_modes[1] // 2 + 1 if len(n_modes) > 1 else n_modes[0]
                weight_shape = (n_heads, self.head_in, self.head_out, n_modes[0], modes_y)
            
            init_std = (2 / (in_channels + out_channels)) ** 0.5
            self.weight = nn.Parameter(
                torch.randn(*weight_shape, dtype=torch.cfloat) * init_std
            )
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels, *((1,) * len(n_modes))))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x, output_shape=None, *args, **kwargs):
        """前向传播，兼容 NeuralOperator 接口"""
        
        if not self.use_mhf:
            # 降级到标准模式
            return self._forward_standard(x)
        
        if x.dim() == 3:
            return self._forward_1d(x)
        elif x.dim() == 4:
            return self._forward_2d(x)
        else:
            raise ValueError(f"Unsupported input dimension: {x.dim()}")
    
    def _forward_1d(self, x):
        """1D 前向传播"""
        B, C, L = x.shape
        
        x_freq = torch.fft.rfft(x, dim=-1)
        
        n_modes = min(self.modes[0], x_freq.shape[-1])
        x_freq = x_freq.view(B, self.n_heads, self.head_in, -1)
        
        out_freq = torch.zeros(B, self.n_heads, self.head_out, x_freq.shape[-1],
                               dtype=x_freq.dtype, device=x.device)
        out_freq[..., :n_modes] = torch.einsum(
            'bhif,hiof->bhof',
            x_freq[..., :n_modes],
            self.weight[..., :n_modes]
        )
        out_freq = out_freq.reshape(B, self.out_channels, -1)
        
        x_out = torch.fft.irfft(out_freq, n=L, dim=-1)
        
        if self.bias is not None:
            x_out = x_out + self.bias
        
        return x_out
    
    def _forward_2d(self, x):
        """2D 前向传播"""
        B, C, H, W = x.shape
        
        x_freq = torch.fft.rfft2(x, dim=(-2, -1))
        
        freq_H, freq_W = x_freq.shape[-2], x_freq.shape[-1]
        m_x = min(self.modes[0], freq_H)
        m_y = min(self.weight.shape[-1], freq_W)
        
        x_freq = x_freq.view(B, self.n_heads, self.head_in, freq_H, freq_W)
        
        out_freq = torch.zeros(B, self.n_heads, self.head_out, freq_H, freq_W,
                               dtype=x_freq.dtype, device=x.device)
        out_freq[:, :, :, :m_x, :m_y] = torch.einsum(
            'bhiXY,hioXY->bhoXY',
            x_freq[:, :, :, :m_x, :m_y],
            self.weight[:, :, :, :m_x, :m_y]
        )
        out_freq = out_freq.reshape(B, self.out_channels, freq_H, freq_W)
        
        x_out = torch.fft.irfft2(out_freq, s=(H, W), dim=(-2, -1))
        
        if self.bias is not None:
            x_out = x_out + self.bias
        
        return x_out
    
    def _forward_standard(self, x):
        """标准模式（当通道数不能被头数整除时）"""
        if x.dim() == 3:
            B, C, L = x.shape
            x_freq = torch.fft.rfft(x, dim=-1)
            n_modes = min(self.weight.shape[-1], x_freq.shape[-1])
            out_freq = torch.zeros(B, self.out_channels, x_freq.shape[-1],
                                   dtype=x_freq.dtype, device=x.device)
            out_freq[..., :n_modes] = torch.einsum('bif,iOf->bOf',
                x_freq[..., :n_modes], self.weight[..., :n_modes])
            x_out = torch.fft.irfft(out_freq, n=L, dim=-1)
        else:
            B, C, H, W = x.shape
            x_freq = torch.fft.rfft2(x, dim=(-2, -1))
            freq_H, freq_W = x_freq.shape[-2], x_freq.shape[-1]
            m_x = min(self.weight.shape[-2], freq_H)
            m_y = min(self.weight.shape[-1], freq_W)
            out_freq = torch.zeros(B, self.out_channels, freq_H, freq_W,
                                   dtype=x_freq.dtype, device=x.device)
            out_freq[:, :, :m_x, :m_y] = torch.einsum('biXY,ioXY->boXY',
                x_freq[:, :, :m_x, :m_y], self.weight[:, :, :m_x, :m_y])
            x_out = torch.fft.irfft2(out_freq, s=(H, W), dim=(-2, -1))
        
        if self.bias is not None:
            x_out = x_out + self.bias
        
        return x_out


def test_with_official_framework():
    """使用 NeuralOperator 官方框架测试 MHF-FNO"""
    
    print("\n" + "=" * 60)
    print(" 使用 NeuralOperator 官方框架测试 MHF-FNO")
    print("=" * 60)
    
    # 加载官方数据
    print("\n加载官方 Darcy Flow 数据...")
    train_loader, test_loaders, data_processor = load_darcy_flow_small(
        n_train=500,
        n_tests=50,
        batch_size=16,
        test_batch_sizes=16,
    )
    
    print(f"训练批次数: {len(train_loader)}")
    print(f"测试批次数: {len(test_loaders[16])}")
    
    # 创建 MHF-FNO (使用官方 FNO 框架 + MHF SpectralConv)
    print("\n创建 MHF-FNO (官方框架)...")
    
    model_mhf = FNO(
        n_modes=(8, 8),
        hidden_channels=32,
        in_channels=1,
        out_channels=1,
        n_layers=3,
        conv_module=MHFSpectralConv,  # 关键：传入 MHF SpectralConv
        n_heads=4,  # MHF 参数
    )
    
    # 创建标准 FNO
    print("创建标准 FNO (官方实现)...")
    
    model_fno = FNO(
        n_modes=(8, 8),
        hidden_channels=32,
        in_channels=1,
        out_channels=1,
        n_layers=3,
    )
    
    # 统计参数
    params_mhf = sum(p.numel() for p in model_mhf.parameters())
    params_fno = sum(p.numel() for p in model_fno.parameters())
    
    print(f"\n参数量对比:")
    print(f"  FNO:     {params_fno:,}")
    print(f"  MHF-FNO: {params_mhf:,}")
    print(f"  减少:    {(1 - params_mhf/params_fno)*100:.1f}%")
    
    # 测试前向传播
    print("\n测试前向传播...")
    for batch in train_loader:
        x = batch['x']
        print(f"  输入: {x.shape}")
        y_mhf = model_mhf(x)
        y_fno = model_fno(x)
        print(f"  MHF-FNO 输出: {y_mhf.shape}")
        print(f"  FNO 输出: {y_fno.shape}")
        break
    
    print("\n✅ MHF-FNO 兼容 NeuralOperator 官方框架！")
    
    return model_mhf, model_fno


if __name__ == "__main__":
    test_with_official_framework()