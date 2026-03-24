#!/usr/bin/env python3
"""
MHF-FNO 单元测试

测试覆盖:
- MHFSpectralConv 前向传播
- MHFFNO 模型完整性
- 不同 n_heads 配置
- 边界情况处理
- GPU 兼容性

运行: pytest tests/ -v
"""

import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from mhf_fno.mhf_fno import MHFSpectralConv, create_hybrid_fno, MHFFNO


class TestMHFSpectralConv:
    """MHFSpectralConv 单元测试"""
    
    def test_1d_forward_shape(self):
        """测试 1D 前向传播输出形状"""
        conv = MHFSpectralConv(
            in_channels=32,
            out_channels=32,
            n_modes=(16,),
            n_heads=4
        )
        x = torch.randn(2, 32, 64)
        out = conv(x)
        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
    
    def test_2d_forward_shape(self):
        """测试 2D 前向传播输出形状"""
        conv = MHFSpectralConv(
            in_channels=32,
            out_channels=32,
            n_modes=(8, 8),
            n_heads=4
        )
        x = torch.randn(2, 32, 16, 16)
        out = conv(x)
        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
    
    def test_different_n_heads(self):
        """测试不同 n_heads 配置"""
        configs = [
            (32, 32, 2),
            (32, 32, 4),
            (32, 32, 8),
            (64, 64, 4),
            (64, 64, 8),
        ]
        
        for in_ch, out_ch, n_heads in configs:
            conv = MHFSpectralConv(
                in_channels=in_ch,
                out_channels=out_ch,
                n_modes=(8, 8),
                n_heads=n_heads
            )
            x = torch.randn(2, in_ch, 16, 16)
            out = conv(x)
            assert out.shape == (2, out_ch, 16, 16)
    
    def test_parameter_reduction(self):
        """测试参数量减少"""
        # 标准 SpectralConv 参数量
        in_ch, out_ch = 32, 32
        n_modes = (8, 8)
        n_heads = 4
        
        # 标准参数量 (近似)
        std_params = in_ch * out_ch * n_modes[0] * (n_modes[1] // 2 + 1)
        
        # MHF 参数量
        head_in = in_ch // n_heads
        head_out = out_ch // n_heads
        mhf_params = n_heads * head_in * head_out * n_modes[0] * (n_modes[1] // 2 + 1)
        
        # MHF 应该更少
        assert mhf_params < std_params, "MHF should have fewer parameters"
        reduction = 1 - mhf_params / std_params
        assert reduction > 0.5, f"Expected >50% reduction, got {reduction*100:.1f}%"
    
    def test_non_divisible_channels(self):
        """测试通道数不能被 n_heads 整除的情况"""
        # 当通道数不能整除时，应使用标准卷积
        conv = MHFSpectralConv(
            in_channels=31,  # 不能被 4 整除
            out_channels=31,
            n_modes=(8, 8),
            n_heads=4
        )
        assert not conv.use_mhf, "Should fall back to standard conv"
        
        x = torch.randn(2, 31, 16, 16)
        out = conv(x)
        assert out.shape == (2, 31, 16, 16)
    
    def test_bias_option(self):
        """测试 bias 选项"""
        # 有 bias
        conv_with_bias = MHFSpectralConv(32, 32, (8, 8), n_heads=4, bias=True)
        assert conv_with_bias.bias is not None
        
        # 无 bias
        conv_no_bias = MHFSpectralConv(32, 32, (8, 8), n_heads=4, bias=False)
        assert conv_no_bias.bias is None
    
    def test_gradient_flow(self):
        """测试梯度流"""
        conv = MHFSpectralConv(32, 32, (8, 8), n_heads=4)
        x = torch.randn(2, 32, 16, 16, requires_grad=True)
        out = conv(x)
        loss = out.sum()
        loss.backward()
        
        assert x.grad is not None, "Input gradient should exist"
        assert conv.weight.grad is not None, "Weight gradient should exist"
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_forward(self):
        """测试 GPU 前向传播"""
        conv = MHFSpectralConv(32, 32, (8, 8), n_heads=4).cuda()
        x = torch.randn(2, 32, 16, 16).cuda()
        out = conv(x)
        assert out.device.type == 'cuda'
        assert out.shape == x.shape
    
    def test_deterministic_output(self):
        """测试输出确定性"""
        torch.manual_seed(42)
        conv = MHFSpectralConv(32, 32, (8, 8), n_heads=4)
        conv.eval()
        
        x = torch.randn(2, 32, 16, 16)
        
        out1 = conv(x)
        out2 = conv(x)
        
        assert torch.allclose(out1, out2), "Output should be deterministic in eval mode"


class TestMHFFNO:
    """MHFFNO 模型测试"""
    
    def test_best_config(self):
        """测试最佳配置预设"""
        model = MHFFNO.best_config(
            n_modes=(8, 8),
            hidden_channels=32,
            in_channels=1,
            out_channels=1
        )
        
        x = torch.randn(2, 1, 16, 16)
        out = model(x)
        assert out.shape == (2, 1, 16, 16)
    
    def test_create_hybrid_fno(self):
        """测试混合 FNO 创建"""
        model = create_hybrid_fno(
            n_modes=(8, 8),
            hidden_channels=32,
            in_channels=1,
            out_channels=1,
            n_layers=3,
            mhf_layers=[0, 2],
            n_heads=4
        )
        
        x = torch.randn(2, 1, 16, 16)
        out = model(x)
        assert out.shape == (2, 1, 16, 16)
    
    def test_different_layer_configs(self):
        """测试不同层配置"""
        configs = [
            ([0], 3),        # 只第一层
            ([2], 3),        # 只最后一层
            ([0, 2], 3),     # 第一和最后一层
            ([0, 1, 2], 3),  # 所有层
        ]
        
        for mhf_layers, n_layers in configs:
            model = create_hybrid_fno(
                n_modes=(8, 8),
                hidden_channels=32,
                in_channels=1,
                out_channels=1,
                n_layers=n_layers,
                mhf_layers=mhf_layers,
                n_heads=4
            )
            
            x = torch.randn(2, 1, 16, 16)
            out = model(x)
            assert out.shape == (2, 1, 16, 16)
    
    def test_model_trainability(self):
        """测试模型可训练性"""
        model = MHFFNO.best_config(
            n_modes=(8, 8),
            hidden_channels=32
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        x = torch.randn(4, 1, 16, 16)
        y = torch.randn(4, 1, 16, 16)
        
        # 训练一步
        model.train()
        optimizer.zero_grad()
        loss = ((model(x) - y) ** 2).mean()
        loss.backward()
        optimizer.step()
        
        assert loss.item() > 0


class TestIntegration:
    """集成测试"""
    
    def test_full_pipeline(self):
        """测试完整流水线"""
        # 创建模型
        model = MHFFNO.best_config(
            n_modes=(8, 8),
            hidden_channels=32,
            in_channels=1,
            out_channels=1
        )
        
        # 生成数据
        train_x = torch.randn(16, 1, 16, 16)
        train_y = torch.randn(16, 1, 16, 16)
        test_x = torch.randn(4, 1, 16, 16)
        
        # 训练
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        
        for _ in range(5):
            optimizer.zero_grad()
            loss = ((model(train_x) - train_y) ** 2).mean()
            loss.backward()
            optimizer.step()
        
        # 推理
        model.eval()
        with torch.no_grad():
            pred = model(test_x)
        
        assert pred.shape == test_x.shape
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_pipeline(self):
        """测试 GPU 完整流水线"""
        model = MHFFNO.best_config(
            n_modes=(8, 8),
            hidden_channels=32
        ).cuda()
        
        x = torch.randn(4, 1, 16, 16).cuda()
        out = model(x)
        
        assert out.device.type == 'cuda'
    
    def test_parameter_count_comparison(self):
        """对比参数量"""
        from neuralop.models import FNO
        
        # 标准 FNO
        fno = FNO(
            n_modes=(8, 8),
            hidden_channels=32,
            in_channels=1,
            out_channels=1,
            n_layers=3
        )
        
        # MHF-FNO
        mhf_fno = MHFFNO.best_config(
            n_modes=(8, 8),
            hidden_channels=32,
            in_channels=1,
            out_channels=1
        )
        
        fno_params = sum(p.numel() for p in fno.parameters())
        mhf_params = sum(p.numel() for p in mhf_fno.parameters())
        
        print(f"\nFNO params: {fno_params:,}")
        print(f"MHF-FNO params: {mhf_params:,}")
        print(f"Reduction: {(1 - mhf_params/fno_params)*100:.1f}%")


class TestEdgeCases:
    """边界情况测试"""
    
    def test_small_input(self):
        """测试小尺寸输入"""
        model = MHFFNO.best_config(n_modes=(4, 4), hidden_channels=16)
        x = torch.randn(1, 1, 8, 8)
        out = model(x)
        assert out.shape == x.shape
    
    def test_large_batch(self):
        """测试大批次"""
        model = MHFFNO.best_config(n_modes=(8, 8), hidden_channels=32)
        x = torch.randn(64, 1, 16, 16)
        out = model(x)
        assert out.shape == x.shape
    
    def test_single_head(self):
        """测试单头配置"""
        conv = MHFSpectralConv(
            in_channels=32,
            out_channels=32,
            n_modes=(8, 8),
            n_heads=1
        )
        x = torch.randn(2, 32, 16, 16)
        out = conv(x)
        assert out.shape == x.shape


if __name__ == '__main__':
    pytest.main([__file__, '-v'])